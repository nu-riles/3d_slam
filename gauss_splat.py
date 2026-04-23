"""
gaussian_splat.py
=================
Trains a 3D Gaussian Splatting model per clip using:
  - The per-clip colorized LiDAR PLY as point cloud initialisation
  - The 3 sampled camera frames as training views
  - Camera poses derived from the sensor extrinsics parquet

Outputs one trained Gaussian PLY per clip to slam_output/gaussians/

Requirements
------------
    pip install gsplat torch torchvision open3d numpy opencv-python-headless

Usage
-----
    python gaussian_splat.py \
        --ply_dir   /scratch/$USER/lidar_fusion/output \
        --img_dir   /scratch/$USER/lidar_fusion/output \
        --calib_dir /scratch/$USER/lidar_fusion/calibration \
        --out_dir   /scratch/$USER/lidar_fusion/gaussians \
        --iters     1000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from gsplat import rasterization

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ], dtype=np.float32)


def poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def ftheta_to_pinhole_K(cx, cy, a2p, W, H):
    """
    Approximate a pinhole K from the ftheta model by computing
    the effective focal length at the image centre.
    """
    # At small angles, r_pix ≈ a2p[1] * theta, and for pinhole f = r/tan(theta) ≈ r/theta
    f = a2p[1] if len(a2p) > 1 else 800.0
    return np.array([
        [f,  0, cx],
        [0,  f, cy],
        [0,  0,  1],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset loader for a single clip
# ---------------------------------------------------------------------------

class ClipDataset:
    """
    Loads the 3 training frames and camera parameters for one clip.
    """

    def __init__(self, clip_uuid: str, img_dir: Path, calib_dir: Path, device: torch.device):
        self.clip_uuid = clip_uuid
        self.device    = device

        import pandas as pd

        # Load calibration
        intr_df = pd.read_parquet(
            calib_dir / "camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet"
        ).reset_index()
        ext_df = pd.read_parquet(
            calib_dir / "sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet"
        ).reset_index()

        row    = intr_df[(intr_df["clip_id"]==clip_uuid) &
                         (intr_df["camera_name"]=="camera_front_wide_120fov")].iloc[0]
        params = json.loads(row["model_parameters"])
        cx, cy    = params["principal_point"]
        a2p       = params["angle_to_pixeldist_poly"]
        self.W, self.H = params["resolution"]
        self.K    = ftheta_to_pinhole_K(cx, cy, a2p, self.W, self.H)

        # Camera-to-world transform
        r = ext_df[(ext_df["clip_id"]==clip_uuid) &
                   (ext_df["sensor_name"]=="camera_front_wide_120fov")].iloc[0]
        R = quat_to_R(r.qx, r.qy, r.qz, r.qw)
        t = np.array([r.x, r.y, r.z], dtype=np.float32)
        # Build 4x4 world-to-camera (view) matrix
        c2w       = np.eye(4, dtype=np.float32)
        c2w[:3,:3] = R
        c2w[:3, 3] = t
        self.w2c  = np.linalg.inv(c2w)

        # Load frames saved during the fusion pipeline
        # Expected naming: <clip_uuid>.camera_front_wide_120fov.frame_<N>.png
        # Fall back to reading from the mp4 if PNGs aren't present
        self.frames = self._load_frames(img_dir)

    def _load_frames(self, img_dir: Path) -> list[np.ndarray]:
        """Try PNG first, fall back to mp4."""
        pngs = sorted(img_dir.glob(f"{self.clip_uuid}.frame_*.png"))
        if pngs:
            return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in pngs]

        mp4 = img_dir / f"{self.clip_uuid}.camera_front_wide_120fov.mp4"
        if not mp4.exists():
            raise FileNotFoundError(f"No frames or mp4 found for {self.clip_uuid}")

        cap   = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs  = np.linspace(0, total - 1, 3, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def get_cameras(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (viewmats, Ks, image_wh) batched for gsplat.
        viewmats: (N, 4, 4) world-to-camera
        Ks:       (N, 3, 3) intrinsics
        """
        N = len(self.frames)
        viewmats = torch.tensor(self.w2c, dtype=torch.float32, device=self.device).unsqueeze(0).expand(N, -1, -1)
        Ks       = torch.tensor(self.K,   dtype=torch.float32, device=self.device).unsqueeze(0).expand(N, -1, -1)
        return viewmats, Ks

    def get_images(self) -> torch.Tensor:
        """Returns (N, H, W, 3) float32 [0,1] tensor."""
        imgs = []
        for f in self.frames:
            resized = cv2.resize(f, (self.W, self.H))
            imgs.append(torch.tensor(resized / 255.0, dtype=torch.float32))
        return torch.stack(imgs).to(self.device)   # (N, H, W, 3)


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------

class GaussianModel:
    """
    Minimal trainable Gaussian model initialised from a PLY point cloud.
    Parameters: means, quats, scales, opacities, colors (SH degree 0)
    """

    def __init__(self, ply_path: Path, device: torch.device):
        pcd    = o3d.io.read_point_cloud(str(ply_path))
        pts    = np.asarray(pcd.points,  dtype=np.float32)
        colors = np.asarray(pcd.colors,  dtype=np.float32)  # (N, 3) [0,1]

        N = len(pts)
        self.device = device

        self.means     = torch.nn.Parameter(torch.tensor(pts,    device=device))
        self.quats     = torch.nn.Parameter(torch.zeros(N, 4,    device=device).fill_(0).index_fill_(1, torch.tensor([0]), 1.0))  # identity quat
        self.log_scales = torch.nn.Parameter(torch.full((N, 3), -3.0, device=device))
        self.logit_opacities = torch.nn.Parameter(torch.zeros(N, device=device))
        # SH degree 0: one coefficient per channel
        self.sh0       = torch.nn.Parameter(
            torch.tensor(
                (colors - 0.5) / 0.28209,  # inverse SH activation
                dtype=torch.float32, device=device
            ).unsqueeze(1)   # (N, 1, 3)
        )

    def params(self):
        return [self.means, self.quats, self.log_scales, self.logit_opacities, self.sh0]

    def get_scales(self):
        return torch.exp(self.log_scales)

    def get_opacities(self):
        return torch.sigmoid(self.logit_opacities)

    def get_colors(self):
        # SH degree 0 activation
        return torch.sigmoid(self.sh0[:, 0, :] * 0.28209 + 0.5)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_clip(
    clip_uuid: str,
    ply_path: Path,
    img_dir: Path,
    calib_dir: Path,
    out_dir: Path,
    iters: int,
    device: torch.device,
):
    print(f"\n[{clip_uuid}] Loading data...")
    dataset = ClipDataset(clip_uuid, img_dir, calib_dir, device)
    gt_imgs = dataset.get_images()        # (N, H, W, 3)
    viewmats, Ks = dataset.get_cameras()
    W, H = dataset.W, dataset.H

    print(f"[{clip_uuid}] Initialising Gaussians from {ply_path.name}...")
    model  = GaussianModel(ply_path, device)
    optim  = torch.optim.Adam(model.params(), lr=1e-3)

    print(f"[{clip_uuid}] Training {iters} iterations...")
    for step in range(iters):
        optim.zero_grad()

        print("means:", model.means.device)
        print("quats:", model.quats.device)
        print("scales:", model.get_scales().device)
        print("opacities:", model.get_opacities().device)
        print("colors:", model.get_colors().device)
        print("viewmats:", viewmats.device)
        print("Ks:", Ks.device)
        print("gt_imgs:", gt_imgs.device)

        renders, alphas, _ = rasterization(
            means     = model.means,
            quats     = F.normalize(model.quats, dim=-1),
            scales    = model.get_scales(),
            opacities = model.get_opacities(),
            colors    = model.get_colors().unsqueeze(0).expand(len(dataset.frames), -1, -1).contiguous().to(device),
            viewmats  = viewmats.contiguous().to(device),
            Ks        = Ks.contiguous().to(device),
            width     = W,
            height    = H,
            sh_degree = 0,
        )

        loss = F.l1_loss(renders, gt_imgs.to(device))
        loss.backward()
        optim.step()

        if step % 200 == 0:
            print(f"  step {step:4d}/{iters}  loss={loss.item():.4f}")

    # Save trained Gaussians as PLY
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / f"{clip_uuid}.gaussian.ply"
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(model.means.detach().cpu().numpy())
    final_pcd.colors = o3d.utility.Vector3dVector(model.get_colors().detach().cpu().numpy())
    o3d.io.write_point_cloud(str(out_ply), final_pcd)
    print(f"[{clip_uuid}] Saved → {out_ply.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_dir",   required=True, help="Directory containing per-clip PLY files")
    parser.add_argument("--img_dir",   required=True, help="Directory containing camera frames or mp4s")
    parser.add_argument("--calib_dir", required=True, help="Directory containing calibration parquets")
    parser.add_argument("--out_dir",   required=True, help="Output directory for Gaussian PLYs")
    parser.add_argument("--iters",     type=int, default=1000)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ply_dir  = Path(args.ply_dir)
    img_dir  = Path(args.img_dir)
    calib_dir = Path(args.calib_dir)
    out_dir  = Path(args.out_dir)

    # Find all per-clip PLYs (exclude any gaussian outputs)
    ply_files = [p for p in ply_dir.glob("*.ply") if "gaussian" not in p.name]
    print(f"Found {len(ply_files)} clip PLYs to process")

    for ply_path in sorted(ply_files):
        clip_uuid = ply_path.stem
        try:
            train_clip(
                clip_uuid = clip_uuid,
                ply_path  = ply_path,
                img_dir   = img_dir,
                calib_dir = calib_dir,
                out_dir   = out_dir,
                iters     = args.iters,
                device    = device,
            )
        except Exception as e:
            print(f"[{clip_uuid}] Skipping: {e}")
            continue

    print(f"\nAll Gaussian PLYs saved to {out_dir}")
