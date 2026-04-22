"""
NVIDIA PhysicalAI-AV → Frame Sampling → 3D SLAM Reconstruction
================================================================
Samples N frames from M video clips in the NVIDIA Physical AI AV dataset,
estimates relative camera poses via ORB feature matching + Essential matrix
decomposition, accumulates a sparse 3D point cloud, and exposes it for
downstream navigation.

Requirements
------------
    pip install physical_ai_av huggingface_hub opencv-python-headless \
                open3d numpy tqdm

Authentication
--------------
    huggingface-cli login          # or set HF_TOKEN env variable
    # Accept the NVIDIA license at:
    # https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

# physical_ai_av is the official NVIDIA devkit for this dataset
# https://github.com/NVlabs/physical_ai_av
from physical_ai_av import PhysicalAIAVDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@dataclass
class SLAMConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN")
    )
    # Camera view to use.  The dataset provides:
    #   front_wide_120fov  |  front_tele_30fov
    #   cross_right_120fov |  cross_left_120fov
    #   rear_right_70fov   |  rear_left_70fov
    camera_view: str = "camera_front_wide_120fov"
    num_videos: int = 25          # clips to pull from the dataset
    frames_per_video: int = 3     # uniformly-spaced frames per clip

    # ── Feature matching ─────────────────────────────────────────────────────
    orb_features: int = 2000      # ORB keypoints per frame
    min_matches: int = 20         # minimum good matches required to estimate pose
    lowe_ratio: float = 0.75      # Lowe's ratio test threshold

    # ── Camera intrinsics (default: approximate 1080p wide-angle) ────────────
    # Override with real calibration data if available in the parquet metadata.
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 960.0
    cy: float = 540.0

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = Path("slam_output")
    pointcloud_file: str = "reconstruction.ply"
    trajectory_file: str = "trajectory.npy"


# ---------------------------------------------------------------------------
# Dataset interface
# ---------------------------------------------------------------------------

class AVDatasetSampler:
    """
    Wraps PhysicalAIAVDataset to yield uniformly-spaced BGR frames
    from the front-wide camera across a configurable number of clips.
    """

    def __init__(self, cfg: SLAMConfig):
        self.cfg = cfg
        log.info("Initialising PhysicalAIAVDataset ...")
        self.dataset = PhysicalAIAVDataset(token=cfg.hf_token)

    def _sample_frames_from_video(self, video_path: str | Path) -> list[np.ndarray]:
        """Return `frames_per_video` evenly-spaced BGR frames from an mp4."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Zero frames reported for {video_path}")

        n = self.cfg.frames_per_video
        indices = np.linspace(0, total - 1, n, dtype=int).tolist()

        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def iter_clip_frames(self):
        """
        Generator that yields (clip_uuid, [frame, ...]) for each requested clip.
        Downloads each clip to a temporary directory and cleans up afterwards.
        """
        clips = self.dataset.list_clips(
            camera=self.cfg.camera_view,
            limit=self.cfg.num_videos,
        )
        log.info("Found %d clips (requested %d)", len(clips), self.cfg.num_videos)

        for clip_meta in tqdm(clips, desc="Clips", unit="clip"):
            clip_uuid = clip_meta["clip_uuid"]
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    video_path = self.dataset.download_clip(
                        clip_uuid=clip_uuid,
                        camera=self.cfg.camera_view,
                        output_dir=tmpdir,
                    )
                    frames = self._sample_frames_from_video(video_path)
                    yield clip_uuid, frames
                except Exception as exc:
                    log.warning("Skipping clip %s: %s", clip_uuid, exc)


# ---------------------------------------------------------------------------
# Feature extraction & pose estimation
# ---------------------------------------------------------------------------

class FramePoseEstimator:
    """
    Estimates relative pose between consecutive frames using ORB feature
    matching and the Essential matrix (5-point algorithm via OpenCV).
    """

    def __init__(self, cfg: SLAMConfig):
        self.cfg = cfg
        self.K = np.array([
            [cfg.fx,    0, cfg.cx],
            [   0, cfg.fy, cfg.cy],
            [   0,     0,      1],
        ], dtype=np.float64)
        self.orb = cv2.ORB_create(nfeatures=cfg.orb_features)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, frame: np.ndarray):
        """Detect ORB keypoints and compute descriptors."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)
        return kps, des

    def match(self, des1, des2):
        """Ratio-test BF match between two descriptor sets."""
        if des1 is None or des2 is None:
            return []
        raw = self.bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < self.cfg.lowe_ratio * n.distance]
        return good

    def estimate_pose(
        self,
        kps1, des1,
        kps2, des2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Returns (R, t, pts3d) where pts3d are triangulated 3-D points in the
        reference frame of camera-1, or None if estimation fails.
        """
        matches = self.match(des1, des2)
        if len(matches) < self.cfg.min_matches:
            return None

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            return None

        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Triangulate inlier matches
        inliers1 = pts1[pose_mask.ravel() > 0]
        inliers2 = pts2[pose_mask.ravel() > 0]
        if len(inliers1) < 4:
            return None

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T   # (N, 3) in camera-1 frame

        return R, t, pts3d


# ---------------------------------------------------------------------------
# SLAM map builder
# ---------------------------------------------------------------------------

class SLAMMap:
    """
    Accumulates 3-D points across frames using the chain of relative poses.
    Stores the global trajectory (camera centres) and a fused point cloud.
    """

    def __init__(self):
        self.global_R = np.eye(3)
        self.global_t = np.zeros((3, 1))
        self.trajectory: list[np.ndarray] = [self.global_t.flatten().copy()]
        self.all_points: list[np.ndarray] = []
        self.all_colors: list[np.ndarray] = []

    def integrate(
        self,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        pts3d_local: np.ndarray,
        colors: np.ndarray | None = None,
    ):
        """Transform local points to world frame and append to the map."""
        self.global_t = self.global_R @ t_rel + self.global_t
        self.global_R = self.global_R @ R_rel
        self.trajectory.append(self.global_t.flatten().copy())

        pts_world = (self.global_R @ pts3d_local.T).T + self.global_t.T
        self.all_points.append(pts_world)

        if colors is not None:
            self.all_colors.append(colors)

    def to_pointcloud(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        if not self.all_points:
            return pcd
        pts = np.vstack(self.all_points)
        pcd.points = o3d.utility.Vector3dVector(pts)
        if self.all_colors and len(self.all_colors) == len(self.all_points):
            cols = np.vstack(self.all_colors).astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    def trajectory_array(self) -> np.ndarray:
        return np.array(self.trajectory)


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------

class BasicNavigator:
    """
    Minimal occupancy-grid navigator derived from the SLAM map.

    Projects 3-D points onto the XZ ground plane (Y is roughly vertical in
    the vehicle frame) and builds a 2-D binary occupancy grid.  Occupied cells
    are marked 1; free space is 0.

    This is intentionally simple — suitable as a foundation for A* or
    potential-field planners.
    """

    def __init__(self, slam_map: SLAMMap, cell_size: float = 0.5):
        self.cell_size = cell_size

        if not slam_map.all_points:
            log.warning("No points in SLAM map; navigator grid will be empty.")
            self.grid = np.zeros((1, 1), dtype=np.uint8)
            self._min = np.zeros(2)
            return

        pts = np.vstack(slam_map.all_points)
        xz  = pts[:, [0, 2]]           # XZ ground-plane projection
        self._min = xz.min(axis=0)
        self._max = xz.max(axis=0)

        shape = np.ceil((self._max - self._min) / cell_size).astype(int) + 1
        self.grid = np.zeros(shape, dtype=np.uint8)

        cells = ((xz - self._min) / cell_size).astype(int)
        cells = np.clip(cells, 0, np.array(shape) - 1)
        self.grid[cells[:, 0], cells[:, 1]] = 1

        log.info(
            "Occupancy grid built: %s cells @ %.2f m resolution",
            self.grid.shape, cell_size,
        )

    def world_to_grid(self, x: float, z: float) -> tuple[int, int]:
        col = int((x - self._min[0]) / self.cell_size)
        row = int((z - self._min[1]) / self.cell_size)
        return col, row

    def is_free(self, x: float, z: float) -> bool:
        """Return True if world position (x, z) maps to a free grid cell."""
        col, row = self.world_to_grid(x, z)
        if 0 <= col < self.grid.shape[0] and 0 <= row < self.grid.shape[1]:
            return self.grid[col, row] == 0
        return False   # out-of-bounds = unknown → treat as blocked


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: SLAMConfig | None = None) -> tuple[SLAMMap, BasicNavigator]:
    """
    Full end-to-end pipeline:
      1. Sample frames from the AV dataset
      2. Extract ORB features per frame
      3. Estimate pairwise relative poses
      4. Accumulate a sparse 3-D point cloud (SLAM map)
      5. Save PLY point cloud + trajectory to disk
      6. Build and return a BasicNavigator over the map
    """
    if cfg is None:
        cfg = SLAMConfig()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    sampler   = AVDatasetSampler(cfg)
    estimator = FramePoseEstimator(cfg)
    slam_map  = SLAMMap()

    total_frames = 0
    total_poses  = 0

    for clip_uuid, frames in sampler.iter_clip_frames():
        log.info("Processing clip %s  (%d frames)", clip_uuid, len(frames))

        prev_kps, prev_des, prev_frame = None, None, None

        for frame in frames:
            kps, des = estimator.extract(frame)
            total_frames += 1

            if prev_kps is not None:
                result = estimator.estimate_pose(prev_kps, prev_des, kps, des)
                if result is not None:
                    R, t, pts3d = result

                    # Sample BGR colours from the previous frame at keypoint locations
                    colors = None
                    if prev_frame is not None and len(pts3d) > 0:
                        h, w = prev_frame.shape[:2]
                        sample_kps = prev_kps[: len(pts3d)]
                        px = np.clip(
                            np.round([k.pt[0] for k in sample_kps]).astype(int), 0, w - 1
                        )
                        py = np.clip(
                            np.round([k.pt[1] for k in sample_kps]).astype(int), 0, h - 1
                        )
                        colors = prev_frame[py, px]  # (N, 3) BGR

                    slam_map.integrate(R, t, pts3d, colors)
                    total_poses += 1

            prev_kps, prev_des, prev_frame = kps, des, frame

    total_pts = sum(len(p) for p in slam_map.all_points)
    log.info(
        "Pipeline complete | frames=%d  poses=%d  3D-pts=%d",
        total_frames, total_poses, total_pts,
    )

    # ── Save outputs ─────────────────────────────────────────────────────────
    pcd = slam_map.to_pointcloud()
    if len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        ply_path = cfg.output_dir / cfg.pointcloud_file
        o3d.io.write_point_cloud(str(ply_path), pcd)
        log.info("Point cloud → %s  (%d pts)", ply_path, len(pcd.points))

    traj_path = cfg.output_dir / cfg.trajectory_file
    np.save(str(traj_path), slam_map.trajectory_array())
    log.info("Trajectory   → %s", traj_path)

    navigator = BasicNavigator(slam_map)
    return slam_map, navigator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = SLAMConfig(
        num_videos=25,
        frames_per_video=3,
        output_dir=Path("slam_output"),
    )
    slam_map, navigator = run_pipeline(config)

    print("\n[Navigator] Origin free?", navigator.is_free(0.0, 0.0))
    print("[Navigator] Grid shape: ", navigator.grid.shape)