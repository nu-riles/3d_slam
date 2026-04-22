"""
visualize_slam.py
=================
Loads the saved PLY point cloud and trajectory produced by slam_pipeline.py
and visualises them with Open3D (interactive viewer) or saves a top-down PNG.

Usage
-----
    python visualize_slam.py                        # interactive viewer
    python visualize_slam.py --save topdown.png     # headless PNG export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_results(output_dir: Path = Path("slam_output")):
    ply_path  = output_dir / "reconstruction.ply"
    traj_path = output_dir / "trajectory.npy"

    pcd  = o3d.io.read_point_cloud(str(ply_path))
    traj = np.load(str(traj_path))          # (N, 3) world positions
    return pcd, traj


def build_trajectory_lineset(traj: np.ndarray) -> o3d.geometry.LineSet:
    """Convert trajectory positions into a coloured LineSet for visualisation."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(traj)
    lines = [[i, i + 1] for i in range(len(traj) - 1)]
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color([1.0, 0.0, 0.0])   # red trajectory
    return ls


def interactive_view(pcd, traj):
    ls = build_trajectory_lineset(traj)
    o3d.visualization.draw_geometries(
        [pcd, ls],
        window_name="SLAM Reconstruction",
        width=1280,
        height=720,
    )


def save_topdown_png(pcd, traj, save_path: str = "topdown.png"):
    """Render a top-down (bird's-eye) view and save to PNG."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    ls = build_trajectory_lineset(traj)
    vis.add_geometry(pcd)
    vis.add_geometry(ls)

    ctr = vis.get_view_control()
    ctr.set_lookat(np.asarray(pcd.points).mean(axis=0).tolist())
    ctr.set_up([0, 0, -1])
    ctr.set_front([0, 1, 0])   # looking down Y
    ctr.set_zoom(0.5)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print(f"Saved top-down view → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise SLAM output")
    parser.add_argument(
        "--output_dir", default="slam_output",
        help="Directory containing reconstruction.ply and trajectory.npy",
    )
    parser.add_argument(
        "--save", default=None,
        help="If set, save a top-down PNG instead of opening the interactive viewer",
    )
    args = parser.parse_args()

    pcd, traj = load_results(Path(args.output_dir))
    print(f"Loaded {len(pcd.points):,} points, {len(traj)} trajectory poses")

    if args.save:
        save_topdown_png(pcd, traj, args.save)
    else:
        interactive_view(pcd, traj)