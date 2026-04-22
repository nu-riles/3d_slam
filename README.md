# NVIDIA PhysicalAI-AV -> 3D SLAM Pipeline

Samples frames from the NVIDIA Physical AI Autonomous Vehicles dataset,
estimates camera poses via ORB feature matching, and accumulates a sparse 3-D
map suitable for basic navigation.

## Architecture

```
PhysicalAIAVDataset
      |  list_clips() -> download_clip()
      v
AVDatasetSampler          <- streams N clips, samples K frames each
      |  (clip_uuid, [frame0, frame1, ...])
      v
FramePoseEstimator        <- ORB detect -> BF match -> Essential mat -> R, t, pts3d
      |  (R_rel, t_rel, pts3d_local)
      v
SLAMMap                   <- chains relative poses, accumulates world-frame pts
      |  reconstruction.ply  +  trajectory.npy
      v
BasicNavigator            <- XZ ground-plane occupancy grid for navigation
```

## Quickstart

### 1. Install dependencies

    pip install git+https://github.com/NVlabs/physical_ai_av
    pip install -r requirements.txt

### 2. Authenticate with HuggingFace

    huggingface-cli login    # paste your HF token when prompted

Then accept the NVIDIA license at:
https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

### 3. Run the pipeline

    python slam_pipeline.py

Default: 25 clips x 3 frames (75 frames total).
Outputs written to slam_output/:
  reconstruction.ply  -- sparse 3-D point cloud (Open3D / MeshLab compatible)
  trajectory.npy      -- (N, 3) NumPy array of camera world positions

### 4. Visualise

    python visualize_slam.py                     # interactive 3-D viewer
    python visualize_slam.py --save topdown.png  # headless top-down PNG

## Configuration (SLAMConfig)

| Parameter        | Default                    | Description                                 |
|------------------|----------------------------|---------------------------------------------|
| camera_view      | camera_front_wide_120fov   | Which AV camera to use                      |
| num_videos       | 25                         | Number of 20-second clips to download       |
| frames_per_video | 3                          | Uniformly-spaced frames per clip            |
| orb_features     | 2000                       | ORB keypoints per frame                     |
| min_matches      | 20                         | Minimum matches to accept a pose estimate   |
| lowe_ratio       | 0.75                       | Lowe ratio-test threshold                   |
| fx/fy/cx/cy      | approx 1080p wide          | Camera intrinsics (override with calib data)|
| output_dir       | slam_output/               | Directory for PLY + NPY outputs             |

## Using the Navigator

    from slam_pipeline import SLAMConfig, run_pipeline

    slam_map, navigator = run_pipeline(SLAMConfig(num_videos=5, frames_per_video=3))

    # Query free space in world coordinates (X, Z plane)
    print(navigator.is_free(0.0, 0.0))    # True / False
    print(navigator.grid.shape)            # (cols, rows) of occupancy grid

## Design notes

Pose chain: relative R, t between consecutive frames within a clip are chained
to build the global trajectory. No global bundle adjustment is performed, which
keeps the implementation simple at the cost of drift over long sequences.

Scale ambiguity: Essential matrix decomposition produces unit-translation poses.
Metric scale is not recovered here. Providing GPS/IMU data from the dataset
parquet files would resolve this.

Intrinsics: Default values approximate a 1080p 120-degree FOV camera. Replace
with the per-clip calibration matrices from the dataset parquet metadata for
best results.

Occupancy grid: Built by projecting 3-D points onto the XZ plane (Y is
vertical in the vehicle coordinate frame). Cell resolution defaults to 0.5 m.
