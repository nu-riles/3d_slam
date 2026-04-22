# NVIDIA PhysicalAI-AV -> 3D SLAM + LiDAR Fusion Pipeline

Samples frames from the NVIDIA Physical AI Autonomous Vehicles dataset,
estimates camera poses via ORB feature matching, accumulates a sparse 3-D
map suitable for basic navigation, and fuses LiDAR point clouds with camera
imagery for colorized 3-D reconstruction.

## Summary

| Item      | Detail |
|-----------|--------|
| Outputs   | Colorized 3D point cloud (.ply), bird's-eye-view (.png), projection overlay (.png) |
| Dataset   | nvidia/PhysicalAI-Autonomous-Vehicles (HuggingFace) |
| Cluster   | Explorer — login.explorer.northeastern.edu |
| Compute   | CPU only, no GPU required |
| Storage   | ~32 GB per chunk (camera + LiDAR) |
| Runtime   | ~5 minutes after download |

## Architecture

```
PhysicalAIAVDataset
      |  list_clips() -> download_clip()
      v
AVDatasetSampler          <- streams N clips, samples K frames each
      |  (clip_uuid, [frame0, frame1, ...]
      v
FramePoseEstimator        <- ORB detect -> BF match -> Essential mat -> R, t, pts3d
      |  (R_rel, t_rel, pts3d_local)
      v
SLAMMap                   <- chains relative poses, accumulates world-frame pts
      |  reconstruction.ply  +  trajectory.npy
      v
BasicNavigator            <- XZ ground-plane occupancy grid for navigation
```

## Prerequisites

### HuggingFace Access

- Create a HuggingFace account at huggingface.co if you don't already have one
- Accept the NVIDIA license at: huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- Create a token with Read access at huggingface.co/settings/tokens
- Export it in your session: export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
- To persist across sessions: echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> ~/.bashrc

### Cluster Access (Northeastern Explorer)

- SSH in: ssh <yournetid>@login.explorer.northeastern.edu
- Request a compute node before running anything:

    srun --partition=courses --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --time=02:00:00 --pty /bin/bash

- Conda is not available on the login node — always srun first

### Conda Environment

The pipeline uses a conda environment with Python 3.11. Required packages:
pandas, numpy, opencv-python, open3d, huggingface_hub, pyarrow

Note: dracopy is not used despite being installable — the native draco_decoder
binary is used instead. Locate it with: find ~/.conda/envs/<env> -name draco_decoder

## Deployment

### 1. Set up workspace

    mkdir -p /scratch/$USER/lidar_fusion
    cd /scratch/$USER/lidar_fusion

### 2. Download calibration metadata

    python3 - <<'PYEOF'
    from huggingface_hub import hf_hub_download, login
    import os
    login(token=os.environ['HF_TOKEN'])
    base = '/scratch/' + os.environ['USER'] + '/lidar_fusion'
    for f in [
        'calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet',
        'calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet',
    ]:
        hf_hub_download(repo_id='nvidia/PhysicalAI-Autonomous-Vehicles',
                        repo_type='dataset', filename=f, local_dir=base)
        print(f'Downloaded {f}')
    PYEOF

### 3. Download camera and LiDAR chunk 0

Warning: chunk 0 is ~32 GB. Check space first with: df -h /scratch/$USER

    python3 - <<'PYEOF'
    from huggingface_hub import hf_hub_download, login
    import os
    login(token=os.environ['HF_TOKEN'])
    base = '/scratch/' + os.environ['USER'] + '/lidar_fusion'
    for f in [
        'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip',
        'lidar/lidar_top_360fov/lidar_top_360fov.chunk_0000.zip',
    ]:
        print(f'Downloading {f}...')
        hf_hub_download(repo_id='nvidia/PhysicalAI-Autonomous-Vehicles',
                        repo_type='dataset', filename=f, local_dir=base)
        print('Done.')
    PYEOF

### 4. Run the fusion pipeline

    bash deployment.sh

The clip UUID 25cd4769-5dcf-4b53-a351-bf2c5deb6124 is hardcoded as the first
clip in chunk 0 for the United States and is known to be valid.

### 5. Run the SLAM pipeline

    python slam_pipeline.py

Default: 25 clips x 3 frames (75 frames total).

## Outputs

| File               | Format | Description |
|--------------------|--------|-------------|
| <clip_uuid>.ply    | .ply   | Colorized 3D point cloud. Open in MeshLab, CloudCompare, or Open3D. |
| bev.png            | .png   | Bird's-eye-view top-down projection with 10m distance rings and ego vehicle marker. |
| projection.png     | .png   | Camera frame with LiDAR points overlaid. Green = near, red = far. |
| reconstruction.ply | .ply   | Sparse SLAM point cloud from ORB feature matching. |
| trajectory.npy     | .npy   | (N, 3) NumPy array of camera world positions. |

### Copy outputs to your local machine

    # Run on your LOCAL machine:
    scp <yournetid>@login.explorer.northeastern.edu:/scratch/<yournetid>/lidar_fusion/output/*.png ./
    scp <yournetid>@login.explorer.northeastern.edu:/scratch/<yournetid>/lidar_fusion/output/*.ply ./

### View the .ply point cloud

- MeshLab (recommended): meshlab.net — free, drag and drop
- CloudCompare: cloudcompare.org — free, handles large point clouds
- Open3D: import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('file.ply')])

### Visualise SLAM output

    python visualize_slam.py                     # interactive 3-D viewer
    python visualize_slam.py --save topdown.png  # headless top-down PNG

## Configuration (SLAMConfig)

| Parameter        | Default                  | Description                                  |
|------------------|--------------------------|----------------------------------------------|
| camera_view      | camera_front_wide_120fov | Which AV camera to use                       |
| num_videos       | 25                       | Number of 20-second clips to download        |
| frames_per_video | 3                        | Uniformly-spaced frames per clip             |
| orb_features     | 2000                     | ORB keypoints per frame                      |
| min_matches      | 20                       | Minimum matches to accept a pose estimate    |
| lowe_ratio       | 0.75                     | Lowe ratio-test threshold                    |
| fx/fy/cx/cy      | approx 1080p wide        | Camera intrinsics (override with calib data) |
| output_dir       | slam_output/             | Directory for PLY + NPY outputs              |

## Using the Navigator

    from slam_pipeline import SLAMConfig, run_pipeline

    slam_map, navigator = run_pipeline(SLAMConfig(num_videos=5, frames_per_video=3))

    print(navigator.is_free(0.0, 0.0))    # True / False
    print(navigator.grid.shape)            # (cols, rows) of occupancy grid

## Troubleshooting

| Error | Fix |
|-------|-----|
| KeyError: 'HF_TOKEN' | export HF_TOKEN=hf_xxxx before running any scripts |
| 401 Unauthorized | Re-accept the NVIDIA license and regenerate your token |
| ModuleNotFoundError: dracopy | Expected — use draco_decoder binary instead |
| draco_decoder not found | find ~/.conda/envs/<env> -name draco_decoder |
| No space left on device | df -h /scratch/$USER to check available space |
| Clip UUID not found in zip | Print zip contents to find a valid UUID for your chunk |
| Zero points projected | Print T_lidar_to_cam and verify Z values are positive |
| gnome-ssh-askpass error on git push | eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519 |

## Technical Notes

### LiDAR decoding
The dracopy Python package fails to import on Explorer despite being installable
— the compiled .so extension is missing from the wheel. The pipeline uses the
draco_decoder CLI binary instead, writing each spin to a temp .drc file and
parsing the resulting PLY directly.

### ftheta camera model
The front-wide 120-degree camera uses NVIDIA's ftheta fisheye model, not
standard pinhole projection:

    theta = arccos(Zc / sqrt(Xc^2 + Yc^2 + Zc^2))
    r_pix = a2p[0] + a2p[1]*theta + a2p[2]*theta^2 + ...
    u = cx + r_pix * (Xc / sqrt(Xc^2 + Yc^2))
    v = cy + r_pix * (Yc / sqrt(Xc^2 + Yc^2))

Coefficients are in camera_intrinsics.offline parquet under model_parameters.

### Coordinate frames
Extrinsics are stored as quaternion + translation in the vehicle frame:

    T_lidar_to_cam = inv(T_cam_in_vehicle) @ T_lidar_in_vehicle

### SLAM design notes
Poses are chained within a clip but there is no loop closure or bundle
adjustment, so drift accumulates across long sequences. Scale is not metric
— Essential matrix decomposition produces unit-translation poses. The
occupancy grid projects points onto the XZ plane at 0.5m cell resolution.
