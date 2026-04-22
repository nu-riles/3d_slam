#!/bin/bash

# Camera-LiDAR Fusion Pipeline - Deployment Script
# Northeastern University Explorer HPC Cluster
#
# Prerequisites:
#   - conda environment activated
#   - HF_TOKEN exported
#   - Running on a compute node (srun --partition=courses ...)

set -e  # exit on error

# ── Workspace ────────────────────────────────────────────────────────
mkdir -p /scratch/$USER/lidar_fusion/output
cd /scratch/$USER/lidar_fusion

# ── Download calibration metadata ───────────────────────────────────
echo "Downloading calibration files..."
python3 - <<'EOF'
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
EOF

# ── Download camera and LiDAR chunk 0 (~32 GB) ──────────────────────
echo "Downloading camera and LiDAR data (this will take a few minutes)..."
python3 - <<'EOF'
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
EOF

# ── Download egomotion for pose stitching ────────────────────────────
echo "Downloading egomotion data..."
python3 - <<'EOF'
from huggingface_hub import hf_hub_download, login
import os
login(token=os.environ['HF_TOKEN'])
base = '/scratch/' + os.environ['USER'] + '/lidar_fusion'
hf_hub_download(repo_id='nvidia/PhysicalAI-Autonomous-Vehicles',
                repo_type='dataset',
                filename='egomotion.offline/egomotion.offline.chunk_0000.parquet',
                local_dir=base)
print('Downloaded egomotion')
EOF

# ── Run the multi-clip fusion pipeline ──────────────────────────────
echo "Running fusion pipeline across 25 clips, 3 frames each..."
python3 - <<'EOF'
import pandas as pd, numpy as np, json, os, cv2, subprocess, tempfile, zipfile
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

NUM_CLIPS   = 25
FRAMES_PER_CLIP = 3

base      = Path('/scratch/' + os.environ['USER'] + '/lidar_fusion')
out_dir   = base / 'output'
decoder   = Path(os.path.expanduser('~/repos/dl/3d_slam/3d_slam/.env/bin/draco_decoder'))
os.makedirs(out_dir, exist_ok=True)

cam_zip   = base / 'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip'
lidar_zip = base / 'lidar/lidar_top_360fov/lidar_top_360fov.chunk_0000.zip'

# --- Load calibration ---
intr_df = pd.read_parquet(base/'calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet').reset_index()
ext_df  = pd.read_parquet(base/'calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet').reset_index()
ego_df  = pd.read_parquet(base/'labels/egomotion.offline/egomotion.offline.chunk_0000.parquet').reset_index()

# --- Get all clip UUIDs from chunk 0 ---
with zipfile.ZipFile(cam_zip) as zf:
    all_uuids = list(dict.fromkeys(
        n.split('.')[0].split('/')[-1] for n in zf.namelist() if n.endswith('.mp4')
    ))
clip_uuids = all_uuids[:NUM_CLIPS]
print(f'Found {len(all_uuids)} clips, processing {len(clip_uuids)}')

# --- Helpers ---
def quat_to_R(qx,qy,qz,qw):
    return np.array([[1-2*(qy**2+qz**2),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)]])

def get_T(df, clip, sensor):
    r = df[(df.clip_id==clip)&(df.sensor_name==sensor)].iloc[0]
    T = np.eye(4); T[:3,:3]=quat_to_R(r.qx,r.qy,r.qz,r.qw); T[:3,3]=[r.x,r.y,r.z]
    return T

def get_ego_T(clip):
    # Use first egomotion entry for the clip as world-frame pose
    rows = ego_df[ego_df.clip_id==clip].sort_values('timestamp_ns')
    if rows.empty:
        return np.eye(4)
    r = rows.iloc[0]
    T = np.eye(4)
    T[:3,:3] = quat_to_R(r.qx, r.qy, r.qz, r.qw)
    T[:3,3]  = [r.x, r.y, r.z]
    return T

def decode_draco(raw):
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp,'s.drc'); outp = os.path.join(tmp,'s.ply')
        open(inp,'wb').write(raw)
        subprocess.run([str(decoder),'-i',inp,'-o',outp],check=True,capture_output=True)
        content = open(outp,'rb').read()
        hend = content.index(b'end_header\n') + len(b'end_header\n')
        hdr  = content[:hend].decode().split('\n')
        nv   = int(next(l for l in hdr if l.startswith('element vertex')).split()[-1])
        np_  = len([l for l in hdr if l.startswith('property float')])
        return np.frombuffer(content[hend:],dtype=np.float32)[:nv*np_].reshape(nv,np_)

def extract_clip(zip_path, uuid, dest):
    with zipfile.ZipFile(zip_path) as zf:
        match = next((n for n in zf.namelist() if uuid in n), None)
        if match is None:
            return None
        out = dest / os.path.basename(match)
        with zf.open(match) as s, open(out, 'wb') as d: d.write(s.read())
        return out

def poly(c, x): return sum(v*x**i for i,v in enumerate(c))

def project_ftheta(points, T_lidar_to_cam, cx, cy, a2p, max_a, W, H):
    N = len(points)
    pts_h = np.hstack([points, np.ones((N,1), dtype=np.float32)])
    pc = (T_lidar_to_cam @ pts_h.T).T
    Xc,Yc,Zc = pc[:,0],pc[:,1],pc[:,2]
    v1 = Zc>0.1; Xc,Yc,Zc = Xc[v1],Yc[v1],Zc[v1]
    theta = np.arccos(np.clip(Zc/np.sqrt(Xc**2+Yc**2+Zc**2),-1,1))
    v2 = theta<max_a; Xc,Yc,Zc,theta = Xc[v2],Yc[v2],Zc[v2],theta[v2]
    r_pix = poly(a2p,theta); r_xy = np.sqrt(Xc**2+Yc**2)+1e-9
    u  = cx + r_pix*(Xc/r_xy)
    v_ = cy + r_pix*(Yc/r_xy)
    d  = np.sqrt(Xc**2+Yc**2+Zc**2)
    v3 = (u>=0)&(u<W)&(v_>=0)&(v_<H)
    u,v_,d = u[v3],v_[v3],d[v3]
    idx = np.where(v1)[0][v2][v3]
    mask = np.zeros(N, dtype=bool); mask[idx] = True
    return np.stack([u,v_,d],axis=1), mask

# --- Accumulate world-frame point cloud ---
all_world_pts = []
all_world_rgb = []

for clip_uuid in tqdm(clip_uuids, desc='Clips'):
    try:
        # Extract files
        cam_mp4  = extract_clip(cam_zip,   clip_uuid, out_dir)
        lidar_pq = extract_clip(lidar_zip, clip_uuid, out_dir)
        if cam_mp4 is None or lidar_pq is None:
            print(f'  Skipping {clip_uuid}: files not found in zip')
            continue

        # Calibration for this clip
        intr_row = intr_df[(intr_df.clip_id==clip_uuid)&(intr_df.camera_name=='camera_front_wide_120fov')]
        if intr_row.empty:
            print(f'  Skipping {clip_uuid}: no intrinsics')
            continue
        params = json.loads(intr_row.iloc[0].model_parameters)
        cx,cy  = params['principal_point']
        a2p    = params['angle_to_pixeldist_poly']
        max_a  = params['max_angle']
        W,H    = params['resolution']

        T_lidar_to_cam = np.linalg.inv(get_T(ext_df,clip_uuid,'camera_front_wide_120fov')) \
                         @ get_T(ext_df,clip_uuid,'lidar_top_360fov')

        # Ego pose in world frame for this clip
        T_ego = get_ego_T(clip_uuid)

        # Decode LiDAR (first 10 spins)
        df_lidar = pd.read_parquet(lidar_pq)
        points   = np.concatenate([decode_draco(bytes(r))
                                   for r in df_lidar['draco_encoded_pointcloud'].iloc[:10]])

        # Sample 3 evenly-spaced frames from the video
        cap   = cv2.VideoCapture(str(cam_mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total-1, FRAMES_PER_CLIP, dtype=int)

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            uvd, mask = project_ftheta(points, T_lidar_to_cam, cx, cy, a2p, max_a, W, H)
            colored_pts = points[mask]

            # Sample RGB
            rgb = frame[uvd[:,1].astype(int), uvd[:,0].astype(int)][:,::-1] / 255.0

            # Transform to world frame via egomotion
            N = len(colored_pts)
            pts_h = np.hstack([colored_pts, np.ones((N,1), dtype=np.float32)])
            world_pts = (T_ego @ pts_h.T).T[:,:3]

            all_world_pts.append(world_pts)
            all_world_rgb.append(rgb)

        cap.release()
        print(f'  {clip_uuid}: {len(colored_pts):,} pts/frame')

    except Exception as e:
        print(f'  Skipping {clip_uuid}: {e}')
        continue

# --- Merge and save global point cloud ---
print(f'\nMerging {sum(len(p) for p in all_world_pts):,} total points...')
all_pts = np.vstack(all_world_pts).astype(np.float64)
all_rgb = np.vstack(all_world_rgb).astype(np.float64)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_pts)
pcd.colors = o3d.utility.Vector3dVector(all_rgb)
pcd = pcd.voxel_down_sample(voxel_size=0.1)
o3d.io.write_point_cloud(str(out_dir/'reconstruction.ply'), pcd)
print(f'Saved reconstruction.ply ({len(pcd.points):,} pts after voxel downsample)')

# --- Global bird's-eye-view ---
pts = np.asarray(pcd.points)
rgb = np.asarray(pcd.colors)
x, y = pts[:,0], pts[:,1]
rm = max(np.abs(x).max(), np.abs(y).max()) + 5
res = 0.2; sz = int(2*rm/res)
bev = np.zeros((sz,sz,3), dtype=np.uint8)
px = ((rm-y)/res).astype(int); py = ((rm-x)/res).astype(int)
valid = (px>=0)&(px<sz)&(py>=0)&(py<sz)
bev[py[valid],px[valid]] = (rgb[valid]*255).astype(np.uint8)[:,::-1]
ci = sz//2
cv2.rectangle(bev,(ci-4,ci-8),(ci+4,ci+8),(0,255,0),-1)
for r_m in np.linspace(10, int(rm), 5):
    cv2.circle(bev,(ci,ci),int(r_m/res),(80,80,80),1)
    cv2.putText(bev,f'{int(r_m)}m',(ci+int(r_m/res)+2,ci),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(120,120,120),1)
cv2.imwrite(str(out_dir/'bev_global.png'), bev)
print(f'Saved bev_global.png')
print(f'\nAll outputs in {out_dir}')
EOF

echo "All done. Outputs in /scratch/$USER/lidar_fusion/output/"
