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

# ── Run the fusion pipeline ──────────────────────────────────────────
echo "Running fusion pipeline..."
python3 - <<'EOF'
import pandas as pd, numpy as np, json, os, cv2, subprocess, tempfile
from pathlib import Path

base      = Path('/scratch/' + os.environ['USER'] + '/lidar_fusion')
out_dir   = base / 'output'
clip_uuid = '25cd4769-5dcf-4b53-a351-bf2c5deb6124'
decoder   = Path(os.path.expanduser('~/repos/dl/3d_slam/3d_slam/.env/bin/draco_decoder'))
os.makedirs(out_dir, exist_ok=True)

# --- Extract clip files from chunk zips ---
import zipfile
def extract_clip(zip_path, uuid, dest):
    with zipfile.ZipFile(zip_path) as zf:
        match = next((n for n in zf.namelist() if uuid in n), None)
        out = dest / os.path.basename(match)
        with zf.open(match) as s, open(out, 'wb') as d: d.write(s.read())
        return out

cam_zip   = base / 'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip'
lidar_zip = base / 'lidar/lidar_top_360fov/lidar_top_360fov.chunk_0000.zip'
cam_mp4   = extract_clip(cam_zip,   clip_uuid, out_dir)
lidar_pq  = extract_clip(lidar_zip, clip_uuid, out_dir)
print('Extracted camera and LiDAR files')

# --- Draco decode ---
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

df_lidar = pd.read_parquet(lidar_pq)
points   = np.concatenate([decode_draco(bytes(r)) for r in df_lidar['draco_encoded_pointcloud'].iloc[:10]])
print(f'Loaded {points.shape[0]:,} points')

# --- Calibration ---
intr_df = pd.read_parquet(base/'calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet').reset_index()
ext_df  = pd.read_parquet(base/'calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet').reset_index()
params  = json.loads(intr_df[(intr_df.clip_id==clip_uuid)&(intr_df.camera_name=='camera_front_wide_120fov')].iloc[0].model_parameters)
cx,cy   = params['principal_point']
a2p     = params['angle_to_pixeldist_poly']
max_a   = params['max_angle']
W,H     = params['resolution']

def quat_to_R(qx,qy,qz,qw):
    return np.array([[1-2*(qy**2+qz**2),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)]])
def get_T(df,clip,sensor):
    r=df[(df.clip_id==clip)&(df.sensor_name==sensor)].iloc[0]
    T=np.eye(4); T[:3,:3]=quat_to_R(r.qx,r.qy,r.qz,r.qw); T[:3,3]=[r.x,r.y,r.z]
    return T
T = np.linalg.inv(get_T(ext_df,clip_uuid,'camera_front_wide_120fov')) @ get_T(ext_df,clip_uuid,'lidar_top_360fov')

# --- ftheta projection ---
def poly(c,x): return sum(v*x**i for i,v in enumerate(c))
N=len(points); pts_h=np.hstack([points,np.ones((N,1),dtype=np.float32)])
pc=(T@pts_h.T).T; Xc,Yc,Zc=pc[:,0],pc[:,1],pc[:,2]
v1=Zc>0.1; Xc,Yc,Zc=Xc[v1],Yc[v1],Zc[v1]
theta=np.arccos(np.clip(Zc/np.sqrt(Xc**2+Yc**2+Zc**2),-1,1))
v2=theta<max_a; Xc,Yc,Zc,theta=Xc[v2],Yc[v2],Zc[v2],theta[v2]
r_pix=poly(a2p,theta); r_xy=np.sqrt(Xc**2+Yc**2)+1e-9
u=cx+r_pix*(Xc/r_xy); v_=cy+r_pix*(Yc/r_xy)
d=np.sqrt(Xc**2+Yc**2+Zc**2)
v3=(u>=0)&(u<W)&(v_>=0)&(v_<H)
u,v_,d=u[v3],v_[v3],d[v3]
idx=np.where(v1)[0][v2][v3]; mask=np.zeros(N,dtype=bool); mask[idx]=True
uvd=np.stack([u,v_,d],axis=1)
print(f'Projected {mask.sum():,} / {N:,} points into image')

# --- Camera frame ---
cap=cv2.VideoCapture(str(cam_mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES,30)
ret,frame=cap.read(); cap.release()
colored_pts=points[mask]
rgb=frame[uvd[:,1].astype(int),uvd[:,0].astype(int)][:,::-1]/255.0

# --- Bird's-eye-view ---
sz=1000; bev=np.zeros((sz,sz,3),dtype=np.uint8)
rm=50.0; res=0.1; ci=sz//2
x,y=colored_pts[:,0],colored_pts[:,1]
in_r=(np.abs(x)<rm)&(np.abs(y)<rm)
px=((rm-y[in_r])/res).astype(int); py=((rm-x[in_r])/res).astype(int)
bev[py,px]=(rgb[in_r]*255).astype(np.uint8)[:,::-1]
cv2.rectangle(bev,(ci-4,ci-8),(ci+4,ci+8),(0,255,0),-1)
for rm_ in [10,20,30,40,50]: cv2.circle(bev,(ci,ci),int(rm_/res),(80,80,80),1)
cv2.imwrite(str(out_dir/'bev.png'),bev)

# --- Projection overlay ---
overlay=frame.copy(); dmin,dmax=uvd[:,2].min(),uvd[:,2].max()
for (u_,v__,d_) in uvd[::5]:
    h=int((1-(d_-dmin)/(dmax-dmin+1e-6))*120)
    c=cv2.cvtColor(np.uint8([[[h,255,255]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
    cv2.circle(overlay,(int(u_),int(v__)),2,c,-1)
cv2.imwrite(str(out_dir/'projection.png'),overlay)

# --- PLY ---
import open3d as o3d
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(colored_pts.astype(np.float64))
pcd.colors=o3d.utility.Vector3dVector(rgb.astype(np.float64))
o3d.io.write_point_cloud(str(out_dir/f'{clip_uuid}.ply'),pcd)
print(f'Done. Outputs in {out_dir}')
EOF

echo "All done. Outputs in /scratch/$USER/lidar_fusion/output/"