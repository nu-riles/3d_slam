~/.conda/envs/ml_proj/bin/python3 - <<'EOF'
import pandas as pd, numpy as np, json, os, cv2, subprocess, tempfile
from pathlib import Path

base      = Path("/scratch/jacob.an/lidar_fusion")
out_dir   = base / "output"
clip_uuid = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
decoder   = Path(os.path.expanduser("~/.conda/envs/ml_proj/bin/draco_decoder"))

# ── draco decoder ────────────────────────────────────────────────────
def decode_draco(raw_bytes):
    with tempfile.TemporaryDirectory() as tmp:
        in_path  = os.path.join(tmp, "spin.drc")
        out_path = os.path.join(tmp, "spin.ply")
        with open(in_path, "wb") as f:
            f.write(raw_bytes)
        subprocess.run([str(decoder), "-i", in_path, "-o", out_path],
                       check=True, capture_output=True)
        with open(out_path, "rb") as f:
            content = f.read()
        header_end = content.index(b"end_header\n") + len(b"end_header\n")
        header     = content[:header_end].decode()
        lines      = header.split("\n")
        n_verts    = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        props      = [l.split()[-1] for l in lines if l.startswith("property float")]
        data       = np.frombuffer(content[header_end:], dtype=np.float32)
        data       = data[:n_verts * len(props)].reshape(n_verts, len(props))
        return data  # (N, 3) xyz

# ── load lidar (first 10 spins ~1 sec) ──────────────────────────────
df_lidar = pd.read_parquet(out_dir / f"{clip_uuid}.lidar_top_360fov.parquet")
print(f"Decoding {min(10, len(df_lidar))} spins...")
all_pts = []
for raw in df_lidar["draco_encoded_pointcloud"].iloc[:10]:
    all_pts.append(decode_draco(bytes(raw)))
points = np.concatenate(all_pts, axis=0)
print(f"Total points: {points.shape[0]:,}")

# ── calibration ──────────────────────────────────────────────────────
intr_df = pd.read_parquet(
    base / "calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet"
).reset_index()
ext_df = pd.read_parquet(
    base / "calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet"
).reset_index()

row    = intr_df[(intr_df["clip_id"]==clip_uuid) &
                 (intr_df["camera_name"]=="camera_front_wide_120fov")].iloc[0]
params = json.loads(row["model_parameters"])
cx, cy    = params["principal_point"]
a2p       = params["angle_to_pixeldist_poly"]
max_angle = params["max_angle"]
W, H      = params["resolution"]

def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def get_T(df, clip, sensor):
    r  = df[(df["clip_id"]==clip) & (df["sensor_name"]==sensor)].iloc[0]
    R  = quat_to_R(r.qx, r.qy, r.qz, r.qw)
    T  = np.eye(4); T[:3,:3]=R; T[:3,3]=[r.x, r.y, r.z]
    return T

T_lidar_to_cam = np.linalg.inv(get_T(ext_df, clip_uuid, "camera_front_wide_120fov")) @ \
                               get_T(ext_df, clip_uuid, "lidar_top_360fov")

# ── ftheta projection ────────────────────────────────────────────────
def poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))

def project_ftheta(pts, T, cx, cy, a2p, max_angle, W, H):
    N     = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N,1), dtype=np.float32)])
    pc    = (T @ pts_h.T).T
    Xc, Yc, Zc = pc[:,0], pc[:,1], pc[:,2]
    v1    = Zc > 0.1
    Xc, Yc, Zc = Xc[v1], Yc[v1], Zc[v1]
    theta = np.arccos(np.clip(Zc / np.sqrt(Xc**2+Yc**2+Zc**2), -1, 1))
    v2    = theta < max_angle
    Xc, Yc, Zc, theta = Xc[v2], Yc[v2], Zc[v2], theta[v2]
    r_pix = poly(a2p, theta)
    r_xy  = np.sqrt(Xc**2+Yc**2) + 1e-9
    u = cx + r_pix * (Xc/r_xy)
    v = cy + r_pix * (Yc/r_xy)
    d = np.sqrt(Xc**2+Yc**2+Zc**2)
    v3    = (u>=0)&(u<W)&(v>=0)&(v<H)
    u, v, d = u[v3], v[v3], d[v3]
    idx   = np.where(v1)[0][v2][v3]
    mask  = np.zeros(N, dtype=bool); mask[idx]=True
    print(f"Projected {mask.sum():,} / {N:,} points into image")
    return np.stack([u,v,d],axis=1), mask

uvd, mask = project_ftheta(points, T_lidar_to_cam, cx, cy, a2p, max_angle, W, H)

# ── camera frame ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(out_dir / f"{clip_uuid}.camera_front_wide_120fov.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
print(f"Camera frame: {frame.shape}")

# ── colorize ─────────────────────────────────────────────────────────
colored_pts = points[mask]
rgb = frame[uvd[:,1].astype(int), uvd[:,0].astype(int)][:, ::-1] / 255.0

# ── bird's-eye-view ──────────────────────────────────────────────────
range_m=50.0; res=0.1; sz=int(2*range_m/res)
bev = np.zeros((sz,sz,3), dtype=np.uint8)
x, y = colored_pts[:,0], colored_pts[:,1]
in_r = (np.abs(x)<range_m)&(np.abs(y)<range_m)
px = ((range_m-y[in_r])/res).astype(int)
py = ((range_m-x[in_r])/res).astype(int)
bev[py,px] = (rgb[in_r]*255).astype(np.uint8)[:,::-1]
ci = sz//2
cv2.rectangle(bev,(ci-4,ci-8),(ci+4,ci+8),(0,255,0),-1)
for r_m in [10,20,30,40,50]:
    cv2.circle(bev,(ci,ci),int(r_m/res),(80,80,80),1)
    cv2.putText(bev,f"{r_m}m",(ci+int(r_m/res)+2,ci),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(120,120,120),1)
cv2.imwrite(str(out_dir/"bev.png"), bev)
print("Saved bev.png")

# ── projection overlay ───────────────────────────────────────────────
overlay = frame.copy()
d_min, d_max = uvd[:,2].min(), uvd[:,2].max()
for (u,v,d) in uvd[::5]:
    norm  = (d-d_min)/(d_max-d_min+1e-6)
    hue   = int((1.0-norm)*120)
    color = cv2.cvtColor(np.uint8([[[hue,255,255]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
    cv2.circle(overlay,(int(u),int(v)),2,color,-1)
cv2.imwrite(str(out_dir/"projection.png"), overlay)
print("Saved projection.png")

# ── ply ──────────────────────────────────────────────────────────────
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(colored_pts.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
o3d.io.write_point_cloud(str(out_dir/f"{clip_uuid}.ply"), pcd)
print(f"Saved {clip_uuid}.ply")

print(f"\nAll outputs in {out_dir}")
EOF