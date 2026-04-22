const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageBreak, UnderlineType
} = require('docx');
const fs = require('fs');

// ── Styles ───────────────────────────────────────────────────────────
const BLUE   = "1a4f8a";
const GRAY   = "4a4a4a";
const LGRAY  = "f4f6f9";
const MGRAY  = "d0d7e3";
const WHITE  = "FFFFFF";
const GREEN  = "1e6b3c";
const border = { style: BorderStyle.SINGLE, size: 1, color: MGRAY };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: WHITE };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 4 } },
    children: [new TextRun({ text, bold: true, size: 32, color: BLUE, font: "Calibri" })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, bold: true, size: 26, color: BLUE, font: "Calibri" })]
  });
}

function h3(text) {
  return new Paragraph({
    spacing: { before: 180, after: 60 },
    children: [new TextRun({ text, bold: true, size: 22, color: GRAY, font: "Calibri" })]
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 100 },
    children: [new TextRun({ text, size: 20, color: GRAY, font: "Calibri", ...opts })]
  });
}

function pRuns(runs) {
  return new Paragraph({
    spacing: { after: 100 },
    children: runs.map(r =>
      typeof r === "string"
        ? new TextRun({ text: r, size: 20, color: GRAY, font: "Calibri" })
        : new TextRun({ size: 20, font: "Calibri", color: GRAY, ...r })
    )
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 20, color: GRAY, font: "Calibri" })]
  });
}

function bulletRuns(runs, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 80 },
    children: runs.map(r =>
      typeof r === "string"
        ? new TextRun({ text: r, size: 20, color: GRAY, font: "Calibri" })
        : new TextRun({ size: 20, font: "Calibri", color: GRAY, ...r })
    )
  });
}

function code(text) {
  return new Paragraph({
    spacing: { after: 0, before: 0 },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "1e3a5f" })]
  });
}

function codeBlock(lines) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders: noBorders,
        shading: { fill: "eef2f7", type: ShadingType.CLEAR },
        margins: { top: 160, bottom: 160, left: 200, right: 200 },
        width: { size: 9360, type: WidthType.DXA },
        children: lines.map(l => code(l))
      })]
    })]
  });
}

function spacer() {
  return new Paragraph({ spacing: { after: 120 }, children: [] });
}

function noteBox(label, text, color = "fff3cd", borderColor = "e6a817") {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders: {
          top: { style: BorderStyle.SINGLE, size: 4, color: borderColor },
          bottom: { style: BorderStyle.SINGLE, size: 4, color: borderColor },
          left: { style: BorderStyle.SINGLE, size: 16, color: borderColor },
          right: { style: BorderStyle.SINGLE, size: 4, color: borderColor },
        },
        shading: { fill: color, type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 180, right: 180 },
        width: { size: 9360, type: WidthType.DXA },
        children: [new Paragraph({
          children: [
            new TextRun({ text: label + " ", bold: true, size: 20, font: "Calibri", color: GRAY }),
            new TextRun({ text, size: 20, font: "Calibri", color: GRAY })
          ]
        })]
      })]
    })]
  });
}

function stepHeader(num, title) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [640, 8720],
    rows: [new TableRow({
      children: [
        new TableCell({
          borders: noBorders,
          shading: { fill: BLUE, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          width: { size: 640, type: WidthType.DXA },
          verticalAlign: "center",
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: String(num), bold: true, size: 26, color: WHITE, font: "Calibri" })]
          })]
        }),
        new TableCell({
          borders: noBorders,
          shading: { fill: LGRAY, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 180, right: 120 },
          width: { size: 8720, type: WidthType.DXA },
          children: [new Paragraph({
            children: [new TextRun({ text: title, bold: true, size: 24, color: BLUE, font: "Calibri" })]
          })]
        })
      ]
    })]
  });
}

// ── Document ─────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }, {
        level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 360 } } }
      }]
    }]
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 20, color: GRAY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Calibri", color: BLUE },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: BLUE },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [

      // ── TITLE PAGE ────────────────────────────────────────────────
      new Paragraph({
        spacing: { before: 480, after: 120 },
        children: [new TextRun({ text: "Camera-LiDAR Fusion Pipeline", bold: true, size: 56, color: BLUE, font: "Calibri" })]
      }),
      new Paragraph({
        spacing: { after: 80 },
        children: [new TextRun({ text: "Deployment Guide", size: 32, color: GRAY, font: "Calibri" })]
      }),
      new Paragraph({
        spacing: { after: 80 },
        children: [new TextRun({ text: "NVIDIA PhysicalAI-Autonomous-Vehicles Dataset", size: 22, color: MGRAY, font: "Calibri" })]
      }),
      new Paragraph({
        spacing: { after: 480 },
        children: [new TextRun({ text: "Northeastern University Explorer HPC Cluster", size: 22, color: MGRAY, font: "Calibri" })]
      }),

      // Summary table
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2400, 6960],
        rows: [
          ["Output", "Colorized 3D point cloud (.ply), bird's-eye-view (.png), projection overlay (.png)"],
          ["Dataset", "nvidia/PhysicalAI-Autonomous-Vehicles (HuggingFace)"],
          ["Cluster", "Explorer — login.explorer.northeastern.edu"],
          ["Compute", "CPU only, no GPU required"],
          ["Storage", "~32 GB per chunk (camera + LiDAR)"],
          ["Runtime", "~5 minutes after download"],
        ].map(([k, v]) => new TableRow({
          children: [
            new TableCell({
              borders, width: { size: 2400, type: WidthType.DXA },
              shading: { fill: LGRAY, type: ShadingType.CLEAR },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: k, bold: true, size: 20, font: "Calibri", color: BLUE })] })]
            }),
            new TableCell({
              borders, width: { size: 6960, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ children: [new TextRun({ text: v, size: 20, font: "Calibri", color: GRAY })] })]
            })
          ]
        }))
      }),

      spacer(),
      new Paragraph({ children: [new PageBreak()] }),

      // ── PREREQUISITES ─────────────────────────────────────────────
      h1("Prerequisites"),
      h2("HuggingFace Access"),
      p("The dataset requires accepting NVIDIA's license agreement before downloading."),
      bullet("Create a HuggingFace account at huggingface.co if you do not already have one"),
      bullet("Navigate to huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"),
      bullet("Click the license agreement banner and accept the NVIDIA Autonomous Vehicle Dataset License"),
      bullet("Go to huggingface.co/settings/tokens and create a new token with Read access"),
      bullet("Copy the token — it will look like: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
      spacer(),

      h2("Cluster Access"),
      bullet("Ensure you have an active Northeastern University account"),
      bullet("Verify you have been added to a PI storage space on Explorer"),
      bulletRuns(["SSH into the login node: ", { text: "ssh <yournetid>@login.explorer.northeastern.edu", font: "Courier New", size: 18, color: "1e3a5f" }]),
      spacer(),

      h2("Conda Environment"),
      p("This project uses an existing conda environment called ml_proj with Python 3.10. The following packages must be present:"),
      bullet("pandas, numpy, opencv-python, open3d, huggingface_hub, pyarrow"),
      bullet("Note: dracopy is not used despite being installed — the native draco_decoder binary is used instead"),
      spacer(),

      new Paragraph({ children: [new PageBreak()] }),

      // ── STEP BY STEP ─────────────────────────────────────────────
      h1("Step-by-Step Deployment"),
      spacer(),

      // Step 1
      stepHeader(1, "SSH into the cluster and set up workspace"),
      spacer(),
      codeBlock([
        "ssh <yournetid>@login.explorer.northeastern.edu",
        "mkdir -p /scratch/$USER/lidar_fusion",
        "cd /scratch/$USER/lidar_fusion",
      ]),
      spacer(),

      // Step 2
      stepHeader(2, "Activate the conda environment"),
      spacer(),
      codeBlock([
        "conda activate ml_proj",
        "which python3   # should show: /home/<user>/.conda/envs/ml_proj/bin/python3",
      ]),
      spacer(),

      // Step 3
      stepHeader(3, "Export your HuggingFace token"),
      spacer(),
      codeBlock([
        "export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "",
        "# To persist across sessions, add to ~/.bashrc:",
        "echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> ~/.bashrc",
        "source ~/.bashrc",
      ]),
      spacer(),
      noteBox("Note:", "Replace hf_xxxxxxxxxxxxxxxx with your actual token from huggingface.co/settings/tokens"),
      spacer(),

      // Step 4
      stepHeader(4, "Request an interactive compute node"),
      spacer(),
      p("Do not run downloads or computation on the login node. Request a compute node first:"),
      spacer(),
      codeBlock([
        "srun --partition=courses-gpu --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash",
      ]),
      spacer(),
      p("Once allocated, re-activate the environment and re-export the token:"),
      spacer(),
      codeBlock([
        "conda activate ml_proj",
        "export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      ]),
      spacer(),

      // Step 5
      stepHeader(5, "Download calibration metadata"),
      spacer(),
      p("The calibration files are small and download quickly. Run the following to fetch chunk 0 of both intrinsics and extrinsics:"),
      spacer(),
      codeBlock([
        "~/.conda/envs/ml_proj/bin/python3 - <<'EOF'",
        "from huggingface_hub import hf_hub_download, login",
        "import os",
        "login(token=os.environ['HF_TOKEN'])",
        "base = '/scratch/' + os.environ['USER'] + '/lidar_fusion'",
        "for f in [",
        "    'calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet',",
        "    'calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet',",
        "]:",
        "    hf_hub_download(repo_id='nvidia/PhysicalAI-Autonomous-Vehicles',",
        "                    repo_type='dataset', filename=f, local_dir=base)",
        "    print(f'Downloaded {f}')",
        "EOF",
      ]),
      spacer(),

      // Step 6
      stepHeader(6, "Download camera and LiDAR chunk 0"),
      spacer(),
      noteBox("Warning:", "Chunk 0 is approximately 32 GB. Ensure you have sufficient scratch space before downloading. This step takes 1-2 minutes on the cluster's network."),
      spacer(),
      codeBlock([
        "~/.conda/envs/ml_proj/bin/python3 - <<'EOF'",
        "from huggingface_hub import hf_hub_download, login",
        "import os",
        "login(token=os.environ['HF_TOKEN'])",
        "base = '/scratch/' + os.environ['USER'] + '/lidar_fusion'",
        "for f in [",
        "    'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip',",
        "    'lidar/lidar_top_360fov/lidar_top_360fov.chunk_0000.zip',",
        "]:",
        "    print(f'Downloading {f}...')",
        "    hf_hub_download(repo_id='nvidia/PhysicalAI-Autonomous-Vehicles',",
        "                    repo_type='dataset', filename=f, local_dir=base)",
        "    print('Done.')",
        "EOF",
      ]),
      spacer(),

      new Paragraph({ children: [new PageBreak()] }),

      // Step 7
      stepHeader(7, "Run the fusion pipeline"),
      spacer(),
      p("The following script extracts clip files from the downloaded zips, decodes LiDAR point clouds using the native draco_decoder binary, projects LiDAR points into the camera image using the ftheta camera model, colorizes the point cloud, and saves three output files."),
      spacer(),
      noteBox("Note:", "The clip UUID 25cd4769-5dcf-4b53-a351-bf2c5deb6124 is the first clip in chunk 0 for the United States. This is hardcoded and known to be valid."),
      spacer(),
      codeBlock([
        "~/.conda/envs/ml_proj/bin/python3 - <<'EOF'",
        "import pandas as pd, numpy as np, json, os, cv2, subprocess, tempfile",
        "from pathlib import Path",
        "",
        "base      = Path('/scratch/' + os.environ['USER'] + '/lidar_fusion')",
        "out_dir   = base / 'output'",
        "clip_uuid = '25cd4769-5dcf-4b53-a351-bf2c5deb6124'",
        "decoder   = Path(os.path.expanduser('~/.conda/envs/ml_proj/bin/draco_decoder'))",
        "os.makedirs(out_dir, exist_ok=True)",
        "",
        "# --- Extract clip files from chunk zips ---",
        "import zipfile",
        "def extract_clip(zip_path, uuid, dest):",
        "    with zipfile.ZipFile(zip_path) as zf:",
        "        match = next((n for n in zf.namelist() if uuid in n), None)",
        "        out = dest / os.path.basename(match)",
        "        with zf.open(match) as s, open(out, 'wb') as d: d.write(s.read())",
        "        return out",
        "",
        "cam_zip   = base / 'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip'",
        "lidar_zip = base / 'lidar/lidar_top_360fov/lidar_top_360fov.chunk_0000.zip'",
        "cam_mp4   = extract_clip(cam_zip,   clip_uuid, out_dir)",
        "lidar_pq  = extract_clip(lidar_zip, clip_uuid, out_dir)",
        "print('Extracted camera and LiDAR files')",
        "",
        "# --- Draco decode ---",
        "def decode_draco(raw):",
        "    with tempfile.TemporaryDirectory() as tmp:",
        "        inp = os.path.join(tmp,'s.drc'); outp = os.path.join(tmp,'s.ply')",
        "        open(inp,'wb').write(raw)",
        "        subprocess.run([str(decoder),'-i',inp,'-o',outp],check=True,capture_output=True)",
        "        content = open(outp,'rb').read()",
        "        hend = content.index(b'end_header\\n') + len(b'end_header\\n')",
        "        hdr  = content[:hend].decode().split('\\n')",
        "        nv   = int(next(l for l in hdr if l.startswith('element vertex')).split()[-1])",
        "        np_  = len([l for l in hdr if l.startswith('property float')])",
        "        return np.frombuffer(content[hend:],dtype=np.float32)[:nv*np_].reshape(nv,np_)",
        "",
        "df_lidar = pd.read_parquet(lidar_pq)",
        "points   = np.concatenate([decode_draco(bytes(r)) for r in df_lidar['draco_encoded_pointcloud'].iloc[:10]])",
        "print(f'Loaded {points.shape[0]:,} points')",
        "",
        "# --- Calibration ---",
        "intr_df = pd.read_parquet(base/'calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet').reset_index()",
        "ext_df  = pd.read_parquet(base/'calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet').reset_index()",
        "params  = json.loads(intr_df[(intr_df.clip_id==clip_uuid)&(intr_df.camera_name=='camera_front_wide_120fov')].iloc[0].model_parameters)",
        "cx,cy   = params['principal_point']",
        "a2p     = params['angle_to_pixeldist_poly']",
        "max_a   = params['max_angle']",
        "W,H     = params['resolution']",
        "",
        "def quat_to_R(qx,qy,qz,qw):",
        "    return np.array([[1-2*(qy**2+qz**2),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],",
        "                     [2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),2*(qy*qz-qx*qw)],",
        "                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)]])",
        "def get_T(df,clip,sensor):",
        "    r=df[(df.clip_id==clip)&(df.sensor_name==sensor)].iloc[0]",
        "    T=np.eye(4); T[:3,:3]=quat_to_R(r.qx,r.qy,r.qz,r.qw); T[:3,3]=[r.x,r.y,r.z]",
        "    return T",
        "T = np.linalg.inv(get_T(ext_df,clip_uuid,'camera_front_wide_120fov')) @ get_T(ext_df,clip_uuid,'lidar_top_360fov')",
        "",
        "# --- ftheta projection ---",
        "def poly(c,x): return sum(v*x**i for i,v in enumerate(c))",
        "N=len(points); pts_h=np.hstack([points,np.ones((N,1),dtype=np.float32)])",
        "pc=(T@pts_h.T).T; Xc,Yc,Zc=pc[:,0],pc[:,1],pc[:,2]",
        "v1=Zc>0.1; Xc,Yc,Zc=Xc[v1],Yc[v1],Zc[v1]",
        "theta=np.arccos(np.clip(Zc/np.sqrt(Xc**2+Yc**2+Zc**2),-1,1))",
        "v2=theta<max_a; Xc,Yc,Zc,theta=Xc[v2],Yc[v2],Zc[v2],theta[v2]",
        "r_pix=poly(a2p,theta); r_xy=np.sqrt(Xc**2+Yc**2)+1e-9",
        "u=cx+r_pix*(Xc/r_xy); v_=cy+r_pix*(Yc/r_xy)",
        "d=np.sqrt(Xc**2+Yc**2+Zc**2)",
        "v3=(u>=0)&(u<W)&(v_>=0)&(v_<H)",
        "u,v_,d=u[v3],v_[v3],d[v3]",
        "idx=np.where(v1)[0][v2][v3]; mask=np.zeros(N,dtype=bool); mask[idx]=True",
        "uvd=np.stack([u,v_,d],axis=1)",
        "print(f'Projected {mask.sum():,} / {N:,} points into image')",
        "",
        "# --- Camera frame ---",
        "cap=cv2.VideoCapture(str(cam_mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES,30)",
        "ret,frame=cap.read(); cap.release()",
        "colored_pts=points[mask]",
        "rgb=frame[uvd[:,1].astype(int),uvd[:,0].astype(int)][:,::-1]/255.0",
        "",
        "# --- Bird's-eye-view ---",
        "sz=1000; bev=np.zeros((sz,sz,3),dtype=np.uint8)",
        "rm=50.0; res=0.1; ci=sz//2",
        "x,y=colored_pts[:,0],colored_pts[:,1]",
        "in_r=(np.abs(x)<rm)&(np.abs(y)<rm)",
        "px=((rm-y[in_r])/res).astype(int); py=((rm-x[in_r])/res).astype(int)",
        "bev[py,px]=(rgb[in_r]*255).astype(np.uint8)[:,::-1]",
        "cv2.rectangle(bev,(ci-4,ci-8),(ci+4,ci+8),(0,255,0),-1)",
        "for rm_ in [10,20,30,40,50]: cv2.circle(bev,(ci,ci),int(rm_/res),(80,80,80),1)",
        "cv2.imwrite(str(out_dir/'bev.png'),bev)",
        "",
        "# --- Projection overlay ---",
        "overlay=frame.copy(); dmin,dmax=uvd[:,2].min(),uvd[:,2].max()",
        "for (u_,v__,d_) in uvd[::5]:",
        "    h=int((1-(d_-dmin)/(dmax-dmin+1e-6))*120)",
        "    c=cv2.cvtColor(np.uint8([[[h,255,255]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()",
        "    cv2.circle(overlay,(int(u_),int(v__)),2,c,-1)",
        "cv2.imwrite(str(out_dir/'projection.png'),overlay)",
        "",
        "# --- PLY ---",
        "import open3d as o3d",
        "pcd=o3d.geometry.PointCloud()",
        "pcd.points=o3d.utility.Vector3dVector(colored_pts.astype(np.float64))",
        "pcd.colors=o3d.utility.Vector3dVector(rgb.astype(np.float64))",
        "o3d.io.write_point_cloud(str(out_dir/f'{clip_uuid}.ply'),pcd)",
        "print(f'Done. Outputs in {out_dir}')",
        "EOF",
      ]),
      spacer(),

      new Paragraph({ children: [new PageBreak()] }),

      // ── OUTPUTS ─────────────────────────────────────────────────
      h1("Outputs"),
      spacer(),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2800, 2000, 4560],
        rows: [
          new TableRow({
            children: [
              new TableCell({
                borders, width: { size: 2800, type: WidthType.DXA },
                shading: { fill: BLUE, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: "File", bold: true, size: 20, color: WHITE, font: "Calibri" })] })]
              }),
              new TableCell({
                borders, width: { size: 2000, type: WidthType.DXA },
                shading: { fill: BLUE, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: "Format", bold: true, size: 20, color: WHITE, font: "Calibri" })] })]
              }),
              new TableCell({
                borders, width: { size: 4560, type: WidthType.DXA },
                shading: { fill: BLUE, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: "Description", bold: true, size: 20, color: WHITE, font: "Calibri" })] })]
              }),
            ]
          }),
          ...[
            ["<clip_uuid>.ply", ".ply", "Colorized 3D point cloud. Open in MeshLab, CloudCompare, or Open3D."],
            ["bev.png", ".png", "Bird's-eye-view top-down projection with 10m distance rings and ego vehicle marker."],
            ["projection.png", ".png", "Camera frame with LiDAR points overlaid. Green = near, red = far."],
          ].map(([f, fmt, desc]) => new TableRow({
            children: [
              new TableCell({
                borders, width: { size: 2800, type: WidthType.DXA },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: f, font: "Courier New", size: 18, color: "1e3a5f" })] })]
              }),
              new TableCell({
                borders, width: { size: 2000, type: WidthType.DXA },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [p(fmt)]
              }),
              new TableCell({
                borders, width: { size: 4560, type: WidthType.DXA },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [p(desc)]
              }),
            ]
          }))
        ]
      }),

      spacer(),
      h2("Copying outputs to your local machine"),
      codeBlock([
        "# Run this on your LOCAL machine:",
        "scp <yournetid>@login.explorer.northeastern.edu:/scratch/<yournetid>/lidar_fusion/output/*.png ./",
        "scp <yournetid>@login.explorer.northeastern.edu:/scratch/<yournetid>/lidar_fusion/output/*.ply ./",
      ]),
      spacer(),
      h2("Viewing the .ply point cloud"),
      bullet("MeshLab (recommended): meshlab.net — free, drag and drop the .ply file"),
      bullet("CloudCompare: cloudcompare.org — free, supports large point clouds"),
      bulletRuns(["Open3D (Python): ", { text: "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('file.ply')])", font: "Courier New", size: 16, color: "1e3a5f" }]),
      spacer(),

      new Paragraph({ children: [new PageBreak()] }),

      // ── TROUBLESHOOTING ──────────────────────────────────────────
      h1("Troubleshooting"),
      spacer(),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3600, 5760],
        rows: [
          new TableRow({
            children: [
              new TableCell({
                borders, width: { size: 3600, type: WidthType.DXA },
                shading: { fill: BLUE, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: "Error", bold: true, size: 20, color: WHITE, font: "Calibri" })] })]
              }),
              new TableCell({
                borders, width: { size: 5760, type: WidthType.DXA },
                shading: { fill: BLUE, type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: "Fix", bold: true, size: 20, color: WHITE, font: "Calibri" })] })]
              }),
            ]
          }),
          ...[
            ["KeyError: 'HF_TOKEN'", "Run: export HF_TOKEN=hf_xxxx before running any Python scripts."],
            ["401 Unauthorized from HuggingFace", "Re-accept the NVIDIA license on the dataset page and regenerate your token."],
            ["ModuleNotFoundError: dracopy", "Expected — do not use dracopy. The pipeline uses the draco_decoder binary at ~/.conda/envs/ml_proj/bin/draco_decoder instead."],
            ["draco_decoder not found", "Run: find ~/.conda/envs/ml_proj -name draco_decoder to locate it, then update the decoder path in the script."],
            ["No space left on device", "Chunk 0 is ~32 GB. Check available scratch space with: df -h /scratch/$USER"],
            ["Clip UUID not found in zip", "The UUID is hardcoded for chunk 0. If you downloaded a different chunk, print the zip contents to find a valid UUID."],
            ["Zero points projected", "Check the extrinsic transform. Print T_lidar_to_cam and verify Z values are positive for points in front of the vehicle."],
          ].map(([err, fix]) => new TableRow({
            children: [
              new TableCell({
                borders, width: { size: 3600, type: WidthType.DXA },
                shading: { fill: "fef9f0", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text: err, font: "Courier New", size: 17, color: "8b3a00" })] })]
              }),
              new TableCell({
                borders, width: { size: 5760, type: WidthType.DXA },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [p(fix)]
              }),
            ]
          }))
        ]
      }),

      spacer(),
      spacer(),

      // ── TECHNICAL NOTES ─────────────────────────────────────────
      h1("Technical Notes"),
      h2("Why dracopy is not used"),
      p("The dracopy Python package (v2.0.0) installs successfully but fails to import on the Explorer cluster despite being listed in site-packages. The compiled .so extension is missing from the installed wheel. Instead, the pipeline uses the draco_decoder CLI binary bundled with the draco conda package, writing each LiDAR spin to a temporary .drc file and parsing the resulting PLY output directly."),
      spacer(),

      h2("ftheta camera model"),
      p("The front-wide 120° camera uses NVIDIA's ftheta fisheye lens model rather than a standard pinhole projection. Points are projected using a polynomial mapping from incidence angle to pixel distance from the principal point:"),
      spacer(),
      codeBlock([
        "theta = arccos(Zc / sqrt(Xc^2 + Yc^2 + Zc^2))   # angle from optical axis",
        "r_pix = a2p[0] + a2p[1]*theta + a2p[2]*theta^2 + ...  # pixel radius",
        "u = cx + r_pix * (Xc / sqrt(Xc^2 + Yc^2))",
        "v = cy + r_pix * (Yc / sqrt(Xc^2 + Yc^2))",
      ]),
      spacer(),
      p("The polynomial coefficients are stored in the camera_intrinsics.offline calibration parquet under the model_parameters JSON field."),
      spacer(),

      h2("Coordinate frames"),
      p("All sensor extrinsics are stored as quaternion + translation representing each sensor's pose in the vehicle frame. The LiDAR-to-camera transform is computed as:"),
      spacer(),
      codeBlock([
        "T_lidar_to_cam = inv(T_cam_in_vehicle) @ T_lidar_in_vehicle",
      ]),
      spacer(),
      p("where both T_cam and T_lidar are read directly from the sensor_extrinsics.offline parquet."),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("deployment_guide.docx", buf);
  console.log("Done: deployment_guide.docx");
});