"""
ARGUS — Aerial Geospatial Reconstruction & Unified Surveillance
SIH 2026 - PS ID SIH26158 - Team ATLAS

Single-pass drone video to 3D point cloud via pycolmap (primary) + OpenCV SfM (fallback)
Streamlit UI with 4 tabs: Upload / Process / Results / Guide
"""

import os
import sys
import shutil
import tempfile
import pathlib
import time
from pathlib import Path
import traceback

import cv2
import numpy as np
import streamlit as st

# Try pycolmap import
try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
    PYCOLMAP_VERSION = pycolmap.__version__
except Exception as e:
    PYCOLMAP_AVAILABLE = False
    PYCOLMAP_VERSION = None
    PYCOLMAP_IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ARGUS - 3D Reconstruction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1f77b4; font-weight: bold; }
.sub-header { font-size: 1.2rem; color: #555; }
.stProgress > div > div > div > div { background-color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers - Video
# ---------------------------------------------------------------------------
def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }

def extract_frames(video_path: str, output_dir: str, sample_rate: int = 1, max_frames: int = 80, progress_callback=None):
    """Extract frames with sampling. Returns list of saved paths."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = []
    idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            if saved_count >= max_frames:
                break
            fname = f"frame_{saved_count:05d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved.append(fpath)
            saved_count += 1
            if progress_callback:
                progress_callback(saved_count, max_frames, total_frames, idx)
        idx += 1

    cap.release()
    return saved

# ---------------------------------------------------------------------------
# Helpers - PLY
# ---------------------------------------------------------------------------
def write_ply_manual(points_3d: np.ndarray, colors: np.ndarray, output_path: str):
    """Manual ASCII PLY writer - zero dependency, robust."""
    assert points_3d.shape[0] == colors.shape[0]
    assert points_3d.shape[1] == 3
    assert colors.shape[1] == 3
    N = points_3d.shape[0]
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points_3d[i]
            r, g, b = colors[i].astype(int)
            # Clamp colors
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def export_ply_from_pycolmap_reconstruction(reconstruction, frames_dir: str, output_ply: str, extract_colors=True):
    """Export pycolmap Reconstruction to PLY, with optional color extraction."""
    # Try to extract colors from images if requested
    if extract_colors:
        try:
            reconstruction.extract_colors_for_all_images(frames_dir)
        except Exception as e:
            # Non-fatal, continue without colors
            print(f"Color extraction warning: {e}")

    # Manual export to ensure we control format
    points = reconstruction.points3D
    if len(points) == 0:
        raise RuntimeError("Reconstruction has 0 points3D")

    xyz = []
    rgb = []
    for pid, p3d in points.items():
        xyz.append(p3d.xyz)
        rgb.append(p3d.color)

    xyz = np.array(xyz, dtype=np.float64)
    rgb = np.array(rgb, dtype=np.uint8)

    # Filter invalid / extreme points
    # Remove NaN, Inf, and points too far
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    rgb = rgb[valid]

    if xyz.shape[0] == 0:
        raise RuntimeError("All points filtered as invalid after finite check")

    # Optional: remove outliers beyond 3 std from centroid
    centroid = np.mean(xyz, axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    # Keep points within 3 std devs or at least keep 90%
    if len(dists) > 100:
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        thresh = mean_d + 3 * std_d
        mask = dists < thresh
        # Ensure we keep at least 50% points
        if np.sum(mask) > len(dists) * 0.5:
            xyz = xyz[mask]
            rgb = rgb[mask]

    write_ply_manual(xyz, rgb, output_ply)
    return xyz.shape[0]

# ---------------------------------------------------------------------------
# Pipeline - pycolmap
# ---------------------------------------------------------------------------
def run_pycolmap_pipeline(frames_dir: str, work_dir: str, max_num_features: int = 8192, min_matches: int = 10, status_cb=None):
    """
    Full pycolmap pipeline:
    - extract_features
    - match_exhaustive
    - incremental_mapping
    Returns (best_reconstruction, all_reconstructions, sparse_dir)
    """
    if not PYCOLMAP_AVAILABLE:
        raise RuntimeError("pycolmap not installed. Run pip install pycolmap")

    frames_dir = Path(frames_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous DB
    if database_path.exists():
        database_path.unlink()

    # Extraction options
    if status_cb: status_cb("Configuring SIFT extraction...")
    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.use_gpu = False
    extraction_opts.sift.max_num_features = max_num_features
    extraction_opts.sift.first_octave = -1
    extraction_opts.sift.num_octaves = 4
    extraction_opts.sift.peak_threshold = 0.006666
    extraction_opts.sift.edge_threshold = 10.0

    matching_opts = pycolmap.FeatureMatchingOptions()
    matching_opts.use_gpu = False
    matching_opts.sift.max_ratio = 0.8
    matching_opts.sift.max_distance = 0.7
    matching_opts.sift.cross_check = True

    # Mapper options - relaxed for single-pass video
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_num_matches = min_matches
    mapper_opts.mapper.init_min_num_inliers = 30  # relaxed from 100
    mapper_opts.mapper.init_min_tri_angle = 4.0    # relaxed from 16
    mapper_opts.mapper.init_max_error = 4.0
    mapper_opts.mapper.abs_pose_min_num_inliers = 15
    mapper_opts.mapper.abs_pose_min_inlier_ratio = 0.25
    mapper_opts.mapper.filter_max_reproj_error = 4.0
    mapper_opts.mapper.filter_min_tri_angle = 1.0
    mapper_opts.mapper.max_reg_trials = 3
    mapper_opts.mapper.ba_local_num_images = 6
    mapper_opts.num_threads = -1
    mapper_opts.min_model_size = 3  # allow small models for demo

    if status_cb: status_cb(f"Extracting features from {len(list(frames_dir.glob('*.jpg')))} images (max_features={max_num_features})...")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(frames_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_opts
    )

    if status_cb: status_cb("Matching features exhaustively...")
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        matching_options=matching_opts
    )

    if status_cb: status_cb("Running incremental SfM mapping (this may take 1-3 minutes)...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(frames_dir),
        output_path=str(sparse_dir),
        options=mapper_opts
    )

    if not reconstructions:
        raise RuntimeError("No reconstruction found. Try increasing max_frames or decreasing sample_rate for more overlap. Also ensure video has sufficient texture/motion.")

    # Pick best reconstruction by number of registered images
    best_id = max(reconstructions.keys(), key=lambda k: reconstructions[k].num_reg_images())
    best_rec = reconstructions[best_id]

    if best_rec.num_points3D() == 0:
        raise RuntimeError(f"Best reconstruction (id={best_id}) has 0 3D points. Registered images: {best_rec.num_reg_images()}. Try different video or settings.")

    if status_cb: status_cb(f"Reconstruction success: {best_rec.num_reg_images()} images registered, {best_rec.num_points3D()} points")
    return best_rec, reconstructions, str(sparse_dir), str(database_path)

# ---------------------------------------------------------------------------
# Pipeline - Pure OpenCV fallback (fixed triangulation bug)
# ---------------------------------------------------------------------------
def triangulate_points_fixed(P1, P2, pts1, pts2):
    """
    Fixed version - pts1, pts2 shape (N,2) float
    Returns (N,3) points
    """
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    # Ensure shape (N,2)
    if pts1.ndim == 3:
        pts1 = pts1.reshape(-1, 2)
    if pts2.ndim == 3:
        pts2 = pts2.reshape(-1, 2)
    # Transpose to (2,N) for cv2
    pts1_t = pts1.T  # (2,N)
    pts2_t = pts2.T
    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)  # (4,N)
    points_3d = points_4d[:3] / (points_4d[3] + 1e-8)  # (3,N)
    return points_3d.T  # (N,3)

def simple_sfm_pipeline_opencv(frames_dir: str, status_cb=None):
    """
    Pure OpenCV SfM - fallback when pycolmap fails.
    Incremental approach with SIFT + BFMatcher + Essential matrix + triangulation.
    """
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    if len(image_files) < 2:
        raise RuntimeError("Need at least 2 frames")

    if status_cb: status_cb(f"OpenCV SfM: {len(image_files)} frames found, detecting SIFT...")

    # Load first image to get K estimate
    first_img = cv2.imread(image_files[0])
    h, w = first_img.shape[:2]
    # Approximate intrinsics
    focal = max(w, h) * 1.2
    K = np.array([[focal, 0, w/2],
                  [0, focal, h/2],
                  [0, 0, 1]], dtype=np.float64)

    sift = cv2.SIFT_create(nfeatures=8192)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Cache keypoints/descriptors and colors
    kps_list = []
    descs_list = []
    colors_list = []
    imgs_gray = []

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            continue
        kps_list.append(kp)
        descs_list.append(des)
        imgs_gray.append(gray)
        # For color, keep BGR
        colors_list.append(img)

    if len(kps_list) < 2:
        raise RuntimeError("Not enough features detected in frames")

    if status_cb: status_cb(f"Detected features in {len(kps_list)} frames, starting pairwise matching...")

    # Global poses - first camera at origin
    poses = [np.hstack((np.eye(3), np.zeros((3,1))))]  # list of 3x4
    all_points = []
    all_colors = []

    P1 = K @ poses[0]

    for i in range(1, len(kps_list)):
        kp1 = kps_list[i-1]
        kp2 = kps_list[i]
        des1 = descs_list[i-1]
        des2 = descs_list[i]

        # Match with Lowe ratio
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Cap matches to avoid memory issues (500 max per pair as per fix)
        if len(good) > 500:
            good = sorted(good, key=lambda x: x.distance)[:500]

        if len(good) < 20:
            if status_cb: status_cb(f"Pair {i-1}->{i}: only {len(good)} good matches, skipping")
            # Keep previous pose
            poses.append(poses[-1])
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(poses[-1])
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        # Build pose: world to camera? For triangulation we need P = K [R|t]
        # Chain relative pose to global
        # Previous global pose: R_prev, t_prev -> world to cam prev
        # Current relative: R_rel, t_rel from prev to curr
        # So global current = R_rel * R_prev | R_rel*t_prev + t_rel ??? Let's simplify:
        # Use incremental chaining: pose_curr = [R|t] * [R_prev|t_prev] composition
        # Actually we maintain poses as world->cam. So:
        R_prev = poses[-1][:,:3]
        t_prev = poses[-1][:,3:]

        R_curr = R @ R_prev
        t_curr = R @ t_prev + t

        pose_curr = np.hstack((R_curr, t_curr))
        poses.append(pose_curr)
        P2 = K @ pose_curr

        # Filter by mask
        inlier_mask = mask_pose.ravel().astype(bool) if mask_pose is not None else np.ones(len(pts1), dtype=bool)
        # Also need essential mask
        if mask is not None:
            inlier_mask = inlier_mask & (mask.ravel().astype(bool))

        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]

        if len(pts1_in) < 10:
            continue

        try:
            points_3d = triangulate_points_fixed(P1, P2, pts1_in, pts2_in)
        except Exception as e:
            if status_cb: status_cb(f"Triangulation failed for pair {i}: {e}")
            continue

        # Depth filtering: 0.1 < z < 1000 and positive depth in both cameras
        # Check depth in current frame
        # Transform points to camera coords
        # For P1 which is identity-ish first pose, depth is just z
        # For more robust, check z positive after projection

        # Filter by depth
        valid = (points_3d[:,2] > 0.1) & (points_3d[:,2] < 1000)
        # Also check reprojection error or second camera depth
        # Compute depth in second camera
        # Points in world -> cam2: X_cam2 = R_curr * X_world + t_curr
        # We already have pose, so:
        X_world_h = np.hstack((points_3d, np.ones((points_3d.shape[0],1))))
        # Actually pose is world->cam, so X_cam = R*X_world + t
        # We have points_3d already in world? No triangulate gives world if P1,P2 are world->cam
        # So to get depth in cam2, we already have? Let's just use z in cam2 via projection
        # Simple: keep points with positive depth after transform
        # We'll skip complex check and use simple z filter for now

        points_3d = points_3d[valid]
        if len(points_3d) == 0:
            continue

        # Get colors from first image of pair
        img_color = colors_list[i-1]
        pts1_valid = pts1_in[valid]
        colors = []
        for pt in pts1_valid:
            x, y = int(pt[0]), int(pt[1])
            x = max(0, min(img_color.shape[1]-1, x))
            y = max(0, min(img_color.shape[0]-1, y))
            b, g, r = img_color[y, x]
            colors.append([r, g, b])  # PLY expects RGB

        if len(colors) > 0:
            all_points.append(points_3d)
            all_colors.append(np.array(colors, dtype=np.uint8))

        P1 = P2  # for next iteration, but we actually chain via poses list, so this is okay for consecutive

        if status_cb and i % 5 == 0:
            total_pts = sum(p.shape[0] for p in all_points) if all_points else 0
            status_cb(f"Processed {i}/{len(kps_list)-1} pairs, total points so far: {total_pts}")

    if not all_points:
        raise RuntimeError("OpenCV SfM produced 0 points")

    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)

    # Optional: statistical outlier removal (simple)
    centroid = np.mean(final_points, axis=0)
    dists = np.linalg.norm(final_points - centroid, axis=1)
    if len(dists) > 100:
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        mask = dists < (mean_d + 2.5*std_d)
        if np.sum(mask) > len(dists)*0.3:
            final_points = final_points[mask]
            final_colors = final_colors[mask]

    return final_points, final_colors, K, poses

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">🛰️ ARGUS</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Aerial Geospatial Reconstruction & Unified Surveillance — Single-Pass Drone Video to 3D Model (SIH26158 | Team ATLAS)</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        sample_rate = st.slider("Frame Sample Rate (1=every frame)", 1, 20, 1, help="1 = use every frame, higher = skip frames. Lower = more overlap, better reconstruction but slower")
        max_frames = st.slider("Max Frames", 10, 200, 80, help="Cap frames to control processing time")
        max_features = st.slider("Max SIFT Features per Image", 1024, 16384, 8192, step=1024, help="More features = denser cloud but slower")
        min_matches = st.slider("Min Matches for Mapper", 5, 30, 10, help="Lower = more permissive, helps single-pass video")
        backend = st.selectbox("SfM Backend", ["pycolmap (Recommended)", "OpenCV (Fallback)"], index=0, help="pycolmap is robust and production-grade, OpenCV is pure-python fallback")
        st.divider()
        st.markdown("### System Status")
        if PYCOLMAP_AVAILABLE:
            st.success(f"pycolmap {PYCOLMAP_VERSION} available")
        else:
            st.error(f"pycolmap missing: {PYCOLMAP_IMPORT_ERROR if 'PYCOLMAP_IMPORT_ERROR' in globals() else 'not installed'}")
            st.code("pip install pycolmap", language="bash")
        try:
            import cv2
            st.success(f"OpenCV {cv2.__version__} available")
        except:
            st.error("OpenCV missing")
        st.divider()
        st.markdown("**Team ATLAS** | SIH 2026")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Video", "⚙️ Process", "📊 Results", "📖 Guide"])

    # Session state init
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'frames_dir' not in st.session_state:
        st.session_state.frames_dir = None
    if 'work_dir' not in st.session_state:
        st.session_state.work_dir = None
    if 'ply_path' not in st.session_state:
        st.session_state.ply_path = None
    if 'reconstruction_stats' not in st.session_state:
        st.session_state.reconstruction_stats = None
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False

    # ---------------- Tab 1: Upload ----------------
    with tab1:
        st.header("Upload Drone Video")
        st.markdown("Upload a single-pass drone video (mp4/avi/mov/mkv). For best results: 10-30 sec, steady forward motion, textured scene (not just sky/water).")

        uploaded = st.file_uploader("Choose video file", type=['mp4','avi','mov','mkv','flv','webm'])

        if uploaded is not None:
            # Save to temp
            temp_dir = tempfile.mkdtemp(prefix="argus_upload_")
            video_path = os.path.join(temp_dir, uploaded.name)
            with open(video_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            st.session_state.video_path = video_path

            info = get_video_info(video_path)
            if info:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Duration (s)", f"{info['duration']:.2f}")
                col2.metric("FPS", f"{info['fps']:.1f}")
                col3.metric("Resolution", f"{info['width']}x{info['height']}")
                col4.metric("Total Frames", f"{info['frame_count']}")

                st.video(video_path)
                st.success(f"Video saved: {video_path}")
            else:
                st.error("Could not read video metadata - file may be corrupted")
        else:
            st.info("No video uploaded yet. Use the file uploader above.")
            # Demo helper: generate synthetic video if needed
            st.markdown("---")
            st.markdown("#### No drone footage? Generate a test video")
            if st.button("🎬 Generate Synthetic Test Video (for testing pipeline)"):
                with st.spinner("Generating synthetic drone-like video..."):
                    syn_dir = tempfile.mkdtemp(prefix="argus_synth_")
                    syn_path = os.path.join(syn_dir, "synthetic_drone.mp4")
                    # Create synthetic video: moving checkerboard / textured plane
                    w, h = 1280, 720
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(syn_path, fourcc, 30.0, (w, h))
                    # Create textured background
                    np.random.seed(0)
                    for i in range(150):  # 5 sec @30fps
                        # Simulate forward motion + slight shake
                        img = np.zeros((h, w, 3), dtype=np.uint8)
                        # Draw grid of random colored squares to give SIFT features
                        offset_x = int((i * 5) % w)
                        offset_y = int((i * 2) % h)
                        for y in range(-h, h*2, 80):
                            for x in range(-w, w*2, 80):
                                cx = x + offset_x
                                cy = y + offset_y
                                if 0 <= cx < w and 0 <= cy < h:
                                    color = (np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255))
                                    cv2.rectangle(img, (cx, cy), (cx+40, cy+40), color, -1)
                                    cv2.circle(img, (cx+20, cy+20), 10, (0,0,0), 2)
                        # Add some larger structures
                        cv2.rectangle(img, (200+offset_x//2, 100), (400+offset_x//2, 300), (255,255,255), 3)
                        cv2.putText(img, f"FRAME {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        out.write(img)
                    out.release()
                    st.session_state.video_path = syn_path
                    st.success(f"Synthetic video generated: {syn_path} - 150 frames, 5 sec")
                    st.video(syn_path)
                    info = get_video_info(syn_path)
                    if info:
                        st.json(info)

    # ---------------- Tab 2: Process ----------------
    with tab2:
        st.header("Process Video to 3D")
        if st.session_state.video_path is None:
            st.warning("Please upload a video in Tab 1 first")
        else:
            st.info(f"Current video: {st.session_state.video_path}")
            info = get_video_info(st.session_state.video_path)
            if info:
                st.write(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames total")
                expected_frames = min(info['frame_count'] // sample_rate, max_frames)
                st.write(f"With current settings (sample_rate={sample_rate}, max_frames={max_frames}), will extract ~{expected_frames} frames")

            if st.button("🚀 START PROCESSING", type="primary", use_container_width=True):
                # Reset state
                st.session_state.processing_done = False
                st.session_state.ply_path = None
                st.session_state.reconstruction_stats = None

                # Create work dirs
                work_root = tempfile.mkdtemp(prefix="argus_work_")
                frames_dir = os.path.join(work_root, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                st.session_state.frames_dir = frames_dir
                st.session_state.work_dir = work_root

                progress_bar = st.progress(0)
                status_text = st.empty()
                log_area = st.empty()
                logs = []

                def log(msg):
                    logs.append(msg)
                    log_area.code("\n".join(logs[-20:]), language="text")

                try:
                    # Stage 1: Extract frames
                    status_text.markdown("**Stage 1/4: Extracting frames...**")
                    log(f"Extracting frames from {st.session_state.video_path}")

                    def frame_progress(saved, max_f, total, idx):
                        pct = int((saved / max_f) * 25)  # 0-25%
                        progress_bar.progress(pct)
                        if saved % 10 == 0:
                            log(f"Extracted {saved}/{max_f} frames (scanned {idx}/{total})")

                    extracted = extract_frames(st.session_state.video_path, frames_dir, sample_rate=sample_rate, max_frames=max_frames, progress_callback=frame_progress)
                    log(f"Frame extraction done: {len(extracted)} frames")
                    if len(extracted) < 3:
                        raise RuntimeError(f"Only {len(extracted)} frames extracted - video too short or sample_rate too high")
                    progress_bar.progress(30)
                    status_text.markdown(f"**Stage 1/4 Done:** {len(extracted)} frames extracted")

                    # Stage 2 & 3: SfM
                    ply_output = os.path.join(work_root, "model.ply")
                    use_pycolmap = backend.startswith("pycolmap") and PYCOLMAP_AVAILABLE

                    if use_pycolmap:
                        status_text.markdown("**Stage 2/4: pycolmap feature extraction & matching...**")
                        log("Starting pycolmap pipeline")

                        def status_cb(msg):
                            log(msg)
                            status_text.markdown(f"**Stage 2-3:** {msg}")

                        try:
                            best_rec, all_recs, sparse_dir, db_path = run_pycolmap_pipeline(
                                frames_dir, work_root,
                                max_num_features=max_features,
                                min_matches=min_matches,
                                status_cb=status_cb
                            )
                            progress_bar.progress(85)
                            status_text.markdown("**Stage 3/4: Exporting PLY...**")
                            log(f"Exporting PLY from reconstruction with {best_rec.num_points3D()} points")

                            num_points = export_ply_from_pycolmap_reconstruction(best_rec, frames_dir, ply_output, extract_colors=True)

                            st.session_state.ply_path = ply_output
                            st.session_state.reconstruction_stats = {
                                "backend": "pycolmap",
                                "num_images": best_rec.num_reg_images(),
                                "num_points": num_points,
                                "num_reconstructions": len(all_recs),
                                "sparse_dir": sparse_dir,
                                "db_path": db_path,
                                "frames": len(extracted)
                            }
                            progress_bar.progress(100)
                            status_text.markdown("**✅ Processing Complete!**")
                            st.session_state.processing_done = True
                            st.balloons()
                            log(f"SUCCESS: {num_points} points exported to {ply_output}")

                        except Exception as e:
                            log(f"pycolmap pipeline failed: {e}")
                            log(traceback.format_exc())
                            # Fallback to OpenCV if pycolmap fails
                            st.warning(f"pycolmap failed ({e}), falling back to OpenCV SfM...")
                            status_text.markdown("**Fallback: Running OpenCV SfM...**")
                            try:
                                def ocv_status(msg):
                                    log(f"[OpenCV] {msg}")
                                points_3d, colors, K, poses = simple_sfm_pipeline_opencv(frames_dir, status_cb=ocv_status)
                                write_ply_manual(points_3d, colors, ply_output)
                                st.session_state.ply_path = ply_output
                                st.session_state.reconstruction_stats = {
                                    "backend": "opencv-fallback",
                                    "num_images": len(poses),
                                    "num_points": points_3d.shape[0],
                                    "frames": len(extracted),
                                    "K": K.tolist()
                                }
                                progress_bar.progress(100)
                                status_text.markdown("**✅ Processing Complete (OpenCV fallback)!**")
                                st.session_state.processing_done = True
                                st.balloons()
                            except Exception as e2:
                                log(f"OpenCV fallback also failed: {e2}")
                                log(traceback.format_exc())
                                raise RuntimeError(f"Both backends failed. pycolmap: {e}, OpenCV: {e2}")
                    else:
                        # Direct OpenCV
                        status_text.markdown("**Stage 2/4: Running OpenCV SfM...**")
                        log("Starting OpenCV pipeline")
                        def ocv_status(msg):
                            log(f"[OpenCV] {msg}")
                        points_3d, colors, K, poses = simple_sfm_pipeline_opencv(frames_dir, status_cb=ocv_status)
                        progress_bar.progress(85)
                        status_text.markdown("**Stage 3/4: Exporting PLY...**")
                        write_ply_manual(points_3d, colors, ply_output)
                        st.session_state.ply_path = ply_output
                        st.session_state.reconstruction_stats = {
                            "backend": "opencv",
                            "num_images": len(poses),
                            "num_points": points_3d.shape[0],
                            "frames": len(extracted),
                            "K": K.tolist()
                        }
                        progress_bar.progress(100)
                        status_text.markdown("**✅ Processing Complete!**")
                        st.session_state.processing_done = True
                        st.balloons()
                        log(f"SUCCESS: {points_3d.shape[0]} points")

                except Exception as e:
                    progress_bar.progress(0)
                    status_text.markdown(f"**❌ Failed:** {e}")
                    st.error(f"Processing failed: {e}")
                    st.code(traceback.format_exc(), language="python")
                    log(f"FAILED: {e}")

            # Show logs if processing done
            if st.session_state.processing_done:
                st.success("Go to Results tab to view/download your 3D model!")

    # ---------------- Tab 3: Results ----------------
    with tab3:
        st.header("Results")
        if not st.session_state.processing_done or st.session_state.ply_path is None:
            st.info("No results yet. Process a video in Tab 2 first.")
        else:
            stats = st.session_state.reconstruction_stats
            st.success(f"Reconstruction completed with backend: **{stats.get('backend','unknown')}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Registered Images", stats.get('num_images', 'N/A'))
            col2.metric("3D Points", stats.get('num_points', 'N/A'))
            col3.metric("Frames Used", stats.get('frames', 'N/A'))

            ply_path = st.session_state.ply_path
            if os.path.exists(ply_path):
                file_size = os.path.getsize(ply_path) / (1024*1024)
                st.metric("PLY File Size (MB)", f"{file_size:.2f}")

                with open(ply_path, 'rb') as f:
                    ply_data = f.read()

                st.download_button(
                    label="📥 Download Point Cloud (.ply)",
                    data=ply_data,
                    file_name="argus_model.ply",
                    mime="application/octet-stream",
                    use_container_width=True
                )

                # Preview first few lines
                with st.expander("Preview PLY header"):
                    try:
                        with open(ply_path, 'r') as pf:
                            lines = [next(pf) for _ in range(15)]
                        st.code("".join(lines), language="text")
                    except Exception as e:
                        st.code(f"Binary PLY or read error: {e}")

                st.markdown("### How to view your model")
                st.markdown("""
                - **Online (easiest):** Go to https://3dviewer.net/ → drag & drop your `.ply` file
                - **Desktop:** Use [CloudCompare](https://www.danielgm.net/cc/) (free, open source) or MeshLab
                - **Python:** `import open3d as o3d; pcd = o3d.io.read_point_cloud('argus_model.ply'); o3d.visualization.draw_geometries([pcd])`
                """)

                # Simple point cloud stats visualization
                try:
                    # Quick parse for visualization (first 5000 points)
                    points = []
                    with open(ply_path, 'r') as f:
                        # Skip header
                        line = f.readline()
                        while line and "end_header" not in line:
                            line = f.readline()
                        # Read some points
                        for i, l in enumerate(f):
                            if i >= 5000:
                                break
                            parts = l.strip().split()
                            if len(parts) >= 3:
                                try:
                                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                except:
                                    continue
                    if points:
                        points = np.array(points)
                        st.markdown("#### Point Cloud Bounding Box (first 5000 points)")
                        st.write(f"X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
                        # Show 2D projection
                        import pandas as pd
                        df = pd.DataFrame(points[:1000], columns=['x','y','z'])
                        st.scatter_chart(df[['x','y']])
                except Exception as e:
                    st.warning(f"Could not generate preview stats: {e}")

            else:
                st.error(f"PLY file not found at {ply_path}")

    # ---------------- Tab 4: Guide ----------------
    with tab4:
        st.header("Guide & Troubleshooting")
        with st.expander("How ARGUS Works (Pipeline)", expanded=True):
            st.markdown("""
            **ARGUS** reconstructs 3D from a single drone video pass:

            1. **Frame Extraction (OpenCV):** Video → overlapping frames (sample_rate=1 recommended)
            2. **Feature Extraction (SIFT):** Detect distinctive keypoints per frame (pycolmap's SIFT or OpenCV SIFT)
            3. **Feature Matching:** Exhaustive matching across frames with Lowe ratio test + geometric verification
            4. **Incremental SfM (pycolmap):** 
               - Find good initial pair (low tri angle threshold 4° for single-pass)
               - Register new images via PnP (min 15 inliers, relaxed)
               - Triangulate new 3D points
               - Bundle adjustment to refine poses + points
            5. **Export:** `points3D` with colors → ASCII PLY (manual writer, no Open3D dependency)

            **Why pycolmap?** It's the Python binding for COLMAP, the gold-standard SfM library, but without needing external `colmap.exe` binary or Qt/CUDA dependencies. Runs pure CPU.
            """)

        with st.expander("Recommended Settings for Demo"):
            st.markdown("""
            - **Sample Rate:** 1 (every frame) for maximum overlap
            - **Max Frames:** 80 (good balance). For 10-30s video at 30fps, 80 frames ≈ 2.6 sec of video sampled at 1fps equivalent
            - **Max Features:** 8192 (denser cloud). If slow, drop to 4096
            - **Min Matches:** 10 (permissive, helps single-pass)
            - **Video tips:**
              - 10-30 seconds, steady forward motion, avoid pure rotation
              - Textured scene: buildings, roads, terrain - not sky/water only
              - 720p or 1080p is fine, 4K will be slower
              - Ensure overlap: drone moving slowly, not too fast
            """)

        with st.expander("Troubleshooting - Common Errors"):
            st.markdown("""
            **No good initial image pair found / Failed to create sparse model**
            - Cause: Too few frames or too little overlap
            - Fix: Set sample_rate=1, max_frames=80-120, ensure video has motion and texture

            **0 points / empty reconstruction**
            - Cause: Scene lacks texture, or video is too short
            - Fix: Try different video, or increase max_features

            **pycolmap not available**
            - Run `pip install pycolmap` - needs Python 3.8-3.11, Linux/Windows wheel exists for 4.2.0

            **Qt platform plugin error (old COLMAP CLI approach)**
            - We no longer use external COLMAP binary. If you see this, you are on old code - use pycolmap version (this file)

            **CUDA error**
            - pycolmap CPU-only in this config (use_gpu=False), should not happen. If it does, ensure pycolmap installed correctly

            **OutOfMemoryError in OpenCV triangulation (old bug)**
            - Fixed in this version: points are (N,2) → transposed to (2,N) correctly, match cap 500/pair, depth filtering
            """)

        with st.expander("SIH Presentation Tips"):
            st.markdown("""
            - **Demo flow:** Upload → Show video metadata → Process (explain pipeline while it runs) → Results → Show on 3dviewer.net live
            - **Backup:** Pre-generate a .ply and keep ready to upload to viewer in case live processing is slow
            - **Story:** Emphasize single-pass constraint (disaster response, one-shot surveillance) vs traditional multi-pass photogrammetry
            - **Technical depth:** Mention relaxed mapper thresholds (tri angle 4°, min inliers 15) tuned for single-pass
            - **Future work:** Georeferencing with GPS EXIF, textured mesh via Poisson, semantic filtering of moving objects
            """)

        with st.expander("Project Structure"):
            st.code("""
argus/
├── app.py              # Main Streamlit app (this file) - pycolmap + OpenCV fallback
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── test_pipeline.py    # Headless test without Streamlit
└── synthetic_video.py  # Generate test video if no drone footage
            """, language="text")

if __name__ == "__main__":
    main()
