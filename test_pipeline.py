"""
Headless test for ARGUS pipeline without Streamlit
Usage: python test_pipeline.py --video /path/to/video.mp4
"""

import argparse
import os
import tempfile
import pathlib
import sys
import traceback

import cv2
import numpy as np

# Import pipeline functions from app.py (reuse)
# To avoid Streamlit import side effects, we re-implement minimal versions here
# But we will try to import pycolmap directly

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except:
    PYCOLMAP_AVAILABLE = False

def extract_frames(video_path, output_dir, sample_rate=1, max_frames=80):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
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
            fpath = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved.append(fpath)
            saved_count += 1
        idx += 1
    cap.release()
    return saved

def write_ply(points, colors, path):
    assert points.shape[0] == colors.shape[0]
    N = points.shape[0]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x,y,z = points[i]
            r,g,b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def run_pycolmap(frames_dir, work_dir, max_features=8192, min_matches=10):
    frames_dir = pathlib.Path(frames_dir)
    work_dir = pathlib.Path(work_dir)
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    ext_opts = pycolmap.FeatureExtractionOptions()
    ext_opts.use_gpu = False
    ext_opts.sift.max_num_features = max_features

    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = False

    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_num_matches = min_matches
    mapper_opts.mapper.init_min_num_inliers = 30
    mapper_opts.mapper.init_min_tri_angle = 4.0
    mapper_opts.mapper.abs_pose_min_num_inliers = 15
    mapper_opts.mapper.abs_pose_min_inlier_ratio = 0.25
    mapper_opts.min_model_size = 3

    print(f"[1] Extracting features from {len(list(frames_dir.glob('*.jpg')))} images...")
    pycolmap.extract_features(str(db_path), str(frames_dir), camera_mode=pycolmap.CameraMode.SINGLE, extraction_options=ext_opts)

    print("[2] Matching...")
    pycolmap.match_exhaustive(str(db_path), matching_options=match_opts)

    print("[3] Mapping...")
    recs = pycolmap.incremental_mapping(str(db_path), str(frames_dir), str(sparse_dir), options=mapper_opts)

    if not recs:
        raise RuntimeError("No reconstructions")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"Best rec: {best.num_reg_images()} images, {best.num_points3D()} points")
    return best, recs, sparse_dir, db_path

def export_ply(rec, frames_dir, out_path):
    try:
        rec.extract_colors_for_all_images(frames_dir)
    except Exception as e:
        print(f"Color extraction warning: {e}")
    pts = rec.points3D
    xyz = np.array([p.xyz for p in pts.values()])
    rgb = np.array([p.color for p in pts.values()])
    print(f"Exporting {len(xyz)} points to {out_path}")
    write_ply(xyz, rgb, out_path)
    return len(xyz)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=False, default=None, help="Path to drone video")
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--max_features", type=int, default=8192)
    parser.add_argument("--work_dir", default=None)
    args = parser.parse_args()

    if args.video is None:
        # Generate synthetic if no video
        print("No video provided, generating synthetic...")
        from synthetic_video import generate_synthetic_video
        tmp = tempfile.mkdtemp(prefix="argus_test_")
        video_path = os.path.join(tmp, "synthetic.mp4")
        generate_synthetic_video(video_path, num_frames=150)
        print(f"Generated synthetic video: {video_path}")
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            sys.exit(1)

    work_root = args.work_dir or tempfile.mkdtemp(prefix="argus_test_work_")
    frames_dir = os.path.join(work_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Work dir: {work_root}")
    print(f"Extracting frames: sample_rate={args.sample_rate}, max_frames={args.max_frames}")
    frames = extract_frames(video_path, frames_dir, sample_rate=args.sample_rate, max_frames=args.max_frames)
    print(f"Extracted {len(frames)} frames")

    if len(frames) < 3:
        print("Too few frames")
        sys.exit(1)

    if not PYCOLMAP_AVAILABLE:
        print("pycolmap not available, cannot run full test")
        sys.exit(1)

    try:
        best_rec, all_recs, sparse_dir, db_path = run_pycolmap(frames_dir, work_root, max_features=args.max_features)
        ply_path = os.path.join(work_root, "model.ply")
        n = export_ply(best_rec, frames_dir, ply_path)
        print(f"\n✅ SUCCESS: {n} points exported")
        print(f"PLY: {ply_path}")
        print(f"Size: {os.path.getsize(ply_path)/1024/1024:.2f} MB")
        print(f"To view: https://3dviewer.net/ -> drag & drop {ply_path}")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
