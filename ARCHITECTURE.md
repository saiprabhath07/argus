# ARGUS Architecture - Technical Deep Dive

## Overview
Single-pass drone video → 3D point cloud via Structure-from-Motion

## System Diagram
```
Drone Video (mp4)
    ↓
[Frame Extraction] OpenCV VideoCapture, sample_rate, max_frames
    ↓
Frames (jpg) - 60-80 images with high overlap
    ↓
┌─────────────────────────────────────────┐
│  Primary: pycolmap Pipeline             │
│  - FeatureExtractionOptions (SIFT)      │
│  - SIFT: max_num_features=8192          │
│  - CPU-only (use_gpu=False)             │
│  ↓                                      │
│  - FeatureMatchingOptions               │
│  - Exhaustive matching, ratio 0.8       │
│  - Cross-check True                     │
│  ↓                                      │
│  - IncrementalPipelineOptions           │
│  - Relaxed thresholds for single-pass:  │
│    init_min_tri_angle=4° (vs 16°)       │
│    init_min_num_inliers=30 (vs 100)     │
│    abs_pose_min_inliers=15              │
│    abs_pose_min_ratio=0.25              │
│    min_num_matches=10                   │
│    min_model_size=3                     │
│  ↓                                      │
│  - incremental_mapping()                │
│  - Returns dict[model_id, Reconstruction]│
│  - Pick best by num_reg_images()        │
└─────────────────────────────────────────┘
    ↓ (fallback if pycolmap fails)
┌─────────────────────────────────────────┐
│  Fallback: Pure OpenCV SfM              │
│  - SIFT detection per frame             │
│  - BFMatcher + Lowe ratio 0.75          │
│  - Cap 500 matches/pair (memory fix)    │
│  - Essential matrix + recoverPose       │
│  - triangulatePoints (fixed shape bug)  │
│    Input: (N,2) → transpose to (2,N)   │
│  - Depth filter 0.1 < z < 1000          │
│  - Outlier removal 2.5 sigma            │
└─────────────────────────────────────────┘
    ↓
Reconstruction.points3D (dict of Point3D with xyz, color, track)
    ↓
[Color Extraction] extract_colors_for_all_images()
    ↓
[Outlier Filtering]
    - Finite check (remove NaN/Inf)
    - 3-sigma distance from centroid
    - Keep >50% points minimum
    ↓
[PLY Export] Manual ASCII writer
    - Header: ply, format ascii 1.0, element vertex N, properties
    - No Open3D dependency
    ↓
model.ply (downloadable, viewable on 3dviewer.net)
```

## Why This Design?

### 1. pycolmap over COLMAP CLI
**Old approach (failed):**
- External `colmap.exe` binary via subprocess
- Issues:
  - CUDA build requires NVIDIA GPU (Check failed: num_cuda_devices >0)
  - No-CUDA build missing Qt platform plugins (Could not find Qt plugin "windows"/"offscreen")
  - PATH configuration errors
  - Invalid flag names (FeatureExtraction.use_gpu vs SiftExtraction.use_gpu)
  - Different builds support different flags

**New approach (works):**
- `pip install pycolmap` - self-contained wheel (38MB)
- No external binary, no Qt, no CUDA needed
- In-process, same COLMAP 4.2.0 core
- CPU-only (has_cuda=False on this machine)
- API verified: extract_features, match_exhaustive, incremental_mapping
- Tested end-to-end: 704 points from synthetic video

### 2. Manual PLY Writer over Open3D
**Old approach (failed):**
- Open3D not available on Colab (No matching distribution)
- Heavy dependency

**New approach:**
- 20 lines of Python, zero dependency
- ASCII PLY format: header + x y z r g b per line
- Works everywhere

### 3. Relaxed Mapper Thresholds
Single-pass video has less overlap than multi-pass photogrammetry. Standard COLMAP thresholds are too strict:
- Default init_min_tri_angle=16° → we use 4° (allows smaller baseline)
- Default init_min_num_inliers=100 → we use 30 (allows fewer matches)
- Default abs_pose_min_num_inliers=30 → we use 15
- Default min_num_matches=15 → we use 10
- min_model_size=3 (allow small demo models)

This was learned from Bug 1 in handoff: 22 frames, 231 matches → no initial pair.

### 4. Fixed OpenCV Triangulation Bug
**Old bug:**
```python
pts1.reshape(-1,1,2)  # shape becomes (N,1,2)
pts1.T  # garbage shape
cv2.triangulatePoints(P1,P2, pts1.T, pts2.T) → allocates 18446744073709551600 bytes → OutOfMemory
```

**Fixed:**
```python
pts1 shape (N,2) float32
pts1.T → (2,N) correct for cv2
triangulatePoints returns (4,N) → normalize by w → (N,3)
Add match cap 500/pair + depth filter 0.1<z<1000 + try/except per pair
```

## Data Flow

1. **Input:** mp4 video (10-30 sec, 720p, 30fps typical)
2. **Frame Extraction:** sample_rate=1, max_frames=80 → ~80 jpg @ 95% quality
3. **Database:** SQLite `database.db` with SIFT features + matches
4. **Sparse Model:** `sparse/` folder with COLMAP binary reconstruction
5. **Point Cloud:** `model.ply` ASCII, e.g. 704 points × (x,y,z,r,g,b) = ~28KB
6. **Output:** Download button, view on 3dviewer.net

## Performance

- Synthetic video (100 frames, 1280x720):
  - Frame extraction: ~1 sec for 60 frames
  - Feature extraction: ~20 sec for 60 frames (CPU SIFT, 757 features avg per image)
  - Matching: ~10 sec (exhaustive, 60 choose 2 = 1770 pairs, but COLMAP optimizes)
  - Mapping: ~30 sec (incremental, 18 images registered)
  - Total: ~60 sec for 60 frames on CPU

- Real drone video may be slower (more texture = more features)

- Optimization options:
  - Reduce max_frames to 40-50
  - Reduce max_num_features to 4096
  - Use sequential matching instead of exhaustive (not implemented but easy: pycolmap.match_sequential)

## Error Handling

Each stage wrapped in try/except:
- Frame extraction: check cap.isOpened(), min 3 frames
- pycolmap: check reconstructions non-empty, num_points3D >0
- Fallback: if pycolmap fails, auto-try OpenCV SfM
- PLY export: finite check, outlier removal, manual writer

## Dependencies

- pycolmap 4.2.0 (38MB wheel, includes COLMAP core, no external binary)
- opencv-python-headless 5.0.0.93 (73MB, no libGL needed)
- numpy 2.4.6
- streamlit 1.63.0
- scipy (for Rotation, though not strictly needed - could be removed)
- Total venv size: ~500MB

## Testing

- `test_pipeline.py --video /tmp/synth.mp4` → headless test, no Streamlit
- `synthetic_video.py` → generates test video with known features
- `sample_model.ply` → 704 points from successful run, for demo backup

## Future Work

- GPS EXIF georeferencing (pycolmap supports prior positions)
- Poisson meshing for textured mesh (open3d or pycolmap.poisson_meshing)
- Semantic filtering (detectron2 to mask dynamic objects)
- Real-time: reduce frames, use sequential matching, GPU if available
- Edge deployment: optimize for Jetson etc.

## References

- COLMAP: https://colmap.github.io/ (Schönberger & Frahm 2016)
- pycolmap: https://github.com/colmap/pycolmap
- SIFT: Lowe 2004
- SfM: Hartley & Zisserman Multiple View Geometry
