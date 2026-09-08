# ARGUS — Aerial Geospatial Reconstruction & Unified Surveillance
### SIH 2026 | Problem Statement ID SIH26158 | Team ATLAS

> Generate a georeferenced, textured 3D model (point cloud / mesh) from a **single-pass drone video** — no multiple overlapping flights.

Traditional photogrammetry needs multiple drone passes with heavy overlap. Real-world scenarios (disaster response, one-shot surveillance, reconnaissance) often allow only **one flight**. ARGUS solves this.

## Features
- 📤 Upload drone video (mp4/avi/mov/mkv) via Streamlit GUI
- 🎬 Frame extraction with configurable sampling
- 🛰️ **pycolmap backend** (primary) — Python bindings for COLMAP, no external binary, no Qt/CUDA issues
- 🔧 **OpenCV SIFT fallback** — pure Python SfM if pycolmap fails
- 💾 Export point cloud as `.ply` (manual ASCII writer, no Open3D dependency)
- 📥 Download & view on https://3dviewer.net/ or CloudCompare

## Quick Start

### 1. Setup environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Run Streamlit app
```bash
streamlit run app.py
# or
.venv/bin/streamlit run app.py
```

App will be at http://localhost:8501

### 3. Use
1. **Upload Video** tab: Upload mp4 (10-30 sec recommended, textured scene)
2. **Process** tab: Keep defaults (sample_rate=1, max_frames=80) → START PROCESSING
3. **Results** tab: Download .ply, view on 3dviewer.net

No drone footage? Use "Generate Synthetic Test Video" button in Upload tab, or run:
```bash
python synthetic_video.py
```

## Pipeline Details

**Stage 1: Frame Extraction**
- OpenCV VideoCapture, sample_rate and max_frames control

**Stage 2: Feature Extraction**
- SIFT with max_num_features=8192 (configurable)
- CPU-only (use_gpu=False) for compatibility

**Stage 3: Matching**
- Exhaustive matching with ratio test 0.8, cross-check

**Stage 4: Incremental Mapping**
- Relaxed thresholds tuned for single-pass:
  - init_min_tri_angle=4° (vs default 16°)
  - init_min_num_inliers=30 (vs 100)
  - abs_pose_min_num_inliers=15
  - abs_pose_min_inlier_ratio=0.25
  - min_num_matches=10
  - min_model_size=3 (allow small demo models)

**Stage 5: PLY Export**
- Extract colors from images
- Filter NaN/Inf and outliers (3 sigma)
- Manual ASCII PLY writer

## Testing without UI

```bash
python test_pipeline.py --video /path/to/drone.mp4 --max_frames 50 --sample_rate 1
```

## Troubleshooting

- **No initial pair / 0 points**: Increase overlap (sample_rate=1, max_frames 80+), use textured video
- **pycolmap missing**: `pip install pycolmap==4.2.0` - requires Python 3.8-3.11, has wheels for Linux/Windows
- **Qt errors**: You are on old COLMAP CLI code - use this pycolmap version (app.py)
- **CUDA errors**: This app is CPU-only, should not need CUDA

## Project Structure

```
argus/
├── app.py              # Main Streamlit app (pycolmap + OpenCV fallback)
├── requirements.txt
├── README.md
├── test_pipeline.py    # Headless test
├── synthetic_video.py  # Generate test video
```

## Team ATLAS - SIH 2026
- Deadline: Sept 10, 2026 demo
- Target: Local Windows laptop, CPU-only, Streamlit GUI

## Future Work
- GPS EXIF georeferencing
- Poisson meshing for textured mesh
- Semantic filtering of dynamic objects
- Real-time edge deployment

## License
MIT - For SIH 2026 submission
