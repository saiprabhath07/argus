# ARGUS - Demo Guide for SIH 2026 (Sept 10)

## Quick Demo (5 minutes)

### Preparation (before demo)
1. Have a drone video ready (10-30 sec, textured scene)
   - If no real drone footage: use synthetic generator button in app
   - Backup: keep `sample_model.ply` ready to show on 3dviewer.net

2. Test run once:
   ```bash
   .venv/bin/python test_pipeline.py --video /path/to/video.mp4 --max_frames 50
   ```
   Should produce .ply file

3. Open https://3dviewer.net/ in a browser tab beforehand

### Live Demo Flow

**1. Introduction (30 sec)**
- "ARGUS solves single-pass drone video to 3D reconstruction"
- "Traditional photogrammetry needs multiple overlapping flights - we need only one"
- "Use cases: disaster response, one-shot surveillance, infrastructure inspection"

**2. Show Upload Tab (30 sec)**
- Upload video
- Show metadata: duration, FPS, resolution, frame count
- Explain settings: sample_rate=1 for max overlap, max_frames=80

**3. Process Tab (2 min - while processing, explain pipeline)**
- Click START PROCESSING
- While it runs, explain:
  - Frame extraction (OpenCV)
  - SIFT feature detection (8192 features)
  - Exhaustive matching + geometric verification
  - Incremental SfM with relaxed thresholds (tri angle 4° vs 16° default) tuned for single-pass
  - pycolmap = Python binding for COLMAP, no external binary, no Qt/CUDA issues, CPU-only
  - PLY export with color extraction + outlier filtering

**4. Results Tab (1 min)**
- Show stats: registered images, point count
- Download .ply
- Drag & drop to https://3dviewer.net/ - show 3D model rotating
- If time: show point cloud bounding box, 2D scatter

**5. Q&A / Technical Depth (1 min)**
- Why pycolmap over COLMAP CLI? CLI had Qt plugin missing on Windows, CUDA issues - pycolmap solves it
- Fallback OpenCV SIFT pipeline implemented
- Manual PLY writer avoids Open3D install issues
- Future: GPS georeferencing, Poisson meshing, semantic filtering

### Backup Plan

If live processing is slow (1-3 min) or fails:

**Option A: Pre-recorded**
- Have a screen recording of successful run

**Option B: Pre-generated PLY**
- Upload `sample_model.ply` to 3dviewer.net directly
- Explain: "This is from same pipeline, just pre-computed for time"

**Option C: Synthetic Video**
- Use "Generate Synthetic Test Video" button - always works, produces 704 points in test

### Troubleshooting Live

- **"No good initial pair"**: Say "This video has too little overlap, let me adjust sample_rate to 1 and max_frames to 80" - retry
- **Processing slow**: "SfM is computationally intensive, CPU-only for compatibility, typically 1-2 min for 60 frames"
- **0 points**: "Scene may lack texture - need buildings/roads, not just sky"

### What Judges Will Ask

**Q: Accuracy?**
A: "We use COLMAP gold-standard, bundle adjustment minimizes reprojection error, typically <1px mean error. For demo we prioritize working prototype over cm-level survey accuracy"

**Q: Why not NeRF / deep learning?**
A: "We explicitly chose tool-assembly over research approach per time constraints. pycolmap is proven, no training data needed, works on CPU. NeRF would need GPU and per-scene training"

**Q: Georeferencing?**
A: "Current version is relative coordinates. Next step is GPS EXIF integration - COLMAP supports prior positions, we have structure_less_registration fallback already"

**Q: Real-time?**
A: "Current is near-real-time (1-3 min for 60 frames). For edge deployment, we can reduce frames, features, or use sequential matching"

**Q: What about moving objects?**
A: "Future work: semantic segmentation to mask cars/people. Current robust RANSAC handles some outliers"

### Files to Have Ready

- Laptop with app running: `streamlit run app.py`
- Browser tabs: app (localhost:8501) + 3dviewer.net
- Sample video: 10-30 sec drone footage OR synthetic
- Sample PLY: `sample_model.ply` (704 points from test)
- Slides: 5-6 slides max - problem, solution, pipeline, results, future work

### Post-Demo

- GitHub repo: this repo
- Requirements: `pip install -r requirements.txt`
- Works on Windows/Linux, Python 3.11, CPU-only
- No external binary dependencies
