# D3S — Drone Single-Shot 3D Reconstruction

**Single-pass drone video → accurate 3D model generation** using [MASt3R-SfM](https://github.com/naver/mast3r).

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/d3s.git
cd d3s

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Run setup (picks GPU or CPU automatically)
python scripts/setup.py --gpu    # If you have NVIDIA GPU (RTX 4060, 3080, etc.)
python scripts/setup.py --cpu    # If CPU only

# 4. Run the pipeline
python scripts/run_pipeline.py --input path/to/images --output output/my_run
```

## Hardware Requirements

| Setup | GPU | RAM | Speed (83 images) |
|-------|-----|-----|-------------------|
| **Recommended** | RTX 4060+ (8GB VRAM) | 16GB | ~15-30 min |
| Minimum | CPU only | 8GB | ~4-8 hours |

## Input Formats

### Image Folder
```bash
python scripts/run_pipeline.py --input path/to/drone_photos/ --output output/result
```

### Video File
```bash
python scripts/run_pipeline.py --input path/to/drone_video.mp4 --output output/result
```

## CLI Options

```bash
python scripts/run_pipeline.py [OPTIONS]

Required:
  --input PATH              Image folder or video file
  --output PATH             Where to save results

Optional:
  --config PATH             Config file (default: config/default.yaml)
  --skip-reconstruction     Run preprocessing only (fast, ~1 min)
  -v, --verbose             Debug logging
```

## Output

```
output/my_run/
├── scene.npz              # Full reconstruction data
├── camera_trajectory.csv  # Camera positions per frame
├── point_cloud.ply        # 3D point cloud (open in MeshLab)
└── metrics.json           # Quality metrics
```

View `.ply` files in [MeshLab](https://www.meshlab.net/) or [CloudCompare](https://www.cloudcompare.org/).

## Project Structure

```
d3s/
├── d3s/                    # Core Python package
│   ├── config.py           # Configuration (auto GPU/CPU detection)
│   ├── data/               # Dataset loading, video ingest, GPS
│   ├── preprocessing/      # Quality filter, keyframe selection
│   ├── reconstruction/     # MASt3R-SfM pipeline, georef, mesh
│   ├── evaluation/         # Metrics, ground truth comparison
│   └── export/             # PLY/CSV export, geotiff
├── config/default.yaml     # Pipeline configuration
├── scripts/
│   ├── run_pipeline.py     # Main CLI entry point
│   └── setup.py            # One-click environment setup
├── requirements.txt
└── pyproject.toml
```

## Configuration

Edit `config/default.yaml` to tune parameters:

```yaml
reconstruction:
  device: auto              # "auto", "cuda", or "cpu"
  image_size: 512           # Input resolution for MASt3R
  scene_graph: swin-5       # "swin-5" (fast) or "complete" (better, slower)
  alignment_iterations: 300
  max_points: 1000000
```

## GPU Auto-Detection

The pipeline automatically:
1. Detects if CUDA is available
2. Checks VRAM (needs ≥3GB for MASt3R ViT-Large)
3. Falls back to CPU if insufficient VRAM

Force a specific device:
```bash
# Environment variable
D3S_DEVICE=cuda python scripts/run_pipeline.py ...

# Or in config/default.yaml
reconstruction:
  device: cuda
```

## Team Setup

1. **Clone the repo** on your machine
2. **Run setup**: `python scripts/setup.py --gpu` (RTX 4060)
3. **First run** downloads the MASt3R model (~2.6 GB, cached after)
4. **Subsequent runs** skip the download

## License

Research use. MASt3R is licensed under the [MASt3R license](https://github.com/naver/mast3r/blob/main/LICENSE).
