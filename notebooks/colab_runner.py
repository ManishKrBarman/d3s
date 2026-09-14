"""
D3S Pipeline — Google Colab Runner
===================================
Run this in Google Colab with GPU runtime:
  Runtime → Change runtime type → T4 GPU

Steps:
  1. Upload this file to Colab (or copy-paste cells)
  2. Set GPU runtime
  3. Run all cells
  4. Upload your images when prompted
  5. Download the 3D results
"""

# %% [markdown]
# # D3S — Drone 3D Reconstruction
# **Runtime → Change runtime type → T4 GPU** before running!

# %% Cell 1: Setup
# ==========================================
# Clone repo & install everything (~3 min)
# ==========================================

import subprocess, sys, os

# Clone the repo (update with your GitHub URL)
REPO_URL = "https://github.com/<your-username>/d3s.git"

if not os.path.exists("/content/d3s"):
    print("📦 Cloning D3S...")
    subprocess.run(
        f"git clone {REPO_URL} /content/d3s",
        shell=True, check=True
    )
else:
    print("✓ D3S already cloned")

os.chdir("/content/d3s")

# Install dependencies (PyTorch CUDA is pre-installed on Colab)
print("📦 Installing dependencies...")
subprocess.run(
    f"{sys.executable} -m pip install -q -r requirements.txt",
    shell=True, check=True
)

# Clone MASt3R
if not os.path.exists("third_party/mast3r/mast3r"):
    print("📦 Cloning MASt3R...")
    os.makedirs("third_party", exist_ok=True)
    subprocess.run(
        "git clone --recursive https://github.com/naver/mast3r third_party/mast3r",
        shell=True, check=True
    )
else:
    print("✓ MASt3R already cloned")

# Install editable
subprocess.run(
    f"{sys.executable} -m pip install -q -e .",
    shell=True, check=True
)

# Verify GPU
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"✅ GPU: {gpu} ({vram:.1f} GB VRAM)")
else:
    print("⚠️ No GPU! Go to Runtime → Change runtime type → T4 GPU")

print("✅ Setup complete!")


# %% Cell 2: Upload Images
# ==========================================
# Upload your drone images
# ==========================================

# pyrefly: ignore [missing-import]
from google.colab import files
import shutil
from pathlib import Path

INPUT_DIR = Path("/content/d3s/input_images")
INPUT_DIR.mkdir(exist_ok=True)

print("📸 Upload your drone images (JPG/PNG)...")
print("   (Select multiple files at once)")
uploaded = files.upload()

for fname, data in uploaded.items():
    dest = INPUT_DIR / fname
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  ✓ {fname} ({len(data) / 1024:.0f} KB)")

print(f"\n✅ {len(uploaded)} images uploaded to {INPUT_DIR}")


# %% Cell 3: Run Pipeline
# ==========================================
# Run the 3D reconstruction
# ==========================================

OUTPUT_DIR = "/content/d3s/output/colab_run"

print("🚀 Running D3S pipeline...")
print(f"   Input:  {INPUT_DIR} ({len(list(INPUT_DIR.glob('*')))} images)")
print(f"   Output: {OUTPUT_DIR}")
print()

result = subprocess.run(
    f"{sys.executable} scripts/run_pipeline.py "
    f"--input {INPUT_DIR} "
    f"--output {OUTPUT_DIR} "
    f"-v",
    shell=True,
    cwd="/content/d3s",
)

if result.returncode == 0:
    print("\n✅ Pipeline complete!")
else:
    print("\n⚠️ Pipeline finished with warnings (check output above)")


# %% Cell 4: Download Results
# ==========================================
# Download the 3D reconstruction
# ==========================================

import zipfile

output_path = Path(OUTPUT_DIR)
zip_path = "/content/d3s_results.zip"

print("📦 Packaging results...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in output_path.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(output_path))
            print(f"  + {f.name} ({f.stat().st_size / 1024:.0f} KB)")

print(f"\n📥 Downloading {zip_path}...")
files.download(zip_path)
print("✅ Done! Open .ply files in MeshLab or CloudCompare.")


# %% Cell 5 (Optional): Run on AGZ Dataset
# ==========================================
# Download and run on the AGZ benchmark dataset
# ==========================================

# Uncomment below if you have the AGZ dataset URL:
#
# !wget -q "YOUR_DRIVE_LINK" -O agz.zip
# !unzip -q agz.zip -d AGZ_subset
# !python scripts/run_pipeline.py --input AGZ_subset --output output/agz_run -v
