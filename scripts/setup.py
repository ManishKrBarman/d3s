"""
D3S Setup Script — Clone MASt3R and verify installation.

Usage:
    python scripts/setup.py          # Auto-detect GPU/CPU
    python scripts/setup.py --gpu    # Force GPU setup
    python scripts/setup.py --cpu    # Force CPU setup
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
MAST3R_DIR = THIRD_PARTY / "mast3r"


def run(cmd, cwd=None, check=True):
    """Run a command and print output."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or PROJECT_ROOT,
        capture_output=False,
    )
    if check and result.returncode != 0:
        print(f"  ✗ Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def clone_mast3r():
    """Clone MASt3R repository if not present."""
    if (MAST3R_DIR / "mast3r").is_dir():
        print("✓ MASt3R already cloned")
        return

    print("Cloning MASt3R...")
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    run(
        "git clone --recursive "
        "https://github.com/naver/mast3r "
        f'"{MAST3R_DIR}"'
    )
    print("✓ MASt3R cloned")


def check_torch():
    """Check PyTorch installation and GPU status."""
    try:
        import torch

        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem
            vram_gb = vram / (1024 ** 3)
            print(f"✓ GPU: {gpu} ({vram_gb:.1f} GB VRAM)")
            return "cuda"
        else:
            print("  No CUDA GPU detected — will use CPU")
            return "cpu"
    except ImportError:
        print("✗ PyTorch not installed!")
        return None


def install_torch(mode):
    """Install PyTorch for GPU or CPU."""
    if mode == "gpu":
        print("Installing PyTorch with CUDA 12.4...")
        run(
            f'"{sys.executable}" -m pip install '
            "torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
    else:
        print("Installing PyTorch CPU...")
        run(
            f'"{sys.executable}" -m pip install '
            "torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )


def install_requirements():
    """Install project requirements."""
    print("Installing requirements...")
    run(
        f'"{sys.executable}" -m pip install '
        f'-r "{PROJECT_ROOT / "requirements.txt"}"'
    )
    print("✓ Requirements installed")


def install_editable():
    """Install d3s package in editable mode."""
    print("Installing d3s package...")
    run(f'"{sys.executable}" -m pip install -e "{PROJECT_ROOT}"')
    print("✓ d3s package installed")


def verify():
    """Verify everything works."""
    print("\n--- Verification ---")

    # Check imports
    checks = [
        ("d3s.config", "Config system"),
        ("d3s.data.dataset", "Dataset loader"),
        ("d3s.preprocessing.quality_filter", "Quality filter"),
        ("d3s.reconstruction.mast3r_pipeline", "MASt3R pipeline"),
    ]

    all_ok = True
    for module, name in checks:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False

    # Check MASt3R
    try:
        mast3r_path = str(MAST3R_DIR)
        dust3r_path = str(MAST3R_DIR / "dust3r")
        if mast3r_path not in sys.path:
            sys.path.insert(0, mast3r_path)
        if dust3r_path not in sys.path:
            sys.path.insert(0, dust3r_path)

        from mast3r.model import AsymmetricMASt3R  # noqa

        print("  ✓ MASt3R model import")
    except ImportError as e:
        print(f"  ✗ MASt3R import: {e}")
        all_ok = False

    # Check device
    import torch

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem
        print(f"  ✓ GPU ready: {gpu} ({vram / 1024**3:.1f} GB)")
    else:
        print("  ⚠ CPU mode (no CUDA GPU)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Set up D3S pipeline"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU (CUDA) PyTorch installation",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only PyTorch installation",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("D3S SETUP")
    print("=" * 50)
    print()

    # Step 1: Check/install PyTorch
    device = check_torch()
    if device is None:
        if args.gpu:
            install_torch("gpu")
        elif args.cpu:
            install_torch("cpu")
        else:
            # Auto-detect: try GPU first
            print(
                "\nPyTorch not found. "
                "Use --gpu or --cpu flag to install.\n"
                "  python scripts/setup.py --gpu   "
                "(NVIDIA GPU with CUDA)\n"
                "  python scripts/setup.py --cpu   "
                "(CPU only)\n"
            )
            sys.exit(1)

    # Step 2: Install requirements
    install_requirements()

    # Step 3: Clone MASt3R
    clone_mast3r()

    # Step 4: Install d3s
    install_editable()

    # Step 5: Verify
    print()
    ok = verify()

    print()
    print("=" * 50)
    if ok:
        print("✓ SETUP COMPLETE — Ready to run!")
        print()
        print("Quick test:")
        print(
            "  python scripts/run_pipeline.py "
            "--input <your_images_folder> "
            "--output output/test"
        )
    else:
        print("✗ Some checks failed. See errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
