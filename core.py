"""Shared utilities for local image search."""

import os
import subprocess
import warnings
from pathlib import Path

import daft
import numpy as np
import pillow_heif
import torch
from daft import DataType, Series
from PIL import Image

pillow_heif.register_heif_opener()  # Enable HEIC/HEIF support in PIL

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from huggingface_hub import logging as hf_hub_logging  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
from transformers import logging as hf_logging  # noqa: E402

hf_hub_logging.set_verbosity_error()
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# Path relative to this file
_CORE_DIR = Path(__file__).parent.resolve()
DB_PATH = str(_CORE_DIR / "embeddings.lance")

# HuggingFace model — SigLIP gives better fine-grained search than CLIP base.
# Override with IMAGE_SEARCH_MODEL env var if needed.
MODEL_NAME = os.environ.get("IMAGE_SEARCH_MODEL", "google/siglip-so400m-patch14-384")
EMBED_DIM = 1152  # SigLIP so400m output dimension

# Image extensions to search for
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg",  # JPEG
    ".png",           # PNG
    ".gif",           # GIF
    ".webp",          # WebP
    ".bmp",           # BMP
    ".tiff", ".tif",  # TIFF
    ".heic", ".heif", # iPhone photos
}

# Rough estimate for --dry-run time estimates (CPU baseline)
IMAGES_PER_SECOND = 10

# Default directories to exclude when scanning home directory
DEFAULT_EXCLUDE_DIRS = [
    "Library",
    ".Trash",
    ".cache",
    "Cache",
    "node_modules",
    ".git",
    ".venv",
    "venv",
]


@daft.cls
class EmbedImages:
    """Daft UDF to generate SigLIP embeddings for images."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), EMBED_DIM))
    def __call__(self, paths: Series):
        """Takes a Series of image paths, returns a list of EMBED_DIM-dim embeddings."""
        path_list = paths.to_pylist()
        images = []
        failed = []
        for p in path_list:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
                failed.append(p)
                images.append(Image.new("RGB", (224, 224)))  # placeholder

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            features = out if isinstance(out, torch.Tensor) else out.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        embeddings = features.cpu().numpy()

        # Zero out embeddings for failed images
        for i, p in enumerate(path_list):
            if p in failed:
                embeddings[i] = np.zeros(EMBED_DIM, dtype=np.float32)

        return [embeddings[i] for i in range(len(path_list))]


def load_model():
    """Load the SigLIP model and processor. Returns (model, processor, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_text(query: str, model, processor, device) -> np.ndarray:
    """Embed a text query using SigLIP text encoder."""
    inputs = processor(text=[query], return_tensors="pt", padding="max_length").to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        features = out if isinstance(out, torch.Tensor) else out.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


def embed_images_batch(images: list, model, processor, device) -> list[np.ndarray]:
    """Embed a batch of PIL images using SigLIP.

    Args:
        images: List of PIL Image objects
        model: Loaded SigLIP model
        processor: SigLIP processor
        device: torch device string

    Returns:
        List of EMBED_DIM-dim numpy embeddings
    """
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        features = out if isinstance(out, torch.Tensor) else out.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    return [features[i].cpu().numpy() for i in range(len(images))]


def find_images(directory: Path, recursive: bool = True, show_progress: bool = True, exclude_dirs: list[str] | None = None) -> list[Path]:
    """Find all image files in a directory using find command.

    Args:
        directory: Root directory to search
        recursive: Whether to search subdirectories
        show_progress: Whether to print progress
        exclude_dirs: List of directory names to exclude (e.g. ["Library", ".cache"])
    """
    # Build -name conditions for each extension
    name_args = []
    for ext in IMAGE_EXTENSIONS:
        if name_args:
            name_args.append("-o")
        name_args.extend(["-name", f"*{ext}"])

    # Build find command
    cmd = ["find", str(directory)]
    if not recursive:
        cmd.extend(["-maxdepth", "1"])

    # Build prune conditions for excluded directories
    prune_args = ["-name", ".*"]  # Always exclude hidden directories
    if exclude_dirs:
        for exclude in exclude_dirs:
            prune_args.extend(["-o", "-name", exclude])

    cmd.extend(["("] + prune_args + [")", "-prune", "-o"])
    cmd.extend(["-type", "f", "("] + name_args + [")", "-print"])

    # Stream output and show progress
    paths = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in process.stdout:
        path = line.strip()
        if path:
            paths.append(Path(path))
            if show_progress and len(paths) % 1000 == 0:
                print(f"\rFound: {len(paths):,} images...", end="", flush=True)
    process.wait()

    if show_progress and len(paths) >= 1000:
        print()  # newline after progress

    return sorted(paths)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
