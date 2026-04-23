#!/usr/bin/env python3
"""MCP server for local image search."""

import fcntl
import gc
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import daft
import numpy as np
from mcp.server.fastmcp import FastMCP

from core import load_model, embed_text, cosine_similarity, DB_PATH, DEFAULT_EXCLUDE_DIRS
from embed import sync_embeddings

# File-based lock to prevent concurrent refreshes across processes
LOCK_FILE = Path(DB_PATH).parent / ".embedding_refresh.lock"

# How long (seconds) to keep model in RAM after last use before unloading
MODEL_IDLE_TIMEOUT = int(os.environ.get("MODEL_IDLE_TIMEOUT", "300"))  # default 5 minutes

# Delay before first embedding refresh (seconds) — avoids CPU spike at Claude startup
REFRESH_STARTUP_DELAY = int(os.environ.get("REFRESH_STARTUP_DELAY", "120"))  # default 2 minutes


def log(msg: str):
    """Log to stderr (stdout is reserved for MCP protocol)."""
    print(msg, file=sys.stderr, flush=True)


# Create MCP server
mcp = FastMCP("local-image-search")

# Global state
model = None
processor = None
device = None
embeddings_df = None
image_dir = None
exclude_dirs = None
model_lock = threading.Lock()       # Protects model load/unload
model_last_used = 0.0               # Timestamp of last search_images call

# Embedding refresh state
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL", "60"))  # default 1 minute


def _load_model_if_needed():
    """Load model lazily. Must be called with model_lock held."""
    global model, processor, device
    if model is not None:
        return
    log("Loading SigLIP model (downloads on first run, ~800MB)...")
    model, processor, device = load_model()
    log(f"Model loaded on {device}")


def _unload_model():
    """Unload model and free RAM. Must be called with model_lock held."""
    global model, processor, device
    if model is None:
        return
    del model, processor
    model = processor = device = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    log("Model unloaded (idle timeout reached)")


def model_idle_watcher():
    """Background thread: unload model after MODEL_IDLE_TIMEOUT seconds of inactivity."""
    while True:
        time.sleep(30)  # check every 30s
        with model_lock:
            if model is not None and model_last_used > 0:
                idle = time.monotonic() - model_last_used
                if idle >= MODEL_IDLE_TIMEOUT:
                    _unload_model()


def get_status_info() -> dict:
    """Get current service status."""
    if embeddings_df is None or len(embeddings_df) == 0:
        return {
            "ready": False,
            "status": "no_embeddings",
            "message": "No embeddings found. Run: uv run python embed.py --drive-folder <url>"
        }
    model_status = "loaded" if model is not None else "unloaded (will load on first search)"
    return {
        "ready": True,
        "status": "ready",
        "total_images": len(embeddings_df),
        "model": model_status,
    }


@mcp.tool()
def get_status() -> dict:
    """Check if the image search service is ready.

    Returns:
        Status dict with 'ready' boolean and 'message' or 'total_images'
    """
    return get_status_info()


@mcp.tool()
def search_images(
    query: str,
    limit: int = 5,
    sort_by: str = "relevance",
    quality_weight: float = 0.5,
    min_relevance: float = 0.0,
) -> list[dict]:
    """Search for images matching a text query.

    Args:
        query: Natural language description of the image to find
        limit: Maximum number of results to return (default: 5)
        sort_by: How to rank results — "relevance" (default), "quality", or "combined"
            - "relevance": rank by SigLIP cosine similarity to query (standard semantic search)
            - "quality": filter by min_relevance, then rank by aesthetic score
            - "combined": weighted blend of relevance and aesthetic score
        quality_weight: Weight for aesthetic score when sort_by="combined" (0.0–1.0, default 0.5)
        min_relevance: Minimum relevance score to include a result (default 0.0).
            Useful with sort_by="quality" to filter out off-topic images before ranking by quality.
            Recommended: 0.05–0.10 for quality/combined modes.

    Returns:
        List of matching images with paths, similarity scores, and aesthetic scores
    """
    global model, processor, device, embeddings_df, model_last_used

    # Check if embeddings are available
    status = get_status_info()
    if not status["ready"]:
        return [status]

    # Lazy-load model on first use (or after idle unload)
    with model_lock:
        _load_model_if_needed()
        model_last_used = time.monotonic()
        # Embed inside the lock so model isn't unloaded mid-search
        query_embedding = embed_text(query, model, processor, device)

    # Get all embeddings and paths
    data = embeddings_df.to_pydict()
    paths = data["path"]
    vectors = data["vector"]
    drive_urls = data.get("drive_url", [None] * len(paths))
    filenames = data.get("filename", [None] * len(paths))
    aesthetic_scores = data.get("aesthetic_score", [None] * len(paths))

    # Compute relevance similarities
    relevance_scores = []
    for vec in vectors:
        vec_array = np.array(vec, dtype=np.float32)
        if np.allclose(vec_array, 0):
            relevance_scores.append(-1.0)
        else:
            relevance_scores.append(cosine_similarity(query_embedding, vec_array))

    # Compute final ranking score based on sort_by
    def _aesthetic(v):
        """Normalize aesthetic score: None/NaN → 0.5 (neutral)."""
        try:
            f = float(v)
            return 0.5 if math.isnan(f) else f
        except (TypeError, ValueError):
            return 0.5

    if sort_by == "quality":
        final_scores = [_aesthetic(a) for a in aesthetic_scores]
    elif sort_by == "combined":
        qw = max(0.0, min(1.0, quality_weight))
        final_scores = [
            (1.0 - qw) * r + qw * _aesthetic(a)
            for r, a in zip(relevance_scores, aesthetic_scores)
        ]
    else:  # "relevance" (default)
        final_scores = relevance_scores

    # Sort by final score descending
    ranked = sorted(
        zip(paths, final_scores, relevance_scores, drive_urls, filenames, aesthetic_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Deduplicate by path (same Drive file_id may appear in multiple folders)
    # ranked is already sorted by score desc, so first occurrence = highest score
    seen_paths: set[str] = set()
    results = []
    for path, final_score, relevance, drive_url, filename, aesthetic in ranked:
        if len(results) >= limit:
            break
        if relevance <= 0:  # always exclude corrupt/failed images (zero vector)
            continue
        if relevance < min_relevance:  # apply caller-specified relevance floor
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        entry = {
            "path": path,
            "score": round(final_score, 3),
            "relevance": round(relevance, 3),
        }
        if aesthetic is not None:  # only include if image has been scored
            entry["aesthetic_score"] = round(_aesthetic(aesthetic), 3)
        if drive_url and str(drive_url) not in ("None", "nan", ""):
            entry["drive_url"] = drive_url
        name = filename if filename and str(filename) not in ("None", "nan", "") else Path(path).name
        entry["filename"] = name
        results.append(entry)

    return results


def reload_embeddings():
    """Reload embeddings from Lance DB."""
    global embeddings_df

    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        log(f"Reloaded {len(embeddings_df)} embeddings")
    else:
        embeddings_df = None
        log("No embeddings found")


def embedding_refresh_loop():
    """Background loop to refresh embeddings periodically."""
    global image_dir, exclude_dirs

    # Wait before first refresh so Claude startup isn't competing with model load
    log(f"Embedding refresh will start in {REFRESH_STARTUP_DELAY}s...")
    time.sleep(REFRESH_STARTUP_DELAY)

    while True:
        # Add random jitter (0-30 seconds) to prevent thundering herd
        jitter = random.uniform(0, 30)
        time.sleep(jitter)

        # Try to acquire file-based lock (non-blocking) to coordinate across processes
        lock_file = None
        try:
            LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(LOCK_FILE, "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            log("Another process is refreshing embeddings, skipping this cycle")
            if lock_file:
                lock_file.close()
            time.sleep(REFRESH_INTERVAL)
            continue

        try:
            if image_dir and image_dir.exists():
                log(f"Starting embedding refresh for {image_dir}...")
                sync_embeddings(image_dir, log_fn=log, exclude_dirs=exclude_dirs)
                reload_embeddings()
            else:
                log(f"Image directory not set or doesn't exist: {image_dir}")
        except Exception as e:
            log(f"Embedding refresh failed: {e}")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

        time.sleep(REFRESH_INTERVAL)


def startup_task():
    """Background task to load embeddings and start background threads.

    Model is NOT loaded here — it loads lazily on first search_images call.
    This keeps startup fast and avoids RAM spike when Claude launches.
    """
    global embeddings_df, image_dir

    # Load embeddings index (Lance DB read is fast, no model needed)
    log("Loading embeddings index...")
    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        log(f"Loaded {len(embeddings_df)} embeddings (model will load on first search)")
    else:
        log("No embeddings found. Run: uv run python embed.py --drive-folder <url>")

    # Start model idle watcher
    watcher_thread = threading.Thread(target=model_idle_watcher, daemon=True)
    watcher_thread.start()

    # Start background embedding refresh thread (with startup delay)
    if image_dir:
        refresh_thread = threading.Thread(target=embedding_refresh_loop, daemon=True)
        refresh_thread.start()
        log(f"Background embedding refresh scheduled (delay={REFRESH_STARTUP_DELAY}s, interval={REFRESH_INTERVAL}s)")


def main():
    """Main entry point."""
    global image_dir, exclude_dirs

    # Parse EXCLUDE_DIRS from environment (comma-separated)
    exclude_env = os.environ.get("EXCLUDE_DIRS", "").strip()
    custom_excludes = [d.strip() for d in exclude_env.split(",") if d.strip()] if exclude_env else None

    # Parse image directory from command line
    if len(sys.argv) > 1:
        # Custom root provided
        image_dir = Path(sys.argv[1]).expanduser().resolve()
        # Use custom excludes if provided, otherwise no excludes
        exclude_dirs = custom_excludes
        log(f"Image directory: {image_dir}")
        if exclude_dirs:
            log(f"Excluding: {', '.join(exclude_dirs)}")
    else:
        # No root provided - use home with defaults (unless custom excludes provided)
        image_dir = Path.home()
        if custom_excludes:
            # Custom excludes override defaults
            exclude_dirs = custom_excludes
            log(f"Image directory: {image_dir} (default)")
            log(f"Excluding: {', '.join(exclude_dirs)}")
        else:
            # Use default excludes
            exclude_dirs = DEFAULT_EXCLUDE_DIRS
            log(f"Image directory: {image_dir} (default)")
            log(f"Excluding (defaults): {', '.join(exclude_dirs)}")

    # Start model loading in background
    startup_thread = threading.Thread(target=startup_task, daemon=True)
    startup_thread.start()

    # Run the MCP server (starts immediately, responds with status while loading)
    mcp.run()


if __name__ == "__main__":
    main()
