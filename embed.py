#!/usr/bin/env python3
"""CLI tool to sync image embeddings from a directory or Google Drive folder."""

import argparse
import sys
import threading
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import daft
import numpy as np
from daft import col, DataType
from PIL import Image

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # allow large images from Drive

from core import EMBED_DIM, EmbedImages, IMAGE_EXTENSIONS, embed_images_batch, find_images, format_time, IMAGES_PER_SECOND, DB_PATH, load_model

# Type for vector column
VECTOR_DTYPE = DataType.embedding(DataType.float32(), EMBED_DIM)

# Batch size for Drive image embedding
DRIVE_BATCH_SIZE = 16

# Parallel download workers for Drive
DRIVE_DOWNLOAD_WORKERS = 4

# SigLIP input resolution — resize on download to save memory
SIGLIP_INPUT_SIZE = 384

# Thread-local storage for Drive service instances (googleapiclient is not thread-safe)
_thread_local = threading.local()


def get_current_files(directory: Path, recursive: bool = True, show_progress: bool = True, exclude_dirs: list[str] | None = None) -> dict[str, float]:
    """Scan directory and return {path: mtime} for all images."""
    images = find_images(directory, recursive=recursive, show_progress=show_progress, exclude_dirs=exclude_dirs)
    return {str(p): p.stat().st_mtime for p in images}


def get_stored_files() -> dict[str, float]:
    """Read Lance DB and return {path: mtime} for stored embeddings."""
    if not Path(DB_PATH).exists():
        return {}

    df = daft.read_lance(DB_PATH)
    results = df.select("path", "mtime").collect()
    data = results.to_pydict()
    return dict(zip(data["path"], data["mtime"]))


def _ensure_drive_url_column(df) -> "daft.DataFrame":
    """Ensure drive_url column exists and has type Utf8 (backward compatibility)."""
    try:
        df.schema()["drive_url"]
    except KeyError:
        df = df.with_column("drive_url", daft.lit(None).cast(DataType.string()))
    # Always cast to string so types match across local and Drive DataFrames
    return df.with_column("drive_url", col("drive_url").cast(DataType.string()))


def _ensure_filename_column(df) -> "daft.DataFrame":
    """Ensure filename column exists (backward compatibility)."""
    try:
        df.schema()["filename"]
    except KeyError:
        df = df.with_column("filename", daft.lit(None).cast(DataType.string()))
    return df.with_column("filename", col("filename").cast(DataType.string()))


def sync_embeddings(directory: Path, recursive: bool = True, log_fn=print, exclude_dirs: list[str] | None = None) -> dict:
    """Sync embeddings for images in a directory.

    Args:
        directory: Directory to scan for images
        recursive: Whether to search subdirectories
        log_fn: Function to use for logging (default: print)
        exclude_dirs: List of directory names to exclude (e.g. ["Library", ".cache"])

    Returns:
        Dict with stats: {new, modified, deleted, unchanged, total, elapsed}
    """
    # Scan current files
    log_fn(f"Scanning: {directory}")
    current = get_current_files(directory, recursive=recursive, show_progress=False, exclude_dirs=exclude_dirs)
    log_fn(f"Found: {len(current):,} images")

    # Load stored embeddings (local paths only — Drive entries use drive:// prefix)
    stored_all = get_stored_files()
    stored = {k: v for k, v in stored_all.items() if not k.startswith("drive://")}
    if stored:
        log_fn(f"Stored: {len(stored):,} embeddings")

    # Compute differences
    current_paths = set(current.keys())
    stored_paths = set(stored.keys())

    new_paths = current_paths - stored_paths
    deleted_paths = stored_paths - current_paths
    common_paths = current_paths & stored_paths

    # Check for modified files (mtime changed)
    modified_paths = {p for p in common_paths if current[p] != stored[p]}
    unchanged_paths = common_paths - modified_paths

    # Paths that need embedding
    to_embed = new_paths | modified_paths

    # Log summary
    log_fn(f"Unchanged: {len(unchanged_paths):,}, New: {len(new_paths):,}, Modified: {len(modified_paths):,}, Removed: {len(deleted_paths):,}")

    if not to_embed and not deleted_paths:
        log_fn("Nothing to do.")
        return {
            "new": 0, "modified": 0, "deleted": 0,
            "unchanged": len(unchanged_paths), "total": len(current), "elapsed": 0
        }

    start = time.perf_counter()

    if to_embed:
        log_fn(f"Embedding {len(to_embed):,} images...")

        # Prepare data for new embeddings (drive_url is null for local files)
        paths_to_embed = sorted(to_embed)
        mtimes_to_embed = [current[p] for p in paths_to_embed]

        # Create DataFrame and embed
        df_new = daft.from_pydict({
            "path": paths_to_embed,
            "mtime": mtimes_to_embed,
            "drive_url": [None] * len(paths_to_embed),
            "filename": [Path(p).name for p in paths_to_embed],
        })
        # Ensure drive_url is Utf8, not Null, so schemas match
        df_new = df_new.with_column("drive_url", col("drive_url").cast(DataType.string()))
        embed_images = EmbedImages()
        df_new = df_new.with_column("vector", embed_images(col("path")))

        # If we have unchanged embeddings, combine them
        if unchanged_paths:
            # Read existing and filter to unchanged only
            df_existing = daft.read_lance(DB_PATH)
            unchanged_list = list(unchanged_paths)
            df_unchanged = df_existing.where(col("path").is_in(unchanged_list))

            # Cast existing vectors to Embedding type (Lance returns as List)
            df_unchanged = df_unchanged.with_column(
                "vector", col("vector").cast(VECTOR_DTYPE)
            )

            # Ensure drive_url column exists for backward compatibility
            df_unchanged = _ensure_drive_url_column(df_unchanged)
            df_unchanged = _ensure_filename_column(df_unchanged)

            # Combine unchanged + new
            df_final = df_unchanged.concat(df_new)
        else:
            df_final = df_new

        # Preserve existing Drive entries (drive:// paths are not local files)
        drive_paths = [p for p in stored_all if p.startswith("drive://")]
        if drive_paths and Path(DB_PATH).exists():
            df_existing = daft.read_lance(DB_PATH)
            df_drive = df_existing.where(col("path").is_in(drive_paths))
            df_drive = df_drive.with_column("vector", col("vector").cast(VECTOR_DTYPE))
            df_drive = _ensure_drive_url_column(df_drive)
            df_drive = _ensure_filename_column(df_drive)
            df_final = df_final.concat(df_drive)
    else:
        # No new embeddings, just filter out deleted local files
        df_existing = daft.read_lance(DB_PATH)
        # Keep current local paths + all Drive entries
        keep_list = list(current_paths) + [p for p in stored_all if p.startswith("drive://")]
        df_final = df_existing.where(col("path").is_in(keep_list))
        df_final = df_final.with_column("vector", col("vector").cast(VECTOR_DTYPE))
        df_final = _ensure_drive_url_column(df_final)
        df_final = _ensure_filename_column(df_final)

    # Write to Lance
    mode = "create" if not Path(DB_PATH).exists() else "overwrite"
    df_final.write_lance(DB_PATH, mode=mode)

    elapsed = time.perf_counter() - start

    log_fn(f"Done in {format_time(elapsed)}")
    if to_embed:
        log_fn(f"Speed: {len(to_embed)/elapsed:.1f} images/second")
    log_fn(f"Total embeddings: {len(current):,}")

    return {
        "new": len(new_paths),
        "modified": len(modified_paths),
        "deleted": len(deleted_paths),
        "unchanged": len(unchanged_paths),
        "total": len(current),
        "elapsed": elapsed
    }


def sync_drive_embeddings(folder_url_or_id: str, log_fn=print) -> dict:
    """Index images directly from a Google Drive folder.

    Downloads images in parallel (DRIVE_DOWNLOAD_WORKERS threads), resizes
    to SigLIP input resolution in-memory, then embeds in batches.

    Args:
        folder_url_or_id: Google Drive folder URL or bare folder ID
        log_fn: Function to use for logging (default: print)

    Returns:
        Dict with stats: {new, skipped, failed, total, elapsed}
    """
    from googleapiclient.http import MediaIoBaseDownload
    from drive import extract_folder_id, list_folder_files_with_ids

    folder_id = extract_folder_id(folder_url_or_id)
    log_fn(f"Fetching file list from Drive folder {folder_id}...")

    all_files = list_folder_files_with_ids(folder_id)
    image_files = [
        f for f in all_files
        if Path(f["name"]).suffix.lower() in IMAGE_EXTENSIONS
    ]
    log_fn(f"Found {len(image_files)} image(s) on Drive")

    # Check which Drive entries are already indexed
    existing_drive_paths: set[str] = set()
    if Path(DB_PATH).exists():
        df = daft.read_lance(DB_PATH)
        data = df.select("path").collect().to_pydict()
        existing_drive_paths = {p for p in data["path"] if p.startswith("drive://")}
        if existing_drive_paths:
            log_fn(f"  {len(existing_drive_paths)} Drive image(s) already indexed — skipping")

    to_download = [f for f in image_files if f"drive://{f['id']}" not in existing_drive_paths]
    if not to_download:
        log_fn("Nothing new to index.")
        return {"new": 0, "skipped": len(existing_drive_paths), "failed": 0, "total": len(image_files), "elapsed": 0}

    log_fn(f"Downloading {len(to_download)} image(s) with {DRIVE_DOWNLOAD_WORKERS} workers...")

    def _get_thread_service():
        """Return a per-thread Drive service (googleapiclient is not thread-safe)."""
        if not hasattr(_thread_local, "drive_service"):
            _thread_local.drive_service = get_service()
        return _thread_local.drive_service

    def _download(file_info: dict) -> tuple[dict, Image.Image | None, str | None]:
        """Download and resize one Drive image. Returns (file_info, image, error)."""
        file_id = file_info["id"]
        name = file_info["name"]
        try:
            svc = _get_thread_service()
            buf = BytesIO()
            downloader = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id))
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            # Resize to SigLIP input size to cut memory and speed up embedding
            img = img.resize((SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE), Image.LANCZOS)
            return file_info, img, None
        except Exception as e:
            return file_info, None, str(e)

    # Download in parallel, preserve order via index
    downloaded: list[tuple[dict, Image.Image | None, str | None]] = [None] * len(to_download)
    index_map = {f["id"]: i for i, f in enumerate(to_download)}
    completed_count = 0
    failed = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=DRIVE_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download, f): f for f in to_download}
        for future in as_completed(futures):
            file_info, img, err = future.result()
            idx = index_map[file_info["id"]]
            downloaded[idx] = (file_info, img, err)
            completed_count += 1
            if err:
                failed += 1
                log_fn(f"  WARNING: could not download {file_info['name']}: {err}")
            else:
                log_fn(f"  [{completed_count}/{len(to_download)}] {file_info['name']}")

    t_download = time.perf_counter() - t0
    log_fn(f"Download done in {format_time(t_download)} ({len(to_download) - failed} ok, {failed} failed)")

    # Load SigLIP model and embed in batches
    log_fn("Loading SigLIP model...")
    model, processor, device = load_model()

    new_rows: list[dict] = []
    batch_images: list[Image.Image] = []
    batch_meta: list[dict] = []

    def flush_batch():
        if not batch_images:
            return
        try:
            embeddings = embed_images_batch(batch_images, model, processor, device)
        except Exception as e:
            log_fn(f"  WARNING: batch embed failed: {e}")
            return
        for meta, emb in zip(batch_meta, embeddings):
            new_rows.append({**meta, "vector": emb.tolist()})
        batch_images.clear()
        batch_meta.clear()

    log_fn(f"Embedding {len(to_download) - failed} image(s)...")
    for file_info, img, err in downloaded:
        if err or img is None:
            continue
        file_id = file_info["id"]
        name_nfc = unicodedata.normalize("NFC", file_info["name"])
        drive_url = file_info.get("webViewLink", f"https://drive.google.com/file/d/{file_id}/view")
        batch_images.append(img)
        batch_meta.append({
            "path": f"drive://{file_id}",
            "filename": name_nfc,
            "mtime": 0.0,
            "drive_url": drive_url,
        })
        if len(batch_images) >= DRIVE_BATCH_SIZE:
            flush_batch()

    flush_batch()

    if not new_rows:
        log_fn("Nothing new to index.")
        return {"new": 0, "skipped": len(existing_drive_paths), "failed": failed, "total": len(image_files), "elapsed": 0}

    # Build DataFrame for new Drive entries
    # Drive entries use drive_url and a synthetic mtime=0
    df_drive_new = daft.from_pydict({
        "path": [r["path"] for r in new_rows],
        "mtime": [r["mtime"] for r in new_rows],
        "drive_url": [r["drive_url"] for r in new_rows],
        "filename": [r["filename"] for r in new_rows],
        "vector": [r["vector"] for r in new_rows],
    })
    df_drive_new = df_drive_new.with_column("vector", col("vector").cast(VECTOR_DTYPE))
    df_drive_new = df_drive_new.with_column("drive_url", col("drive_url").cast(DataType.string()))

    # Merge with existing DB (if any)
    if Path(DB_PATH).exists():
        df_existing = daft.read_lance(DB_PATH)
        df_existing = df_existing.with_column("vector", col("vector").cast(VECTOR_DTYPE))
        df_existing = _ensure_drive_url_column(df_existing)
        df_existing = _ensure_filename_column(df_existing)
        df_final = df_existing.concat(df_drive_new)
        mode = "overwrite"
    else:
        df_final = df_drive_new
        mode = "create"

    df_final.write_lance(DB_PATH, mode=mode)

    elapsed = time.perf_counter() - t0
    log_fn(f"\nDone. Indexed {len(new_rows)} Drive image(s) in {format_time(elapsed)}")

    return {
        "new": len(new_rows),
        "skipped": len(existing_drive_paths),
        "failed": failed,
        "total": len(image_files),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sync image embeddings from a local directory and/or Google Drive"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        help="Local directory to search for images",
    )
    parser.add_argument(
        "--drive-folder",
        metavar="FOLDER_ID_OR_URL",
        help="Google Drive folder ID or URL to index (requires credentials.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually embedding",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories (local only)",
    )

    args = parser.parse_args()

    if not args.directory and not args.drive_folder:
        parser.error("Provide a local directory and/or --drive-folder")

    if args.directory:
        directory = Path(args.directory).resolve()

        if not directory.exists():
            print(f"Error: Directory '{directory}' does not exist")
            sys.exit(1)

        if not directory.is_dir():
            print(f"Error: '{directory}' is not a directory")
            sys.exit(1)

        if args.dry_run:
            current = get_current_files(directory, recursive=not args.no_recursive)
            stored_all = get_stored_files()
            stored = {k: v for k, v in stored_all.items() if not k.startswith("drive://")}

            current_paths = set(current.keys())
            stored_paths = set(stored.keys())
            new_paths = current_paths - stored_paths
            deleted_paths = stored_paths - current_paths
            common_paths = current_paths & stored_paths
            modified_paths = {p for p in common_paths if current[p] != stored[p]}
            unchanged_paths = common_paths - modified_paths
            to_embed = new_paths | modified_paths

            print(f"Found: {len(current):,} images")
            print(f"Stored: {len(stored):,} embeddings")
            print(f"\nUnchanged: {len(unchanged_paths):,}")
            print(f"New: {len(new_paths):,}")
            print(f"Modified: {len(modified_paths):,}")
            print(f"Removed: {len(deleted_paths):,}")
            if to_embed:
                estimated = len(to_embed) / IMAGES_PER_SECOND
                print(f"\nTo embed: {len(to_embed):,} images (~{format_time(estimated)})")
        else:
            sync_embeddings(directory, recursive=not args.no_recursive)

    if args.drive_folder and not args.dry_run:
        sync_drive_embeddings(args.drive_folder)


if __name__ == "__main__":
    main()
