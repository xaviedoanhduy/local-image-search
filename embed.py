#!/usr/bin/env python3
"""CLI tool to sync image embeddings from a directory or Google Drive folder."""

import argparse
import shutil
import sys
import time
import unicodedata
import warnings
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
    except (KeyError, ValueError):
        df = df.with_column("drive_url", daft.lit(None).cast(DataType.string()))
    # Always cast to string so types match across local and Drive DataFrames
    return df.with_column("drive_url", col("drive_url").cast(DataType.string()))


def _ensure_filename_column(df) -> "daft.DataFrame":
    """Ensure filename column exists (backward compatibility)."""
    try:
        df.schema()["filename"]
    except (KeyError, ValueError):
        df = df.with_column("filename", daft.lit(None).cast(DataType.string()))
    return df.with_column("filename", col("filename").cast(DataType.string()))


def _ensure_drive_folder_id_column(df) -> "daft.DataFrame":
    """Ensure drive_folder_id column exists (backward compatibility)."""
    try:
        df.schema()["drive_folder_id"]
    except (KeyError, ValueError):
        df = df.with_column("drive_folder_id", daft.lit(None).cast(DataType.string()))
    return df.with_column("drive_folder_id", col("drive_folder_id").cast(DataType.string()))


# Canonical column order for all DataFrames written to the DB
_CANONICAL_COLS = ["path", "mtime", "drive_url", "drive_folder_id", "filename", "vector", "aesthetic_score", "md5"]


def _ensure_aesthetic_score_column(df) -> "daft.DataFrame":
    """Ensure aesthetic_score column exists (backward compatibility)."""
    try:
        df.schema()["aesthetic_score"]
    except (KeyError, ValueError):
        df = df.with_column("aesthetic_score", daft.lit(None).cast(DataType.float32()))
    return df.with_column("aesthetic_score", col("aesthetic_score").cast(DataType.float32()))


def _ensure_md5_column(df) -> "daft.DataFrame":
    """Ensure md5 column exists (backward compatibility)."""
    try:
        df.schema()["md5"]
    except (KeyError, ValueError):
        df = df.with_column("md5", daft.lit(None).cast(DataType.string()))
    return df.with_column("md5", col("md5").cast(DataType.string()))


def _normalize_schema(df) -> "daft.DataFrame":
    """Ensure optional columns exist and reorder to canonical schema."""
    df = _ensure_drive_url_column(df)
    df = _ensure_filename_column(df)
    df = _ensure_drive_folder_id_column(df)
    df = _ensure_aesthetic_score_column(df)
    df = _ensure_md5_column(df)
    return df.select(*_CANONICAL_COLS)


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
        df_new = _normalize_schema(df_new)  # add drive_folder_id=None, aesthetic_score=None

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

            # Ensure drive_url and filename columns exist for backward compatibility
            df_unchanged = _normalize_schema(df_unchanged)

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
            df_drive = _normalize_schema(df_drive)
            df_final = df_final.concat(df_drive)
    else:
        # No new embeddings, just filter out deleted local files
        df_existing = daft.read_lance(DB_PATH)
        # Keep current local paths + all Drive entries
        keep_list = list(current_paths) + [p for p in stored_all if p.startswith("drive://")]
        df_final = df_existing.where(col("path").is_in(keep_list))
        df_final = df_final.with_column("vector", col("vector").cast(VECTOR_DTYPE))
        df_final = _normalize_schema(df_final)

    # Collect into memory before deleting source DB (Daft DataFrames are lazy)
    df_final = df_final.collect()
    # Write to Lance — delete first if it exists so schema changes don't cause conflicts
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)
    df_final.write_lance(DB_PATH, mode="create")

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

    Downloads each image to memory (no local storage needed), embeds it
    using SigLIP, and stores the result in the Lance DB alongside any
    locally-indexed images.

    Args:
        folder_url_or_id: Google Drive folder URL or bare folder ID
        log_fn: Function to use for logging (default: print)

    Returns:
        Dict with stats: {new, skipped, failed, total, elapsed}
    """
    from googleapiclient.http import MediaIoBaseDownload
    from drive import extract_folder_id, get_service, list_folder_files_with_ids

    folder_id = extract_folder_id(folder_url_or_id)
    log_fn(f"Fetching file list from Drive folder {folder_id}...")

    service = get_service()
    all_files = list_folder_files_with_ids(folder_id)
    image_files = [
        f for f in all_files
        if Path(f["name"]).suffix.lower() in IMAGE_EXTENSIONS
    ]
    log_fn(f"Found {len(image_files)} image(s) on Drive")

    # Load existing DB and classify each file from the Drive listing:
    #   - "needs_embed"   : file_id not in DB at all → download + embed
    #   - "needs_backfill": file_id in DB but drive_folder_id is None → metadata-only update
    #   - "up_to_date"    : file_id in DB with correct drive_folder_id → skip
    needs_embed: list[dict] = []    # full file_info dicts
    needs_backfill: list[str] = []  # drive paths that need folder_id written

    if Path(DB_PATH).exists():
        df_existing_check = daft.read_lance(DB_PATH)
        df_existing_check = _ensure_drive_folder_id_column(df_existing_check)
        df_existing_check = _ensure_md5_column(df_existing_check)
        check_data = df_existing_check.select("path", "drive_folder_id", "md5").collect().to_pydict()
        existing_folder_by_path: dict[str, str | None] = dict(
            zip(check_data["path"], check_data["drive_folder_id"])
        )
        existing_md5s: set[str] = {m for m in check_data["md5"] if m}
    else:
        existing_folder_by_path = {}
        existing_md5s = set()

    skipped_duplicates = 0
    for file_info in image_files:
        drive_path = f"drive://{file_info['id']}"
        md5 = file_info.get("md5Checksum")
        if drive_path not in existing_folder_by_path:
            # Skip if same content (different file_id, same md5 — e.g. "Copy of X")
            if md5 and md5 in existing_md5s:
                skipped_duplicates += 1
                continue
            needs_embed.append(file_info)
        elif existing_folder_by_path[drive_path] is None:
            needs_backfill.append(drive_path)
        # else: already has correct folder_id → skip

    log_fn(f"  To embed (new): {len(needs_embed)} | "
           f"Metadata backfill: {len(needs_backfill)} | "
           f"Duplicate content skipped: {skipped_duplicates} | "
           f"Up-to-date: {len(image_files) - len(needs_embed) - len(needs_backfill) - skipped_duplicates}")

    t0 = time.perf_counter()
    failed = 0
    new_rows: list[dict] = []

    # --- Step 1: embed new files (download + embed) ---
    if needs_embed:
        log_fn("Loading SigLIP model...")
        model, processor, device = load_model()

        batch_images: list[Image.Image] = []
        batch_meta: list[dict] = []

        def flush_batch():
            nonlocal new_rows
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

        for i, file_info in enumerate(needs_embed):
            file_id = file_info["id"]
            name = file_info["name"]
            drive_path = f"drive://{file_id}"
            name_nfc = unicodedata.normalize("NFC", name)
            drive_url = file_info.get(
                "webViewLink", f"https://drive.google.com/file/d/{file_id}/view"
            )
            try:
                request = service.files().get_media(fileId=file_id)
                buf = BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                buf.seek(0)
                image = Image.open(buf).convert("RGB")
            except Exception as e:
                log_fn(f"  WARNING: could not download {name}: {e}")
                failed += 1
                continue

            batch_images.append(image)
            batch_meta.append({
                "path": drive_path,
                "filename": name_nfc,
                "mtime": 0.0,
                "drive_url": drive_url,
                "drive_folder_id": folder_id,
                "md5": file_info.get("md5Checksum"),
            })
            log_fn(f"  [{i + 1}/{len(needs_embed)}] {name}")

            if len(batch_images) >= DRIVE_BATCH_SIZE:
                flush_batch()

        flush_batch()

    # --- Step 2: metadata-only backfill (no download, no embed) ---
    # Read full DB, update drive_folder_id for matching paths in memory
    if needs_embed or needs_backfill:
        if Path(DB_PATH).exists():
            df_existing = daft.read_lance(DB_PATH)
            df_existing = df_existing.with_column("vector", col("vector").cast(VECTOR_DTYPE))
            df_existing = _normalize_schema(df_existing)
            existing_data = df_existing.collect().to_pydict()
        else:
            existing_data = {c: [] for c in _CANONICAL_COLS}

        if needs_backfill:
            backfill_set = set(needs_backfill)
            log_fn(f"  Backfilling drive_folder_id for {len(needs_backfill)} entries (no re-embed)...")
            existing_data["drive_folder_id"] = [
                folder_id if (p in backfill_set) else fid
                for p, fid in zip(existing_data["path"], existing_data["drive_folder_id"])
            ]

        if new_rows:
            for key in _CANONICAL_COLS:
                if key == "vector":
                    existing_data[key] = existing_data[key] + [r["vector"] for r in new_rows]
                else:
                    existing_data[key] = existing_data[key] + [r.get(key) for r in new_rows]

        df_final = daft.from_pydict(existing_data)
        df_final = df_final.with_column("vector", col("vector").cast(VECTOR_DTYPE))
        df_final = _normalize_schema(df_final)

    if not needs_embed and not needs_backfill:
        log_fn("Nothing to do.")
        return {"new": 0, "backfilled": 0, "failed": 0, "total": len(image_files), "elapsed": 0}

    # Collect into memory before deleting source DB (Daft DataFrames are lazy)
    df_final = df_final.collect()
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)
    df_final.write_lance(DB_PATH, mode="create")

    elapsed = time.perf_counter() - t0
    log_fn(f"\nDone in {format_time(elapsed)}: "
           f"{len(new_rows)} new embedded, {len(needs_backfill)} metadata backfilled")

    return {
        "new": len(new_rows),
        "backfilled": len(needs_backfill),
        "failed": failed,
        "total": len(image_files),
        "elapsed": elapsed,
    }


AESTHETIC_MODEL = "cafeai/cafe_aesthetic"
AESTHETIC_BATCH_SIZE = 8
AESTHETIC_SAVE_INTERVAL = 500  # save checkpoint to DB every N images scored


def _flushing_print(msg):
    """print() with flush=True so progress is visible when stdout is redirected to a file."""
    print(msg, flush=True)


def _write_db(data: dict, log_fn) -> None:
    """Write data dict to Lance DB (delete + recreate)."""
    df = daft.from_pydict(data)
    df = df.with_column("vector", col("vector").cast(VECTOR_DTYPE))
    df = _normalize_schema(df)
    df = df.collect()
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)
    df.write_lance(DB_PATH, mode="create")
    log_fn(f"  [checkpoint] saved to DB")


def add_aesthetic_scores(log_fn=_flushing_print) -> dict:
    """Score all unscored images with cafeai/cafe_aesthetic and store results in the DB.

    Processes local images from disk (batched) and Drive images by downloading
    to memory. Saves a checkpoint to DB every AESTHETIC_SAVE_INTERVAL images so
    the run can be safely resumed if interrupted.

    Returns:
        Dict with stats: {scored, skipped, failed}
    """
    from transformers import pipeline

    if not Path(DB_PATH).exists():
        log_fn("No embeddings DB found.")
        return {"scored": 0, "skipped": 0, "failed": 0}

    # Load full DB
    df = daft.read_lance(DB_PATH)
    df = df.with_column("vector", col("vector").cast(VECTOR_DTYPE))
    df = _normalize_schema(df)
    data = df.collect().to_pydict()
    total = len(data["path"])

    def _needs_score(v) -> bool:
        if v is None:
            return True
        try:
            import math
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return True

    unscored_indices = [i for i, s in enumerate(data["aesthetic_score"]) if _needs_score(s)]
    skipped = total - len(unscored_indices)

    if not unscored_indices:
        log_fn(f"All {total} images already scored — nothing to do.")
        return {"scored": 0, "skipped": total, "failed": 0}

    log_fn(f"Scoring {len(unscored_indices)} images (skipping {skipped} already scored)...")
    log_fn(f"Loading aesthetic model {AESTHETIC_MODEL}...")

    scorer = pipeline("image-classification", model=AESTHETIC_MODEL)

    # id2label maps int → str, e.g. {0: "not_aesthetic", 1: "aesthetic"}
    aesthetic_label = next(
        (lbl for lbl in scorer.model.config.id2label.values()
         if "aesthetic" in lbl.lower() and "not" not in lbl.lower()),
        "aesthetic"
    )

    failed = 0
    scored = 0
    since_last_save = 0  # images scored since last checkpoint

    def _score_images(images: list) -> list[float]:
        """Run scorer on a batch of PIL images, return list of aesthetic scores."""
        try:
            results = scorer(images)
            if isinstance(results[0], dict):
                results = [results]
            scores = []
            for res in results:
                s = next((r["score"] for r in res if r["label"] == aesthetic_label), 0.5)
                scores.append(float(s))
            return scores
        except Exception as e:
            log_fn(f"  WARNING: batch scoring failed: {e}")
            return [None] * len(images)

    def _flush_batch(batch_idx: list[int], batch_img: list) -> int:
        """Score a batch, update data in-place. Returns number of newly scored images."""
        nonlocal scored, failed
        if not batch_img:
            return 0
        results = _score_images(batch_img)
        n = 0
        for idx, score in zip(batch_idx, results):
            if score is None:
                data["aesthetic_score"][idx] = None
                failed += 1
            else:
                data["aesthetic_score"][idx] = score
                scored += 1
                n += 1
        return n

    def _maybe_save(n_new: int) -> None:
        """Save checkpoint if enough images have been scored since the last save."""
        nonlocal since_last_save
        since_last_save += n_new
        if since_last_save >= AESTHETIC_SAVE_INTERVAL:
            _write_db(data, log_fn)
            since_last_save = 0

    # --- Local images (read from disk, batch scoring) ---
    local_indices = [i for i in unscored_indices if not data["path"][i].startswith("drive://")]
    if local_indices:
        log_fn(f"Scoring {len(local_indices)} local images...")
        batch_idx: list[int] = []
        batch_img: list = []

        for i, idx in enumerate(local_indices):
            path = data["path"][idx]
            try:
                batch_img.append(Image.open(path).convert("RGB"))
                batch_idx.append(idx)
            except Exception as e:
                log_fn(f"  WARNING: could not load {path}: {e}")
                data["aesthetic_score"][idx] = None
                failed += 1
                continue

            if len(batch_img) >= AESTHETIC_BATCH_SIZE:
                n = _flush_batch(batch_idx, batch_img)
                batch_idx.clear()
                batch_img.clear()
                _maybe_save(n)

            if (i + 1) % 100 == 0:
                log_fn(f"  Local: {i + 1}/{len(local_indices)}")

        n = _flush_batch(batch_idx, batch_img)
        _maybe_save(n)

    # --- Drive images (download to memory, score in small batches) ---
    drive_indices = [i for i in unscored_indices if data["path"][i].startswith("drive://")]
    if drive_indices:
        log_fn(f"Scoring {len(drive_indices)} Drive images (downloading to memory)...")
        try:
            from drive import get_service
            from googleapiclient.http import MediaIoBaseDownload
            service = get_service()
        except Exception as e:
            log_fn(f"  WARNING: could not connect to Google Drive: {e}")
            for idx in drive_indices:
                data["aesthetic_score"][idx] = None
            failed += len(drive_indices)
            drive_indices = []

        batch_idx = []
        batch_img = []

        for i, idx in enumerate(drive_indices):
            file_id = data["path"][idx].removeprefix("drive://")
            try:
                request = service.files().get_media(fileId=file_id)
                buf = BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                buf.seek(0)
                batch_img.append(Image.open(buf).convert("RGB"))
                batch_idx.append(idx)
            except Exception as e:
                log_fn(f"  WARNING: could not download {file_id}: {e}")
                data["aesthetic_score"][idx] = None
                failed += 1
                continue

            if len(batch_img) >= AESTHETIC_BATCH_SIZE:
                n = _flush_batch(batch_idx, batch_img)
                batch_idx.clear()
                batch_img.clear()
                _maybe_save(n)

            if (i + 1) % 100 == 0:
                log_fn(f"  Drive: {i + 1}/{len(drive_indices)}")

        n = _flush_batch(batch_idx, batch_img)
        _maybe_save(n)

    log_fn(f"\nAesthetic scoring done: {scored} scored, {skipped} skipped, {failed} failed")

    # Final save — only needed if there are unsaved scores since the last checkpoint
    if since_last_save > 0:
        _write_db(data, log_fn)
    log_fn(f"DB updated — {total} total entries.")
    return {"scored": scored, "skipped": skipped, "failed": failed}


def dedup_by_md5(log_fn=print):
    """Fetch md5Checksum from Drive for all indexed Drive files, then remove duplicate-content entries."""
    from drive import get_service

    if not Path(DB_PATH).exists():
        log_fn("No DB found.")
        return

    df = daft.read_lance(DB_PATH)
    df = _normalize_schema(df)
    data = df.collect().to_pydict()

    drive_paths = [p for p in data["path"] if p.startswith("drive://")]
    log_fn(f"Fetching md5Checksum for {len(drive_paths)} Drive entries...")

    service = get_service()

    # Batch fetch metadata — files.get per file_id
    md5_by_path: dict[str, str] = {}
    for i, path in enumerate(drive_paths):
        file_id = path.removeprefix("drive://")
        try:
            meta = service.files().get(fileId=file_id, fields="id,md5Checksum").execute()
            md5 = meta.get("md5Checksum")
            if md5:
                md5_by_path[path] = md5
        except Exception as e:
            log_fn(f"  WARNING: could not fetch md5 for {file_id}: {e}")
        if (i + 1) % 500 == 0:
            log_fn(f"  {i + 1}/{len(drive_paths)} fetched...")

    log_fn(f"Got md5 for {len(md5_by_path)}/{len(drive_paths)} Drive files")

    # Find duplicates: for each md5, keep the first occurrence (by index), drop the rest
    seen_md5: dict[str, int] = {}  # md5 → index of kept row
    duplicate_paths: set[str] = set()
    for path, md5 in md5_by_path.items():
        if md5 in seen_md5:
            duplicate_paths.add(path)
        else:
            seen_md5[md5] = 1

    if not duplicate_paths:
        log_fn("No duplicates found.")
        # Still write md5 values to DB
    else:
        log_fn(f"Found {len(duplicate_paths)} duplicate entries — removing...")

    # Write md5 values into data and drop duplicate rows
    new_data: dict[str, list] = {c: [] for c in _CANONICAL_COLS}
    for i, path in enumerate(data["path"]):
        if path in duplicate_paths:
            continue
        for col_name in _CANONICAL_COLS:
            val = data[col_name][i]
            if col_name == "md5" and path in md5_by_path:
                val = md5_by_path[path]
            new_data[col_name].append(val)

    df_final = daft.from_pydict(new_data)
    df_final = df_final.with_column("vector", col("vector").cast(VECTOR_DTYPE))
    df_final = _normalize_schema(df_final)
    df_final = df_final.collect()

    shutil.rmtree(DB_PATH)
    df_final.write_lance(DB_PATH, mode="create")
    log_fn(f"Done — {len(new_data['path'])} entries remaining (removed {len(duplicate_paths)}).")


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
    parser.add_argument(
        "--add-aesthetic-scores",
        action="store_true",
        help="Score all unscored images with cafeai/cafe_aesthetic and store results",
    )
    parser.add_argument(
        "--dedup-md5",
        action="store_true",
        help="Fetch md5Checksum from Drive for all indexed files and remove duplicate-content entries",
    )

    args = parser.parse_args()

    if not args.directory and not args.drive_folder and not args.add_aesthetic_scores and not args.dedup_md5:
        parser.error("Provide a local directory, --drive-folder, --add-aesthetic-scores, and/or --dedup-md5")

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

    if args.add_aesthetic_scores:
        add_aesthetic_scores()

    if args.dedup_md5:
        dedup_by_md5()


if __name__ == "__main__":
    main()
