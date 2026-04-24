"""FastAPI server for image search — Drive image proxy, web UI, REST API."""

import logging
import logging.handlers
import math
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import daft
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
from pydantic import BaseModel

from core import load_model, embed_text, cosine_similarity, DB_PATH
from drive import get_service

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "server.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)
log = logging.getLogger("image-search")

# Global state
model = None
processor = None
device = None
embeddings_df = None
_drive_service = None

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, device, embeddings_df

    log.info("Loading SigLIP model...")
    model, processor, device = load_model()
    log.info("Model loaded on %s", device)

    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        log.info("Loaded %d embeddings from %s", len(embeddings_df), DB_PATH)
    else:
        log.warning("No embeddings found at %s — run embed.py first", DB_PATH)

    yield

    log.info("Server shutting down")


app = FastAPI(title="Trobz Image Search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _drive_svc():
    global _drive_service
    if _drive_service is None:
        _drive_service = get_service()
    return _drive_service


class SearchRequest(BaseModel):
    query: str
    limit: int = 12
    sort_by: str = "combined"
    quality_weight: float = 0.6
    min_relevance: float = 0.0


class SearchResult(BaseModel):
    filename: str
    path: str
    score: float
    relevance: float
    aesthetic_score: float | None = None
    drive_url: str | None = None
    file_id: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_images: int


def _aesthetic(v) -> float:
    """Normalise aesthetic score: None/NaN → 0.5 (neutral)."""
    try:
        f = float(v)
        return 0.5 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.5


def _extract_file_id(path: str) -> str | None:
    """Extract Drive file ID from a path like 'drive://1abc...' or 'drive/1abc...'."""
    if path.startswith("drive:"):
        return path.split("//", 1)[-1] if "//" in path else path.split("/", 1)[-1]
    if path.startswith("drive/"):
        return path.split("/", 1)[1]
    return None


@app.get("/health")
async def health():
    return {"status": "ok", "embeddings": len(embeddings_df) if embeddings_df is not None else 0}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if embeddings_df is None:
        return SearchResponse(results=[], total_images=0)

    log.info("Search: query=%r sort_by=%s limit=%d", request.query, request.sort_by, request.limit)

    query_embedding = embed_text(request.query, model, processor, device)

    data = embeddings_df.to_pydict()
    paths = data["path"]
    vectors = data["vector"]
    drive_urls = data.get("drive_url", [None] * len(paths))
    filenames = data.get("filename", [None] * len(paths))
    aesthetic_scores = data.get("aesthetic_score", [None] * len(paths))

    relevance_scores = [
        -1.0 if np.allclose(arr := np.array(vec, dtype=np.float32), 0)
        else cosine_similarity(query_embedding, arr)
        for vec in vectors
    ]

    qw = max(0.0, min(1.0, request.quality_weight))
    if request.sort_by == "quality":
        final_scores = [_aesthetic(a) for a in aesthetic_scores]
    elif request.sort_by == "combined":
        final_scores = [(1.0 - qw) * r + qw * _aesthetic(a) for r, a in zip(relevance_scores, aesthetic_scores)]
    else:
        final_scores = relevance_scores

    ranked = sorted(
        zip(paths, final_scores, relevance_scores, drive_urls, filenames, aesthetic_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    seen: set[str] = set()
    results = []
    for path, final_score, relevance, drive_url, filename, aesthetic in ranked:
        if len(results) >= request.limit:
            break
        if relevance <= 0 or relevance < request.min_relevance:
            continue
        if path in seen:
            continue
        seen.add(path)

        url = drive_url if drive_url and str(drive_url) not in ("None", "nan", "") else None
        name = filename if filename and str(filename) not in ("None", "nan", "") else Path(path).name

        results.append(SearchResult(
            filename=name,
            path=path,
            score=round(final_score, 3),
            relevance=round(relevance, 3),
            aesthetic_score=round(_aesthetic(aesthetic), 3) if aesthetic is not None else None,
            drive_url=url,
            file_id=_extract_file_id(path),
        ))

    log.info("Search returned %d results", len(results))
    return SearchResponse(results=results, total_images=len(paths))


@app.get("/image/{file_id}")
async def proxy_image(file_id: str, size: int = Query(default=800, le=1600)):
    """Proxy a Drive image — stream from Drive, resize, no local cache required."""
    import time
    t0 = time.monotonic()
    log.info("drive proxy start file_id=%s size=%d", file_id, size)
    try:
        svc = _drive_svc()
        request_obj = svc.files().get_media(fileId=file_id)
        buf = BytesIO()
        dl = MediaIoBaseDownload(buf, request_obj)
        done = False
        while not done:
            _, done = dl.next_chunk()
        t_download = time.monotonic() - t0
        log.info("drive proxy download done file_id=%s download=%.2fs bytes=%d",
                 file_id, t_download, buf.tell())
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        out = BytesIO()
        img.save(out, format="JPEG", quality=85)
        out.seek(0)

        t_total = time.monotonic() - t0
        log.info("drive proxy done file_id=%s total=%.2fs", file_id, t_total)
        return StreamingResponse(out, media_type="image/jpeg", headers={
            "Cache-Control": "public, max-age=3600",
        })
    except Exception as e:
        log.warning("drive proxy failed file_id=%s elapsed=%.2fs error=%s",
                    file_id, time.monotonic() - t0, e)
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
async def ui():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Trobz Image Search server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"], help="Log level (default: info)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
