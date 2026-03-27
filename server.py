"""FastAPI server for image search."""

from pathlib import Path

import daft
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from core import load_model, embed_text, cosine_similarity, DB_PATH

app = FastAPI(title="Local Image Search")

# Global state - loaded on startup
model = None
processor = None
device = None
embeddings_df = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    filename: str
    path: str
    score: float
    drive_url: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_images: int


@app.on_event("startup")
async def startup():
    """Load model and embeddings on startup."""
    global model, processor, device, embeddings_df

    print("Loading SigLIP model...")
    model, processor, device = load_model()

    print("Loading embeddings...")
    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        print(f"Loaded {len(embeddings_df)} embeddings")
    else:
        print("No embeddings found. Run embed.py first.")
        embeddings_df = None


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "embeddings_loaded": embeddings_df is not None}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for images matching the query."""
    if embeddings_df is None:
        return SearchResponse(results=[], total_images=0)

    # Embed the query text
    query_embedding = embed_text(request.query, model, processor, device)

    # Get all embeddings and paths
    data = embeddings_df.to_pydict()
    paths = data["path"]
    vectors = data["vector"]
    drive_urls = data.get("drive_url", [None] * len(paths))
    filenames = data.get("filename", [None] * len(paths))

    # Compute similarities
    scores = []
    for i, vec in enumerate(vectors):
        vec_array = np.array(vec, dtype=np.float32)
        # Skip zero vectors (failed images)
        if np.allclose(vec_array, 0):
            scores.append(-1.0)
        else:
            scores.append(cosine_similarity(query_embedding, vec_array))

    # Sort by score descending
    ranked = sorted(zip(paths, scores, drive_urls, filenames), key=lambda x: x[1], reverse=True)

    # Return top results
    results = []
    for path, score, drive_url, filename in ranked[:request.limit]:
        if score <= 0:
            continue
        url = drive_url if drive_url and str(drive_url) not in ("None", "nan", "") else None
        name = filename if filename and str(filename) not in ("None", "nan", "") else Path(path).name
        results.append(SearchResult(filename=name, path=path, score=score, drive_url=url))

    return SearchResponse(results=results, total_images=len(paths))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
