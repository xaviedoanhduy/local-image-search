# Local Image Search MCP

Give your AI coding agent the ability to search through your local images and Google Drive photos. Cross-platform MCP server (Linux, macOS, Windows). Uses SigLIP (HuggingFace) for embeddings, Daft for batch processing, and Lance for vector storage.

https://github.com/user-attachments/assets/41e167f0-bb73-4310-8c1c-4be07af21cc1

## Features

- **Cross-platform** — runs on Linux, macOS, and Windows (CPU or CUDA)
- **Google Drive support** — index photos directly from multiple Drive folders, no local download needed
- **Recursive folder traversal** — automatically indexes images at any subfolder depth
- **Multi-folder** — index multiple Drive folders independently; no duplicate entries
- **MCP Server** — works with Claude Code and Claude Desktop
- **Natural language search** — find images by describing them ("person wearing a watch", "sunset by the ocean")
- **Better quality** — SigLIP so400m-patch14-384 outperforms CLIP base for fine-grained details
- **Incremental sync** — re-runs skip unchanged files; new metadata (e.g. folder ID) backfills in milliseconds without re-embedding
- **Lightweight idle** — model unloads from RAM automatically after inactivity; only ~50 MB used at rest

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- CUDA GPU (optional, but speeds up embedding significantly)

## Quick Start

### Claude Code

**Option 1: CLI**
```bash
claude mcp add local-image-search -- uvx local-image-search ~/Pictures
```

**Option 2: Manual** — add to `~/.claude.json`:
```json
{
  "mcpServers": {
    "local-image-search": {
      "command": "uvx",
      "args": ["local-image-search", "/home/user/Pictures"]
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config:
```json
{
  "mcpServers": {
    "local-image-search": {
      "command": "uvx",
      "args": ["local-image-search", "/home/user/Pictures"]
    }
  }
}
```

Restart Claude after setup. On first use of `search_images`, the SigLIP model (~800MB) downloads and loads automatically. After 5 minutes of inactivity the model unloads to free RAM — it reloads on the next search.

### Development Setup (without uvx)

If you're running from a local clone instead of a published package:

```bash
# Claude Code
claude mcp add local-image-search -- uv --directory /path/to/local-image-search run python mcp_server.py

# Or manually in ~/.claude.json
{
  "mcpServers": {
    "local-image-search": {
      "command": "uv",
      "args": ["--directory", "/path/to/local-image-search", "run", "python", "mcp_server.py"]
    }
  }
}
```

### Custom Configuration

**Scan a specific folder:**
```json
{ "args": ["local-image-search", "~/Pictures"] }
```

**Custom excludes:**
```json
{
  "args": ["local-image-search"],
  "env": { "EXCLUDE_DIRS": "Downloads,Desktop,Movies" }
}
```

**Faster refresh:**
```json
{
  "env": { "REFRESH_INTERVAL": "30" }
}
```

**Tune memory behaviour:**
```json
{
  "env": {
    "MODEL_IDLE_TIMEOUT": "300",
    "REFRESH_STARTUP_DELAY": "120"
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REFRESH_INTERVAL` | `60` | Seconds between embedding refresh cycles |
| `REFRESH_STARTUP_DELAY` | `120` | Seconds to wait before the first refresh after startup |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds of inactivity before unloading the model from RAM |
| `EXCLUDE_DIRS` | (none) | Comma-separated directory names to exclude when scanning |

### Configuration Logic

| Options | Root | Excludes |
|---------|------|----------|
| None | `~` (home) | Default excludes |
| Root only | Custom root | None |
| Excludes only | `~` (home) | Custom excludes |
| Root + Excludes | Custom root | Custom excludes |

**Default excludes:** Library, .Trash, .cache, Cache, node_modules, .git, .venv, venv

### MCP Tools

- `search_images(query, limit, sort_by, quality_weight, min_relevance)` — search for images by text description
  - `sort_by`: `"relevance"` (default), `"quality"`, or `"combined"`
  - `quality_weight`: blend factor for `"combined"` mode (0.0–1.0, default `0.5`)
  - `min_relevance`: relevance floor before quality ranking — recommended `0.08–0.10` with `sort_by="quality"` to avoid returning beautiful but off-topic results
  - Returns `filename`, `path`, `score`, `relevance`, `aesthetic_score`, and `drive_url` (Drive results only)
- `get_status()` — check if the service is ready; shows `total_images` and current `model` state

### Aesthetic Scoring (Quality Ranking)

After indexing, you can score all images with an aesthetic quality model:

```bash
uv run --extra drive python embed.py --add-aesthetic-scores
```

This runs [`cafeai/cafe_aesthetic`](https://huggingface.co/cafeai/cafe_aesthetic) (ViT classifier) on every image and stores a `0–1` quality score in the DB. Local images are scored from disk; Drive images are downloaded to memory. Re-running only scores new (unscored) entries.

Once scored, you can use `sort_by` in `search_images`:
- `"relevance"` — pure semantic match (default, classic search)
- `"quality"` — highest aesthetic scores regardless of topic (Nils-style: best photos first)
- `"combined"` — blend of relevance + quality (e.g. `quality_weight=0.7` for quality-biased results)

### Memory & Performance

The MCP server is designed to be lightweight when idle:

| State | RAM usage |
|-------|-----------|
| Startup | ~50 MB (Lance DB index only) |
| Active search | ~900 MB (SigLIP model loaded) |
| Idle > 5 min | ~50 MB (model unloaded automatically) |

- **Startup**: only the embeddings index is loaded — the SigLIP model (~800 MB) is **not** loaded until the first `search_images` call
- **Idle unload**: after `MODEL_IDLE_TIMEOUT` seconds without a search, the model is unloaded and RAM is freed
- **Refresh delay**: the background embedding refresh waits `REFRESH_STARTUP_DELAY` seconds before its first run, so it doesn't compete with Claude's own startup

## Development Setup

```bash
git clone https://github.com/Eventual-Inc/local-image-search.git
cd local-image-search
uv sync
```

The SigLIP model (~800MB) downloads automatically from HuggingFace on first use.

## CLI Usage

### Index images from a local directory
```bash
uv run python embed.py ~/Pictures              # embed all images
uv run python embed.py ~/Pictures --dry-run    # count and estimate time
uv run python embed.py . --no-recursive        # current dir only
uv run --extra drive python embed.py --add-aesthetic-scores  # score quality (run once after indexing)
```

Embeddings are cached in `embeddings.lance/`. Re-running skips unchanged files.

### Index images from Google Drive

```bash
# One-time setup — see "Google Drive Setup" below
uv run --extra drive python embed.py --drive-folder <folder-id-or-url>

# Or combine local + Drive in one pass
uv run --extra drive python embed.py ~/Pictures --drive-folder <folder-id-or-url>
```

Drive entries are stored alongside local entries with a `drive_url` pointing back to the original file. They are preserved when re-running local syncs.

Nested folders are traversed recursively — all images at any depth are indexed.

> **Note:** Video files (`.mp4`) and metadata files (`.xml`) are skipped automatically. Only image formats listed in the table below are indexed.

### Web UI & REST API

Start the FastAPI server (loads model once at startup):
```bash
uv run --extra drive python server.py
```

Then open **http://localhost:8000** in your browser — a search UI with image grid, sort controls, and lightbox preview.

Drive images are proxied server-side (`/image/<file_id>`) so visitors don't need their own Google Drive access.

Search via CLI:
```bash
uv run python search.py "sunset"        # list results
uv run python search.py "people" -n 10  # show 10 results
```

Or via REST API:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "team meeting", "limit": 5, "sort_by": "combined", "quality_weight": 0.6}'
```

### Supported formats

| Format | Extensions |
|--------|------------|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| GIF | `.gif` |
| WebP | `.webp` |
| BMP | `.bmp` |
| TIFF | `.tiff`, `.tif` |
| HEIC/HEIF | `.heic`, `.heif` |

Corrupted or unreadable images get zero vectors (won't appear in search results).

## Google Drive Setup

Google Drive indexing requires a one-time OAuth2 setup:

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → enable **Google Drive API**
3. Create credentials → **OAuth client ID** → **Desktop app** → download JSON
4. Save the downloaded file as `credentials.json` in the project directory
5. Run `embed.py --drive-folder <id>` — a browser window opens for consent on first run
6. The token is saved to `token.json` (auto-refreshed on subsequent runs)

```bash
# Folder ID from URL: https://drive.google.com/drive/folders/<FOLDER_ID>
uv run --extra drive python embed.py --drive-folder 1ABC123xyz...
# Or pass the full URL directly
uv run --extra drive python embed.py --drive-folder "https://drive.google.com/drive/folders/1ABC123xyz..."
```

> `credentials.json` and `token.json` are gitignored — never commit them.

## Project Structure

```
local-image-search/
├── core.py              # Shared utilities: EmbedImages UDF, load_model, embed_text/images
├── embed.py             # CLI: sync embeddings from local dir and/or Google Drive
├── drive.py             # Google Drive helpers: OAuth2, recursive folder listing
├── mcp_server.py        # MCP server: lazy model load, idle unload, search_images tool
├── server.py            # FastAPI server: REST API + Drive image proxy + web UI
├── static/
│   ├── index.html       # Web UI markup
│   ├── style.css        # Web UI styles
│   └── app.js           # Web UI logic (search, grid, lightbox)
├── search.py            # CLI search client (queries server.py)
├── data/
│   └── pokemon/         # Pokemon artwork (1025 images, for testing)
├── embeddings.lance/    # Lance DB — embeddings + metadata (generated, gitignored)
├── pyproject.toml       # Project dependencies
└── uv.lock              # Dependency lockfile
```

## Data Attribution

### Pokemon Artwork
- **Source**: [PokeAPI/sprites](https://github.com/PokeAPI/sprites)
- **License**: CC0 1.0 Universal
- **Copyright**: All Pokemon images are Copyright The Pokemon Company
