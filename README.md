# Local Image Search MCP

Give your AI coding agent the ability to search through your local images. Cross-platform MCP server (Linux, macOS, Windows). Uses SigLIP (HuggingFace) for embeddings, Daft for batch processing, and Lance for vector storage.

https://github.com/user-attachments/assets/41e167f0-bb73-4310-8c1c-4be07af21cc1

## Features

- **Cross-platform** — runs on Linux, macOS, and Windows (CPU or CUDA)
- **MCP Server** — works with Claude Code and Claude Desktop
- **Natural language search** — find images by describing them ("person wearing a watch", "sunset by the ocean")
- **Better quality** — SigLIP so400m-patch14-384 outperforms CLIP base for fine-grained details
- **Incremental sync** — re-runs skip unchanged files

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

Restart Claude after setup. The first run downloads the SigLIP model (~800MB) and embeds your images. After that, only new or changed files are processed.

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

**Different model** (e.g. a smaller/faster one):
```json
{
  "env": { "IMAGE_SEARCH_MODEL": "openai/clip-vit-large-patch14" }
}
```

### Configuration Logic

| Options | Root | Excludes |
|---------|------|----------|
| None | `~` (home) | Default excludes |
| Root only | Custom root | None |
| Excludes only | `~` (home) | Custom excludes |
| Root + Excludes | Custom root | Custom excludes |

**Default excludes:** Library, .Trash, .cache, Cache, node_modules, .git, .venv, venv

### MCP Tools

- `search_images(query, limit)` — search for images by text description; returns `filename`, `path`, and `score`
- `get_status()` — check if the service is ready (model loaded, embeddings synced)

## Development Setup

```bash
git clone https://github.com/xaviedoanhduy/local-image-search.git
cd local-image-search
uv sync
```

The SigLIP model (~800MB) downloads automatically from HuggingFace on first use.

## CLI Usage

### Index images from a local directory
```bash
uv run python embed.py ~/Pictures           # embed all images
uv run python embed.py ~/Pictures --dry-run # count and estimate time
uv run python embed.py . --no-recursive     # current dir only
```

Embeddings are cached in `embeddings.lance/`. Re-running skips unchanged files.

### Search

Start the FastAPI server (loads model once):
```bash
uv run python server.py
```

Search via CLI:
```bash
uv run python search.py "sunset"        # list results
uv run python search.py "people" -n 10  # show 10 results
```

Or via API:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "yellow mouse", "limit": 5}'
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

## Project Structure

```
local-image-search/
├── core.py              # Shared utilities: EmbedImages UDF, load_model, embed_text
├── embed.py             # CLI: sync embeddings from local directory
├── mcp_server.py        # MCP server entry point (background model load + refresh)
├── server.py            # FastAPI server for local REST API
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
