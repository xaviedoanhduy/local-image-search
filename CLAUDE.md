# Local Image Search - Project Notes

See README.md for project structure and setup instructions.

## What We've Done

- Replaced MLX/CLIP with HuggingFace SigLIP (`google/siglip-so400m-patch14-384`) for cross-platform support (Linux, macOS, Windows)
- Created Daft-based batch embedding with `@daft.cls`
- Benchmarked performance (see benchmark_plot.png)
- Added Pokemon dataset (1025 images) for testing
- Implemented Lance DB storage for embeddings
- Added FastAPI server, web UI, and search CLI
- Added incremental embedding (skips unchanged files by path + mtime)
- Added error handling for corrupted/unreadable images
- Added Google Drive integration (OAuth2, download-to-memory, no local copy)
- Drive folder traversal is **recursive** — all subfolders at any depth are indexed
- Multi-folder Drive support — `drive_folder_id` column tracks which folder each image belongs to
- Smart re-index: metadata-only backfill (~100ms) when only new columns need updating, no re-embed
- Dedup search results by `path` (Drive file_id) — same file in multiple folders appears once
- MCP server: lazy model load (only on first `search_images` call)
- MCP server: idle unload — model freed after `MODEL_IDLE_TIMEOUT` seconds (default 5 min)
- MCP server: startup refresh delayed by `REFRESH_STARTUP_DELAY` (default 2 min)
- Aesthetic scoring with `cafeai/cafe_aesthetic` — 0–1 quality score per image, checkpoints every 500
- `search_images` supports `sort_by`: relevance | quality | combined, with `min_relevance` floor
- Web UI at `/` — image grid with lightbox, sort controls, Drive proxy thumbnails
- Drive image proxy at `/image/<file_id>` — streams Drive images server-side, no user auth needed

## MCP Server Setup (Development)

To test locally during development:

```bash
# Add to Claude Code (must split command and args properly)
claude mcp add -s user local-image-search -- uv --directory /path/to/local-image-search run python mcp_server.py

# Restart Claude Code to load the server
```

**Memory behaviour (important for low-RAM machines):**
- Startup: only Lance DB is read (~50 MB). Model is NOT loaded.
- First `search_images` call: model loads (~800 MB, ~5-10s delay).
- After `MODEL_IDLE_TIMEOUT` seconds idle: model unloads automatically, RAM freed.
- Embedding refresh starts `REFRESH_STARTUP_DELAY` seconds after launch (not immediately).

**Gotchas we encountered:**

1. **stdout is reserved for MCP protocol** - Any `print()` statements corrupt the JSON-RPC communication. Use `print(..., file=sys.stderr)` for logging.

2. **Command must be split from args** - This is wrong:
   ```json
   "command": "uv --directory /path run python mcp_server.py",
   "args": []
   ```
   This is correct:
   ```json
   "command": "uv",
   "args": ["--directory", "/path", "run", "python", "mcp_server.py"]
   ```

3. **Restart required** - Claude Code must be fully restarted (not just new conversation) to pick up MCP config changes.

