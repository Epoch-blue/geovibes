# CLAUDE.md - GeoVibes Project Guide

## Notification

- IMPORTANT: YOU MUST ALWAYS DO THIS: When you need to send me a notification because you need input or when you have finished a task, please use terminal-notifier tool like this: terminal-notifier -title "🔔 Claude Code: request" -message "Claude needs your permission to use ...", or terminal-notifier -title "✅ Claude Code: done" -message "The task has been completed"
- Always customise the message using a short summary of the input needed or the task just completed


After making changes, commit them with a useful commit message.
Use `uv run` to run with the correct env strategy
Do not leave comments unless explicitly asked to
Do not implement classes unless explicitly necessary, where possible use functions and keep things simple.
Be parsimonious with code.

## Coding Preferences

- **Fail Fast**: Do not use try-except statements. Let errors surface immediately for faster debugging.
- **Time All Steps**: When implementing workflows/pipelines, add timing to each step to understand performance characteristics.
- **Memory Constraints**: Target M1 Mac with 32GB RAM. Plan batch sizes and memory usage accordingly.
- **Physical Units**: Use proper physical units (e.g., meters for distances, not degrees).
- **CLI Scripts**: Use argparse for command-line interfaces.
- **Config Files**: Use YAML/JSON config files for complex parameter sets rather than many CLI arguments.
- **Docstrings**: Docstrings are acceptable when they add value; prefer them over inline comments.
- **Integration Tests**: Prefer integration tests where possible over mocking everything.

## Ultrathink Mode

When I say **"ultrathink"**, use extended thinking/deep analysis mode. This means:
- Thoroughly analyze the problem before proposing solutions
- Consider multiple approaches and their trade-offs
- Profile or benchmark when optimizing
- Run experiments in parallel using subagents when comparing alternatives

## Brainstorm Mode

When I say **"brainstorm"** or **"don't generate any code"**, just discuss ideas without writing implementation code.

## Autonomous Debugging

When I say **"run this yourself and debug issues"** or **"fix issues until it works"**:
- Execute the code
- Identify errors
- Fix them iteratively
- Continue until the workflow completes successfully

## Clarifying Questions

**Ask clarifying questions when unsure** before proceeding with implementation. Better to clarify upfront than to implement the wrong thing.

## Git Workflow

- **Git Worktrees**: Use git worktrees for feature branches when requested.
- **Clean Artifacts**: Do not commit generated files, output folders, benchmark scripts, or test data files. Add them to .gitignore instead.
- **PR Summaries**: When asked, generate summary PR messages suitable for GitHub.

## Development Approach

- **Test-Driven Development Strategy**: 
  - For each code generation session, create a structured approach:
    - Generate code and modify types
    - Create functions with stubbed return values
    - Create unit tests that check for expected values (may initially fail)
    - Implement functions progressively
    - Run unit tests to verify implementation
  - When modifying existing functions:
    - Ensure a unit test exists
    - Create a unit test if none exists
    - Modify the function
    - Run tests to verify changes
  - **Continuous TDD Commitment**: Consistently apply test-driven development principles throughout the project lifecycle
  - **Session Guideline**: Lets use TDD principles for this session

- **Parallelization & Subagent Strategy**:
  - **Maximize Parallel Execution**: When multiple independent tasks exist, execute them in parallel rather than sequentially
  - **Deploy Subagents for Complex Tasks**: Use the Task tool with specialized subagents for:
    - `Explore` agent: Codebase exploration, finding files, understanding architecture
    - `general-purpose` agent: Multi-step research tasks, complex searches
    - `Plan` agent: Designing implementation approaches for complex features
  - **Parallel Subagent Deployment**: Launch multiple subagents simultaneously when tasks are independent (e.g., searching for different patterns, exploring different parts of the codebase)
  - **When to Parallelize**:
    - Running independent tests or linting checks
    - Searching for multiple patterns or files
    - Reading multiple independent files
    - Exploring different modules or components
    - Making independent code changes across files
  - **Example Parallel Patterns**:
    - Run `pytest`, `black --check`, and `ruff check` in parallel
    - Deploy multiple Explore agents to investigate different subsystems simultaneously
    - Read multiple test files in parallel when understanding test coverage

## UI/UX Design Guidelines

### Color Palette (Tailwind-inspired)
- **Primary blue**: `#3b82f6` (buttons, focus states, hover borders)
- **Primary blue dark**: `#2563eb`, `#1d4ed8` (gradients, hover)
- **Success green**: `#22c55e` (positive labels)
- **Danger red**: `#ef4444` (negative labels)
- **Gray scale**: `#f8fafc`, `#f1f5f9`, `#e2e8f0`, `#cbd5e1`, `#94a3b8`, `#64748b`, `#374151`
- **Borders**: `rgba(0,0,0,0.06)`, `#e2e8f0`, `#d1d5db`

### ipywidgets Styling Patterns
- **CSS injection**: Use `HTML` widget with `<style>` tags for custom CSS
- **Class-based styling**: Use `widget.add_class("class-name")` for CSS targeting
- **Layout constraints**: Always set explicit `overflow: hidden` to prevent unwanted scrollbars
- **Gradients**: Use `linear-gradient()` for subtle depth on containers

### Card Design
- **Border radius**: 6-8px for cards, 12px for containers
- **Shadows**: `0 1px 3px rgba(0,0,0,0.08)` (subtle), `0 4px 20px rgba(0,0,0,0.15)` (elevated)
- **Hover states**: Lift with `transform: translateY(-2px)` and enhanced shadow
- **Border feedback**: 2px transparent border, colored on hover/active states

### Typography
- **Font sizes**: 10px (badges), 11px (small labels), 12px (body/buttons)
- **Font weights**: 500 (medium), 600 (semibold for emphasis/buttons)
- **Number formatting**: Use `:,` format specifier for thousands separators
- **Button font consistency**: All button types (action buttons, toggle buttons, etc.) must share identical `font-family`, `font-size`, `font-weight`, and `letter-spacing` properties
- **Font families**:
  - Primary: `'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
  - Use `@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600&display=swap')` for web font loading in CSS
- **Letter spacing**: 0.3px for button text (improves readability)

### Layout Principles
- **Image-first**: Show visual content before actions (natural reading order)
- **Rank indicators**: Display position (#1, #2...) for ranked results
- **Fixed headers/footers**: Keep controls accessible, only scroll content
- **Match heights**: Panel components should align with adjacent UI elements

### Alternative Layouts Considered
- **Horizontal carousel**: Better for ranked results, preserves map width, familiar pattern (Netflix/YouTube)
- **Vertical sidebar** (current): Familiar pattern, doesn't take vertical space

## Project Overview

GeoVibes is an interactive geospatial similarity search tool that lets users "vibe check" satellite foundation model embeddings through an intuitive Jupyter notebook interface. Instead of relying solely on academic benchmarks, it enables hands-on exploration to assess which embedding models suit specific use cases.

**Author**: Chris Ren (chris@demeterlabs.io)
**Status**: Experimental research code (not production-grade)

## Quick Start Commands

```bash
# Install and setup
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python -m ipykernel install --user --name geovibes --display-name "Python (geovibes)"

# Download embeddings (interactive)
uv run download_embeddings.py

# Launch notebook
uv run jupyter lab
# Open vibe_checker.ipynb, select "Python (geovibes)" kernel

# Build FAISS index from embeddings
python geovibes/database/faiss_db.py \
  --roi-file geometries/your_region.geojson \
  --mgrs-reference-file geometries/mgrs_tiles.parquet \
  --embedding-dir s3://path/to/embeddings/ \
  --name your_model_name \
  --tile-pixels 32 --tile-overlap 16 --tile-resolution 10 \
  --output_dir local_databases

# Run tests
pytest
```

## Architecture

### Core Components

```
geovibes/
├── __init__.py           # Exports GeoVibes class
├── ui/
│   ├── app.py            # Main GeoVibes class - orchestrates the UI
│   ├── state.py          # AppState dataclass - mutable UI state
│   ├── data_manager.py   # DataManager - database/FAISS/config operations
│   ├── map_manager.py    # MapManager - ipyleaflet map and layers
│   ├── tiles.py          # TilePanel - search result thumbnails
│   ├── datasets.py       # DatasetManager - save/load GeoJSON datasets
│   ├── status.py         # StatusBus - status bar messaging
│   ├── xyz.py            # XYZ tile fetching utilities
│   └── utils.py          # Helper functions
├── ui_config/
│   ├── constants.py      # UIConstants, BasemapConfig, DatabaseConstants, LayerStyles
│   └── settings.py       # GeoVibesConfig YAML loader
├── database/
│   ├── faiss_db.py       # CLI script to build FAISS index + DuckDB metadata
│   └── cloud.py          # Cloud storage utilities (S3, GCS)
├── tiling.py             # MGRS tile grid generation
└── ee_tools.py           # Earth Engine basemap helpers
```

### Data Flow

1. **User clicks map** → `_on_map_interaction()` → `label_point()` → queries DuckDB for nearest embedding
2. **Label applied** → `AppState.apply_label()` → updates `pos_ids`/`neg_ids` → `_update_query_vector()`
3. **Search clicked** → `_search_faiss()` → FAISS index search → `_process_search_results()` → map + tile panel update
4. **Query vector formula**: `2 * positive_avg - negative_avg` (in `AppState.update_query_vector()`)

### Key Classes

- **GeoVibes** (`ui/app.py`): Main entry point, orchestrates UI widgets and event handling
- **AppState** (`ui/state.py`): Holds `pos_ids`, `neg_ids`, `cached_embeddings`, `query_vector`
- **DataManager** (`ui/data_manager.py`): Manages DuckDB connections, FAISS index loading, database discovery
- **MapManager** (`ui/map_manager.py`): ipyleaflet map, basemap switching, layer management
- **TilePanel** (`ui/tiles.py`): Async tile thumbnail loading with ThreadPoolExecutor

## Database Schema

DuckDB table `geo_embeddings`:
```sql
CREATE TABLE geo_embeddings (
    id BIGINT PRIMARY KEY,
    source_id VARCHAR,           -- Optional: original tile_id from source
    embedding FLOAT[N],          -- or UTINYINT[N] for INT8 quantized
    geometry GEOMETRY            -- Point centroid of the tile
);
```

FAISS index uses IVF with:
- `IndexIVFPQ` for float embeddings (Product Quantization)
- `IndexIVFScalarQuantizer` for int8 embeddings (Scalar Quantization)

## Configuration

### config.yaml
```yaml
start_date: "2024-01-01"
end_date: "2025-01-01"
enable_ee: true  # Optional: enable Earth Engine basemaps
```

### Environment Variables
- `MAPTILER_API_KEY`: Required for MapTiler satellite basemap
- `GEOVIBES_ENABLE_EE`: Enable Earth Engine (also controllable via config)
- `GCS_ACCESS_KEY_ID` / `GCS_SECRET_ACCESS_KEY`: For GCS-hosted databases

## Key Implementation Details

### Threading Model
- TilePanel uses `concurrent.futures.ThreadPoolExecutor` (8 workers) for async tile image fetching
- Widget updates dispatch to UI thread via `asyncio.get_event_loop().call_soon_threadsafe()`
- Context variables (`contextvars.copy_context()`) preserve state across threads

### Memory Management
- DuckDB memory limit: 24GB (`DatabaseConstants.MEMORY_LIMIT`)
- Embedding fetch chunk size: 10,000 (`DatabaseConstants.EMBEDDING_CHUNK_SIZE`)
- FAISS search uses `nprobe=4096` for IVF index

### Tile Specification
- Default: 32px tiles, 16px overlap, 10m resolution (320m × 320m ground tiles)
- Inferred from database name pattern: `{name}_{pixels}_{overlap}_{resolution}`
- Stored in `DataManager.tile_spec` as `{"tile_size_px": N, "meters_per_pixel": M}`

## Testing

```bash
pytest                           # Run all tests
pytest -v --tb=short            # Verbose with short tracebacks
pytest tests/test_specific.py   # Run specific test file
```

Test directory: `tests/`

## Code Style

- Formatter: `black` (line-length 88)
- Linter: `ruff` (E, F rules)
- Type checking: `mypy` (Python 3.10+)

```bash
black geovibes/
ruff check geovibes/
mypy geovibes/
```

## Common Development Tasks

### Adding a New Basemap
1. Add URL template to `BasemapConfig.BASEMAP_TILES` in `ui_config/constants.py`
2. If Earth Engine based, add to `_setup_basemap_tiles()` in `map_manager.py`

### Adding New Database Support
1. Database discovery logic in `DataManager._discover_databases()`
2. FAISS path inference in `DataManager._infer_faiss_from_db()`
3. Manifest entries in `manifest.csv`

### Modifying Search Algorithm
- Query vector computation: `AppState.update_query_vector()` in `ui/state.py`
- FAISS search execution: `GeoVibes._search_faiss()` in `ui/app.py`
- Result processing: `GeoVibes._process_search_results()` in `ui/app.py`

## File Locations

- Main notebook: `vibe_checker.ipynb`
- Embeddings downloader: `download_embeddings.py`
- Database builder: `geovibes/database/faiss_db.py`
- Geometry files: `geometries/` (GeoJSON, Parquet)
- Downloaded databases: `local_databases/`
- Manifest: `manifest.csv`

## Supported Embedding Models

Pre-built indexes available for:
- Earth Genome SSL4EO DINO ViT (Sentinel-2)
- Google Satellite Embeddings v1 (AlphaEarth)
- Clay v1.5 NAIP embeddings

## Troubleshooting

### "No downloaded models found"
Run `uv run download_embeddings.py` to fetch databases from manifest.

### Earth Engine basemaps not loading
1. Run `earthengine authenticate`
2. Set `enable_ee: true` in config.yaml or `GEOVIBES_ENABLE_EE=1`

### Memory issues with large databases
- Reduce `DatabaseConstants.MEMORY_LIMIT`
- Use quantized (INT8) embeddings for smaller memory footprint

### Tile images not loading
- Check `MAPTILER_API_KEY` is set in `.env`
- Verify network connectivity
- Check TilePanel's `_fetch_tile_image_bytes()` for errors
