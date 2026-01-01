# CLAUDE.md - GeoVibes Project Guide

## Notification

- IMPORTANT: YOU MUST ALWAYS DO THIS: When you need to send me a notification because you need input or when you have finished a task, please use terminal-notifier tool like this: terminal-notifier -title "🔔 Claude Code: request" -message "Claude needs your permission to use ...", or terminal-notifier -title "✅ Claude Code: done" -message "The task has been completed"
- Always customise the message using a short summary of the input needed or the task just completed


After making changes, commit them with a useful commit message.
Use `uv run` to run with the correct env strategy
Do not leave comments unless explicitly asked to
Do not implement classes unless explicitly necessary, where possible use functions and keep things simple.
Be parsimonious with code.

## Developer Documentation (`docs/`)

**IMPORTANT**: Before implementing features, check the relevant documentation in `docs/`:

| If you're working on... | Read this doc |
|-------------------------|---------------|
| Search vs Detection mode behavior | `docs/modes-and-state.md` |
| AppState fields, mode transitions | `docs/modes-and-state.md` |
| How user clicks trigger method chains | `docs/event-flow.md` |
| Adding new event handlers | `docs/event-flow.md` |
| GeoJSON input/output formats | `docs/data-formats.md` |
| DuckDB schema, FAISS index types | `docs/data-formats.md` |
| ipyvuetify patterns, BtnToggle events | `docs/ui-widgets.md` |
| Widget hierarchy, CSS injection | `docs/ui-widgets.md` |
| MDI icons (NOT FontAwesome in side panel) | `docs/ui-widgets.md` |
| CLI tools (faiss_db.py, download_embeddings.py) | `docs/scripts.md` |
| Building new databases | `docs/scripts.md` |

### Quick Reference

- **Mode-specific behavior**: Always check `state.detection_mode` before implementing click handlers
- **BtnToggle returns index**: `v_model=0` means first option, not string value
- **Icon systems differ**: ipyvuetify uses `mdi-*`, ipywidgets uses `fa-*`
- **Query vector formula**: `2 * mean(positives) - mean(negatives)`

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

## Bug Analysis Guidelines

When analyzing code for bugs (especially when using subagents), apply rigorous verification to avoid false positives:

### Verification Checklist (MANDATORY before reporting a bug)

1. **Trace actual execution paths** - Don't flag code based on pattern matching alone. Follow the control flow:
   - What conditions must be true to reach this code?
   - What state exists when this code executes?
   - Example: `if key in dict: dict[key]` is NOT a KeyError bug - the condition guarantees the key exists

2. **Read docstrings and comments** - Check if behavior is intentional:
   - A function named `toggle_label` that removes then re-adds is a feature, not a bug
   - Docstrings like "Toggle label assignment" indicate intentional toggle behavior

3. **Understand library-specific behaviors**:
   - **DuckDB**: `execute(sql, [a, b, c])` correctly binds list elements to `?` placeholders
   - **Regex**: Non-greedy quantifiers (`\d+?`) still produce correct results due to backtracking
   - **ipyleaflet**: Layer removal patterns with `if layer in map.layers` are idiomatic

4. **Recognize intentional design patterns**:
   - **Graceful degradation**: Silent exception handling in async/background tasks is often intentional
   - **Cleanup-then-return**: Removing old state before early return is correct cleanup
   - **Toggle vs Set**: UI controls often toggle off when clicked twice - this is UX, not a bug

5. **Consider UX intent** - Some "bugs" are features:
   - Clicking the same label twice to remove it
   - Cancelling pending operations when starting new ones
   - Displaying empty state instead of errors

### Common False Positive Patterns to Avoid

| Pattern | Why It's Usually NOT a Bug |
|---------|---------------------------|
| `if key in dict: use dict[key]` | Condition guarantees key exists |
| `list.remove(x); list.append(x)` | Ensures no duplicates, moves to end |
| `connection.close(); connection = new()` | Old connection should close before reconnect |
| `try: ... except: pass` (in async) | Intentional graceful degradation |
| `str(value)` where value might be None | Creates "None" string - edge case, not crash |

### Subagent Bug Analysis Protocol

When deploying subagents for bug analysis:
1. Instruct them to apply this verification checklist
2. Require them to explain WHY something is a bug, not just WHAT looks suspicious
3. Have them categorize findings as: **Confirmed Bug**, **Potential Issue**, or **Needs Verification**
4. Always manually verify "critical" findings before acting on them

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

### ipyvuetify Side Panel (PR: ui-redesign-claude)

The side panel was redesigned using ipyvuetify for a more polished, Material Design-inspired look. Key design decisions:

#### Widget Library Strategy
- **Side panel**: Uses `ipyvuetify` (v.Btn, v.BtnToggle, v.Select, v.Card, v.Icon) for polished Material Design components
- **Tile panel**: Remains `ipywidgets` (Button, HBox, VBox) for lightweight tile cards
- **Hybrid approach**: ipywidgets can be embedded inside ipyvuetify containers (e.g., sliders inside v.Card)

#### Icon Systems (Critical)
- **ipyvuetify uses MDI icons** (Material Design Icons): `mdi-thumb-up-outline`, `mdi-magnify`, etc.
- **ipywidgets uses FontAwesome**: `icon="fa-thumbs-up"` in Button widget
- **Do NOT mix**: FontAwesome icons will not render inside ipyvuetify containers due to CSS isolation
- **Reference**: https://materialdesignicons.com/ for MDI icon names

#### Component Patterns

**Search Button (Style L)**:
```python
v.Btn(
    block=True,           # Full width
    color="primary",
    depressed=True,       # Flat style, no shadow
    children=[
        v.Icon(small=True, class_="mr-2", children=["mdi-magnify"]),
        "Search",
    ],
)
# Layout: cols=10 for search, cols=2 for tiles icon button
```

**Label Toggle (BtnToggle with icons)**:
```python
v.BtnToggle(
    v_model=0,
    mandatory=True,
    class_="d-flex",
    children=[
        v.Btn(small=True, children=[v.Icon(small=True, children=["mdi-thumb-up-outline"])]),
        v.Btn(small=True, children=[v.Icon(small=True, children=["mdi-thumb-down-outline"])]),
        v.Btn(small=True, children=[v.Icon(small=True, children=["mdi-eraser"])]),
    ],
)
# Event: observe(handler, names="v_model") - returns index
```

**Mode Toggle (BtnToggle with text)**:
```python
v.BtnToggle(
    v_model=0,
    mandatory=True,
    children=[
        v.Btn(small=True, children=["• Point"]),
        v.Btn(small=True, children=["▢ Polygon"]),
    ],
)
```

**Dropdown Select**:
```python
v.Select(
    v_model="value",
    items=["Option1", "Option2"],
    dense=True,
    hide_details=True,
)
# Event: observe(handler, names="v_model") - returns selected value
```

**Section Cards**:
```python
v.Card(
    outlined=True,
    class_="section-card pa-3",
    children=[
        v.Html(tag="span", class_="section-label", children=["LABEL"]),
        # ... widgets
    ],
)
```

#### Event Handling Differences
| Widget Type | Event Method | Callback Pattern |
|-------------|--------------|------------------|
| `v.Btn` | `on_event("click", handler)` | `lambda *args: ...` |
| `v.BtnToggle` | `observe(handler, names="v_model")` | `change["new"]` = index |
| `v.Select` | `observe(handler, names="v_model")` | `change["new"]` = value |
| ipywidgets Button | `on_click(handler)` | `lambda btn: ...` |

#### CSS Customization
Custom CSS is injected via `HTML` widget with `<style>` tags. Key classes:
- `.geovibes-panel` - Root container class
- `.section-label` - Uppercase gray section headers (10px, #64748b)
- `.search-btn` - Prominent search button (40px height, 14px font)
- `.v-btn` overrides - Remove text-transform, set letter-spacing

#### Design Rationale
1. **Expansion panels → Cards**: Replaced v.ExpansionPanels with always-visible v.Card sections for immediate access
2. **Basemap expansion → Dropdown**: Changed from expansion panel to v.Select for compact selection
3. **Icon-only label buttons**: More compact, relies on tooltips for clarity
4. **Full-width search button**: More prominent call-to-action (Style L from comparison)
5. **Depressed button style**: Flat appearance without shadows for cleaner look

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

## UI Preview Workflow (Visual Testing)

For iterating on UI changes, use Voila + Playwright to preview and capture screenshots.

**Tools location:** `claude_ui_dev/`
- `test_tile_panel.ipynb` - Test notebook that programmatically triggers UI state
- `screenshot_ui.py` - Playwright-based headless screenshot capture
- `preview_ui.sh` - Convenience wrapper script

### Quick Start

```bash
# Install playwright (one-time)
uv add playwright && uv run playwright install chromium

# Start Voila in background (port 8866)
cd claude_ui_dev
uv run voila test_tile_panel.ipynb --port=8866 --no-browser &

# Take screenshot after UI loads
./preview_ui.sh my_screenshot.png 10

# Or with full control
uv run python screenshot_ui.py --output tile_panel_v2.png --wait 15 --width 1600 --height 900

# Kill Voila when done
pkill -f "voila test_tile_panel"
```

### Iterative Design Loop

1. Make UI changes in `geovibes/ui/*.py`
2. Restart Voila (or use `--autoreload=True`)
3. Take screenshot: `./preview_ui.sh iteration_N.png 10`
4. Review screenshot, iterate
5. Compare versions side-by-side

### Test Notebook Customization

Edit `claude_ui_dev/test_tile_panel.ipynb` to:
- Change coordinates to match your database coverage
- Trigger specific UI states (detection mode, different basemaps, etc.)
- Test specific workflows programmatically

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
