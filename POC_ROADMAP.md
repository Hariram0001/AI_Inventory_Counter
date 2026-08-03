# AI Inventory Counter — POC Roadmap Notes

## Model Catalog

- Tabs: **Foundation Models** · **My Workspace** · **Public Models**
- YOLO-World is the primary generic prompt-driven Foundation model (Ready when live-validated)
- Workspace Refresh discovers trained object-detection versions; entries start as **Metadata only**
- Public models are registered by explicit model ID (no bulk Universe import)
- Fixed-class models require trained-class → inventory mapping; Custom Item stays on dynamic models
- Live Test stamps Ready; discovery alone never enables analysis selection
- Diagnostics shows sync counts without duplicating full model cards

## Compare Models

- **Single Model** — select one compatible validated model, **Run Analysis**
- **Compare Models** — multiselect **2–3** enabled, **live-validated**, inventory-compatible peers
- Peers may include Roboflow workflow/model adapters and confirmed local inference (Local Picket Counter for Fence Panel)
- Demo/mock models are hidden when `DEMO_MODE=false`
- If only YOLO-World is available for an inventory, Compare stays correctly unavailable
- Comparison runs each model independently on the same original image bytes
- Failures are isolated; a failed model never becomes a fake `Count: 0` result
- Review uses model tabs (optional side-by-side for exactly two models) without re-running inference
- **Use This Result** chooses the model output for review/save (no silent highest-count pick)
- Save embeds comparison metadata in `AIC_META` (compatible with older history rows)

### How to add a second model

1. Train a workspace object-detection model or select a suitable public model.
2. Refresh Workspace or register the model ID.
3. Inspect its classes.
4. Map compatible inventories.
5. Run a live test.
6. Enable the model.
7. Open Compare Models.

## Built-in Sample Images

Images are stored under `assets/sample_images/`.

Metadata is stored in `assets/sample_images/manifest.json`.

Git-tracked files in that folder are included in deployment (see `.gitignore` exceptions). Runtime user uploads are **not** permanently saved by this feature.

### How to add a new sample image

1. Copy the image into `assets/sample_images/`
2. Add its metadata to `manifest.json`
3. Start the app
4. Verify it appears under **Add Photos → Sample Images**
5. Run tests: `.\.venv\Scripts\python.exe -m pytest -q`
6. Commit **both** the image and the manifest update

### Supported formats

- JPEG (`.jpg` / `.jpeg`) — preferred for photographs
- PNG (`.png`) — only when necessary

### Recommended sizes

- About **5–15** built-in samples for the POC
- Reasonably compressed; keep enough resolution for object detection
- Do not repeatedly recompress during inference
- If the library becomes large later, move it to object storage rather than the application repository

### Licensing

You are responsible for ensuring each sample may be bundled with this project.

### Inventory compatibility

The gallery filters by `inventory_type` / optional `benchmark.inventory_key`. Fence Panel samples use `fence_panels`; Gates samples use `gates`. Unrelated inventory samples stay hidden unless selected in Detection Benchmark (which lists all enabled samples).

## Shape Detection (Work in progress)

- Left sidebar icon (Work in progress) — open to all signed-in users; not on Home
- Does not alter the inventory wizard
- Local OpenCV only (Hough + contours); no Roboflow / OpenRouter / YOLO
- Multi-shape registry enabled (circle, rectangle, square, triangle, polygon, line, ellipse)
- Per-user shape-test history (`shape_detection_runs`); Experimental Features still covers save/history limits
- Details: [`docs/SHAPE_DETECTION_TESTING.md`](docs/SHAPE_DETECTION_TESTING.md)

## Review UX (current)

- Solo canvas by default; opt-in “Show all markers on the picture”
- Focused detection outlined in red; confidence on Prev/Next stepper
- Dense scenes (≥12 detections) hide stacked class chips except the focused item
- **Exclude this item** / **Include this item** advances the navigator
- Custom Item multi-type: one scan for all types; Review type focus / navigation

## OpenRouter (admin-managed)

- Single deployment key in SQLite `deployment_secrets` (Admin → API Keys)
- Direct VLM via `openrouter_vlm.py` (not Roboflow Workflow injection)
- Model Access seeds OpenRouter **disabled** until an administrator enables it
- Details: [`docs/OPENROUTER_BYOK.md`](docs/OPENROUTER_BYOK.md)

## Persistence (future)

Streamlit Community Cloud local files are ephemeral. A later phase should add durable storage (for example Supabase, PostgreSQL, or cloud object storage). No provider is selected or integrated in the current POC.

## Detection Benchmark

Dynamic prompt **execution** into YOLO-World `class_names` is verified. Object-level
**detection quality** still needs human-validated benchmarks on real images.

### Location

Settings → AI Configuration → **Detection Benchmark**

Modes: **Single Image** (default) and **Batch Benchmark** (multi-image + threshold sweep).

Isolated from the inventory wizard (does not change uploads, run context, or analysis results).

### What it measures (per image)

- Count difference vs expected ground truth  
- True positives / false positives / false negatives (after visual review)  
- Precision, recall, count accuracy for **that image only**  

Do not treat a single-image score as universal model accuracy.

### Prompt wording

Good prompts name the **individual countable object**. Ambiguous scene terms
(`fence`, `road equipment`) may yield one structure-level box. YOLO-World may
still need custom training for specialized inventory.

### Storage

Results append to `data/benchmarks.json` (gitignored). Streamlit Community Cloud
filesystem is ephemeral — history may reset unless external storage is added.

Profile promotion writes `inventory_profiles.json` atomically after backing up to
`data/inventory_profile_backups/` and `inventory_profiles.backup.json`.

### How to validate a new inventory type

1. Add or select an inventory profile  
2. Upload a representative image  
3. Enter expected count  
4. Test up to three prompt sets  
5. Inspect numbered boxes  
6. Record false positives and missed objects  
7. Save benchmark  
8. Promote the best prompt set only after several images  

### Recommended objects to test

Fence Panels, Traffic Cones, Chairs, Boxes, Pallets, Cars, Bottles, Gates, Poles, Custom Item.

Add real images and verified `benchmark` metadata before claiming accuracy.
