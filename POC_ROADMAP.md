# AI Inventory Counter — POC Roadmap Notes

## Compare Models

- **Single Model** — select one model, **Run Analysis**
- **Compare Models** — multiselect **2–3** enabled, valid, inventory-compatible peers (Roboflow workflow/model and confirmed local inference such as Local Picket Counter)
- Demo/mock models are hidden when `DEMO_MODE=false`
- Comparison runs sequentially: every photo × every selected model
- Failures are isolated; a failed model never becomes a fake `Count: 0` result
- Review uses model tabs (optional side-by-side for exactly two models) without re-running inference
- **Use This Result** chooses the model output for review/save
- Save embeds comparison metadata in `AIC_META` (compatible with older history rows)

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

## Detection Benchmark

Dynamic prompt **execution** into YOLO-World `class_names` is verified. Object-level
**detection quality** still needs human-validated benchmarks on real images.

### Location

Settings → AI Configuration → **Detection Benchmark**

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
