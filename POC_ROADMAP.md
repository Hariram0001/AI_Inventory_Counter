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

The gallery currently shows samples with `inventory_type = "fence_panels"` (Fence Panels). Unrelated inventory samples stay hidden or can be marked disabled.
