# Built-in Sample Images

Project-owned sample photos for **Add Photos → Sample Images**.

These files are git-tracked and shipped with deployment. Runtime user uploads are **not** stored here.

## Recommended set

- About **5–15** curated images
- Prefer **JPEG** for photographs; use **PNG** only when needed
- Keep individual files reasonably small for deployment while preserving detection resolution
- Do not commit huge original camera dumps
- If the library grows large later, move it to object storage instead of the application repository

## Add a new sample

1. Copy the image into `assets/sample_images/`
2. Add metadata to `manifest.json` (see example below)
3. Start the app
4. Verify it appears under **Add Photos → Sample Images**
5. Run tests: `.\.venv\Scripts\python.exe -m pytest -q`
6. Commit **both** the image file and the manifest update

### Manifest entry example

```json
{
  "id": "fence_backyard_01",
  "filename": "fence_backyard_01.jpg",
  "title": "Backyard Wooden Fence",
  "description": "Wide view of a wooden fence in a backyard.",
  "inventory_type": "fence_panels",
  "enabled": true,
  "featured": true,
  "source": "project_sample",
  "license": "Provided for this project"
}
```

Only register files that genuinely exist. Do not invent fake samples.

## Supported formats

- `.jpg` / `.jpeg`
- `.png`

## Licensing

You are responsible for ensuring each sample may be bundled with this project.
