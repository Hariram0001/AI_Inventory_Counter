# Shape Detection (Testing Phase)

Local, free circle detection for the AI Inventory Counter POC.

## Where to find it

1. Sign in (administrators or regular users).
2. Open **Home**.
3. Immediately below **Get Started**, choose **Shape Detection**.
4. Caption: **Testing Phase · Local computer vision · No API key required**.

The button does not appear on the login page or inside Admin Console navigation. Administrators control availability under **Admin Console → Experimental Features**.

## What it is

- Detects **likely visible circular shapes and circular objects** using OpenCV on the server/CPU.
- **Does not** use AI foundation models, OpenRouter, Roboflow, YOLO-World, or paid APIs.
- Review results before using the final count.

Recommended wording in the product:

> Detect likely visible circular shapes and circular objects. Review the results before using the final count.

## Supported shapes (this release)

| Status | Shapes |
|--------|--------|
| Enabled | Circles (aliases below) |
| Coming soon | Rectangle, Square, Triangle, Polygon, Line, Ellipse |

Accepted circle terms (case-insensitive):

- circle, circles
- circular object, circular objects
- round object, round objects

Unsupported shapes show:

> Only circle detection is available during the current testing phase.

Circle detection is **not** run silently for unsupported shapes.

## Image input

- Upload Image (JPG / JPEG / PNG / WEBP)
- Camera
- Built-in Test Sample (synthetic circles generated in code)

Uploads are validated by decoding (not by extension alone). Images are not permanently stored unless you **Save Shape Test**.

## Target types

- **Circular objects** — coins, wheels, plates, bottle caps, physical round items
- **Drawn or outlined circles** — diagrams, rings, printed outlines
- **Both** (default) — combine strategies and deduplicate overlaps

## Modes

| Mode | Behavior |
|------|----------|
| Strict | Fewer detections, fewer false positives |
| Balanced | Default for most test images |
| Sensitive | More candidates; may increase false positives |

Detailed OpenCV-style controls live under **Advanced Settings**, with **Reset to Balanced Defaults**.

## Size, partial, and concentric circles

- Minimum / maximum diameter: **Auto** (default) or **Custom** (% of shortest side).
- **Include partially visible circles at image edges** — default On; marked **Partial**.
- **Count concentric circles separately** — default Off (inner/outer ring ≈ one object).

## Review and final count

1. Inspect numbered annotations.
2. Use the review table (include/exclude, review status).
3. Optionally add **Additional missed circles**.
4. Final count = included detections + manual additions.

Shape quality scores are geometric (circularity / method agreement), **not** model confidence.

## Exports and history

- Annotated PNG, CSV, JSON (no passwords, API keys, or private paths).
- **Shape Test History** is separate from Inventory History.
- Regular users see only their own runs; administrators can view all and filter by user.
- Opening a saved result does **not** rerun detection.

## Administrator policy

**Admin Console → Experimental Features → Shape Detection**

- Enabled for administrators
- Enabled for regular users
- Maximum image size
- Save history enabled
- Notes

No model-access quota counters apply (detection is local).

## Limitations

- Does not guarantee detection of every circle.
- Shadows, reflections, and curved textures may create false positives.
- Occluded or distorted circles may be missed.
- Strong perspective can make a circle appear elliptical.
- Future versions may add rectangles, triangles, polygons, and ellipses.

## API usage

During Shape Detection:

- Roboflow requests: **0**
- OpenRouter requests: **0**
- Paid inference: **0**
