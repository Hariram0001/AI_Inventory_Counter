# Stakeholder Demo Script (5–7 minutes)

**Product:** AI Inventory Counter — Proof of Concept  
**Audience:** Business stakeholders  
**Goal:** Show how AI can draft a visible-object count that a person reviews and saves.

---

## 1. The problem (30–45 seconds)

Yards still count many inventory items by hand from photos or walk-throughs. That is slow, inconsistent, and hard to audit. This POC shows a practical assistive flow: photograph → AI draft count → human review → saved record.

---

## 2. Open the app (15 seconds)

Open the deployed Streamlit app (or local `app.py`).  
Landing page should show:

- **AI Inventory Counter**
- Short product sentence
- **Get Started** and **Try a Sample**
- **Shape Detection** directly under Get Started (badge: Testing Phase — local, no API key)
- Compact POC notice (results need review)

Do **not** dig into Diagnostics or Workflow IDs during this walkthrough.

Optional 30-second aside: open **Shape Detection**, load the built-in circle sample, run **Detect Circles**, and note that this path uses free local computer vision (not Roboflow/OpenRouter).

---

## 3. Fence Panel sample (primary demo) (2–3 minutes)

> Built-in **Cardboard Boxes** sample is not shipped yet. Use **Fence Panel** as the verified happy path.

1. Under **Try a Sample**, choose **Fence Panel**.
2. Confirm the wizard opens on **Photos** with the sample loaded (inventory already set).
3. Continue to **Analyze**.
4. Keep **Single Model** and **YOLO-World**.
5. Note detection terms (fence panel prompts).
6. Click **Run Analysis** (this is the paid/live call — say so briefly).
7. On **Running**, mention the clear progress phases.
8. Continue to **Review & Save**.

### What to point out

- Numbered markers on the photo
- Count is a draft, not an official inventory figure
- Confidence is guidance, not measured accuracy
- Reviewer can adjust before save

---

## 4. Adjust and save (45–60 seconds)

1. If a marker looks wrong, exclude it or note a manual adjustment.
2. Emphasize the **final count**.
3. Click **Save Result** (not “Run again”).
4. Open **Settings → Inventory History** and show the saved row.
5. Mention CSV export if useful for the audience.

---

## 5. Optional: difficult Gate sample (30–45 seconds)

Only if time remains:

1. Home → **Fence Gate** sample (labeled difficult).
2. Run YOLO-World for **Gates**.
3. If the model returns zero or a coarse structure box, treat that as an honest POC limitation — do not invent a success.

---

## 6. Compare Models (Fence Panel) (45–60 seconds)

1. Start a Fence Panel run again (sample or prior photos).
2. Choose **Compare Models**.
3. Select **YOLO-World** and **Local Picket Counter**.
4. Run comparison — explain each model runs independently.
5. Use **Use This Result** on the preferred output.
6. Confirm Review shows that selected result.

If the audience asks about Boxes or other inventories: comparison may be unavailable until a second compatible model is validated for that inventory.

---

## 7. Model Catalog (20–30 seconds)

Settings → AI Configuration → **Model Catalog**:

- YOLO-World is the generic prompt-driven model
- Workspace refresh finds trained projects when they exist
- Specialized trained models can be added later, live-tested, then enabled for comparison

---

## 8. Close with limitations (20–30 seconds)

- Visible objects only; occlusion and stacking are hard
- Prompt wording and confidence thresholds matter
- Human review is required before operational use
- This is a **proof of concept**, not a production accuracy guarantee

---

## Technical appendix (engineers only)

- YOLO-World path: Roboflow Workflow `custom-workflow` with injected `class_names`
- No silent fallback to hardcoded fence defaults when dynamic prompts are requested
- Local Picket Counter is a classical heuristic for Fence Panel only
- Streamlit Community Cloud local files may reset; durable storage is a future roadmap item
