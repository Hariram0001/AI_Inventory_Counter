# Stakeholder Demo Script (5–7 minutes)

**Product:** AI Inventory Counter — Proof of Concept  
**Audience:** Business stakeholders  
**Goal:** Show how AI can draft a visible-object count that a person reviews and saves.

---

## 1. The problem (30–45 seconds)

Yards still count many inventory items by hand from photos or walk-throughs. That is slow, inconsistent, and hard to audit. This POC shows a practical assistive flow: photograph → AI draft count → human review → saved record.

---

## 2. Open the app (15–30 seconds)

Open the deployed Streamlit app (or local `app.py`).  
**First screen is login** (not the dashboard):

- **AI Inventory Counter** brand + short product sentence
- **Sign in**
- Expandable **Create an account** / **Forgot password?** (mention briefly that admins approve these)

Sign in with a prepared demo account (or the bootstrap admin). After sign-in the home dashboard shows:

- **Get Started** and **Try a Sample**
- Compact POC notice (results need review)
- **Shape Detection** is a left-sidebar icon (Work in progress — local, no paid API key)

Do **not** dig into Diagnostics or Workflow IDs during this walkthrough.

Optional 30-second aside: open **Shape Detection** from the left panel, load a built-in sample, run detection, and note that this path uses free local computer vision (not Roboflow/OpenRouter).

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

- Default is **one item at a time**; the focused box is outlined in **red**
- Confidence % on the stepper is guidance, not measured accuracy
- Dense piles hide stacked class labels until you focus an item
- Count is a draft, not an official inventory figure
- Reviewer can **Exclude this item** before save

---

## 4. Adjust and save (45–60 seconds)

1. Step Prev/Next; if a marker looks wrong, click **Exclude this item**.
2. Emphasize the **final / included count**.
3. Click **Save Result** (not “Run again”).
4. Open **Settings → Inventory History** and show the saved row (private to this account).
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

## 7. Model Catalog / OpenRouter (20–40 seconds)

Settings → AI Configuration → **Model Catalog**:

- YOLO-World is the generic prompt-driven model
- Workspace refresh finds trained projects when they exist

Optional (admin only): mention that OpenRouter VLM uses an **administrator-managed** key on **API Keys** — users never paste a key.

---

## 8. Close with limitations (20–30 seconds)

- Visible objects only; occlusion and stacking are hard
- Prompt wording and confidence thresholds matter
- Human review is required before operational use
- Streamlit Cloud local storage may reset after redeploy
- This is a **proof of concept**, not a production accuracy guarantee

---

## Technical appendix (engineers only)

- YOLO-World path: Roboflow Workflow `custom-workflow` with injected `class_names`
- OpenRouter path: direct `openrouter_vlm.py` chat/completions (admin deployment key)
- No silent fallback to hardcoded fence defaults when dynamic prompts are requested
- Local Picket Counter is a classical heuristic for Fence Panel only
- Review: solo canvas, red focus, dense-label threshold 12, Exclude this item
- Schema v8 (auth, deployment secrets, shape detection, signup/reset)
- Streamlit Community Cloud local files may reset; durable storage is a future roadmap item
