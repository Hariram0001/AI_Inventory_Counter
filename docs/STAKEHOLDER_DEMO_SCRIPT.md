# Stakeholder Demo Script (7–10 minutes)

**Product:** AI Inventory Counter — Proof of Concept
**Audience:** Business stakeholders
**Goal:** Show assistive visible-object counting with mandatory human review.
**Reliable path:** YOLO-World + Fence Panel sample (avoid OpenRouter as the main demo unless already configured).

---

## 1. Problem (30–45 s)

Yards still count many items by hand from photos. This POC drafts a count from
a photo, lets a person review markers, then saves a private record.

---

## 2. Open the app (30 s)

1. Open the Streamlit URL.
2. **Login screen first** — Sign in (mention Create an account / Forgot password need admin approval).
3. After login: **Home** (admins land here too — Administration is a sidebar icon).
4. Point out: **Get Started**, left icons (History, AI Config, Shape Detection WIP, and Admin if applicable).

---

## 3. Fence Panel happy path (3–4 min)

1. **Get Started** → choose **Fence Panel** (or Add Photos → Sample Images).
2. Continue to **Analyze** → **YOLO-World** → **Run Analysis** (paid/live — say so).
3. Progress phases → **Review & Save**.
4. Call out: one-at-a-time focus, **red** outline, confidence %, **Exclude this item**.
5. Save → **Inventory History** (private to this account).

Fallback if live API down: set `DEMO_MODE=true` for UI-only walkthrough, or use Local Picket on Fence Panel.

---

## 4. Optional Compare (45–60 s)

Fence Panel → Compare YOLO-World + Local Picket → **Use This Result**.

---

## 5. Admin / OpenRouter explanation (45 s) — do not depend on a live Luna run

- Sidebar **Administration**: users, pending signups, samples, model access, audit.
- **API Keys**: admin-managed OpenRouter key for the whole deployment; users never paste a key.
- Model Access must enable OpenRouter before anyone can run it.

---

## 6. Shape Detection aside (30 s, optional)

Sidebar **Shape Detection · Work in progress** — local OpenCV, no paid API. Label it experimental.

---

## 7. Close with limitations (30 s)

- Visible objects only; stacking/occlusion hard
- Confidence ≠ measured accuracy
- Cloud storage can reset after redeploy
- **Proof of concept**, not a production guarantee

---

## Technical appendix (engineers)

- YOLO-World: Roboflow Workflow + injected `class_names`
- OpenRouter: direct VLM (`openrouter_vlm.py`), admin `deployment_secrets`
- Multi Custom Item: one scan; Review type focus
- Schema v8; health `/_stcore/health`
