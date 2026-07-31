# POC Acceptance Checklist

Proof of Concept — AI Inventory Counter  
Date of verification: 2026-07-31  
Base commit before this release pass: `14a8d43`

Status values: **Passed** · **Failed** · **Deferred**

---

## Core flow

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Preset inventory | Passed | Offline profiles + UI contract | `inventory_profiles.json` enabled presets; Setup stage selectable |
| Custom Item | Passed | Offline prompt tests + live Custom Item run | Prompt validation rejects HTML; dynamic YOLO-World prompts |
| Image upload | Passed | Code path + upload validator | `validate_upload` in image processing; Photos stage uploader |
| Sample image | Passed | Home Try a Sample + sample library | Fence Panel + Fence Gate verified on disk |
| Camera | Passed | Code inspection | Photos stage `st.camera_input` present |
| Dynamic YOLO-World prompts | Passed | Live inference | Injection VERIFIED; `published_specification_with_prompt`; no fallback |
| Detections | Passed | Live Fence Panel run | Successful boxes returned when model finds objects |
| Numbered markers | Passed | Code + prior review tests | Review viz styles: Numbered / Boxes / Both |
| Manual review | Passed | Code inspection + prior tests | Exclude/include, adjustments, final count |
| Save | Passed | History CSV path + DB insert | SQLite `inventory_counts` |
| History | Passed | Settings → Inventory History | Opens without mutating wizard; CSV export exists |

---

## Models

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| YOLO-World Ready | Passed | Catalog + live probe | Foundation Ready; adapter `yolo_world_workflow` |
| Model compatibility filtering | Passed | Offline pytest | Custom Item → YOLO-World only; Boxes excludes Local Picket |
| Compare Models with two compatible models | Passed | Fence Panel selectable peers | YOLO-World + Local Picket Counter |
| Comparison unavailable with one model | Passed | Boxes inventory + UI message | Exact stakeholder message shown |
| Partial comparison failure | Passed | Offline compare helpers | Failures show `—`, not fake zero |
| Use This Result | Passed | Review stage contract tests | Selects accepted result without re-inference |

---

## Benchmark

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Single-image benchmark | Passed | Prior phase + Settings tab | Detection Benchmark UI |
| Batch benchmark | Passed | Prior phase tests | Batch mode + threshold sweep |
| Threshold sweep | Passed | Prior phase tests | Offline + live fence validation previously |
| Prompt comparison | Passed | Prior phase | Up to 3 prompt sets |
| Export | Passed | Prior phase | CSV/JSON session export |

---

## Reliability

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| No secrets exposed | Passed | Sanitizer tests + git review | `.env` ignored; catalog never stores keys |
| No silent dynamic-prompt fallback | Passed | Live YOLO-World | `FALLBACK False`; fail-closed injection |
| Zero detections ≠ failure | Passed | UI + helpers | Distinct empty-state copy |
| Old history opens | Passed | History label helper | Deleted models show stored metadata |
| Cloud startup healthy | Passed | Streamlit `/_stcore/health` | Returns `ok` |

---

## Demo experience (this pass)

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Landing page clarity | Passed | Manual UI review | Get Started + Try a Sample; no Workflow IDs on home |
| Sample does not auto-run | Passed | Offline test + code | Navigates to Photos only |
| Progress phases | Passed | Code inspection | Preparing → prompts → model → detections → review |
| Connection test isolation | Passed | Offline test | Wizard uploads/results restored after probe |
| Empty states | Passed | Code inspection | Photos, models, compare, history, benchmark, workspace |

---

## Deferred (intentional)

| Item | Status | Reason |
|------|--------|--------|
| Built-in Cardboard Boxes sample | Deferred | No verified box image in repo; do not fabricate assets |
| Persistent cloud database | Deferred | Roadmap only (Supabase/PostgreSQL/object storage) |
| Additional trained workspace OD models | Deferred | Workspace currently has 0 trained OD projects |
| Production accuracy claims | Deferred | POC estimates require human review |
