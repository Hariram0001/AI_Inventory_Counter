# AI Inventory Counter — POC Roadmap

Status markers: **Complete** · **In progress** · **Blocked** · **Deferred** · **Planned**

Deep detail: [`docs/COMPLETE_APP_OVERVIEW.md`](docs/COMPLETE_APP_OVERVIEW.md) ·
[`docs/FEATURE_MATRIX.md`](docs/FEATURE_MATRIX.md)

---

## Completed POC milestones

| Item | Status |
|------|--------|
| Streamlit wizard (Setup → Photos → Analyze → Running → Review) | Complete |
| Login-first auth, Argon2id, lockout, sessions | Complete |
| Self-signup pending + admin password-reset queue | Complete |
| Admin console (8 tabs) + audit log | Complete |
| Per-user inventory history + CSV | Complete |
| Dynamic YOLO-World prompts (no silent fence fallback) | Complete |
| Model Catalog (Foundation / Workspace / Public) | Complete |
| Single + Compare Models | Complete |
| Detection Benchmark (single + batch) | Complete |
| Review solo focus, red outline, Exclude, dense labels | Complete |
| Custom multi-type **one scan** + Review type focus | Complete |
| Preset synonyms as internal detection terms | Complete |
| Admin-managed OpenRouter key + direct VLM adapter | Complete |
| Local Picket Counter | Complete |
| Demo Mode mocks | Complete |
| Shape Detection local OpenCV (sidebar WIP, all signed-in) | Complete (experimental) |
| Home landing for all roles | Complete |
| Documentation pack (Overview, Index, Matrix, Troubleshooting) | Complete |

---

## Model Catalog

- Tabs: **Foundation Models** · **My Workspace** · **Public Models** — Complete
- YOLO-World Ready when live-validated — Complete
- Workspace Refresh → Metadata only until live Test — Complete
- Fixed-class mapping; Custom Item on dynamic models — Complete

## Compare Models

- 2–3 peers, independent runs, Use This Result — Complete
- Demo fixtures hidden when `DEMO_MODE=false` — Complete

## Built-in Sample Images

- `assets/sample_images/` + manifest — Complete
- How to add samples — see `assets/sample_images/README.md`

## Shape Detection (Work in progress)

- Sidebar icon; open to all signed-in users — Complete
- Multi-shape registry — Complete
- Accuracy / false positives — **In progress** (experimental)
- Details: [`docs/SHAPE_DETECTION_TESTING.md`](docs/SHAPE_DETECTION_TESTING.md)

## Review UX

- Solo / red / dense / Exclude / one-scan multi-type — Complete

## OpenRouter (admin-managed)

- Deployment secret + direct VLM — Complete
- Model Access seed disabled until admin enables — Complete
- Filename `OPENROUTER_BYOK.md` historical — documented

## Persistence

| Item | Status |
|------|--------|
| Durable Cloud DB / object storage | Deferred |
| Ephemeral Cloud warning | Complete (documented) |

## Detection Benchmark

- Single + batch + promotion — Complete
- Image-specific metrics warning — Complete

## Productionization (out of POC scope)

| Item | Status |
|------|--------|
| PostgreSQL + backups | Planned |
| Object storage for samples/uploads | Planned |
| OIDC/SSO + MFA | Planned |
| Tamper-evident audit sink | Planned |
| Spend reconciliation | Planned |
| Mobile-first capture | Planned |

---

## How to validate a new inventory type

1. Add/select profile
2. Representative images + expected counts
3. Benchmark prompt sets
4. Inspect boxes; record FP/FN
5. Promote prompts only after several images
