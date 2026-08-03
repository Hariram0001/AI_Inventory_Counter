# POC Acceptance Checklist

Proof of Concept — AI Inventory Counter
Documentation refresh: starting `ebd5358`, ending `f462ce2` on `main`.
Last checklist revision date: 2026-08-03
Offline tests: **463 passed** (`pytest -q`)

Status: **Passed** · **Failed** · **Deferred** · **Disabled** · **Config required**

Verification: prefer offline pytest + AppTest; paid live calls are manual.

---

## Core flow

| Item | Status | Method | Evidence |
|------|--------|--------|----------|
| Preset inventories | Passed | Profiles + UI | All presets enabled in `inventory_profiles.json` |
| Custom Item one scan | Passed | Offline tests | `test_per_type_analysis.py` |
| Upload / camera | Passed | Code + tests | Photos stage |
| Built-in samples | Passed | Manifest on disk | Fence Panel / Gate samples |
| Dynamic YOLO-World prompts | Passed | Prior live + offline guards | No silent fence fallback |
| Review solo / red / Exclude | Passed | Offline viz tests | `test_review_visualization.py` |
| Save + History | Passed | DB + AppTest | Per-user `inventory_counts` |
| Shape Detection WIP | Passed | Offline pytest | Sidebar; open to signed-in users |

---

## Models

| Item | Status | Method | Evidence |
|------|--------|--------|----------|
| YOLO-World Ready path | Passed | Catalog + prior live | Adapter `yolo_world_workflow` |
| Compare with two peers | Passed | Fence Panel | YOLO-World + Local Picket |
| Use This Result | Passed | Review contracts | No auto highest-count |
| OpenRouter direct VLM | Config required | Offline VLM tests | Needs admin key + Model Access enable |
| Paid OpenRouter live demo | Deferred | Manual | Do not mark Passed without live evidence |

---

## Authentication

| Item | Status | Method | Evidence |
|------|--------|--------|----------|
| Login-first | Passed | AppTest | `test_app_opens_on_login_screen` |
| Bootstrap admin | Passed | Offline + AppTest | Empty users only |
| Self-signup pending | Passed | `test_auth_signup_reset.py` | Approve/Reject |
| Password reset request | Passed | Same | Admin authorize |
| Argon2id | Passed | Offline | Hash prefix |
| Password complexity | Disabled | Design | Any non-empty |
| Lockout 5 / 15 min | Passed | Offline + AppTest | Unlock |
| Forced change | Passed | AppTest | Admin-created users |
| Home landing (all roles) | Passed | AppTest | `app_view == welcome` |
| Session invalidation | Passed | Offline | `session_version` |
| Logout cleanup | Passed | AppTest | Wizard cleared |

---

## Authorization / admin

| Item | Status | Method | Evidence |
|------|--------|--------|----------|
| User denied admin / API Keys | Passed | AppTest | `authz.denied` |
| Eight console tabs | Passed | AppTest | After `nav_admin` |
| History isolation | Passed | Offline | `user_id` filter |
| Admin OpenRouter key | Passed | Offline | `deployment_secrets` |
| Quotas | Passed | Offline | `policy.quota.blocked` |

---

## Deployment / secrets

| Item | Status | Method | Evidence |
|------|--------|--------|----------|
| Schema migrations v8 | Passed | Offline | `database.py` |
| Bootstrap warning copy | Passed | AppTest | Missing secrets |
| Ephemeral Cloud warning | Passed | Docs + Storage tab | Documented |
| Secrets not in git | Passed | `.gitignore` + scan | No `.env` committed |

---

## Manual sequence (fresh DATA_DIR)

1. Empty DB + bootstrap secrets → login names admin.
2. Sign in → **Home** (not Admin).
3. Sidebar Administration → Overview.
4. Create user / approve signup / authorize reset as needed.
5. Fence Panel sample → YOLO-World → Review Exclude → Save → own History.
6. Optional: API Keys verify OpenRouter → Model Access enable → one run.
7. Shape Detection from sidebar (WIP).
8. Confirm audit events without secrets in details.

---

## Deferred

| Item | Reason |
|------|--------|
| Built-in Boxes sample | No verified asset |
| Durable Cloud DB | Roadmap |
| SSO/MFA | Roadmap |
| Per-user BYOK | Superseded by admin key |
| Production accuracy claims | POC only |
