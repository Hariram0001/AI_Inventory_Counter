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
| Shape Detection (Testing Phase) | Passed | Offline pytest + local OpenCV | Home button under Get Started; circles only; 0 paid API calls |

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

## Authentication and roles (auth rework)

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| App opens on a login screen | Passed | AppTest | `test_app_opens_on_login_screen`; no dashboard controls before sign-in |
| No default or hardcoded account | Passed | Offline pytest | Bootstrap only while `users` is empty; missing config warns instead of creating |
| First-admin bootstrap | Passed | Offline pytest + AppTest | Creates one admin, usable immediately; never re-runs |
| Argon2id hashing, no plaintext | Passed | Offline pytest | Hash prefix asserted; temporary password absent from the database file |
| Password policy | Disabled | Offline pytest | Complexity intentionally removed; only blank is refused |
| Generic login failure | Passed | AppTest | Unknown user, wrong password and disabled account are indistinguishable |
| Lockout after 5 failures, 15 min | Passed | Offline pytest + AppTest | Admin unlock clears it |
| Forced password change blocks the app | Passed | AppTest | Rejects reuse, weak values and a wrong current password |
| Idle and absolute session timeouts | Passed | Offline pytest + AppTest | Returns to login with an inactivity notice |
| Session-version invalidation | Passed | Offline pytest | Password change and admin reset end other sessions |
| Deactivation revokes a live session | Passed | AppTest | Next rerun signs the user out |
| Logout cleanup | Passed | AppTest | Identity, BYOK key, wizard, benchmark, admin and transient state cleared |
| Final-admin protection | Passed | Offline pytest + AppTest | Role, deactivate and delete disabled and refused in the store |

---

## Authorization and data ownership

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Regular user denied the admin console | Passed | AppTest | Hidden in navigation and refused when forced; `authz.denied` audited |
| Admin permitted | Passed | AppTest | All seven console tabs render |
| History scoped per user | Passed | Offline pytest | Query filters by `user_id` for every account; legacy/unowned rows are never shared |
| New records attributed | Passed | Offline pytest | `user_id` and `username` stored on save |

---

## User management and audit

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Create user with one-time password | Passed | AppTest | Shown once, verifies, absent from the database file |
| Duplicate username and email rejected | Passed | Offline pytest + AppTest | Explained, not silently overwritten |
| Admin reset forces change | Passed | AppTest | Old password stops working; `session_version` incremented |
| Disable / enable | Passed | AppTest | Round-trip verified |
| Role update | Passed | Offline pytest | Audited as `user.updated` |
| Audit events recorded | Passed | AppTest | Bootstrap, sign-in and password change present |
| No secrets in the audit log | Passed | Offline pytest | Recursive redaction of details |

---

## OpenRouter BYOK and cost control

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Key format validated before any call | Passed | Offline pytest | Non-OpenRouter values rejected locally |
| Verification uses `GET /api/v1/key` | Passed | Offline pytest | No model call; free metadata read |
| Invalid, forbidden, rate-limited, timeout, malformed handled | Passed | Offline pytest | Each maps to a distinct actionable message |
| Session-only storage | Passed | AppTest | Key absent from every `.db` / `.json` under `DATA_DIR` |
| Logout clears the key and cost notice | Passed | AppTest | Cleared with the rest of the session |
| Key redacted from nested diagnostics and errors | Passed | Offline pytest | Recursive redaction, including bare `Bearer` tokens |
| Cost notice required before the first run | Passed | AppTest | Acknowledgement recorded as `cost.acknowledged` |
| Model gated on all availability conditions | Passed | Offline pytest | Role, policy, key, consent, metadata, inventory, quota |
| Daily per-user quota | Passed | Offline pytest | Blocks at the limit; `quota.blocked` audited |

---

## Workflow adapter (OpenRouter VLM Detector)

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| `image`, `classes`, `model_api_key` inputs | Passed | Offline pytest | Parameters asserted on the workflow call |
| `predictions` and `error_status` outputs | Passed | Offline pytest | Errors extracted and translated for users |
| Visualization-only / text-only rejected | Passed | Offline pytest | Not converted into a count |
| Malformed coordinates rejected | Passed | Offline pytest | Normalization guards unchanged |
| YOLO-World and Local Picket unchanged | Passed | Offline pytest | Existing suites still pass |

---

## Manual acceptance sequence

Run against a fresh `DATA_DIR` before a demo. Each step should behave as
described; anything else is a failure worth investigating.

1. Start with no database. The app opens on the login screen and names the
   bootstrap administrator it created.
2. Remove the bootstrap variables, restart, delete the database: the login page
   explains which variables to set and creates nothing.
3. Sign in as the administrator with the bootstrap password → the dashboard
   loads directly, with no forced password change.
4. Sign in with a wrong password five times → the account locks for 15 minutes
   with a generic message throughout.
5. As administrator, unlock the account, then create a regular user and copy the
   one-time password.
6. Sign in as the regular user → forced change → dashboard. Confirm there is no
   administrator console entry.
7. Run a YOLO-World analysis on a sample, review, save. Open Inventory History
   and confirm only that user's record is listed.
8. Sign back in as the administrator and confirm History shows only the
   administrator's own saves — not the other user's.
9. On **API Keys**, verify an OpenRouter key. Confirm only a masked form is
   shown, the cost notice must be acknowledged, and the OpenRouter VLM Detector
   becomes selectable.
10. Run one OpenRouter analysis. Confirm detections are reviewable and saveable
    like any other model.
11. Sign out and back in. Confirm the key is gone and the model is unavailable
    until it is verified again.
12. In **Model Access**, set the OpenRouter daily quota to 1, run once as a
    user, and confirm the second attempt is blocked with a clear reason.
13. Deactivate the regular user while they are signed in. Their next
    interaction returns them to the login screen.
14. In **Audit Log**, confirm the sign-ins, password changes, key verification,
    policy update and quota block are present, and that no key or password
    appears in any detail.
15. Leave a session idle past the timeout and confirm it returns to login with
    an inactivity notice.

---

## Deferred (intentional)

| Item | Status | Reason |
|------|--------|--------|
| Built-in Cardboard Boxes sample | Deferred | No verified box image in repo; do not fabricate assets |
| Persistent cloud database | Deferred | Roadmap only (Supabase/PostgreSQL/object storage) |
| Additional trained workspace OD models | Deferred | Workspace currently has 0 trained OD projects |
| Production accuracy claims | Deferred | POC estimates require human review |
| SSO / OIDC and MFA | Deferred | Provider interface exists; local passwords only for the POC |
| Encryption at rest and tamper-evident audit | Deferred | Requires managed storage; documented in `SECURITY_MODEL.md` |
| Spend estimation and reconciliation | Deferred | Quotas and a cost notice only |
