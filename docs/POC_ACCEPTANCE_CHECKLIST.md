# POC Acceptance Checklist

Proof of Concept — AI Inventory Counter  
Date of verification: 2026-08-02  
Status values: **Passed** · **Failed** · **Deferred** · **Disabled**

Re-run the offline suites and the manual sequence below before a stakeholder
demo. Prefer `.\.venv\Scripts\python.exe -m pytest -q` against a fresh
`DATA_DIR`.

---

## Core flow

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Preset inventory | Passed | Offline profiles + UI contract | Enabled presets in `inventory_profiles.json` (Fence Panel, Boxes, Gates, …) |
| Custom Item | Passed | Offline prompt + per-type tests | Multi-type / separate analysis passes; synonym canonicalize |
| Image upload | Passed | Code path + upload validator | `validate_upload`; Photos stage uploader |
| Sample image | Passed | Home Try a Sample + sample library | Fence Panel + Fence Gate verified on disk |
| Camera | Passed | Code inspection | Photos stage `st.camera_input` |
| Dynamic YOLO-World prompts | Passed | Live inference | Injection VERIFIED; no silent fence fallback |
| Detections | Passed | Live Fence Panel run | Boxes returned when model finds objects |
| Numbered markers / solo focus | Passed | Review viz tests | Solo default; red selected outline; dense labels ≥12 |
| Manual review / exclude | Passed | Code + navigation tests | **Exclude this item** on stepper; filters; final count |
| Save | Passed | History CSV path + DB insert | SQLite `inventory_counts` |
| History | Passed | Settings → Inventory History | Per-user only; CSV export |
| Shape Detection (Testing Phase) | Passed | Offline pytest + local OpenCV | Home under Get Started; multi-shape registry; 0 paid API calls |

---

## Models

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| YOLO-World Ready | Passed | Catalog + live probe | Foundation Ready; adapter `yolo_world_workflow` |
| Model compatibility filtering | Passed | Offline pytest | Custom Item → dynamic peers; Boxes excludes Local Picket |
| Compare Models with two compatible models | Passed | Fence Panel selectable peers | YOLO-World + Local Picket Counter |
| Comparison unavailable with one model | Passed | Boxes inventory + UI message | Exact stakeholder message shown |
| Partial comparison failure | Passed | Offline compare helpers | Failures show `—`, not fake zero |
| Use This Result | Passed | Review stage contract tests | Selects accepted result without re-inference |
| OpenRouter direct VLM | Passed | Offline VLM + catalog tests | `openrouter_vlm_detector`; chat/completions path |

---

## Benchmark

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Single-image benchmark | Passed | Settings tab | Detection Benchmark UI |
| Batch benchmark | Passed | Prior phase tests | Batch mode + threshold sweep |
| Threshold sweep | Passed | Prior phase tests | Offline + live fence validation previously |
| Prompt comparison | Passed | Prior phase | Up to 3 prompt sets |
| Export | Passed | Prior phase | CSV/JSON session export |

---

## Reliability

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| No secrets exposed | Passed | Sanitizer tests + git review | `.env` ignored; catalog never stores keys |
| No silent dynamic-prompt fallback | Passed | Live YOLO-World | Fail-closed injection |
| Zero detections ≠ failure | Passed | UI + helpers | Distinct empty-state copy |
| Old history opens | Passed | History label helper | Deleted models show stored metadata |
| Cloud startup healthy | Passed | Streamlit `/_stcore/health` | Returns `ok` |
| Schema migrations | Passed | Offline pytest | Schema **v8** (through signup/reset) |

---

## Demo experience

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Login-first landing | Passed | AppTest | Sign in + Create an account / Forgot password; no dashboard before auth |
| Sample does not auto-run | Passed | Offline test + code | Navigates to Photos only |
| Progress phases | Passed | Code inspection | Preparing → prompts → model → detections → review |
| Connection test isolation | Passed | Offline test | Wizard uploads/results restored after probe |
| Empty states | Passed | Code inspection | Photos, models, compare, history, benchmark, workspace |

---

## Authentication and roles

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| App opens on a login screen | Passed | AppTest | `test_app_opens_on_login_screen` |
| No default or hardcoded account | Passed | Offline pytest | Bootstrap only while `users` is empty |
| First-admin bootstrap | Passed | Offline pytest + AppTest | Creates one admin, usable immediately; never re-runs |
| Argon2id hashing, no plaintext | Passed | Offline pytest | Hash prefix asserted; temp password absent from DB file |
| Password policy | Disabled | Offline pytest | Complexity intentionally removed; only blank is refused |
| Self-signup pending + admin approve/reject | Passed | AppTest / signup tests | `test_auth_signup_reset.py` |
| Password reset request + admin authorize | Passed | AppTest / signup tests | Temp password once; force change |
| Generic login failure | Passed | AppTest | Unknown / wrong / disabled indistinguishable |
| Lockout after 5 failures, 15 min | Passed | Offline pytest + AppTest | Admin unlock clears it |
| Forced password change blocks the app | Passed | AppTest | Wrong current password refused; blank new password refused |
| Idle and absolute session timeouts | Passed | Offline pytest + AppTest | Returns to login with an inactivity notice |
| Session-version invalidation | Passed | Offline pytest | Password change and admin reset end other sessions |
| Deactivation revokes a live session | Passed | AppTest | Next rerun signs the user out |
| Logout cleanup | Passed | AppTest | Identity, wizard, benchmark, admin and transient state cleared |
| Final-admin protection | Passed | Offline pytest + AppTest | Role, deactivate and delete disabled and refused in the store |

---

## Authorization and data ownership

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Regular user denied the admin console | Passed | AppTest | Hidden in navigation and refused when forced; `authz.denied` |
| Regular user denied API Keys | Passed | AppTest / UI | Admin-only OpenRouter key page |
| Admin permitted | Passed | AppTest | All **eight** console tabs render |
| History scoped per user | Passed | Offline pytest | Query filters by `user_id`; legacy rows never shared |
| New records attributed | Passed | Offline pytest | `user_id` and `username` stored on save |

---

## User management and audit

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Create user with one-time password | Passed | AppTest | Shown once; absent from the database file |
| Duplicate username and email rejected | Passed | Offline pytest + AppTest | Explained, not silently overwritten |
| Admin reset forces change | Passed | AppTest | Old password stops working; `session_version` incremented |
| Disable / enable | Passed | AppTest | Round-trip verified |
| Role update | Passed | Offline pytest | Audited as `admin.user.updated` |
| Audit events recorded | Passed | AppTest | Bootstrap, sign-in, signup/reset, password change present |
| No secrets in the audit log | Passed | Offline pytest | Recursive redaction of details |

---

## OpenRouter (administrator-managed key)

| Item | Status | Verification method | Evidence / note |
|------|--------|---------------------|-----------------|
| Key format validated before any call | Passed | Offline pytest | Non-OpenRouter values rejected locally |
| Verification uses `GET /api/v1/key` | Passed | Offline pytest | No model call; free metadata read |
| Invalid / forbidden / rate-limited / timeout handled | Passed | Offline pytest | Distinct actionable messages |
| Stored in `deployment_secrets` | Passed | Offline pytest / migration 6 | Survives admin sign-out; not shown to users |
| Key redacted from diagnostics and errors | Passed | Offline pytest | Recursive redaction |
| Model seeds disabled until admin enables | Passed | Offline pytest | `ensure_default_policies` |
| Daily per-user quota | Passed | Offline pytest | Blocks at the limit; `policy.quota.blocked` |
| Direct VLM path (not Roboflow Workflow) | Passed | Offline VLM tests | `openrouter_vlm.py` chat/completions |

---

## Manual acceptance sequence

Run against a fresh `DATA_DIR` before a demo.

1. Start with no database. The app opens on the login screen and names the
   bootstrap administrator it created.
2. Remove the bootstrap variables, restart, delete the database: the login page
   explains which variables to set and creates nothing.
3. Sign in as the administrator with the bootstrap password → the dashboard
   loads directly, with no forced password change.
4. From a signed-out browser, **Create an account** → confirm it cannot sign in
   until approved. As admin, Approve under Pending sign-ups.
5. Sign in with a wrong password five times → the account locks for 15 minutes.
6. As administrator, unlock, then create a regular user and copy the one-time
   password.
7. Sign in as the regular user → forced change → dashboard. Confirm there is no
   administrator console or API Keys entry.
8. Run a YOLO-World analysis on a sample. On Review: step Prev/Next, confirm
   red focus outline, **Exclude this item**, then save. History shows only that
   user's record.
9. Sign back in as the administrator and confirm History shows only the
   administrator's own saves.
10. As admin on **API Keys**, verify an OpenRouter key. Under **Model Access**,
    enable OpenRouter VLM Detector. As a regular user, run one analysis — the
    key is never visible.
11. In **Model Access**, set the OpenRouter daily quota to 1, run once as a
    user, and confirm the second attempt is blocked with a clear reason.
12. Deactivate the regular user while they are signed in. Their next
    interaction returns them to the login screen.
13. In **Audit Log**, confirm sign-ins, signup/reset events, key verification,
    policy updates and quota blocks — with no key or password in details.
14. Leave a session idle past the timeout and confirm it returns to login with
    an inactivity notice.

---

## Deferred (intentional)

| Item | Status | Reason |
|------|--------|--------|
| Built-in Cardboard Boxes sample | Deferred | No verified box image in repo; do not fabricate assets |
| Persistent cloud database | Deferred | Roadmap only (Supabase/PostgreSQL/object storage) |
| Additional trained workspace OD models | Deferred | Workspace may have 0 trained OD projects |
| Production accuracy claims | Deferred | POC estimates require human review |
| SSO / OIDC and MFA | Deferred | Provider interface exists; local passwords only for the POC |
| Encryption at rest and tamper-evident audit | Deferred | Requires managed storage; documented in `SECURITY_MODEL.md` |
| Spend estimation and reconciliation | Deferred | Quotas only; admin key billing |
| Per-user OpenRouter BYOK | Deferred / superseded | Replaced by admin deployment key |
