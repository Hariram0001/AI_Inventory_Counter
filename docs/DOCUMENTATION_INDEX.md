# Documentation Index

Authoritative map of project Markdown. Prefer the **Authoritative for** column
when documents overlap.

**Code truth:** current `main` branch.
**Deepest product+technical description:** [`COMPLETE_APP_OVERVIEW.md`](COMPLETE_APP_OVERVIEW.md)

---

## Stakeholders

| Document | Purpose | Intended reader | Authoritative for | Related |
|----------|---------|-----------------|-------------------|---------|
| [`../README.md`](../README.md) | Entry point, setup, capability summary | Everyone | Quick start, repo orientation | Overview, Deployment |
| [`COMPLETE_APP_OVERVIEW.md`](COMPLETE_APP_OVERVIEW.md) | Full factual product + technical description | Stakeholders + engineers | End-to-end behavior | Feature Matrix, Security |
| [`STAKEHOLDER_DEMO_SCRIPT.md`](STAKEHOLDER_DEMO_SCRIPT.md) | Timed live demo path | Presenters | Demo sequence | Overview |
| [`POC_ACCEPTANCE_CHECKLIST.md`](POC_ACCEPTANCE_CHECKLIST.md) | Pass/fail acceptance evidence | Reviewers | Verification status | Feature Matrix |
| [`FEATURE_MATRIX.md`](FEATURE_MATRIX.md) | Compact capability grid | Stakeholders + PM | Feature status labels | Overview |

---

## Administrators

| Document | Purpose | Intended reader | Authoritative for | Related |
|----------|---------|-----------------|-------------------|---------|
| [`ADMIN_GUIDE.md`](ADMIN_GUIDE.md) | Day-to-day console operations | Admins | Console tabs & user ops | Auth, OpenRouter |
| [`AUTHENTICATION_AND_ROLES.md`](AUTHENTICATION_AND_ROLES.md) | Login, roles, sessions, audit events | Admins + security | Auth semantics | Security Model |
| [`OPENROUTER_BYOK.md`](OPENROUTER_BYOK.md) | Admin-managed OpenRouter key (filename historical) | Admins | OpenRouter credential flow | Admin Guide, Overview |
| [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md) | Cloud deploy, secrets, ephemeral DB | Deployers | Streamlit Cloud ops | Troubleshooting |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Symptom → fix tree | Admins + deployers | Operational fixes | Deployment, Auth |

---

## Developers

| Document | Purpose | Intended reader | Authoritative for | Related |
|----------|---------|-----------------|-------------------|---------|
| [`COMPLETE_APP_OVERVIEW.md`](COMPLETE_APP_OVERVIEW.md) | Architecture, schema, adapters | Developers | Implementation map | README, Security |
| [`SECURITY_MODEL.md`](SECURITY_MODEL.md) | Trust boundaries and gaps | Developers + security | Security claims | Auth |
| [`SHAPE_DETECTION_TESTING.md`](SHAPE_DETECTION_TESTING.md) | Local OpenCV WIP feature | Developers + testers | Shape Detection | Overview |
| [`../POC_ROADMAP.md`](../POC_ROADMAP.md) | Completed vs planned work | Developers + PM | Roadmap status | Feature Matrix |
| [`../assets/sample_images/README.md`](../assets/sample_images/README.md) | Built-in sample packaging | Developers | Sample assets | Photos flow |
| [`../vendor/opencv-python-cloud-stub/README.md`](../vendor/opencv-python-cloud-stub/README.md) | Cloud OpenCV stub rationale | Developers | Vendor stub only | Deployment |

---

## Topic → authoritative document

| Topic | Authoritative document |
|-------|------------------------|
| What the app does end-to-end | `COMPLETE_APP_OVERVIEW.md` |
| Feature Available/Blocked labels | `FEATURE_MATRIX.md` |
| Auth / signup / lockout / sessions | `AUTHENTICATION_AND_ROLES.md` |
| Admin console operations | `ADMIN_GUIDE.md` |
| OpenRouter key (admin-managed) | `OPENROUTER_BYOK.md` |
| Security gaps / redaction | `SECURITY_MODEL.md` |
| Streamlit Secrets / ephemeral DB | `STREAMLIT_DEPLOYMENT.md` |
| Shape Detection WIP | `SHAPE_DETECTION_TESTING.md` |
| Demo script | `STAKEHOLDER_DEMO_SCRIPT.md` |
| Acceptance evidence | `POC_ACCEPTANCE_CHECKLIST.md` |
| Roadmap Complete/Planned | `POC_ROADMAP.md` |
| Break/fix runbook | `TROUBLESHOOTING.md` |
| Local quick start | `README.md` |

---

## Classification of existing Markdown (audit)

| File | Classification | Notes |
|------|----------------|-------|
| README.md | Current after this refresh | Links to Overview / Index / Matrix |
| POC_ROADMAP.md | Current after this refresh | Complete vs WIP vs Planned |
| docs/COMPLETE_APP_OVERVIEW.md | Current (new) | Authoritative deep dive |
| docs/DOCUMENTATION_INDEX.md | Current (new) | This file |
| docs/FEATURE_MATRIX.md | Current (new) | Status grid |
| docs/TROUBLESHOOTING.md | Current (new) | Ops runbook |
| docs/AUTHENTICATION_AND_ROLES.md | Current after this refresh | Signup + Home landing |
| docs/ADMIN_GUIDE.md | Current after this refresh | Sidebar entry; OpenRouter seeds |
| docs/OPENROUTER_BYOK.md | Current (name historical) | Admin-managed + direct VLM |
| docs/SECURITY_MODEL.md | Current after this refresh | Trust model includes signup |
| docs/STREAMLIT_DEPLOYMENT.md | Current | Schema v8; admin OpenRouter |
| docs/POC_ACCEPTANCE_CHECKLIST.md | Current after this refresh | Match one-scan / sidebar Shape |
| docs/STAKEHOLDER_DEMO_SCRIPT.md | Current after this refresh | Login → Home; no Home Shape |
| docs/SHAPE_DETECTION_TESTING.md | Current | Sidebar WIP |
| assets/sample_images/README.md | Current | Samples packaging |
| vendor/.../README.md | Current | Stub only |

No `docs/ROBOFLOW_MODEL_CAPABILITIES.md` exists; model matrix lives in the Overview and Feature Matrix.
