# Feature Matrix

Statuses reflect **code + offline tests** on current `main`.
Paid live OpenRouter/Roboflow inference is not marked Available without a current live pass.

| Feature | Status | Regular user | Administrator | External dependency | Credential required | Paid usage possible | Data stored | Cloud persistence | Known limitation |
|---------|--------|--------------|---------------|---------------------|---------------------|---------------------|-------------|-------------------|------------------|
| Login-first gate | Available | ✓ | ✓ | — | Account | No | Session | Session only | No SSO/MFA |
| Self-signup (pending) | Available | Request | Approve/Reject | — | — | No | `users` pending | Ephemeral DB | No email |
| Password reset request | Available | Request | Authorize/Reject | — | — | No | `password_reset_requests` | Ephemeral DB | Out-of-band temp password |
| Bootstrap first admin | Available with configuration | — | Via secrets | — | Bootstrap secrets | No | `users` | Ephemeral DB | Missing secrets → warning |
| Home dashboard | Available | ✓ | ✓ | — | Session | No | — | — | Admins land here too |
| Get Started wizard | Available | ✓ | ✓ | Models as chosen | Per model | Maybe | Session + history on save | History ephemeral | Human review required |
| Preset inventories | Available | ✓ | ✓ | — | — | No | Profiles in git | Profiles in git | Quality varies |
| Custom Item multi-type one scan | Available | ✓ | ✓ | Dynamic model | Per model | Maybe | Prompts in session | — | Model multi-class quality |
| Upload / camera | Available | ✓ | ✓ | — | — | No | Session bytes | No | Size/format limits |
| Built-in samples | Available | ✓ | ✓ | — | — | No | Git assets | Yes (git) | Small curated set |
| Admin samples | Admin only | View if enabled | Upload/manage | — | — | No | `admin_samples` + files | Ephemeral | Lost on Cloud reset |
| YOLO-World | Available with configuration | ✓ | ✓ | Roboflow | `ROBOFLOW_API_KEY` | Yes | Results session/history | History ephemeral | Threshold / granularity |
| Local Picket Counter | Available | ✓ if compatible | ✓ | Local CPU | None | No | Results session/history | History ephemeral | Fence Panel oriented |
| OpenRouter VLM Detector | Available with configuration | Run if enabled | Key + policy | OpenRouter | Admin deployment key | Yes | `deployment_secrets` + usage | Ephemeral | Seeds disabled; admin must enable |
| Demo Mode mocks | Available with configuration | ✓ | ✓ | — | `DEMO_MODE=true` | No | Mock JSON | — | Not live accuracy |
| Model Catalog | Available | ✓ | ✓ | Roboflow Management API for sync | Roboflow for refresh/test | Probe may cost | `model_catalog.json` | Ephemeral | Discovery ≠ Ready |
| Compare Models | Available with configuration | ✓ | ✓ | Peers | Per peer | Yes | Comparison meta | Ephemeral | Needs ≥2 validated peers |
| Review solo / red focus / Exclude | Available | ✓ | ✓ | — | — | No | `review_edits` session | — | Dense scenes still hard |
| Inventory History + CSV | Available | Own rows | Own rows | — | Session | No | `inventory_counts` | Ephemeral | No shared admin view of others |
| Detection Benchmark | Available with configuration | ✓ | ✓ | YOLO-World typically | Roboflow | Yes | `benchmarks*.json` | Ephemeral | Image-specific metrics |
| Shape Detection | Experimental | ✓ | ✓ | OpenCV local | None | No | Optional shape tables | Ephemeral | WIP accuracy |
| API Keys (OpenRouter) | Admin only | Hidden | ✓ | OpenRouter verify API | OpenRouter key | Verify free | `deployment_secrets` | Ephemeral | Users never see key |
| Admin Console (8 tabs) | Admin only | Denied | ✓ | — | Admin role | Connectivity opt-in | Policies / audit / samples | Ephemeral | Final-admin rules |
| Model Access policies / quotas | Admin only | Affected | Configure | — | — | Gates paid models | `model_access_policies`, usage | Ephemeral | Not billing reconciliation |
| Audit log | Admin only | — | Filter/export | — | — | No | `audit_events` | Ephemeral | Not tamper-evident |
| Theme light/dark | Available | ✓ | ✓ | — | — | No | Session | — | — |
| SSO / MFA / durable DB | Deferred | — | — | — | — | — | — | — | Roadmap only |
| Per-user OpenRouter BYOK | Deferred / superseded | — | — | — | — | — | — | — | Replaced by admin key |

### Status legend

- **Available** — usable in current code paths with offline coverage
- **Available with configuration** — needs secrets, policy enable, or compatible peers
- **Experimental** — shipped but explicitly WIP
- **Admin only** — role-gated
- **Blocked** — cannot run until a named blocker is cleared
- **Planned / Deferred** — not implemented in this POC
