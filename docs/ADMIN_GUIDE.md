# Administrator Guide

Everything an administrator can do in the AI Inventory Counter POC, and what
each action means for the people using the app.

Open it from the dashboard: **Open administrator console**. The console is
hidden from regular users and refuses to render for them even if the view is
requested directly.

---

## Console tabs

| Tab | Purpose |
|-----|---------|
| Overview | Live counts: users, active, administrators, locked, saved counts, samples, schema version, recent activity |
| Users | Create, manage, reset, unlock and delete accounts |
| Samples | Upload and curate demo images (inventory or Shape Detection classification) |
| Model Access | Decide which models each role may run, and under what conditions |
| Experimental Features | Enable/disable Shape Detection per role; local CV policy (no API quota) |
| Connectivity | Check Roboflow and OpenRouter reachability and workflow schemas |
| Audit Log | Filter and export security events |
| Storage and System | Storage paths, sizes and runtime information |

Shape Detection itself is opened from **Home** (not from this console). See
[`SHAPE_DETECTION_TESTING.md`](SHAPE_DETECTION_TESTING.md).

---

## Users

### Creating a user

**Users → Create a user**: username, role, optional display name, optional
email.

The console generates a temporary password and shows it **once**, in a copyable
block. It is not stored and cannot be retrieved again — copy it before leaving
the page and deliver it to the person through a channel you trust. The account
is created with `force_password_change`, so their first sign-in immediately
asks them to choose their own password.

Usernames must be unique and are normalised to lowercase. Email addresses, when
supplied, must also be unique. Duplicates and malformed values are rejected
with an explanation rather than silently overwriting anything.

Note that this POC enforces **no password complexity** — whatever the user
chooses at their first sign-in is accepted, including single characters. See
[`SECURITY_MODEL.md`](SECURITY_MODEL.md).

### Day-to-day management

Select a user from the table to reveal the controls:

- **Apply role** — switch between `user` and `admin`. Takes effect on that
  user's next interaction.
- **Deactivate / Activate** — deactivating ends any live session for that
  account at once and blocks future sign-in. The user's saved inventory history
  is untouched. This is the reversible way to remove access, and usually the
  right one.
- **Unlock** — clears a lockout from repeated failed sign-ins without waiting
  for the 15-minute window to pass.
- **Generate temporary password** — see below.
- **Delete user** — permanent. Requires typing the exact username to confirm.
  Their saved inventory history remains, still attributed to the username, so
  the audit trail stays intact.

### Resetting a password

Administrators cannot read passwords. **Generate temporary password** creates a
new one, shows it once, sets `force_password_change` and invalidates every
active session for that account. The user signs in with the temporary password
and is required to replace it before reaching anything else.

### Protection against lockout

The last active administrator cannot be demoted, deactivated or deleted, and
you cannot deactivate or delete your own account. The console disables those
buttons and explains why. Promote a second administrator first if you need to
remove the current one.

---

## Model Access

Each model has a policy row that governs who may run it. Defaults are seeded on
first launch:

| Model | Enabled | Roles | User key required | Cost confirmation | Daily runs per user |
|-------|---------|-------|-------------------|-------------------|---------------------|
| YOLO-World | Yes | admin, user | No | No | Unlimited |
| Local Picket Counter | Yes | admin, user | No | No | Unlimited |
| OpenRouter VLM Detector | Yes | admin, user | **Yes** | **Yes** | 25 |

For each model you can set:

- **Enabled** — disabling hides it from every user, immediately.
- **Allowed roles** — restrict a model to administrators while evaluating it.
- **Requires a user API key** — the model only appears for users who have
  verified their own OpenRouter key in this session (see
  [`OPENROUTER_BYOK.md`](OPENROUTER_BYOK.md)).
- **Requires cost confirmation** — the user must acknowledge the paid-usage
  notice before the first run.
- **Maximum runs per user per day** — a per-user, per-UTC-day counter. When the
  quota is exhausted, the model becomes unselectable for that user until the
  next day and a `quota.blocked` event is recorded.

Policy changes are audited (`policy.updated`) and take effect on the next
rerun. When a model is unavailable, the user is told **why** — wrong role, no
verified key, cost notice not accepted, quota exhausted, or unsupported for the
chosen inventory — rather than the model silently disappearing.

---

## Samples

**Samples** manages the demo images offered on the home page.

Uploads are validated before anything is written: the file must be a decodable
JPEG or PNG, at most 15 MB, and between 64 and 8000 pixels on each side.
Filenames are slugified, so a crafted name cannot escape the samples directory.
Each sample stores a title, description, inventory type, expected count and an
enabled flag; disabling hides it from users without deleting the file.

Samples live in `DATA_DIR/admin_samples/` and are registered in the
`admin_samples` table. On Streamlit Community Cloud both are lost on redeploy —
the built-in samples committed under `assets/sample_images/` are the durable
ones.

---

## Connectivity

Shows configuration state for Roboflow and OpenRouter without revealing
secrets: whether a deployment key is configured, which workspace and workflow
are targeted, and the inputs and outputs the workflow declares.

**Run Roboflow connectivity test** performs a live call using the deployment's
own key. It is opt-in because it consumes a Roboflow credit. There is no live
OpenRouter test here: the deployment holds no OpenRouter key, since every user
brings their own.

---

## Audit Log

Filter by event type and outcome, adjust the row limit, and export to CSV.

Every entry carries the actor, the target, the outcome and a UTC timestamp.
Detail payloads pass through recursive redaction, so API keys, passwords and
tokens never appear — a redacted placeholder does instead. Use this tab to
answer "who changed this account", "when did this user sign in", and "which
model runs were blocked by quota".

---

## Storage and System

Reports the database path and size, the samples directory, the schema version,
the Python and Streamlit versions, demo mode, and the active session policy.

It also carries the standing warning that Streamlit Community Cloud storage is
ephemeral. Read [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md) before
promising anyone that data will still be there tomorrow.

---

## Routine tasks

**Onboard someone:** Users → Create a user → copy the temporary password →
deliver it → confirm they signed in and changed it.

**Someone is locked out:** Users → select them → Unlock. If they have forgotten
the password, generate a temporary one instead.

**Someone leaves:** Deactivate rather than delete. Their history stays readable
and their access ends immediately.

**Evaluate a new model safely:** Model Access → restrict roles to `admin`, set
a low daily quota, then widen it once you are satisfied.

**Investigate an incident:** Audit Log → filter by event type or outcome →
export to CSV.
