# Authentication and Roles

How sign-in, sessions and roles work in the AI Inventory Counter POC.

> **Not production-ready.** This is a local username/password proof of concept
> intended for a small, trusted group of demo users. See
> [`SECURITY_MODEL.md`](SECURITY_MODEL.md) for what is and is not defended
> against, and the [future direction](#future-direction) section below for the
> intended path to SSO.

---

## Overview

The app is login-first. An unauthenticated visitor sees the product name, a
short description, **Sign in**, and expanders for **Create an account** and
**Forgot password?** — no dashboard, wizard, history, or model list.

Users may **create an account** with their own password; the account stays
**pending** until an administrator approves it. Password resets are
**requested** from the login page and likewise require administrator
authorization (a temporary password is generated and shown once in the admin
console).

| Role | Can do |
|------|--------|
| `user` | Run analyses, review detections, save counts, see **their own** inventory history, change their own password |
| `admin` | Everything a user can, plus the [administrator console](ADMIN_GUIDE.md) and **API Keys** (OpenRouter deployment key). Inventory history stays private to each account. |

Roles are stored per user in the `users` table and re-read from the database on
every rerun, so a role change or deactivation takes effect on the affected
user's next interaction without waiting for their session to expire.

---

## First administrator bootstrap

The database ships with no users and no default password. The first
administrator is created from configuration, once, and only while the `users`
table is completely empty.

Set these before first launch, in `.env` locally or in **App → Settings →
Secrets** on Streamlit Community Cloud:

```bash
BOOTSTRAP_ADMIN_USERNAME=admin
BOOTSTRAP_ADMIN_PASSWORD=admin
BOOTSTRAP_ADMIN_EMAIL=admin@example.com
```

On startup the app checks whether any user exists:

- **No users, and the variables are set** — the administrator is created, an
  `admin.bootstrap` audit event is recorded, and the login page confirms the
  username. The account is usable immediately; there is no forced change.
- **No users, and the variables are missing** — the login page explains which
  variables to set. Nothing is created, and no fallback account exists.
- **At least one user already exists** — bootstrap does nothing, even if the
  variables are still present. It cannot be used to overwrite or re-create an
  account.

> The default is `admin`/`admin`. That is a demo convenience, not a safe
> configuration — anyone who reaches the URL can sign in as an administrator.
> Choose a real password before exposing this anywhere.

> On Streamlit Community Cloud the database is ephemeral, so expect to re-run
> bootstrap after a redeploy or a container restart. See
> [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md).

---

## Signing in

1. Open the app and enter a username and password.
2. On success, the session is established and the role-appropriate dashboard
   loads.
3. If `force_password_change` is set (accounts an administrator created, and
   administrator password resets), a change-password screen blocks everything
   else until a new password is chosen. Bootstrap accounts skip this.
4. Pending (not yet approved) accounts cannot sign in — they see a waiting-for-
   approval message.

Failures for unknown / wrong / deactivated accounts are deliberately vague — a
single *"Invalid username or password."* — so the login form cannot be used to
discover which accounts exist.

### Lockout

| Setting | Value |
|---------|-------|
| Failed attempts before lock | 5 |
| Lock duration | 15 minutes |
| Counter reset | On any successful sign-in |

While locked, the form reports the remaining minutes rather than accepting more
attempts. An administrator can clear a lock immediately from **Admin Console →
Users → Unlock**. Both the lock (`auth.login.locked`) and the unlock
(`admin.user.unlocked`) are audited.

---

## Self-signup and password reset requests

| Flow | User action | Admin action |
|------|-------------|--------------|
| Sign up | **Create an account** → chooses username + password | **Users → Pending sign-ups** → Approve or Reject |
| Forgot password | **Forgot password?** → submits username | **Users → Password reset requests** → Authorize reset (temp password once) or Reject |

Approve activates the account with the password they chose. Authorize reset
sets `force_password_change`, invalidates sessions, and shows a temporary
password once — deliver it out of band. There is no email delivery in this POC.

---

## Passwords

Passwords are hashed with **Argon2id** (`argon2-cffi`). Plaintext is never
written to the database, the audit log, session state or any log line, and the
`UserRecord` object the application works with does not even carry a hash
field.

**There is no password policy.** Complexity rules are disabled for this POC, so
any non-empty value is accepted: length, character classes, reuse of your own
username, and repeating your previous password are all permitted. The only
rejected value is an empty one, because Argon2 cannot hash it and the account
would become unreachable.

This is a deliberate trade for demo convenience. It means the strength of every
account rests entirely on what the person typing chooses. Reinstate
`security.validate_password_policy` before this handles anything real.

Hash parameters are re-checked at sign-in; if they fall behind the configured
cost, the password is transparently re-hashed using the verified plaintext.

### Changing your own password

**Account → Change password.** Requires the current password. On success the
user's `session_version` is incremented, which invalidates every other session
for that account, and an `auth.password.changed` event is recorded.

### Administrator reset

**Admin Console → Users → Generate temporary password** (or Authorize reset on
a request). The console displays a generated temporary password exactly once —
it is not stored anywhere and cannot be shown again. The account is flagged
`force_password_change`, and `session_version` is incremented so any active
session for that user is immediately invalidated.

Administrators cannot read existing passwords; reset is the only recovery path.

---

## Sessions

Identity lives in Streamlit session state and is re-validated on every rerun
against the database.

| Control | Default | Override |
|---------|---------|----------|
| Idle timeout | 30 minutes since last activity | `SESSION_IDLE_TIMEOUT_MINUTES` |
| Absolute timeout | 12 hours since sign-in | `SESSION_ABSOLUTE_TIMEOUT_HOURS` |

A session also ends immediately, mid-use, when the account is deactivated,
deleted, or has its `session_version` bumped by a password change or
administrator reset. In every case the user is returned to the login screen
with a short explanation ("signed out due to inactivity", "session is no longer
valid"), and the sign-out cleanup below runs.

`last_activity_at` is written back to the database at most once per minute per
user to avoid a write on every rerun.

### What sign-out clears

Logout, timeout and invalidation all run the same cleanup, so nothing survives
into the next session on a shared browser:

- Authenticated identity and activity timestamps.
- Any leftover session OpenRouter key fields (legacy cleanup — the live key is
  the admin deployment secret in SQLite).
- Wizard state: uploaded photos, form, analysis results, review edits, saved
  record, inference cache.
- Benchmark state, including batch images and ground-truth edits.
- Administrator console state and notices.
- Transient UI state and prefixed widget values (`login_`, `pwchange_`,
  `admin_user_`, `sample_sel_`, `prompt_`, and similar).

---

## Authorization

Administrator surfaces are guarded in two places, not one:

- **Navigation** hides what the user cannot use — regular users never see the
  administrator console or **API Keys** entry.
- **The page itself** re-checks the role before rendering anything. Forcing the
  view directly (for example by setting the view in session state) shows a
  permission-denied message and records an `authz.denied` audit event naming
  the actor.

The same principle applies to data: history queries are scoped by `user_id` in
the query itself, not filtered in the UI. Every account — administrator or not —
sees only the inventory counts it saved.

---

## Audit log

Security-relevant actions are appended to the `audit_events` table with an
actor, a target, an outcome, a UTC timestamp and a redacted detail payload.
Details pass through the same recursive redaction used everywhere else, so API
keys, passwords and tokens cannot reach the log even by accident.

Recorded events include:

- Bootstrap / auth: `admin.bootstrap`, `auth.login.success`,
  `auth.login.failure`, `auth.login.locked`, `auth.logout`,
  `auth.session.timeout`, `auth.session.invalidated`, `authz.denied`,
  `auth.password.changed`, `auth.signup.requested`,
  `auth.password.reset_requested`
- Admin user lifecycle: `admin.user.created`, `admin.user.updated`,
  `admin.user.activated`, `admin.user.deactivated`, `admin.user.deleted`,
  `admin.user.unlocked`, `admin.password.reset`, `admin.signup.approved`,
  `admin.signup.rejected`
- Policy / samples / keys: `admin.policy.updated`, `admin.sample.uploaded`,
  `admin.sample.updated`, `admin.sample.deleted`, `byok.key.verified`,
  `byok.key.verify_failed`, `byok.key.removed`, `byok.cost.acknowledged`
  (legacy), `inference.run`, `policy.quota.blocked`

Administrators can filter and export the log from **Admin Console → Audit Log**.

---

## Lockout protection for administrators

The last active administrator cannot be demoted, deactivated or deleted, and
administrators cannot deactivate or delete themselves. The console disables
those controls and explains why, and the underlying store rejects the operation
as well, so the rule holds even if the UI is bypassed.

---

## Future direction

The authentication layer sits behind a small provider interface
(`AuthenticationProvider`), and the rest of the app only ever sees a canonical
`AuthenticatedUser` object. Swapping the local password provider for OIDC/SSO
is intended to be a change in one module, with `auth_provider` already recorded
per user so both can coexist during a migration. PostgreSQL and object storage
are the intended replacements for SQLite and local sample files; see
[`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md).
