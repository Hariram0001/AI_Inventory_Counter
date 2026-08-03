# Authentication and Roles

How sign-in, sessions, and roles work in the AI Inventory Counter POC.

> **Not production-ready.** Local username/password for a small trusted demo
> group. See [`SECURITY_MODEL.md`](SECURITY_MODEL.md).

**Authoritative deeper context:** [`COMPLETE_APP_OVERVIEW.md`](COMPLETE_APP_OVERVIEW.md) §3C.

---

## Overview

The app is **login-first**. Visitors see the product name, a short description,
**Sign in**, and expanders for **Create an account** and **Forgot password?** —
no dashboard or models until authenticated.

| Role | Can do |
|------|--------|
| `user` | Analyses, review/save, own history, own password, Shape Detection (WIP) |
| `admin` | All of the above, plus Admin Console and **API Keys** (OpenRouter deployment key). History remains private per account. |

After sign-in (and after forced password change), **everyone lands on Home**
(`welcome`), including administrators. Admins open Administration from the
left sidebar.

Roles are re-read from SQLite on every rerun.

---

## First administrator bootstrap

Used only while the `users` table is empty:

```bash
BOOTSTRAP_ADMIN_USERNAME=admin
BOOTSTRAP_ADMIN_PASSWORD=admin
BOOTSTRAP_ADMIN_EMAIL=admin@example.com
```

- Creates one admin, audit event `admin.bootstrap`, **no** forced password change.
- Missing vars → login warning; nothing created.
- Once any user exists, bootstrap never overwrites accounts.

Default `admin`/`admin` is a demo convenience — change before any public URL.
Cloud DB wipe requires bootstrap secrets again. See
[`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md).

---

## Account creation paths

1. **Bootstrap** — first admin from secrets.
2. **Admin Create a user** — temp password once; `force_password_change`.
3. **Self-signup** — user picks password; `account_status=pending` until Approve/Reject.

---

## Password reset requests

**Forgot password?** stores a request. Admin **Authorize reset** shows a
temporary password once, forces change, invalidates sessions; or **Reject**.

No email delivery — out of band only.

---

## Signing in

1. Username + password.
2. Success → Home.
3. `force_password_change` → change screen blocks everything else.
4. Pending accounts cannot sign in (waiting-for-approval message).

Unknown user / wrong password / deactivated → same generic failure text.

### Lockout

| Setting | Value |
|---------|-------|
| Failures | 5 |
| Duration | 15 minutes |
| Unlock | Admin console or wait |

Events: `auth.login.locked`, `admin.user.unlocked`.

---

## Passwords

- **Argon2id** (`argon2-cffi`).
- **No complexity policy** — any non-empty value.
- Transparent rehash if parameters fall behind.
- Own change: **Profile → Change password** → `auth.password.changed` + session bump.
- Admin reset: `admin.password.reset`.

---

## Sessions

| Control | Default | Env |
|---------|---------|-----|
| Idle | 30 minutes | `SESSION_IDLE_TIMEOUT_MINUTES` |
| Absolute | 12 hours | `SESSION_ABSOLUTE_TIMEOUT_HOURS` |

Also ends on deactivate/delete/`session_version` bump.

### Sign-out cleanup

Clears identity, wizard/analysis/review state, inference cache, benchmark
transients, admin UI state, leftover legacy OpenRouter session fields, and
prefixed widgets. The live OpenRouter key is the admin **deployment** secret
in SQLite (not cleared by user logout).

---

## Authorization

- Navigation hides admin surfaces from regular users.
- Pages re-check role (`authz.denied` if forced).
- History queries filter by `user_id`.
- API Keys view is admin-only.

---

## Audit events (current names)

Include among others:

`admin.bootstrap`, `auth.login.success`, `auth.login.failure`, `auth.login.locked`,
`auth.logout`, `auth.session.timeout`, `auth.session.invalidated`, `authz.denied`,
`auth.password.changed`, `auth.signup.requested`, `auth.password.reset_requested`,
`admin.signup.approved`, `admin.signup.rejected`, `admin.user.created`,
`admin.user.updated`, `admin.user.activated`, `admin.user.deactivated`,
`admin.user.deleted`, `admin.user.unlocked`, `admin.password.reset`,
`admin.policy.updated`, `admin.sample.*`, `byok.key.verified`,
`byok.key.verify_failed`, `byok.key.removed`, `inference.run`,
`policy.quota.blocked`

(`byok.*` names are historical; the key is deployment-scoped.)

---

## Final-admin protection

The last active administrator cannot be demoted, deactivated, or deleted; admins
cannot deactivate/delete themselves.

---

## Future direction

`AuthenticationProvider` interface exists for a future OIDC/SSO swap. Durable
identity/storage is not implemented. See roadmap.
