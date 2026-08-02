# Security Model

What this POC defends against, how, and — just as importantly — what it does
not.

> **This is a proof of concept, not a production system.** It has not been
> penetration tested, threat modelled by a security team, or reviewed for
> compliance with any standard. Do not put regulated, personal or commercially
> sensitive data into it. Treat every deployment as disposable.

---

## Trust model

The app assumes a **small group of known, trusted users** who each have an
account created by an administrator. It defends against accidents and casual
misuse — a user seeing another user's data, a secret ending up in a log, one
person burning through shared API credit — rather than against a determined
attacker with access to the host.

Anyone with filesystem access to the deployment can read the SQLite database
directly. Password hashes are Argon2id, so passwords are not recoverable, but
inventory history and audit events are not encrypted at rest.

---

## Authentication

- Local username/password, hashed with **Argon2id** (`argon2-cffi`).
- No default, hardcoded or fallback account exists. The first administrator
  comes from one-time bootstrap configuration; without it, nobody can sign in.
- **Password complexity is disabled.** Any non-empty value is accepted, and the
  default deployment ships with `admin`/`admin`. This is a deliberate POC
  convenience and the single largest weakness here: anyone who can reach the URL
  can sign in as an administrator. Do not expose this build publicly.
- Lockout after 5 failed attempts for 15 minutes, with administrator unlock.
- Login failures are uniform: unknown username, wrong password and deactivated
  account are indistinguishable, so accounts cannot be enumerated.
- Timing is not constant across all paths; a determined attacker could in
  principle distinguish some cases by response time. Accepted for a POC.

See [`AUTHENTICATION_AND_ROLES.md`](AUTHENTICATION_AND_ROLES.md) for the full
flow.

---

## Sessions

- Identity is held in Streamlit session state and **re-validated against the
  database on every rerun**, so deactivation, deletion and role changes take
  effect immediately rather than at the next sign-in.
- Idle timeout (30 minutes) and absolute timeout (12 hours), both configurable.
- `session_version` per user: a password change or administrator reset
  increments it, invalidating every other live session for that account.
- Sign-out, timeout and invalidation all run the same cleanup, clearing
  identity, the BYOK key, wizard state, benchmark state, admin state and
  prefixed widget values.

Session state is Streamlit's own, held server-side per connection. This POC
does not implement CSRF tokens, cookie hardening or device/session listings.

---

## Authorization

- Two-layer enforcement: navigation hides what a role cannot use, **and** every
  protected page re-checks the role before rendering. Forcing a view directly
  yields a permission-denied message and an `authz.denied` audit event.
- Data access is scoped in the query, not the UI. `get_inventory_history`
  always filters by the signed-in `user_id`, so no account — including an
  administrator — can read another user's inventory history through the app.
- The last active administrator cannot be demoted, deactivated or deleted, and
  administrators cannot deactivate or delete themselves — enforced in the store
  as well as the console.

---

## Secret handling

Three classes of secret exist, treated differently:

| Secret | Where it lives | Persisted? |
|--------|----------------|-----------|
| Roboflow deployment key | Environment / Streamlit Secrets | Not by the app |
| Admin OpenRouter key | SQLite `deployment_secrets` (admin-only UI) | Yes (deployment-scoped; never shown to users) |
| Passwords | Argon2id hash in SQLite | Hash only |

### Redaction

A single recursive redaction routine (`security.redact_secrets` /
`security.redact_text`) is applied at every boundary where data could escape:

- Audit event details.
- Diagnostics payloads and workflow parameter dumps.
- Exception and error text shown to users, including text returned by upstream
  APIs.
- Saved catalog and registry JSON.

It walks nested dictionaries and lists, replaces values under sensitive keys,
and scrubs secret-shaped substrings from free text — `Bearer` tokens,
`api_key=` query parameters, `Authorization` headers, and `sk-`-style keys —
regardless of nesting depth. Key-name matching deliberately ignores metadata
fields such as `api_key_parameter_name` so that configuration remains readable.

Masking for display (`sk-o…4f2a`) is separate from redaction and never reveals
enough to reconstruct a key.

---

## Data ownership

- New inventory records store `user_id` and `username`.
- Every account sees only its own records. Pre-authentication / unowned rows
  are not shared into any user's history.
- Deleting a user does not delete their history. Records stay attributed to the
  username so the trail survives the account.

---

## Audit trail

Security-relevant actions are appended to `audit_events` with actor, target,
outcome, UTC timestamp and a redacted detail payload: bootstrap, sign-in
success and failure, lockout, logout, timeout, invalidation, permission denial,
user lifecycle changes, password changes and resets, policy updates, sample
changes, key verification and removal, cost acknowledgement, inference runs and
quota blocks.

The log is append-only in practice — nothing in the app updates or deletes
rows — but it is not tamper-evident. Anyone with database access can alter it.

---

## Cost controls

Model access policies gate every model by role, and can require a user-supplied
key, require cost acknowledgement, and cap runs per user per UTC day (25 by
default for the OpenRouter detector). Quota blocks are audited. These are
guardrails against accidental overspend, not a billing system: they do not
estimate cost, reconcile against provider invoices, or stop a user spending
their own money at the provider directly.

---

## Input validation

- Sample uploads must decode as JPEG or PNG, be at most 15 MB, and measure
  between 64 and 8000 pixels per side. Filenames are slugified so they cannot
  traverse out of the samples directory.
- Usernames and emails are normalised and uniqueness-checked.
- Workflow responses are validated before use; visualization-only, text-only
  and malformed-coordinate responses are rejected rather than turned into a
  count.

---

## Database safety

Schema changes run through versioned, idempotent, transactional migrations. The
database is backed up before an upgrade, migrations re-run safely, and a failed
migration rolls back rather than leaving a half-applied schema. No migration
drops user data.

---

## Known limitations

Stated plainly, because a POC that pretends otherwise is worse than one that
does not:

- **Ephemeral storage on Streamlit Community Cloud.** The database, samples and
  audit log are lost on redeploy, restart or sleep. See
  [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md).
- **No SSO/MFA.** Local passwords only. The provider interface exists so OIDC
  can replace it, but it is not implemented.
- **No encryption at rest** for inventory data or audit events.
- **No rate limiting** on the application itself beyond login lockout and daily
  model quotas.
- **No email delivery.** Temporary passwords are handed over out of band by the
  administrator.
- **No password expiry, history or reuse prevention** across time.
- **Not tamper-evident.** Host access defeats the audit trail.
- **Photographs are not stored**, so a saved count cannot be re-verified
  against its original images.

---

## If you were to productionise this

In rough priority order: move to OIDC/SSO with MFA; move the database to
managed PostgreSQL with encryption at rest and backups; move samples to object
storage; put the app behind a reverse proxy with TLS termination, rate limiting
and CSRF protection; ship audit events to an append-only external sink;
introduce real cost accounting per user; and commission an independent security
review before any of it handles real inventory data.
