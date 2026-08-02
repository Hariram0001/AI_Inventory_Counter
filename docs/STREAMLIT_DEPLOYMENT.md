# Streamlit Deployment

Deploying the AI Inventory Counter POC to Streamlit Community Cloud, and the
storage limitation you must plan around.

---

## The limitation, first

**Streamlit Community Cloud storage is ephemeral.** The container's local disk
is wiped whenever the app is redeployed, restarted, or wakes from sleep after
inactivity.

That means these are all lost:

- `data/inventory_counts.db` — user accounts, saved inventory counts, audit
  events, model access policies, usage counters
- `data/admin_samples/` — administrator-uploaded sample images
- Database backups written before migrations

The practical consequences:

- **You will re-run the first-admin bootstrap after every reset.** Keep
  `BOOTSTRAP_ADMIN_USERNAME` / `BOOTSTRAP_ADMIN_PASSWORD` in Secrets for as
  long as you are demoing, and treat the password as rotatable.
- **All user accounts disappear** with the database. Re-create them from the
  console after a reset.
- **Export anything you need to keep.** Inventory history exports to CSV and
  the audit log exports to CSV; do it before a redeploy, not after.
- Only files committed to Git survive — which is why the built-in samples under
  `assets/sample_images/` are durable and administrator uploads are not.

Say this out loud in any stakeholder demo. It is a proof of concept, and
pretending the data is durable is the fastest way to lose someone's trust.

---

## Deploying

1. Push to GitHub.
2. In Streamlit Community Cloud, create an app pointing at the repository,
   branch and `app.py`.
3. Set Python 3.11 (`.python-version` is committed).
4. Add secrets (below).
5. Deploy, then open the app and complete the first-admin bootstrap.

Health check: `https://<your-app>.streamlit.app/_stcore/health` should return
`ok`.

---

## Secrets

**App → Settings → Secrets** takes the same keys as `.env`, in TOML form. The
app reads `st.secrets` when available and falls back to environment variables,
so local and cloud configuration stay identical.

```toml
# Roboflow — the deployment's own key, used for YOLO-World and the
# OpenRouter workflow execution.
ROBOFLOW_API_KEY = "your-roboflow-key"
ROBOFLOW_WORKSPACE = "hariram-s-mzhvc"
ROBOFLOW_WORKFLOW_ID = "custom-workflow"

# First administrator bootstrap — used once, only while no users exist.
BOOTSTRAP_ADMIN_USERNAME = "admin"
BOOTSTRAP_ADMIN_PASSWORD = "choose_a_real_password_for_any_public_url"
BOOTSTRAP_ADMIN_EMAIL = "admin@example.com"

# Optional
# SESSION_IDLE_TIMEOUT_MINUTES = 30
# SESSION_ABSOLUTE_TIMEOUT_HOURS = 12
# OPENROUTER_MODELS_ENABLED = true
# OPENROUTER_WORKFLOW_ID = "playground-gpt-5-6-luna-od"
# DEMO_MODE = false
```

**No OpenRouter API key belongs here.** Every user supplies their own on the
API Connections page, and it is held in their session only. See
[`OPENROUTER_BYOK.md`](OPENROUTER_BYOK.md).

Changing secrets restarts the app — which, per the section above, wipes the
database. Batch your changes.

**This POC enforces no password complexity, and the shipped default is
`admin`/`admin`.** The bootstrap account is usable immediately with no forced
change, so whatever you put in `BOOTSTRAP_ADMIN_PASSWORD` is the password —
permanently, until someone changes it in the app. On a public Streamlit Cloud
URL that means anyone who finds the app is an administrator. Set a real value
here, or do not deploy this build publicly.

---

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then edit
streamlit run app.py
```

Locally `DATA_DIR` defaults to `./data` and **is** durable, so accounts and
history persist across restarts. Database backups are written alongside it
before each migration. `data/` is gitignored — do not commit it.

---

## Database migrations

The schema is versioned (currently version 5) and migrations run automatically
at startup. They are idempotent and transactional: re-running is safe, a
failure rolls back rather than leaving a partial schema, and the database is
backed up before an upgrade.

Migration 1 covers the original inventory table; 2 adds users and audit events;
3 adds per-user ownership to inventory counts; 4 adds model access policies and
usage counters; 5 adds administrator samples. Existing records are preserved —
inventory rows saved before authentication existed stay in the database but are
not shown in any user's private history (history is never a shared pool).

---

## Dependencies

`requirements.txt` pins what the deployment needs, including `argon2-cffi` for
password hashing. If you fork or vendor dependencies, keep that pin: without
it, the app cannot hash or verify passwords and nobody can sign in.

---

## Future durable storage

The path off ephemeral storage, when this stops being a POC:

- **Database** — managed PostgreSQL. `database.py` isolates connection handling
  and the migration framework, so the schema can move without touching callers.
- **Samples** — object storage (S3 or equivalent). `admin_samples.py` isolates
  file reads and writes behind a small interface for the same reason.
- **Sessions and identity** — OIDC/SSO through the existing
  `AuthenticationProvider` interface, with `auth_provider` already recorded per
  user so local and federated accounts can coexist during a migration.
- **Audit** — ship events to an append-only external sink so they survive the
  application and cannot be edited from it.
- **Hosting** — a platform with persistent volumes and real backups, behind a
  reverse proxy with TLS, rate limiting and CSRF protection.

None of this is implemented. It is written down so the next person knows where
the seams were deliberately left.
