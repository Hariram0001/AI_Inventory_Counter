# OpenRouter — Administrator-Managed Key

The OpenRouter VLM Detector uses a single API key configured by an
**administrator**. Regular users never see, enter, or manage that key. They
can run OpenRouter models only after an administrator:

1. Adds and verifies an OpenRouter key, and
2. Enables the model under **Admin Console → Model Access**.

---

## Who can do what

| Action | Administrator | Regular user |
|--------|---------------|--------------|
| Open **API Keys** / enter an OpenRouter key | Yes | No — page is hidden and blocked |
| See the plaintext or masked key | Yes (masked in UI) | No |
| Enable / disable OpenRouter models | Yes (Model Access) | No |
| Run an enabled OpenRouter model | Yes | Yes (key used under the hood) |
| Accept a personal cost notice | N/A | N/A — billing is on the admin key |

---

## Administrator setup

1. Sign in as an administrator.
2. Open **API Keys** (or Admin Console → Connectivity → Open API Keys page).
3. Paste an OpenRouter inference key (`sk-or-…`) and choose **Verify and save key**.
4. Verification calls `GET https://openrouter.ai/api/v1/key` — a free metadata
   check that does **not** run a model.
5. Open **Admin Console → Model Access**, enable **OpenRouter VLM Detector**
   for the roles that should use it, and optionally set a daily run quota.

Remove the key from the same API Keys page when you want OpenRouter models to
stop working for everyone.

---

## How runs work for users

When a user selects an enabled OpenRouter model, the app injects the
administrator's deployment key into the Roboflow Workflow as `model_api_key`.
The user never sees that parameter. Results flow into the same review and save
path as YOLO-World.

Charges land on the OpenRouter account that owns the administrator key, and
the same run also consumes Roboflow Workflow credits from this deployment.

---

## Storage

The verified key is stored in the SQLite `deployment_secrets` table so it
survives administrator sign-out (otherwise users could not run models). It is:

- Never shown to regular users
- Never written into audit-event details (only a masked form and status)
- Redacted from diagnostics and error text like any other secret

On Streamlit Community Cloud the database is ephemeral — expect to re-enter
the key after a redeploy. See [`STREAMLIT_DEPLOYMENT.md`](STREAMLIT_DEPLOYMENT.md).

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Model unavailable for users | Admin has not enabled it under Model Access, or no deployment key is configured |
| "OpenRouter is not configured" | Administrator must add a key on **API Keys** |
| User sees no API Keys button | Expected — only administrators manage the key |
| Key gone after Cloud redeploy | Ephemeral storage; re-verify the key and re-enable the model |
