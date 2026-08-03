# OpenRouter — Administrator-Managed Key

> Filename is historical (`BYOK`). The current POC uses a **single
> administrator-managed** OpenRouter key for the whole deployment. Regular
> users never see, enter, or manage that key.

The OpenRouter VLM Detector calls OpenRouter’s
`/api/v1/chat/completions` API directly (`openrouter_vlm.py`). It is **not**
routed through a Roboflow Workflow. Users can run it only after an
administrator:

1. Adds and verifies an OpenRouter key on **API Keys**, and
2. Enables the model under **Admin Console → Model Access** (it seeds
   **disabled**).

---

## Who can do what

| Action | Administrator | Regular user |
|--------|---------------|--------------|
| Open **API Keys** / enter an OpenRouter key | Yes | No — page is hidden and blocked |
| See the plaintext or masked key | Yes (masked in UI) | No |
| Enable / disable OpenRouter models | Yes (Model Access) | No |
| Run an enabled OpenRouter model | Yes | Yes (deployment key used under the hood) |
| Accept a personal cost notice for Analyze | N/A | N/A — billing is on the admin key |
| Catalog Test paid-inference checkbox | Yes (settings probe) | May see checkbox when running Catalog Test |

---

## Administrator setup

1. Sign in as an administrator.
2. Open **API Keys** (or Admin Console → Connectivity → Open API Keys page).
3. Paste an OpenRouter inference key (`sk-or-…`) and choose **Verify and save key**.
4. Verification calls `GET https://openrouter.ai/api/v1/key` — a free metadata
   check that does **not** run a model.
5. Open **Admin Console → Model Access**, enable **OpenRouter VLM Detector**
   for the roles that should use it, and optionally set a daily run quota
   (default seed: 25 runs per user per UTC day).

Remove the key from the same API Keys page when you want OpenRouter models to
stop working for everyone.

Optional env toggles (see `.env.example`): `OPENROUTER_MODELS_ENABLED`,
`OPENROUTER_MODEL_ID` (default `openai/gpt-5.6-luna`).

---

## How runs work for users

When a user selects an enabled OpenRouter model:

1. The app loads the administrator’s verified key from `deployment_secrets`.
2. `openrouter_vlm.py` sends the image and a strict JSON detection prompt to
   OpenRouter chat completions.
3. Normalized boxes flow into the same Review and Save path as YOLO-World
   (solo focus, red selection, Exclude this item, history).

The user never sees the key or any OpenRouter request headers. Charges land on
the OpenRouter account that owns the administrator key. YOLO-World and other
Roboflow models still use the deployment’s `ROBOFLOW_API_KEY` separately.

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
