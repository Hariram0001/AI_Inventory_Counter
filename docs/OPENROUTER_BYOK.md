# OpenRouter — Bring Your Own Key

The OpenRouter VLM Detector runs on **your** OpenRouter account. This
deployment holds no OpenRouter key of its own: you supply one, it lives in your
browser session only, and any charges land on your account.

---

## Why BYOK

Vision-language inference is billed per call. Rather than sharing one funded
key across every user — with no way to attribute spend or stop one person
exhausting it — each user brings their own key and pays for exactly what they
run.

---

## Adding your key

1. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys). An
   inference key is what you want, and it starts with `sk-or-`.
2. In the app, open **API Keys** (the API Connections page).
3. Read the cost notice and tick the acknowledgement.
4. Paste the key and choose **Verify key**.

Once verified, the OpenRouter VLM Detector becomes selectable in the Analyze
step. The page shows only a masked form of the key (for example
`sk-o…4f2a`) — the full value is never displayed back to you.

**Remove key** clears it immediately. Signing out, an idle timeout and an
absolute timeout all do the same.

---

## What verification actually does

Verification issues a single `GET https://openrouter.ai/api/v1/key` request
with your key.

**This does not run an AI model.** It reads the key's own metadata — validity,
label, usage and limits — so the check is free. The response is reduced to a
small sanitized object (valid or not, a masked key, and any free-tier or limit
flags) before anything is stored in session state; the raw response is not
retained.

Charges only occur when you actually run an analysis. See
[Costs](#costs-what-you-are-agreeing-to) below.

### Verification outcomes

| Result | What it means |
|--------|---------------|
| Verified | The key is valid and usable for inference |
| Rejected (401) | The key is wrong, revoked, or was mistyped |
| Rejected (403) | The key is valid but not permitted here — commonly a provisioning/management key rather than an inference key |
| Rate limited (429) | Too many requests to OpenRouter right now; wait and retry |
| Service error (5xx) | An OpenRouter-side problem; retry later |
| Network error / timeout | The app could not reach OpenRouter |
| Malformed response | OpenRouter returned something unexpected; treated as unverified |

Every outcome is audited as `key.verified` or `key.verify.failed` — recording
that a verification happened and whether it succeeded, never the key itself.

A key that does not look like an OpenRouter key is rejected locally, before any
network call is made.

---

## Where your key lives

**Session memory only.** Specifically:

- It is **never** written to the SQLite database, `models.json`, the catalog,
  saved history records, diagnostics dumps, exported CSVs or any file on disk.
- It is **never** written to the audit log — only the fact of a verification.
- It is scoped to your session. Another user, including an administrator,
  cannot read it, and administrators have no view that exposes it.
- It is cleared on sign-out, idle timeout, absolute timeout and session
  invalidation.
- If it is ever interpolated into an error message or a workflow parameter dump,
  recursive redaction replaces it with a placeholder before that text is shown
  or logged.

The practical consequence: **you re-enter your key each session.** That is the
intended trade-off. Nothing to store means nothing to leak.

---

## Costs — what you are agreeing to

Before your first OpenRouter run you must acknowledge the cost notice. It says,
in short:

- Running the OpenRouter VLM Detector calls a paid model on **your** OpenRouter
  account, and **you** are billed.
- Cost scales with the number of images and their size — a batch or a
  comparison run multiplies it.
- The same run also executes a Roboflow Workflow, which consumes **Roboflow**
  credits from this deployment in addition to your OpenRouter charge.
- Verifying your key is free; only inference costs money.
- This POC does not estimate, cap or reconcile spend. Set limits on your key at
  OpenRouter if you want a hard ceiling.

Acknowledgement is per session and recorded as `cost.acknowledged`. It is
cleared along with the key on sign-out.

---

## When the model is selectable

The OpenRouter VLM Detector appears only when **all** of the following hold:

1. You are signed in and your account is active.
2. OpenRouter is enabled for the deployment (`OPENROUTER_MODELS_ENABLED`).
3. The administrator's policy enables the model for your role.
4. You have a verified key in this session.
5. You have accepted the cost notice.
6. The workflow metadata is valid — the workflow declares the `image`,
   `classes` and `model_api_key` inputs the adapter needs.
7. The model supports the chosen inventory type.
8. You are under your daily run quota (25 runs per user per UTC day by
   default).

When it is not selectable, the app names the specific reason instead of hiding
the model without explanation.

---

## How a run works

The adapter calls the Roboflow Workflow
(`playground-gpt-5-6-luna-od`, override with `OPENROUTER_WORKFLOW_ID`) with
three inputs:

| Input | Value |
|-------|-------|
| `image` | The photo being analysed |
| `classes` | The prompt classes derived from the chosen inventory type |
| `model_api_key` | Your session key, passed through and never persisted |

and reads three outputs:

| Output | Use |
|--------|-----|
| `predictions` | Bounding boxes, fed into the normal review and compare flow |
| `label_visualization` | Optional annotated preview |
| `error_status` | Workflow-side failure signal |

Results flow into the same review, edit and save path as YOLO-World, so
detections stay editable and comparable. A response that contains only a
visualization, only text, or malformed coordinates is rejected rather than
being turned into a bogus count.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| "OpenRouter rejected your API key" | Key is invalid, revoked or mistyped. Create a fresh inference key and verify again. |
| "Not permitted" on a key you just made | You likely created a provisioning/management key. Create an inference key instead. |
| "Not enough credit" | Add credit at OpenRouter, or use a key with available balance. |
| "Rate limited" | OpenRouter is throttling; wait and retry. |
| Model disappeared mid-session | Session expired (key cleared), the administrator changed the policy, or you hit your daily quota. |
| Model not listed at all | OpenRouter disabled deployment-wide, or the model does not support the selected inventory type. |
| Key gone after leaving the tab open | Idle timeout cleared the session. Sign in and verify again. |

Error messages are sanitized before display: upstream text is passed through
redaction, so a key echoed back by an API cannot surface in the UI or the logs.
