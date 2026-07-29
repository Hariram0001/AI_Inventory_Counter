"""Unified Model Catalog: workspace sync, foundation, and approved public models.

Uses Roboflow REST management APIs (api.roboflow.com) for discovery and
inference-sdk / existing adapters for execution. Never stores API keys.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests

from config import (
    DEMO_MODE,
    MODELS_JSON_PATH,
    PROJECT_ROOT,
    ROBOFLOW_API_KEY,
    api_key_configured,
)
from model_registry import (
    MODEL_NAME_ALIASES,
    get_enabled_valid_models,
    load_models_from_file,
    load_models_raw,
    normalize_model_name,
    save_models_to_file,
    sanitize_selected_model_names,
)
from schemas import ModelConfig

CATALOG_PATH = PROJECT_ROOT / "data" / "model_catalog.json"
PUBLIC_MODELS_PATH = PROJECT_ROOT / "data" / "approved_public_models.json"
SYNC_REPORT_PATH = PROJECT_ROOT / "data" / "catalog_sync_report.json"
ROBOFLOW_MANAGEMENT_API = "https://api.roboflow.com"
DEFAULT_WORKSPACE = "hariram-s-mzhvc"

SOURCE_WORKSPACE = "workspace"
SOURCE_FOUNDATION = "foundation"
SOURCE_UNIVERSE = "universe"
SOURCE_DEMO = "demo"
SOURCE_LOCAL = "local"

STATUS_READY = "ready"
STATUS_NEEDS_CONFIG = "needs_configuration"
STATUS_FAILED = "failed_validation"
STATUS_UNAVAILABLE = "unavailable"
STATUS_STALE = "stale"
STATUS_METADATA_ONLY = "metadata_only"

# Fence-related class tokens for fixed-class compatibility heuristics
FENCE_CLASS_HINTS = {
    "fence",
    "fence-panel",
    "fence_panel",
    "fence panel",
    "panel",
    "wood fence",
    "wooden fence",
    "picket",
    "privacy",
}


@dataclass
class CatalogEntry:
    key: str
    display_name: str
    source: str
    provider: str = "roboflow"
    task_type: str = "object_detection"
    adapter_type: str = "roboflow_model"
    workspace: str | None = None
    project_id: str | None = None
    version: str | int | None = None
    model_id: str | None = None
    workflow_id: str | None = None
    enabled: bool = False
    validated: bool = False
    demo_only: bool = False
    dynamic_classes: bool = False
    supported_classes: list[str] = field(default_factory=list)
    supported_inventory_types: list[str] = field(default_factory=list)
    architecture: str | None = None
    last_tested_at: str | None = None
    last_test_status: str | None = None
    # Extended / preserved fields
    kind: str = "model"
    status: str = STATUS_NEEDS_CONFIG
    deployment: str | None = None
    is_default: bool = False
    default_confidence: float | None = 0.25
    default_iou: float | None = 0.5
    supports_prompt: bool = False
    image_input_name: str = "image"
    prompt_parameter_name: str = "classes"
    counting_strategy: str | None = "Object Detection"
    annotation_support: bool = True
    segmentation_support: bool = False
    license: str | None = None
    stale: bool = False
    sync_note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        # Flatten unknown preserved fields without overwriting known keys
        for k, v in extra.items():
            if k not in d:
                d[k] = v
        d.pop("api_key", None)
        d.pop("ROBOFLOW_API_KEY", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogEntry":
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        core = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = {k: v for k, v in data.items() if k not in known and k != "extra"}
        if isinstance(data.get("extra"), dict):
            extra.update(data["extra"])
        core.setdefault("key", data.get("key") or data.get("name") or "unknown")
        core.setdefault("display_name", data.get("display_name") or data.get("name") or core["key"])
        core["extra"] = extra
        # Coerce lists
        for list_key in ("supported_classes", "supported_inventory_types"):
            if list_key in core and core[list_key] is None:
                core[list_key] = []
        return cls(**core)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            name=self.display_name,
            kind=self.kind,
            enabled=bool(self.enabled) and not self.stale,
            model_id=self.model_id,
            workspace_name=self.workspace,
            workflow_id=self.workflow_id,
            image_input_name=self.image_input_name or "image",
            prompt_parameter_name=self.prompt_parameter_name or "classes",
            supported_inventory_types=list(self.supported_inventory_types or []),
            allowed_classes=list(self.supported_classes or []),
            supports_prompt=bool(self.supports_prompt or self.dynamic_classes),
            key=self.key,
            provider=self.provider,
            is_default=bool(self.is_default),
            default_confidence=self.default_confidence,
            default_iou=self.default_iou,
            counting_strategy=self.counting_strategy,
            annotation_support=bool(self.annotation_support),
            segmentation_support=bool(self.segmentation_support),
            demo_only=bool(self.demo_only),
            dynamic_classes=bool(self.dynamic_classes),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_error(msg: str | None) -> str:
    if not msg:
        return ""
    text = str(msg)
    # Redact key-like query params / bearer tokens
    text = re.sub(r"(api_key=)[^&\s]+", r"\1***", text, flags=re.I)
    text = re.sub(r"(Bearer\s+)\S+", r"\1***", text, flags=re.I)
    text = re.sub(r"[A-Za-z0-9]{32,}", "***", text)
    return text[:500]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    json.loads(text)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent), suffix=".tmp"
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


# --- Foundation models (verified adapters only as Ready) -----------------------

def load_registered_foundation_models() -> list[CatalogEntry]:
    """Explicit foundation/base models this app can genuinely execute or list.

    Only YOLO-World is Ready — verified via Workflow + inference-sdk.
    Other architectures are listed as Deployment unavailable without inventing IDs.
    """
    return [
        CatalogEntry(
            key="workflow:hariram-s-mzhvc/custom-workflow",
            display_name="YOLO-World",
            source=SOURCE_FOUNDATION,
            provider="roboflow",
            task_type="object_detection",
            adapter_type="roboflow_workflow",
            workspace=DEFAULT_WORKSPACE,
            workflow_id="custom-workflow",
            enabled=True,
            validated=True,
            demo_only=False,
            dynamic_classes=True,
            supports_prompt=True,
            supported_classes=[],
            supported_inventory_types=[],
            architecture="YOLO-World",
            kind="workflow",
            status=STATUS_READY,
            deployment="serverless_hosted_api + workflow",
            is_default=True,
            prompt_parameter_name="class_names",
            image_input_name="image",
            last_test_status="verified_workflow",
        ),
        # Known Roboflow families — not Ready until a real adapter/workflow exists
        CatalogEntry(
            key="foundation:rf-detr",
            display_name="RF-DETR",
            source=SOURCE_FOUNDATION,
            task_type="object_detection",
            adapter_type="none",
            architecture="RF-DETR",
            kind="model",
            status=STATUS_UNAVAILABLE,
            deployment="Deployment unavailable",
            enabled=False,
            validated=False,
            sync_note="No verified adapter or Workflow in this POC.",
        ),
        CatalogEntry(
            key="foundation:yolo11",
            display_name="YOLO11",
            source=SOURCE_FOUNDATION,
            task_type="object_detection",
            adapter_type="none",
            architecture="YOLO11",
            kind="model",
            status=STATUS_UNAVAILABLE,
            deployment="Deployment unavailable",
            enabled=False,
            validated=False,
            sync_note="No verified adapter or Workflow in this POC.",
        ),
        CatalogEntry(
            key="foundation:yolo26",
            display_name="YOLO26",
            source=SOURCE_FOUNDATION,
            task_type="object_detection",
            adapter_type="none",
            architecture="YOLO26",
            kind="model",
            status=STATUS_UNAVAILABLE,
            deployment="Deployment unavailable",
            enabled=False,
            validated=False,
            sync_note="No verified adapter or Workflow in this POC.",
        ),
        CatalogEntry(
            key="foundation:sam3",
            display_name="SAM3",
            source=SOURCE_FOUNDATION,
            task_type="instance_segmentation",
            adapter_type="none",
            architecture="SAM3",
            kind="model",
            status=STATUS_UNAVAILABLE,
            deployment="Deployment unavailable",
            enabled=False,
            validated=False,
            sync_note="No verified adapter or Workflow in this POC.",
        ),
        CatalogEntry(
            key="foundation:florence-2",
            display_name="Florence 2",
            source=SOURCE_FOUNDATION,
            task_type="multimodal",
            adapter_type="none",
            architecture="Florence-2",
            kind="model",
            status=STATUS_UNAVAILABLE,
            deployment="Deployment unavailable",
            enabled=False,
            validated=False,
            sync_note="No verified adapter or Workflow in this POC.",
        ),
    ]


def load_local_demo_catalog_entries() -> list[CatalogEntry]:
    """Local classical counter + demo fixtures.

    Local Picket Counter is a real optional local tool (selectable in Analysis).
    Demo Fence Detector remains demo-only and is excluded when DEMO_MODE=false.
    """
    return [
        CatalogEntry(
            key="local:local-picket-counter",
            display_name="Local Picket Counter",
            source=SOURCE_LOCAL,
            provider="local",
            task_type="custom_counting",
            adapter_type="local_classical",
            model_id="local-picket-counter",
            enabled=True,
            validated=True,
            demo_only=False,
            dynamic_classes=False,
            supported_classes=["fence-picket"],
            supported_inventory_types=["Fence Panel"],
            architecture="Classical tip-peak (NumPy/PIL)",
            kind="local",
            status=STATUS_READY,
            deployment="local_only",
            sync_note=(
                "Classical picket heuristic in picket_counter.py — not Roboflow. "
                "Optional for pointed (dog-ear) pickets; no API key required."
            ),
        ),
        CatalogEntry(
            key="model:demo-fence-panels/1",
            display_name="Demo Fence Detector",
            source=SOURCE_DEMO,
            provider="demo",
            task_type="object_detection",
            adapter_type="demo_fixture",
            model_id="demo-fence-panels/1",
            enabled=False,
            validated=False,
            demo_only=True,
            supported_classes=[
                "fence-panel",
                "fence",
                "panel",
                "pole",
                "gate",
                "clamp",
                "sandbag",
                "chain-link-roll",
            ],
            supported_inventory_types=["Fence Panel"],
            architecture="Fixture",
            kind="model",
            status=STATUS_METADATA_ONLY,
            deployment="demo_mock",
            sync_note="Fixture-based demo only.",
        ),
    ]


# --- Public / Universe ---------------------------------------------------------

def load_approved_public_models() -> list[CatalogEntry]:
    raw = _read_json(PUBLIC_MODELS_PATH, [])
    if not isinstance(raw, list):
        return []
    out: list[CatalogEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = CatalogEntry.from_dict(item)
        entry.source = SOURCE_UNIVERSE
        out.append(entry)
    return out


def save_approved_public_models(entries: list[CatalogEntry]) -> None:
    _write_json(PUBLIC_MODELS_PATH, [e.to_dict() for e in entries])


def validate_universe_model_id(model_id: str) -> tuple[bool, str, dict[str, Any]]:
    """Validate a Universe/public model ID via management API (no invented IDs)."""
    mid = (model_id or "").strip()
    if not mid or "/" not in mid:
        return False, "Model ID must look like 'project-slug/version'.", {}
    if "replace-with" in mid.lower() or mid.startswith("demo-"):
        return False, "Invalid or demo model ID.", {}
    if not api_key_configured():
        return False, "API key not configured.", {}

    # project/version — fetch project then version
    parts = mid.split("/")
    if len(parts) < 2:
        return False, "Model ID must include project and version.", {}
    project_slug, version = parts[0], parts[1]
    # Prefer workspace-scoped lookup; Universe models may use owner/project/version
    key = ROBOFLOW_API_KEY
    try:
        # Root auth
        root = requests.get(
            f"{ROBOFLOW_MANAGEMENT_API}/",
            params={"api_key": key},
            timeout=30,
        )
        if root.status_code != 200:
            return False, _sanitize_error(f"Auth failed ({root.status_code})"), {}
        workspace = (root.json() or {}).get("workspace") or DEFAULT_WORKSPACE

        # Try workspace/project
        proj_url = f"{ROBOFLOW_MANAGEMENT_API}/{quote(workspace)}/{quote(project_slug)}"
        resp = requests.get(proj_url, params={"api_key": key}, timeout=30)
        meta: dict[str, Any] = {}
        if resp.status_code == 200:
            meta = resp.json() or {}
            versions = meta.get("versions") or meta.get("project", {}).get("versions") or []
            # versions may be list of dicts
            ver_ok = False
            classes: list[str] = []
            proj = meta.get("project") or meta
            if isinstance(proj.get("classes"), dict):
                classes = list(proj["classes"].keys())
            task = str(proj.get("type") or "object-detection")
            if isinstance(versions, list):
                for v in versions:
                    if isinstance(v, dict):
                        vid = str(v.get("id") or v.get("name") or v.get("version") or "")
                        if vid.endswith(f"/{version}") or str(v.get("version")) == str(version):
                            ver_ok = bool(v.get("model") or v.get("models") or v.get("trained"))
                            break
                    elif str(v) == str(version):
                        ver_ok = True
            # Fetch version detail
            ver_url = f"{proj_url}/{quote(str(version))}"
            vresp = requests.get(ver_url, params={"api_key": key}, timeout=30)
            if vresp.status_code == 200:
                vmeta = vresp.json() or {}
                model_block = vmeta.get("model") or vmeta.get("version", {}).get("model")
                ver_ok = bool(model_block) or ver_ok
                if not classes and isinstance(vmeta.get("classes"), dict):
                    classes = list(vmeta["classes"].keys())
            if not ver_ok and vresp.status_code != 200:
                # Try as universe-style id without inventing — report failure honestly
                return (
                    False,
                    _sanitize_error(
                        f"Could not confirm trained model for {mid} "
                        f"(project HTTP {resp.status_code}, version HTTP {vresp.status_code})."
                    ),
                    {"workspace": workspace, "project": project_slug},
                )
            return True, "Validated against Roboflow management API.", {
                "workspace": workspace,
                "project": project_slug,
                "version": version,
                "classes": classes,
                "task_type": _normalize_task(task),
                "has_model": ver_ok,
            }

        # Fallback: try direct model id path used by some Universe entries
        # workspace may differ — attempt GET /:owner/:project/:version if 3-part
        if len(parts) >= 3:
            owner, project_slug, version = parts[0], parts[1], parts[2]
            ver_url = (
                f"{ROBOFLOW_MANAGEMENT_API}/{quote(owner)}/"
                f"{quote(project_slug)}/{quote(version)}"
            )
            vresp = requests.get(ver_url, params={"api_key": key}, timeout=30)
            if vresp.status_code == 200:
                vmeta = vresp.json() or {}
                model_block = vmeta.get("model")
                if not model_block:
                    return False, "Version exists but no trained model is available.", {}
                return True, "Validated Universe model.", {
                    "workspace": owner,
                    "project": project_slug,
                    "version": version,
                    "classes": list((vmeta.get("classes") or {}).keys())
                    if isinstance(vmeta.get("classes"), dict)
                    else [],
                    "task_type": "object_detection",
                    "has_model": True,
                }
        return False, _sanitize_error(
            f"Project not found in workspace ({resp.status_code})."
        ), {}
    except requests.RequestException as exc:
        return False, _sanitize_error(str(exc)), {}


def add_approved_public_model(
    *,
    model_id: str,
    display_name: str,
    task_type: str = "object_detection",
    supported_classes: list[str] | None = None,
    supported_inventory_types: list[str] | None = None,
    license_info: str | None = None,
    require_live_validation: bool = True,
) -> tuple[bool, str, CatalogEntry | None]:
    ok, msg, meta = (True, "Skipped live validation.", {})
    if require_live_validation:
        ok, msg, meta = validate_universe_model_id(model_id)
    if not ok:
        return False, msg, None
    if meta.get("has_model") is False:
        return False, "Version has no usable trained model.", None

    classes = list(supported_classes or meta.get("classes") or [])
    inv = list(supported_inventory_types or [])
    if not inv and _classes_compatible_with_fence(classes):
        inv = ["Fence Panel"]

    entry = CatalogEntry(
        key=f"model:{model_id.strip()}",
        display_name=(display_name or model_id).strip(),
        source=SOURCE_UNIVERSE,
        provider="roboflow",
        task_type=_normalize_task(task_type or meta.get("task_type") or "object_detection"),
        adapter_type="roboflow_model",
        workspace=meta.get("workspace"),
        project_id=meta.get("project"),
        version=meta.get("version"),
        model_id=model_id.strip(),
        enabled=True,
        validated=bool(require_live_validation and ok),
        demo_only=False,
        dynamic_classes=False,
        supported_classes=classes,
        supported_inventory_types=inv,
        kind="model",
        status=STATUS_READY if (require_live_validation and ok) else STATUS_METADATA_ONLY,
        deployment="serverless_hosted_api",
        license=license_info,
        last_tested_at=_utc_now() if require_live_validation else None,
        last_test_status="validated" if ok else None,
    )
    existing = [e for e in load_approved_public_models() if e.key != entry.key]
    existing.append(entry)
    save_approved_public_models(existing)
    _merge_entry_into_models_json(entry)
    return True, msg, entry


# --- Workspace sync ------------------------------------------------------------

def _normalize_task(raw: str | None) -> str:
    t = (raw or "object-detection").lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "object_detection": "object_detection",
        "objectdetection": "object_detection",
        "instance_segmentation": "instance_segmentation",
        "instancesegmentation": "instance_segmentation",
        "classification": "classification",
        "semantic_segmentation": "instance_segmentation",
        "keypoint_detection": "object_detection",
        "multimodal": "multimodal",
    }
    return mapping.get(t, t)


def _classes_compatible_with_fence(classes: Iterable[str]) -> bool:
    for c in classes:
        token = str(c).lower().replace("_", " ").replace("-", " ")
        for hint in FENCE_CLASS_HINTS:
            if hint in token or token in hint:
                return True
    return False


def _version_has_trained_model(version_payload: dict[str, Any]) -> bool:
    if not isinstance(version_payload, dict):
        return False
    if version_payload.get("model"):
        return True
    if version_payload.get("models"):
        return True
    # Some payloads nest under "version"
    ver = version_payload.get("version")
    if isinstance(ver, dict) and (ver.get("model") or ver.get("trained")):
        return True
    train = version_payload.get("train") or {}
    if isinstance(train, dict) and train.get("status") in {"finished", "success", "trained"}:
        # Still require a model block when possible
        return bool(version_payload.get("model") or version_payload.get("models"))
    return False


def fetch_workspace_projects(
    workspace: str = DEFAULT_WORKSPACE,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """GET /:workspace — returns project list + sanitized report meta."""
    key = api_key or ROBOFLOW_API_KEY
    report: dict[str, Any] = {"workspace": workspace, "ok": False, "error": None}
    if not key:
        report["error"] = "API key not configured."
        return [], report
    sess = session or requests.Session()
    url = f"{ROBOFLOW_MANAGEMENT_API}/{quote(workspace)}"
    try:
        resp = sess.get(url, params={"api_key": key}, timeout=45)
        if resp.status_code != 200:
            report["error"] = _sanitize_error(f"HTTP {resp.status_code}")
            return [], report
        payload = resp.json() or {}
        ws = payload.get("workspace") or payload
        projects = ws.get("projects") if isinstance(ws, dict) else None
        if projects is None and isinstance(payload.get("projects"), list):
            projects = payload["projects"]
        if not isinstance(projects, list):
            projects = []
        report["ok"] = True
        report["project_count"] = len(projects)
        return projects, report
    except requests.RequestException as exc:
        report["error"] = _sanitize_error(str(exc))
        return [], report


def fetch_project_versions(
    workspace: str,
    project_slug: str,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    key = api_key or ROBOFLOW_API_KEY
    sess = session or requests.Session()
    url = f"{ROBOFLOW_MANAGEMENT_API}/{quote(workspace)}/{quote(project_slug)}"
    try:
        resp = sess.get(url, params={"api_key": key}, timeout=45)
        if resp.status_code != 200:
            return {}, [], _sanitize_error(f"HTTP {resp.status_code}")
        payload = resp.json() or {}
        project = payload.get("project") or payload
        versions = payload.get("versions") or project.get("versions") or []
        if not isinstance(versions, list):
            versions = []
        # Normalize version entries to dicts when API returns ids only
        norm: list[dict[str, Any]] = []
        for v in versions:
            if isinstance(v, dict):
                norm.append(v)
            else:
                # Need detail fetch
                vid = str(v)
                ver_num = vid.split("/")[-1] if "/" in vid else vid
                vurl = f"{url}/{quote(ver_num)}"
                vresp = sess.get(vurl, params={"api_key": key}, timeout=45)
                if vresp.status_code == 200:
                    norm.append(vresp.json() or {"version": ver_num})
                else:
                    norm.append({"version": ver_num, "id": vid, "_unresolved": True})
        return project if isinstance(project, dict) else {}, norm, None
    except requests.RequestException as exc:
        return {}, [], _sanitize_error(str(exc))


def normalize_workspace_version(
    workspace: str,
    project: dict[str, Any],
    version: dict[str, Any],
) -> CatalogEntry | None:
    """Build a catalog entry only when a usable trained model exists."""
    if version.get("_unresolved"):
        return None
    inner = version.get("version") if isinstance(version.get("version"), dict) else None
    has_model = _version_has_trained_model(version) or (
        isinstance(inner, dict) and _version_has_trained_model(inner)
    )
    if not has_model:
        return None

    proj_id = str(project.get("id") or project.get("url") or "")
    proj_name = str(project.get("name") or proj_id or "Project")
    slug = proj_id.split("/")[-1] if "/" in proj_id else str(project.get("url") or proj_name)
    slug = slug.strip() or re.sub(r"[^a-z0-9-]+", "-", proj_name.lower()).strip("-")

    ver_id = version.get("id")
    if not ver_id and isinstance(inner, dict):
        ver_id = inner.get("id")
    ver_num = None
    if isinstance(ver_id, str) and "/" in ver_id:
        ver_num = ver_id.split("/")[-1]
    else:
        ver_num = version.get("version")
        if ver_num is None and isinstance(inner, dict):
            ver_num = inner.get("version")
        if ver_num is None:
            ver_num = ver_id
    if ver_num is None:
        return None
    ver_num = str(ver_num)

    model_id = f"{slug}/{ver_num}"
    classes_raw = project.get("classes") or version.get("classes") or {}
    if isinstance(classes_raw, dict):
        classes = list(classes_raw.keys())
    elif isinstance(classes_raw, list):
        classes = [str(c) for c in classes_raw]
    else:
        classes = []

    task = _normalize_task(str(project.get("type") or "object-detection"))
    inv: list[str] = []
    if _classes_compatible_with_fence(classes):
        inv = ["Fence Panel"]

    model_block = version.get("model") or (inner or {}).get("model") or {}
    arch = None
    if isinstance(model_block, dict):
        arch = (
            model_block.get("type")
            or model_block.get("modelType")
            or model_block.get("architecture")
        )

    updated = project.get("updated") or version.get("updated")
    updated_s = None
    if isinstance(updated, (int, float)):
        try:
            updated_s = datetime.fromtimestamp(float(updated), tz=timezone.utc).isoformat()
        except (OSError, ValueError, OverflowError):
            updated_s = str(updated)

    return CatalogEntry(
        key=f"model:{model_id}",
        display_name=f"{proj_name} v{ver_num}",
        source=SOURCE_WORKSPACE,
        provider="roboflow",
        task_type=task,
        adapter_type="roboflow_model",
        workspace=workspace,
        project_id=slug,
        version=ver_num,
        model_id=model_id,
        enabled=False,
        validated=False,
        demo_only=False,
        dynamic_classes=False,
        supported_classes=classes,
        supported_inventory_types=inv,
        architecture=str(arch) if arch else None,
        kind="model",
        status=STATUS_NEEDS_CONFIG,
        deployment="serverless_hosted_api",
        sync_note=f"Discovered from workspace {workspace}. Enable and Test before use.",
        extra={"project_name": proj_name, "project_updated": updated_s},
    )


def fetch_workspace_workflows(
    workspace: str = DEFAULT_WORKSPACE,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """GET /:workspace/workflows — list deployed Workflows (not OD projects)."""
    key = api_key or ROBOFLOW_API_KEY
    if not key:
        return [], "API key not configured."
    sess = session or requests.Session()
    url = f"{ROBOFLOW_MANAGEMENT_API}/{quote(workspace)}/workflows"
    try:
        resp = sess.get(url, params={"api_key": key}, timeout=45)
        if resp.status_code != 200:
            return [], _sanitize_error(f"HTTP {resp.status_code}")
        payload = resp.json() or {}
        workflows = payload.get("workflows") or []
        if not isinstance(workflows, list):
            return [], "Unexpected workflows payload."
        return [w for w in workflows if isinstance(w, dict)], None
    except requests.RequestException as exc:
        return [], _sanitize_error(str(exc))


def normalize_workspace_workflow(
    workspace: str, workflow: dict[str, Any]
) -> CatalogEntry:
    url_slug = str(workflow.get("url") or workflow.get("id") or "").strip()
    name = str(workflow.get("name") or url_slug or "Workflow").strip()
    # Prefer friendly foundation name for the verified YOLO-World workflow
    display = "YOLO-World" if url_slug == "custom-workflow" else name
    is_known = url_slug == "custom-workflow"
    return CatalogEntry(
        key=f"workflow:{workspace}/{url_slug}",
        display_name=display,
        source=SOURCE_WORKSPACE if not is_known else SOURCE_FOUNDATION,
        provider="roboflow",
        task_type="object_detection",
        adapter_type="roboflow_workflow",
        workspace=workspace,
        workflow_id=url_slug,
        enabled=is_known,
        validated=is_known,
        demo_only=False,
        dynamic_classes=is_known,
        supports_prompt=is_known,
        prompt_parameter_name="class_names" if is_known else "classes",
        image_input_name="image",
        kind="workflow",
        status=STATUS_READY if is_known else STATUS_NEEDS_CONFIG,
        deployment="serverless_hosted_api + workflow",
        architecture="YOLO-World" if is_known else None,
        is_default=is_known,
        sync_note=(
            "Verified Workflow used by this app (dynamic class injection)."
            if is_known
            else "Discovered workspace Workflow — Test before enabling for Analysis."
        ),
        extra={"workflow_remote_id": workflow.get("id"), "workflow_name": name},
    )


def sync_workspace_models(
    workspace: str = DEFAULT_WORKSPACE,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Discover trained workspace models and merge into the catalog.

    Preserves local Enabled / inventory compatibility for matching keys.
    Marks previously synced workspace entries missing from this sync as stale.
    """
    report: dict[str, Any] = {
        "workspace": workspace,
        "started_at": _utc_now(),
        "ok": False,
        "projects_found": 0,
        "versions_found": 0,
        "models_registered": 0,
        "workflows_found": 0,
        "workflows_registered": 0,
        "versions_skipped_unusable": 0,
        "errors": [],
        "method": (
            "GET api.roboflow.com/:workspace, /:workspace/:project, "
            "and /:workspace/workflows"
        ),
    }
    projects, meta = fetch_workspace_projects(workspace, api_key=api_key, session=session)
    if not meta.get("ok"):
        report["errors"].append(meta.get("error") or "Workspace fetch failed.")
        report["finished_at"] = _utc_now()
        if persist:
            _write_json(SYNC_REPORT_PATH, report)
        return report

    report["projects_found"] = len(projects)
    previous = {e.key: e for e in load_catalog_entries()}
    discovered: list[CatalogEntry] = []
    sess = session or requests.Session()

    # Workflows (this workspace currently has Workflows even when OD projects=0)
    workflows, wf_err = fetch_workspace_workflows(
        workspace, api_key=api_key, session=sess
    )
    if wf_err:
        report["errors"].append(f"workflows: {wf_err}")
    else:
        report["workflows_found"] = len(workflows)
        for wf in workflows:
            entry = normalize_workspace_workflow(workspace, wf)
            prev = previous.get(entry.key)
            if prev and entry.workflow_id != "custom-workflow":
                entry.enabled = prev.enabled
                entry.validated = prev.validated
                entry.last_tested_at = prev.last_tested_at
                entry.last_test_status = prev.last_test_status
                if prev.validated:
                    entry.status = STATUS_READY
            discovered.append(entry)
        report["workflows_registered"] = len(workflows)

    for proj_summary in projects:
        if not isinstance(proj_summary, dict):
            continue
        proj_id = str(proj_summary.get("id") or "")
        slug = proj_id.split("/")[-1] if proj_id else str(proj_summary.get("url") or "")
        if not slug:
            continue
        project, versions, err = fetch_project_versions(
            workspace, slug, api_key=api_key, session=sess
        )
        if err:
            report["errors"].append(_sanitize_error(f"{slug}: {err}"))
            # Do not delete existing entries on partial failure
            continue
        if not project:
            project = proj_summary
        report["versions_found"] += len(versions)
        for ver in versions:
            # Enrich version detail when model field missing
            if isinstance(ver, dict) and not ver.get("model") and not ver.get("_unresolved"):
                ver_num = None
                vid = ver.get("id")
                if isinstance(vid, str) and "/" in vid:
                    ver_num = vid.split("/")[-1]
                else:
                    ver_num = ver.get("version")
                if ver_num is not None:
                    vurl = (
                        f"{ROBOFLOW_MANAGEMENT_API}/{quote(workspace)}/"
                        f"{quote(slug)}/{quote(str(ver_num))}"
                    )
                    try:
                        key = api_key or ROBOFLOW_API_KEY
                        vresp = sess.get(vurl, params={"api_key": key}, timeout=45)
                        if vresp.status_code == 200:
                            detail = vresp.json() or {}
                            if isinstance(detail, dict):
                                ver = {**ver, **detail}
                    except requests.RequestException:
                        pass
            entry = normalize_workspace_version(workspace, project, ver)
            if entry is None:
                report["versions_skipped_unusable"] += 1
                continue
            # Preserve local settings
            prev = previous.get(entry.key)
            if prev:
                entry.enabled = prev.enabled
                entry.supported_inventory_types = list(
                    prev.supported_inventory_types or entry.supported_inventory_types
                )
                entry.validated = prev.validated
                entry.last_tested_at = prev.last_tested_at
                entry.last_test_status = prev.last_test_status
                if prev.validated and prev.last_test_status in {"OK", "validated", "ready"}:
                    entry.status = STATUS_READY
            discovered.append(entry)

    report["models_registered"] = sum(
        1 for e in discovered if e.kind == "model"
    )

    # Merge: keep non-workspace entries; update workspace; mark missing workspace stale
    others = [
        e
        for e in previous.values()
        if e.source not in {SOURCE_WORKSPACE}
        and not (
            e.source == SOURCE_FOUNDATION
            and e.kind == "workflow"
            and e.workflow_id == "custom-workflow"
        )
    ]
    discovered_keys = {e.key for e in discovered}
    for key, prev in previous.items():
        if prev.source == SOURCE_WORKSPACE and key not in discovered_keys:
            stale = CatalogEntry.from_dict(prev.to_dict())
            stale.stale = True
            stale.status = STATUS_STALE
            stale.sync_note = "Unavailable during last sync"
            others.append(stale)

    merged = others + discovered
    # Deduplicate by key (prefer newer discovered)
    by_key: dict[str, CatalogEntry] = {}
    for e in merged:
        by_key[e.key] = e

    # Ensure foundation YOLO-World + local/demo metadata always present
    for base in load_registered_foundation_models() + load_local_demo_catalog_entries():
        if base.key not in by_key:
            by_key[base.key] = base
        elif base.source == SOURCE_FOUNDATION and base.status == STATUS_READY:
            # Preserve user enabled/default from models.json merge later
            cur = by_key[base.key]
            if cur.source != SOURCE_FOUNDATION:
                pass
            else:
                cur.architecture = cur.architecture or base.architecture
                if base.status == STATUS_READY and cur.status in {STATUS_NEEDS_CONFIG, STATUS_STALE}:
                    if cur.validated or cur.enabled:
                        cur.status = STATUS_READY

    # Overlay models.json enabled/default flags
    for m in load_models_from_file():
        mk = m.key or f"{m.kind}:{m.model_id or m.workflow_id or m.name}"
        from model_adapters import model_key as mk_fn

        mk = m.key or mk_fn(m)
        if mk in by_key:
            by_key[mk].enabled = bool(m.enabled)
            by_key[mk].is_default = bool(m.is_default)
            by_key[mk].display_name = normalize_model_name(m.name) or by_key[mk].display_name
            if m.allowed_classes:
                by_key[mk].supported_classes = list(m.allowed_classes)
            if m.supported_inventory_types:
                by_key[mk].supported_inventory_types = list(m.supported_inventory_types)
            by_key[mk].dynamic_classes = bool(m.dynamic_classes)
            by_key[mk].supports_prompt = bool(m.supports_prompt)
            by_key[mk].demo_only = bool(m.demo_only)

    entries = list(by_key.values())
    report["ok"] = True
    report["finished_at"] = _utc_now()
    report["catalog_size"] = len(entries)
    # Never include api key
    report.pop("api_key", None)

    if persist:
        save_catalog_entries(entries)
        _write_json(SYNC_REPORT_PATH, report)
        # Merge executable workspace+foundation+public into models.json carefully
        _sync_models_json_from_catalog(entries)
    return report


def _merge_entry_into_models_json(entry: CatalogEntry) -> None:
    if entry.adapter_type in {"none", "demo_fixture"} and entry.source == SOURCE_FOUNDATION:
        if entry.status != STATUS_READY:
            return
    if entry.demo_only and entry.source in {SOURCE_DEMO, SOURCE_LOCAL}:
        # Keep demo/local in models.json for DEMO_MODE but disabled
        pass
    models = load_models_from_file()
    cfg = entry.to_model_config()
    replaced = False
    out: list[ModelConfig] = []
    for m in models:
        from model_adapters import model_key as mk_fn

        if (m.key or mk_fn(m)) == entry.key or m.name == cfg.name:
            # Preserve unknown fields via save_models_to_file name match
            cfg.enabled = m.enabled if entry.source == SOURCE_WORKSPACE else cfg.enabled
            out.append(cfg)
            replaced = True
        else:
            out.append(m)
    if not replaced and entry.status in {STATUS_READY, STATUS_NEEDS_CONFIG} and entry.kind in {
        "model",
        "workflow",
        "local",
    }:
        if entry.adapter_type != "none":
            out.append(cfg)
    save_models_to_file(out)


def _sync_models_json_from_catalog(entries: list[CatalogEntry]) -> None:
    """Update models.json for executable catalog entries; preserve unknowns."""
    existing_raw = {str(d.get("name")): d for d in load_models_raw()}
    from model_adapters import model_key as mk_fn

    existing_by_key: dict[str, ModelConfig] = {}
    for m in load_models_from_file():
        existing_by_key[m.key or mk_fn(m)] = m

    # Start from current models.json order
    out_cfgs: list[ModelConfig] = []
    seen: set[str] = set()

    # Always keep YOLO-World foundation ready entry first if present
    for e in entries:
        if e.adapter_type == "none":
            continue
        if e.source == SOURCE_FOUNDATION and e.status != STATUS_READY:
            continue
        cfg = e.to_model_config()
        key = e.key
        prev = existing_by_key.get(key)
        if prev:
            # Preserve enable unless stale
            if e.stale:
                cfg.enabled = False
            else:
                cfg.enabled = prev.enabled if e.source == SOURCE_WORKSPACE else e.enabled
            cfg.is_default = prev.is_default or e.is_default
        else:
            if e.source == SOURCE_WORKSPACE:
                cfg.enabled = False
        # Demo fixtures only: keep out of live selector when DEMO_MODE=false.
        # Local Picket Counter is a real optional local tool — keep enabled as catalog says.
        if e.source == SOURCE_DEMO or (e.demo_only and e.source != SOURCE_LOCAL):
            cfg.demo_only = True
            if not DEMO_MODE:
                cfg.enabled = False
        if e.source == SOURCE_LOCAL or e.kind == "local":
            cfg.demo_only = False
            cfg.enabled = bool(e.enabled) and not e.stale
        out_cfgs.append(cfg)
        seen.add(key)

    # Preserve any models.json entries not represented (custom)
    for m in load_models_from_file():
        key = m.key or mk_fn(m)
        if key not in seen:
            # Rename alias
            m.name = normalize_model_name(m.name) or m.name
            if m.is_demo_model_id() or (m.demo_only and (m.kind or "").lower() != "local"):
                m.demo_only = True
                if not DEMO_MODE:
                    m.enabled = False
            out_cfgs.append(m)

    save_models_to_file(out_cfgs)
    # Preserve unknown fields already handled by save_models_to_file


# --- Catalog load / query ------------------------------------------------------

def load_catalog_entries() -> list[CatalogEntry]:
    raw = _read_json(CATALOG_PATH, None)
    if isinstance(raw, dict) and isinstance(raw.get("models"), list):
        items = raw["models"]
    elif isinstance(raw, list):
        items = raw
    else:
        # Bootstrap from foundation + models.json + public + local
        items = []
        boot: list[CatalogEntry] = []
        boot.extend(load_registered_foundation_models())
        boot.extend(load_local_demo_catalog_entries())
        boot.extend(load_approved_public_models())
        for m in load_models_from_file():
            from model_adapters import model_key as mk_fn

            key = m.key or mk_fn(m)
            if any(b.key == key for b in boot):
                # Overlay flags
                for b in boot:
                    if b.key == key:
                        b.enabled = m.enabled
                        b.is_default = m.is_default
                        b.demo_only = m.demo_only or b.demo_only
                        b.display_name = normalize_model_name(m.name) or b.display_name
                        if m.dynamic_classes:
                            b.dynamic_classes = True
                            b.supports_prompt = True
                            b.status = STATUS_READY if m.enabled else b.status
                        break
                continue
            source = SOURCE_WORKSPACE
            if m.demo_only or m.is_demo_model_id():
                source = SOURCE_DEMO
            elif (m.kind or "").lower() == "local":
                source = SOURCE_LOCAL
            elif (m.kind or "").lower() == "workflow":
                source = SOURCE_FOUNDATION
            boot.append(
                CatalogEntry(
                    key=key,
                    display_name=normalize_model_name(m.name) or m.name,
                    source=source,
                    provider=(m.provider or "roboflow").lower(),
                    task_type="object_detection",
                    adapter_type=(
                        "roboflow_workflow"
                        if m.kind == "workflow"
                        else ("local_classical" if m.kind == "local" else "roboflow_model")
                    ),
                    workspace=m.workspace_name,
                    workflow_id=m.workflow_id,
                    model_id=m.model_id,
                    enabled=m.enabled,
                    validated=bool(m.kind == "workflow" and m.enabled),
                    demo_only=m.demo_only or m.is_demo_model_id(),
                    dynamic_classes=m.dynamic_classes,
                    supports_prompt=m.supports_prompt,
                    supported_classes=list(m.allowed_classes or []),
                    supported_inventory_types=list(m.supported_inventory_types or []),
                    kind=m.kind,
                    status=STATUS_READY if (m.enabled and m.is_valid(allow_demo_ids=DEMO_MODE)) else STATUS_NEEDS_CONFIG,
                    is_default=m.is_default,
                    prompt_parameter_name=m.prompt_parameter_name,
                    image_input_name=m.image_input_name,
                )
            )
        save_catalog_entries(boot)
        return boot

    return [CatalogEntry.from_dict(i) for i in items if isinstance(i, dict)]


def save_catalog_entries(entries: list[CatalogEntry]) -> None:
    payload = {
        "updated_at": _utc_now(),
        "models": [e.to_dict() for e in entries],
    }
    _write_json(CATALOG_PATH, payload)


def get_all_catalog_models() -> list[CatalogEntry]:
    return load_catalog_entries()


def get_selectable_models(
    inventory_key: str | None = "Fence Panel",
    *,
    allow_demo: bool | None = None,
) -> list[ModelConfig]:
    """Models for the Analysis selector.

    Includes enabled Roboflow workflow/model adapters and the optional Local
    Picket Counter. Demo fixtures stay out unless allow_demo/DEMO_MODE.
    """
    if allow_demo is None:
        allow_demo = bool(DEMO_MODE)
    models = load_models_from_file()
    enabled = get_enabled_valid_models(models, allow_demo_ids=allow_demo)
    out: list[ModelConfig] = []
    for m in enabled:
        kind = (m.kind or "").lower()
        if kind not in {"workflow", "model", "local"}:
            continue
        if m.is_demo_model_id() or (m.demo_only and kind != "local"):
            if not allow_demo:
                continue
        # Fixed-class / local inventory filter
        supported = list(m.supported_inventory_types or [])
        if supported and inventory_key and inventory_key not in supported:
            continue
        if (
            kind != "local"
            and not m.dynamic_classes
            and not m.supports_prompt
            and m.allowed_classes
            and inventory_key == "Fence Panel"
            and not _classes_compatible_with_fence(m.allowed_classes)
            and (not supported or inventory_key not in supported)
        ):
            continue
        out.append(m)
    return out


def validate_model(model_key: str) -> dict[str, Any]:
    entries = {e.key: e for e in get_all_catalog_models()}
    entry = entries.get(model_key)
    if entry is None:
        # Try by display name
        for e in entries.values():
            if e.display_name == model_key:
                entry = e
                break
    if entry is None:
        return {
            "ok": False,
            "status": STATUS_UNAVAILABLE,
            "message": "Model no longer configured",
            "model_key": model_key,
        }
    if entry.stale:
        return {
            "ok": False,
            "status": STATUS_STALE,
            "message": entry.sync_note or "Unavailable during last sync",
            "model_key": entry.key,
            "display_name": entry.display_name,
        }
    if entry.adapter_type == "none" or entry.status == STATUS_UNAVAILABLE:
        return {
            "ok": False,
            "status": STATUS_UNAVAILABLE,
            "message": entry.deployment or "Deployment unavailable",
            "model_key": entry.key,
            "display_name": entry.display_name,
        }
    cfg = entry.to_model_config()
    errors = cfg.validation_errors(allow_demo_ids=bool(DEMO_MODE))
    if errors:
        return {
            "ok": False,
            "status": STATUS_NEEDS_CONFIG,
            "message": "; ".join(errors),
            "model_key": entry.key,
            "display_name": entry.display_name,
        }
    return {
        "ok": True,
        "status": entry.status if entry.validated or entry.status == STATUS_READY else STATUS_NEEDS_CONFIG,
        "message": "OK",
        "model_key": entry.key,
        "display_name": entry.display_name,
        "entry": entry.to_dict(),
    }


def remove_stale_model_selection(
    selected_names: list[str] | None,
    inventory_key: str | None = "Fence Panel",
) -> tuple[list[str], str | None]:
    """Clear stale selections; fall back to inventory default or first compatible live model."""
    selectable = get_selectable_models(inventory_key, allow_demo=bool(DEMO_MODE))
    names = [m.name for m in selectable]
    cleaned = sanitize_selected_model_names(selected_names, names)
    note = None
    if selected_names and not cleaned:
        note = "No compatible live model is configured."
        # Try inventory default
        from config import INVENTORY_PROFILES

        default_name = (INVENTORY_PROFILES.get(inventory_key or "") or {}).get("default_model")
        default_name = normalize_model_name(default_name)
        if default_name in names:
            cleaned = [default_name]
            note = f"Fell back to default model: {default_name}"
        elif names:
            cleaned = [names[0]]
            note = f"Fell back to first compatible model: {names[0]}"
        else:
            cleaned = []
            note = "No compatible live model is configured."
    elif selected_names and len(cleaned) < len(
        [normalize_model_name(n) for n in (selected_names or []) if n]
    ):
        note = "Removed model(s) that are no longer configured."
    return cleaned, note


def set_catalog_entry_enabled(model_key: str, enabled: bool) -> bool:
    entries = load_catalog_entries()
    found = False
    for e in entries:
        if e.key == model_key:
            e.enabled = bool(enabled)
            if e.stale and enabled:
                return False
            found = True
            break
    if not found:
        return False
    save_catalog_entries(entries)
    # Mirror to models.json
    models = load_models_from_file()
    from model_adapters import model_key as mk_fn

    for m in models:
        if (m.key or mk_fn(m)) == model_key:
            m.enabled = bool(enabled)
    save_models_to_file(models)
    return True


def remove_from_catalog(model_key: str) -> bool:
    """Remove from local catalog / disable in models.json — never deletes Roboflow remote."""
    entries = [e for e in load_catalog_entries() if e.key != model_key]
    save_catalog_entries(entries)
    # Public store
    pubs = [e for e in load_approved_public_models() if e.key != model_key]
    save_approved_public_models(pubs)
    models = load_models_from_file()
    from model_adapters import model_key as mk_fn

    kept: list[ModelConfig] = []
    for m in models:
        if (m.key or mk_fn(m)) == model_key:
            # Workspace: disable rather than delete
            if (m.kind or "").lower() == "model" and m.workspace_name:
                m.enabled = False
                kept.append(m)
            # Public/session: drop; foundation workflow: disable
            elif (m.kind or "").lower() == "workflow":
                m.enabled = False
                kept.append(m)
            continue
        kept.append(m)
    save_models_to_file(kept)
    return True


def filter_catalog_entries(
    entries: Iterable[CatalogEntry],
    *,
    search: str = "",
    source: str | None = None,
    task_type: str | None = None,
    status: str | None = None,
    compatible_fence: bool = False,
    enabled_only: bool = False,
) -> list[CatalogEntry]:
    q = (search or "").strip().lower()
    out: list[CatalogEntry] = []
    for e in entries:
        if source and source != "all" and e.source != source:
            continue
        if task_type and task_type != "all" and e.task_type != task_type:
            continue
        if enabled_only and not e.enabled:
            continue
        if status and status != "all":
            if status == "enabled" and not e.enabled:
                continue
            if status == "needs_configuration" and e.status != STATUS_NEEDS_CONFIG:
                continue
            if status == "failed_validation" and e.status != STATUS_FAILED:
                continue
            if status == "compatible_fence":
                compatible_fence = True
            elif status not in {"enabled", "all"} and e.status != status and status != "compatible_fence":
                if status == STATUS_READY and e.status != STATUS_READY:
                    continue
        if compatible_fence:
            if e.dynamic_classes or e.supports_prompt:
                pass
            elif "Fence Panel" in (e.supported_inventory_types or []):
                pass
            elif _classes_compatible_with_fence(e.supported_classes or []):
                pass
            else:
                continue
        if q:
            blob = " ".join(
                [
                    e.display_name,
                    e.key,
                    e.source,
                    e.task_type,
                    e.architecture or "",
                    " ".join(e.supported_classes or []),
                    e.project_id or "",
                    e.workflow_id or "",
                ]
            ).lower()
            if q not in blob:
                continue
        if not DEMO_MODE and e.demo_only and e.source in {SOURCE_DEMO, SOURCE_LOCAL}:
            # Still show in catalog with Demo badge, but mark clearly
            pass
        out.append(e)
    return out


def last_sync_report() -> dict[str, Any]:
    return _read_json(SYNC_REPORT_PATH, {})


def history_model_label(model_name: str | None) -> str:
    """Label for history when model may no longer exist."""
    name = normalize_model_name(model_name)
    if not name:
        return "Model no longer configured"
    entries = {e.display_name: e for e in get_all_catalog_models()}
    file_names = {m.name for m in load_models_from_file()}
    if name in entries or name in file_names:
        return name
    if name in MODEL_NAME_ALIASES.values():
        return name
    return f"{name} (Model no longer configured)"
