"""Verify import boundary: constants vs ui_helpers under Streamlit-like load order."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    sys.path.insert(0, str(ROOT))
    for name in list(sys.modules):
        if name in {
            "ui_helpers",
            "navigation",
            "settings_constants",
            "app_constants",
            "app",
            "app_audit",
        }:
            del sys.modules[name]

    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_src)

    by_module: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "ui_helpers",
            "app_constants",
            "navigation",
            "settings_constants",
        }:
            by_module.setdefault(node.module, []).extend(a.name for a in node.names)

    # 1) Constants must come from app_constants, not depend on Streamlit
    import app_constants

    assert hasattr(app_constants, "SETTINGS_SECTION_LABELS")
    print("app_constants.SETTINGS_SECTION_LABELS OK")

    # 2) Streamlit then ui_helpers (Streamlit startup order)
    import streamlit  # noqa: F401

    ui = importlib.import_module("ui_helpers")
    missing_ui = [n for n in by_module.get("ui_helpers", []) if not hasattr(ui, n)]
    missing_c = [n for n in by_module.get("app_constants", []) if not hasattr(app_constants, n)]
    print("ui_helpers imports:", by_module.get("ui_helpers", []))
    print("app_constants imports:", by_module.get("app_constants", []))
    print("missing ui_helpers:", missing_ui)
    print("missing app_constants:", missing_c)

    # 3) settings_constants must NOT import ui_helpers (breaks cycles)
    sc_src = (ROOT / "settings_constants.py").read_text(encoding="utf-8")
    if "from ui_helpers" in sc_src or "import ui_helpers" in sc_src:
        print("FAIL: settings_constants imports ui_helpers (circular risk)")
        return 1

    # 4) ui_helpers must not top-level import streamlit
    uh_src = (ROOT / "ui_helpers.py").read_text(encoding="utf-8")
    for line in uh_src.splitlines():
        if line.startswith("import streamlit") or line.startswith("from streamlit "):
            print("FAIL: top-level streamlit in ui_helpers:", line)
            return 1

    if missing_ui or missing_c:
        return 1

    # 5) Full app exec
    spec = importlib.util.spec_from_file_location("app_audit", ROOT / "app.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app_audit"] = mod
    spec.loader.exec_module(mod)
    print("APP_LOAD_OK")
    print("IMPORT_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
