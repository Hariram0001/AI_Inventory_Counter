"""Automatic Roboflow connectivity probe with session cache."""

from __future__ import annotations

from typing import Any


def _run_lightweight_auth_probe() -> dict[str, Any]:
    """Auth-only Roboflow check — fast enough to run automatically."""
    import time

    import config
    from detector import RoboflowDetector

    started = time.perf_counter()
    result: dict[str, Any] = {
        "ok": False,
        "auth": "Unknown",
        "auth_ok": False,
        "workflow": "—",
        "response_source": "demo source" if config.DEMO_MODE else "live Roboflow",
        "message": "",
        "processing_time": 0.0,
    }
    try:
        detector = RoboflowDetector()
        ok, msg = detector.test_connectivity()
        result["auth"] = "Successful" if ok else "Failed"
        result["auth_ok"] = bool(ok)
        result["ok"] = bool(ok)
        result["message"] = msg
        if config.DEMO_MODE:
            result["auth"] = "Demo Mode"
            result["auth_ok"] = True
            result["ok"] = True
            result["message"] = "Demo Mode active — live API not required."
    except Exception as exc:  # noqa: BLE001
        from detector import sanitize_exception_text

        result["message"] = sanitize_exception_text(f"{type(exc).__name__}: {exc}")
        result["auth"] = "Failed"
        result["auth_ok"] = False
        result["ok"] = False
    result["processing_time"] = time.perf_counter() - started
    return result


def ensure_roboflow_probe(*, force: bool = False) -> dict[str, Any]:
    """Return a connection probe, auto-running one when missing or stale."""
    import streamlit as st

    from poc_ux import probe_is_fresh, stamp_connection_probe

    existing = st.session_state.get("connection_probe")
    if not force and probe_is_fresh(existing if isinstance(existing, dict) else None):
        return dict(existing)

    # Avoid stacking probes if several panels render in one run.
    if st.session_state.get("_connection_probe_running") and isinstance(existing, dict):
        return dict(existing)

    st.session_state._connection_probe_running = True
    try:
        with st.spinner("Checking Roboflow connection…"):
            stamped = stamp_connection_probe(_run_lightweight_auth_probe())
        st.session_state.connection_probe = stamped
        return stamped
    finally:
        st.session_state._connection_probe_running = False


__all__ = ["ensure_roboflow_probe"]
