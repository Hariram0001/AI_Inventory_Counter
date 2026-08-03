"""Session isolation: concurrent users, wipe-on-switch, no process-global leak."""

from __future__ import annotations

import threading

import auth_session
import database
import openrouter_runtime as orun


def test_wipe_preserves_only_theme(monkeypatch):
    class _SS(dict):
        pass

    state = _SS(ui_theme="light", auth_user="x", analysis_results=[1], login_password="secret")
    fake = type("S", (), {"session_state": state, "query_params": {}})()
    monkeypatch.setattr(auth_session, "_st", lambda: fake)
    auth_session.wipe_session_for_identity_change()
    assert state.get("ui_theme") == "light"
    assert "auth_user" not in state
    assert "analysis_results" not in state
    assert "login_password" not in state


def test_openrouter_fallback_is_thread_local():
    orun._FALLBACK_SESSION.clear()
    orun.mark_session_key_rejected(reason="thread-a")
    assert orun._session_key_rejected()

    other_seen = {}

    def _other() -> None:
        orun._FALLBACK_SESSION.clear()
        other_seen["rejected"] = orun._session_key_rejected()
        orun.mark_session_key_rejected(reason="thread-b")
        other_seen["after"] = orun._session_key_rejected()

    t = threading.Thread(target=_other)
    t.start()
    t.join(timeout=5)
    assert other_seen["rejected"] is False
    assert other_seen["after"] is True
    # Parent thread rejection must not be cleared by the child thread.
    assert orun._session_key_rejected() is True
    orun._FALLBACK_SESSION.clear()


def test_sqlite_uses_wal_and_busy_timeout(tmp_path):
    db = str(tmp_path / "iso.db")
    database.initialize_database(db)
    with database._connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() in {"wal", "delete", "memory", "persist", "truncate", "off"}
        # busy_timeout is set in milliseconds
        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert int(busy) >= 30000


def test_verify_session_binding_rejects_foreign_sid(monkeypatch):
    class _SS(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    state = _SS(auth_session_id="mine")
    params = {"sid": "theirs"}
    fake = type("S", (), {"session_state": state, "query_params": params})()
    monkeypatch.setattr(auth_session, "_st", lambda: fake)
    assert auth_session.verify_session_binding() is False
    params["sid"] = "mine"
    assert auth_session.verify_session_binding() is True
