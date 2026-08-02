"""Dark / light theme toggle — helpers, CSS coverage, sidebar control."""

from __future__ import annotations

import inspect

import auth_ui
import ui_helpers
from ui_helpers import (
    DEFAULT_UI_THEME,
    get_ui_theme,
    set_ui_theme,
    toggle_ui_theme,
    _theme_override_css,
    inject_css,
)


def test_default_theme_is_dark(monkeypatch):
    class _SS(dict):
        pass

    fake = type("S", (), {"session_state": _SS()})()
    monkeypatch.setattr(ui_helpers, "_st", lambda: fake)
    assert get_ui_theme() == "dark"
    assert DEFAULT_UI_THEME == "dark"


def test_set_and_toggle_theme(monkeypatch):
    class _SS(dict):
        pass

    fake = type("S", (), {"session_state": _SS()})()
    monkeypatch.setattr(ui_helpers, "_st", lambda: fake)

    assert set_ui_theme("light") == "light"
    assert get_ui_theme() == "light"
    assert toggle_ui_theme() == "dark"
    assert get_ui_theme() == "dark"
    assert toggle_ui_theme() == "light"
    assert set_ui_theme("nope") == "dark"


def test_theme_css_covers_icons_in_both_modes():
    light = _theme_override_css("light")
    dark = _theme_override_css("dark")
    for css in (light, dark):
        assert "stIconMaterial" in css
        assert "stSidebar" in css
        assert "fill: currentColor" in css or "fill: #ffffff" in css
        assert "color:" in css
    assert "#ffffff" in light
    assert "#31333F" in light
    assert "#0e1117" in dark
    assert "#fafafa" in dark
    # Role badges get readable colors in both themes
    assert "aic-role-admin" in light and "aic-role-admin" in dark
    assert "#0369a1" in light  # light-mode admin badge text
    assert "#7dd3fc" in dark


def test_sidebar_has_theme_toggle_at_bottom():
    src = inspect.getsource(auth_ui.render_app_sidebar)
    assert "nav_theme_toggle" in src
    assert "toggle_ui_theme" in src
    assert "light_mode" in src
    assert "dark_mode" in src
    # Theme control is after sign-out
    assert src.index("menu_signout") < src.index("nav_theme_toggle")


def test_theme_not_cleared_with_user_scoped_state():
    import auth_session

    src = inspect.getsource(auth_session.clear_user_scoped_state)
    assert "ui_theme" not in src
    assert "clear_shape_detection_state" in src


def test_inject_css_uses_active_theme():
    src = inspect.getsource(inject_css)
    assert "get_ui_theme" in src
    assert "aic-theme-root" in src
    assert "_theme_override_css" in src
