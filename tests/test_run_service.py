import pytest

from run_service import covariance_mode


def test_covariance_mode_defaults_to_diagonal(monkeypatch):
    monkeypatch.delenv("TOOLBANDIT_COVARIANCE", raising=False)
    assert covariance_mode() == ("diagonal", True)


def test_covariance_mode_supports_full(monkeypatch):
    monkeypatch.setenv("TOOLBANDIT_COVARIANCE", "full")
    assert covariance_mode() == ("full", False)


def test_covariance_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("TOOLBANDIT_COVARIANCE", "dense-ish")
    with pytest.raises(SystemExit, match="diagonal.*full"):
        covariance_mode()
