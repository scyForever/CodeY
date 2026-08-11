from pathlib import Path

import pytest

from CodeY.storage.session import SessionConflictError, SessionStore


def test_session_store_rejects_stale_writer_without_mutating_its_revision(tmp_path):
    store = SessionStore(tmp_path)
    session = {"id": "session-1", "revision": 0, "value": "initial"}
    store.save(session)
    first = store.load("session-1")
    stale = store.load("session-1")

    first["value"] = "first writer"
    store.save(first)
    stale["value"] = "stale writer"

    with pytest.raises(SessionConflictError, match="revision conflict"):
        store.save(stale)

    assert stale["revision"] == 1
    assert store.load("session-1")["value"] == "first writer"


def test_session_store_does_not_advance_revision_when_atomic_replace_fails(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    session = {"id": "session-1", "revision": 0}

    def fail_replace(self, target):
        del self, target
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.save(session)

    assert session["revision"] == 0
    assert not store.path("session-1").exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.lock")) == []
