"""Versioned, atomic session persistence with optimistic concurrency checks."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path


class SessionConflictError(RuntimeError):
    """Raised when a writer would overwrite a newer session revision."""


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session, expected_revision=None):
        if not isinstance(session, dict) or not str(session.get("id", "")).strip():
            raise ValueError("session must contain a non-empty id")
        path = self.path(session["id"])
        lock_path = path.with_suffix(path.suffix + ".lock")
        with self._lock, self._exclusive_lock(lock_path):
            disk_revision = 0
            if path.exists():
                try:
                    disk_revision = int(json.loads(path.read_text(encoding="utf-8")).get("revision", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise SessionConflictError(f"cannot read current session revision: {path}") from exc
            local_revision = int(session.get("revision", 0))
            expected = local_revision if expected_revision is None else int(expected_revision)
            if disk_revision != expected:
                raise SessionConflictError(
                    f"session revision conflict for {session['id']}: expected {expected}, current {disk_revision}"
                )
            next_revision = disk_revision + 1
            payload = dict(session)
            payload["revision"] = next_revision
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=str(path.parent),
                prefix=path.name + ".",
                suffix=".tmp",
            ) as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                temp_name = handle.name
            try:
                Path(temp_name).replace(path)
            except Exception:
                try:
                    Path(temp_name).unlink()
                except FileNotFoundError:
                    pass
                raise
            session["revision"] = next_revision
            return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None

    @staticmethod
    def _exclusive_lock(lock_path, timeout=5.0):
        class _Lock:
            def __enter__(self_inner):
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        self_inner.fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        return self_inner
                    except FileExistsError:
                        if time.monotonic() >= deadline:
                            raise SessionConflictError(f"session lock timeout: {lock_path}")
                        time.sleep(0.01)

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    os.close(self_inner.fd)
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass

        return _Lock()
