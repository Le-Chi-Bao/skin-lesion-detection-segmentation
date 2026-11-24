import logging
import os
import re
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from threading import Event, Lock, Thread
from typing import BinaryIO
from uuid import UUID, uuid4

from filelock import FileLock, Timeout

from app.core.errors import AppError, ErrorCode
from app.core.logging_config import log_error

logger = logging.getLogger(__name__)
SESSION_NAME = re.compile(r"[a-f0-9]{32}")
IMAGE_NAME = re.compile(r"[a-f0-9]{32}\.png")


def checked_path(path: Path) -> Path:
    """Require an absolute path with no filesystem redirection."""
    try:
        if not path.is_absolute() or path.resolve() != path or path.is_symlink() or path.is_junction():
            raise AppError(ErrorCode.UNSAFE_PATH)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AppError(ErrorCode.UNSAFE_PATH) from exc
    return path


class TemporaryFiles:
    def __init__(self, runtime_dir: Path, ttl_seconds: int):
        self.runtime_dir = runtime_dir
        self.root = runtime_dir / "sessions"
        self.ttl_seconds = ttl_seconds
        self.cache_dir = runtime_dir / "gradio-cache"
        self._lock = Lock()
        self._active: dict[UUID, int] = {}
        self._discarded: dict[UUID, set[str]] = {}
        self._closing: set[UUID] = set()

    def _directory(self, session_id: UUID) -> Path:
        if not isinstance(session_id, UUID):
            raise AppError(ErrorCode.INVALID_SESSION)
        return checked_path(self.root / session_id.hex)

    def _image_path(self, session_id: UUID, name: str) -> Path:
        if not IMAGE_NAME.fullmatch(name):
            raise AppError(ErrorCode.UNSAFE_PATH)
        return checked_path(self._directory(session_id) / name)

    @contextmanager
    def operation(self, session_id: UUID) -> Iterator[None]:
        with self._lock:
            self._directory(session_id)
            if session_id in self._closing:
                raise AppError(ErrorCode.INVALID_SESSION)
            self._active[session_id] = self._active.get(session_id, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._active[session_id] -= 1
                if self._active[session_id] == 0:
                    del self._active[session_id]
                    self._finish_operation(session_id)

    def _finish_operation(self, session_id: UUID) -> None:
        obsolete = self._discarded.pop(session_id, ())
        if session_id in self._closing:
            self._closing.remove(session_id)
            self._remove_session(session_id)
            return
        for name in obsolete:
            self._remove_image(session_id, name)
        try:
            directory = self._directory(session_id)
            os.utime(directory)
            # A reset can leave an empty session directory.
            if not any(directory.iterdir()):
                directory.rmdir()
        except FileNotFoundError:
            pass
        except (AppError, OSError) as exc:
            self._log_cleanup_error(exc, session_id)

    def write(self, session_id: UUID, writer: Callable[[BinaryIO], None]) -> str:
        """Create a private PNG. Call within an operation for this session."""
        with self._lock:
            self._require_operation(session_id)
            path = self._image_path(session_id, f"{uuid4().hex}.png")
        created = False
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as stream:
                created = True
                writer(stream)
        except Exception as exc:
            if created:
                self.discard(session_id, path.name)
            if isinstance(exc, (OSError, ValueError)):
                raise AppError(ErrorCode.TEMPORARY_FILE) from exc
            raise
        return path.name

    def write_mask(self, session_id: UUID, writer: Callable[[BinaryIO], None]) -> str:
        return self.write(session_id, writer).removesuffix(".png")

    def open_mask(self, session_id: UUID, reference: str) -> AbstractContextManager[BinaryIO]:
        return self.open_image(session_id, f"{reference}.png")

    def discard_mask(self, session_id: UUID, reference: str) -> None:
        self.discard(session_id, f"{reference}.png")

    @contextmanager
    def open_image(self, session_id: UUID, name: str) -> Iterator[BinaryIO]:
        with self.operation(session_id):
            path = self._image_path(session_id, name)
            try:
                stream = path.open("rb")
            except FileNotFoundError as exc:
                raise AppError(ErrorCode.MISSING_FILE) from exc
            except OSError as exc:
                raise AppError(ErrorCode.TEMPORARY_FILE) from exc
            with stream:
                yield stream

    def _require_operation(self, session_id: UUID) -> None:
        if session_id not in self._active:
            raise AppError(ErrorCode.INVALID_SESSION)

    def discard(self, session_id: UUID, name: str) -> None:
        if not IMAGE_NAME.fullmatch(name):
            raise AppError(ErrorCode.UNSAFE_PATH)
        with self._lock:
            self._require_operation(session_id)
            self._discarded.setdefault(session_id, set()).add(name)

    def close_session(self, session_id: UUID) -> None:
        with self._lock:
            if session_id in self._active:
                self._closing.add(session_id)
            else:
                self._remove_session(session_id)

    def _remove_image(self, session_id: UUID, name: str) -> None:
        try:
            self._image_path(session_id, name).unlink(missing_ok=True)
        except (AppError, OSError) as exc:
            self._log_cleanup_error(exc, session_id)

    def _remove_session(self, session_id: UUID) -> None:
        try:
            shutil.rmtree(self._directory(session_id))
        except FileNotFoundError:
            pass
        except (AppError, OSError) as exc:
            self._log_cleanup_error(exc, session_id)

    @staticmethod
    def _log_cleanup_error(exc: Exception, session_id: UUID | None = None) -> None:
        if isinstance(exc, OSError):
            original = exc
            exc = AppError(ErrorCode.TEMPORARY_FILE)
            exc.__cause__ = original
        log_error(logger, exc, session_id)

    def sweep_expired(self, *, now: float | None = None) -> None:
        cutoff = (time.time() if now is None else now) - self.ttl_seconds
        with self._lock:
            try:
                directories = list(checked_path(self.root).iterdir())
            except FileNotFoundError:
                return
            except (AppError, OSError) as exc:
                self._log_cleanup_error(exc)
                return
            for directory in directories:
                if not SESSION_NAME.fullmatch(directory.name):
                    continue
                session_id = UUID(hex=directory.name)
                if session_id in self._active:
                    continue
                try:
                    if checked_path(directory).is_dir() and directory.stat().st_mtime <= cutoff:
                        self._remove_session(session_id)
                except FileNotFoundError:
                    continue
                except (AppError, OSError) as exc:
                    self._log_cleanup_error(exc, session_id)

    @contextmanager
    def maintain(self, interval: int) -> Iterator[None]:
        """Reserve this runtime for one app process and run a background janitor."""
        try:
            checked_path(self.runtime_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
            lock = FileLock(checked_path(self.runtime_dir / ".wp1.lock"), timeout=0)
            lock.acquire()
        except Timeout as exc:
            raise AppError(ErrorCode.TEMPORARY_FILE, field="RUNTIME_ALREADY_IN_USE") from exc
        except OSError as exc:
            raise AppError(ErrorCode.TEMPORARY_FILE, field="RUNTIME_DIR") from exc
        try:
            self.sweep_expired()
            stop = Event()

            def cleanup() -> None:
                while not stop.wait(interval):
                    self.sweep_expired()

            janitor = Thread(target=cleanup, name="session-cleanup", daemon=True)
            janitor.start()
            try:
                yield
            finally:
                stop.set()
                janitor.join()
        finally:
            lock.release()
