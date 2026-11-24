import os
import time
from uuid import uuid4

import pytest

from app.core.errors import AppError, ErrorCode
from app.core.temporary_files import TemporaryFiles


def put(files, session, data):
    with files.operation(session):
        return files.write(session, lambda stream: stream.write(data))


def test_sessions_and_obsolete_files_are_isolated(tmp_path):
    files = TemporaryFiles(tmp_path, 60)
    a, b = uuid4(), uuid4()
    old = put(files, a, b"old")
    current = put(files, a, b"current")
    other = put(files, b, b"other")
    with files.operation(a):
        with pytest.raises(AppError) as error:
            files.discard(a, "../outside.png")
        assert error.value.code is ErrorCode.UNSAFE_PATH
        files.discard(a, old)
        assert (files.root / a.hex / old).exists()
    assert not (files.root / a.hex / old).exists()
    with files.open_image(a, current) as image:
        assert image.read() == b"current"
    with pytest.raises(AppError) as error, files.open_image(a, other):
        pytest.fail("A file reference must be scoped to its owning session")
    assert error.value.code is ErrorCode.MISSING_FILE
    files.close_session(a)
    files.close_session(a)
    assert not (files.root / a.hex).exists()
    assert (files.root / b.hex / other).exists()


def test_ttl_uses_last_use_and_defers_removal_during_an_operation(tmp_path):
    files = TemporaryFiles(tmp_path, 60)
    session = uuid4()
    name = put(files, session, b"active")
    directory = files.root / session.hex
    now = time.time()
    os.utime(directory, (now - 120, now - 120))
    with files.open_image(session, name) as stream:
        files.sweep_expired(now=now)
        assert directory.exists() and stream.read() == b"active"
    files.sweep_expired(now=now + 10)
    assert directory.exists()
    files.sweep_expired(now=now + 120)
    files.sweep_expired(now=now + 120)
    assert not directory.exists()


def test_reset_discard_and_session_close_wait_for_open_file(tmp_path):
    files = TemporaryFiles(tmp_path, 60)
    session = uuid4()
    name = put(files, session, b"active")
    path = files.root / session.hex / name
    with files.open_image(session, name) as stream:
        with files.operation(session):
            files.discard(session, name)
        files.close_session(session)
        assert path.exists() and stream.read() == b"active"
    assert not path.parent.exists()


def test_failed_write_removes_partial_file(tmp_path):
    files = TemporaryFiles(tmp_path, 60)
    session = uuid4()

    def fail(stream):
        stream.write(b"partial")
        raise OSError("test failure")

    with files.operation(session), pytest.raises(AppError) as error:
        files.write(session, fail)
    assert error.value.code is ErrorCode.TEMPORARY_FILE
    assert not (files.root / session.hex).exists()


def test_cleanup_never_traverses_external_directory(tmp_path):
    files = TemporaryFiles(tmp_path / "runtime", 60)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    session = uuid4()
    files.root.mkdir(parents=True)
    linked_session = files.root / session.hex
    try:
        linked_session.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows junctions can be created without symlink privileges.
        import _winapi

        _winapi.CreateJunction(str(outside), str(linked_session))
    files.close_session(session)
    files.sweep_expired(now=time.time() + 120)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert linked_session.exists()
    session_dir = files.root / uuid4().hex
    linked_session.rename(session_dir)
    if session_dir.is_junction():
        session_dir.rmdir()
    else:
        session_dir.unlink()


def test_runtime_is_reserved_while_janitor_is_running(tmp_path):
    first, other = TemporaryFiles(tmp_path, 60), TemporaryFiles(tmp_path, 60)
    expired = uuid4()
    put(first, expired, b"old")
    directory = first.root / expired.hex
    os.utime(directory, (0, 0))
    with first.maintain(60):
        assert not directory.exists()
        with pytest.raises(AppError) as error, other.maintain(60):
            pytest.fail("Two app instances cannot manage the same runtime")
        assert error.value.field == "RUNTIME_ALREADY_IN_USE"
    with other.maintain(60):
        pass
