from dataclasses import replace
from uuid import uuid4

import pytest

from app.core.errors import AppError
from app.core.image_validation import ImageMetadata
from app.core.session import SessionState
from app.cv.availability import RunStatus
from app.cv.schemas import CVResult, SegmentationResult

METADATA = ImageMetadata(50, 30, "JPEG", 118, "a" * 64)


def result_for(state, run):
    return CVResult(
        **state.image.identity.model_dump(),
        run_id=run.run_id,
        segmentation=SegmentationResult(image_width=50, image_height=30),
    )


def test_create_replace_and_reset():
    state = SessionState()
    assert state.image is state.result is state.processing_status is state.current_run_id is None
    assert state.image_revision == 0
    with pytest.raises(AppError):
        state.begin_run()
    assert state.replace_image(METADATA, "first.png") is None
    first = state.image
    assert state.image_revision == 1 and first.metadata is METADATA
    run = state.begin_run()
    assert state.processing_status is RunStatus.RUNNING
    assert state.current_run_id == run.run_id
    assert state.accept_result(run, result_for(state, run))
    assert state.current_run_id == run.run_id and state.processing_status is RunStatus.NOT_AVAILABLE
    assert state.replace_image(METADATA, "second.png") is first
    assert state.image_revision == 2 and state.image.image_id != first.image_id
    assert state.result is state.current_run_id is None
    assert state.reset().temporary_file == "second.png"
    assert state.image_revision == 3 and state.image is state.result is state.current_run_id is None


def test_sessions_runs_and_image_revisions_are_independent():
    first, other = SessionState(), SessionState()
    first.replace_image(METADATA, "first.png")
    other.replace_image(METADATA, "other.png")
    old_run = first.begin_run()
    late_result = result_for(first, old_run)
    assert first.session_id != other.session_id and other.result is None
    with pytest.raises(AppError):
        first.begin_run()
    first.replace_image(METADATA, "new.png")
    current_run = first.begin_run()
    current_result = result_for(first, current_run)
    assert not first.accept_result(old_run, late_result)
    assert not first.accept_result(current_run, late_result)
    assert not first.accept_result(replace(current_run, session_id=other.session_id), current_result)
    assert not first.accept_result(replace(current_run, image_revision=1), current_result)
    assert not first.accept_result(current_run, current_result.model_copy(update={"image_id": uuid4()}))
    assert first.accept_result(current_run, current_result)
    assert not first.accept_result(current_run, current_result)
    next_run = first.begin_run()
    next_result = result_for(first, next_run)
    first.reset()
    assert not first.accept_result(next_run, next_result)
    assert first.result is None and other.image.temporary_file == "other.png"
