from _thread import LockType
from dataclasses import dataclass, field
from threading import Lock
from uuid import UUID, uuid4

from app.core.errors import AppError, ErrorCode
from app.core.image_validation import ImageMetadata
from app.cv.availability import RunStatus
from app.cv.schemas import CVResult, ImageIdentity


@dataclass(frozen=True, slots=True)
class SessionImage:
    image_id: UUID
    metadata: ImageMetadata
    temporary_file: str

    @property
    def identity(self) -> ImageIdentity:
        return ImageIdentity(
            image_id=self.image_id,
            image_width=self.metadata.width,
            image_height=self.metadata.height,
            image_hash=self.metadata.sha256,
        )


@dataclass(frozen=True, slots=True)
class RunToken:
    session_id: UUID
    image_revision: int
    run_id: UUID


@dataclass(slots=True)
class SessionState:
    session_id: UUID = field(default_factory=uuid4, init=False)
    image_revision: int = field(default=0, init=False)
    image: SessionImage | None = field(default=None, init=False)
    active_run: RunToken | None = field(default=None, init=False)
    result: CVResult | None = field(default=None, init=False)
    _latest_image_action: int = field(default=0, init=False, repr=False)
    lock: LockType = field(default_factory=Lock, init=False, repr=False, compare=False)
    _run_claimed: bool = field(default=False, init=False, repr=False)

    def accept_image_action(self, action: int) -> bool:
        """Reject browser edits that reached the queue after a newer edit."""
        if type(action) is not int or action <= self._latest_image_action:
            return False
        self._latest_image_action = action
        return True

    def matches_image_action(self, action: int) -> bool:
        return type(action) is int and action == self._latest_image_action

    @property
    def processing_status(self) -> RunStatus | None:
        if self.active_run is not None:
            return RunStatus.RUNNING
        return self.result.metadata.status if self.result is not None else None

    @property
    def current_run_id(self) -> UUID | None:
        if self.active_run is not None:
            return self.active_run.run_id
        return self.result.run_id if self.result is not None else None

    def replace_image(self, metadata: ImageMetadata, temporary_file: str) -> SessionImage | None:
        previous = self.reset()
        self.image = SessionImage(uuid4(), metadata, temporary_file)
        return previous

    def reset(self) -> SessionImage | None:
        previous = self.image
        self.image_revision += 1
        self.image = self.active_run = self.result = None
        return previous

    def begin_run(self) -> RunToken:
        if self.image is None or self.active_run is not None:
            raise AppError(ErrorCode.INVALID_SESSION)
        self.result = None
        self.active_run = RunToken(self.session_id, self.image_revision, uuid4())
        self._run_claimed = False
        return self.active_run

    def claim_run(self) -> tuple[RunToken, SessionImage] | None:
        if self.active_run is None or self.image is None or self._run_claimed:
            return None
        self._run_claimed = True
        return self.active_run, self.image

    def accept_result(self, token: RunToken, result: CVResult) -> bool:
        if (
            self.active_run != token
            or self.image is None
            or result.run_id != token.run_id
            or result.metadata.status is RunStatus.RUNNING
        ):
            return False
        image = self.image
        if (result.image_id, result.image_width, result.image_height, result.image_hash) != (
            image.image_id, image.metadata.width, image.metadata.height, image.metadata.sha256
        ):
            return False
        self.result = result
        self.active_run = None
        return True
