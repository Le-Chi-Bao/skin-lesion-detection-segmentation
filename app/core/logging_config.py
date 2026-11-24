import logging
from uuid import UUID

from app.core.errors import AppError, ErrorCode


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def log_error(logger: logging.Logger, error: Exception, session_id: UUID | None = None) -> None:
    code = error.code if isinstance(error, AppError) else ErrorCode.INTERNAL
    cause = error.__cause__ or error
    logger.log(
        logging.ERROR if code is ErrorCode.INTERNAL else logging.WARNING,
        "code=%s session=%s field=%s exception=%s errno=%s",
        code,
        session_id or "-",
        error.field if isinstance(error, AppError) else "",
        type(cause).__name__,
        getattr(cause, "errno", None),
    )
