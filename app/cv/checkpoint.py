"""Pinned SegFormer artifact and explicit acquisition command."""

import argparse
import hashlib
import http.client
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Iterator
from urllib.error import URLError
from urllib.request import urlopen

from filelock import FileLock, Timeout

from app.config import AppConfig
from app.core.errors import AppError, ErrorCode
from app.core.temporary_files import checked_path

REPOSITORY = "mokshhere/skin-lesion-segformer-gate2"
REVISION = "19ca826598cb59a6ac5b18b6a5636ba9298851b5"
FILENAME = "real_only/best.pt"
SIZE_BYTES = 99_002_337
SHA256 = "d5a92fa31296a4f7c32e87cdf84683119d07dc64410253103bfa285aacb93f87"
CHECKPOINT_ID = f"{REPOSITORY}@{REVISION}/{FILENAME}"
DOWNLOAD_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"
CHUNK_SIZE = 1024 * 1024


def checkpoint_path(config: AppConfig) -> Path:
    return config.segformer_checkpoint or checked_path(config.model_store_dir / "segformer" / FILENAME)


@contextmanager
def verified_checkpoint(path: Path) -> Iterator[BinaryIO]:
    try:
        stream = path.open("rb")
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.CHECKPOINT_MISSING) from exc
    except OSError as exc:
        raise AppError(ErrorCode.CHECKPOINT_READ) from exc
    with stream:
        try:
            if stream.seek(0, 2) != SIZE_BYTES:
                raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="size_bytes")
            stream.seek(0)
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="sha256")
            stream.seek(0)
        except OSError as exc:
            raise AppError(ErrorCode.CHECKPOINT_READ) from exc
        yield stream


def acquire(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(".lock")), timeout=120):
        if path.exists():
            with verified_checkpoint(path):
                return

        staging = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
                staging = Path(stream.name)
                with urlopen(DOWNLOAD_URL, timeout=30) as response:
                    total = 0
                    while True:
                        try:
                            chunk = response.read(CHUNK_SIZE)
                        except (OSError, http.client.HTTPException) as exc:
                            raise URLError(exc) from exc
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > SIZE_BYTES:
                            raise AppError(ErrorCode.CHECKPOINT_INCOMPATIBLE, field="size_bytes")
                        stream.write(chunk)
            with verified_checkpoint(staging):
                pass
            staging.replace(path)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải và xác minh checkpoint SegFormer đã ghim phiên bản.")
    parser.add_argument("--verify-only", action="store_true", help="Xác minh tệp cục bộ qua cấu hình, không tải.")
    args = parser.parse_args()
    try:
        config = AppConfig.from_env()
        path = checkpoint_path(config)
        if args.verify_only or config.segformer_checkpoint is not None:
            with verified_checkpoint(path):
                pass
        else:
            acquire(path)
    except AppError as exc:
        parser.exit(1, f"{exc}\n")
    except Timeout:
        parser.exit(1, "Một tiến trình khác đang chuẩn bị checkpoint. Vui lòng thử lại.\n")
    except (URLError, TimeoutError, http.client.HTTPException) as exc:
        parser.exit(1, f"Không thể tải checkpoint từ Hugging Face ({exc}).\n")
    except OSError as exc:
        parser.exit(1, f"Không thể ghi checkpoint vào ổ đĩa ({exc}).\n")

    print(json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "filename": FILENAME,
        "size_bytes": SIZE_BYTES, "sha256": SHA256,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
