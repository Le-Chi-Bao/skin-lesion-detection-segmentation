"""Stage official archives locally and derive a checksummed ISIC Task 1 manifest."""

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from tqdm.auto import tqdm

from src.annotations import build_annotations, digest_json, sha256_file, write_json
from src.dataset import load_manifest, read_groups, split_records

WORKSPACE = Path(__file__).resolve().parents[1]
OFFICIAL = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
GROUPS = "ISIC2018_Task3_Training_LesionGroupings.csv"
ARCHIVES = {
    "training": ("ISIC2018_Task1-2_Training_Input.zip", "ISIC2018_Task1_Training_GroundTruth.zip", 2594),
    "test": ("ISIC2018_Task1-2_Test_Input.zip", "ISIC2018_Task1_Test_GroundTruth.zip", 1000),
}


def check_space(path, needed):
    reserve = 1024**3
    free = shutil.disk_usage(path).free
    if free < needed + reserve:
        raise OSError(f"Need {(needed + reserve) / 1024**3:.2f} GiB free in {path}; have {free / 1024**3:.2f} GiB")


def stage_file(name, cache, archives, download):
    destination = cache / name
    source = (Path(archives) / name) if archives else None
    if destination.is_file() and (not source or not source.is_file() or destination.stat().st_size == source.stat().st_size):
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    if source and source.is_file():
        stream, size = source.open("rb"), source.stat().st_size
    elif download:
        stream = urllib.request.urlopen(OFFICIAL + name, timeout=90)
        content_length = stream.headers.get("Content-Length")
        if content_length is None:
            stream.close()
            raise OSError(f"Unable to determine size of official file: {name}")
        size = int(content_length)
    else:
        raise FileNotFoundError(f"Missing {name}; put it in the Drive archives directory or enable official download.")
    try:
        with stream:
            check_space(cache, size)
            with temporary.open("wb") as output, tqdm(total=size, unit="B", unit_scale=True, desc=name, mininterval=1) as bar:
                for block in iter(lambda: stream.read(2**20), b""):
                    output.write(block)
                    bar.update(len(block))
        if temporary.stat().st_size != size:
            raise OSError(f"Unexpected archive length: {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_archive(archive, root):
    output = root / archive.stem
    checksum = sha256_file(archive)
    marker = output / ".archive_sha256"
    if marker.is_file() and marker.read_text(encoding="ascii") == checksum:
        return output
    if output.exists():
        raise ValueError(f"Extracted cache differs from {archive.name}; choose a fresh data root: {output}")
    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        check_space(root, sum(item.file_size for item in members))
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as temporary:
            staged = Path(temporary) / archive.stem
            staged.mkdir()
            seen = set()
            for member in tqdm(members, desc=f"Extract {archive.stem}", mininterval=1):
                name = member.filename
                parts = Path(name).parts
                if (not parts or Path(name).is_absolute() or ".." in name.split("/")
                        or any(char in name for char in ("\\", ":")) or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError(f"Unsafe archive entry: {name}")
                if name.casefold() in seen:
                    raise ValueError(f"Duplicate archive entry: {name}")
                seen.add(name.casefold())
                zipped.extract(member, staged)
            (staged / marker.name).write_text(checksum, encoding="ascii")
            staged.rename(output)
    return output


def prepare_dataset(root, archives=None, download=False, grouping_csv=None, invalid_policy="fail", kind="training"):
    if invalid_policy not in {"fail", "exclude"} or (kind == "test" and invalid_policy != "fail"):
        raise ValueError("Invalid mask policy. Official test evaluation requires every sample.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "archives"
    cache.mkdir(exist_ok=True)
    images_name, masks_name, expected = ARCHIVES[kind]
    images = extract_archive(stage_file(images_name, cache, archives, download), root)
    masks = extract_archive(stage_file(masks_name, cache, archives, download), root)
    groups, group_source = {}, None
    if kind == "training":
        if grouping_csv is None:
            available = (archives and (Path(archives) / GROUPS).is_file()) or (cache / GROUPS).is_file()
            if available or download:
                grouping_csv = stage_file(GROUPS, cache, archives, download)
        groups = read_groups(grouping_csv)
        if grouping_csv:
            group_source = {"file": Path(grouping_csv).name, "sha256": sha256_file(grouping_csv)}
    records, invalid, image_count, mask_count = build_annotations(root, images, masks)
    write_json(root / "invalid_samples.json", {
        "images": image_count, "masks": mask_count, "expected": expected, "invalid": invalid,
    })
    if image_count != expected or mask_count != expected:
        raise ValueError(f"Expected {expected} images and masks, found {image_count}/{mask_count}. See {root / 'invalid_samples.json'}")
    if invalid and invalid_policy == "fail":
        raise ValueError(f"{len(invalid)} invalid samples. See {root / 'invalid_samples.json'}. Review before opting into exclusion.")
    for row in records:
        if row["id"] in groups:
            row["lesion_group"] = groups[row["id"]]
    manifest = {
        "version": 1, "dataset": kind, "source": OFFICIAL, "seed": 42 if kind == "training" else None,
        "archives": {name: (directory / ".archive_sha256").read_text(encoding="ascii")
                     for name, directory in ((images_name, images), (masks_name, masks))},
        "invalid_policy": invalid_policy, "invalid": invalid, "image_count": image_count,
        "group_source": group_source, "group_coverage": sum("lesion_group" in row for row in records),
        "records": records, "splits": split_records(records, groups) if kind == "training"
        else {"test": [row["id"] for row in records]},
    }
    manifest["sha256"] = digest_json(manifest)
    path = root / "manifest.json"
    if path.exists() and load_manifest(path, expected=kind)["sha256"] != manifest["sha256"]:
        raise ValueError(f"Existing split differs. Keep the original {path} or select a new data root.")
    write_json(path, manifest)
    return path


def persist_manifest(path, destination):
    existing = Path(destination)
    manifest = load_manifest(path)
    if existing.exists():
        if load_manifest(existing)["sha256"] != manifest["sha256"]:
            raise ValueError(f"Persistent split differs: {existing}. Restore the original data and grouping CSV.")
        return
    write_json(existing, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data" / "training")
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--grouping-csv", type=Path)
    parser.add_argument("--invalid-policy", choices=["fail", "exclude"], default="fail")
    parser.add_argument("--persist-manifest", type=Path)
    args = parser.parse_args()
    path = prepare_dataset(args.data, args.archives, args.download, args.grouping_csv, args.invalid_policy)
    if args.persist_manifest:
        persist_manifest(path, args.persist_manifest)
    print(path)


if __name__ == "__main__":
    main()
