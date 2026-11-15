"""Download only the approved checkpoint and one licensed smoke-test image."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs" / "phase2a1"
HF_REPO = "mokshhere/skin-lesion-segformer-gate2"
HF_REVISION = "19ca826598cb59a6ac5b18b6a5636ba9298851b5"
CODE_REVISION = "d3d3195f13b02366cbc1332cbb57d7ca9234b7a7"
ISIC_ID = "ISIC_0012169"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(url, path, expected_size, expected_sha256=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and expected_sha256:
        with path.open("rb") as existing:
            checksum = hashlib.file_digest(existing, "sha256").hexdigest()
        if path.stat().st_size == expected_size and checksum == expected_sha256:
            return {"path": str(path), "bytes": expected_size, "sha256": checksum, "url": url}
    temporary = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    checksum = digest.hexdigest()
    if size != expected_size or (expected_sha256 and checksum != expected_sha256):
        raise RuntimeError(f"Download verification failed: {path}, size={size}, sha256={checksum}")
    temporary.replace(path)
    return {"path": str(path), "bytes": size, "sha256": checksum, "url": url}


AUDIT.mkdir(parents=True, exist_ok=True)
tree_url = f"https://huggingface.co/api/models/{HF_REPO}/tree/{HF_REVISION}/real_only"
tree = fetch_json(tree_url)
entry = next(item for item in tree if item["path"] == "real_only/best.pt")
checkpoint_url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/real_only/best.pt"
checkpoint_path = ROOT / "checkpoints" / "segformer-gate2" / "real_only" / "best.pt"
checkpoint = download(checkpoint_url, checkpoint_path, entry["size"], entry["lfs"]["oid"])
checkpoint.update({"repo": HF_REPO, "revision": HF_REVISION,
                   "upstream_lfs_sha256": entry["lfs"]["oid"], "tree_url": tree_url})
save_json(checkpoint_path.with_name("provenance.json"), checkpoint)
print("CHECKPOINT_DOWNLOAD_VERIFIED:", json.dumps(checkpoint), flush=True)

for name in ("config.json", "summary.json", "eval_test.json"):
    url = f"https://huggingface.co/{HF_REPO}/raw/{HF_REVISION}/real_only/{name}"
    save_json(AUDIT / ("upstream_" + name), fetch_json(url))

metadata_url = f"https://api.isic-archive.com/api/v2/images/{ISIC_ID}/"
metadata = fetch_json(metadata_url)
assert metadata["isic_id"] == ISIC_ID and metadata["public"]
assert metadata["copyright_license"] == "CC-0", "Unverified license; do not download image"
assert metadata["metadata"]["acquisition"]["image_type"] == "dermoscopic"
image_dir = ROOT / "data" / "samples"
save_json(image_dir / f"{ISIC_ID}.metadata.json", metadata)
file_info = metadata["files"]["full"]
with urllib.request.urlopen(urllib.request.Request(file_info["url"], method="HEAD"), timeout=60) as response:
    s3_size = int(response.headers["Content-Length"])
    s3_etag = response.headers["ETag"].strip('"')
    s3_last_modified = response.headers["Last-Modified"]
sample = download(file_info["url"], image_dir / f"{ISIC_ID}.jpg", s3_size)
with Path(sample["path"]).open("rb") as image_file:
    md5 = hashlib.file_digest(image_file, "md5").hexdigest()
assert md5 == s3_etag, "Downloaded image does not match S3 ETag"
sample.update({"api_advertised_bytes": file_info["size"], "s3_content_length": s3_size,
               "s3_etag": s3_etag, "downloaded_md5": md5, "s3_last_modified": s3_last_modified,
               "size_discrepancy": file_info["size"] != s3_size,
               "integrity_note": "API size differs from S3 object size; actual length and MD5 verified against S3 HEAD"})
with Image.open(sample["path"]) as image:
    image.verify()
with Image.open(sample["path"]) as image:
    assert image.size == (metadata["metadata"]["acquisition"]["pixels_x"],
                          metadata["metadata"]["acquisition"]["pixels_y"])
    sample.update({"image_format": image.format, "image_mode": image.mode,
                   "width": image.width, "height": image.height})
test_ids_url = (f"https://raw.githubusercontent.com/ishaannk/skin-lesion-seg/{CODE_REVISION}/"
                "experiments/segformer_real_s1337/perimage_test_t0.5_native.csv")
with urllib.request.urlopen(test_ids_url, timeout=60) as response:
    csv_text = response.read().decode("utf-8")
test_rows = [row for row in csv_text.splitlines() if row.endswith("," + ISIC_ID)]
assert len(test_rows) == 1
sample.update({"isic_id": ISIC_ID, "metadata_url": metadata_url,
               "copyright_license": metadata["copyright_license"],
               "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
               "attribution": metadata["attribution"],
               "credit": f"{ISIC_ID} - {metadata['attribution']}, via ISIC Archive, CC0 1.0",
               "upstream_test_list_url": test_ids_url, "upstream_test_row": test_rows[0],
               "split_status": "Listed as test by upstream; official challenge split not independently verified",
               "intended_use": "Local smoke test only; not model evaluation",
               "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
save_json(image_dir / f"{ISIC_ID}.provenance.json", sample)
print("SAMPLE_DOWNLOAD_VERIFIED:", json.dumps(sample), flush=True)
save_json(AUDIT / "assets.json", {"checkpoint": checkpoint, "sample": sample})
