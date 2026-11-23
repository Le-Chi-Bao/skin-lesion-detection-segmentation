"""Check the training manifest, disjointness, annotations and file hashes."""

import argparse
from pathlib import Path

from scripts.prepare_data import WORKSPACE
from src.dataset import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=WORKSPACE / "data/training/manifest.json")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, verify_files=True)
    print(f"Verified {manifest['sha256']}")
    print({split: len(ids) for split, ids in manifest["splits"].items()})
    print(f"Valid: {len(manifest['records'])}; invalid: {len(manifest['invalid'])}; grouped: {manifest['group_coverage']}")


if __name__ == "__main__":
    main()
