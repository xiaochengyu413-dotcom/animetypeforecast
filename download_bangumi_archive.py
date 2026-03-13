from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "bangumi_archive"
LATEST_METADATA_URL = "https://raw.githubusercontent.com/bangumi/Archive/refs/heads/master/aux/latest.json"
USER_AGENT = "Mozilla/5.0 (compatible; CodexBangumiArchiveSync/1.0)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the latest Bangumi Archive dump and extract subject.jsonlines."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def fetch_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def extract_member(zip_path: Path, member_name: str, output_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member_name) as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def resolve_archive_name(download_url: str) -> str:
    parsed = urlparse(download_url)
    return Path(parsed.path).name


def sync_latest_archive(
    data_dir: Path,
    force_download: bool = False,
    skip_extract: bool = False,
) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_json(LATEST_METADATA_URL)
    download_url = str(metadata.get("browser_download_url") or metadata.get("url"))
    archive_name = str(metadata.get("name") or resolve_archive_name(download_url))
    archive_path = data_dir / archive_name
    subject_path = data_dir / "subject.jsonlines"
    metadata_path = data_dir / "latest_metadata.json"

    expected_hash = str(metadata.get("hash") or metadata.get("digest") or "").strip().lower()
    if expected_hash.startswith("sha256:"):
        expected_hash = expected_hash.split(":", 1)[1]
    needs_download = force_download or not archive_path.exists()
    if not needs_download and expected_hash:
        needs_download = sha256_of_file(archive_path).lower() != expected_hash

    if needs_download:
        print(f"downloading {download_url}")
        download_file(download_url, archive_path)
    else:
        print(f"archive already up to date: {archive_path}")

    if expected_hash:
        actual_hash = sha256_of_file(archive_path).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Downloaded archive hash mismatch. Expected {expected_hash}, got {actual_hash}."
            )

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved metadata to {metadata_path}")

    if not skip_extract:
        if force_download or not subject_path.exists():
            print(f"extracting subject.jsonlines to {subject_path}")
            extract_member(archive_path, "subject.jsonlines", subject_path)
        else:
            print(f"subject file already exists: {subject_path}")

    return {
        "metadata": metadata_path,
        "archive": archive_path,
        "subject": subject_path,
    }


def main() -> None:
    args = parse_args()
    sync_latest_archive(
        data_dir=args.data_dir,
        force_download=args.force_download,
        skip_extract=args.skip_extract,
    )


if __name__ == "__main__":
    main()
