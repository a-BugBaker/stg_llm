from __future__ import annotations

"""下载并整理 OpenEQA 测试集资源。

脚本会把题目表与视觉 episode 资源统一下载到仓库内目录，
并在解压后写出一份 `download_manifest.json` 供后续脚本复用。
"""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from .common import now_utc_iso, write_json
from .defaults import (
    DEFAULT_OPENEQA_ARCHIVE_DIR,
    DEFAULT_OPENEQA_ARCHIVE_FILENAMES,
    DEFAULT_OPENEQA_DATASET_REPO,
    DEFAULT_OPENEQA_DOWNLOAD_MANIFEST,
    DEFAULT_OPENEQA_EPISODES_DIR,
    DEFAULT_OPENEQA_PARQUET_DIR,
    DEFAULT_OPENEQA_PARQUET_FILENAME,
    DEFAULT_OPENEQA_RAW_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并解压 OpenEQA 数据资源。")
    parser.add_argument("--repo-id", default=DEFAULT_OPENEQA_DATASET_REPO)
    parser.add_argument("--raw-root", default=str(DEFAULT_OPENEQA_RAW_ROOT))
    parser.add_argument("--parquet-filename", default=DEFAULT_OPENEQA_PARQUET_FILENAME)
    parser.add_argument("--archive-filenames", nargs="*", default=list(DEFAULT_OPENEQA_ARCHIVE_FILENAMES))
    parser.add_argument("--skip-archives", action="store_true", help="只下载 parquet，不下载视觉资源压缩包。")
    parser.add_argument("--force-extract", action="store_true", help="即使已存在解压目录，也重新解压。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    parquet_dir = raw_root / DEFAULT_OPENEQA_PARQUET_DIR.name
    archive_dir = raw_root / DEFAULT_OPENEQA_ARCHIVE_DIR.name
    episodes_dir = raw_root / DEFAULT_OPENEQA_EPISODES_DIR.name
    parquet_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    manifest = download_openeqa_assets(
        repo_id=args.repo_id,
        parquet_filename=args.parquet_filename,
        archive_filenames=[] if args.skip_archives else list(args.archive_filenames),
        parquet_dir=parquet_dir,
        archive_dir=archive_dir,
        episodes_dir=episodes_dir,
        force_extract=args.force_extract,
    )
    manifest_path = raw_root / DEFAULT_OPENEQA_DOWNLOAD_MANIFEST.name
    write_json(manifest_path, manifest)

    print("OpenEQA download finished")
    print(f"Manifest: {manifest_path}")
    print(f"Parquet: {manifest['parquet']['local_path']}")
    print(f"Archives downloaded: {len(manifest['archives'])}")


def download_openeqa_assets(
    *,
    repo_id: str,
    parquet_filename: str,
    archive_filenames: list[str],
    parquet_dir: Path,
    archive_dir: Path,
    episodes_dir: Path,
    force_extract: bool,
) -> dict[str, object]:
    """下载 OpenEQA 所需文件并按目录落盘。"""
    parquet_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=parquet_filename,
            local_dir=str(parquet_dir),
        )
    )

    archive_entries: list[dict[str, object]] = []
    for filename in archive_filenames:
        archive_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(archive_dir),
            )
        )
        extracted_root = episodes_dir / filename.replace(".tar.gz", "")
        _extract_tar_if_needed(archive_path, episodes_dir, extracted_root, force_extract=force_extract)
        archive_entries.append(
            {
                "filename": filename,
                "local_path": str(archive_path.resolve()),
                "size_bytes": archive_path.stat().st_size,
                "extracted_root": str(extracted_root.resolve()),
                "extracted_exists": extracted_root.exists(),
            }
        )

    return {
        "generated_at": now_utc_iso(),
        "repo_id": repo_id,
        "raw_root": str(parquet_dir.parent.resolve()),
        "parquet": {
            "filename": parquet_filename,
            "local_path": str(parquet_path.resolve()),
            "size_bytes": parquet_path.stat().st_size,
        },
        "archives": archive_entries,
        "episodes_dir": str(episodes_dir.resolve()),
    }


def _extract_tar_if_needed(
    archive_path: Path,
    episodes_dir: Path,
    extracted_root: Path,
    *,
    force_extract: bool,
) -> None:
    """解压 tar.gz，并做基本的路径安全检查。"""
    if extracted_root.exists() and not force_extract:
        return

    with tarfile.open(archive_path, mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (episodes_dir / member.name).resolve()
            if episodes_dir.resolve() not in member_path.parents and member_path != episodes_dir.resolve():
                raise RuntimeError(f"Unsafe tar member path detected: {member.name}")
        tar.extractall(path=episodes_dir)


if __name__ == "__main__":
    main()
