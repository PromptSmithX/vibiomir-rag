from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from crawler.utils.files import replace_with_retry

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "backups",
    "data",
    "dist",
    "logs",
    "partitions",
    "partitions_v2",
    "tests",
}
ROOT_FILES = ("pyproject.toml", "uv.lock", "README.md")
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _requirements(project_root: Path) -> bytes:
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-header",
        "--no-annotate",
        "--no-hashes",
    ]
    try:
        environment = dict(os.environ)
        environment["UV_CACHE_DIR"] = str(project_root / ".uv-cache")
        result = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("uv is required to export Kaggle dependencies") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"uv export failed: {exc.stderr.strip()}") from exc
    value = result.stdout.strip() + "\n"
    if not value.strip():
        raise RuntimeError("uv export produced an empty requirements file")
    return value.encode("utf-8")


def _source_files(project_root: Path) -> list[Path]:
    files = [path for path in (project_root / "crawler").rglob("*.py") if path.is_file()]
    files.append(project_root / "config" / "crawler.yaml")
    files.extend(project_root / name for name in ROOT_FILES)
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"bundle inputs are missing: {missing}")
    return sorted(files, key=lambda path: path.relative_to(project_root).as_posix())


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build_bundle(project_root: Path, output: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, bytes] = {}
    for path in _source_files(project_root):
        archive_name = path.relative_to(project_root).as_posix()
        parts = set(PurePosixPath(archive_name).parts)
        if parts & FORBIDDEN_PARTS:
            raise RuntimeError(f"forbidden path selected for bundle: {archive_name}")
        entries[archive_name] = path.read_bytes()
    entries["requirements-kaggle.txt"] = _requirements(project_root)

    manifest = {
        "bundle_format": 1,
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in sorted(entries.items())
        },
    }
    entries["bundle_manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(_zip_info(name), data)
    replace_with_retry(temporary, output)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        for name in names:
            if set(PurePosixPath(name).parts) & FORBIDDEN_PARTS:
                raise RuntimeError(f"forbidden output found in bundle: {name}")
        if set(names) != set(entries):
            raise RuntimeError("bundle verification found a mismatched file list")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"bundle contains a corrupt member: {bad}")
    return {
        "output": str(output),
        "files": len(entries),
        "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the ViBioMIR Kaggle crawler bundle")
    parser.add_argument("--project-root", default=Path(__file__).parent, type=Path)
    parser.add_argument("--output", default=Path("dist/crawler_bundle.zip"), type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(build_bundle(args.project_root, args.output), indent=2))


if __name__ == "__main__":
    main()
