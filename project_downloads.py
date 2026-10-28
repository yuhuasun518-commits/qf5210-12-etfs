"""Package only explicitly listed project files, never runtime secrets or git history."""
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent


def project_archive(root=ROOT):
    root = Path(root).resolve()
    files = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(files, list) or len(files) != len(set(files)):
        raise ValueError("The project download manifest must be a unique list of paths.")
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for relative in files:
            part = PurePosixPath(relative)
            if part.is_absolute() or ".." in part.parts or "\\" in relative:
                raise ValueError("Invalid project download path.")
            if any(p in {".git", ".venv", "__pycache__", "secrets.toml"} or p.startswith(".env") for p in part.parts):
                raise ValueError("Private runtime files cannot be downloaded.")
            path = root.joinpath(*part.parts)
            if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
                raise ValueError("Symlinks cannot be included in the project download.")
            if not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError(f"A project file is missing: {relative}")
            archive.write(path, "qf5210-12-etfs/" + relative)
    return output.getvalue()
