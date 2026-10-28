from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_downloads import project_archive
from share_access import token_matches


def test_complete_download_has_original_report_code_and_results():
    with ZipFile(BytesIO(project_archive())) as archive:
        prefix = "qf5210-12-etfs/"
        expected = json.loads((ROOT / "download_manifest.json").read_text())
        assert set(archive.namelist()) == {prefix + name for name in expected}
        for name in expected:
            assert archive.read(prefix + name) == (ROOT / name).read_bytes()
        assert prefix + "QF5210_Report.docx" in archive.namelist()
        assert prefix + "Strategy Code/output_four_models/performance_metrics.csv" in archive.namelist()


def test_download_uses_allowlist_and_excludes_runtime_files(tmp_path):
    (tmp_path / "README.md").write_text("Project")
    (tmp_path / ".env").write_text("not for download")
    (tmp_path / ".streamlit").mkdir()
    (tmp_path / ".streamlit/secrets.toml").write_text("not for download")
    (tmp_path / "download_manifest.json").write_text(json.dumps(["README.md"]))
    with ZipFile(BytesIO(project_archive(tmp_path))) as archive:
        assert archive.namelist() == ["qf5210-12-etfs/README.md"]


@pytest.mark.parametrize("bad_path", ["../README.md", "/README.md", ".env", ".streamlit/secrets.toml", ".git/config"])
def test_download_rejects_unsafe_manifest_paths(tmp_path, bad_path):
    (tmp_path / "download_manifest.json").write_text(json.dumps([bad_path]))
    with pytest.raises(ValueError):
        project_archive(tmp_path)


def test_download_rejects_symlinks(tmp_path):
    (tmp_path / "target.txt").write_text("Project")
    (tmp_path / "link.txt").symlink_to(tmp_path / "target.txt")
    (tmp_path / "download_manifest.json").write_text(json.dumps(["link.txt"]))
    with pytest.raises(ValueError, match="Symlinks"):
        project_archive(tmp_path)


@pytest.mark.parametrize("expected,supplied,result", [("a"*48, "a"*48, True),
    ("short", "short", False), (None, "", False), ("a"*48, "wrong", False),
    ("a"*48, "中文", False), ("a"*48, ["a"*48], False)])
def test_link_token_validation(expected, supplied, result):
    assert token_matches(expected, supplied) is result


@pytest.mark.parametrize("configured,supplied", [(False, None), (True, None), (True, "wrong"), (True, "中文")])
def test_hosted_content_is_blocked_without_valid_link(monkeypatch, configured, supplied):
    monkeypatch.delenv("QF5210_LOCAL_MODE", raising=False)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=40)
    if configured:
        app.secrets["share_token"] = "test-only-token-" * 4
    if supplied is not None:
        app.query_params["access"] = supplied
    app.run()
    assert not app.exception
    assert not app.metric
    assert not app.selectbox
    assert len(app.info) == 1
    assert len(app.get("download_button")) == 0


def test_valid_link_opens_dashboard_and_project_download(monkeypatch):
    monkeypatch.delenv("QF5210_LOCAL_MODE", raising=False)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=40)
    app.secrets["share_token"] = "test-only-token-" * 4
    app.query_params["access"] = "test-only-token-" * 4
    app.run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) >= 8
    downloads = app.get("download_button")
    assert any(d.label == "Download full project (ZIP)" for d in downloads)
