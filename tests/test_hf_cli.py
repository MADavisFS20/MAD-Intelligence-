import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_hf_cli as f

SRC = f.FILE_SOURCES[0]
CODE = "package a\nimport android.os.Bundle\n" + "// pad\n" * 40


def row(**kw):
    base = dict(new_file="A.kt", old_contents="", new_contents=CODE, subject="Add main screen",
                license="mit", repos="o/r", commit="c")
    base.update(kw)
    return base


def conv(r, seen=None):
    return f.convert(SRC, r, seen if seen is not None else set(), {})[0]


def test_keeps_permissive_android_row():
    rec = conv(row())
    assert rec["instruction"] == "Add main screen" and rec["license"] == "mit" and rec["output"] == CODE


def test_rejects_bad_rows():
    assert conv(row(license="gpl-3.0")) is None
    assert conv(row(subject="Fix")) is None
    assert conv(row(new_file="A.py")) is None
    assert conv(row(new_contents="x" * 400)) is None  # not android-related


def test_dedupes():
    seen = set()
    assert conv(row(), seen) and conv(row(), seen) is None


def test_http_fallback_and_xet_disabled(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            ok = self.path == "/datasets/o/r/resolve/main/data/kotlin/data.jsonl"
            self.send_response(200 if ok else 404)
            self.end_headers()
            if ok:
                self.wfile.write(b'{"a": 1}\n')

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_port}")
    try:
        dest = tmp_path / "data/kotlin/data.jsonl"
        assert f.http_download("o/r", "data/kotlin/data.jsonl", dest) and dest.read_text() == '{"a": 1}\n'
        assert not f.http_download("o/r", "missing.jsonl", tmp_path / "x") and not (tmp_path / "x.part").exists()
    finally:
        srv.shutdown()


def test_hf_subprocess_disables_xet(tmp_path, monkeypatch):
    stub = tmp_path / "hf"
    stub.write_text('#!/bin/sh\necho "xet=$HF_HUB_DISABLE_XET"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    assert "xet=1" in f.run_hf("anything", capture=True).stdout
