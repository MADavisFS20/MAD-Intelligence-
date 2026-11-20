"""Local stand-ins for GitHub and the `hf` CLI so the whole pipeline can be tested offline."""
import io
import json
import os
import stat
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

APACHE = ("Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n"
          "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n")
MIT = "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"
GPL = "GNU General Public License Version 3\n"

REPOS = {  # full_name -> (api spdx or None, license file text, n_files)
    "good/apache-app": ("Apache-2.0", APACHE, 40),
    "good/mit-lib": ("MIT", MIT, 30),
    "good/sniffed": (None, MIT, 20),        # API gives no license; the LICENSE file decides
    "evil/gpl-app": ("GPL-3.0", GPL, 20),   # must be rejected
    "evil/unlicensed": (None, "", 20),      # must be rejected
}


def kotlin_file(repo, i):
    name = repo.split("/")[1].replace("-", "_")
    return (f"package com.example.{name}.f{i}\n\nimport android.os.Bundle\nimport android.widget.Toast\n"
            f"import androidx.appcompat.app.AppCompatActivity\n\n"
            f"/** Shows a short toast with the message number {i} for the {name} screen */\n"
            f"fun showMessage{i}(activity: AppCompatActivity, msg: String) {{\n"
            f"    Toast.makeText(activity, msg + \" {i}\", Toast.LENGTH_SHORT).show()\n}}\n\n"
            f"class Screen{i}Activity : AppCompatActivity() {{\n"
            f"    override fun onCreate(savedInstanceState: Bundle?) {{\n"
            f"        super.onCreate(savedInstanceState)\n        val count = {i}\n"
            f"        showMessage{i}(this, \"opened $count\")\n    }}\n}}\n")


def make_zip(repo):
    _, lic, n = REPOS[repo]
    top = repo.split("/")[1] + "-abc1234"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        if lic:
            z.writestr(f"{top}/LICENSE", lic)
        for i in range(n):
            z.writestr(f"{top}/app/src/main/java/com/example/F{i}.kt", kotlin_file(repo, i))
        z.writestr(f"{top}/app/build/generated/Skip.kt", kotlin_file(repo, 999))  # build/ must be skipped
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        if parts[:1] == ["repos"] and len(parts) == 3:
            full = f"{parts[1]}/{parts[2]}"
            if full not in REPOS:
                return self.reply(404, {"message": "Not Found"})
            spdx = REPOS[full][0]
            return self.reply(200, {"full_name": full, "fork": False, "size": 100,
                                    "license": {"spdx_id": spdx} if spdx else None})
        if parts[:2] == ["search", "repositories"]:
            items = [{"full_name": f, "fork": False, "size": 100, "license": {"spdx_id": REPOS[f][0]}}
                     for f in ("good/apache-app", "evil/gpl-app")] if "page=1" in u.query else []
            return self.reply(200, {"items": items})
        if len(parts) == 4 and parts[2] == "zip":
            full = f"{parts[0]}/{parts[1]}"
            if full in REPOS:
                return self.reply(200, make_zip(full), "application/zip")
        self.reply(404, {"message": "Not Found"})


def start_github_mock():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


HF_STUB = r'''#!/usr/bin/env python3
"""Fake `hf`: 'uploads' copy into $HF_STUB_DIR/repo; 'download' copies back. Logs every call."""
import json, os, shutil, sys
from pathlib import Path
root = Path(os.environ["HF_STUB_DIR"]); repo = root / "repo"; root.mkdir(exist_ok=True)
with (root / "calls.log").open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
a = sys.argv[1:]
if a[:2] == ["auth", "whoami"]:
    if os.environ.get("HF_STUB_LOGGED_OUT"):
        print("Error: Not logged in."); sys.exit(1)
    print(json.dumps({"name": "tester"}))
elif a[:2] == ["repos", "create"]:
    repo.mkdir(exist_ok=True); print("ok")
elif a[0] == "upload":
    if os.environ.get("HF_STUB_FAIL_UPLOAD"):
        print("boom: connection reset", file=sys.stderr); sys.exit(1)
    src = Path(a[2])
    for p in src.rglob("*"):
        if p.is_file():
            dst = repo / p.relative_to(src); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
elif a[0] == "download":
    if not repo.exists() or not any(repo.iterdir()):
        print("Repository not found", file=sys.stderr); sys.exit(1)
    dest = Path(a[a.index("--local-dir") + 1]); shutil.copytree(repo, dest, dirs_exist_ok=True)
else:
    print("stub: unsupported", a); sys.exit(1)
'''


def install_hf_stub(bin_dir):
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "hf"
    exe.write_text(HF_STUB)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe
