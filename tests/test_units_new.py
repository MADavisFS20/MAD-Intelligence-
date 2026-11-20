import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import fetch_github_data as gh  # noqa: E402
import hf_sync  # noqa: E402
import mocks  # noqa: E402
os.environ.setdefault("MI_MIN_RAW_BYTES", "1")
import mi_run  # noqa: E402


@pytest.mark.parametrize("tail,rc,kind", [
    ("Training stopped: Loss became nan at step 3.", 1, "nan"),
    ("", 137, "oom"),
    ("DefaultCPUAllocator: can't allocate memory", 1, "oom"),
    ("OSError: [Errno 28] No space left on device", 1, "disk"),
    ("Could not read checkpoint x.pt: bad", 1, "bad_checkpoint"),
    ("Checkpoint architecture does not match config.json", 1, "arch_mismatch"),
    ("Tokenizer size 900 does not match model vocab_size 1024", 1, "vocab_mismatch"),
    ("Corpus has 10 training tokens; need at least 258.", 1, "small_corpus"),
    ("ModuleNotFoundError: No module named 'torch'", 1, "no_torch"),
    ("ModuleNotFoundError: No module named 'numpy'", 1, "missing_module"),
    ("Corpus not found: x. Run `python mad.py prepare-data` first.", 1, "missing_input"),
    ("urllib.error.URLError: <urlopen error Name resolution failed>", 1, "network"),
    ("something odd", 1, "unknown"),
])
def test_classify(tail, rc, kind):
    assert mi_run.classify(tail, rc) == kind


def test_license_detection():
    assert gh.detect_license(mocks.APACHE) == "apache-2.0"
    assert gh.detect_license(mocks.MIT) == "mit"
    assert gh.detect_license(mocks.GPL) is None
    assert gh.detect_license(mocks.MIT + mocks.APACHE) is None      # dual/mixed -> refuse to guess
    assert gh.detect_license(mocks.MIT + "GNU General Public License") is None
    assert gh.detect_license("") is None


def test_path_filter():
    w = gh.wanted_path
    assert w("app/src/main/java/A.kt") == ".kt"
    assert w("app/build/generated/A.kt") is None
    assert w("app/src/main/res/layout/main.xml") == ".xml"
    assert w("app/src/main/res/drawable/vec.xml") is None
    assert w("app/src/main/AndroidManifest.xml") == ".xml"
    assert w("docs/readme.md") is None
    assert w("gradle/libs.versions.toml") == ".toml" and w("pyproject.toml") is None
    assert w("app/proguard-rules.pro") == ".pro" and w("random.pro") is None
    assert w("app/src/main/res/xml/backup_rules.xml") == ".xml"
    assert w("app/src/main/res/values-night/themes.xml") == ".xml"
    assert w("app/src/main/res/values-es/strings.xml") is None   # translations skipped
    assert w("app/src/main/res/values/strings.xml") == ".xml"
    assert w("app/src/main/aidl/com/x/IFoo.aidl") == ".aidl"


def test_github_rejects_bad_licenses(tmp_path):
    srv, url = mocks.start_github_mock()
    gh.API = gh.CODELOAD = url
    gh.SEED_REPOS[:] = ["good/apache-app", "evil/gpl-app", "evil/unlicensed", "gone/missing", "good/sniffed"]
    try:
        rc = gh.main(["--raw", str(tmp_path / "raw"), "--licenses", str(tmp_path / "lic"),
                      "--state", str(tmp_path / "s.json"), "--seeds-only", "--mb", "5"])
    finally:
        srv.shutdown()
    st = json.loads((tmp_path / "s.json").read_text())
    assert rc == 0 and set(st["done"]) == {"good/apache-app", "good/sniffed"}
    assert st["done"]["good/sniffed"]["license"] == "mit"
    assert st["skipped"]["evil/gpl-app"].startswith("license")
    assert st["skipped"]["gone/missing"] == "not found"
    first = json.loads((tmp_path / "raw/gh_good__apache-app.jsonl").read_text().splitlines()[0])
    assert first["license"] == "apache-2.0" and first["source"].startswith("github:good/apache-app@")


def test_val_trend(tmp_path):
    log = tmp_path / "t.jsonl"
    rows = [{"step": i * 50, "val_loss": v} for i, v in enumerate([3.0, 2.5, 2.4] + [2.6] * 13, 1)]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    out = subprocess.run([sys.executable, str(ROOT / "mi_tools.py"), "val-trend", "--log", str(log), "--patience", "12"],
                         capture_output=True, text=True)
    res = json.loads(out.stdout)
    assert res["stop"] is True and res["best_step"] == 150


def test_hf_stage_excludes_raw_data_and_redacts(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/gh_x.jsonl").write_text("third party code")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/mad_latest.pt").write_bytes(b"x")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/mi_run.log").write_text("token hf_" + "b" * 30 + " ok\n")
    h = hf_sync.HFSync(root=tmp_path)
    stage = h.stage(heavy=True)
    files = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    assert "checkpoints/mad_latest.pt" in files and not any(f.startswith("data/raw") for f in files)
    assert "hf_" + "b" * 30 not in (stage / "logs/mi_run.log").read_text()
    assert "checkpoints/mad_latest.pt" not in {p.relative_to(h.stage(heavy=False)).as_posix()
                                               for p in h.stage(heavy=False).rglob("*") if p.is_file()}


def test_hf_auth_uses_cli_login_and_gated_message(tmp_path, monkeypatch):
    import hf_auth
    for n in hf_auth.ENV_NAMES:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    hf_auth._cache.clear()
    assert hf_auth.get_token() is None
    assert "hf auth login" in hf_auth.gated_message("bigcode/the-stack-smol")
    (tmp_path / "hfhome").mkdir(exist_ok=True)
    (tmp_path / "hfhome/token").write_text("hf_" + "c" * 30)   # what `hf auth login` leaves behind
    hf_auth._cache.clear()
    assert hf_auth.get_token() == "hf_" + "c" * 30
    msg = hf_auth.gated_message("bigcode/the-stack-smol")
    assert "https://huggingface.co/datasets/bigcode/the-stack-smol" in msg and "accepted" in msg
    assert "c" * 30 not in msg                                  # token never printed
    monkeypatch.setenv("HF_TOKEN", "hf_" + "d" * 30)
    hf_auth._cache.clear()
    assert hf_auth.get_token() == "hf_" + "d" * 30              # env wins


def test_backup_link_announced_and_saved(tmp_path):
    lines = []
    h = hf_sync.HFSync(root=tmp_path, repo="me/MI-android-coder", log=lines.append)
    h.user = "me"
    h.announce()
    text = "\n".join(lines)
    assert "https://huggingface.co/me/MI-android-coder" in text and "PRIVATE" in text
    assert "me/MI-android-coder/commits/main" in (tmp_path / "HF_BACKUP_LINK.txt").read_text()


def test_data_report_lists_types_and_dropped(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    files = {"gh_a.jsonl#1": {"path": "app/A.kt", "bytes": 3000, "source": "github:o/r@abc path=app/A.kt"},
             "gh_a.jsonl#2": {"path": "app/res/layout/m.xml", "bytes": 1000, "source": "github:o/r@abc path=x"},
             "notes.md": {"bytes": 500, "source": ""}}
    (proc / "manifest.json").write_text(json.dumps({"files": files}))
    (proc / "dropped.txt").write_text("a.xyz\tunsupported extension\nb.xyz\tunsupported extension\nc.kt\tduplicate of d\n")
    out = subprocess.run([sys.executable, str(ROOT / "mi_tools.py"), "data-report", "--manifest",
                          str(proc / "manifest.json"), "--no-tokens"], capture_output=True, text=True).stdout
    assert ".kt" in out and ".xml" in out and ".md" in out and "github" in out and "local files" in out
    assert "unsupported file types found in data/raw/: .xyz x2" in out


def test_maybe_refresh_triggers(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(mi_run, "ROOT", tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/a.jsonl").write_text("x" * 5000)
    calls = []
    monkeypatch.setattr(mi_run, "run_fetch", lambda name, cmd, attempts=1: calls.append(name) or True)
    monkeypatch.setattr(mi_run, "stage_prepare", lambda a, st: calls.append("prepare"))
    monkeypatch.setattr(mi_run, "tool_json", lambda *a: {"tokens": 123})
    monkeypatch.setattr(mi_run.State, "save", lambda self: None)
    args = types.SimpleNamespace(refresh_hours=6.0, refresh_mb=10.0, hf_data=True)
    st = mi_run.State()
    st.clear(); st.update(done={}, prepared_raw_bytes=5000, last_refresh=time.time())
    assert mi_run.maybe_refresh(args, st) is False and calls == []          # nothing due
    (tmp_path / "data/raw/mine.kt").write_text("fun a() {}\n" * 200)       # a file added by hand
    assert mi_run.maybe_refresh(args, st) is True and calls == ["prepare"]  # prepares, does not fetch
    calls.clear()
    st["prepared_raw_bytes"] = mi_run.raw_bytes()
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data/REFRESH_NOW").write_text("")
    (tmp_path / "data/raw/new.jsonl").write_text("y" * 4000)
    assert mi_run.maybe_refresh(args, st) is True
    assert calls[:2] == ["github", "huggingface (REST)"] and "prepare" in calls
    assert not (tmp_path / "data/REFRESH_NOW").exists()
    calls.clear(); st["prepared_raw_bytes"] = mi_run.raw_bytes()
    st["last_refresh"] = time.time() - 7 * 3600                             # scheduled refresh due
    assert mi_run.maybe_refresh(args, st) is True and calls[0] == "github"
