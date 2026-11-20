"""End-to-end: mock GitHub + stub hf CLI + (optionally) injected failures, real training on a tiny model."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mocks  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
COPY = ["mad", "mad.py", "mi_run.py", "mi_tools.py", "hf_sync.py", "hf_auth.py", "fetch_github_data.py", "fetch_hf_data.py",
        "fetch_hf_cli.py", "prepare_data.py", "tests/flaky_mad.py"]


@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "MI"
    (root / "tests").mkdir(parents=True)
    for rel in COPY:
        s = SRC / rel
        d = root / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__")) if s.is_dir() else shutil.copy2(s, d)
    (root / "tokenizer").mkdir()
    srv, url = mocks.start_github_mock()
    mocks.install_hf_stub(tmp_path / "bin")
    env = {**os.environ, "MI_GITHUB_API": url, "MI_GITHUB_CODELOAD": url, "MI_SEARCH_DELAY": "0",
           "MI_GH_SEEDS": "good/apache-app,good/mit-lib,good/sniffed,evil/gpl-app,evil/unlicensed,gone/missing",
           "MI_MIN_RAW_BYTES": "20000", "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
           "HF_STUB_DIR": str(tmp_path / "hfstub"), "HF_TOKEN": "hf_" + "a" * 30,
           "MI_FLAKY_DIR": str(tmp_path / "flaky")}
    yield root, env, tmp_path
    srv.shutdown()


def run_mi(root, env, *extra, flaky=None, timeout=600):
    e = dict(env)
    if flaky:
        e["MI_FLAKY"] = flaky
        e["MI_MAD_SCRIPT"] = str(root / "tests/flaky_mad.py")
    args = [sys.executable, "mi_run.py", "--preset", "tiny", "--steps", "80", "--sft-steps", "30",
            "--no-hf-data", "--mb", "1", "--chunk-minutes", "0.05", "--epochs", "1",
            "--hf-repo", "tester/MI-test", *extra]
    return subprocess.run(args, cwd=root, env=e, capture_output=True, text=True, timeout=timeout)


def test_full_run_with_backup_and_resume(project):
    root, env, tmp = project
    r = run_mi(root, env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    # licensing guard rails
    raw = {p.name for p in (root / "data/raw").glob("*.jsonl")}
    assert raw == {"gh_good__apache-app.jsonl", "gh_good__mit-lib.jsonl", "gh_good__sniffed.jsonl"}, raw
    assert "build/generated" not in "".join(p.read_text() for p in (root / "data/raw").glob("*.jsonl"))
    assert (root / "data/licenses/good__apache-app.txt").is_file()
    state = json.loads((root / "state/pipeline.json").read_text())
    assert state["done"]["pretrain"] and state["done"]["sft"] and state["done"]["finished"]
    assert (root / "checkpoints/mad_sft_best.pt").is_file() or (root / "checkpoints/mad_sft_latest.pt").is_file()
    # frequent HF commits, repo created private, files present in the (fake) repo
    calls = (tmp / "hfstub/calls.log").read_text()
    assert "repos create tester/MI-test --type model --exist-ok --private" in calls
    assert calls.count("upload tester/MI-test") >= 2
    repo = tmp / "hfstub/repo"
    for rel in ("checkpoints/mad_latest.pt", "checkpoints/mad_sft_latest.pt", "config.json", "tokenizer/mad_bpe.json",
                "state/pipeline.json", "data/THIRD_PARTY_NOTICES.md", "README.md"):
        assert (repo / rel).is_file(), rel
    assert "evil/gpl-app" not in (repo / "data/THIRD_PARTY_NOTICES.md").read_text()
    assert "hf_" + "a" * 30 not in "".join(p.read_text(errors="ignore") for p in repo.rglob("*") if p.suffix in (".log", ".md", ".json"))
    # losing every local model file -> a re-run restores from the HF repo instead of starting over
    for rel in ("checkpoints", "state", "config.json", "tokenizer"):
        p = root / rel
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    (root / "tokenizer").mkdir()
    r2 = run_mi(root, env)
    assert r2.returncode == 0, (r2.stdout + r2.stderr)[-3000:]
    assert "restored from tester/MI-test" in r2.stdout
    assert (root / "checkpoints/mad_sft_latest.pt").is_file() and (root / "config.json").is_file()
    assert "== pretrain: step" not in r2.stdout  # the backup said the run was already finished


@pytest.mark.parametrize("flaky,needle", [
    ("nan:1", "learning rate ->"),
    ("inf:1", "infinite gradients"),
    ("oom:1", "out of memory; batch"),
    ("kill:1", "out of memory"),
    ("unknown:1", "unrecognised failure"),
    ("corrupt:1", "checkpoint unreadable"),
])
def test_self_healing(project, flaky, needle):
    root, env, _ = project
    r = run_mi(root, env, "--no-sft", flaky=flaky, timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert needle in (root / "reports/heals.md").read_text(), out[-2000:]
    state = json.loads((root / "state/pipeline.json").read_text())
    assert state["done"]["pretrain"]


def test_hf_down_never_stops_training(project):
    root, env, _ = project
    env = {**env, "HF_STUB_FAIL_UPLOAD": "1"}
    r = run_mi(root, env, "--no-sft")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "Hugging Face push failed" in r.stdout


def test_no_login_trains_locally(project):
    root, env, _ = project
    env = {**env, "HF_STUB_LOGGED_OUT": "1"}
    r = run_mi(root, env, "--no-sft")
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "not logged in" in r.stdout


def test_live_refresh_during_training(project):
    root, env, _ = project
    r = run_mi(root, env, "--no-sft", "--refresh-hours", "0.00001", "--steps", "120")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "live data refresh (scheduled)" in out and "training continues" in out
    assert "What the model trains on" in out and ".kt" in out
