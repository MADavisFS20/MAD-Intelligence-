import json

import pytest
import torch

from mad import data
from mad.cli import build_parser
from mad.cli import main as cli_main

KOTLIN = """package com.example.app

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }
}
"""


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(6):
        (raw / f"Screen{i}.kt").write_text(KOTLIN.replace("MainActivity", f"Screen{i}Activity") * 40)
    (raw / "layout.xml").write_text('<LinearLayout android:id="@+id/root" />\n' * 200)
    (raw / "dup.kt").write_text((raw / "Screen0.kt").read_text())          # duplicate content
    (raw / "image.png").write_bytes(b"\x89PNG\r\n")                        # unsupported
    (raw / ".git").mkdir()
    (raw / ".git" / "config.txt").write_text("should be skipped")
    (raw / "blob.json").write_bytes(bytes(range(256)) * 40)                # binary-looking
    return tmp_path


def run(*argv):
    cli_main([str(a) for a in argv])


def test_collect_files_filters(project):
    records, skipped = data.collect_files(project / "raw")
    names = {r["path"].rsplit("/", 1)[-1] for r in records}
    assert names == {f"Screen{i}.kt" for i in range(6)} | {"layout.xml"}
    reasons = {why for _, why in skipped}
    assert "duplicate content" in reasons and "looks binary" in reasons


def test_full_pipeline_and_resume(project, capsys):
    raw, proc = project / "raw", project / "processed"
    cfg, tok, ck = project / "config.json", project / "tok.json", project / "ck"
    corpus = proc / "corpus.txt"
    run("init", "--config", cfg, "--force", "--checkpoint-dir", project / "ck0")
    run("prepare-data", "--input", raw, "--output", corpus)
    manifest = json.loads((proc / "manifest.json").read_text())
    assert len(manifest["files"]) == 7 and manifest["corpus_sha256"] == data.sha256_file(corpus)
    # user-edited source notes survive a re-run
    manifest["files"][0]["source"] = "my own app"
    (proc / "manifest.json").write_text(json.dumps(manifest))
    run("prepare-data", "--input", raw, "--output", corpus)
    assert json.loads((proc / "manifest.json").read_text())["files"][0]["source"] == "my own app"

    run("train-tokenizer", "--corpus", corpus, "--output", tok, "--vocab-size", 600, "--config", cfg)
    assert json.loads(cfg.read_text())["vocab_size"] <= 600  # config synced to tokenizer

    base = ["train", "--config", cfg, "--tokenizer", tok, "--corpus", corpus, "--output-dir", ck,
            "--batch-size", 2, "--seq-len", 32, "--grad-accum", 1, "--eval-every", 5,
            "--eval-batches", 2, "--checkpoint-every", 5, "--threads", 1]
    run(*base, "--steps", 10)
    assert (ck / "mad_latest.pt").is_file() and (ck / "mad_best.pt").is_file()
    rows = [json.loads(x) for x in (ck / "training.jsonl").read_text().splitlines()]
    assert rows[-1]["step"] == 10 and any("val_loss" in r for r in rows)
    assert all(math_ok(r["loss"]) for r in rows)

    run(*base, "--steps", 14, "--resume", ck / "mad_latest.pt")
    assert json.loads((ck / "training.jsonl").read_text().splitlines()[-1])["step"] == 14

    run("eval", "--checkpoint", ck / "mad_latest.pt", "--tokenizer", tok, "--corpus", corpus,
        "--batches", 2, "--seq-len", 32, "--threads", 1)
    assert "held-out loss" in capsys.readouterr().out
    run("generate", "class ", "--checkpoint", ck / "mad_latest.pt", "--tokenizer", tok,
        "--max-new-tokens", 5, "--seed", 1, "--threads", 1)
    run("status", "--config", cfg, "--corpus", corpus, "--tokenizer", tok)
    assert "tokenizer" in capsys.readouterr().out


def math_ok(x):
    return x == x and abs(x) < 1e6


def test_time_limit_stops_and_saves(project):
    raw, proc = project / "raw", project / "processed"
    cfg, tok, ck, corpus = project / "c.json", project / "t.json", project / "ck", proc / "corpus.txt"
    run("init", "--config", cfg, "--force", "--checkpoint-dir", project / "ck0")
    run("prepare-data", "--input", raw, "--output", corpus)
    run("train-tokenizer", "--corpus", corpus, "--output", tok, "--vocab-size", 500, "--config", cfg)
    run("train", "--config", cfg, "--tokenizer", tok, "--corpus", corpus, "--output-dir", ck,
        "--steps", 100000, "--batch-size", 2, "--seq-len", 32, "--grad-accum", 1,
        "--max-minutes", 0.02, "--threads", 1, "--eval-every", 0)
    state = torch.load(ck / "mad_latest.pt", weights_only=True)
    assert 0 < state["step"] < 100000


def test_errors_are_friendly(project):
    with pytest.raises(SystemExit, match="prepare-data"):
        run("train-tokenizer", "--corpus", project / "nope.txt")
    with pytest.raises(SystemExit, match="init"):
        run("train", "--config", project / "nope.json")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bogus"])


def test_secret_scan_skips_keys_and_passwords(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ok.kt").write_text("class Login { var password: String = \"\"\n fun check(password: String) = password.length > 3 }")
    (raw / "google-services.json").write_text('{"api_key": [{"current_key": "AIza' + "a" * 35 + '"}]}')
    (raw / "build.gradle").write_text('signingConfigs { release { storePassword "hunter2hunter2" } }\nstorePassword = "hunter2hunter2"')
    (raw / "key.txt").write_text("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----")
    records, skipped = data.collect_files(raw)
    assert [r["path"].rsplit("/", 1)[-1] for r in records] == ["ok.kt"]
    assert sum(why.startswith("possible secret") for _, why in skipped) == 3
    records, _ = data.collect_files(raw, scan_secrets=False)
    assert len(records) == 4
