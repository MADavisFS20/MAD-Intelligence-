#!/usr/bin/env python3
"""Back up and restore the project on a Hugging Face model repo, using the `hf` CLI (stdlib only).

Used by mi_run.py after every training chunk so a crash, reboot or lost phone never loses more than
a few minutes. Every failure here is non-fatal: training always continues locally.

What is pushed (one atomic commit per push):
  checkpoints/*.pt  training.jsonl  mad_sft.jsonl  config.json  tokenizer/mad_bpe.json
  state/pipeline.json  data/manifest.json  data/THIRD_PARTY_NOTICES.md  data/licenses/*
  logs/mi_run.log (tail, secrets redacted)  reports/*  README.md (model card)
Raw third-party code is NOT uploaded (re-fetched from its original source when needed).

CLI:
  python3 hf_sync.py status            # who am I, which repo, does it exist
  python3 hf_sync.py push [-m MSG]     # push now
  python3 hf_sync.py pull [--force]    # restore missing files from the repo
Environment: HF_TOKEN (needs write access), MI_HF_REPO (owner/name), MI_HF_PUBLIC=1
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_NAME = "MI-android-coder"
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
HEAVY = ["checkpoints/mad_latest.pt", "checkpoints/mad_best.pt",
         "checkpoints/mad_sft_latest.pt", "checkpoints/mad_sft_best.pt"]
LIGHT = ["checkpoints/training.jsonl", "checkpoints/mad_sft.jsonl", "config.json",
         "tokenizer/mad_bpe.json", "state/pipeline.json"]
RESTORE_MAP = HEAVY + LIGHT  # repo path == local path


def redact(text):
    return TOKEN_RE.sub("<redacted>", text)


class HFSync:
    def __init__(self, root=ROOT, repo=None, private=True, enabled=True, log=print):
        self.root = Path(root)
        self.repo = repo or os.environ.get("MI_HF_REPO")
        self.private = private and os.environ.get("MI_HF_PUBLIC") != "1"
        self.enabled = enabled
        self.log = log
        self.ready = False
        self.failures = 0
        self.last_push = 0.0
        self.user = None

    # ------------------------------------------------------------------ plumbing
    def hf(self, *args, timeout=1800):
        exe = shutil.which("hf")
        if not exe:
            return 127, "hf command not found"
        try:
            p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "HF_HUB_DISABLE_XET": "1"})  # hf_xet has no Termux build
        except subprocess.TimeoutExpired:
            return 124, "timed out"
        return p.returncode, redact((p.stdout or "") + (p.stderr or ""))

    def install_cli(self):
        if shutil.which("hf"):
            return True
        self.log("hf CLI missing; installing huggingface_hub with pip ...")
        for extra in ([], ["--break-system-packages"], ["--user"]):
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", *extra],
                               capture_output=True, text=True)
            if r.returncode == 0 and shutil.which("hf"):
                return True
        self.log("could not install the hf CLI automatically (run: pip install -U huggingface_hub)")
        return False

    def whoami(self):
        rc, text = self.hf("auth", "whoami", "--format", "json", timeout=60)
        if rc != 0 or "not logged in" in text.lower():
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                for key in ("name", "user", "username", "login"):
                    if isinstance(obj.get(key), str) and obj[key]:
                        return obj[key]
        except json.JSONDecodeError:
            pass
        m = re.search(r'(?im)^\W*(?:user|name|username)\W+([\w.-]+)', text)
        if m:
            return m.group(1)
        rc, text = self.hf("auth", "whoami", "--quiet", timeout=60)
        word = text.split()[0] if rc == 0 and text.split() else None
        return word if word and re.fullmatch(r"[\w.-]+", word) else None

    # ------------------------------------------------------------------ setup
    def ensure(self):
        """Resolve the repo id and create the repo if needed. Returns True when pushes can work."""
        if not self.enabled:
            return False
        if self.ready:
            return True
        if not self.install_cli():
            return False
        self.user = self.whoami()
        if not self.user:
            self.log("Hugging Face: not logged in (set HF_TOKEN or run `hf auth login`). "
                     "Training continues locally without cloud backup.")
            return False
        if not self.repo:
            self.repo = f"{self.user}/{DEFAULT_NAME}"
        args = ["repos", "create", self.repo, "--type", "model", "--exist-ok",
                "--private" if self.private else "--public"]
        rc, text = self.hf(*args, timeout=120)
        if rc != 0:
            self.log(f"Hugging Face: could not create/open {self.repo}: {text.strip()[-300:]}\n"
                     f"  Your token needs WRITE access (fine-grained tokens: permission to create and write repos "
                     f"under '{self.user}'). Create one at https://huggingface.co/settings/tokens, then `hf auth login`. "
                     f"Training continues locally meanwhile.")
            return False
        self.ready = True
        self.announce()
        return True

    @property
    def url(self):
        return f"https://huggingface.co/{self.repo}" if self.repo else None

    def announce(self):
        """Print (once per run) and save the link to the cloud backup repo."""
        kind = "PRIVATE" if self.private else "PUBLIC"
        lines = ["=" * 64, f" Cloud backup: {kind} model repo for account '{self.user}'",
                 f"   Repo:     {self.url}", f"   Files:    {self.url}/tree/main",
                 f"   Commits:  {self.url}/commits/main",
                 "   (open while signed in as that account; private repos show 404 to everyone else)", "=" * 64]
        for line in lines:
            self.log(line)
        try:
            (self.root / "HF_BACKUP_LINK.txt").write_text(
                f"{self.url}\n{self.url}/tree/main\n{self.url}/commits/main\nvisibility: {kind.lower()}\n"
                f"account: {self.user}\n", encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------ push
    def _link(self, src, dst):
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except (AttributeError, OSError):
            shutil.copy2(src, dst)

    def stage(self, heavy=True, card=None):
        stage = self.root / f".hf_stage-{os.getpid()}"  # per process: a manual push can run during training
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        files = (HEAVY if heavy else []) + LIGHT
        for rel in files:
            src = self.root / rel
            if src.is_file():
                # checkpoints are COPIED, not hard-linked: a hard link shares the inode, so a trainer that
                # rewrites the file in place would corrupt the upload while it is in flight
                if rel.endswith(".pt"):
                    (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, stage / rel)
                else:
                    self._link(src, stage / rel)
        extra = {"data/processed/manifest.json": "data/manifest.json",
                 "data/THIRD_PARTY_NOTICES.md": "data/THIRD_PARTY_NOTICES.md"}
        for src_rel, dst_rel in extra.items():
            if (self.root / src_rel).is_file():
                self._link(self.root / src_rel, stage / dst_rel)
        lic = self.root / "data/licenses"
        if lic.is_dir():
            for f in lic.glob("*.txt"):
                self._link(f, stage / "data/licenses" / f.name)
        reports = self.root / "reports"
        if reports.is_dir():
            for f in reports.glob("*.md"):
                self._link(f, stage / "reports" / f.name)
        log = self.root / "logs/mi_run.log"
        if log.is_file():
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
            (stage / "logs").mkdir(exist_ok=True)
            (stage / "logs/mi_run.log").write_text(redact("\n".join(tail)) + "\n", encoding="utf-8")
        if card:
            (stage / "README.md").write_text(card, encoding="utf-8")
        return stage

    def push(self, message, heavy=True, card=None):
        """Commit the current state to the Hub. Never raises; returns True on success."""
        if not self.ensure():
            return False
        stage = self.stage(heavy, card)
        try:
            rc, text = self.hf("upload", self.repo, str(stage), ".", "--type", "model",
                               "--commit-message", message[:200])
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        if rc != 0:
            self.failures += 1
            self.log(f"Hugging Face push failed ({self.failures} so far; will retry next time): {text.strip()[-300:]}")
            return False
        self.failures = 0
        self.last_push = time.time()
        self.log(f"pushed to {self.repo}: {message}")
        return True

    # ------------------------------------------------------------------ pull
    def pull(self, force=False):
        """Restore files that are missing locally (or all, with force). Returns the list restored."""
        if not self.ensure():
            return []
        tmp = self.root / ".hf_restore"
        shutil.rmtree(tmp, ignore_errors=True)
        rc, text = self.hf("download", self.repo, "--type", "model", "--local-dir", str(tmp))
        restored = []
        if rc == 0:
            for rel in RESTORE_MAP:
                src, dst = tmp / rel, self.root / rel
                if src.is_file() and (force or not dst.is_file()):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    restored.append(rel)
        else:
            self.log(f"nothing restored from {self.repo}: {text.strip()[-200:]}")
        shutil.rmtree(tmp, ignore_errors=True)
        if restored:
            self.log(f"restored from {self.repo}: {', '.join(restored)}")
        return restored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("push")
    p.add_argument("-m", "--message", default="manual push")
    p = sub.add_parser("pull")
    p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    h = HFSync()
    if args.cmd == "status":
        ok = h.ensure()
        print(json.dumps({"logged_in_as": h.user, "repo": h.repo, "ready": ok, "private": h.private}))
        return 0 if ok else 1
    if args.cmd == "push":
        return 0 if h.push(args.message) else 1
    h.pull(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
