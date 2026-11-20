#!/usr/bin/env python3
"""Populate data/raw/ from Hugging Face datasets with the `hf` CLI (stdlib + the `hf` command).

Complements fetch_hf_data.py (which streams parquet-backed datasets through the REST Datasets
Server). This script covers datasets stored as plain JSONL files, using `hf download`, and lets you
discover more datasets with `hf datasets list`.

Sources (edit FILE_SOURCES below):
  bigcode/commitpackft   MIT dataset of real commits; each row has its own license. Human-written
                         commit message = instruction, file before = input, file after = output.

Only datasets are fetched. No models or checkpoints (the MI independence rules forbid them), and
only rows whose license is on the permissive allowlist are kept.

Usage:
  python3 fetch_hf_cli.py                         # all default sources
  python3 fetch_hf_cli.py --only commitpackft-kotlin
  python3 fetch_hf_cli.py --discover "android kotlin"     # find more datasets (prints license tags)
  python3 fetch_hf_cli.py --card bigcode/commitpackft     # show a dataset card's metadata
  python3 fetch_hf_cli.py --from-dir DIR --only commitpackft-kotlin   # parse files already downloaded

Install the CLI:  pip install -U huggingface_hub   (or: curl -LsSf https://hf.co/cli/install.sh | bash)
Optional login for rate limits / gated data:  export HF_TOKEN=hf_...
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import hf_auth
from fetch_hf_data import ANDROID_RE, MAX_BYTES, MIN_BYTES, PERMISSIVE, licenses_of

FILE_SOURCES = [
    dict(name="commitpackft-kotlin", repo="bigcode/commitpackft", include="data/kotlin/data.jsonl",
         exts={".kt", ".kts"}, android=True, default=True),
    dict(name="commitpackft-java", repo="bigcode/commitpackft", include="data/java/data.jsonl",
         exts={".java"}, android=True, default=False),  # 58 MB download: opt in with --only
]
MIN_INSTRUCTION_WORDS = 3
MAX_INPUT_BYTES = 6_000


def find_hf():
    exe = shutil.which("hf")
    if not exe:
        sys.exit("The `hf` command was not found. Install it with:  pip install -U huggingface_hub\n"
                 "(or: curl -LsSf https://hf.co/cli/install.sh | bash), then re-run.")
    return exe


def run_hf(*args, capture=False):
    cmd = [find_hf(), *args]
    # hf_xet (the "optimized" Xet downloader) has no Android/Termux build; plain HTTP downloads work everywhere.
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, env=env)
    return subprocess.run(cmd, env=env)


def http_download(repo, path, dest):
    """Stdlib fallback when the hf CLI fails: fetch one file from the dataset's resolve URL."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
    headers = {"User-Agent": "mi-fetch/1.0"}
    token = hf_auth.get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  direct download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


def discover(query, limit):
    res = run_hf("datasets", "list", "--search", query, "--sort", "downloads",
                 "--limit", str(limit), "--format", "json", capture=True)
    if res.returncode != 0:
        sys.exit(res.stderr.strip() or "hf datasets list failed")
    try:
        rows = json.loads(res.stdout)
    except json.JSONDecodeError:
        print(res.stdout)
        return
    print(f"{'dataset':52} {'downloads':>10}  license")
    for r in rows if isinstance(r, list) else rows.get("items", []):
        tags = r.get("tags") or []
        lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "?")
        flag = "" if lic in PERMISSIVE else "   <- check before use"
        print(f"{str(r.get('id')):52} {r.get('downloads', 0):>10}  {lic}{flag}")
    print("\nInspect one with: python3 fetch_hf_cli.py --card OWNER/NAME")


def convert(src, row, seen, stats):
    path = row.get("new_file") or row.get("old_file") or ""
    ext = os.path.splitext(path)[1].lower()
    if ext not in src["exts"]:
        stats["other extension"] = stats.get("other extension", 0) + 1
        return None, None
    new, old = row.get("new_contents"), row.get("old_contents") or ""
    subject = (row.get("subject") or "").strip()
    reason = None
    if not isinstance(new, str):
        reason = "no content"
    elif not MIN_BYTES <= len(new.encode("utf-8")) <= MAX_BYTES:
        reason = "size out of range"
    elif len(subject.split()) < MIN_INSTRUCTION_WORDS:
        reason = "message too short"
    elif len(old.encode("utf-8")) > MAX_INPUT_BYTES:
        reason = "input too large"
    else:
        lics = licenses_of(row.get("license"))
        if not lics or any(l not in PERMISSIVE for l in lics):
            reason = "license: " + (",".join(lics) or "none")
        elif src["android"] and not ANDROID_RE.search(new):
            reason = "not android-related"
        else:
            digest = hashlib.sha256(new.encode("utf-8")).hexdigest()
            if digest in seen:
                reason = "duplicate"
            else:
                seen.add(digest)
    if reason:
        stats[reason] = stats.get(reason, 0) + 1
        return None, None
    return {
        "source_path": path, "extension": ext.lstrip("."),
        "instruction": subject, "input": old, "output": new,
        "license": ",".join(lics),
        "source": f"hf:{src['repo']} repos={row.get('repos', '')} commit={row.get('commit', '')} path={path}",
    }, None


def process(src, jsonl, raw_dir, budget, seen):
    out_path = raw_dir / f"hf_{src['name']}.jsonl"
    written = files = 0
    stats = {}
    with jsonl.open(encoding="utf-8", errors="replace") as fh, out_path.open("w", encoding="utf-8") as out:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad json"] = stats.get("bad json", 0) + 1
                continue
            rec, _ = convert(src, row, seen, stats)
            if not rec:
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += len(rec["output"].encode("utf-8"))
            files += 1
            if written >= budget:
                break
    if files == 0:
        out_path.unlink(missing_ok=True)
    print(f"  {src['name']}: wrote {files} records, {written/1e6:.2f} MB -> {out_path}")
    rej = sorted(stats.items(), key=lambda kv: -kv[1])[:6]
    if rej:
        print("  rejected: " + "; ".join(f"{k} x{v}" for k, v in rej))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/hf_cache", help="where `hf download` puts files")
    ap.add_argument("--mb", type=float, default=5.0, help="max MB of code to keep per source")
    ap.add_argument("--only", nargs="*", help="source names to fetch")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded files after parsing")
    ap.add_argument("--from-dir", help="parse already-downloaded files from this folder (no download)")
    ap.add_argument("--discover", metavar="QUERY", help="search Hub datasets and show their licenses")
    ap.add_argument("--card", metavar="DATASET", help="print a dataset's card metadata")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.discover:
        return discover(args.discover, args.limit)
    if args.card:
        return sys.exit(run_hf("datasets", "card", args.card, "--metadata").returncode)

    raw_dir = Path(args.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for src in FILE_SOURCES:
        if (args.only and src["name"] not in args.only) or (not args.only and not src["default"]):
            continue
        print(f"[{src['name']}] {src['repo']} ({src['include']})")
        base = Path(args.from_dir) if args.from_dir else Path(args.cache) / src["name"]
        jsonl = base / src["include"]
        if not args.from_dir:
            res = run_hf("download", src["repo"], "--type", "dataset", "--include", src["include"],
                         "--local-dir", str(base))
            if res.returncode != 0 or not (base / src["include"]).is_file():
                print("  hf download failed; trying a direct HTTPS download ...")
                if not http_download(src["repo"], src["include"], base / src["include"]):
                    print("  skipping (check network, or `hf auth login`)")
                    continue
        if not jsonl.is_file():
            print(f"  expected file not found: {jsonl}; skipping")
            continue
        process(src, jsonl, raw_dir, int(args.mb * 1_000_000), seen)
        if not args.from_dir and not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    print("done. Next: python3 prepare_data.py")


if __name__ == "__main__":
    main()
