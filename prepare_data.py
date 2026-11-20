#!/usr/bin/env python3
"""prepare-data for MAD Intelligence (stdlib only, Termux friendly).

Reads data/raw/ (source files AND .jsonl records with instruction/input/output),
filters and de-duplicates, and writes to data/processed/:
  train.txt, val.txt              plain-text corpus (code, with a path header per item)
  sft_train.jsonl, sft_val.jsonl  instruction/input/output records (from .jsonl only)
  dropped.txt                     every dropped item and why
  manifest.json                   source/license/notes per item (preserved on re-run)

Standalone:  python prepare_data.py [--raw data/raw] [--out data/processed]
Inside mad.py:  from prepare_data import add_arguments, run
"""
import argparse
import hashlib
import json
import os
import re
import time
import sys
from pathlib import Path

EXTENSIONS = {
    ".kt", ".java", ".xml", ".gradle", ".kts", ".json", ".py", ".md", ".txt",
    ".properties", ".pro", ".aidl", ".toml", ".yml", ".yaml", ".sh", ".cfg",
    # general Linux/open-source languages (hand-added files; fetched .jsonl records are accepted for any type)
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".go", ".rs", ".rb", ".pl", ".pm", ".lua", ".js", ".mjs",
    ".ts", ".php", ".scala", ".swift", ".cs", ".zig", ".hs", ".ml", ".ex", ".exs", ".mk", ".cmake", ".bash",
}
SKIP_DIRS = {
    "build", ".git", ".gradle", ".idea", ".cxx", ".externalNativeBuild",
    "node_modules", "__pycache__", ".venv", "venv", "env", "out", "dist",
    "captures", ".svn", ".hg",
}
SKIP_FILENAMES = {
    "google-services.json", "local.properties", ".env", "keystore.properties",
    "gradle-wrapper.jar", "package-lock.json", "yarn.lock",
}
SKIP_SUFFIXES = (".jks", ".keystore", ".pem", ".p12", ".pfx", ".min.js")

FILE_START = "<|file|> "
FILE_END = "\n<|endoftext|>\n"
MANIFEST_USER_FIELDS = ("source", "license", "notes")

SECRET_PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("Stripe/OpenAI-style key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b|\bsk-[A-Za-z0-9]{32,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("signing password", re.compile(r"\b(?:storePassword|keyPassword)\s*[=:]\s*[\"']?[^\s\"'$][^\s\"']*", re.I)),
    ("hard-coded credential", re.compile(
        r"""(?ix)\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd)\b
        \s*[=:]\s*["']([^"'\s]{8,})["']""")),
]
PLACEHOLDER = re.compile(r"(?i)(your[_-]|example|changeme|placeholder|xxxx|<.*>|\$\{|\{\{|dummy|sample|test)")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n") + "\n"


_LICENSE_WORDS = re.compile(r"(?i)copyright|licen[sc]e|all rights reserved|spdx-license")
_LINE_COMMENT = re.compile(r"^\s*(?://+|#(?=\s|$))")


def strip_license_header(text):
    """Remove a leading license/copyright comment (/* ... */ block or a run of // or # lines).

    Every Android file starts with the same ~15-line Apache header. Left in, it is the most
    "learnable" text in the corpus and eats a large share of a small model's context window."""
    m = re.match(r"\s*/\*.*?\*/[ \t]*\n?", text, re.S)
    if m and m.end() <= 4000 and _LICENSE_WORDS.search(m.group(0)):
        return text[m.end():].lstrip("\n")
    m = re.match(r"(<\?xml[^>]*\?>\s*)(<!--.*?-->)\s*", text, re.S)
    if m and m.end(2) <= 4000 and _LICENSE_WORDS.search(m.group(2)):
        return m.group(1).rstrip() + "\n" + text[m.end():]
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (_LINE_COMMENT.match(lines[i]) or (not lines[i].strip() and i)):
        i += 1
    head = "\n".join(lines[:i])
    if i and len(head) <= 4000 and _LICENSE_WORDS.search(head):
        return "\n".join(lines[i:]).lstrip("\n")
    return text


def split_group(key, source):
    """Items from one repo must fall on the same side of the train/val split, otherwise near-identical
    code from the same project leaks into validation and the held-out loss looks better than it is."""
    if source.startswith("github:"):
        return source.split(" path=")[0].split("@")[0]
    if source.startswith("hf:"):
        m = re.search(r"repos=(\S+)", source)
        return "hf:" + m.group(1) if m else key
    base = key.split("#")[0]
    return base.split("/")[0] if "#" not in key and "/" in base else key


def looks_binary(raw):
    if b"\x00" in raw[:8192]:
        return True
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return True
    sample = raw[:8192]
    if not sample:
        return False
    ctrl = sum(1 for b in sample if b < 9 or 13 < b < 32)
    return ctrl / len(sample) > 0.05


def find_secret(text):
    for label, pat in SECRET_PATTERNS:
        for m in pat.finditer(text):
            if PLACEHOLDER.search(m.group(0)):
                continue
            return label
    return None


def walk(raw_dir):
    for root, dirs, files in os.walk(raw_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            yield Path(root) / name


def load_manifest(path):
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("files"), dict):
                return data
        except json.JSONDecodeError:
            backup = path.with_suffix(".json.bak")
            path.replace(backup)
            print(f"warning: manifest was invalid JSON, saved as {backup}", file=sys.stderr)
    return {"files": {}}


def is_val(key, val_frac):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_frac


def iter_units(raw_dir, args, drop):
    """Yield dicts: key, label, text, scan, record(None for plain files), source, license."""
    for path in walk(raw_dir):
        rel = path.relative_to(raw_dir).as_posix()
        name = path.name
        suffix = path.suffix.lower()

        if name in SKIP_FILENAMES or name.lower().endswith(SKIP_SUFFIXES):
            drop(rel, "excluded filename/type (credentials or generated)")
            continue
        if suffix != ".jsonl" and suffix not in EXTENSIONS:
            drop(rel, "unsupported extension")
            continue
        size = path.stat().st_size
        if size == 0:
            drop(rel, "empty")
            continue

        if suffix == ".jsonl":
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for n, line in enumerate(fh, 1):
                        key = f"{rel}#{n}"
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            drop(key, "invalid JSON line")
                            continue
                        out = rec.get("output") if isinstance(rec, dict) else None
                        if not isinstance(out, str) or not out.strip():
                            drop(key, "record has no 'output' text")
                            continue
                        if len(out.encode("utf-8")) > args.max_bytes:
                            drop(key, f"output larger than {args.max_bytes} bytes")
                            continue
                        text = strip_license_header(normalize(out)) if args.strip_license else normalize(out)
                        inp = str(rec.get("input") or "")
                        if args.strip_license and inp:
                            inp = strip_license_header(normalize(inp))
                        scan = "\n".join((str(rec.get("instruction") or ""), inp)) + "\n" + text
                        yield {
                            "key": key,
                            "label": str(rec.get("source_path") or key),
                            "text": text, "scan": scan,
                            "record": {"instruction": str(rec.get("instruction") or ""),
                                       "input": inp,
                                       "output": text},
                            "source": str(rec.get("source") or rec.get("repo") or ""),
                            "license": str(rec.get("license") or ""),
                        }
            except UnicodeDecodeError:
                drop(rel, "not valid UTF-8")
            continue

        if size > args.max_bytes:
            drop(rel, f"larger than {args.max_bytes} bytes")
            continue
        data = path.read_bytes()
        if looks_binary(data):
            drop(rel, "binary-looking")
            continue
        text = normalize(data.decode("utf-8"))
        scan = text
        if args.strip_license:
            text = strip_license_header(text)
        if not text.strip():
            drop(rel, "empty after removing license header")
            continue
        yield {"key": rel, "label": rel, "text": text, "scan": scan,
               "record": None, "source": "", "license": ""}


def run(args):
    raw_dir, out_dir = Path(args.raw), Path(args.out)
    if not raw_dir.is_dir():
        sys.exit(f"error: {raw_dir} does not exist. Put your own / licensed code there.")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    old = manifest.get("files", {})

    dropped = []
    drop = lambda key, why: dropped.append((key, why))
    seen, new_files = {}, {}
    kept_count = train_count = val_count = 0
    sft_train_count = sft_val_count = 0
    tb = vb = 0

    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    sft_train_path = out_dir / "sft_train.jsonl"
    sft_val_path = out_dir / "sft_val.jsonl"

    with (train_path.open("w", encoding="utf-8") as f_train,
          val_path.open("w", encoding="utf-8") as f_val,
          sft_train_path.open("w", encoding="utf-8") as f_sft_train,
          sft_val_path.open("w", encoding="utf-8") as f_sft_val):
        t_start, scanned, scanned_bytes = time.time(), 0, 0
        for u in iter_units(raw_dir, args, drop):
            scanned += 1
            scanned_bytes += len(u["text"])
            if scanned % 1000 == 0:
                print(f"  ... scanned {scanned} items ({scanned_bytes / 1e6:.1f} MB), "
                      f"{time.time() - t_start:.0f}s elapsed", flush=True)
            key = u["key"]
            if not args.no_secret_scan:
                secret = find_secret(u["scan"])
                if secret:
                    drop(key, f"possible secret: {secret}")
                    continue
            digest = sha(u["text"])
            if digest in seen:
                drop(key, f"duplicate of {seen[digest]}")
                continue
            seen[digest] = key

            prev = old.get(key, {})
            entry = {
                "source": prev.get("source") or u["source"] or "UNKNOWN",
                "license": prev.get("license") or u["license"] or "UNKNOWN",
                "notes": prev.get("notes", ""),
                "sha256": digest,
                "bytes": len(u["text"].encode("utf-8")),
            }
            if u["label"] != key:
                entry["path"] = u["label"]
            new_files[key] = entry

            block = FILE_START + u["label"] + "\n" + u["text"] + FILE_END
            block_bytes = len(block.encode("utf-8"))
            val = is_val(split_group(key, entry["source"] if entry["source"] != "UNKNOWN" else u["source"]), args.val_frac)
            if val:
                f_val.write(block)
                val_count += 1
                vb += block_bytes
                if u["record"] and u["record"]["instruction"].strip():
                    f_sft_val.write(json.dumps(u["record"], ensure_ascii=False) + "\n")
                    sft_val_count += 1
            else:
                f_train.write(block)
                train_count += 1
                tb += block_bytes
                if u["record"] and u["record"]["instruction"].strip():
                    f_sft_train.write(json.dumps(u["record"], ensure_ascii=False) + "\n")
                    sft_train_count += 1
            kept_count += 1

    if sft_train_count == 0 and sft_train_path.exists():
        sft_train_path.unlink()
    if sft_val_count == 0 and sft_val_path.exists():
        sft_val_path.unlink()

    # keep manifest notes for items that were not in this run
    for key, entry in old.items():
        if key not in new_files:
            entry = dict(entry)
            entry["status"] = "not in last run"
            new_files[key] = entry

    (out_dir / "dropped.txt").write_text(
        "\n".join(f"{k}\t{why}" for k, why in dropped) + ("\n" if dropped else ""), encoding="utf-8")
    manifest["files"] = new_files
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"kept {kept_count} items | train {tb/1024:.1f} KB ({train_count}) | val {vb/1024:.1f} KB ({val_count})")
    print(f"sft records: train {sft_train_count}, val {sft_val_count}")
    print(f"dropped {len(dropped)} items (full list: {out_dir / 'dropped.txt'})")
    for key, why in dropped:
        if why.startswith("possible secret"):
            print(f"  REVIEW  {key}  -> {why}")
    if args.no_secret_scan:
        print("WARNING: secret scan disabled. Make sure you checked every file.")
    unknown = [k for k, e in new_files.items()
               if "status" not in e and "UNKNOWN" in (e["source"], e["license"])]
    if unknown:
        print(f"{len(unknown)} kept items have source/license = UNKNOWN. Fill them in: {manifest_path}")
    if tb + vb and not 0.02 <= vb / (tb + vb) <= 0.10:
        print(f"note: validation is {100 * vb / (tb + vb):.1f}% of the text (target {100 * args.val_frac:.0f}%). "
              f"Splitting by repo makes this vary when there are few repos.")
    if tb < 200 * 1024:
        print("note: under ~200 KB of training text. Enough for formatting only; add more varied code.")


def add_arguments(p):
    p.add_argument("--raw", default="data/raw", help="input folder (default data/raw)")
    p.add_argument("--out", default="data/processed", help="output folder (default data/processed)")
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction of items held out for eval")
    p.add_argument("--max-bytes", type=int, default=1_000_000, help="skip items larger than this")
    p.add_argument("--keep-license-headers", dest="strip_license", action="store_false",
                   help="keep the leading license/copyright comment of each file (default: remove it)")
    p.add_argument("--no-secret-scan", action="store_true",
                   help="disable the secret scan (only after reviewing every file)")
    return p


if __name__ == "__main__":
    run(add_arguments(argparse.ArgumentParser(description=__doc__.splitlines()[0])).parse_args())
