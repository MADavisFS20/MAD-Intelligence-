"""Local corpus ingestion: collect, dedupe, manifest, encode, split."""
import gc
import hashlib
import json
import re
from array import array
from datetime import datetime, timezone
from pathlib import Path

import torch

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".kt", ".java", ".xml", ".gradle", ".kts", ".json",
    ".js", ".ts", ".c", ".h", ".cpp", ".sh", ".html", ".css", ".yaml", ".yml",
    ".toml", ".pro", ".rst", ".cfg", ".ini",
}
SKIP_DIRS = {"__pycache__", "node_modules", "build", "venv", "env"}  # plus any dot-directory
CHUNK_CHARS = 1_000_000

# Files matching any of these are left out of the corpus so keys and passwords never get memorized.
SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(r"gh[pousr]_[0-9A-Za-z]{36,}")),
    ("secret-style API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("hard-coded password", re.compile(r"(?i)\b\w*(?:password|passwd|secret)\w*\s*[=:]\s*[\"'][^\"'$\s]{4,}[\"']")),
]


def find_secret(text):
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return label
    return None


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _skipped_dir(relative_parts):
    return any(p in SKIP_DIRS or p.startswith(".") for p in relative_parts[:-1])


def _iter_collected_items(roots, max_file_bytes=5_000_000, scan_secrets=True):
    """Yield (record_dict_or_None, skip_tuple_or_None) lazily so large corpora stream cleanly."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    seen = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            yield None, (str(root), "input path does not exist")
            continue
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix != ".jsonl" and suffix not in TEXT_EXTENSIONS:
                continue
            if path != root and _skipped_dir(path.relative_to(root).parts):
                continue
            try:
                size = path.stat().st_size
                if size == 0:
                    continue
                if suffix == ".jsonl":
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        for n, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            where = f"{path}#{n}"
                            try:
                                rec = json.loads(line)
                            except ValueError:
                                yield None, (where, "invalid JSON line")
                                continue
                            if not isinstance(rec, dict):
                                continue
                            text = None
                            for field in ("output", "response", "code", "text"):
                                val = rec.get(field)
                                if isinstance(val, str) and val.strip():
                                    text = val.replace("\x00", "").strip()
                                    break
                            if not text:
                                continue
                            raw_bytes = len(text.encode("utf-8"))
                            if raw_bytes > max_file_bytes:
                                yield None, (where, f"larger than {max_file_bytes // 1_000_000} MB")
                                continue
                            if scan_secrets:
                                scan_text = "\n".join(
                                    str(rec.get(k) or "") for k in ("instruction", "input", "prompt", "context")
                                ) + "\n" + text
                                secret = find_secret(scan_text)
                                if secret:
                                    yield None, (where, f"possible secret ({secret})")
                                    continue
                            digest = sha256_bytes(text.encode("utf-8"))
                            if digest in seen:
                                yield None, (where, "duplicate content")
                                continue
                            seen.add(digest)
                            yield {
                                "path": str(rec.get("source_path") or where),
                                "sha256": digest,
                                "bytes": raw_bytes,
                                "text": text,
                                "source": str(rec.get("source") or rec.get("repo") or ""),
                                "license": str(rec.get("license") or ""),
                            }, None
                    continue
                if size > max_file_bytes:
                    yield None, (str(path), f"larger than {max_file_bytes // 1_000_000} MB")
                    continue
                raw = path.read_bytes()
            except OSError as e:
                yield None, (str(path), f"unreadable ({e.__class__.__name__})")
                continue
            text = raw.decode("utf-8", errors="replace").replace("\x00", "").strip()
            if not text:
                continue
            if text.count("\ufffd") > 0.01 * len(text):
                yield None, (str(path), "looks binary")
                continue
            if scan_secrets:
                secret = find_secret(text)
                if secret:
                    yield None, (str(path), f"possible secret ({secret})")
                    continue
            digest = sha256_bytes(text.encode("utf-8"))
            if digest in seen:
                yield None, (str(path), "duplicate content")
                continue
            seen.add(digest)
            yield {"path": str(path), "sha256": digest, "bytes": len(raw), "text": text}, None


def collect_files(roots, max_file_bytes=5_000_000, scan_secrets=True):
    """Return (records, skipped). Each record: path, sha256, bytes, text. Duplicates are dropped."""
    records, skipped = [], []
    for rec, skip in _iter_collected_items(roots, max_file_bytes, scan_secrets):
        if rec is not None:
            records.append(rec)
        if skip is not None:
            skipped.append(skip)
    return records, skipped


def load_local_texts(root):
    """Compatibility helper: just the texts."""
    records, _ = collect_files(root)
    return [r["text"] for r in records]


def write_corpus(texts, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\x00", "") + "\n\n")


def stream_prepare_corpus(roots, corpus_path, manifest_path, max_file_bytes=5_000_000, scan_secrets=True):
    """Stream collected texts directly to corpus_path to keep peak RAM low on mobile devices."""
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    meta_records, skipped = [], []
    with corpus_path.open("w", encoding="utf-8") as f:
        for rec, skip in _iter_collected_items(roots, max_file_bytes, scan_secrets):
            if skip is not None:
                skipped.append(skip)
                continue
            text = rec.pop("text")
            f.write(text.replace("\x00", "") + "\n\n")
            meta_records.append(rec)
    if not meta_records:
        if corpus_path.exists():
            corpus_path.unlink()
        return [], skipped, None
    manifest = write_manifest(meta_records, skipped, manifest_path, corpus_path)
    return meta_records, skipped, manifest


def write_manifest(records, skipped, manifest_path, corpus_path):
    """Write the corpus manifest, keeping any source/license notes you already filled in."""
    manifest_path = Path(manifest_path)
    previous = {}
    if manifest_path.is_file():
        try:
            files_data = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
            if isinstance(files_data, list):
                for entry in files_data:
                    previous[entry["sha256"]] = entry
        except (ValueError, KeyError, TypeError):
            pass
    files = []
    for r in records:
        old = previous.get(r["sha256"], {})
        files.append({
            "path": r["path"], "sha256": r["sha256"], "bytes": r["bytes"],
            "source": old.get("source") or r.get("source", ""),
            "license": old.get("license") or r.get("license", ""),
            "acquired": old.get("acquired", ""),
        })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_bytes": Path(corpus_path).stat().st_size,
        "files": files,
        "skipped": [{"path": p, "reason": why} for p, why in skipped],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def iter_file_chunks(path, chunk_chars=CHUNK_CHARS):
    """Stream a text file in ~chunk_chars pieces that always end at a line boundary."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                return
            if not chunk.endswith("\n"):
                chunk += f.readline()
            yield chunk


def encode_corpus(tokenizer, path):
    """Encode a corpus file into a compact integer tensor (int16 when the vocabulary allows)."""
    small = tokenizer.size <= 32767
    buf = array("h" if small else "i")
    for chunk in iter_file_chunks(path):
        buf.extend(tokenizer.encode(chunk))
    dtype = torch.int16 if small else torch.int32
    if not len(buf):
        return torch.zeros(0, dtype=dtype)
    tensor = torch.frombuffer(buf, dtype=dtype).clone()
    del buf
    gc.collect()
    return tensor


def load_or_encode_tokens(tokenizer, tokenizer_path, corpus_path, cache_path):
    """Return (tokens, was_cached). The cache is keyed by tokenizer + corpus content hashes."""
    key = sha256_file(tokenizer_path) + ":" + sha256_file(corpus_path)
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached.get("key") == key:
                return cached["tokens"], True
        except Exception:
            pass
    tokens = encode_corpus(tokenizer, corpus_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "tokens": tokens}, cache_path)
    return tokens, False


def split_tokens(tokens, val_fraction=0.05, min_val_tokens=256):
    """Contiguous tail split. Returns (train, val); val is None when the corpus is too small."""
    n_val = int(tokens.numel() * val_fraction)
    if val_fraction <= 0 or n_val < min_val_tokens:
        return tokens, None
    return tokens[:-n_val], tokens[-n_val:]
