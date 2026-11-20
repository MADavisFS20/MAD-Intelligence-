#!/usr/bin/env python3
"""Small helpers used by start.sh / mi_run.py. Each prints one JSON line (except `notices`).

  count-tokens   encode the corpus (cached) and report the token count
  ckpt-step P    read a checkpoint and report its step (fails if unreadable or non-finite)
  pairs-from-raw turn documented functions/classes in data/raw/*.jsonl into request->code pairs
  val-trend      look at checkpoints/training.jsonl and say whether validation loss stopped improving
  data-report    show what the model trains on: file types, sources, what was left out and why
  notices        write data/THIRD_PARTY_NOTICES.md from data/processed/manifest.json
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def out(obj):
    print(json.dumps(obj))


def cmd_count_tokens(a):
    from mad import data
    from mad.tokenizer import MADTokenizer
    tok = MADTokenizer.load(a.tokenizer)
    tokens, cached = data.load_or_encode_tokens(tok, a.tokenizer, Path(a.corpus), Path(a.corpus).with_name("tokens.pt"))
    out({"tokens": int(tokens.numel()), "cached": cached})


def cmd_ckpt_step(a):
    import torch
    from mad.trainer import load_checkpoint
    state = load_checkpoint(a.path)
    for name, tensor in state["model"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise SystemExit(f"non-finite weights in {name}")
    out({"step": int(state["step"]), "has_optimizer": "optimizer" in state,
         "best_val_loss": state.get("metadata", {}).get("best_val_loss")})


def cmd_pairs_from_raw(a):
    from mad import instructions
    pairs, files = [], 0
    for path in sorted(Path(a.raw).glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lang = instructions.LANGUAGES.get("." + str(rec.get("extension", "")).lower())
                if not lang or not isinstance(rec.get("output"), str):
                    continue
                files += 1
                for p in instructions.extract_pairs(rec["output"], lang):
                    pairs.append(p)
                    if len(pairs) >= a.max_pairs:
                        break
                if len(pairs) >= a.max_pairs:
                    break
    pairs, skipped = instructions.clean_pairs(pairs, scan_secrets=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    instructions.write_pairs(pairs, a.out)
    out({"files_scanned": files, "pairs": len(pairs), "skipped": len(skipped)})


def cmd_val_trend(a):
    log = Path(a.log)
    vals = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" in r:
                vals.append((r["step"], r["val_loss"]))
    if not vals:
        return out({"evals": 0, "stop": False})
    best_i = min(range(len(vals)), key=lambda i: vals[i][1])
    since = len(vals) - 1 - best_i
    stop = since >= a.patience and vals[-1][1] > vals[best_i][1] * 1.02
    out({"evals": len(vals), "best_step": vals[best_i][0], "best_val": vals[best_i][1],
         "last_step": vals[-1][0], "last_val": vals[-1][1], "evals_since_best": since, "stop": bool(stop)})


def cmd_notices(a):
    manifest = Path(a.manifest)
    groups = OrderedDict()
    if manifest.is_file():
        for entry in json.loads(manifest.read_text()).get("files", {}).values():
            src = (entry.get("source") or "UNKNOWN").split(" path=")[0]
            if src.startswith("hf:"):
                parts = src.split()
                src = parts[0] + (" " + next((p for p in parts if p.startswith("repo")), "") if len(parts) > 1 else "")
            elif src.startswith("github:"):
                src = src.split("@")[0]
            key = (src.strip(), entry.get("license") or "UNKNOWN")
            groups[key] = groups.get(key, 0) + 1
    lines = ["# Third-party notices", "",
             "Training data for this model came from the public sources below. Each is used under its own",
             "license (see data/licenses/ for the license texts saved at download time).", "",
             "| Source | License | Files |", "|---|---|---:|"]
    for (src, lic), n in sorted(groups.items()):
        link = "https://github.com/" + src[len("github:"):] if src.startswith("github:") else src
        lines.append(f"| {link} | {lic} | {n} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(groups)} sources)")


def _source_group(src):
    src = src or "UNKNOWN"
    if src.startswith("github:"):
        return "github"
    if src.startswith("hf:"):
        return "hf:" + src[3:].split()[0].split(":")[0]
    return "local files" if src in ("UNKNOWN", "") else src.split()[0]


def cmd_data_report(a):
    """Show exactly what the model trains on: file types, sources, what was left out and why."""
    manifest = Path(a.manifest)
    if not manifest.is_file():
        raise SystemExit("No manifest yet: run python3 prepare_data.py first.")
    files = json.loads(manifest.read_text()).get("files", {})
    by_ext, by_src = {}, {}
    for key, e in files.items():
        name = (e.get("path") or key).split("#")[0]
        ext = ("." + name.rsplit(".", 1)[1].lower()) if "." in name.rsplit("/", 1)[-1] else "(none)"
        for table, k in ((by_ext, ext), (by_src, _source_group(e.get("source")))):
            n, b = table.get(k, (0, 0))
            table[k] = (n + 1, b + e.get("bytes", 0))
    total = sum(b for _, b in by_ext.values()) or 1
    print("== What the model trains on ==")
    print(f"{len(files)} items, {total / 1e6:.2f} MB of text after cleaning")
    print("\nBy file type:")
    for k, (n, b) in sorted(by_ext.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:10} {n:7} files {b / 1e6:8.2f} MB {100 * b / total:5.1f}%")
    print("\nBy source:")
    for k, (n, b) in sorted(by_src.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:34} {n:7} files {b / 1e6:8.2f} MB")
    dropped = Path(a.manifest).with_name("dropped.txt")
    if dropped.is_file():
        reasons, exts = {}, {}
        for line in dropped.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            key, why = line.split("\t", 1)
            why = why.split(":")[0] if why.startswith("possible secret") else why
            reasons[why] = reasons.get(why, 0) + 1
            if why == "unsupported extension":
                e = ("." + key.rsplit(".", 1)[1].lower()) if "." in key.rsplit("/", 1)[-1] else "(none)"
                exts[e] = exts.get(e, 0) + 1
        if reasons:
            print("\nLeft out (details: data/processed/dropped.txt):")
            for why, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {n:7}  {why}")
        if exts:
            print("  unsupported file types found in data/raw/: " + ", ".join(f"{e} x{n}" for e, n in sorted(exts.items(), key=lambda kv: -kv[1])[:10]))
    corpus = Path(a.manifest).with_name("corpus.txt")
    if a.tokens and corpus.is_file() and (ROOT / "tokenizer/mad_bpe.json").is_file():
        try:
            from mad import data
            from mad.tokenizer import MADTokenizer
            tok = MADTokenizer.load(str(ROOT / "tokenizer/mad_bpe.json"))
            tokens, _ = data.load_or_encode_tokens(tok, str(ROOT / "tokenizer/mad_bpe.json"), corpus, corpus.with_name("tokens.pt"))
            print(f"\nTraining corpus: {corpus.stat().st_size / 1e6:.2f} MB -> {int(tokens.numel()):,} tokens (95% train / 5% validation)")
        except Exception as e:  # report is informational only
            print(f"\n(token count unavailable: {e})")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("count-tokens")
    s.add_argument("--tokenizer", default=str(ROOT / "tokenizer/mad_bpe.json"))
    s.add_argument("--corpus", default=str(ROOT / "data/processed/corpus.txt"))
    s.set_defaults(fn=cmd_count_tokens)
    s = sub.add_parser("ckpt-step")
    s.add_argument("path")
    s.set_defaults(fn=cmd_ckpt_step)
    s = sub.add_parser("pairs-from-raw")
    s.add_argument("--raw", default=str(ROOT / "data/raw"))
    s.add_argument("--out", default=str(ROOT / "data/instructions/auto_from_raw.jsonl"))
    s.add_argument("--max-pairs", type=int, default=20000)
    s.set_defaults(fn=cmd_pairs_from_raw)
    s = sub.add_parser("val-trend")
    s.add_argument("--log", default=str(ROOT / "checkpoints/training.jsonl"))
    s.add_argument("--patience", type=int, default=12, help="evaluations without a new best before stopping")
    s.set_defaults(fn=cmd_val_trend)
    s = sub.add_parser("data-report")
    s.add_argument("--manifest", default=str(ROOT / "data/processed/manifest.json"))
    s.add_argument("--no-tokens", dest="tokens", action="store_false")
    s.set_defaults(fn=cmd_data_report)
    s = sub.add_parser("notices")
    s.add_argument("--manifest", default=str(ROOT / "data/processed/manifest.json"))
    s.add_argument("--out", default=str(ROOT / "data/THIRD_PARTY_NOTICES.md"))
    s.set_defaults(fn=cmd_notices)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
