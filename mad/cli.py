import argparse
import gc
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

from . import __version__
from . import data
from . import instructions
from .config import PRESETS, ModelConfig
from .model import MADIntelligence
from .tokenizer import MADTokenizer
from .trainer import (configure_threads, default_threads, evaluate_loss,
                      load_checkpoint, save_checkpoint, train_model)

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
VAL_FRACTION = 0.05


def rp(relative):
    """Resolve a default path against the project root so commands work from any directory."""
    return str(ROOT / relative)


def die(message):
    raise SystemExit(message)


def load_config(path):
    try:
        return ModelConfig.load(path)
    except (FileNotFoundError, ValueError) as e:
        die(str(e))


def read_tokenizer(path):
    if not Path(path).is_file():
        die(f"Tokenizer not found: {path}. Run `python mad.py train-tokenizer` first.")
    try:
        return MADTokenizer.load(path)
    except ValueError as e:
        die(str(e))


def sync_vocab(config, tokenizer, config_path):
    """The tokenizer may learn fewer merges than requested; make config.json match it."""
    if config.vocab_size != tokenizer.size:
        print(f"Note: setting vocab_size {config.vocab_size} -> {tokenizer.size} in {config_path} "
              f"to match the tokenizer.")
        config.vocab_size = tokenizer.size
        config.save(config_path)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def memory_info():
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) * 1024
        return info.get("MemTotal"), info.get("MemAvailable")
    except (OSError, ValueError):
        return None, None


PRETRAIN_CHECKPOINTS = ("mad_best.pt", "mad_latest.pt")
SFT_CHECKPOINTS = ("mad_sft_best.pt", "mad_sft_latest.pt") + PRETRAIN_CHECKPOINTS


def resolve_checkpoint(path, names=PRETRAIN_CHECKPOINTS):
    if path:
        return Path(path)
    for name in names:
        if (CKPT_DIR / name).is_file():
            return CKPT_DIR / name
    die("No checkpoint found. Run `python mad.py train` first.")


def load_inference_model(checkpoint, tokenizer_path, threads, names=PRETRAIN_CHECKPOINTS):
    ckpt = resolve_checkpoint(checkpoint, names)
    try:
        state = load_checkpoint(ckpt)
    except (FileNotFoundError, RuntimeError) as e:
        die(str(e))
    config = ModelConfig(**state["config"])
    tokenizer = read_tokenizer(tokenizer_path)
    if tokenizer.size != config.vocab_size:
        die(f"Tokenizer size {tokenizer.size} does not match checkpoint vocab_size {config.vocab_size}. "
            f"If you retrained the tokenizer, the model must be retrained too.")
    expected = state.get("metadata", {}).get("tokenizer_sha256")
    if expected and expected != data.sha256_file(tokenizer_path):
        print("Warning: this tokenizer differs from the one used for training; output may be garbled.")
    configure_threads(threads)
    model = MADIntelligence(config)
    model.load_state_dict(state["model"])
    del state
    gc.collect()
    return model, tokenizer, config, ckpt


def sft_texts(path):
    """Instruction examples as text chunks (so the tokenizer also learns English and the control tokens)."""
    pairs, _ = instructions.load_pairs(path)
    for i in range(0, len(pairs), 200):
        yield "\n".join(instructions.format_example(p) for p in pairs[i:i + 200]) + "\n"


# ----------------------------------------------------------------------------- commands
def command_init(args):
    config_path = Path(args.config)
    if config_path.exists() and not args.force:
        die(f"{config_path} already exists. Use --force to overwrite it (or edit it by hand).")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ModelConfig.from_preset(args.preset, vocab_size=args.vocab_size)
    config.save(config_path)
    model = MADIntelligence(config)
    save_checkpoint(Path(args.checkpoint_dir) / "random_init.pt", model, None, 0, config,
                    {"origin": "random_initialization", "seed": args.seed, "preset": args.preset},
                    save_optimizer=False)
    n = model.num_parameters()
    print(f"Initialized '{args.preset}' model: {n:,} parameters from random initialization.")
    print(f"Weights+gradients+optimizer need about {human_bytes(n * 16)} of RAM (activations extra).")
    print(f"Saved {config_path} and {Path(args.checkpoint_dir) / 'random_init.pt'}")


def command_prepare(args):
    records, skipped, manifest = data.stream_prepare_corpus(
        args.input, args.output, Path(args.output).with_name("manifest.json"),
        int(args.max_file_mb * 1_000_000), scan_secrets=not args.no_secret_scan)
    if not records:
        die("No supported text files found. Put files in data/raw/ or pass --input DIR [DIR ...].")
    total = sum(r["bytes"] for r in records)
    print(f"Prepared {len(records)} files ({human_bytes(total)}) into {args.output}")
    reasons = {}
    for _, why in skipped:
        reasons[why] = reasons.get(why, 0) + 1
    for why, count in sorted(reasons.items()):
        print(f"  skipped {count}: {why}")
    flagged = [path for path, why in skipped if why.startswith("possible secret")]
    for path in flagged[:10]:
        print(f"    review: {path}")
    if len(flagged) > 10:
        print(f"    ... and {len(flagged) - 10} more (see manifest.json)")
    print(f"Manifest: {Path(args.output).with_name('manifest.json')} "
          f"(fill in each file's source/license before serious training)")
    if total < 200_000:
        print("Warning: under 200 KB of text. The model will memorize rather than learn; add more code.")
    return manifest


def command_tokenizer(args):
    corpus = Path(args.corpus)
    if not corpus.is_file():
        die(f"Corpus not found: {corpus}. Run `python mad.py prepare-data` first.")
    start = time.time()
    chunks = data.iter_file_chunks(corpus)
    sft = Path(args.sft)
    if sft.is_file():
        import itertools
        print(f"Including instruction examples from {sft}.")
        chunks = itertools.chain(chunks, sft_texts(sft))
    tokenizer = MADTokenizer().train(chunks, args.vocab_size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print(f"Tokenizer trained in {time.time() - start:.1f}s. Vocabulary size: {tokenizer.size} "
          f"(requested {args.vocab_size}).")
    if tokenizer.size < args.vocab_size:
        print("The corpus did not have enough repeated byte pairs for the full vocabulary; "
              "the model will use the smaller size automatically.")
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        sync_vocab(load_config(cfg_path), tokenizer, cfg_path)
    print(f"Saved: {args.output}")


def command_train(args):
    config = load_config(args.config)
    tokenizer = read_tokenizer(args.tokenizer)
    if tokenizer.size != config.vocab_size:
        if args.resume:
            die(f"Tokenizer size {tokenizer.size} does not match model vocab_size {config.vocab_size}; "
                f"cannot resume.")
        sync_vocab(config, tokenizer, args.config)
    corpus = Path(args.corpus)
    if not corpus.is_file():
        die(f"Corpus not found: {corpus}. Run `python mad.py prepare-data` first.")
    start = time.time()
    tokens, cached = data.load_or_encode_tokens(tokenizer, args.tokenizer, corpus,
                                                Path(corpus).with_name("tokens.pt"))
    print(f"{tokens.numel():,} tokens ({'cached' if cached else f'encoded in {time.time() - start:.1f}s'}).")
    seq_len = args.seq_len or config.max_seq_len
    train_tokens, val_tokens = data.split_tokens(tokens, VAL_FRACTION)
    del tokens, tokenizer
    gc.collect()
    if val_tokens is not None and val_tokens.numel() < seq_len + 2:
        val_tokens = None
    if val_tokens is None:
        print("Corpus too small for a validation split; training without validation loss.")
    if train_tokens.numel() < seq_len + 2:
        die(f"Corpus has {train_tokens.numel()} training tokens; need at least {seq_len + 2}. Add more local text.")
    warmup = args.warmup if args.warmup is not None else max(1, min(100, args.steps // 10))
    metadata = {"origin": "random_initialization_then_local_training", "seed": args.seed,
                "tokenizer_sha256": data.sha256_file(args.tokenizer),
                "corpus_sha256": data.sha256_file(corpus), "version": __version__}
    try:
        train_model(train_tokens, val_tokens, config, args.output_dir, steps=args.steps,
                    batch_size=args.batch_size, seq_len=seq_len, grad_accum=args.grad_accum,
                    learning_rate=args.learning_rate, warmup=warmup, eval_every=args.eval_every,
                    eval_batches=args.eval_batches, checkpoint_every=args.checkpoint_every,
                    resume=args.resume, threads=args.threads, max_minutes=args.max_minutes,
                    seed=args.seed, metadata=metadata)
    except (FloatingPointError, ValueError, FileNotFoundError, RuntimeError) as e:
        die(f"Training stopped: {e}")


def command_eval(args):
    ckpt = resolve_checkpoint(args.checkpoint)
    state = load_checkpoint(ckpt)
    config = ModelConfig(**state["config"])
    tokenizer = read_tokenizer(args.tokenizer)
    if tokenizer.size != config.vocab_size:
        die(f"Tokenizer size {tokenizer.size} does not match checkpoint vocab_size {config.vocab_size}.")
    corpus = Path(args.corpus)
    if not corpus.is_file():
        die(f"Corpus not found: {corpus}. Run `python mad.py prepare-data` first.")
    tokens, _ = data.load_or_encode_tokens(tokenizer, args.tokenizer, corpus, corpus.with_name("tokens.pt"))
    train_tokens, val_tokens = data.split_tokens(tokens, VAL_FRACTION)
    del tokens, tokenizer
    seq_len = args.seq_len or config.max_seq_len
    if val_tokens is None or val_tokens.numel() < seq_len + 2:
        die("Corpus too small for a held-out validation split.")
    configure_threads(args.threads)
    model = MADIntelligence(config)
    model.load_state_dict(state["model"])
    step = state["step"]
    del state
    gc.collect()
    val = evaluate_loss(model, val_tokens, args.batch_size, seq_len, args.batches)
    train = evaluate_loss(model, train_tokens, args.batch_size, seq_len, args.batches)
    print(f"checkpoint: {ckpt} (step {step})")
    print(f"held-out loss {val:.4f}  perplexity {math.exp(min(val, 50)):.1f}")
    print(f"train    loss {train:.4f}  perplexity {math.exp(min(train, 50)):.1f}")
    if val - train > 0.5:
        print("Large train/held-out gap: the model is memorizing. Add more varied data or train less.")


def command_generate(args):
    model, tokenizer, config, ckpt = load_inference_model(args.checkpoint, args.tokenizer, args.threads)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ids = tokenizer.encode(args.prompt) or [tokenizer.special_to_id["<MAD_START>"]]
    out = model.generate(torch.tensor([ids], dtype=torch.long), args.max_new_tokens, args.temperature,
                         args.top_k, args.repetition_penalty,
                         None if args.no_stop else tokenizer.special_to_id["<MAD_END>"])[0].tolist()
    print(tokenizer.decode(out[len(ids):]))


def command_make_pairs(args):
    pairs, scanned = instructions.extract_from_files(args.input)
    if not pairs:
        die(f"No documented Kotlin/Java declarations found in {scanned} files. Pairs come from a /** doc */ "
            f"or // comment of 3+ words directly above a fun/class/method. Add such comments, or write "
            f"request/code pairs by hand (see README).")
    pairs, skipped = instructions.clean_pairs(pairs, scan_secrets=not args.no_secret_scan)
    instructions.write_pairs(pairs, args.output)
    print(f"Scanned {scanned} files, wrote {len(pairs)} request/code pairs to {args.output}")
    reasons = {}
    for _, why in skipped:
        reasons[why] = reasons.get(why, 0) + 1
    for why, count in sorted(reasons.items()):
        print(f"  skipped {count}: {why}")
    for p in pairs[:3]:
        print(f"  e.g. {p['instruction'][:90]!r} -> {p['response'].splitlines()[0][:60]!r}")
    print("Review the file: the request text is your comment, so poor comments make poor training pairs.")


def command_prepare_instructions(args):
    if not Path(args.input).exists():
        die(f"{args.input} not found. Create it and add .jsonl files, or run `python mad.py make-pairs`.")
    pairs, problems = instructions.load_pairs(args.input)
    for where, why in problems[:10]:
        print(f"  ignored {where}: {why}")
    if len(problems) > 10:
        print(f"  ... and {len(problems) - 10} more ignored lines")
    if not pairs:
        die("No valid request/code pairs found.")
    pairs, skipped = instructions.clean_pairs(pairs, scan_secrets=not args.no_secret_scan)
    instructions.write_pairs(pairs, args.output)
    reasons = {}
    for _, why in skipped:
        reasons[why] = reasons.get(why, 0) + 1
    print(f"Wrote {len(pairs)} examples to {args.output}")
    for why, count in sorted(reasons.items()):
        print(f"  skipped {count}: {why}")
    if len(pairs) < 100:
        print("Warning: under 100 examples. The model will memorize them rather than generalize; aim for 500+.")
    print("If you added instructions after training the tokenizer, retrain tokenizer and model so it learns your wording.")


def command_finetune(args):
    config = load_config(args.config)
    tokenizer = read_tokenizer(args.tokenizer)
    if tokenizer.size != config.vocab_size:
        die(f"Tokenizer size {tokenizer.size} does not match model vocab_size {config.vocab_size}. "
            f"Run `python mad.py train` after retraining the tokenizer.")
    sft = Path(args.sft)
    if not sft.is_file():
        die(f"{sft} not found. Run `python mad.py prepare-instructions` first.")
    pairs, _ = instructions.load_pairs(sft)
    seq_len = args.seq_len or config.max_seq_len
    train_pairs, val_pairs = instructions.split_pairs(pairs)
    train_ex, d1, t1 = instructions.encode_examples(train_pairs, tokenizer, seq_len)
    val_ex, d2, t2 = instructions.encode_examples(val_pairs, tokenizer, seq_len)
    if len(train_ex) < 8:
        die(f"Only {len(train_ex)} usable training examples (need at least 8).")
    print(f"{len(train_ex)} training and {len(val_ex)} held-out examples"
          f" ({d1 + d2} dropped: request longer than the context; {t1 + t2} truncated).")
    base = None
    if not args.resume and not args.from_scratch:
        out_dir = Path(args.output_dir)
        base = Path(args.base) if args.base else next(
            (out_dir / n for n in PRETRAIN_CHECKPOINTS if (out_dir / n).is_file()), None)
        if base is None:
            die("No pretrained checkpoint to start from. Run `python mad.py train` on your code first, "
                "or pass --from-scratch.")
    pad = tokenizer.special_to_id["<MAD_PAD>"]
    metadata = {"stage": "instruction_tuning", "seed": args.seed, "version": __version__,
                "tokenizer_sha256": data.sha256_file(args.tokenizer), "sft_examples": len(pairs),
                "base_checkpoint": str(base) if base else None}
    warmup = args.warmup if args.warmup is not None else max(1, min(50, args.steps // 10))
    del pairs, train_pairs, val_pairs, tokenizer
    gc.collect()
    try:
        train_model(None, None, config, args.output_dir, steps=args.steps, batch_size=args.batch_size,
                    seq_len=seq_len, grad_accum=args.grad_accum, learning_rate=args.learning_rate,
                    warmup=warmup, eval_every=args.eval_every, eval_batches=args.eval_batches,
                    checkpoint_every=args.checkpoint_every, resume=args.resume, threads=args.threads,
                    max_minutes=args.max_minutes, seed=args.seed, metadata=metadata,
                    train_batch_fn=instructions.make_batch_fn(train_ex, args.batch_size, pad),
                    val_batch_fn=instructions.make_batch_fn(val_ex, args.batch_size, pad) if val_ex else None,
                    init_weights=base, prefix="mad_sft")
    except (FloatingPointError, ValueError, FileNotFoundError, RuntimeError) as e:
        die(f"Fine-tuning stopped: {e}")


def command_ask(args):
    model, tokenizer, config, ckpt = load_inference_model(
        args.checkpoint, args.tokenizer, args.threads, SFT_CHECKPOINTS)
    if not ckpt.name.startswith("mad_sft"):
        print(f"Note: {ckpt.name} has not been instruction-tuned, so it will continue text rather than answer "
              f"requests. Run `python mad.py prepare-instructions` and `python mad.py finetune`.\n")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    context = Path(args.context_file).read_text(encoding="utf-8") if args.context_file else None
    end_id = tokenizer.special_to_id["<MAD_END>"]

    def answer(request):
        ids = tokenizer.encode(instructions.build_prompt(request, context))
        if len(ids) >= config.max_seq_len - 8:
            print(f"Request{' + context' if context else ''} is {len(ids)} tokens; the model context is "
                  f"{config.max_seq_len}. Shorten it.")
            return
        out = model.generate(torch.tensor([ids], dtype=torch.long), args.max_new_tokens, args.temperature,
                             args.top_k, args.repetition_penalty, end_id)[0].tolist()
        code = instructions.strip_special(tokenizer.decode(out[len(ids):])).strip("\n")
        print(code)
        if code and not instructions.balanced(code):
            print("\n[note] Brackets do not pair up: the output is probably cut off or wrong. "
                  "Review it by hand, or try --max-new-tokens or a lower --temperature.")

    if args.request:
        answer(" ".join(args.request))
        return
    print(f"MAD ({ckpt.name}). Type a request, or an empty line to quit.")
    while True:
        try:
            request = input("mad> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if request.lower() in ("", "exit", "quit"):
            break
        answer(request)
        print()


def command_inspect(args):
    config, from_file = ModelConfig.load_or_default(args.config)
    if not from_file:
        print(f"Note: {args.config} not found; showing built-in default configuration. "
              f"Run `python mad.py init` to create it.\n")
    model = MADIntelligence(config)
    count = model.num_parameters()
    print(json.dumps({"architecture": asdict(config), "parameters": count,
                      "parameter_storage_fp32_bytes": count * 4,
                      "training_ram_floor_bytes": count * 16,
                      "device": "cpu"}, indent=2))


def command_status(args):
    def line(ok, label, detail=""):
        print(f"  [{'x' if ok else ' '}] {label:<18}{detail}")

    print(f"MAD Intelligence {__version__}  ({ROOT})")
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        n = MADIntelligence(cfg).num_parameters()
        line(True, "config", f"{n:,} params, d_model={cfg.d_model}, layers={cfg.n_layers}, ctx={cfg.max_seq_len}")
    else:
        line(False, "config", "run: python mad.py init")
    raw = Path(ROOT / "data" / "raw")
    n_raw = sum(1 for p in raw.rglob("*")
                if p.is_file() and (p.suffix.lower() in data.TEXT_EXTENSIONS or p.suffix.lower() == ".jsonl")) if raw.is_dir() else 0
    line(n_raw > 0, "data/raw", f"{n_raw} supported files" if n_raw else "add code files here")
    corpus = Path(args.corpus)
    line(corpus.is_file(), "corpus", human_bytes(corpus.stat().st_size) if corpus.is_file() else "run: python mad.py prepare-data")
    tok = Path(args.tokenizer)
    if tok.is_file():
        try:
            line(True, "tokenizer", f"{MADTokenizer.load(tok).size} tokens")
        except ValueError as e:
            line(False, "tokenizer", f"{e}")
    else:
        line(False, "tokenizer", "run: python mad.py train-tokenizer")
    for name in ("mad_best.pt", "mad_latest.pt"):
        p = CKPT_DIR / name
        line(p.is_file(), name, f"{human_bytes(p.stat().st_size)}, {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}" if p.is_file() else "")

    def last_run(log, label):
        if log.is_file():
            with log.open("r", encoding="utf-8", errors="replace") as f:
                rows = [json.loads(x) for x in deque(f, maxlen=500) if x.strip()]
            if rows:
                last = rows[-1]
                vals = [r for r in rows if "val_loss" in r]
                msg = f"step {last['step']}, loss {last['loss']}"
                if vals:
                    msg += f", last val_loss {vals[-1]['val_loss']} (step {vals[-1]['step']})"
                print(f"  {label}: {msg}")

    last_run(CKPT_DIR / "training.jsonl", "last pretraining")
    print("  --- natural-language stage ---")
    inst_dir = ROOT / "data" / "instructions"
    n_files = len(list(inst_dir.rglob("*.jsonl"))) if inst_dir.is_dir() else 0
    line(n_files > 0, "instructions", f"{n_files} .jsonl files in data/instructions" if n_files else "run: python mad.py make-pairs (or write pairs by hand)")
    sft = ROOT / "data" / "processed" / "sft.jsonl"
    if sft.is_file():
        line(True, "sft.jsonl", f"{sum(1 for _ in sft.open(encoding='utf-8'))} examples")
    else:
        line(False, "sft.jsonl", "run: python mad.py prepare-instructions")
    for name in ("mad_sft_best.pt", "mad_sft_latest.pt"):
        p = CKPT_DIR / name
        line(p.is_file(), name, f"{human_bytes(p.stat().st_size)}, {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}" if p.is_file() else "")
    last_run(CKPT_DIR / "mad_sft.jsonl", "last fine-tune")


def command_doctor(args):
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"PyTorch {torch.__version__}; CPU cores: {os.cpu_count()}; default training threads: {default_threads()}")
    total, avail = memory_info()
    if total:
        print(f"RAM: {human_bytes(avail)} available of {human_bytes(total)} (changes as Android runs other apps)")
    print(f"Free disk: {human_bytes(shutil.disk_usage(ROOT).free)}")
    config, _ = ModelConfig.load_or_default(args.config)
    configure_threads(args.threads)
    model = MADIntelligence(config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    seq = min(config.max_seq_len, 128)
    x = torch.randint(0, config.vocab_size, (4, seq))
    times = []
    for _ in range(4):
        t = time.time()
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        times.append(time.time() - t)
    per = sum(times[1:]) / 3
    n = model.num_parameters()
    del model, opt, x, loss
    gc.collect()
    print(f"Speed test ({n:,} params, batch 4 x {seq}): {per:.2f}s/step, ~{4 * seq / per:.0f} tokens/s")
    print(f"Estimate with defaults (grad-accum 4): ~{per * 4 * 500 / 60:.0f} min for 500 steps")
    if avail and n * 16 > avail * 0.5:
        print("Warning: this model would use over half of the free RAM; use a smaller preset.")


# ----------------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(prog="mad", description="MAD Intelligence: offline model training and inference")
    p.add_argument("--version", action="version", version=f"MAD Intelligence {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        s = sub.add_parser(name, help=help_text)
        s.set_defaults(func=func)
        return s

    s = add("init", command_init, "create config.json for a model preset")
    s.add_argument("--preset", choices=list(PRESETS), default="tiny")
    s.add_argument("--vocab-size", type=int, default=None)
    s.add_argument("--seed", type=int, default=12403)
    s.add_argument("--config", default=rp("config.json"))
    s.add_argument("--checkpoint-dir", default=str(CKPT_DIR))
    s.add_argument("--force", action="store_true", help="overwrite an existing config.json")

    s = add("prepare-data", command_prepare, "collect, dedupe and write the training corpus + manifest")
    s.add_argument("--input", nargs="+", default=[rp("data/raw")], help="one or more folders or files")
    s.add_argument("--output", default=rp("data/processed/corpus.txt"))
    s.add_argument("--max-file-mb", type=float, default=5.0)
    s.add_argument("--no-secret-scan", action="store_true", help="include files that look like they hold secrets (not recommended)")

    s = add("train-tokenizer", command_tokenizer, "train the BPE tokenizer on the prepared corpus")
    s.add_argument("--sft", default=rp("data/processed/sft.jsonl"), help="instruction examples to include, if present")
    s.add_argument("--corpus", default=rp("data/processed/corpus.txt"))
    s.add_argument("--output", default=rp("tokenizer/mad_bpe.json"))
    s.add_argument("--vocab-size", type=int, default=1024)
    s.add_argument("--config", default=rp("config.json"))

    s = add("train", command_train, "train the model")
    s.add_argument("--config", default=rp("config.json"))
    s.add_argument("--tokenizer", default=rp("tokenizer/mad_bpe.json"))
    s.add_argument("--corpus", default=rp("data/processed/corpus.txt"))
    s.add_argument("--output-dir", default=str(CKPT_DIR))
    s.add_argument("--steps", type=int, default=500, help="total target step number")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--seq-len", type=int, default=None, help="default: the model context length")
    s.add_argument("--grad-accum", type=int, default=4)
    s.add_argument("--learning-rate", type=float, default=5e-4)
    s.add_argument("--warmup", type=int, default=None, help="warmup steps (default: 10%% of steps, max 100)")
    s.add_argument("--eval-every", type=int, default=50)
    s.add_argument("--eval-batches", type=int, default=10)
    s.add_argument("--checkpoint-every", type=int, default=50)
    s.add_argument("--threads", type=int, default=None, help="CPU threads (default: half the cores)")
    s.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this many minutes")
    s.add_argument("--seed", type=int, default=12403)
    s.add_argument("--resume", default=None)

    s = add("eval", command_eval, "held-out loss and perplexity of a checkpoint")
    s.add_argument("--checkpoint", default=None, help="default: mad_best.pt, else mad_latest.pt")
    s.add_argument("--tokenizer", default=rp("tokenizer/mad_bpe.json"))
    s.add_argument("--corpus", default=rp("data/processed/corpus.txt"))
    s.add_argument("--batches", type=int, default=50)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--seq-len", type=int, default=None)
    s.add_argument("--threads", type=int, default=None)

    s = add("generate", command_generate, "generate text from a prompt")
    s.add_argument("prompt")
    s.add_argument("--checkpoint", default=None, help="default: mad_best.pt, else mad_latest.pt")
    s.add_argument("--tokenizer", default=rp("tokenizer/mad_bpe.json"))
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--repetition-penalty", type=float, default=1.1)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--no-stop", action="store_true", help="do not stop at <MAD_END>")

    s = add("make-pairs", command_make_pairs, "extract request/code pairs from documented Kotlin/Java in your projects")
    s.add_argument("--input", nargs="+", default=[rp("data/raw")], help="folders or files with .kt/.java")
    s.add_argument("--output", default=rp("data/instructions/auto_from_code.jsonl"))
    s.add_argument("--no-secret-scan", action="store_true")

    s = add("prepare-instructions", command_prepare_instructions, "validate and clean request/code pairs for fine-tuning")
    s.add_argument("--input", default=rp("data/instructions"), help="a .jsonl file or a folder of them")
    s.add_argument("--output", default=rp("data/processed/sft.jsonl"))
    s.add_argument("--no-secret-scan", action="store_true")

    s = add("finetune", command_finetune, "teach the model to answer English requests with code")
    s.add_argument("--config", default=rp("config.json"))
    s.add_argument("--tokenizer", default=rp("tokenizer/mad_bpe.json"))
    s.add_argument("--sft", default=rp("data/processed/sft.jsonl"))
    s.add_argument("--base", default=None, help="pretrained checkpoint to start from (default: mad_best.pt)")
    s.add_argument("--from-scratch", action="store_true", help="skip pretraining (only sensible with lots of pairs)")
    s.add_argument("--output-dir", default=str(CKPT_DIR))
    s.add_argument("--steps", type=int, default=300, help="total target step number")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--seq-len", type=int, default=None)
    s.add_argument("--grad-accum", type=int, default=2)
    s.add_argument("--learning-rate", type=float, default=3e-4)
    s.add_argument("--warmup", type=int, default=None)
    s.add_argument("--eval-every", type=int, default=25)
    s.add_argument("--eval-batches", type=int, default=5)
    s.add_argument("--checkpoint-every", type=int, default=50)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--max-minutes", type=float, default=None)
    s.add_argument("--seed", type=int, default=12403)
    s.add_argument("--resume", default=None, help="resume an interrupted fine-tune (checkpoints/mad_sft_latest.pt)")

    s = add("ask", command_ask, "ask in plain English; the model answers with code")
    s.add_argument("request", nargs="*", help="what you want; omit for an interactive prompt")
    s.add_argument("--context-file", default=None, help="existing code the request refers to")
    s.add_argument("--checkpoint", default=None, help="default: mad_sft_best.pt, else the latest available")
    s.add_argument("--tokenizer", default=rp("tokenizer/mad_bpe.json"))
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.4)
    s.add_argument("--top-k", type=int, default=20)
    s.add_argument("--repetition-penalty", type=float, default=1.05)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--threads", type=int, default=None)

    s = add("inspect", command_inspect, "show architecture and parameter count")
    s.add_argument("--config", default=rp("config.json"))

    s = add("status", command_status, "show what exists in the pipeline")
    s.add_argument("--config", default=rp("config.json"))
    s.add_argument("--corpus", default=rp("data/processed/corpus.txt"))
    s.add_argument("--tokenizer", default=rp("tokenizer/mad_bpe.json"))

    s = add("doctor", command_doctor, "check the environment and measure training speed")
    s.add_argument("--config", default=rp("config.json"))
    s.add_argument("--threads", type=int, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
