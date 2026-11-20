#!/usr/bin/env python3
"""Is the model learning?  Run from anywhere:  python3 ~/MI/diagnose.py [--checkpoint PATH] [--skip-overfit]

1. Held-out loss of the trained checkpoint next to trivial unigram / bigram models, using the same
   token cache and the same train/held-out split the trainer uses.
2. Summary of checkpoints/training.jsonl: loss, grad_norm spikes, non-finite steps, where runs stopped.
3. Overfit test: can this architecture + optimizer memorize one fixed batch? If not, the problem is
   the model or optimizer, not the data.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mad import data  # noqa: E402
from mad.cli import VAL_FRACTION, read_tokenizer, resolve_checkpoint  # noqa: E402
from mad.config import ModelConfig  # noqa: E402
from mad.model import MADIntelligence  # noqa: E402
from mad.trainer import evaluate_loss, load_checkpoint  # noqa: E402

MAX_BIGRAM_VOCAB = 8192  # the bigram table is vocab^2 floats


def baselines(train, val, vocab):
    """Held-out cross-entropy (nats/token) of add-k smoothed unigram and bigram models."""
    counts = torch.bincount(train, minlength=vocab).float() + 0.5
    uni = -torch.log((counts / counts.sum())[val]).mean().item()
    if vocab > MAX_BIGRAM_VOCAB:
        return uni, None
    pairs = torch.bincount(train[:-1] * vocab + train[1:], minlength=vocab * vocab).float().view(vocab, vocab) + 0.05
    cond = pairs / pairs.sum(1, keepdim=True)
    return uni, -torch.log(cond[val[:-1], val[1:]]).mean().item()


def ppl(loss):
    return math.exp(min(loss, 50))


def section_model(args):
    print("== 1. trained model vs trivial baselines ==")
    ckpt = resolve_checkpoint(args.checkpoint)
    state = load_checkpoint(ckpt)
    config = ModelConfig(**state["config"])
    tok_path = ROOT / "tokenizer/mad_bpe.json"
    corpus = ROOT / "data/processed/corpus.txt"
    if not corpus.is_file():
        print(f"no corpus at {corpus}; skipping")
        return None, None, None
    tokenizer = read_tokenizer(str(tok_path))
    if tokenizer.size != config.vocab_size:
        print(f"tokenizer size {tokenizer.size} != checkpoint vocab {config.vocab_size}; skipping")
        return None, None, None
    tokens, _ = data.load_or_encode_tokens(tokenizer, str(tok_path), corpus, corpus.with_name("tokens.pt"))
    train, val = data.split_tokens(tokens, VAL_FRACTION)
    if val is None:
        print("corpus too small for a held-out split; skipping")
        return None, None, None
    train, val = train.long(), val.long()
    uni, big = baselines(train, val, config.vocab_size)

    model = MADIntelligence(config)
    model.load_state_dict(state["model"])
    seq = min(config.max_seq_len, val.numel() - 2)
    loss = evaluate_loss(model, val, 4, seq, args.batches)

    print(f"checkpoint {ckpt.name} (step {state['step']}), {model.num_parameters():,} params, "
          f"{train.numel():,} train / {val.numel():,} held-out tokens, vocab {config.vocab_size}")
    print(f"  uniform guess   loss {math.log(config.vocab_size):.3f}  perplexity {config.vocab_size}")
    print(f"  unigram model   loss {uni:.3f}  perplexity {ppl(uni):.0f}")
    if big is not None:
        print(f"  bigram model    loss {big:.3f}  perplexity {ppl(big):.0f}")
    print(f"  this checkpoint loss {loss:.3f}  perplexity {ppl(loss):.1f}")
    ref = big if big is not None else uni
    name = "bigram" if big is not None else "unigram"
    if loss >= uni:
        print("  VERDICT: no better than a unigram model: it has learned nothing useful.")
    elif loss >= ref:
        print(f"  VERDICT: better than unigram but not than {name}: it has barely learned context.")
    else:
        print(f"  VERDICT: learning. {ref - loss:.2f} nats/token below the {name} baseline.")
    return tokens, config, uni


def section_log():
    print("\n== 2. training log ==")
    path = ROOT / "checkpoints/training.jsonl"
    if not path.is_file():
        print("checkpoints/training.jsonl not found")
        return
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = [r for r in rows if "step" in r]
    print(f"{len(rows)} logged rows (all runs appended together)")
    norms = sorted(r["grad_norm"] for r in rows if isinstance(r.get("grad_norm"), (int, float)) and r["grad_norm"] == r["grad_norm"])
    median = norms[len(norms) // 2] if norms else 0.0
    if norms:
        print(f"grad_norm: median {median:.3f}, max {norms[-1]:.3f}")
    losses = [r["loss"] for r in rows if isinstance(r.get("loss"), (int, float))]
    if len(losses) >= 200:
        k = max(50, len(losses) // 20)
        print(f"loss: first {k} steps avg {sum(losses[:k]) / k:.3f} -> last {k} steps avg {sum(losses[-k:]) / k:.3f}")
    vals = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    if vals:
        best = min(vals, key=lambda v: v[1])
        print(f"val_loss: best {best[1]:.4f} at step {best[0]}; last {vals[-1][1]:.4f} at step {vals[-1][0]}")
    shown = 0
    for i, r in enumerate(rows):
        gn = r.get("grad_norm")
        bad = gn is None or gn != gn or (median and gn > 5 * median)
        last = i + 1 == len(rows) or rows[i + 1]["step"] <= r["step"]
        if (bad and gn is not None) or last:
            tag = "  <- last step of a run" if last else "  <- grad_norm spike / non-finite"
            print(f"  step {r['step']:>6}  loss {r.get('loss')}  grad_norm {gn}  lr {r.get('lr', 0):.2e}{tag}")
            shown += 1
            if shown >= 25:
                print("  ... (more omitted)")
                break


def section_overfit(tokens, config, uni):
    print("\n== 3. can the model + optimizer learn at all? ==")
    if tokens is None or config is None:
        print("skipped (needs the token cache and a checkpoint/config)")
        return
    torch.manual_seed(0)
    cfg = config
    model = MADIntelligence(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    seq = min(cfg.max_seq_len, 64)
    starts = torch.randint(0, tokens.numel() - seq - 1, (8,))
    idx = starts[:, None] + torch.arange(seq)[None, :]
    x, y = tokens[idx].long(), tokens[idx + 1].long()
    print(f"8 x {seq} tokens, 150 steps (unigram baseline {uni:.2f}); a healthy model goes below 1.0")
    first = last = None
    for step in range(1, 151):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
        if step in (1, 10, 50, 100, 150):
            print(f"  step {step:>3}  loss {last:.3f}  grad_norm {norm:.2f}")
    verdict = "PASS" if last < 1.0 else "FAIL"
    print(f"{verdict}: loss {first:.2f} -> {last:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="default: mad_best.pt, else mad_latest.pt")
    ap.add_argument("--batches", type=int, default=50, help="held-out batches to score")
    ap.add_argument("--skip-overfit", action="store_true")
    args = ap.parse_args()
    tokens, config, uni = section_model(args)
    section_log()
    if not args.skip_overfit:
        section_overfit(tokens, config, uni)


if __name__ == "__main__":
    main()
