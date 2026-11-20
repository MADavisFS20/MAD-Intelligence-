import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .model import MADIntelligence


def default_threads():
    """Half the cores (min 1): a good speed/heat balance on a phone."""
    return max(1, (os.cpu_count() or 2) // 2)


def configure_threads(threads=None):
    torch.set_num_threads(threads or default_threads())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def get_batch(tokens, batch_size, seq_len, generator=None):
    max_start = tokens.numel() - seq_len - 1
    if max_start < 0:
        raise ValueError("Encoded corpus is too short for the selected context length")
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    idx = starts[:, None] + torch.arange(seq_len)[None, :]
    return tokens[idx].long(), tokens[idx + 1].long()


def lr_at(step, total_steps, base_lr, warmup, min_ratio=0.1):
    """Linear warmup then cosine decay to min_ratio * base_lr."""
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    if total_steps <= warmup:
        return base_lr
    progress = min(1.0, (step - warmup) / (total_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate_batches(model, batch_fn, batches, seed=1234):
    """Mean loss over `batches` batches from batch_fn(generator); deterministic for a given seed."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(batches):
        x, y = batch_fn(gen)
        _, loss = model(x, y)
        total += float(loss)
    model.train(was_training)
    return total / batches


def evaluate_loss(model, tokens, batch_size, seq_len, batches, seed=1234):
    return evaluate_batches(model, lambda g: get_batch(tokens, batch_size, seq_len, g), batches, seed)


def save_checkpoint(path, model, optimizer, step, config, metadata=None, save_optimizer=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    payload = {
        "step": step,
        "model": model.state_dict(),
        "config": asdict(config),
        "metadata": metadata or {},
    }
    if save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, tmp)
    Path(tmp).replace(path)
    del payload
    gc.collect()


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # corrupt or foreign file
        raise RuntimeError(f"Could not read checkpoint {path}: {e}") from e


def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def train_model(train_tokens, val_tokens, config, output_dir, *, steps, batch_size, seq_len,
                grad_accum, learning_rate, warmup, eval_every=50, eval_batches=10,
                checkpoint_every=50, resume=None, threads=None, max_minutes=None,
                seed=12403, metadata=None, log_every=10,
                train_batch_fn=None, val_batch_fn=None, init_weights=None, prefix="mad"):
    """Train from token tensors, or from custom batch functions (used for instruction tuning).

    prefix names the files: <prefix>_latest.pt, <prefix>_best.pt. init_weights loads model
    weights only (new optimizer, step 0); resume restores weights, optimizer and step.
    """
    if seq_len > config.max_seq_len:
        raise ValueError(f"--seq-len {seq_len} exceeds the model context length {config.max_seq_len}")
    configure_threads(threads)
    torch.manual_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    if train_batch_fn is None:
        train_batch_fn = lambda g=None: get_batch(train_tokens, batch_size, seq_len, g)
    if val_batch_fn is None and val_tokens is not None:
        val_batch_fn = lambda g=None: get_batch(val_tokens, batch_size, seq_len, g)

    model = MADIntelligence(config)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    start_step, best_val = 0, float("inf")
    if init_weights:
        base = load_checkpoint(init_weights)
        if base["config"] != asdict(config):
            raise ValueError("Base checkpoint architecture does not match config.json.")
        model.load_state_dict(base["model"])
        base_step = base["step"]
        del base
        gc.collect()
        print(f"Starting from weights in {init_weights} (step {base_step}).")
    if resume:
        state = load_checkpoint(resume)
        if state["config"] != asdict(config):
            raise ValueError("Checkpoint architecture does not match config.json; cannot resume.")
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except ValueError:
                print("Warning: optimizer state is from an older version; starting the optimizer fresh.")
        else:
            print("Note: checkpoint has no optimizer state; starting the optimizer fresh.")
        start_step = int(state["step"])
        best_val = float(state.get("metadata", {}).get("best_val_loss", best_val))
        del state
        gc.collect()
        print(f"Resumed from {resume} at step {start_step}.")
    if start_step >= steps:
        print(f"Checkpoint is already at step {start_step} >= --steps {steps}; nothing to train. "
              f"--steps is the total target, so raise it.")
        return {"step": start_step, "stop_reason": "nothing to do", "best_val_loss": best_val}

    def meta(extra=None):
        m = dict(metadata)
        m["best_val_loss"] = best_val
        m.update(extra or {})
        return m

    latest, best = out / f"{prefix}_latest.pt", out / f"{prefix}_best.pt"
    log_path = out / ("training.jsonl" if prefix == "mad" else f"{prefix}.jsonl")
    tokens_per_step = batch_size * seq_len * grad_accum
    started = time.time()
    completed, last_saved, stop_reason, last_loss = start_step, start_step, "finished", None
    model.train()
    try:
        for step in range(start_step + 1, steps + 1):
            lr = lr_at(step, steps, learning_rate, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(grad_accum):
                x, y = train_batch_fn()
                _, loss = model(x, y)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"Loss became {value} at step {step}. Lower --learning-rate and resume "
                        f"from the last checkpoint.")
                (loss / grad_accum).backward()
                total += value
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(grad_norm):
                print(f"Warning: non-finite grad_norm ({grad_norm}) at step {step}; skipping optimizer step.",
                      flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            completed = step
            last_loss = total / grad_accum
            elapsed = time.time() - started
            record = {"step": step, "loss": round(last_loss, 5), "lr": lr,
                      "grad_norm": round(grad_norm, 4),
                      "tokens_per_sec": round(tokens_per_step * (step - start_step) / max(elapsed, 1e-9), 1),
                      "elapsed_seconds": round(elapsed, 2)}
            line = None
            if val_batch_fn is not None and eval_every and (step % eval_every == 0 or step == steps):
                val = evaluate_batches(model, val_batch_fn, eval_batches)
                record["val_loss"] = round(val, 5)
                line = f"  validation: loss={val:.4f} perplexity={math.exp(min(val, 50)):.1f}"
                if val < best_val:
                    best_val = val
                    save_checkpoint(best, model, None, step, config, meta({"val_loss": val}),
                                    save_optimizer=False)
                    line += f"  (new best, saved {best.name})"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if step == start_step + 1 or step % log_every == 0 or step == steps:
                per_step = elapsed / (step - start_step)
                print(f"step={step}/{steps} loss={last_loss:.4f} lr={lr:.2e} "
                      f"{record['tokens_per_sec']:.0f} tok/s elapsed={_fmt_time(elapsed)} "
                      f"eta={_fmt_time(per_step * (steps - step))}", flush=True)
            if line:
                print(line, flush=True)
            if step % checkpoint_every == 0:
                save_checkpoint(latest, model, optimizer, step, config, meta())
                last_saved = step
            if max_minutes and elapsed > max_minutes * 60:
                stop_reason = f"time limit of {max_minutes} min reached"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted (Ctrl+C)"
    if completed > last_saved:
        save_checkpoint(latest, model, optimizer, completed, config, meta())
    if completed > start_step:
        print(f"Stopped at step {completed}: {stop_reason}. Saved {latest}"
              f" (resume with --resume {latest} --steps {steps}).")
    else:
        print(f"Stopped before completing a step: {stop_reason}. Nothing new was saved.")
    return {"step": completed, "stop_reason": stop_reason, "last_loss": last_loss,
            "best_val_loss": best_val}
