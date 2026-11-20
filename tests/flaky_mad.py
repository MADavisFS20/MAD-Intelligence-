"""Wrapper around mad.py that injects failures, to test self-healing.

MI_FLAKY="nan:1,oom:1,corrupt:1" -> the first N train/finetune calls of each kind fail with the
message the real trainer would print. Counters live in $MI_FLAKY_DIR so they survive across calls.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd in ("train", "finetune") and os.environ.get("MI_FLAKY"):
    d = Path(os.environ["MI_FLAKY_DIR"]); d.mkdir(exist_ok=True)
    for item in os.environ["MI_FLAKY"].split(","):
        kind, n = item.split(":")
        c = d / f"{kind}.count"
        done = int(c.read_text()) if c.exists() else 0
        if done < int(n):
            c.write_text(str(done + 1))
            if kind == "inf":  # noisy but not fatal: trainer prints skip warnings and carries on
                for i in range(12):
                    print(f"Warning: non-finite grad_norm (inf) at step {i}; skipping optimizer step.", flush=True)
                continue
            if kind == "nan":
                sys.exit("Training stopped: Loss became nan at step 3. Lower --learning-rate and resume from the last checkpoint.")
            if kind == "oom":
                print("RuntimeError: [enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate memory", file=sys.stderr)
                sys.exit(1)
            if kind == "kill":
                os._exit(137)
            if kind == "unknown":
                sys.exit("Something strange happened")
            if kind == "corrupt":
                p = ROOT / "checkpoints" / ("mad_sft_latest.pt" if cmd == "finetune" else "mad_latest.pt")
                if p.exists():
                    p.write_bytes(b"garbage")
                sys.exit(f"Training stopped: Could not read checkpoint {p}: invalid")
from mad.cli import main
main()
