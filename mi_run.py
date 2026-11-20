#!/usr/bin/env python3
"""MI one-command pipeline: fetch legal data -> prepare -> tokenizer -> train -> fine-tune -> back up.

Normally started through ./start.sh. Safe to stop and re-run at any time: state lives in
state/pipeline.json, checkpoints in checkpoints/, and (if Hugging Face is set up) in your HF repo.

It heals known problems by itself and keeps going (see `classify` / `Healer` below):
  NaN loss -> halve the learning rate and resume from the last good checkpoint
  out of memory -> smaller batch (more accumulation), then fewer threads
  corrupt checkpoint -> fall back to the previous good one, then the best one, then the HF copy
  disk full -> delete regenerable caches and retry
  network down -> wait and retry; training proceeds with whatever data exists
  missing python package -> install it
  anything unknown -> retry a few times, back up the logs to HF, then stop with a clear message

Exit codes: 0 done | 2 needs a human (message printed) | 130 stopped by you | other: start.sh restarts it.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import hf_sync  # noqa: E402

PY = sys.executable
MAD = os.environ.get("MI_MAD_SCRIPT", str(ROOT / "mad.py"))
TOOLS = str(ROOT / "mi_tools.py")
CK = ROOT / "checkpoints"
STATE_PATH = ROOT / "state/pipeline.json"
LOG_PATH = ROOT / "logs/mi_run.log"
MIN_RAW_BYTES = int(os.environ.get("MI_MIN_RAW_BYTES", 400_000))
STOP = threading.Event()
CURRENT = {"proc": None}
LAST_RUN = {"skipped": 0}  # "non-finite grad_norm" warnings seen in the most recent run()
VERSION = "0.4.5"
import socket

HOLD_FILE = ROOT / "state/HOLD"
PENDING_SYNC_FILE = ROOT / "state/PENDING_HF_SYNC"


class Fatal(Exception):
    """Needs a human. Exit code 2."""


class Redo(Exception):
    """Re-run the earlier pipeline stages (e.g. more data needed)."""


# ----------------------------------------------------------------------------- logging
def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    write_log(line + "\n")


def write_log(text):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 8_000_000:
            LOG_PATH.replace(LOG_PATH.with_suffix(".log.1"))
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(hf_sync.redact(text))
    except OSError:
        pass


# ----------------------------------------------------------------------------- state
class State(dict):
    def __init__(self):
        super().__init__(done={}, hp={}, heals=[], train_seconds=0.0, version=VERSION)
        if STATE_PATH.exists():
            try:
                self.update(json.loads(STATE_PATH.read_text()))
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self, indent=1))
        tmp.replace(STATE_PATH)


# ----------------------------------------------------------------------------- processes
def on_signal(signum, _frame):
    STOP.set()
    p = CURRENT["proc"]
    if p and p.poll() is None:
        try:
            p.send_signal(signal.SIGINT)  # the trainer saves a checkpoint on SIGINT
        except OSError:
            pass


def run(cmd, timeout=None, env=None, echo=True):
    """Run a command, stream and log its output, return (returncode, last_lines_text)."""
    tail = deque(maxlen=150)
    LAST_RUN["skipped"] = 0
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"}, cwd=ROOT)
    CURRENT["proc"] = p
    timer = None
    if timeout:
        timer = threading.Timer(timeout, p.kill)
        timer.start()
    try:
        for line in p.stdout:
            tail.append(line)
            if "non-finite grad_norm" in line:
                LAST_RUN["skipped"] += 1
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            write_log(line)
        rc = p.wait()
    finally:
        if timer:
            timer.cancel()
        CURRENT["proc"] = None
    return rc, "".join(tail)


def mad(*args, **kw):
    return run([PY, MAD, *map(str, args)], **kw)


def tool_json(*args):
    r = subprocess.run([PY, TOOLS, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return None
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def ckpt_info(path):
    return tool_json("ckpt-step", path) if Path(path).is_file() else None


def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError):
        pass
    return None


def mem_total_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError):
        pass
    return None


def sleep_interruptible(seconds):
    end = time.time() + seconds
    while time.time() < end and not STOP.is_set():
        time.sleep(min(2, end - time.time()))


def has_wifi_or_net(require_wifi: bool = True) -> bool:
    """Return True if network is reachable (and on Wi-Fi if require_wifi=True)."""
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=2.5):
            pass
    except OSError:
        return False

    if not require_wifi:
        return True

    if shutil.which("termux-wifi-connectioninfo"):
        try:
            r = subprocess.run(
                ["termux-wifi-connectioninfo"],
                capture_output=True, text=True, timeout=4
            )
            if r.returncode == 0 and r.stdout.strip():
                info = json.loads(r.stdout)
                return bool(info.get("supplicant_state") == "COMPLETED" and info.get("ip"))
        except Exception:
            pass

    try:
        route = Path("/proc/net/route").read_text()
        for line in route.splitlines()[1:]:
            cols = line.split()
            if len(cols) >= 4 and cols[0].startswith("wlan") and cols[1] == "00000000":
                return True
        return False
    except OSError:
        return True


def wait_if_held(args, st, reason_prefix: str = ""):
    """Pause cleanly while state/HOLD exists or (if --pause-without-wifi) Wi-Fi is off."""
    announced = False
    while not STOP.is_set():
        manual_hold = HOLD_FILE.exists()
        wifi_hold = getattr(args, "pause_without_wifi", False) and not has_wifi_or_net(require_wifi=True)
        if not manual_hold and not wifi_hold:
            if announced:
                log(f"{reason_prefix}resuming now.")
            return
        if not announced:
            why = "state/HOLD present" if manual_hold else "off Wi-Fi (--pause-without-wifi)"
            log(f"{reason_prefix}HOLD active ({why}). Progress saved; waiting to resume...")
            st.save()
            announced = True
        sleep_interruptible(10)


# ----------------------------------------------------------------------------- failure handling
def classify(tail, rc):
    """Map a failed command's output to a failure kind (pure function, unit tested)."""
    t = (tail or "").lower()
    if rc in (-9, 137) or "out of memory" in t or "cannot allocate memory" in t or "memoryerror" in t \
            or "defaultcpuallocator" in t:
        return "oom"
    if "loss became" in t or "non-finite" in t or "floatingpointerror" in t:
        return "nan"
    if "no space left" in t or "disk quota" in t:
        return "disk"
    if "could not read checkpoint" in t or "checkpoint not found" in t or "pytorchstreamreader" in t:
        return "bad_checkpoint"
    if "architecture does not match" in t or "size mismatch" in t:
        return "arch_mismatch"
    if "tokenizer size" in t and "does not match" in t:
        return "vocab_mismatch"
    if "need at least" in t and "training tokens" in t:
        return "small_corpus"
    if "pytorch is not installed" in t or "no module named 'torch'" in t:
        return "no_torch"
    if "no module named" in t:
        return "missing_module"
    if "corpus not found" in t or "tokenizer not found" in t or "config file not found" in t:
        return "missing_input"
    if any(w in t for w in ("urlopen error", "connection reset", "timed out", "name resolution",
                            "temporary failure", "network is unreachable", "max retries")):
        return "network"
    return "unknown"


def free_disk(st):
    freed = 0
    for stale in ROOT.glob(".hf_stage*"):
        shutil.rmtree(stale, ignore_errors=True)
    for rel in ("data/processed/tokens.pt", "checkpoints/random_init.pt", ".hf_restore",
                "data/hf_cache", "logs/mi_run.log.1"):
        p = ROOT / rel
        try:
            if p.is_dir():
                freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                freed += p.stat().st_size
                p.unlink()
        except OSError:
            pass
    return freed


class Healer:
    """Decides what to change before retrying a failed training chunk."""

    LIMITS = {"nan": 6, "oom": 4, "disk": 2, "bad_checkpoint": 4, "network": 8, "unknown": 3,
              "no_torch": 1, "missing_module": 2, "missing_input": 2, "small_corpus": 1}

    def __init__(self, st, args, hfs):
        self.st, self.args, self.hfs = st, args, hfs
        self.consec = {}

    def success(self):
        self.consec.clear()

    def record(self, kind, note):
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind, "action": note}
        self.st["heals"] = (self.st.get("heals", []) + [entry])[-200:]
        self.st.save()
        rep = ROOT / "reports"
        rep.mkdir(exist_ok=True)
        with (rep / "heals.md").open("a", encoding="utf-8") as f:
            f.write(f"- {entry['time']} **{kind}**: {note}\n")
        log(f"SELF-HEAL [{kind}]: {note}")

    def handle(self, kind, tail, stage):
        n = self.consec[kind] = self.consec.get(kind, 0) + 1
        if n > self.LIMITS.get(kind, 3):
            raise Fatal(f"{stage}: '{kind}' failed {n - 1} times in a row even after automatic fixes.\n"
                        f"Last output:\n{tail[-1500:]}\nLogs were backed up to Hugging Face if it is set up. "
                        f"See logs/mi_run.log; fix the cause and re-run ./start.sh (it resumes).")
        hp = self.st["hp"].setdefault(stage, {})
        if kind == "nan":
            base = self.args.lr * (0.6 if stage == "sft" else 1.0)
            hp["lr"] = max(1e-5, hp.get("lr", base) * 0.5)
            self.record(kind, f"loss went non-finite; learning rate -> {hp['lr']:.2e}, resuming from last good checkpoint")
        elif kind == "oom":
            b = hp.get("batch_size", self.args.batch_size)
            a = hp.get("grad_accum", self.args.grad_accum)
            t = hp.get("threads", self.args.threads or max(1, (os.cpu_count() or 2) // 2))
            if b > 1:
                hp["batch_size"], hp["grad_accum"] = b // 2, a * 2
                self.record(kind, f"out of memory; batch {b}->{b // 2}, grad-accum {a}->{a * 2} (same effective batch)")
            elif t > 1:
                hp["threads"] = max(1, t // 2)
                self.record(kind, f"out of memory at batch 1; threads {t}->{hp['threads']}")
            else:
                raise Fatal(f"{stage}: out of memory even at batch size 1 and 1 thread. Close other apps, or use "
                            f"a smaller model: ./start.sh --fresh --preset tiny")
        elif kind == "disk":
            freed = free_disk(self.st)
            self.record(kind, f"disk full; deleted regenerable caches ({freed / 1e6:.0f} MB)")
            if freed == 0:
                raise Fatal("Disk is full and nothing regenerable could be deleted. Free some space and re-run.")
        elif kind == "bad_checkpoint":
            for name in ("mad_latest.pt", "mad_sft_latest.pt"):
                p = CK / name
                if p.is_file() and not ckpt_info(p):
                    p.replace(p.with_name(p.name + f".corrupt-{int(time.time())}"))
            restored = self.hfs.pull(force=False) if self.hfs.enabled else []
            self.record(kind, f"checkpoint unreadable; quarantined it and fell back to previous/best"
                              f"{' / restored from Hugging Face: ' + ', '.join(restored) if restored else ''}")
        elif kind == "network":
            wait = min(600, 30 * 2 ** (n - 1))
            self.record(kind, f"network problem; waiting {wait}s then retrying")
            sleep_interruptible(wait)
        elif kind == "no_torch":
            if not install_torch():
                raise Fatal("PyTorch is missing and could not be installed. In Termux: pkg install python-torch")
            self.record(kind, "installed PyTorch")
        elif kind == "missing_module":
            m = re.search(r"no module named '([\w.]+)'", tail.lower())
            name = m.group(1).split(".")[0] if m else ""
            if name not in ("numpy", "huggingface_hub", "pytest"):
                raise Fatal(f"{stage}: missing python module '{name}'. Install it and re-run.")
            pip_install(name)
            self.record(kind, f"installed {name}")
        elif kind in ("arch_mismatch", "vocab_mismatch"):
            raise Fatal(f"{stage}: the model files do not match config.json/tokenizer (you changed the preset or "
                        f"data after training started). To start over: ./start.sh --fresh (old files are moved to "
                        f"backup_*/). To keep training, restore the original config.json and tokenizer/.")
        elif kind == "small_corpus":
            self.args.mb += 20
            for k in ("data", "tokenizer", "pretrain"):
                self.st["done"].pop(k, None)
            self.st.pop("total_steps", None)
            self.record(kind, f"corpus too small for this model; fetching {self.args.mb:.0f} MB total and redoing prep")
            raise Redo()
        elif kind == "missing_input":
            for k in ("prepare", "tokenizer"):
                self.st["done"].pop(k, None)
            self.record(kind, "a pipeline input file was missing; rebuilding data prep and tokenizer")
            raise Redo()
        else:
            wait = 20 * n
            self.record(kind, f"unrecognised failure; retry {n}/{self.LIMITS['unknown']} in {wait}s. "
                              f"Last line: {tail.strip().splitlines()[-1][:160] if tail.strip() else ''}")
            sleep_interruptible(wait)
        self.st.save()


def pip_install(name):
    for extra in ([], ["--break-system-packages"], ["--user"]):
        if subprocess.run([PY, "-m", "pip", "install", "-q", name, *extra], capture_output=True).returncode == 0:
            return True
    return False


def install_torch():
    if shutil.which("pkg"):
        subprocess.run(["pkg", "install", "-y", "python-torch", "python-numpy"], capture_output=True)
    elif subprocess.run([PY, "-c", "import torch"], capture_output=True).returncode != 0:
        pip_install("torch")
    return subprocess.run([PY, "-c", "import torch"], capture_output=True).returncode == 0


# ----------------------------------------------------------------------------- background pushes
class Pusher:
    def __init__(self, hfs):
        self.hfs, self.thread = hfs, None

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def submit(self, message, heavy=True, card=None, wait=False):
        if not self.hfs.enabled:
            return False
        if self.busy():
            if not wait:
                return False
            self.thread.join()
        self.thread = threading.Thread(target=self.hfs.push, args=(message, heavy, card), daemon=False)
        self.thread.start()
        if wait:
            self.thread.join()
        return True

    def drain(self):
        if self.busy():
            log("waiting for the last Hugging Face upload to finish ...")
            self.thread.join()


def model_card(st, args):
    ev = st.get("eval", {})
    lines = ["---", "library_name: pytorch", "tags: [code, linux, android, kotlin, from-scratch]", "license: other", "---", "",
             "# MI coder", "",
             "A small decoder-only Transformer trained **from random initialization** on permissively licensed "
             "open-source code (Android-focused, plus general Linux languages) by the MI pipeline. GitHub and "
             "Hugging Face datasets are used only as data sources; no pretrained weights were used.", "",
             f"- Preset: `{st.get('preset')}` | steps (pretrain): {st.get('pretrain_step', 0)}/{st.get('total_steps', '?')}"
             f" | steps (fine-tune): {st.get('sft_step', 0)}/{st.get('sft_total', '?')}",
             f"- Held-out loss: {ev.get('held_out_loss', 'n/a')} | pipeline version {VERSION}",
             "- Training data sources and licenses: `data/THIRD_PARTY_NOTICES.md`, `data/manifest.json`, `data/licenses/`",
             "- Resume anywhere: `git clone` the MI project, set `HF_TOKEN`, run `./start.sh --hf-repo THIS_REPO`.", "",
             "This is a small experimental model; check its output before using it."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- stages
def stage_preflight(args, st):
    if sys.version_info < (3, 9):
        raise Fatal("Python 3.9+ is required.")
    if subprocess.run([PY, "-c", "import torch"], capture_output=True).returncode != 0:
        log("PyTorch not found; trying to install it ...")
        if not install_torch():
            raise Fatal("PyTorch is not installed. In Termux run: pkg install python python-numpy python-torch")
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 0.3:
        freed = free_disk(st)
        if shutil.disk_usage(ROOT).free / 1e9 < 0.3:
            raise Fatal(f"Only {free:.2f} GB free disk; free some space and re-run.")
        log(f"low disk; cleared {freed / 1e6:.0f} MB of caches")
    elif free < 1.5:
        log(f"warning: only {free:.1f} GB free disk")
    ram, total = mem_available_gb(), mem_total_gb()
    # Android's MemAvailable swings with whatever apps are cached, so it alone can pick "tiny" on a
    # phone with plenty of RAM. Use the larger of it and 40% of total RAM for the choice.
    usable = max(ram or 0.0, 0.4 * (total or 0.0)) or ram
    prev = st.get("preset")
    if have_pretrained():
        st["preset"] = prev or choose_preset(args, usable)   # weights exist: keep the model size
    else:
        st["preset"] = choose_preset(args, usable)           # nothing trained yet: free to change
        if prev and prev != st["preset"]:
            st["done"].pop("tokenizer", None)                # vocab size depends on the preset
            st.pop("total_steps", None)
            log(f"preset changed {prev} -> {st['preset']} before any training; rebuilding the tokenizer/config")
    log(f"MI {VERSION} | python {sys.version.split()[0]} | RAM {ram:.1f} GB free of {total:.1f} GB | preset {st['preset']}"
        if ram and total else f"MI {VERSION} | python {sys.version.split()[0]} | preset {st['preset']}")
    if st["preset"] == "tiny" and args.preset == "auto":
        log("note: auto-selected the 'tiny' preset (~0.5M parameters), which plateaus at high loss. "
            "If the phone can spare the RAM, use --preset small (with --fresh if training already started).")


def choose_preset(args, ram):
    if args.preset != "auto":
        return args.preset
    if ram is None:
        return "small"
    return "medium" if ram >= 6 else "small" if ram >= 2.5 else "tiny"


def raw_bytes():
    """Total size of everything in data/raw/ (fetched .jsonl and any files you added by hand)."""
    raw = ROOT / "data/raw"
    if not raw.is_dir():
        return 0
    return sum(f.stat().st_size for f in raw.rglob("*") if f.is_file() and not f.name.endswith(".keep"))


def stage_restore(args, st, hfs):
    if args.fresh or not hfs.enabled:
        return
    have_local = (CK / "mad_latest.pt").is_file() or (CK / "mad_best.pt").is_file()
    if not have_local and hfs.ensure():
        restored = hfs.pull()
        if restored:
            st.update(json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {})
            log("resuming from the copy saved on Hugging Face")


# A fetcher only needs the token for its own host; the HF token has WRITE scope (for the backup).
NO_HF_TOKEN = {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "", "HUGGINGFACE_HUB_TOKEN": ""}
NO_GH_TOKEN = {"GITHUB_TOKEN": "", "GH_TOKEN": ""}


def run_fetch(name, cmd, attempts=3, env=None):
    for i in range(attempts):
        if STOP.is_set():
            return False
        rc, tail = run(cmd, timeout=3600, env=env)
        if rc == 0:
            return True
        log(f"{name} exited with {rc} (attempt {i + 1}/{attempts})")
        sleep_interruptible([20, 60, 180][i])
    return False


def stage_data(args, st):
    if st["done"].get("data") and raw_bytes() >= MIN_RAW_BYTES and not args.refresh_data:
        log("data: already fetched (use --refresh-data to add more)")
        return
    if args.skip_data:
        log("data: fetching skipped (--skip-data)")
        if raw_bytes() < MIN_RAW_BYTES:
            raise Fatal("No training data in data/raw and --skip-data was given.")
        return
    waited = 0.0
    while True:
        before = raw_bytes()
        mb = args.mb
        log(f"== data: fetching about {mb:.0f} MB of permissively licensed code (GitHub + Hugging Face) ==")
        wait_if_held(args, st, "data: ")
        ls = min(max(args.linux_share, 0.0), 0.8)
        rest = mb * (1 - ls)
        run_fetch("github", [PY, "fetch_data.py", "github", "--mb", f"{rest * 0.5:.1f}"], env=NO_HF_TOKEN)
        if ls > 0:
            run_fetch("linux code (all languages)", [PY, "fetch_data.py", "linux", "--mb", f"{mb * ls:.1f}"], attempts=2,
                      env=NO_HF_TOKEN)
        if args.hf_data:
            run_fetch("huggingface (REST)", [PY, "fetch_data.py", "hf", "--mb", f"{rest * 0.35:.1f}"], env=NO_GH_TOKEN)
        if args.hf_data and args.hf_cli:
            cli_args = [PY, "fetch_data.py", "hf-files", "--mb", f"{max(1.0, rest * 0.15):.1f}"]
            if args.hf_java:
                cli_args += ["--only", "commitpackft-kotlin", "commitpackft-java"]
            run_fetch("huggingface (hf CLI)", cli_args, attempts=2, env=NO_GH_TOKEN)
        now = raw_bytes()
        log(f"data: raw code {before / 1e6:.2f} -> {now / 1e6:.2f} MB")
        if now >= MIN_RAW_BYTES:
            break
        if STOP.is_set():
            return
        if waited >= args.network_wait_min * 60:
            raise Fatal(f"Only {now / 1e3:.0f} KB of data could be fetched (need >= {MIN_RAW_BYTES // 1000} KB). "
                        f"Check the network, or put licensed code in data/raw/ and run ./start.sh --skip-data")
        log("not enough data yet (network down or rate-limited?); waiting 3 minutes and retrying")
        sleep_interruptible(180)
        waited += 180
    st["done"]["data"] = True
    st["done"].pop("prepare", None)
    st.save()


def stage_prepare(args, st):
    corpus = ROOT / "data/processed/corpus.txt"
    if st["done"].get("prepare") and corpus.is_file():
        return
    log("== prepare: dedupe, secret-scan, license manifest, corpus ==")
    rc, tail = run([PY, "prepare_data.py"])
    if rc != 0:
        raise Fatal(f"prepare_data.py failed:\n{tail[-800:]}")
    proc = ROOT / "data/processed"
    with corpus.open("wb") as out:
        for name in ("train.txt", "val.txt"):
            with (proc / name).open("rb") as f:
                shutil.copyfileobj(f, out)
    (ROOT / "data/instructions").mkdir(parents=True, exist_ok=True)
    pairs = tool_json("pairs-from-raw") or {}
    n_commit = 0
    merged = ROOT / "data/instructions/hf_commits.jsonl"
    merged.unlink(missing_ok=True)
    with merged.open("w", encoding="utf-8") as out:
        for name in ("sft_train.jsonl", "sft_val.jsonl"):
            p = proc / name
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.write(line + "\n")
                        n_commit += 1
    log(f"instruction pairs: {pairs.get('pairs', 0)} from documented code, {n_commit} from real commits")
    st["sft_ready"] = False
    if (pairs.get("pairs", 0) + n_commit) >= 8:
        rc, _ = mad("prepare-instructions", "--input", "data/instructions")
        st["sft_ready"] = rc == 0 and (proc / "sft.jsonl").is_file()
    else:
        (proc / "sft.jsonl").unlink(missing_ok=True)
    subprocess.run([PY, TOOLS, "notices"], capture_output=True)
    run([PY, TOOLS, "data-report", "--no-tokens"])
    st["prepared_raw_bytes"] = raw_bytes()
    st["done"]["prepare"] = True
    st.save()


def have_pretrained():
    return (CK / "mad_latest.pt").is_file() or (CK / "mad_best.pt").is_file()


def stage_tokenizer(args, st):
    tok, cfg = ROOT / "tokenizer/mad_bpe.json", ROOT / "config.json"
    if st["done"].get("tokenizer") and tok.is_file() and cfg.is_file():
        return
    if have_pretrained() and tok.is_file() and cfg.is_file():
        st["done"]["tokenizer"] = True  # keep the tokenizer that matches the existing checkpoints
        return
    preset = st["preset"]
    vocab = args.vocab_size or (1024 if preset == "tiny" else 4096)
    log(f"== tokenizer: preset {preset}, vocab {vocab} ==")
    for stale in ("data/processed/tokens.pt",):
        (ROOT / stale).unlink(missing_ok=True)
    rc, tail = mad("init", "--preset", preset, "--vocab-size", vocab, "--force")
    if rc != 0:
        raise Fatal(f"init failed:\n{tail[-600:]}")
    rc, tail = mad("train-tokenizer", "--vocab-size", vocab)
    if rc != 0:
        raise Fatal(f"train-tokenizer failed:\n{tail[-600:]}")
    st["done"]["tokenizer"] = True
    st.pop("total_steps", None)
    st.save()


def maybe_refresh(args, st):
    """Between chunks: fetch new data and/or re-prepare when due, asked for, or files were added by hand."""
    flag = ROOT / "data/REFRESH_NOW"
    now = time.time()
    st.setdefault("last_refresh", now)
    due = bool(args.refresh_hours) and now - st["last_refresh"] >= args.refresh_hours * 3600
    by_hand = raw_bytes() > st.get("prepared_raw_bytes", raw_bytes()) + 1000   # files dropped into data/raw/
    asked = flag.exists()
    if not (due or asked or by_hand):
        return False
    why = "scheduled" if due else "requested (data/REFRESH_NOW)" if asked else "new files in data/raw/"
    if (due or asked) and not has_wifi_or_net(require_wifi=not getattr(args, "allow_mobile_data", False)):
        log("refresh: off Wi-Fi; skipping network fetch and continuing with local data")
        return False
    log(f"== live data refresh ({why}); training is paused for a moment ==")
    before = raw_bytes()
    try:
        if due or asked:
            mb = args.refresh_mb
            run_fetch("github", [PY, "fetch_data.py", "github", "--mb", f"{mb * 0.5:.1f}"], attempts=1, env=NO_HF_TOKEN)
            if args.hf_data:
                run_fetch("huggingface (REST)", [PY, "fetch_data.py", "hf", "--mb", f"{mb * 0.5:.1f}"], attempts=1, env=NO_GH_TOKEN)
        grew = raw_bytes() - st.get("prepared_raw_bytes", before)
        if grew > 1000:
            st["done"].pop("prepare", None)
            stage_prepare(args, st)
            info = tool_json("count-tokens") or {}
            log(f"refresh: +{grew / 1e6:.2f} MB raw code; corpus is now {info.get('tokens', 0):,} tokens. "
                f"The tokenizer stays fixed; training continues on the larger corpus.")
        else:
            log("refresh: nothing new found; training continues")
    except Fatal as e:
        log(f"refresh failed ({str(e)[:200]}); continuing with the previous corpus")
    except Exception as e:
        log(f"refresh failed ({type(e).__name__}: {e}); continuing with the previous corpus")
    flag.unlink(missing_ok=True)
    st["last_refresh"] = time.time()
    st.save()
    return True


def best_resume(latest, prev, best):
    """First readable checkpoint among latest, previous-good, best. Returns (path, step) or (None, 0)."""
    for p in (latest, prev, best):
        info = ckpt_info(p)
        if info:
            return str(p), info["step"]
    return None, 0


def train_loop(stage, args, st, hfs, pusher, healer):
    """Chunked training with checkpoints, HF commits, early stopping and self-healing."""
    sft = stage == "sft"
    prefix = "mad_sft" if sft else "mad"
    latest, prev, best = CK / f"{prefix}_latest.pt", CK / f"{prefix}_latest.prev.pt", CK / f"{prefix}_best.pt"
    log_file = CK / ("mad_sft.jsonl" if sft else "training.jsonl")
    total_key, step_key = ("sft_total", "sft_step") if sft else ("total_steps", "pretrain_step")
    hp = st["hp"].setdefault(stage, {})
    chunk_min = args.chunk_minutes
    stalled = 0
    deadline = st.get("deadline")
    while True:
        if STOP.is_set():
            return "stopped"
        wait_if_held(args, st, f"{stage}: ")
        if STOP.is_set():
            return "stopped"
        total = st[total_key]
        resume, step = best_resume(latest, prev, best)
        st[step_key] = step
        if step >= total:
            return "finished"
        trend = tool_json("val-trend", "--log", log_file, "--patience", args.patience if not sft else 8)
        if trend and trend.get("stop"):
            log(f"{stage}: validation loss has not improved for {trend['evals_since_best']} evaluations "
                f"(best {trend['best_val']} at step {trend['best_step']}); stopping early to avoid memorizing")
            return "early_stop"
        if deadline and time.time() > deadline and not sft:
            log(f"{stage}: time budget reached")
            return "deadline"
        cmd = ["finetune" if sft else "train", "--steps", total, "--batch-size", hp.get("batch_size", args.batch_size),
               "--grad-accum", hp.get("grad_accum", args.grad_accum if not sft else max(1, args.grad_accum // 2)),
               "--learning-rate", f"{hp.get('lr', args.lr if not sft else args.lr * 0.6):.3e}",
               "--checkpoint-every", 25, "--eval-every", 50 if not sft else 25, "--max-minutes", chunk_min]
        threads = hp.get("threads", args.threads)
        if threads:
            cmd += ["--threads", threads]
        if resume:
            cmd += ["--resume", resume]
        log(f"== {stage}: step {step}/{total}, running a {chunk_min:g}-minute chunk ==")
        t0 = time.time()
        rc, tail = mad(*cmd)
        elapsed = time.time() - t0
        st["train_seconds"] = st.get("train_seconds", 0) + elapsed
        if rc != 0:
            if STOP.is_set():
                return "stopped"
            kind = classify(tail, rc)
            log(f"{stage}: chunk failed (exit {rc}, kind={kind})")
            try:
                healer.handle(kind, tail, stage)
            except Fatal:
                pusher.submit(f"{stage}: fatal error ({kind})", heavy=False, wait=True)
                raise
            continue
        info = ckpt_info(latest)
        if not info:
            stalled += 1
            if stalled >= 2:
                chunk_min *= 2
                log(f"{stage}: no checkpoint produced; doubling chunk length to {chunk_min:g} min")
            if stalled >= 4:
                raise Fatal(f"{stage}: training exits cleanly but writes no checkpoint:\n{tail[-800:]}")
            continue
        new_step = info["step"]
        if new_step <= step:
            stalled += 1
            if stalled >= 2:
                chunk_min *= 2
                log(f"{stage}: no progress in a chunk; doubling chunk length to {chunk_min:g} min")
            if stalled >= 5:
                raise Fatal(f"{stage}: no training progress after several chunks:\n{tail[-800:]}")
            continue
        stalled = 0
        healer.success()
        skipped, done_steps = LAST_RUN["skipped"], new_step - step
        if done_steps >= 10 and skipped / done_steps > 0.05:
            old = hp.get("lr", args.lr if not sft else args.lr * 0.6)
            hp["lr"] = max(1e-5, old * 0.7)
            if hp["lr"] < old:
                healer.record("unstable_grads", f"{skipped} of {done_steps} steps had infinite gradients and were "
                                                f"skipped; learning rate {old:.2e} -> {hp['lr']:.2e}")
        shutil.copy2(latest, prev)
        st[step_key] = new_step
        # fit the schedule to the time budget once, from the measured speed
        if not sft and args.hours and not st.get("steps_fitted") and new_step > step and not args.steps:
            sec_per_step = elapsed / (new_step - step)
            fit = int(0.85 * args.hours * 3600 / sec_per_step)
            if fit < st[total_key]:
                st[total_key] = max(new_step + 50, fit)
                log(f"{stage}: {sec_per_step:.1f}s/step -> fitting the plan to --hours {args.hours:g}: {st[total_key]} total steps")
            st["steps_fitted"] = True
        st.save()
        val = (tool_json("val-trend", "--log", log_file, "--patience", 999) or {}).get("best_val")
        msg = f"{stage} step {new_step}/{st[total_key]}" + (f", best val loss {val}" if val else "")
        log(msg)
        pusher.submit(msg, heavy=True, card=model_card(st, args))
        if new_step >= st[total_key]:
            return "finished"
        if not sft and not STOP.is_set():
            maybe_refresh(args, st)


def stage_pretrain(args, st, hfs, pusher, healer):
    if st["done"].get("pretrain") and have_pretrained():
        return
    if "total_steps" not in st or args.steps:
        if args.steps:
            st["total_steps"] = args.steps
            st["steps_fitted"] = True
        else:
            info = tool_json("count-tokens") or {}
            n = info.get("tokens", 0)
            hp = st["hp"].get("pretrain", {})
            try:
                ctx = json.loads((ROOT / "config.json").read_text())["max_seq_len"]
            except (OSError, ValueError, KeyError):
                ctx = 256
            tps = hp.get("batch_size", args.batch_size) * hp.get("grad_accum", args.grad_accum) * ctx
            st["total_steps"] = max(200, min(60000, int(args.epochs * 0.95 * n / tps))) if n else 2000
            log(f"corpus: {n:,} tokens -> {st['total_steps']} steps (~{args.epochs:g} epochs)")
        if not args.steps:
            st.pop("steps_fitted", None)
        st.save()
    st.pop("deadline", None)
    if args.hours:
        # count time already trained, so restarts/resumes do not reset the --hours budget
        st["deadline"] = time.time() + max(0.0, 0.85 * args.hours * 3600 - st.get("train_seconds", 0.0))
    result = train_loop("pretrain", args, st, hfs, pusher, healer)
    if result in ("finished", "early_stop"):
        st["done"]["pretrain"] = True
    log(f"pretraining: {result} at step {st.get('pretrain_step')}")
    st.save()


def stage_sft(args, st, hfs, pusher, healer):
    sft_file = ROOT / "data/processed/sft.jsonl"
    if st["done"].get("sft") or args.no_sft:
        return
    if not st.get("sft_ready") or not sft_file.is_file() or not have_pretrained():
        log("fine-tune: skipped (no instruction pairs or no pretrained model yet)")
        return
    n = sum(1 for _ in sft_file.open(encoding="utf-8"))
    if n < 8:
        log(f"fine-tune: skipped (only {n} pairs)")
        return
    if "sft_total" not in st:
        st["sft_total"] = args.sft_steps or max(200, min(3000, int(3 * n / 4)))
        log(f"fine-tune: {n} pairs -> {st['sft_total']} steps")
    result = train_loop("sft", args, st, hfs, pusher, healer)
    if result in ("finished", "early_stop"):
        st["done"]["sft"] = True
    st.save()


SMOKE_PROMPTS = [
    "In Kotlin: Show a toast with a message",
    "In Kotlin: RecyclerView adapter that shows a list of strings",
    "In Kotlin: Check whether the device has an internet connection",
]


def stage_finish(args, st, hfs, pusher):
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    out = []
    if have_pretrained():
        rc, tail = mad("eval", echo=False)
        m = re.search(r"held-out loss ([\d.]+)", tail)
        if m:
            st.setdefault("eval", {})["held_out_loss"] = float(m.group(1))
        out.append("## Eval\n```\n" + tail.strip() + "\n```")
        if (CK / "mad_sft_best.pt").is_file() or (CK / "mad_sft_latest.pt").is_file():
            out.append("## Sample answers (python mad.py ask)")
            for prompt in SMOKE_PROMPTS:
                rc, tail = mad("ask", prompt, echo=False)
                out.append(f"### {prompt}\n```\n{tail.strip()[:1500]}\n```")
    (rep / "latest_report.md").write_text("# MI report\n\n" + "\n\n".join(out) + "\n", encoding="utf-8")
    st["done"]["finished"] = bool(st["done"].get("pretrain"))
    st.save()
    pusher.submit("final: weights, logs, report, license notices", heavy=True, card=model_card(st, args), wait=True)
    pusher.drain()


# ----------------------------------------------------------------------------- main
def fresh_start():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = ROOT / f"backup_{stamp}"
    moved = []
    for rel in ("config.json", "tokenizer", "checkpoints", "data/processed", "state", "reports", "logs"):
        p = ROOT / rel
        if p.exists():
            dest.mkdir(exist_ok=True)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest / rel))
            moved.append(rel)
    (ROOT / "tokenizer").mkdir(exist_ok=True)
    log(f"--fresh: moved {', '.join(moved) or 'nothing'} to {dest.name}/ (raw data kept)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MI: fetch data, train and back up your Android coder.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("hugging face backup")
    g.add_argument("--no-hf", action="store_true", help="no cloud backup (local only)")
    g.add_argument("--hf-repo", default=None, help="owner/name (default: <you>/MI-android-coder)")
    g.add_argument("--public", action="store_true", help="make the HF repo public (default private)")
    g = ap.add_argument_group("data")
    g.add_argument("--mb", type=float, default=40.0, help="MB of code to fetch in total (GitHub 50%%, HF 50%%)")
    g.add_argument("--skip-data", action="store_true", help="do not fetch; use what is in data/raw")
    g.add_argument("--refresh-data", action="store_true", help="fetch more data even if already fetched")
    g.add_argument("--no-hf-data", dest="hf_data", action="store_false", help="GitHub only; skip all Hugging Face datasets")
    g.add_argument("--no-hf-cli", dest="hf_cli", action="store_false", help="skip the hf-CLI commit dataset")
    g.add_argument("--no-hf-java", dest="hf_java", action="store_false", help="skip the 58 MB Java commit file")
    g.add_argument("--refresh-hours", type=float, default=6.0,
                   help="while training, fetch more data and rebuild the corpus this often (0 = never)")
    g.add_argument("--refresh-mb", type=float, default=10.0, help="MB of new code per refresh")
    g.add_argument("--linux-share", type=float, default=0.2,
                   help="fraction of --mb fetched as general Linux code in all languages (0 = Android only)")
    g.add_argument("--pause-without-wifi", action="store_true", help="pause training while off Wi-Fi")
    g.add_argument("--allow-mobile-data", action="store_true", help="allow data refreshes over mobile data")
    g.add_argument("--network-wait-min", type=float, default=90, help="how long to wait for a missing network")
    g = ap.add_argument_group("model and training")
    g.add_argument("--preset", default="auto", choices=["auto", "tiny", "small", "medium", "large"])
    g.add_argument("--vocab-size", type=int, default=None, help="default 1024 (tiny) or 4096")
    g.add_argument("--steps", type=int, default=0, help="pretraining steps (default: from corpus size and --epochs)")
    g.add_argument("--epochs", type=float, default=3.0,
                   help="target passes over the corpus (small corpora memorize quickly; add data instead of epochs)")
    g.add_argument("--sft-steps", type=int, default=0, help="fine-tune steps (default from pair count)")
    g.add_argument("--no-sft", action="store_true", help="skip the English-request fine-tune stage")
    g.add_argument("--hours", type=float, default=0, help="time budget for the plan (0 = no limit)")
    g.add_argument("--chunk-minutes", type=float, default=15, help="minutes between saves/HF commits")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--grad-accum", type=int, default=4)
    g.add_argument("--threads", type=int, default=None)
    g.add_argument("--patience", type=int, default=12, help="evals without improvement before early stop")
    g = ap.add_argument_group("control")
    g.add_argument("--fresh", action="store_true", help="start over (moves old model files to backup_*/)")
    g.add_argument("--status", action="store_true", help="print progress and exit")
    return ap.parse_args(argv)


def print_status(st):
    print(json.dumps({k: st.get(k) for k in ("version", "preset", "done", "total_steps", "pretrain_step", "sft_total",
                                              "sft_step", "hp", "eval", "hf_repo")}, indent=1))
    if st.get("hf_repo"):
        print(f"cloud backup: https://huggingface.co/{st['hf_repo']}  (also in HF_BACKUP_LINK.txt)")
    for h in st.get("heals", [])[-5:]:
        print("heal:", h["time"], h["kind"], "-", h["action"])


def pipeline(args, st, hfs, pusher, healer):
    stage_preflight(args, st)
    stage_restore(args, st, hfs)
    stage_data(args, st)
    if STOP.is_set():
        return
    stage_prepare(args, st)
    stage_tokenizer(args, st)
    stage_pretrain(args, st, hfs, pusher, healer)
    if STOP.is_set():
        return
    stage_sft(args, st, hfs, pusher, healer)


def main(argv=None):
    args = parse_args(argv)
    for s in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(s, on_signal)
    if args.fresh:
        fresh_start()
    st = State()
    if args.status:
        print_status(st)
        return 0
    if args.steps and st.get("total_steps") != args.steps:
        st["done"].pop("pretrain", None)
    hfs = hf_sync.HFSync(repo=args.hf_repo, private=not args.public, enabled=not args.no_hf, log=log)
    if hfs.enabled and hfs.ensure():
        st["hf_repo"] = hfs.repo
    pusher = Pusher(hfs)
    healer = Healer(st, args, hfs)
    try:
        for attempt in range(3):
            try:
                pipeline(args, st, hfs, pusher, healer)
                break
            except Redo:
                log(f"re-running earlier stages (pass {attempt + 2}/3)")
        if not STOP.is_set():
            stage_finish(args, st, hfs, pusher)
    except Fatal as e:
        log(f"STOPPED: needs your attention:\n{e}")
        st.save()
        pusher.drain()
        return 2
    except Exception:
        tb = traceback.format_exc()
        log("unexpected error in the pipeline (it will be restarted and resumes from the last checkpoint):\n" + tb)
        st.save()
        pusher.submit("crash report", heavy=False, wait=True)
        return 1
    st.save()
    if STOP.is_set():
        log("stopped by you; progress is saved. Run ./start.sh again to continue.")
        pusher.drain()
        return 130
    if st["done"].get("finished"):
        log(f"DONE. Ask your model:  python mad.py ask \"In Kotlin: ...\"   | report: reports/latest_report.md"
            + (f" | backup: https://huggingface.co/{hfs.repo}" if hfs.ready else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
