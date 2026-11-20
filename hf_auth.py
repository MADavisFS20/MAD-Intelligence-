"""Find your Hugging Face token the way the `hf` CLI does (stdlib only). The token is never printed.

Order: HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_HUB_TOKEN environment variables, then the login saved
by `hf auth login` (asked from `hf auth token`, or read from the token file).
"""
import os
import shutil
import subprocess
from pathlib import Path

ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_cache = {}


def _token_files():
    home = Path(os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface")
    return [home / "token", Path.home() / ".huggingface" / "token"]


def get_token():
    """Return the token string or None."""
    if "t" in _cache:
        return _cache["t"]
    token = next((os.environ[n].strip() for n in ENV_NAMES if os.environ.get(n, "").strip()), None)
    if not token:
        for f in _token_files():
            try:
                if f.is_file() and f.read_text().strip():
                    token = f.read_text().strip()
                    break
            except OSError:
                pass
    if not token and shutil.which("hf"):
        try:
            r = subprocess.run(["hf", "auth", "token"], capture_output=True, text=True, timeout=30)
            out = r.stdout.strip().splitlines()
            if r.returncode == 0 and out and out[-1].startswith("hf_"):
                token = out[-1].strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    _cache["t"] = token
    return token


def dataset_url(dataset):
    return f"https://huggingface.co/datasets/{dataset}"


def gated_message(dataset):
    """Explain a gated-dataset refusal precisely, including the link where the terms are accepted."""
    url = dataset_url(dataset)
    if get_token():
        return (f"access denied: '{dataset}' is gated. Your Hugging Face login was found, but this account has not "
                f"accepted the dataset's terms yet.\n  Open {url} in a browser, signed in to the SAME account as "
                f"your token, and click the accept/agree button (the hf CLI cannot do this step). Then re-run. "
                f"If it says 'pending', the owner must approve you first. This dataset is optional.")
    return (f"access denied: '{dataset}' is gated and no Hugging Face login was found. Run `hf auth login` "
            f"(or export HF_TOKEN=hf_...), then accept the terms at {url} in a browser. This dataset is optional.")
