#!/usr/bin/env python3
"""MI data fetcher: permissively licensed training data into data/raw/ (stdlib only).

One script replacing fetch_github_data.py, fetch_hf_data.py, fetch_hf_cli.py, fetch_teams.py,
fetch_termux_android_fs_data.py and fetch_termux.py.

Subcommands:
  github     Android source from public GitHub repos (curated seeds + search). Was fetch_github_data.py
  hf         Code from Hugging Face via the Datasets Server REST API.          Was fetch_hf_data.py
  hf-files   JSONL datasets via the `hf` CLI (commitpackft) + dataset discovery. Was fetch_hf_cli.py
  teams      Blue/Red team security datasets and repos.                         Was fetch_teams.py
  termux     Termux CLI / Android filesystem data (built-in + HF + GitHub).     Was fetch_termux*.py
  linux      Open-source Linux code in all common languages (GitHub search).
  all        Priority order: Android (github, hf-files, hf) -> Python/Linux (linux) -> termux -> teams (last)

Every output record uses the same schema:
  {source_path, extension, instruction, input, output, license, source}

Legal guard rails (enforced in code, identical for every subcommand):
  * A repo or dataset is used only if its license is on the permissive allowlist. The license comes
    from the API; for GitHub, when the API gives no answer, the repo's LICENSE file must match exactly
    one permissive family. Unknown license = skipped (never assumed, never defaulted to "mit").
  * Forks are skipped. Only public archives/endpoints are used. Gated datasets need your own HF_TOKEN.
  * Every record keeps its repo/dataset, commit and license; LICENSE texts go to data/licenses/.

Examples:
  python3 fetch_data.py github --mb 50
  python3 fetch_data.py github --seeds-only --max-repos 5
  python3 fetch_data.py hf --mb 5 --only kstack-kotlin
  python3 fetch_data.py hf-files --only commitpackft-kotlin
  python3 fetch_data.py hf-files --discover "android kotlin"
  python3 fetch_data.py teams --mb 20
  python3 fetch_data.py teams --team blue --per-item-mb 20 --mb 0   # --mb 0 = no total cap
  python3 fetch_data.py linux --mb 100 --langs C Rust Go
  python3 fetch_data.py termux --discover --dry-run
  python3 fetch_data.py all

Optional env: GITHUB_TOKEN / GH_TOKEN (API rate limits), HF_TOKEN (gated data, rate limits).
Test overrides: MI_GITHUB_API, MI_GITHUB_CODELOAD, MI_GH_SEEDS (comma list), MI_SEARCH_DELAY.
Next step after fetching: python3 prepare_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterator

UA = "mi-fetch/3.0 (Termux-Android)"
API = os.environ.get("MI_GITHUB_API", "https://api.github.com").rstrip("/")
CODELOAD = os.environ.get("MI_GITHUB_CODELOAD", "https://codeload.github.com").rstrip("/")
HF_HUB = "https://huggingface.co"
HF_DS_API = "https://datasets-server.huggingface.co"

MIN_BYTES, MAX_BYTES = 200, 60_000
MAX_ZIP_BYTES = 80_000_000
MAX_LINE = 500

# Permissive licenses (lowercase SPDX ids) allowed for code.
PERMISSIVE = {
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "isc", "unlicense", "cc0-1.0", "0bsd", "zlib", "bsl-1.0", "wtfpl", "postgresql",
}
# Text/instruction datasets may additionally be attribution-only Creative Commons.
DATA_PERMISSIVE = PERMISSIVE | {"cc-by-4.0", "cc-by-3.0"}

ANDROID_RE = re.compile(r"\bandroidx?\.|@Composable|\bcom\.android\.|AndroidManifest|\bandroid\s*\{")
# Extension priority when a per-repo byte budget cuts a repo short: Android first, then Python,
# then other Linux code, then docs/config last.
ANDROID_EXTS = {".kt", ".kts", ".java", ".gradle", ".aidl", ".xml", ".pro"}
DOC_EXTS = {".md", ".txt", ".yml", ".yaml", ".json", ".yar", ".yara", ".rules", ".conf", ".toml"}


def ext_rank(ext):
    ext = ext.lower()
    return 0 if ext in ANDROID_EXTS else 1 if ext == ".py" else 3 if ext in DOC_EXTS else 2


SKIP_DIRS = {"build", ".git", ".gradle", ".idea", "node_modules", "third_party", "thirdparty", "vendor",
             "generated", "gen", "out", "dist", "captures", "intermediates", "__pycache__",
             "venv", "target", "assets", "images", "img"}

# Some datasets store GitHub-style license names; map them to the SPDX ids above.
LICENSE_ALIASES = {
    "apache license 2.0": "apache-2.0", "apache 2.0": "apache-2.0",
    "mit license": "mit", "mit no attribution": "mit-0", "the unlicense": "unlicense",
    "creative commons zero v1.0 universal": "cc0-1.0",
    'bsd 3-clause "new" or "revised" license': "bsd-3-clause",
    'bsd 2-clause "simplified" license': "bsd-2-clause",
    "bsd 3-clause clear license": "bsd-3-clause-clear",
    "bsd zero clause license": "0bsd", "isc license": "isc", "zlib license": "zlib",
    "boost software license 1.0": "bsl-1.0", "postgresql license": "postgresql",
    "do what the f*ck you want to public license": "wtfpl",
}


# ============================================================================= shared: errors, http
class HttpStatus(Exception):
    def __init__(self, code, headers=None):
        super().__init__(f"HTTP {code}")
        self.code, self.headers = code, headers


class NetworkError(Exception):
    pass


class RateLimited(Exception):
    pass


class Gated(Exception):
    pass


class Unavailable(Exception):
    pass


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """urllib copies every request header, including Authorization, to the redirect target. HF and
    GitHub redirect downloads to CDN hosts, so a token would be sent to a third party (and signed CDN
    URLs reject it). Drop credentials whenever the redirect leaves the original host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            for h in list(new.headers):
                if h.lower() in ("authorization", "cookie"):
                    del new.headers[h]
        return new


_OPENER = urllib.request.build_opener(_StripAuthOnRedirect)


def _urlopen(req, timeout=60):
    return _OPENER.open(req, timeout=timeout)


def http_get(url, headers=None, timeout=60, tries=4) -> bytes:
    h = {"User-Agent": UA, **(headers or {})}
    last = None
    for attempt in range(tries):
        try:
            with _urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                ra = (e.headers.get("Retry-After") if e.headers else None) or ""
                wait = min(90, int(ra)) if ra.isdigit() else 2 ** attempt * 2 * (3 if e.code == 429 else 1)
                time.sleep(wait)
                continue
            raise HttpStatus(e.code, e.headers) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            if attempt < tries - 1:
                time.sleep(2 ** attempt * 2)
    raise NetworkError(str(last))


def http_json(url, headers=None, **kw):
    return json.loads(http_get(url, headers, **kw).decode("utf-8", "replace"))


LAST_DL = {"reason": ""}  # why the last download_zip() returned None: "too large", "gone" or "transient"


def download_zip(url, headers=None, tries=4, cap=MAX_ZIP_BYTES):
    """Stream an archive into a temp file (size capped, never held in memory). Path or None."""
    LAST_DL["reason"] = "transient"
    for attempt in range(tries):
        tmp = tempfile.NamedTemporaryFile(prefix="mi_", suffix=".zip", delete=False)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with _urlopen(req, timeout=90) as r:
                total = 0
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > cap:
                        raise ValueError("archive larger than cap")
                    tmp.write(chunk)
            tmp.close()
            return tmp.name
        except ValueError:
            tmp.close()
            os.unlink(tmp.name)
            LAST_DL["reason"] = "too large"
            return None
        except urllib.error.HTTPError as e:
            tmp.close()
            os.unlink(tmp.name)
            if e.code in (404, 403, 451):
                LAST_DL["reason"] = "gone"
                return None
            time.sleep(2 ** attempt * 2)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            tmp.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            time.sleep(2 ** attempt * 2)
    return None


# ============================================================================= shared: auth, paths, state
def hf_token() -> str:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k, "").strip():
            return os.environ[k].strip()
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        try:
            if p.is_file():
                return p.read_text().strip()
        except OSError:
            pass
    return ""


def gh_token() -> str:
    return (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()


def hf_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def gated_message(ds):
    return (f"{ds} is gated: create a free Hugging Face account, accept the terms at "
            f"{HF_HUB}/datasets/{ds}, then `export HF_TOKEN=hf_...` and re-run.")


def default_base() -> Path:
    if any(Path(p).exists() for p in ("data/raw", "mi_run.py", "prepare_data.py")):
        return Path(".")
    home_mi = Path.home() / "MI"
    return home_mi if home_mi.exists() else Path(".")


def paths(args, state_name):
    data = Path(args.data_dir) if args.data_dir else default_base() / "data"
    raw = Path(args.raw) if args.raw else data / "raw"
    lic = Path(args.licenses) if args.licenses else data / "licenses"
    state = Path(args.state) if args.state else data / state_name
    raw.mkdir(parents=True, exist_ok=True)
    lic.mkdir(parents=True, exist_ok=True)
    return data, raw, lic, state


def load_state(path: Path, default: dict) -> dict:
    state = json.loads(json.dumps(default))
    if path.exists():
        try:
            state.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return state


def save_state(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def known_repos(data: Path) -> set[str]:
    """Repos already handled by any subcommand (lowercase owner/repo)."""
    seen: set[str] = set()
    for fname, keys in (("gh_state.json", ("done", "skipped")), ("teams_state.json", ("gh_done",)),
                        ("termux_state.json", ("gh_done",)),
                        ("gh_linux_state.json", ("done", "skipped"))):
        try:
            st = json.loads((data / fname).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for k in keys:
            v = st.get(k, {})
            seen.update(str(x).lower() for x in (v.keys() if isinstance(v, dict) else v))
    return seen


# ============================================================================= shared: licenses, records
def licenses_of(value):
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    names = [str(x).strip().lower().removeprefix("license:") for x in items if str(x).strip()]
    return [LICENSE_ALIASES.get(n, n) for n in names]


def license_from_tags(tags) -> str:
    lics = [t.split(":", 1)[1].lower() for t in (tags or []) if isinstance(t, str) and t.startswith("license:")]
    return lics[0] if lics else ""


LICENSE_FAMILIES = {
    "apache-2.0": re.compile(r"Apache License[\s,]*Version 2\.0", re.I),
    "mit": re.compile(r"Permission is hereby granted, free of charge", re.I),
    "bsd-3-clause": re.compile(r"Redistribution and use in source and binary forms.*?Neither the name", re.I | re.S),
    "bsd-2-clause": re.compile(r"Redistribution and use in source and binary forms", re.I),
    "isc": re.compile(r"Permission to use, copy, modify, and/or distribute this software", re.I),
    "unlicense": re.compile(r"This is free and unencumbered software released into the public domain", re.I),
}
NON_PERMISSIVE_HINT = re.compile(r"(?i)GNU (Affero |Lesser |Library )?General Public License|Mozilla Public License|"
                                 r"Creative Commons (Attribution-)?(Non|Share)|Eclipse Public License|Server Side Public")


def detect_license(text):
    """One SPDX id only if exactly one permissive family matches and no copyleft text appears."""
    if not text or NON_PERMISSIVE_HINT.search(text):
        return None
    found = {k for k, rx in LICENSE_FAMILIES.items() if rx.search(text)}
    if "bsd-3-clause" in found:
        found.discard("bsd-2-clause")
    return found.pop() if len(found) == 1 else None


def mi_record(path, output, lic, source, instruction="", inp=""):
    return {"source_path": path, "extension": (os.path.splitext(path)[1].lstrip(".") or "txt").lower(),
            "instruction": instruction, "input": inp, "output": output, "license": lic, "source": source}


class Sink:
    """Writes de-duplicated MI records to one JSONL file until a byte budget is reached."""

    def __init__(self, path, budget, seen, mode="w"):
        self.path, self.budget, self.seen, self.mode = Path(path), budget, seen, mode
        self.rows = self.bytes = 0
        self._f = None

    def add(self, rec) -> bool:
        digest = hashlib.sha256(rec["output"].encode("utf-8")).hexdigest()
        if digest in self.seen:
            return False
        self.seen.add(digest)
        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open(self.mode, encoding="utf-8")
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.rows += 1
        self.bytes += len(rec["output"].encode("utf-8"))
        return True

    @property
    def full(self):
        return self.bytes >= self.budget

    def close(self):
        if self._f:
            self._f.close()


def bump(stats, key):
    stats[key] = stats.get(key, 0) + 1


def top_rejections(stats, n=6):
    return "; ".join(f"{k} x{v}" for k, v in sorted(stats.items(), key=lambda kv: -kv[1])[:n])


# ============================================================================= shared: GitHub
def gh_api(path, token, params=None, tries=4):
    url = f"{API}{path}" + (("?" + urllib.parse.urlencode(params)) if params else "")
    headers = {"User-Agent": UA, "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last = None
    for attempt in range(tries):
        try:
            with _urlopen(urllib.request.Request(url, headers=headers), timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                remaining = e.headers.get("X-RateLimit-Remaining")
                reset, retry_after = e.headers.get("X-RateLimit-Reset"), e.headers.get("Retry-After")
                wait = 0
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                elif reset and reset.isdigit():
                    wait = max(0, int(reset) - int(time.time())) + 1
                if remaining == "0" or retry_after:
                    if wait and wait <= 90 and attempt < tries - 1:
                        time.sleep(wait)
                        continue
                    raise RateLimited(f"GitHub API rate limit (resets in ~{wait}s). Set GITHUB_TOKEN to raise it.")
                raise RuntimeError(f"HTTP {e.code} for {path}")
            if e.code == 404:
                return None
            last = e
            if e.code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {e.code} for {path}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"network failure for {path}: {last}")


def search_delay(token):
    # unauthenticated search allows ~10 requests/minute
    time.sleep(float(os.environ.get("MI_SEARCH_DELAY", 1 if token else 6.5)))


def zip_text_records(zpath, full, lic, exts, budget):
    """Generic repo-archive reader: text files with the given extensions. Returns (records, license_text)."""
    recs, used = [], 0
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        lic_text = ""
        for i in infos:  # license first, so a byte budget can never skip it
            parts = i.filename.split("/")
            if len(parts) == 2 and parts[1].upper().startswith(("LICENSE", "LICENCE", "COPYING")):
                lic_text = z.read(i).decode("utf-8", "replace")
                break
        infos.sort(key=lambda i: ext_rank(os.path.splitext(i.filename)[1]))  # stable: keeps archive order within a rank
        for i in infos:
            parts = i.filename.split("/")
            if len(parts) < 2 or any(p in SKIP_DIRS or p.startswith(".") for p in parts[1:-1]):
                continue
            rel, ext = "/".join(parts[1:]), os.path.splitext(parts[-1])[1].lower()
            if ext not in exts or not MIN_BYTES <= i.file_size <= MAX_BYTES:
                continue
            try:
                text = z.read(i).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if "\x00" in text:
                continue
            recs.append(mi_record(rel, text, lic, f"github:{full} path={rel}"))
            used += i.file_size
            if used >= budget:
                break
    return recs, lic_text


def gh_repo_to_jsonl(full, meta, raw_dir, lic_dir, exts, budget, prefix):
    """Check license/fork/size from API metadata, fetch the archive, write one JSONL. (rows, bytes, note)."""
    if meta is None:
        return 0, 0, "not found"
    if meta.get("fork"):
        return 0, 0, "fork"
    lic = ((meta.get("license") or {}).get("spdx_id") or "").lower()
    if lic not in PERMISSIVE:
        return 0, 0, f"license {lic or 'unknown'}"
    if meta.get("size", 0) > 400_000:
        return 0, 0, "repo too large"
    owner, repo = full.split("/", 1)
    zpath = download_zip(f"{CODELOAD}/{owner}/{repo}/zip/HEAD")
    if not zpath:
        return 0, 0, "download failed" if LAST_DL["reason"] == "transient" else f"archive {LAST_DL['reason']}"
    try:
        recs, lic_text = zip_text_records(zpath, full, lic, exts, budget)
    except zipfile.BadZipFile:
        return 0, 0, "bad archive"
    finally:
        os.unlink(zpath)
    if not recs:
        return 0, 0, "no usable files"
    safe = f"{owner}__{repo}"
    nbytes = 0
    with (raw_dir / f"{prefix}{safe}.jsonl").open("w", encoding="utf-8") as out:
        for r in recs:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            nbytes += len(r["output"].encode("utf-8"))
    if lic_text:
        (lic_dir / f"{safe}.txt").write_text(lic_text, encoding="utf-8")
    return len(recs), nbytes, "ok"


# ============================================================================= subcommand: github
SEED_REPOS = [
    "android/sunflower", "android/architecture-samples", "android/compose-samples",
    "android/nowinandroid", "android/architecture-components-samples", "android/views-widgets-samples",
    "android/user-interface-samples", "android/connectivity-samples", "android/camera-samples",
    "android/location-samples", "android/storage-samples", "android/testing-samples",
    "android/platform-samples", "android/codelab-android-compose", "android/media-samples",
    "android/android-test", "android/input-samples", "android/wear-os-samples",
    "square/okhttp", "square/retrofit", "square/moshi", "square/leakcanary", "square/picasso",
    "coil-kt/coil", "airbnb/lottie-android", "google/accompanist", "InsertKoinIO/koin",
    "material-components/material-components-android", "Kotlin/kotlinx.coroutines",
    "Kotlin/kotlinx.serialization", "cashapp/turbine", "skydoves/Pokedex", "skydoves/landscapist",
    "chrisbanes/tivi", "Kotlin/anko", "ReactiveX/RxAndroid", "greenrobot/EventBus",
    "JakeWharton/timber", "JakeWharton/butterknife", "bumptech/glide", "google/flexbox-layout",
    "android/android-ktx", "google/ExoPlayer", "androidx/media",
]
if os.environ.get("MI_GH_SEEDS") is not None:  # test/override hook
    SEED_REPOS = [r for r in os.environ["MI_GH_SEEDS"].split(",") if r]
SEARCH_LICENSES = ["apache-2.0", "mit", "bsd-3-clause"]
SEARCH_LANGS = ["Kotlin", "Java"]
CODE_EXT = {".kt", ".java", ".kts", ".gradle", ".aidl"}
GENERATED = re.compile(r"(?i)(auto-?generated|generated by|do not edit|this file is generated|@generated)")
SKIP_NAMES = {"R.java", "BuildConfig.java", "BuildConfig.kt", "Manifest.java"}
THEME_DIR = re.compile(r"^values(-night)?(-v\d+)?$")


def wanted_path(rel):
    """Android files worth learning from: code, Gradle config, manifest and useful resources."""
    parts = rel.split("/")
    if any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
        return None
    name = parts[-1]
    ext = os.path.splitext(name)[1].lower()
    if name in SKIP_NAMES:
        return None
    if ext in CODE_EXT or name == "libs.versions.toml":
        return ext
    if ext == ".pro" and name.startswith("proguard"):
        return ext
    if ext == ".xml":
        if name == "AndroidManifest.xml":
            return ext
        if "res" in parts:
            i = parts.index("res") + 1
            res_dir = parts[i] if i < len(parts) - 1 else ""
            if res_dir == "layout" or res_dir.startswith("layout-") or res_dir in ("menu", "navigation", "xml", "values"):
                return ext
            if THEME_DIR.match(res_dir) and name in ("themes.xml", "styles.xml", "colors.xml", "dimens.xml"):
                return ext
    return None


def good_text(text):
    lines = text.splitlines()
    return len(lines) >= 8 and not GENERATED.search(text[:800]) and max(len(l) for l in lines) <= MAX_LINE


def priority(rel):
    low = rel.lower()
    return (1 if ("/test/" in low or "androidtest" in low or "/sample" in low) else 0, rel)


def extract_android_repo(zpath, per_repo_bytes):
    """Return (top_dir, license_text, [(rel, ext, text)]) with main sources first, up to a byte cap."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            return "", "", []
        top = names[0].split("/", 1)[0]
        lic_text = ""
        for cand in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING", "LICENSE-APACHE"):
            if f"{top}/{cand}" in names:
                lic_text = z.read(f"{top}/{cand}").decode("utf-8", "replace")
                break
        members = []
        for n in names:
            rel = n.split("/", 1)[1] if "/" in n else n
            ext = wanted_path(rel)
            if ext:
                members.append((priority(rel), n, rel, ext))
        members.sort()
        out, used = [], 0
        for _, n, rel, ext in members:
            size = z.getinfo(n).file_size
            if not MIN_BYTES <= size <= MAX_BYTES:
                continue
            try:
                text = z.read(n).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if "\x00" in text or not good_text(text):
                continue
            out.append((rel, ext, text))
            used += size
            if used >= per_repo_bytes:
                break
    return top, lic_text, out


def process_gh_repo(full, meta, args, state, raw_dir, lic_dir):
    owner, repo = full.split("/", 1)
    spdx_api = ((meta or {}).get("license") or {}).get("spdx_id")
    if meta:
        if meta.get("fork"):
            return "fork"
        if spdx_api and spdx_api.lower() not in PERMISSIVE and spdx_api not in ("NOASSERTION", "NONE"):
            return f"license {spdx_api}"
        if meta.get("size", 0) > 400_000:
            return "repo too large"
    zpath = download_zip(f"{CODELOAD}/{owner}/{repo}/zip/HEAD")
    if not zpath:
        return "download failed" if LAST_DL["reason"] == "transient" else f"archive {LAST_DL['reason']}"
    try:
        top, lic_text, files = extract_android_repo(zpath, int(args.per_repo_mb * 1_000_000))
    except zipfile.BadZipFile:
        return "bad archive"
    finally:
        os.unlink(zpath)
    if spdx_api and spdx_api not in ("NOASSERTION", "NONE", "null"):
        lic = spdx_api.lower()
    else:
        lic = detect_license(lic_text)
    if not lic or lic not in PERMISSIVE:
        return f"license {lic or 'unknown'}"
    if not files:
        return "no usable files"
    sha = top.rsplit("-", 1)[-1] if "-" in top else top
    safe, written = f"{owner}__{repo}", 0
    with (raw_dir / f"gh_{safe}.jsonl").open("w", encoding="utf-8") as out:
        for rel, ext, text in files:
            out.write(json.dumps(mi_record(rel, text, lic, f"github:{owner}/{repo}@{sha} path={rel}"),
                                 ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
    if lic_text:
        (lic_dir / f"{safe}.txt").write_text(lic_text, encoding="utf-8")
    state["done"][full] = {"license": lic, "files": len(files), "bytes": written, "ts": int(time.time())}
    return f"ok {len(files)} files, {written / 1e6:.2f} MB, {lic}"


def gh_candidates(args, state, token):
    done, skipped = state["done"], state.setdefault("skipped", {})
    for full in SEED_REPOS:
        if full in done or full in skipped:
            continue
        try:
            meta = gh_api(f"/repos/{full}", token)
            if meta is None:
                skipped[full] = "not found"
                continue
        except (RateLimited, RuntimeError):
            meta = None  # fall back to LICENSE-file detection
        yield full, meta
    if args.seeds_only:
        return
    pages = state.setdefault("search_page", {})
    for n in range(args.searches):
        lic = SEARCH_LICENSES[n % len(SEARCH_LICENSES)]
        lang = SEARCH_LANGS[(n // len(SEARCH_LICENSES)) % len(SEARCH_LANGS)]
        key, page = f"{lang}/{lic}", pages.get(f"{lang}/{lic}", 0) + 1
        q = f"language:{lang} topic:android license:{lic} stars:>100 fork:false"
        try:
            res = gh_api("/search/repositories", token,
                         {"q": q, "sort": "stars", "order": "desc", "per_page": 30, "page": page})
        except RateLimited as e:
            print(f"  search stopped: {e}")
            return
        except RuntimeError as e:
            print(f"  search failed: {e}")
            continue
        pages[key] = page
        for item in (res or {}).get("items", []):
            full = item.get("full_name")
            if full and full not in done and full not in skipped:
                yield full, item
        search_delay(token)


def cmd_github(args):
    data, raw_dir, lic_dir, state_path = paths(args, "gh_state.json")
    token = gh_token()
    state = load_state(state_path, {"done": {}, "skipped": {}, "search_page": {}})
    budget, got, saved, seen_now = int(args.mb * 1_000_000), 0, 0, set()
    try:
        for full, meta in gh_candidates(args, state, token):
            if full in seen_now:
                continue
            seen_now.add(full)
            if got >= budget or saved >= args.max_repos:
                break
            try:
                result = process_gh_repo(full, meta, args, state, raw_dir, lic_dir)
            except Exception as e:  # one bad repo must never stop the run
                result = f"error {type(e).__name__}: {e}"
            if result.startswith("ok"):
                saved += 1
                got += state["done"][full]["bytes"]
            elif not result.startswith(("download failed", "error")):  # transient failures retry next run
                state["skipped"][full] = result
            print(f"  {full}: {result}", flush=True)
            save_state(state_path, state)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("  interrupted")
    finally:
        save_state(state_path, state)
    print(f"github: saved {saved} repos, {got / 1e6:.2f} MB new code "
          f"({len(state['done'])} repos total, {len(state['skipped'])} skipped)")
    return 0


# ============================================================================= subcommand: hf (Datasets Server)
HF_PAGE = 100


SMOL_DIRS = {"C++": ["c++", "cpp"], "C": ["c"], "Shell": ["shell"], "Markdown": ["markdown"]}


def smol(name, ext, lang, share, android):
    return dict(name=name, dataset="bigcode/the-stack-smol", exts={ext}, share=share, android=android,
                file_dirs=SMOL_DIRS.get(lang, [lang.lower()]),
                where=[f"\"ext\"='{ext.lstrip('.')}'", f"\"lang\"='{lang}'", f"\"lang\"='{lang.lower()}'"],
                fields=dict(content="content", path="path", license="licenses", repo=("repository_name",)))


HF_SOURCES = [
    # Android first
    dict(name="kstack-kotlin", dataset="JetBrains/KStack", exts={".kt", ".kts"}, share=0.30, android=True,
         where=[], fields=dict(content="content", path="path", license="license", repo=("owner", "name"))),
    smol("smol-java", ".java", "Java", 0.14, True),
    # then Python
    smol("smol-python", ".py", "Python", 0.12, False),
    # then Linux / everything else
    smol("smol-shell", ".sh", "Shell", 0.08, False),
    smol("smol-c", ".c", "C", 0.07, False),
    smol("smol-cpp", ".cpp", "C++", 0.07, False),
    smol("smol-go", ".go", "Go", 0.05, False),
    smol("smol-rust", ".rs", "Rust", 0.05, False),
    smol("smol-javascript", ".js", "JavaScript", 0.03, False),
    smol("smol-typescript", ".ts", "TypeScript", 0.02, False),
    smol("smol-perl", ".pl", "Perl", 0.02, False),
    smol("smol-ruby", ".rb", "Ruby", 0.02, False),
    smol("smol-lua", ".lua", "Lua", 0.01, False),
    smol("smol-php", ".php", "PHP", 0.01, False),
    smol("smol-markdown", ".md", "Markdown", 0.01, False),
]


def ds_api(endpoint, params, token):
    """Datasets Server GET with Gated/Unavailable translation."""
    try:
        return http_json(f"{HF_DS_API}/{endpoint}?{urllib.parse.urlencode(params)}", hf_headers(token), tries=6)
    except HttpStatus as e:
        if e.code in (401, 403):
            raise Gated(f"HTTP {e.code}")
        if e.code in (400, 404, 422):
            raise Unavailable(f"HTTP {e.code}")
        raise
    except NetworkError as e:
        raise RuntimeError(f"network failure: {e}")


def make_hf_record(src, row, truncated, seen, stats):
    f = src["fields"]
    content, path = row.get(f["content"]), row.get(f["path"]) or ""
    if not isinstance(content, str) or f["content"] in truncated:
        bump(stats, "no content")
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in src["exts"]:
        bump(stats, "other extension")
        return None
    size = len(content.encode("utf-8"))
    if not MIN_BYTES <= size <= MAX_BYTES:
        bump(stats, "size out of range")
        return None
    lics = licenses_of(row.get(f["license"]))
    if not lics or any(l not in PERMISSIVE for l in lics):
        bump(stats, "license: " + (",".join(lics) or "none"))
        return None
    if src["android"] and not ANDROID_RE.search(content):
        bump(stats, "not android-related")
        return None
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest in seen:
        bump(stats, "duplicate")
        return None
    seen.add(digest)
    repo = "/".join(str(row.get(k) or "") for k in f["repo"]).strip("/")
    return mi_record(path, content, ",".join(lics), f"hf:{src['dataset']} repo={repo} path={path}"), size


def hf_tree(ds, path, token):
    url = f"{HF_HUB}/api/datasets/{ds}/tree/main/{urllib.parse.quote(path)}"
    try:
        data = http_json(url, hf_headers(token))
    except HttpStatus as e:
        if e.code in (401, 403):
            raise Gated(f"HTTP {e.code}")
        raise Unavailable(f"tree HTTP {e.code}")
    except NetworkError as e:
        raise Unavailable(str(e))
    return data if isinstance(data, list) else []


def hf_collect_file(src, args, token, state, seen, raw_dir):
    """Stream the first rows of a dataset's per-language JSON-lines file (no random slice scanning, no 429s)."""
    ds = src["dataset"]
    dirs = {e["path"].rsplit("/", 1)[-1].lower(): e["path"] for e in hf_tree(ds, "data", token) if e.get("type") == "directory"}
    folder = next((dirs[c] for c in src["file_dirs"] if c in dirs), None)
    if not folder:
        raise Unavailable("no language folder in repo")
    files = [e["path"] for e in hf_tree(ds, folder, token) if e.get("type") == "file" and e["path"].endswith((".json", ".jsonl"))]
    if not files:
        raise Unavailable("no JSON-lines file in folder")
    path, budget = files[0], int(args.mb * 1_000_000 * src["share"])
    url = f"{HF_HUB}/datasets/{ds}/resolve/main/{urllib.parse.quote(path)}"
    out_path, stats, written, nfiles, scanned = raw_dir / f"hf_{src['name']}.jsonl", {}, 0, 0, 0
    print(f"  streaming {path} (budget {budget / 1e6:.2f} MB)")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, **hf_headers(token)})
        with _urlopen(req, timeout=120) as r, out_path.open("w", encoding="utf-8") as out:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip().rstrip(",")
                if not line or line in ("[", "]"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bump(stats, "bad json")
                    continue
                if not isinstance(row, dict):
                    raise Unavailable("file is not JSON lines")
                scanned += 1
                made = make_hf_record(src, row, [], seen, stats)
                if made:
                    rec, size = made
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += size
                    nfiles += 1
                if written >= budget or scanned >= args.max_scan:
                    break
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise Gated(f"HTTP {e.code}")
        raise Unavailable(f"HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Unavailable(f"network: {e}")
    finally:
        if nfiles == 0 and out_path.exists():
            out_path.unlink()
    if nfiles:
        state[f"{src['name']}#file"] = True
    print(f"  {src['name']}: wrote {nfiles} files, {written / 1e6:.2f} MB (scanned {scanned} rows)")
    if stats:
        print("  rejected: " + top_rejections(stats))


def hf_collect(src, args, token, state, seen, raw_dir):
    ds = src["dataset"]
    splits = ds_api("splits", {"dataset": ds}, token).get("splits", [])
    train = [s for s in splits if s.get("split") == "train"] or splits
    if not train:
        raise Unavailable("no splits")
    # the-stack-smol has one config per language; prefer the one matching this source
    lang_hint = next(iter(src["exts"])).lstrip(".")
    cfg = (next((s for s in train if s["config"] == "default"), None) if not src["where"] else None) \
        or next((s for s in train if lang_hint in s["config"].lower() or src["name"].split("-")[-1] in s["config"].lower()), None) \
        or next((s for s in train if s["config"] == "default"), train[0])
    base = {"dataset": ds, "config": cfg["config"], "split": cfg["split"]}

    endpoint, extra, total = "rows", {}, 0
    for where in src["where"]:
        try:
            probe = ds_api("filter", {**base, "where": where, "offset": 0, "length": 1}, token)
        except Gated:
            raise
        except Exception:
            continue
        if probe.get("num_rows_total", 0) > 0:
            endpoint, extra, total = "filter", {"where": where}, probe["num_rows_total"]
            print(f"  filter: {where} -> {total} rows")
            break
    if endpoint == "rows":
        total = ds_api("rows", {**base, "offset": 0, "length": 1}, token).get("num_rows_total", 0)
        if src["where"]:
            print("  no working filter; scanning random slices (low yield, may be slow)")
    if total <= 0:
        raise Unavailable("no rows")

    budget = int(args.mb * 1_000_000 * src["share"])
    used = set(state.setdefault(src["name"], []))
    offsets = [o for o in range(0, total, HF_PAGE) if o not in used]
    random.Random(args.seed + len(used)).shuffle(offsets)

    out_path = raw_dir / f"hf_{src['name']}.jsonl"
    written = files = 0
    stats: dict = {}
    with out_path.open("a", encoding="utf-8") as out:  # append: each run adds new slices
        for n, off in enumerate(offsets[: args.max_requests], 1):
            data = ds_api(endpoint, {**base, **extra, "offset": off, "length": HF_PAGE}, token)
            for item in data.get("rows", []):
                made = make_hf_record(src, item.get("row", {}), item.get("truncated_cells", []), seen, stats)
                if made:
                    rec, size = made
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += size
                    files += 1
            used.add(off)
            state[src["name"]] = sorted(used)
            if n % 5 == 0:
                print(f"  {src['name']}: {files} files, {written / 1e6:.2f}/{budget / 1e6:.2f} MB ({n} requests)")
                if files == 0 and n >= 10:
                    print(f"    all rows rejected so far: {top_rejections(stats, 3)}")
            if written >= budget:
                break
            time.sleep(1.0)  # the Datasets Server rate-limits (HTTP 429) fast scanners
    if out_path.exists() and out_path.stat().st_size == 0:
        out_path.unlink()
    print(f"  {src['name']}: wrote {files} files, {written / 1e6:.2f} MB")
    if stats:
        print("  rejected: " + top_rejections(stats))


def cmd_hf(args):
    token = hf_token()
    print("Hugging Face login: " + ("found" if token else "none (public datasets only)"))
    if args.list_configs:
        print(json.dumps(ds_api("splits", {"dataset": args.list_configs}, token), indent=2)[:4000])
        return 0
    data, raw_dir, _, state_path = paths(args, "hf_state.json")
    state, seen, refused = load_state(state_path, {}), set(), set()
    for src in HF_SOURCES:
        if args.only and src["name"] not in args.only:
            continue
        print(f"[{src['name']}] {src['dataset']}")
        if src["dataset"] in refused:
            print("  skipped (gated, see above)")
            continue
        try:
            if src.get("file_dirs"):
                if state.get(f"{src['name']}#file"):
                    print("  already collected (remove its '#file' key from hf_state.json to refetch)")
                    continue
                try:
                    hf_collect_file(src, args, token, state, seen, raw_dir)
                except Unavailable as e:
                    print(f"  file route unavailable ({e}); falling back to slice scanning")
                    hf_collect(src, args, token, state, seen, raw_dir)
            else:
                hf_collect(src, args, token, state, seen, raw_dir)
        except Gated:
            refused.add(src["dataset"])
            print("  " + gated_message(src["dataset"]))
        except Unavailable as e:
            print(f"  not available through the Datasets Server ({e}); skipping")
        except KeyboardInterrupt:
            print("  interrupted")
            break
        except Exception as e:
            print(f"  error: {e}; skipping")
        finally:
            save_state(state_path, state)
    print("done. Next: python3 prepare_data.py")
    return 0


# ============================================================================= subcommand: hf-files
FILE_SOURCES = [
    dict(name="commitpackft-kotlin", repo="bigcode/commitpackft", include="data/kotlin/data.jsonl",
         exts={".kt", ".kts"}, android=True, default=True),
    dict(name="commitpackft-java", repo="bigcode/commitpackft", include="data/java/data.jsonl",
         exts={".java"}, android=True, default=False),  # 58 MB download: opt in with --only
]
MIN_INSTRUCTION_WORDS, MAX_INPUT_BYTES = 3, 6_000


def run_hf_cli(*a, capture=False):
    exe = shutil.which("hf")
    if not exe:
        return None
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}  # hf_xet has no Android/Termux build
    return subprocess.run([exe, *a], capture_output=capture, text=capture, env=env)


def http_download(repo, path, dest, token):
    endpoint = os.environ.get("HF_ENDPOINT", HF_HUB).rstrip("/")
    url = f"{endpoint}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, **hf_headers(token)})
        with _urlopen(req, timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  direct download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


def hf_discover(query, limit, token):
    """Search Hub datasets via the REST API (no CLI needed) and show license tags."""
    url = f"{HF_HUB}/api/datasets?{urllib.parse.urlencode({'search': query, 'sort': 'downloads', 'direction': -1, 'limit': limit})}"
    try:
        rows = http_json(url, hf_headers(token))
    except (HttpStatus, NetworkError) as e:
        sys.exit(f"dataset search failed: {e}")
    print(f"{'dataset':52} {'downloads':>10}  license")
    for r in rows if isinstance(rows, list) else []:
        lic = license_from_tags(r.get("tags")) or "?"
        flag = "" if lic in DATA_PERMISSIVE else "   <- check before use"
        print(f"{str(r.get('id')):52} {r.get('downloads', 0):>10}  {lic}{flag}")
    print("\nInspect one with: python3 fetch_data.py hf-files --card OWNER/NAME")


def hf_card(ds, token):
    try:
        info = http_json(f"{HF_HUB}/api/datasets/{urllib.parse.quote(ds, safe='/')}", hf_headers(token))
    except (HttpStatus, NetworkError) as e:
        sys.exit(f"could not read card for {ds}: {e}")
    print(json.dumps({"id": info.get("id"), "license": license_from_tags(info.get("tags")),
                      "gated": info.get("gated"), "cardData": info.get("cardData")}, indent=2))


def convert_commit(src, row, seen, stats):
    path = row.get("new_file") or row.get("old_file") or ""
    ext = os.path.splitext(path)[1].lower()
    if ext not in src["exts"]:
        bump(stats, "other extension")
        return None
    new, old = row.get("new_contents"), row.get("old_contents") or ""
    subject = (row.get("subject") or "").strip()
    lics = licenses_of(row.get("license"))
    reason = None
    if not isinstance(new, str):
        reason = "no content"
    elif not MIN_BYTES <= len(new.encode("utf-8")) <= MAX_BYTES:
        reason = "size out of range"
    elif len(subject.split()) < MIN_INSTRUCTION_WORDS:
        reason = "message too short"
    elif len(old.encode("utf-8")) > MAX_INPUT_BYTES:
        reason = "input too large"
    elif not lics or any(l not in PERMISSIVE for l in lics):
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
        bump(stats, reason)
        return None
    return mi_record(path, new, ",".join(lics),
                     f"hf:{src['repo']} repos={row.get('repos', '')} commit={row.get('commit', '')} path={path}",
                     instruction=subject, inp=old)


def cmd_hf_files(args):
    token = hf_token()
    if args.discover:
        return hf_discover(args.discover, args.limit, token) or 0
    if args.card:
        return hf_card(args.card, token) or 0
    data, raw_dir, _, _ = paths(args, "hf_files_state.json")
    cache, seen = Path(args.cache) if args.cache else data / "hf_cache", set()
    for src in FILE_SOURCES:
        if (args.only and src["name"] not in args.only) or (not args.only and not src["default"]):
            continue
        print(f"[{src['name']}] {src['repo']} ({src['include']})")
        base = Path(args.from_dir) if args.from_dir else cache / src["name"]
        jsonl = base / src["include"]
        if not args.from_dir:
            res = run_hf_cli("download", src["repo"], "--type", "dataset", "--include", src["include"],
                             "--local-dir", str(base))
            if res is None or res.returncode != 0 or not jsonl.is_file():
                print("  `hf` CLI unavailable or failed; trying a direct HTTPS download ...")
                if not http_download(src["repo"], src["include"], jsonl, token):
                    print("  skipping (check network, or set HF_TOKEN)")
                    continue
        if not jsonl.is_file():
            print(f"  expected file not found: {jsonl}; skipping")
            continue
        sink, stats = Sink(raw_dir / f"hf_{src['name']}.jsonl", int(args.mb * 1_000_000), set()), {}
        with jsonl.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bump(stats, "bad json")
                    continue
                rec = convert_commit(src, row, seen, stats)
                if rec:
                    sink.add(rec)
                    if sink.full:
                        break
        sink.close()
        print(f"  {src['name']}: wrote {sink.rows} records, {sink.bytes / 1e6:.2f} MB")
        if stats:
            print("  rejected: " + top_rejections(stats))
        if not args.from_dir and not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    print("done. Next: python3 prepare_data.py")
    return 0


# ============================================================================= instruction-dataset helpers
def hf_dataset_info(ds_id, token):
    try:
        return http_json(f"{HF_HUB}/api/datasets/{urllib.parse.quote(ds_id, safe='/')}", hf_headers(token))
    except (HttpStatus, NetworkError):
        return None


def hf_iter_rows(ds_id, token, max_scan=50_000):
    """Yield (row_idx, row) from the first train split via the Datasets Server."""
    hd = hf_headers(token)
    try:
        splits = http_json(f"{HF_DS_API}/splits?{urllib.parse.urlencode({'dataset': ds_id})}", hd).get("splits") or []
    except (HttpStatus, NetworkError):
        return
    if not splits:
        return
    s = next((x for x in splits if x.get("split") == "train"), splits[0])
    offset = 0
    while offset < max_scan:
        params = {"dataset": ds_id, "config": s["config"], "split": s["split"], "offset": offset, "length": HF_PAGE}
        try:
            res = http_json(f"{HF_DS_API}/rows?{urllib.parse.urlencode(params)}", hd)
        except (HttpStatus, NetworkError):
            return
        rows = res.get("rows") or []
        if not rows:
            return
        for it in rows:
            yield it.get("row_idx", offset), it.get("row", {})
        if len(rows) < HF_PAGE:
            return
        offset += HF_PAGE
        time.sleep(0.2)


def hf_row_to_record(row, ds_id, idx, lic, tag, min_out, transform=None):
    pick = lambda *ks: next((str(row[k]).strip() for k in ks if row.get(k)), "")
    instr = pick("instruction", "prompt", "question", "description")
    inp = pick("input", "context") if instr else ""
    out = pick("output", "response", "answer", "completion", "command", "bash", "code", "content", "text")
    if len(out.encode("utf-8", "ignore")) < min_out:
        return None
    if len(out.encode("utf-8", "ignore")) > MAX_BYTES:
        out = out[:MAX_BYTES]
    row_lics = licenses_of(row.get("license") or row.get("licenses"))
    if row_lics and any(l not in DATA_PERMISSIVE for l in row_lics):
        return None
    use_lic = ",".join(row_lics) or lic
    if transform:
        instr, inp, out = transform(instr), transform(inp), transform(out)
    path = str(row.get("path") or row.get("source_path") or f"{tag}/{ds_id}/row_{idx}")
    ext = os.path.splitext(path)[1].lstrip(".") or ("sh" if out.startswith("#!") or "$ " in out[:8] else "txt")
    return {"source_path": path, "extension": ext, "instruction": instr, "input": inp, "output": out,
            "license": use_lic, "source": f"hf:{ds_id}"}


_TERMUX_SUBS = [
    (re.compile(r"\bsudo\s+apt(?:-get)?\s+(update|upgrade)\b"), r"pkg \1"),
    (re.compile(r"\bsudo\s+apt(?:-get)?\s+install\b"), "pkg install"),
    (re.compile(r"\bapt(?:-get)?\s+install\b"), "pkg install"),
    (re.compile(r"(?<![\w$])(?<!#!)/usr/(?:local/)?(bin|lib|share|etc)\b"), r"$PREFIX/\1"),
    (re.compile(r"(?<![\w$/.])/home/[A-Za-z0-9_.-]+"), "$HOME"),
]


def adapt_to_termux(text: str) -> str:
    """Rewrite generic Linux commands/paths to Termux conventions. Command Q&A only; never run on source
    code (the old version rewrote shebangs and Kotlin/Java strings)."""
    for rx, rep in _TERMUX_SUBS:
        text = rx.sub(rep, text)
    return text


# ============================================================================= subcommand: teams
TEAM_QUERIES = {
    "blue": {
        "hf": ["cybersecurity", "soc", "sigma", "yara", "forensics", "threat", "intrusion", "malware"],
        "gh": ["topic:blue-team stars:>20", "topic:threat-hunting stars:>30", "topic:incident-response stars:>30",
               "topic:sigma-rules stars:>20", "topic:yara-rules stars:>20", "topic:dfir stars:>20",
               "topic:siem stars:>20", "termux hardening OR firewall OR defense stars:>5"],
    },
    "red": {
        "hf": ["pentest", "redteam", "ctf", "exploit", "vulnerability", "offensive", "security", "bash"],
        "gh": ["topic:red-team stars:>30", "topic:pentesting stars:>50", "topic:ctf-tools stars:>30",
               "topic:security-audit stars:>30", "topic:reconnaissance stars:>30",
               "topic:android-security stars:>20", "termux pentest OR recon OR scanner stars:>10"],
    },
}
LINUX_EXTS = {".py", ".sh", ".bash", ".zsh", ".fish", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".go", ".rs",
              ".rb", ".pl", ".pm", ".lua", ".js", ".mjs", ".ts", ".php", ".java", ".kt", ".kts", ".scala", ".swift",
              ".cs", ".zig", ".nim", ".hs", ".ml", ".ex", ".exs", ".erl", ".clj", ".r", ".jl", ".tcl", ".awk",
              ".s", ".asm", ".mk", ".cmake", ".sql", ".ps1"}
TEAM_EXTS = LINUX_EXTS | {".md", ".txt", ".yml", ".yaml", ".json", ".yar", ".yara", ".rules", ".conf", ".toml"}
TEAM_MIN_BYTES = 120


def search_hf_datasets(team, token, limit_per_query=12):
    seen = {}
    for q in TEAM_QUERIES[team]["hf"]:
        url = f"{HF_HUB}/api/datasets?{urllib.parse.urlencode({'search': q, 'limit': limit_per_query, 'sort': 'downloads', 'direction': -1})}"
        try:
            data = http_json(url, hf_headers(token))
        except (HttpStatus, NetworkError):
            continue
        for item in data if isinstance(data, list) else []:
            ds_id = item.get("id")
            lic = license_from_tags(item.get("tags"))
            # strict: unknown license is skipped (the old script let it through as "permissive-check-needed")
            if not ds_id or ds_id in seen or lic not in DATA_PERMISSIVE:
                continue
            seen[ds_id] = {"id": ds_id, "team": team, "downloads": item.get("downloads", 0),
                           "likes": item.get("likes", 0), "license": lic, "gated": bool(item.get("gated"))}
        time.sleep(0.2)
    return sorted(seen.values(), key=lambda d: (d["downloads"], d["likes"]), reverse=True)


def search_github_repos(team, token, per_query=10):
    seen = {}
    for q in TEAM_QUERIES[team]["gh"]:
        try:
            res = gh_api("/search/repositories", token,
                         {"q": f"{q} fork:false", "sort": "stars", "order": "desc", "per_page": per_query})
        except RateLimited as e:
            print(f"  GitHub search stopped: {e}")
            break
        except RuntimeError:
            continue
        for repo in (res or {}).get("items", []):
            full = repo.get("full_name")
            lic = ((repo.get("license") or {}).get("spdx_id") or "").lower()
            if not full or full in seen or repo.get("fork") or lic not in PERMISSIVE:
                continue
            seen[full] = {"full_name": full, "team": team, "stars": repo.get("stargazers_count", 0),
                          "description": (repo.get("description") or "")[:100], "meta": repo}
        search_delay(token)
    return sorted(seen.values(), key=lambda r: r["stars"], reverse=True)


def cmd_teams(args):
    data, raw_dir, lic_dir, state_path = paths(args, "teams_state.json")
    htok, gtok = hf_token(), gh_token()
    teams = ["blue", "red"] if args.team == "both" else [args.team]
    state = load_state(state_path, {"hf_done": [], "gh_done": []})
    skip_repos = known_repos(data)
    hf_by, gh_by = {}, {}
    for t in teams:
        hf_by[t] = search_hf_datasets(t, htok)[: args.max_items] if args.source in ("hf", "both") else []
        gh_by[t] = [r for r in search_github_repos(t, gtok)
                    if r["full_name"].lower() not in skip_repos][: args.max_items] if args.source in ("github", "both") else []
    print(f"Found {sum(map(len, hf_by.values()))} Hugging Face datasets and {sum(map(len, gh_by.values()))} GitHub repos.")
    for t in teams:
        for ds in hf_by[t]:
            print(f"  [HF:{t.upper()}] {ds['id']}  (dl={ds['downloads']}, lic={ds['license']}{' [gated]' if ds['gated'] else ''})")
        for r in gh_by[t]:
            print(f"  [GH:{t.upper()}] {r['full_name']}  (*{r['stars']}) - {r['description']}")
    if args.check_only:
        return 0

    team_budget = int(args.mb / len(teams) * 1024 * 1024) if args.mb else float("inf")  # no total cap by default
    per_item = int(args.per_item_mb * 1024 * 1024)
    grand_rows = grand_bytes = 0
    seen = set()
    try:
        for t in teams:
            t_bytes = t_rows = 0
            print(f"\n--- Fetching {t.upper()} team (budget: {'unlimited' if team_budget == float('inf') else f'{team_budget / 1048576:.1f} MB'}) ---")
            for ds in hf_by[t]:
                if t_bytes >= team_budget:
                    break
                if ds["id"] in state["hf_done"]:
                    continue
                if ds["gated"] and not htok:
                    print(f"  Skipping gated HF dataset {ds['id']} (set HF_TOKEN to enable).")
                    continue
                slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ds["id"])
                sink = Sink(raw_dir / f"hf_team_{t}_{slug}.jsonl", int(min(per_item, team_budget - t_bytes)), seen)
                for idx, row in hf_iter_rows(ds["id"], htok):
                    rec = hf_row_to_record(row, ds["id"], idx, ds["license"], t, TEAM_MIN_BYTES)
                    if rec:
                        sink.add(rec)
                    if sink.full:
                        break
                sink.close()
                print(f"  + HF {ds['id']}: {sink.rows} records ({sink.bytes / 1024:.1f} KB)")
                t_rows, t_bytes = t_rows + sink.rows, t_bytes + sink.bytes
                if sink.rows:
                    state["hf_done"].append(ds["id"])
                    save_state(state_path, state)
            for r in gh_by[t]:
                if t_bytes >= team_budget:
                    break
                if r["full_name"] in state["gh_done"]:
                    continue
                rows, nbytes, note = gh_repo_to_jsonl(r["full_name"], r["meta"], raw_dir, lic_dir, TEAM_EXTS,
                                                      int(min(per_item, team_budget - t_bytes)), f"gh_team_{t}_")
                print(f"  + GH {r['full_name']}: {rows} files ({nbytes / 1024:.1f} KB) {'' if note == 'ok' else note}")
                t_rows, t_bytes = t_rows + rows, t_bytes + nbytes
                if rows:
                    state["gh_done"].append(r["full_name"])
                    save_state(state_path, state)
            grand_rows += t_rows
            grand_bytes += t_bytes
            print(f"  {t.upper()} subtotal: {t_rows} records ({t_bytes / 1048576:.2f} MB)")
    except KeyboardInterrupt:
        print("  interrupted")
    finally:
        save_state(state_path, state)
    print(f"\nDone: added {grand_rows} records ({grand_bytes / 1048576:.2f} MB) to {raw_dir}")
    print("Next: python3 prepare_data.py")
    return 0


# ============================================================================= subcommand: linux
LINUX_LANGS = ["Python", "C", "C++", "Shell", "Go", "Rust", "Perl", "Ruby", "Lua", "JavaScript", "TypeScript",
               "PHP", "Java", "Kotlin", "Scala", "Zig", "Haskell", "OCaml", "Elixir", "Makefile", "CMake", "Assembly"]


def cmd_linux(args):
    """General Linux/open-source code in every common language: GitHub search, language x license."""
    data, raw_dir, lic_dir, state_path = paths(args, "gh_linux_state.json")
    token = gh_token()
    state = load_state(state_path, {"done": {}, "skipped": {}, "search_page": {}})
    skip = known_repos(data)
    langs = args.langs or LINUX_LANGS
    budget, got, saved = int(args.mb * 1_000_000), 0, 0
    try:
        for rnd in range(args.rounds):
            for lang in langs:
                for lic in SEARCH_LICENSES:
                    if got >= budget or saved >= args.max_repos:
                        raise StopIteration
                    key = f"{lang}/{lic}"
                    page = state["search_page"].get(key, 0) + 1
                    q = f'language:"{lang}" topic:linux license:{lic} stars:>100 fork:false'
                    try:
                        res = gh_api("/search/repositories", token,
                                     {"q": q, "sort": "stars", "order": "desc", "per_page": 20, "page": page})
                    except RateLimited as e:
                        print(f"  search stopped: {e}")
                        raise StopIteration
                    except RuntimeError as e:
                        print(f"  search failed: {e}")
                        continue
                    state["search_page"][key] = page
                    for item in (res or {}).get("items", []):
                        full = item.get("full_name")
                        if not full or full in state["done"] or full in state["skipped"] or full.lower() in skip:
                            continue
                        if got >= budget or saved >= args.max_repos:
                            break
                        rows, nbytes, note = gh_repo_to_jsonl(full, item, raw_dir, lic_dir, LINUX_EXTS,
                                                              int(args.per_repo_mb * 1_000_000), "gh_linux_")
                        if rows:
                            saved += 1
                            got += nbytes
                            state["done"][full] = {"lang": lang, "files": rows, "bytes": nbytes}
                        elif note != "download failed":
                            state["skipped"][full] = note
                        print(f"  {full} [{lang}]: {rows} files {'' if rows else note}", flush=True)
                        save_state(state_path, state)
                        time.sleep(0.3)
                    search_delay(token)
    except (StopIteration, KeyboardInterrupt):
        pass
    finally:
        save_state(state_path, state)
    print(f"linux: saved {saved} repos, {got / 1e6:.2f} MB ({len(state['done'])} total)")
    return 0


# ============================================================================= subcommand: termux
CURATED_HF_SHELL = ["aelhalili/bash-commands-dataset", "mecha-org/linux-command-dataset"]
CURATED_GH_TERMUX = ["termux/termux-packages", "termux-user-repository/tur"]  # license verified at run time
TERMUX_HF_QUERIES = ["termux", "bash commands", "linux shell cli", "android filesystem", "android storage saf", "adb shell"]
TERMUX_GH_QUERIES = ["termux cli", "termux scripts", "termux packages", "android scoped storage",
                     "android storage access framework", "android filesystem kotlin", "adb shell scripts"]
TERMUX_EXTS = {".kt", ".java", ".sh", ".py", ".md", ".xml"}
OWN_LICENSE = "mit"  # first-party text written for this project; change if you license it differently

ANDROID_FS_QA = [
    ("Explain Android app-private internal storage paths and how to access them in Kotlin and Termux.",
     "Android app-private internal storage resides under /data/data/<package_name>/ (and /data/user/0/<package_name>/). "
     "In Kotlin/Android, use context.filesDir (/data/data/<pkg>/files) for persistent app files and context.cacheDir "
     "(/data/data/<pkg>/cache) for temporary cache files that the system may evict under storage pressure. "
     "Other apps and unrooted Termux shells cannot read /data/data/<other_pkg> directly. To inspect the files of your "
     "own debuggable app, use `adb shell run-as <debuggable_pkg>`."),
    ("How does Termux access shared Android storage such as Downloads, DCIM, and Pictures?",
     "Run `termux-setup-storage` in Termux and grant the Android storage permission prompt. "
     "This creates symlinks under ~/storage/ (such as ~/storage/downloads, ~/storage/dcim, ~/storage/pictures, "
     "~/storage/movies, ~/storage/music, and ~/storage/shared -> /storage/emulated/0). "
     "Note that Android's FUSE/sdcardfs layer on /storage/emulated/0 does not support POSIX chmod/chown or executable "
     "permissions; keep executable scripts and binaries inside $PREFIX or $HOME."),
    ("What is the difference between $PREFIX, $HOME, and /sdcard in Termux?",
     "In Termux:\n"
     "- $PREFIX is /data/data/com.termux/files/usr, containing installed packages (bin/, etc/, lib/, share/, tmp/).\n"
     "- $HOME is /data/data/com.termux/files/home, the default user home directory supporting standard POSIX permissions.\n"
     "- /sdcard (typically a symlink to /storage/emulated/0) is Android external/shared storage governed by Android "
     "media/scoped storage rules, where POSIX permission bits and symlinks behave differently than in $HOME."),
    ("How do you query or inspect Android storage mounts, free space, and paths from Termux CLI?",
     "Use standard POSIX and Android utilities in Termux:\n"
     "- `df -h $HOME $PREFIX /sdcard` to check filesystem usage and mount targets.\n"
     "- `mount | grep -E 'sdcard|emulated|fuse|data'` to inspect active mount flags.\n"
     "- `ls -la ~/storage` to verify storage symlink targets after `termux-setup-storage`.\n"
     "- `termux-storage-get <output_path>` (with Termux:API installed) to pick a file via the Storage Access Framework."),
    ("Compare MediaStore and Storage Access Framework (SAF) for reading and writing files on modern Android.",
     "On modern Android targeting API 30+ (Android 11+) with Scoped Storage:\n"
     "- Storage Access Framework (SAF) uses `Intent.ACTION_OPEN_DOCUMENT`, `Intent.ACTION_CREATE_DOCUMENT`, or "
     "`Intent.ACTION_OPEN_DOCUMENT_TREE` and returns a `content://` URI. Persist access with "
     "`contentResolver.takePersistableUriPermission(uri, flags)` and read/write via `contentResolver.openInputStream(uri)` "
     "or `openOutputStream(uri)` rather than assuming direct file paths.\n"
     "- MediaStore (`MediaStore.Downloads`, `MediaStore.Images`, `MediaStore.Audio`, `MediaStore.Video`) inserts and "
     "queries media/download entries via `ContentResolver` without broad filesystem path access."),
    ("How do you push, pull, and list files on an Android device filesystem using adb?",
     "Common `adb` filesystem commands:\n"
     "- `adb push local_file.txt /sdcard/Download/local_file.txt`\n"
     "- `adb pull /sdcard/Download/remote_file.txt ./local_copy.txt`\n"
     "- `adb shell ls -la /sdcard/Download`\n"
     "- `adb shell run-as com.example.app ls -la files/` (only works on apps with `android:debuggable=\"true\"`)."),
]


def write_android_fs(raw_dir) -> int:
    path = raw_dir / "hf_android_filesystem_cli.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, (q, a) in enumerate(ANDROID_FS_QA):
            fh.write(json.dumps(mi_record(f"android_fs_spec/{i}.md", a, OWN_LICENSE, "builtin:android_fs_spec",
                                          instruction=q), ensure_ascii=False) + "\n")
    return len(ANDROID_FS_QA)


def discover_termux(known_hf, known_gh, htok, gtok, min_stars, per_query=8):
    hf, gh = {}, {}
    for q in TERMUX_HF_QUERIES:
        url = f"{HF_HUB}/api/datasets?{urllib.parse.urlencode({'search': q, 'sort': 'downloads', 'direction': -1, 'limit': 10})}"
        try:
            res = http_json(url, hf_headers(htok))
        except (HttpStatus, NetworkError):
            continue
        for e in res if isinstance(res, list) else []:
            ds_id = str(e.get("id") or "")
            if ds_id and ds_id.lower() not in known_hf and not e.get("gated") and not e.get("private"):
                hf[ds_id] = {"id": ds_id, "downloads": e.get("downloads", 0), "license": license_from_tags(e.get("tags"))}
        time.sleep(0.2)
    for q in TERMUX_GH_QUERIES:
        try:
            res = gh_api("/search/repositories", gtok, {"q": f"{q} stars:>={min_stars} archived:false fork:false",
                                                        "sort": "stars", "order": "desc", "per_page": per_query})
        except RateLimited as e:
            print(f"  GitHub search stopped: {e}")
            break
        except RuntimeError:
            continue
        for r in (res or {}).get("items", []):
            full = str(r.get("full_name") or "")
            if full and full.lower() not in known_gh:
                gh[full] = r
        search_delay(gtok)
    return (sorted(hf.values(), key=lambda x: x["downloads"], reverse=True),
            sorted(gh.values(), key=lambda r: r.get("stargazers_count", 0), reverse=True))


def cmd_termux(args):
    data, raw_dir, lic_dir, state_path = paths(args, "termux_state.json")
    htok, gtok = hf_token(), gh_token()
    state = load_state(state_path, {"hf_done": [], "gh_done": []})
    known_gh = known_repos(data) | {x.lower() for x in state["gh_done"]}
    known_hf = {x.lower() for x in state["hf_done"]}
    for p in raw_dir.glob("hf_*.jsonl"):
        known_hf.add(p.stem[3:].lower())

    hf_ids = [{"id": i} for i in CURATED_HF_SHELL if i.lower() not in known_hf]
    gh_names = [(n, None) for n in CURATED_GH_TERMUX if n.lower() not in known_gh]
    if args.discover:
        new_hf, new_gh = discover_termux(known_hf, known_gh, htok, gtok, args.min_stars)
        hf_ids += [d for d in new_hf if d["id"] not in {x["id"] for x in hf_ids}]
        gh_names += [(r["full_name"], r) for r in new_gh if r["full_name"] not in {n for n, _ in gh_names}]
    hf_ids, gh_names = hf_ids[: args.max_hf_datasets], gh_names[: args.max_gh_repos]

    if args.dry_run:
        print(json.dumps({"hf_candidates": [d["id"] for d in hf_ids], "github_candidates": [n for n, _ in gh_names]}, indent=2))
        return 0

    summary: dict[str, Any] = {}
    if not args.skip_android_fs:
        summary["hf_android_filesystem_cli.jsonl"] = write_android_fs(raw_dir)
    seen: set = set()
    if not args.skip_hf:
        for d in hf_ids:
            ds_id = d["id"]
            info = hf_dataset_info(ds_id, htok)
            lic = license_from_tags((info or {}).get("tags"))
            if lic not in DATA_PERMISSIVE:
                summary[ds_id] = f"skipped: license {lic or 'unknown'}"
                continue
            if (info or {}).get("gated") and not htok:
                summary[ds_id] = "skipped: gated (set HF_TOKEN)"
                continue
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ds_id)
            sink = Sink(raw_dir / f"hf_{slug}.jsonl", 10 ** 12, seen)
            for idx, row in hf_iter_rows(ds_id, htok, max_scan=args.max_hf_rows):
                rec = hf_row_to_record(row, ds_id, idx, lic, "termux", 3, transform=adapt_to_termux)
                if rec:
                    sink.add(rec)
            sink.close()
            summary[ds_id] = sink.rows
            if sink.rows:
                state["hf_done"].append(ds_id)
    if not args.skip_gh:
        for full, meta in gh_names:
            try:
                meta = meta or gh_api(f"/repos/{full}", gtok)
            except (RateLimited, RuntimeError) as e:
                summary[full] = f"skipped: {e}"
                continue
            rows, nbytes, note = gh_repo_to_jsonl(full, meta, raw_dir, lic_dir, TERMUX_EXTS,
                                                  int(args.per_repo_mb * 1_000_000), "gh_termux_")
            summary[full] = rows if note == "ok" else f"skipped: {note}"
            if rows:
                state["gh_done"].append(full)
            time.sleep(0.3)
    save_state(state_path, state)
    print(json.dumps({"raw_dir": str(raw_dir), "records_written": summary}, indent=2))
    return 0


# ============================================================================= cli
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="data directory (default: ./data, or ~/MI/data)")
    common.add_argument("--raw", help="output dir for JSONL (default: <data-dir>/raw)")
    common.add_argument("--licenses", help="license text dir (default: <data-dir>/licenses)")
    common.add_argument("--state", help="state file path (default depends on subcommand)")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", metavar="{github,linux,hf,hf-files,teams,termux,all}  (default: all)")

    p = sub.add_parser("github", parents=[common], help="Android code from GitHub")
    p.add_argument("--mb", type=float, default=50.0, help="stop after this many MB of new code")
    p.add_argument("--per-repo-mb", type=float, default=1.5, help="cap per repository (variety over volume)")
    p.add_argument("--max-repos", type=int, default=60)
    p.add_argument("--searches", type=int, default=6, help="GitHub search requests per run")
    p.add_argument("--seeds-only", action="store_true")
    p.set_defaults(fn=cmd_github)

    p = sub.add_parser("hf", parents=[common], help="code from Hugging Face (Datasets Server)")
    p.add_argument("--mb", type=float, default=50.0, help="total new MB across sources")
    p.add_argument("--only", nargs="*", help="source names: " + ", ".join(s["name"] for s in HF_SOURCES))
    p.add_argument("--max-requests", type=int, default=300, help="cap per source")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-scan", type=int, default=60_000, help="max rows read per streamed language file")
    p.add_argument("--list-configs", metavar="DATASET")
    p.set_defaults(fn=cmd_hf)

    p = sub.add_parser("hf-files", parents=[common], help="JSONL datasets via hf CLI (commitpackft) + discovery")
    p.add_argument("--mb", type=float, default=50.0, help="max MB per source")
    p.add_argument("--only", nargs="*", help="source names: " + ", ".join(s["name"] for s in FILE_SOURCES))
    p.add_argument("--cache", help="download cache (default: <data-dir>/hf_cache)")
    p.add_argument("--keep", action="store_true", help="keep downloaded files after parsing")
    p.add_argument("--from-dir", help="parse already-downloaded files from this folder")
    p.add_argument("--discover", metavar="QUERY", help="search Hub datasets and show licenses")
    p.add_argument("--card", metavar="DATASET", help="print a dataset's license/card metadata")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_hf_files)

    p = sub.add_parser("linux", parents=[common], help="Linux/open-source code in all common languages")
    p.add_argument("--mb", type=float, default=50.0, help="stop after this many MB of new code")
    p.add_argument("--per-repo-mb", type=float, default=1.5)
    p.add_argument("--max-repos", type=int, default=100)
    p.add_argument("--rounds", type=int, default=1, help="passes over the language list (next page each pass)")
    p.add_argument("--langs", nargs="*", help="override language list (default: " + ", ".join(LINUX_LANGS[:8]) + ", ...)")
    p.set_defaults(fn=cmd_linux)

    p = sub.add_parser("teams", parents=[common], help="Blue/Red team security datasets and repos")
    p.add_argument("--team", choices=["blue", "red", "both"], default="both")
    p.add_argument("--source", choices=["hf", "github", "both"], default="both")
    p.add_argument("--mb", type=float, default=50.0, help="total MB cap, split evenly across teams (0 = no cap)")
    p.add_argument("--per-item-mb", type=float, default=25.0)
    p.add_argument("--max-items", type=int, default=25, help="max datasets/repos per source per team")
    p.add_argument("--check-only", action="store_true", help="only list matches")
    p.set_defaults(fn=cmd_teams)

    p = sub.add_parser("termux", parents=[common], help="Termux CLI + Android filesystem data")
    p.add_argument("--discover", action="store_true", help="also search HF/GitHub for new sources")
    p.add_argument("--dry-run", action="store_true", help="list candidates, download nothing")
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--skip-gh", action="store_true")
    p.add_argument("--skip-android-fs", action="store_true")
    p.add_argument("--max-hf-datasets", type=int, default=5)
    p.add_argument("--max-hf-rows", type=int, default=2500, help="max rows scanned per HF dataset")
    p.add_argument("--max-gh-repos", type=int, default=8)
    p.add_argument("--per-repo-mb", type=float, default=1.5)
    p.add_argument("--min-stars", type=int, default=15)
    p.set_defaults(fn=cmd_termux)

    p = sub.add_parser("all", parents=[common], help="android -> python/linux -> termux -> teams, 50 MB limits")
    p.set_defaults(fn=None)
    return ap


def main(argv=None):
    ap = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0].startswith("-") and not any(a in ("-h", "--help") for a in argv[:1]):
        argv = ["all", *argv]  # bare invocation (or only global options) means "all"
    args = ap.parse_args(argv)
    if args.cmd == "all":
        rc = 0
        for name in ("github", "hf-files", "hf", "linux", "termux", "teams"):  # priority: Android, Python/Linux, Termux, teams last
            sub_args = ap.parse_args([name])
            for k in ("data_dir", "raw", "licenses"):
                setattr(sub_args, k, getattr(args, k))
            print(f"\n===== {name} =====")
            rc = sub_args.fn(sub_args) or rc
        return rc
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
