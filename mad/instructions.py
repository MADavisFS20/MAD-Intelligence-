"""Natural-language -> code training data.

An example is a request in plain English and the code that answers it. It is stored one per
line as JSON (JSONL):

    {"instruction": "Write a Kotlin function that checks if a string is a valid email", "response": "fun isValidEmail(s: String): Boolean { ... }"}

Optional field "context" holds code the request refers to. Aliases accepted for the fields:
instruction = prompt | question, response = output | completion | answer | code, context = input.

Training text layout (the loss is computed on the response only):

    <MAD_USER>{instruction}[\\n\\n{context}]<MAD_ASSISTANT><MAD_CODE>{response}<MAD_END>
"""
import hashlib
import json
import re
import textwrap
from pathlib import Path

import torch

from .data import find_secret
from .tokenizer import SPECIAL_TOKENS

PROMPT_OPEN = "<MAD_USER>"
PROMPT_CLOSE = "<MAD_ASSISTANT><MAD_CODE>"
END = "<MAD_END>"
IGNORE = -100  # label value ignored by the loss

MAX_INSTRUCTION_CHARS = 2000
MAX_RESPONSE_CHARS = 20000
_SPECIAL_STRIP = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))
_ALIASES = {
    "instruction": ("instruction", "prompt", "question"),
    "response": ("response", "output", "completion", "answer", "code"),
    "context": ("context", "input"),
}


def strip_special(text):
    """Remove control-token look-alikes from user text so they cannot inject structure."""
    return _SPECIAL_STRIP.sub("", text)


def build_prompt(instruction, context=None):
    text = strip_special(instruction).strip()
    if context and context.strip():
        text += "\n\n" + strip_special(context).strip()
    return f"{PROMPT_OPEN}{text}{PROMPT_CLOSE}"


def format_example(pair):
    return build_prompt(pair["instruction"], pair.get("context")) + strip_special(pair["response"]).strip("\n") + END


# --------------------------------------------------------------------------- loading
def _pick(record, field):
    for key in _ALIASES[field]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_pairs(path):
    """Read a JSONL file or every *.jsonl in a folder. Returns (pairs, problems)."""
    path = Path(path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    pairs, problems = [], []
    for f in files:
        if not f.is_file():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f.name}:{n}"
            try:
                record = json.loads(line)
            except ValueError:
                problems.append((where, "not valid JSON"))
                continue
            if not isinstance(record, dict):
                problems.append((where, "line is not a JSON object"))
                continue
            instruction, response = _pick(record, "instruction"), _pick(record, "response")
            if not instruction or not response:
                problems.append((where, "needs both an instruction and a response"))
                continue
            if len(instruction) > MAX_INSTRUCTION_CHARS or len(response) > MAX_RESPONSE_CHARS:
                problems.append((where, "too long"))
                continue
            pair = {"instruction": instruction.strip(), "response": response.strip("\n")}
            context = _pick(record, "context")
            if context:
                pair["context"] = context.strip("\n")
            pairs.append(pair)
    return pairs, problems


def clean_pairs(pairs, scan_secrets=True):
    """Drop duplicates and (optionally) pairs that look like they contain secrets."""
    seen, kept, skipped = set(), [], []
    for p in pairs:
        key = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        if key in seen:
            skipped.append((p["instruction"][:60], "duplicate"))
            continue
        seen.add(key)
        if scan_secrets:
            secret = find_secret(p["instruction"] + "\n" + p.get("context", "") + "\n" + p["response"])
            if secret:
                skipped.append((p["instruction"][:60], f"possible secret ({secret})"))
                continue
        kept.append(p)
    return kept, skipped


def write_pairs(pairs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def split_pairs(pairs, val_fraction=0.1):
    """Deterministic split by content hash, so the held-out set never changes between runs."""
    def score(p):
        return int(hashlib.sha256((p["instruction"] + p["response"]).encode()).hexdigest()[:8], 16)
    if len(pairs) < 10 or val_fraction <= 0:
        return list(pairs), []
    ranked = sorted(pairs, key=score)
    n_val = max(1, int(len(pairs) * val_fraction))
    val = ranked[:n_val]
    ids = {id(p) for p in val}
    return [p for p in pairs if id(p) not in ids], val


# ----------------------------------------------------------- extraction from your code
_DOC_BLOCK = re.compile(r"/\*\*(.*?)\*/", re.S)
_LINE_COMMENTS = re.compile(r"(?:^[ \t]*//[^\n]*\n)+", re.M)
_STATEMENT_START = re.compile(r"^(?:if|for|while|when|return|else|try|catch|switch|import|package)\b")
_CLASS_LIKE = re.compile(r"\b(?:class|interface|object|enum)\b")
LANGUAGES = {".kt": "Kotlin", ".kts": "Kotlin", ".java": "Java"}


def _match_brace(text, start):
    """Index just past the } that closes the { at text[start], skipping strings and comments."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            if text.startswith('"""', i):
                j = text.find('"""', i + 3)
                i = n if j < 0 else j + 3
                continue
            i += 1
            while i < n and text[i] != '"' and text[i] != "\n":
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if c == "'":
            i += 1
            while i < n and text[i] != "'" and text[i] != "\n":
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _declaration_end(source, start):
    """End index of the declaration that begins at `start`, or -1 if it does not look like one."""
    window = source[start:start + 800]
    brace, semi, blank = window.find("{"), window.find(";"), window.find("\n\n")
    if brace != -1 and (semi == -1 or brace < semi) and (blank == -1 or brace < blank):
        header = window[:brace].strip()
        header_body = re.sub(r"^(?:@\w+(?:\([^)\n]*\))?\s*)+", "", header)
        if _STATEMENT_START.match(header_body) or not ("(" in header_body or _CLASS_LIKE.search(header_body)):
            return -1
        end = _match_brace(source, start + brace)
        return end
    first_line = window.split("\n", 1)[0]
    if "fun " in first_line and re.search(r"\)\s*(?::[^=]+)?=(?:\s|$)", first_line):  # Kotlin expression body
        lines = window.split("\n")
        keep = []
        for line in lines[:15]:
            if not line.strip():
                break
            keep.append(line)
        return start + len("\n".join(keep))
    plain = re.sub(r"^(?:@\w+(?:\([^)\n]*\))?\s*)+", "", first_line)
    if re.match(r"(?:(?:data|sealed|enum|open|abstract|inner|value|private|internal|public)\s+)*class\s+\w+", plain) and "(" in first_line:
        # Brace-less class such as `data class Note(val id: Long, val name: String)`
        open_at = start + window.index("(")
        depth, i = 0, open_at
        while i < len(source) and i < open_at + 800:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    line_end = source.find("\n", i)
                    return len(source) if line_end < 0 else line_end
            i += 1
    return -1


def _clean_doc(raw, line_comment=False):
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^(?://+|\*+)\s?", "", line).strip()
        if line.startswith("@"):
            continue
        lines.append(line)
    text = " ".join(l for l in lines if l)
    text = re.sub(r"\{@\w+\s+([^}]*)\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pairs(source, language, min_words=3, max_words=60, min_chars=25, max_chars=6000):
    """Turn documented declarations (/** doc */ or // comment above a fun/class/method) into pairs."""
    anchors = [(m.end(), _clean_doc(m.group(1))) for m in _DOC_BLOCK.finditer(source)]
    anchors += [(m.end(), _clean_doc(m.group(0), True)) for m in _LINE_COMMENTS.finditer(source)]
    pairs = []
    for pos, doc in sorted(anchors):
        words = doc.split()
        if not (min_words <= len(words) <= max_words) or doc.lower().startswith(("todo", "fixme", "copyright", "license")):
            continue
        start = pos
        while start < len(source) and source[start] in " \t\r\n":
            start += 1
        end = _declaration_end(source, start)
        if end <= start:
            continue
        line_start = source.rfind("\n", 0, start) + 1
        code = textwrap.dedent(source[line_start:end]).strip("\n")
        if not (min_chars <= len(code) <= max_chars):
            continue
        pairs.append({"instruction": f"In {language}: {doc}", "response": code})
    return pairs


def extract_from_files(roots, max_file_bytes=2_000_000):
    """Walk Kotlin/Java files and extract pairs. Returns (pairs, files_scanned)."""
    from .data import _skipped_dir  # same folder-skipping rules as the corpus builder
    if isinstance(roots, (str, Path)):
        roots = [roots]
    pairs, scanned = [], 0
    for root in roots:
        root = Path(root).expanduser()
        files = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in files:
            language = LANGUAGES.get(path.suffix.lower())
            if not language or not path.is_file():
                continue
            if path != root and _skipped_dir(path.relative_to(root).parts):
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            pairs.extend(extract_pairs(text, language))
    return pairs, scanned


# ------------------------------------------------------------------------ batching
def encode_examples(pairs, tokenizer, seq_len):
    """Return (examples, dropped, truncated). Each example is (token_ids, prompt_length)."""
    end_id = tokenizer.special_to_id[END]
    examples, dropped, truncated = [], 0, 0
    for p in pairs:
        prompt = tokenizer.encode(build_prompt(p["instruction"], p.get("context")))
        response = tokenizer.encode(strip_special(p["response"]).strip("\n")) + [end_id]
        if len(prompt) >= seq_len - 1:
            dropped += 1  # request alone fills the whole context
            continue
        ids = prompt + response
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            truncated += 1
        examples.append((torch.tensor(ids, dtype=torch.long), len(prompt)))
    return examples, dropped, truncated


def make_batch_fn(examples, batch_size, pad_id):
    """Batches of padded examples whose labels cover the response only."""
    def batch(generator=None):
        picks = torch.randint(0, len(examples), (batch_size,), generator=generator)
        chosen = [examples[int(i)] for i in picks]
        width = max(len(ids) for ids, _ in chosen) - 1
        x = torch.full((batch_size, width), pad_id, dtype=torch.long)
        y = torch.full((batch_size, width), IGNORE, dtype=torch.long)
        for row, (ids, prompt_len) in enumerate(chosen):
            n = len(ids) - 1
            x[row, :n] = ids[:-1]
            target = ids[1:].clone()
            target[: prompt_len - 1] = IGNORE
            y[row, :n] = target
        return x, y
    return batch


def balanced(code):
    """Cheap sanity check for generated code: do brackets pair up?"""
    for open_c, close_c in ("{}", "()", "[]"):
        if code.count(open_c) != code.count(close_c):
            return False
    return True
