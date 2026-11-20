"""Byte-level BPE tokenizer trained from local text; no external vocabulary.

Format MAD-BPE-2: text is first split into "words" (a word may carry one leading
space, runs of punctuation, or runs of whitespace). Merges never cross word
boundaries, which makes training and encoding fast enough for a phone.
"""
import gc
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SPECIAL_TOKENS = [
    "<MAD_START>", "<MAD_USER>", "<MAD_ASSISTANT>",
    "<MAD_CODE>", "<MAD_END>", "<MAD_PAD>", "<MAD_UNKNOWN>"
]
FORMAT = "MAD-BPE-2"

_WORD = re.compile(r" ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")
_CACHE_LIMIT = 25_000


def _merge_word(ids, a, b, new_id):
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class MADTokenizer:
    def __init__(self, merges=None, vocab_size=None):
        self.merges = [list(map(int, m)) for m in (merges or [])]
        self.vocab_size = int(vocab_size) if vocab_size else self.size
        self._refresh()

    def _refresh(self):
        self.special_to_id = {t: 256 + len(self.merges) + i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self._ranks = {tuple(m): r for r, m in enumerate(self.merges)}
        table = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            table.append(table[a] + table[b])
        self._bytes = table
        self._cache = {}

    @property
    def size(self):
        return 256 + len(self.merges) + len(SPECIAL_TOKENS)

    # ---------------------------------------------------------------- training
    def train(self, texts, vocab_size=None, min_frequency=2):
        """Learn merges from an iterable of text chunks."""
        target = int(vocab_size or self.vocab_size)
        max_merges = max(0, target - 256 - len(SPECIAL_TOKENS))
        word_counts = Counter()
        for text in texts:
            if not text:
                continue
            for part in _SPECIAL_RE.split(text)[::2]:  # even indexes = ordinary text
                word_counts.update(_WORD.findall(part))
        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())
        del word_counts

        pair_counts = Counter()
        where = defaultdict(set)
        for i, w in enumerate(words):
            f = freqs[i]
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(i)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = []
        while len(merges) < max_merges and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current <= 0:
                continue
            if -neg != current:  # stale entry; counts only ever shrink for old pairs
                heapq.heappush(heap, (-current, pair))
                continue
            if current < min_frequency:
                break
            a, b = pair
            new_id = 256 + len(merges)
            merges.append([a, b])
            touched = set()
            for i in list(where.pop(pair, ())):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                w = _merge_word(w, a, b, new_id)
                words[i] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    if new_id in p:
                        touched.add(p)
            pair_counts[pair] = 0
            for p in touched:
                heapq.heappush(heap, (-pair_counts[p], p))

        del words, freqs, pair_counts, where, heap
        gc.collect()
        self.merges = merges
        self.vocab_size = 256 + len(merges) + len(SPECIAL_TOKENS)
        self._refresh()
        return self

    # ---------------------------------------------------------------- encoding
    def _encode_word(self, word):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word.encode("utf-8"))
        ranks = self._ranks
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = _merge_word(ids, best[0], best[1], 256 + best_rank)
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def encode(self, text, add_special_tokens=False):
        result = [self.special_to_id["<MAD_START>"]] if add_special_tokens else []
        for i, part in enumerate(_SPECIAL_RE.split(text)):
            if i % 2:
                result.append(self.special_to_id[part])
            elif part:
                for word in _WORD.findall(part):
                    result.extend(self._encode_word(word))
        return result

    # ---------------------------------------------------------------- decoding
    def decode(self, ids):
        chunks = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                chunks.append(self.id_to_special[token_id].encode("utf-8"))
            elif 0 <= token_id < 256 + len(self.merges):
                chunks.append(self._bytes[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ------------------------------------------------------------- persistence
    def save(self, path):
        payload = {
            "format": FORMAT,
            "vocab_size": self.size,
            "merges": self.merges,
            "special_tokens": SPECIAL_TOKENS,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(
                f"Unsupported tokenizer format {payload.get('format')!r} (need {FORMAT}). "
                "Retrain it with `python mad.py train-tokenizer`.")
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("Special-token definition mismatch")
        return cls(payload["merges"], payload.get("vocab_size"))
