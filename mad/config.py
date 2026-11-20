from dataclasses import asdict, dataclass
import json
from pathlib import Path

# Rough sizes (vocab 1024): tiny ~0.5M, small ~2.9M, medium ~6.7M, large ~19M parameters.
PRESETS = {
    "tiny":   dict(d_model=96,  n_heads=3, n_layers=4, d_ff=256, max_seq_len=128),
    "small":  dict(d_model=192, n_heads=6, n_layers=6, d_ff=512, max_seq_len=256),
    "medium": dict(d_model=256, n_heads=8, n_layers=8, d_ff=704, max_seq_len=256),
    "large":  dict(d_model=384, n_heads=6, n_layers=10, d_ff=1024, max_seq_len=512),  # PC / strong phone
}


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    d_model: int = 96
    n_heads: int = 3
    n_layers: int = 4
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.0
    tie_embeddings: bool = True

    @classmethod
    def from_preset(cls, name, **overrides):
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}")
        return cls(**{**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}})

    def validate(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension (d_model / n_heads) must be even for rotary embeddings")
        if self.vocab_size < 263:
            raise ValueError("vocab_size must allow 256 byte tokens and special tokens")
        if min(self.d_model, self.n_heads, self.n_layers, self.d_ff, self.max_seq_len) <= 0:
            raise ValueError("Model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Config file not found: {path}. Run `python mad.py init` first to create it."
            )
        try:
            obj = cls(**json.loads(path.read_text(encoding="utf-8")))
        except TypeError as e:
            raise ValueError(f"Invalid config file {path}: {e}") from e
        obj.validate()
        return obj

    @classmethod
    def load_or_default(cls, path):
        """Return (config, loaded_from_file). Falls back to built-in defaults if no file exists."""
        if Path(path).is_file():
            return cls.load(path), True
        obj = cls()
        obj.validate()
        return obj, False
