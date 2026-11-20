import math

import pytest
import torch

from mad.config import PRESETS, ModelConfig
from mad.model import MADIntelligence
from mad.tokenizer import SPECIAL_TOKENS, MADTokenizer
from mad.trainer import get_batch, lr_at

SAMPLES = [
    'fun main() { println("MAD") }',
    "Android Kotlin UTF-8: café ✓",
    "class A {\n    val x = 1\n\n    fun f() = x\n}\n",
]


def small_cfg(**kw):
    base = dict(vocab_size=300, d_model=24, n_heads=3, n_layers=1, d_ff=48, max_seq_len=16)
    base.update(kw)
    return ModelConfig(**base)


# ------------------------------------------------------------------ tokenizer
def test_tokenizer_roundtrip():
    tok = MADTokenizer(vocab_size=300).train(SAMPLES, 300)
    for sample in SAMPLES:
        assert tok.decode(tok.encode(sample)) == sample


def test_tokenizer_roundtrip_untrained_and_odd_text():
    tok = MADTokenizer()
    for text in ["", " ", "\n\n\n", "  leading", "trailing  ", "\t\ttabs", "日本語 テキスト", "a" * 500]:
        assert tok.decode(tok.encode(text)) == text


def test_tokenizer_compresses_and_is_deterministic():
    text = "fun onCreate(savedInstanceState: Bundle?) {\n    super.onCreate(savedInstanceState)\n}\n" * 40
    a = MADTokenizer().train([text], 400)
    b = MADTokenizer().train([text], 400)
    assert a.merges == b.merges and a.merges
    assert len(a.encode(text)) < len(text.encode()) / 3


def test_special_tokens_roundtrip_and_ids():
    tok = MADTokenizer().train(SAMPLES * 5, 300)
    ids = tok.encode("<MAD_USER>hi<MAD_END>")
    assert ids[0] == tok.special_to_id["<MAD_USER>"] and ids[-1] == tok.special_to_id["<MAD_END>"]
    assert tok.decode(ids) == "<MAD_USER>hi<MAD_END>"
    assert tok.size == 256 + len(tok.merges) + len(SPECIAL_TOKENS)


def test_tokenizer_save_load(tmp_path):
    tok = MADTokenizer().train(SAMPLES * 5, 300)
    path = tmp_path / "t.json"
    tok.save(path)
    again = MADTokenizer.load(path)
    assert again.merges == tok.merges and again.encode(SAMPLES[0]) == tok.encode(SAMPLES[0])


def test_old_tokenizer_format_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"format": "MAD-BPE-1", "merges": [], "special_tokens": []}')
    with pytest.raises(ValueError, match="train-tokenizer"):
        MADTokenizer.load(path)


# ---------------------------------------------------------------------- model
def test_model_forward_and_backward():
    cfg = small_cfg(vocab_size=263)
    model = MADIntelligence(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    _, loss = model(x, x)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_random_initialization_differs():
    cfg = small_cfg(vocab_size=263)
    torch.manual_seed(1); a = MADIntelligence(cfg)
    torch.manual_seed(2); b = MADIntelligence(cfg)
    assert not torch.equal(a.token_embedding.weight, b.token_embedding.weight)


def test_model_is_causal():
    cfg = small_cfg()
    model = MADIntelligence(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    y = x.clone()
    y[0, 7:] = (y[0, 7:] + 1) % cfg.vocab_size
    with torch.no_grad():
        la, _ = model(x)
        lb, _ = model(y)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


def test_generate_stop_token_and_mode_restored():
    cfg = small_cfg()
    model = MADIntelligence(cfg)
    model.train()
    ids = torch.tensor([[1, 2, 3]])
    out = model.generate(ids, max_new_tokens=5, temperature=0, stop_token_id=None)
    assert out.shape[1] == 8 and model.training
    first = int(out[0, 3])
    short = model.generate(ids, max_new_tokens=5, temperature=0, stop_token_id=first)
    assert short.shape[1] == 4
    model.generate(ids, max_new_tokens=3, repetition_penalty=1.3)  # penalty path runs


def test_presets_valid_and_sizes():
    sizes = {}
    for name in PRESETS:
        sizes[name] = MADIntelligence(ModelConfig.from_preset(name)).num_parameters()
    assert sizes["tiny"] < sizes["small"] < sizes["medium"]
    assert 400_000 < sizes["tiny"] < 700_000


def test_old_checkpoint_weights_still_load():
    cfg = small_cfg()
    state = MADIntelligence(cfg).state_dict()
    assert not any("rope" in k for k in state)  # buffers are not persisted
    MADIntelligence(cfg).load_state_dict(state)


# --------------------------------------------------------------------- config
def test_config_missing_file_is_clear_and_inspect_default(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="python mad.py init"):
        ModelConfig.load(missing)
    cfg, from_file = ModelConfig.load_or_default(missing)
    assert from_file is False and cfg.vocab_size == 1024


def test_config_roundtrip_and_validation(tmp_path):
    path = tmp_path / "config.json"
    ModelConfig().save(path)
    cfg, from_file = ModelConfig.load_or_default(path)
    assert from_file is True and cfg == ModelConfig()
    with pytest.raises(ValueError):
        ModelConfig(d_model=21, n_heads=3).validate()  # odd head dim
    path.write_text('{"bogus": 1}')
    with pytest.raises(ValueError, match="Invalid config"):
        ModelConfig.load(path)


# -------------------------------------------------------------------- trainer
def test_lr_schedule_shape():
    assert lr_at(1, 100, 1e-3, 10) == pytest.approx(1e-4)
    assert lr_at(10, 100, 1e-3, 10) == pytest.approx(1e-3)
    assert lr_at(100, 100, 1e-3, 10) == pytest.approx(1e-4)
    mid = lr_at(55, 100, 1e-3, 10)
    assert 1e-4 < mid < 1e-3


def test_get_batch_alignment_and_bounds():
    tokens = torch.arange(50, dtype=torch.int16)
    for _ in range(50):
        x, y = get_batch(tokens, 3, 10)
        assert x.dtype == torch.long and torch.equal(y, x + 1) and int(y.max()) <= 49
    with pytest.raises(ValueError):
        get_batch(torch.arange(5), 1, 10)
