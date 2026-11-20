import math
import torch
from torch import nn
from torch.nn import functional as F


def build_rope(max_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: [batch, heads, time, head_dim]; cos/sin: [time, head_dim/2]; rotates (even, odd) pairs.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q = apply_rope(q.view(shape).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(shape).transpose(1, 2), cos, sin)
        v = v.view(shape).transpose(1, 2)
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        present_kv = (k, v) if use_cache else None
        is_causal = (past_kv is None) and (t > 1)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        out = self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return (out, present_kv) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        if use_cache:
            attn_out, present_kv = self.attn(self.norm1(x), cos, sin, past_kv=past_kv, use_cache=True)
            x = x + attn_out
            return x + self.mlp(self.norm2(x)), present_kv
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class MADIntelligence(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        cos, sin = build_rope(config.max_seq_len, config.d_model // config.n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._initialize)
        # Scale residual-output projections so deeper stacks start stable.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, targets=None, past_kv=None, use_cache=False):
        t = input_ids.shape[1]
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        if past_len + t > self.config.max_seq_len:
            raise ValueError(f"Sequence exceeds context length {self.config.max_seq_len}")
        cos = self.rope_cos[past_len:past_len + t]
        sin = self.rope_sin[past_len:past_len + t]
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        next_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            if use_cache:
                x, layer_kv = block(x, cos, sin, past_kv=layer_past, use_cache=True)
                next_kv.append(layer_kv)
            else:
                x = block(x, cos, sin)
        logits = self.lm_head(self.final_norm(x))
        # Soft-cap logits to prevent extreme values and gradient spikes with tied embeddings.
        logits = 30.0 * torch.tanh(logits / 30.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, next_kv
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=80, temperature=0.8, top_k=40,
                 repetition_penalty=1.0, stop_token_id=None):
        was_training = self.training
        self.eval()
        idx = input_ids[:, -self.config.max_seq_len:]
        logits, _, past_kv = self(idx, use_cache=True)
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :].clone()
            if repetition_penalty != 1.0:
                window = input_ids[:, -self.config.max_seq_len:]
                seen = next_logits.gather(1, window)
                seen = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
                next_logits.scatter_(1, window, seen)
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k and top_k < next_logits.size(-1):
                    values, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < values[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if stop_token_id is not None and bool((next_id == stop_token_id).all()):
                break
            if step + 1 < max_new_tokens:
                if past_kv[0][0].shape[2] >= self.config.max_seq_len:
                    logits, _, past_kv = self(input_ids[:, -self.config.max_seq_len:], use_cache=True)
                else:
                    logits, _, past_kv = self(next_id, past_kv=past_kv, use_cache=True)
        self.train(was_training)
        return input_ids
