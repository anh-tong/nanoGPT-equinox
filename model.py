import math
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom


@dataclass
class GPTConfig:

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(eqx.Module):

    config: GPTConfig

    c_attn: nn.Linear
    c_proj: nn.Linear
    attn_dropout: nn.Dropout
    resid_dropout: nn.Dropout
    n_head: int
    n_embd: int

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey):
        attn_key, proj_key = jrandom.split(key)
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd * 3, use_bias=config.bias, key=attn_key
        )
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=proj_key
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x, *, key: jrandom.PRNGKey = None, inference: bool = None):
        attn_key, dropout_key = (None, None) if key is None else jrandom.split(key)
        seq_len, _ = x.shape

        x = jax.vmap(self.c_attn)(x)
        q, k, v = jnp.split(x, 3, axis=-1)
        q = q.reshape(seq_len, self.n_head, -1)
        k = k.reshape(seq_len, self.n_head, -1)
        v = v.reshape(seq_len, self.n_head, -1)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_fn = partial(
            nn.attention.dot_product_attention,
            mask=mask,
            dropout=self.attn_dropout,
            inference=inference,
        )
        keys = None if attn_key is None else jrandom.split(attn_key, self.n_head)
        x = jax.vmap(attn_fn, in_axes=1, out_axes=1)(q, k, v, key=keys)

        x = x.reshape(seq_len, -1)

        x = jax.vmap(self.c_proj)(x)
        x = self.resid_dropout(x, key=dropout_key, inference=inference)

        return x


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0)))
        )
    )


class MLP(eqx.Module):

    c_fc: nn.Linear
    c_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey) -> None:
        fc_key, proj_key = jrandom.split(key)
        self.c_fc = nn.Linear(
            config.n_embd, config.n_embd * 4, use_bias=config.bias, key=fc_key
        )
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, use_bias=config.bias, key=proj_key
        )
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x, *, key: jrandom.PRNGKey = None, inference: bool = None):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key, inference=inference)

        return x


class Block(eqx.Module):

    config: GPTConfig
    ln_1: nn.LayerNorm
    ln_2: nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey) -> None:
        attn_key, mlp_key = jrandom.split(key)
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, key=attn_key)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, key=mlp_key)

    def __call__(self, x, *, key: jrandom.PRNGKey = None, inference: bool = None):
        key = None if key is None else jrandom.split(key, 1)[0]
        x = x + self.attn(self.ln_1(x), key=key, inference=inference)
        key = None if key is None else jrandom.split(key, 1)[0]
        x = x + jax.vmap(partial(self.mlp, key=key, inference=inference))(self.ln_2(x))
        return x


class GPT(eqx.Module):

    config: GPTConfig
    transfomer: eqx.Module
    lm_head: nn.Linear

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey):

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config, key=k) for k in jrandom.split(key, config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)

    def __call__(
        self, idx, target=None, *, key: jrandom.PRNGKey = None, inference: bool = None
    ):

        t = idx.shape[0]
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        x = self.drop(tok_emb + pos_emb, key=key, inference=inference)

        for block in self.h:
            key = jrandom.split(key, 1)[0]
            x = block(x, key=key, inference=inference)

        x = self.ln_f(x)


config = GPTConfig()
key = jrandom.PRNGKey(0)
attn = CausalSelfAttention(config, key=key)
block = Block(config=config, key=key)
x = jrandom.normal(key, shape=(20, config.n_embd))
block(x, key=key, inference=False)
block(x, key=key, inference=True)
