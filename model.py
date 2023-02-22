import math
from dataclasses import dataclass
from functools import partial
from typing import List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import optax


@dataclass
class GPTConfig:

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    dtype: str = "bfloat16"


def convert_bfloat16(module: eqx.Module):
    module = eqx.tree_at(lambda l: l.weight, module, module.weight.astype(jnp.bfloat16))
    if hasattr(module, "bias"):
        if module.bias is not None:
            module = eqx.tree_at(
                lambda l: l.bias, module, module.bias.astype(jnp.bfloat16)
            )
    return module


def init_weight(module: eqx.Module, *, key: jrandom.PRNGKey):
    module = eqx.tree_at(
        lambda l: l.weight,
        module,
        jrandom.normal(key=key, shape=module.weight.shape) * 0.02,
    )
    if hasattr(module, "bias"):
        if module.bias is not None:
            key = jrandom.split(key, 1)[0]
            module = eqx.tree_at(lambda l: l.bias, module, jnp.zeros_like(module.bias))
    return module


class CausalSelfAttention(eqx.Module):

    config: GPTConfig = eqx.static_field()

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
        c_attn = nn.Linear(
            config.n_embd, config.n_embd * 3, use_bias=config.bias, key=attn_key
        )
        c_proj = nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=proj_key
        )
        c_attn = init_weight(c_attn, key=attn_key)
        c_proj = init_weight(c_proj, key=proj_key)
        c_proj = eqx.tree_at(
            lambda l: l.weight,
            c_proj,
            jrandom.normal(key=proj_key, shape=c_proj.weight.shape)
            * 0.02
            / math.sqrt(2 * config.n_layer),
        )

        if config.dtype == "bfloat16":
            self.c_attn = convert_bfloat16(c_attn)
            self.c_proj = convert_bfloat16(c_proj)
        else:
            self.c_attn = c_attn
            self.c_proj = c_proj

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
        c_fc = nn.Linear(
            config.n_embd, config.n_embd * 4, use_bias=config.bias, key=fc_key
        )
        c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, use_bias=config.bias, key=proj_key
        )

        if config.dtype == "bfloat16":
            self.c_fc = convert_bfloat16(c_fc)
            self.c_proj = convert_bfloat16(c_proj)
        else:
            self.c_fc = c_fc
            self.c_proj = c_proj

        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x, *, key: jrandom.PRNGKey = None, inference: bool = None):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key, inference=inference)

        return x


class Block(eqx.Module):

    config: GPTConfig = eqx.static_field()
    ln_1: nn.LayerNorm
    ln_2: nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey) -> None:
        attn_key, mlp_key = jrandom.split(key)
        self.config = config
        ln_1 = nn.LayerNorm(config.n_embd)
        ln_2 = nn.LayerNorm(config.n_embd)
        if config.dtype == "bfloat16":
            self.ln_1 = convert_bfloat16(ln_1)
            self.ln_2 = convert_bfloat16(ln_2)
        self.attn = CausalSelfAttention(config, key=attn_key)
        self.mlp = MLP(config, key=mlp_key)

    def __call__(self, x, *, key: jrandom.PRNGKey = None, inference: bool = None):
        key = None if key is None else jrandom.split(key, 1)[0]
        x = x + self.attn(self.ln_1(x), key=key, inference=inference)
        key = None if key is None else jrandom.split(key, 1)[0]
        x = x + jax.vmap(partial(self.mlp, key=key, inference=inference))(self.ln_2(x))
        return x


class GPT(eqx.Module):

    config: GPTConfig = eqx.static_field()
    lm_head: nn.Linear
    wte: nn.Embedding
    wpe: nn.Embedding
    drop: nn.Dropout
    h: List[Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: GPTConfig, *, key: jrandom.PRNGKey):
        emb_key_1, emb_key_2, block_key, lin_key = jrandom.split(key, 4)
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        wte = init_weight(
            nn.Embedding(config.vocab_size, config.n_embd, key=emb_key_1), key=emb_key_1
        )
        wpe = init_weight(
            nn.Embedding(config.block_size, config.n_embd, key=emb_key_2), key=emb_key_2
        )
        lm_head = init_weight(
            nn.Linear(
                config.n_embd, config.vocab_size, use_bias=config.bias, key=lin_key
            ),
            key=lin_key,
        )
        ln_f = nn.LayerNorm(config.n_embd)
        if config.dtype == "bfloat16":
            self.wte = convert_bfloat16(wte)
            self.wpe = convert_bfloat16(wpe)
            self.lm_head = convert_bfloat16(lm_head)
            self.ln_f = convert_bfloat16(ln_f)
        else:
            self.wte = wte
            self.wpe = wpe
            self.lm_head = lm_head
            self.ln_f = ln_f
        self.drop = nn.Dropout(config.dropout)
        self.h = [
            Block(config, key=k) for k in jrandom.split(block_key, config.n_layer)
        ]

    def forward(
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

        if target is not None:
            logits = jax.vmap(self.lm_head)(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, target.squeeze()
            )
        else:
            logits = self.lm_head(x[-1])
            loss = None

        return loss

    def configure_optimizers(
        self,
        model,
        weight_decay,
        learning_rate,
        betas,
        decay_lr=None,
        warmup_iters=None,
        lr_decay_iters=None,
        min_lr=None,
    ):
        """Equinox will partition PyTree later"""

        if decay_lr:
            assert (
                warmup_iters is not None
                and lr_decay_iters is not None
                and min_lr is not None
            )
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_iters,
                decay_steps=lr_decay_iters,
                end_value=min_lr,
            )

        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate,
                b1=betas[0],
                b2=betas[1],
                weight_decay=decay,
            )

        def where_is_linear_weight(tree: GPT, return_replace=True):
            all = [tree.lm_head.weight]
            for block in tree.h:
                all.append(block.mlp.c_fc.weight)
                all.append(block.mlp.c_proj.weight)
                all.append(block.attn.c_attn.weight)
                all.append(block.attn.c_proj.weight)
            if return_replace:
                return ("decay",) * len(all)
            else:
                return all

        param_labels = jtu.tree_map(lambda _: "no_decay", model)
        param_labels = eqx.tree_at(
            where=partial(where_is_linear_weight, return_replace=False),
            pytree=param_labels,
            replace=where_is_linear_weight(param_labels),
        )

        return optax.multi_transform(
            transforms={
                "decay": get_optimizer(decay=weight_decay),
                "no_decay": get_optimizer(decay=0),
            },
            param_labels=param_labels,
        )
