"""
Sample from a trained model
"""
import os
import pickle
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import tiktoken
from model import GPT, GPTConfig


# -----------------------------------------------------------------------------
out_dir = "out"
start = "\n"  # or "<|endoftext|>" or whatever you like
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
dtype = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------
checkpoint_path = os.path.join(out_dir, "checkpoint")
checkpoint_file = os.path.join(checkpoint_path, "model.eqx")
checkpoint_params_file = os.path.join(checkpoint_path, "params.pkl")
with open(checkpoint_params_file, "rb") as f:
    checkpoint_params = pickle.load(f)


gptconf = GPTConfig(**checkpoint_params["model_args"])
model = GPT(gptconf, key=jrandom.PRNGKey(seed))
model = eqx.tree_deserialise_leaves(checkpoint_file, model)


# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    "config" in checkpoint_params and "dataset" in checkpoint_params["config"]
):  # older checkpoints might not have these...
    meta_path = Path("data", checkpoint_params["config"]["dataset"], "meta.pkl")
    load_meta = meta_path.exists()
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = jnp.array(start_ids, dtype=jnp.int32)[None]
key = jax.random.PRNGKey(seed)


def sample(model: GPT, token, key: jrandom.PRNGKey):
    generate_fn = partial(
        model.generate,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        temperature=temperature,
    )
    if token.shape[0] == 1:
        generated = generate_fn(token[0], key=key)
    else:
        key = jrandom.split(key, token.shape[0])
        generated = jax.vmap(generate_fn(token, key=key))
    return decode(generated)


start_ids = encode(start)
x = jnp.array(start_ids, dtype=jnp.int32)[None]
key = jrandom.PRNGKey(seed)

# run generation
for k in range(num_samples):
    step_key = jax.random.fold_in(key, k)
    sample_str = sample(model, x, key=key)
    print(sample_str)
    print("---------------")
