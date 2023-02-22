import os
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from model import GPT, GPTConfig


out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2"

dataset = "shakespeare"
batch_size = 15
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
bias = False
dropout = 0.0

learning_rate = 6e-4
max_iters = 6e6
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 6e5
min_lr = 6e-5

dtype = "bfloat16"

max_new_tokens = 100
temperature = 0.8
top_k = 200

model_key, train_key, eval_key = jrandom.split(jrandom.PRNGKey(seed=123), 3)

data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i : i + block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int32)] for i in ix)
    return jnp.array(x), jnp.array(y)


iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["meta_vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} inside meta path {meta_path}")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    dtype=dtype,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, key=model_key)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # TODO need implement
    raise NotImplementedError
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # TODO need implement
    raise NotImplementedError
else:
    raise ValueError(f"init_from={init_from} not supported")


optimizer = model.configure_optimizers(
    model=model,
    weight_decay=weight_decay,
    betas=[beta1, beta2],
    learning_rate=learning_rate,
    decay_lr=decay_lr,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_value_and_grad
def compute_loss(model: GPT, inputs, labels, *, key: jrandom.PRNGKey):
    batch_size = batch[0].shape[0]
    key = jrandom.split(key, batch_size)
    loss = jax.vmap(model.forward)(inputs, labels, key=key)
    return jnp.mean(loss)


@eqx.filter_jit
def make_step(model: GPT, x, y, opt_state, key):
    loss, grads = compute_loss(model, x, y, key=key)
    updates, opt_state = optimizer.update(
        grads, opt_state, model
    )  # multi transform needs 3 params
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state


while True:

    batch = get_batch(split="train")
    model, loss, opt_state = make_step(
        model, batch[0], batch[1], opt_state, key=jrandom.fold_in(train_key, iter_num)
    )

    iter_num += 1

    if iter_num % log_interval == 0:
        lossf = loss.item()
        print(f"iter {iter_num} \t loss: {lossf:.4f}")

    if iter_num > max_iters:
        break
