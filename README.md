
This is a version of GPT model using [Equinox](https://github.com/patrick-kidger/equinox) in JAX. This repo is inspired from

- [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [https://github.com/cgarciae/nanoGPT-jax](https://github.com/cgarciae/nanoGPT-jax)

## Scale-up

This project initially aims for a simple implementation. As a nature of LLMs, scaling is one of the importants aspects. After doing a quick look over the internet, there are not many clear and concise implementations, tutorials focusing on JAX. Howerver, learning resources are really available with some starting points:

- [JAX documentation for multiple GPUs and hosts](https://jax.readthedocs.io/en/latest/multi_process.html)
- [Equinox filter pmap](https://docs.kidger.site/equinox/api/filtering/transformations/#equinox.filter_pmap)


UPDATE: multiple GPUs training is included! 

## Some small tricks and practice

In Equinox, if we optimize different groups with different optimizers using `optax`. We first need to assign labels for leaves of PyTree. The `optax.multi_transform` will using information of labels to decide which optimizers to choose. Here, we have two optimizers: having weight decay and not having weight decay.

Here is an example
```
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
```

To train model under floating point precision `bfloat16`, we may need to convert type of network weight

```
def convert_bfloat16(module: eqx.Module):
    module = eqx.tree_at(lambda l: l.weight, module, module.weight.astype(jnp.bfloat16))
    if hasattr(module, "bias"):
        if module.bias is not None:
            module = eqx.tree_at(
                lambda l: l.bias, module, module.bias.astype(jnp.bfloat16)
            )
    return module
```