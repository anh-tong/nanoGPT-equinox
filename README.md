
This is a version of GPT model using [Equinox](https://github.com/patrick-kidger/equinox) in JAX. This repo is inspired from

- [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [https://github.com/cgarciae/nanoGPT-jax](https://github.com/cgarciae/nanoGPT-jax)

## Scale-up

This project initially aims for a simple implementation. As a nature of LLMs, scaling is one of the importants aspects. After doing a quick look over the internet, there are not many clear and concise implementations, tutorials focusing on JAX. Howerver, learning resources are really available with some starting points:

- [JAX documentation for multiple GPUs and hosts](https://jax.readthedocs.io/en/latest/multi_process.html)
- [Equinox filter pmap](https://docs.kidger.site/equinox/api/filtering/transformations/#equinox.filter_pmap)