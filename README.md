# peekllm

An interactive playground that runs in the browser, built around a small
GPT style language model trained from scratch, visualizing what the model
is doing token by token (probability heatmap, attention patterns,
calibration, training curves), instead of just claiming "it works."

## Status

Early stage. Working through the model implementation first
(`training/`), following *Build a Large Language Model (From Scratch)* by
Sebastian Raschka, reimplemented in JAX. The browser frontend (`web/`) comes
later, once there's a trained model to export and serve.

## Repo layout

* `training/`: JAX implementation of the GPT model: tokenizer, attention,
  training loop, evaluation and calibration.
* `web/`: (not yet created) React and TypeScript playground frontend,
  added once v1 (text input plus token probability heatmap) is ready to
  build.

## Compute strategy

Developed and debugged locally on an M1 Pro using JAX on CPU with tiny toy
configs (JAX's Metal/GPU backend on Apple Silicon is experimental and not
reliable enough for this). The actual training run happens on a rented CUDA
GPU, where JAX's support is solid.
