# Continuous Autoregressive Language Model POC

This workspace contains a small, practical proof of concept for the idea behind a continuous autoregressive language model.

## What it is

A standard LLM predicts the next token from a sequence of previous tokens. A continuous autoregressive language model extends that idea by working in a continuous vector space:

- tokens are first mapped to dense embeddings;
- the model predicts the next embedding or next-token distribution step by step;
- generation is autoregressive: each new step depends on the previously generated context.

In practice, this is the core idea behind many modern sequence models, and it is a good starting point for research into latent, continuous, or diffusion-style language generation.

## Why this POC matters

This demo is intentionally small so you can understand the mechanics:

1. Convert text into tokens.
2. Learn dense embeddings.
3. Train an autoregressive transformer to predict the next token.
4. Also train the model to predict continuous next-token embeddings.
5. Generate new text from a seed prompt using both discrete and continuous decoding.

It is not a production-grade GPT, but it is a valuable learning and demonstration project.

## Project files

- `demo_continuous_autoregressive.py` — runnable demo that trains a small transformer autoregressive model and demonstrates two generation paths.
- `requirements.txt` — dependencies to install.

## Run it

```bash
cd /workspace
make install
make run
```

If you are using the devcontainer, open the repository in Codespaces and the dependencies will be installed automatically.

## Run with helper script

```bash
./run_demo.sh
```

## Possible extensions

1. Replace the tiny character-level model with a small transformer.
2. Use continuous latent vectors instead of discrete tokens.
3. Compare standard next-token prediction vs latent continuous generation.
4. Build a small web UI for interactive prompting.
