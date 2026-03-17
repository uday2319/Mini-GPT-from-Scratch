# Mini-GPT from Scratch

This project implements a GPT-style Transformer language model from scratch using PyTorch. It covers the full pipeline: tokenization,Attention mechanism, Resiudal connection ,Normalization model architecture, training, and text generation.

---
## Overview

The goal of this project is to understand how modern language models work internally by building a minimal GPT-like model without relying on high-level libraries.

The model learns to predict the next token in a sequence using a causal (autoregressive) Transformer decoder.
---
## Architecture

The model follows a standard GPT pipeline:

Text → Tokenizer → Embedding → Transformer Blocks → Linear Projection → Next Token

Each Transformer block consists of:

* Layer Normalization
* Multi-Head Self-Attention
* Feedforward Network (GELU activation)
* Residual Connections

Self-attention allows each token to attend to previous tokens, enabling context-aware predictions.
---

## Project Structure
```
Mini-GPT
│
├── model/
│   ├── gpt_model.py
│   ├── transformer_block.py
│   ├── attention.py
│   ├── feedforward.py
│   └── config.py
│
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer.py
│
├── training/
│   ├── dataset.py
│   └── train.py
│
├── data/
│   └── dataset.txt
│
├── generate.py
├── README.md
└── .gitignore
```

---

## Configuration

Example configuration used for training:

```
vocab_size = 5000
block_size = 128

embed_dim = 256
heads = 8
layers = 4

batch_size = 16
learning_rate = 3e-4
max_iters = 10000
```
---
## Dataset
The model is trained on the Tiny Shakespeare dataset (~1M characters).
This dataset is suitable for learning grammar, structure, and dialogue patterns.
--

## Training

used colab to train significantly faster on GPU .

Typical loss progression:
```
Initial loss: ~7.0
Mid training: ~3.0
Final loss: ~0.4 – 2.5
```
---
The model generates text autoregressively, predicting one token at a time.
---
## Key Concepts Implemented

* Tokenization using Byte Pair Encoding (BPE)
* Embedding and positional encoding
* Multi-head self-attention
* Feedforward neural networks
* Residual connections and layer normalization
* Cross-entropy loss for language modeling
* Autoregressive text generation
---
## Limitations

* Small dataset limits generalization
* Token splitting may occur due to limited vocabulary size
* Model capacity is relatively small compared to modern LLMs

---

## Possible Improvements

* Increase vocabulary size and dataset scale
* Add top-k and temperature sampling
* Implement top-p (nucleus) sampling
* Add checkpoint saving and resume training
* Train deeper and wider models
* Add evaluation metrics (perplexity)
* Build an interactive interface for generation

---

## Summary

This project demonstrates how GPT-style models work internally by implementing every component from scratch. It provides a practical understanding of Transformers, attention mechanisms, and language model training.

---
