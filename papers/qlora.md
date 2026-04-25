---
slug: qlora
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
authors:
  - Tim Dettmers
  - Artidoro Pagnoni
  - Ari Holtzman
  - Luke Zettlemoyer
venue: NeurIPS
year: 2023
arxiv_id: "2305.14314"
doi: null
url: https://arxiv.org/abs/2305.14314
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: parameter-efficient
  secondary:
    - efficient-inference
    - alignment
tags:
  - qlora
  - 4-bit
  - nf4
  - double-quantization
  - paged-optimizer
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microqlora
  target_path: 02-alignment/microqlora.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microqlora.py
    script_slug: microqlora
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: lora
---

## TL;DR

QLoRA fine-tunes a 4-bit quantized base model with full-precision LoRA adapters, combining a new 4-bit data type, double quantization, and paged optimizer states to fit a 65B-parameter model on a single 48 GB GPU without quality loss.

## Problem

LoRA cut the parameter count of fine-tuning, but the frozen base model still had to live in GPU memory at full or half precision. For a 65B-parameter model that meant roughly 130 GB of weights, far beyond a single consumer or workstation GPU. Existing 8-bit and 4-bit quantization schemes (LLM.int8, GPTQ) targeted inference, not fine-tuning, and did not preserve quality when gradients flowed through the quantized weights. The paper asks whether 4-bit quantization can be combined with parameter-efficient fine-tuning to run training on a single device while matching 16-bit fine-tuning quality.

## Contribution

QLoRA introduces three innovations and combines them with LoRA adapters. First, NF4 (4-bit NormalFloat), a quantization data type optimized for the empirically observed approximately-normal distribution of pretrained weights; the quantization levels are chosen to be the quantiles of a standard normal so each level is equally likely. Second, double quantization, which quantizes the per-block quantization constants themselves to save additional memory at negligible quality cost. Third, paged optimizers, which use unified memory paging to handle gradient checkpointing memory spikes without out-of-memory errors. The base model is stored in NF4 and frozen; LoRA adapters in BFloat16 are added to every linear layer; gradients flow through the dequantized weights to the adapters. The paper trains Guanaco, a chat model, this way and shows that QLoRA matches 16-bit full fine-tuning across model scales and tasks.

## Method summary

- Quantize each linear layer's weight matrix into NF4 in blocks of 64 elements; store one Float32 (or quantized) absmax per block.
- Apply double quantization: quantize the absmaxes themselves to 8-bit with another set of constants. Memory cost per parameter drops from 4.5 bits to about 4.1 bits.
- Wrap each linear layer so the forward pass dequantizes weights on the fly to the compute dtype (BFloat16) and computes (W_dequant + B·A)·x with LoRA adapters A, B.
- Train only the LoRA adapters; freeze the quantized base.
- Use paged optimizer states (CUDA unified memory) so gradient checkpointing spikes are absorbed by host memory rather than crashing the run.
- Optionally dequantize and merge adapters at the end to produce a 16-bit adapted model.

## Key results

QLoRA matches 16-bit full fine-tuning on the GLUE benchmark and on instruction-following evaluations. Guanaco-65B trained with QLoRA on a single 48 GB GPU reaches 99% of ChatGPT performance on the Vicuna evaluation. Memory savings: a 65B-parameter model that requires ~780 GB for 16-bit Adam fine-tuning fits in under 48 GB with QLoRA. The paper also gives ablations isolating each contribution: NF4 outperforms FP4, double quantization saves an additional ~3 GB at no quality cost, and paged optimizers eliminate out-of-memory failures during training.

## Relation to existing work

QLoRA combines LoRA (Hu et al. 2021) with the quantization line that includes LLM.int8 (Dettmers et al. 2022) and GPTQ (Frantar et al. 2022). The key shift is to use quantization as a memory-saving primitive during training rather than only at inference, and to design the quantization data type (NF4) for the actual weight distribution rather than for a generic uniform grid. Subsequent work — IR-QLoRA, LoftQ, QA-LoRA — refines initialization and quantization-error compensation; the QLoRA recipe remains the dominant single-GPU large-model fine-tuning workflow.

## Implementation notes

A pedagogical version can simulate 4-bit quantization on a small model: define an NF4 quantize/dequantize pair, apply it to the linear weights at load time, store per-block absmaxes, and add a small LoRA adapter as in the LoRA card. The toy script does not need real CUDA paged memory; it can omit the optimizer-paging step and still demonstrate the quantize-dequantize-with-adapter pattern. Key pitfalls: dequantizing inside the forward pass on every step (production uses fused kernels; the pedagogical version pays the cost), and forgetting that gradients only flow to A and B because the dequantized W has no requires_grad. A useful ablation shows model perplexity for FP16, NF4-without-adapters, and NF4 + LoRA — the third should match the first while the second visibly degrades.

## Open questions

The paper does not analyze how NF4 interacts with extreme-scale models trained with newer optimizers, or with mixture-of-experts architectures whose weight distributions differ from dense transformer weights. Activation quantization remains out of scope; QLoRA quantizes only weights.
