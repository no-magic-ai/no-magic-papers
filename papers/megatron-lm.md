---
slug: megatron-lm
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
authors:
  - Mohammad Shoeybi
  - Mostofa Patwary
  - Raul Puri
  - Patrick LeGresley
  - Jared Casper
  - Bryan Catanzaro
venue: arXiv
year: 2019
arxiv_id: "1909.08053"
doi: null
url: https://arxiv.org/abs/1909.08053
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - tensor-parallelism
  - model-parallelism
  - megatron
  - transformer-training
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microparallel
  target_path: 03-systems/microparallel.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microparallel.py
    script_slug: microparallel
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: transformer
---

## TL;DR

Megatron-LM trains transformer language models with billions of parameters by partitioning the matrix multiplications inside attention and MLP layers across multiple GPUs (tensor parallelism), using a partitioning pattern that requires only two AllReduce operations per transformer layer.

## Problem

Transformer models had grown beyond what fit in single-GPU memory. Three forms of parallelism existed: data parallelism (replicate the model, split the batch — limited by per-GPU memory), pipeline parallelism (split layers across GPUs — introduces pipeline-bubble inefficiency), and tensor model parallelism (split each layer's matrix multiplies across GPUs — requires careful communication design to be efficient). Existing tensor-parallel approaches relied on heavy inter-GPU communication that limited scalability beyond a small number of GPUs. The paper asks whether a tensor-parallel scheme can scale efficiently to dozens of GPUs by minimizing the number of AllReduce operations per layer.

## Contribution

Megatron-LM introduces a tensor-parallel partitioning of transformer layers that requires only two AllReduce operations per transformer block: one in the self-attention sublayer, one in the feedforward sublayer. The pattern exploits the structure of a transformer layer. For the MLP — Y = ReLU(X · W_1) · W_2 — partition W_1 column-wise across N GPUs (each GPU computes a slice of the hidden dimension); the ReLU is element-wise so no communication is needed; partition W_2 row-wise (each GPU computes a partial sum of the output); a single AllReduce sums the partial sums. For self-attention, partition the QKV projection across heads (each GPU holds a contiguous subset of attention heads) and the output projection row-wise; one AllReduce sums the head outputs. The KV-cache for a given layer lives entirely on one GPU per head subset, so attention computation is local. The combination scales to 8-way tensor parallelism with > 75% scaling efficiency on the 8-GPU NVLink topology of the time, enabling 8.3-billion-parameter GPT-2 training on a single DGX-2.

## Method summary

- Partition the MLP first weight matrix W_1 column-wise: each of N GPUs holds W_1[:, i*h/N : (i+1)*h/N] where h is hidden size.
- Forward: each GPU computes a partial Y_partial = ReLU(X · W_1_local) — no communication, X is replicated.
- Partition W_2 row-wise: each GPU holds W_2[i*h/N : (i+1)*h/N, :].
- Each GPU computes its slice's contribution Y_partial · W_2_local.
- AllReduce the partial sums across GPUs to produce the final MLP output.
- Self-attention: partition Q, K, V projection matrices across heads (column-wise on a per-head basis); each GPU computes attention for its assigned heads locally including the KV-cache; output projection W_o is row-wise; AllReduce sums across GPUs.
- Two AllReduces per transformer layer total. The pattern composes with data parallelism (orthogonal axis) and with pipeline parallelism (different layers on different stages); 3D parallelism — data + tensor + pipeline — combines all three.

## Key results

The paper trains an 8.3-billion-parameter GPT-2-style language model with 8-way tensor parallelism on a single DGX-2 (16 V100s with NVLink), reaching 76% scaling efficiency from 1 to 8 GPUs — meaning each added GPU contributes roughly 0.76× of an additional GPU's worth of throughput. The same code trains a 3.9-billion-parameter BERT model with substantial throughput gains. The paper documents how communication scales sublinearly because the AllReduce volumes grow linearly with hidden size while compute grows quadratically, so larger hidden sizes amortize communication better.

## Relation to existing work

Megatron-LM sits beside other parallelism families: data parallelism (Goyal et al. 2017 large-batch training), pipeline parallelism (GPipe by Huang et al. 2018, PipeDream by Narayanan et al. 2019), and earlier tensor-parallel work (Mesh-TensorFlow by Shazeer et al. 2018, which provided the abstraction layer but did not optimize transformer-specific patterns). The 3D-parallelism combination — Megatron tensor parallelism + GPipe-style pipeline + data parallelism — became the standard recipe for very large model training (Megatron-Turing NLG 530B, BLOOM, Llama). Subsequent work refined the same primitives: ZeRO (Rajbhandari et al. 2020) shards optimizer states, gradients, and parameters across data-parallel ranks; sequence parallelism (Korthikanti et al. 2022) extends Megatron's tensor partitioning to layer-norm and dropout activations; FSDP (Fully Sharded Data Parallel) generalizes ZeRO into mainstream frameworks.

## Implementation notes

A pedagogical script can simulate tensor parallelism on a single device by partitioning weight matrices manually and inserting fake AllReduce calls (just sums across the partitions). Minimum viable demo: a small MLP, partition W_1 column-wise and W_2 row-wise, compute partials per partition, AllReduce-sum, verify the result equals the unpartitioned forward. Pitfalls: forgetting that the input X is replicated (each partition uses the full X, only the weights are split); placing the AllReduce in the wrong place (between W_1 and W_2 when there should be no communication there); using row-wise partition where column-wise is correct (the AllReduce is over a different axis). The same script can demonstrate pipeline parallelism by sequentially passing activations through stage-partitioned layers.

## Open questions

The paper does not address sequence parallelism or activation sharding for very long sequences; subsequent work fills these gaps. Communication-vs-compute trade-offs change with newer interconnects (NVLink-NVSwitch, InfiniBand, host-side interconnect); modern very-large-scale training requires tuning that the paper does not cover.
