---
slug: moe-shazeer
title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
authors:
  - Noam Shazeer
  - Azalia Mirhoseini
  - Krzysztof Maziarz
  - Andy Davis
  - Quoc Le
  - Geoffrey Hinton
  - Jeff Dean
venue: ICLR
year: 2017
arxiv_id: "1701.06538"
doi: null
url: https://arxiv.org/abs/1701.06538
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary:
    - efficient-inference
tags:
  - moe
  - sparse-gating
  - top-k-routing
  - load-balancing
  - conditional-computation
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micromoe
  target_path: 02-alignment/micromoe.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/micromoe.py
    script_slug: micromoe
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The paper introduces the sparsely-gated mixture-of-experts layer: a bank of expert subnetworks with a learned gating function that routes each input to a small subset of experts, scaling parameter count by orders of magnitude while keeping per-example compute roughly constant.

## Problem

Conventional dense neural networks pay compute proportional to parameter count: doubling parameters doubles FLOPs. To keep training tractable, model size had been bounded by available compute, leaving most parameters underused for any given input. Earlier conditional-computation work proposed gating mechanisms but ran into three problems: gradients did not flow cleanly through hard routing decisions, expert utilization collapsed onto a few favorites, and existing batch parallelism patterns broke when different examples activated different parameters.

## Contribution

Shazeer et al. design a layer that addresses all three. The gating function emits a soft top-K distribution: a learned linear projection plus tunable Gaussian noise, followed by keep-top-K and softmax. K is small (typically 2 to 4) out of hundreds or thousands of experts, so per-example compute is small. Two auxiliary losses prevent collapse: an importance loss penalizes the variance of total gate weights across the batch, and a load-balancing loss penalizes the variance of the number of examples each expert receives. The paper also gives a concrete distributed-computation pattern: the gate sorts inputs by chosen expert, dispatches them as a contiguous batch to that expert's device, then reassembles outputs. The full model interposes the MoE layer between LSTM stacks; total parameter counts reach 137 billion at the time of publication.

## Method summary

- Define E expert subnetworks (the paper uses two-layer feedforward experts) and a learned gating projection W_g.
- For input x, compute logits g(x) = W_g·x + Normal(0, softplus(W_noise·x)); the noise is data-dependent and helps load-balancing during training.
- KeepTopK(g, K) zeros all but the largest K entries; softmax over the surviving entries gives the gate weights.
- The layer output is Σ_i softmax(g)_i · expert_i(x), summed only over the K surviving experts.
- Add two auxiliary losses: L_importance = (CV(Σ_batch gate))² and L_load = (CV(load))², each scaled by a small coefficient. CV is the coefficient of variation.
- Distribute experts across devices; route each example's chosen experts using the dispatch-gather pattern.

## Key results

The paper trains language models with up to 137B parameters. On the One-Billion-Word benchmark, MoE language models with K=4 outperform dense LSTM baselines at the same compute budget by a substantial perplexity margin and reach state-of-the-art at the time. Translation experiments on WMT show similar gains: a 32-expert MoE model matches a deep dense ensemble while using less compute. The auxiliary losses are critical; without them expert utilization collapses onto a small favored subset.

## Relation to existing work

The paper builds on classical mixture-of-experts (Jacobs et al. 1991) and on conditional-computation work (Bengio et al. 2013, Davis & Arel 2013). Its contributions are the sparse top-K gate, the load-balancing losses, and the engineering pattern for distributed routing. It is the direct ancestor of GShard, Switch Transformer (Fedus et al. 2021, which simplifies to top-1 gating), GLaM, Mixtral, and DeepSeek-V3's MoE design. Modern designs vary the gate (top-K, expert-choice, hash routing), the load-balance term (auxiliary loss vs auxiliary-free balancing), and the expert capacity policy, but the conceptual frame from this paper carries through.

## Implementation notes

A pedagogical script can use a small number of tiny feedforward experts (E = 4 to 8, K = 2) inside a small transformer. The minimum viable layer is a learned gate, top-K + softmax, expert MLPs, and a sum weighted by gate values. Pitfalls: forgetting the auxiliary load-balancing loss collapses experts within a few hundred steps; using a hard top-K without softmax-over-survivors is non-differentiable through the kept indices; failing to scale gradients to experts by their gate weight assigns them all-or-nothing gradient even with soft routing. A useful diagnostic: log per-expert example count per batch and watch the load-balancing loss act on the variance.

## Open questions

The auxiliary losses are heuristic and require coefficient tuning. Subsequent work explores auxiliary-loss-free balancing and routing schemes that avoid the hard top-K bottleneck. The paper does not address training instability that arises at very large expert counts, which Switch Transformer and later work addressed.
