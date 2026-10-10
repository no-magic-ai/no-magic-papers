---
slug: self-consistency
title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
authors:
  - Xuezhi Wang
  - Jason Wei
  - Dale Schuurmans
  - Quoc Le
  - Ed H. Chi
  - Sharan Narang
  - Aakanksha Chowdhery
  - Denny Zhou
venue: ICLR
year: 2023
arxiv_id: "2203.11171"
doi: null
url: https://arxiv.org/abs/2203.11171
discovered_via: maintainer
discovered_date: 2026-10-10
status: implemented
themes:
  primary: reasoning
  secondary: []
tags:
  - chain-of-thought
  - self-consistency
  - decoding
  - majority-vote
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microcot
  target_path: 03-systems/microcot.py
  target_tier: 03-systems
  batch_label: t3-gap-fill
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microcot.py
    script_slug: microcot
    commit: null
    release: null
    media_repo: no-magic-viz
    media_status: linked
    scene_path: scenes/scene_microcot.py
    preview_path: previews/microcot.gif
    media_note: null
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: chain-of-thought
---

## TL;DR

Instead of decoding one chain of thought greedily, sample several reasoning paths from the same prompted model, read each final answer, and return the answer most paths agree on. The vote marginalizes out the reasoning path; nothing extra is trained.

## Problem

Chain-of-thought prompting is decoded greedily, so one wrong step in the single decoded path decides the answer. Earlier remedies train an extra verifier or re-ranker (Sec. 1). Can decoding exploit that hard problems usually admit several reasoning routes to one correct answer?

## Contribution

- A decoding strategy for chain-of-thought prompting: sample diverse reasoning paths r_i with answers a_i, then take the most consistent answer, argmax_a Σ_i 1(a_i = a) (Sec. 2).
- A comparison of aggregations (Table 1): majority vote, and weighted sums or averages using each path's probability, unnormalized or length-normalized by Eq. 1, P(r_i, a_i | prompt, question) = exp((1/K) Σ_k log P(t_k | prompt, question, t_1 … t_(k−1))).
- Gains on arithmetic and commonsense benchmarks across four language models, with analyses of sampling, path count, robustness and diversity (Sec. 3).

## Method summary

- Prompt a frozen language model with few-shot chain-of-thought exemplars.
- Sample m outputs (temperature 0.5 or 0.7, top-k 40 where used; Sec. 3.1); a task-specific parser reads the answer, for arithmetic the first number after "The answer is" (footnote 1).
- Aggregate over a fixed answer set (Sec. 2). The majority vote and the normalized weighted sum score almost identically because the normalized path probabilities are nearly equal; footnote 2 adds that this also means the model is "not well calibrated". Unnormalized sums and weighted averages do worse (Table 1).
- Diversity of the sampled paths is the key: beam search, which yields less diverse paths, helps less, and short equation-only paths leave "less opportunity" for diversity (Sec. 3.4–3.5).

## Key results

With PaLM 540B on GSM8K, accuracy rises from 56.5 (greedy chain of thought) to 74.4 with 40 sampled paths (+17.9; Tables 1–2, averaged over 10 runs); the abstract reports gains on SVAMP (+11.0), AQuA (+12.2), StrategyQA (+6.4) and ARC-challenge (+3.9). Most of the gain arrives with 5 to 10 paths (Fig. 2); each path costs a decode, and sampled paths can be incorrect or nonsensical (Sec. 5).

## Relation to existing work

It replaces the greedy decoding of chain-of-thought prompting (Wei et al., 2022) with sampling and marginalization. Unlike a trained verifier (Cobbe et al., 2021) or re-ranker (Thoppilan et al., 2022), it needs no extra training or model. Sample-and-rank, which keeps the sampled sequence with the highest log probability, also needs none but gains much less from the same samples (Sec. 3.4, Fig. 3). Supervised scratchpads (Nye et al., 2021) are a separate, training-based line.

## Implementation notes

`no-magic/03-systems/microcot.py` is a toy adaptation, not a reproduction. It trains two arms of one 17,345-parameter position-slot MLP (no attention) from one initialization on four-digit sums modulo 10: direct answers, and Nye-style written partial sums. It decodes the trace arm greedily and by self-consistency (40 samples at temperature 0.7, answer parsed after `#`, majority vote, ties by summed Eq. 1 weights then smallest digit, abstention scored wrong). The majority vote is the gated result; the other four non-greedy Table 1 aggregations are report-only.

Deviations: no pretrained model, prompting or top-k. Path diversity is built into the training data (three fixed operand orders announced by a route token the model must emit), not sampled free-form reasoning. Eq. 1 uses temperature-1 probabilities; the tie rule is the program's own; 40 samples come from one run.

History: a first single-route design was recorded as a design no-go (the vote tied greedy at 0.904; a pre-registered trace-use criterion scored 0.351 against 0.80). The multi-route idea arose only after that result, so the shipped design was pre-registered and evaluated once on a fresh multiset partition; its defaults sit at the edge of a development grid, and all results are one frozen seed under CPython 3.12.8.

At seed 42 the vote beats greedy by +0.104 (exact McNemar p = 8.5e-09), partly because greedy picks its route poorly; against the best forced route, chosen after seeing the test slice, the margin is +0.068. The vote costs 40 decodes per problem and saturates near k = 20 (0.810, against 0.806 at k = 40). Trace versus direct (+0.052, p = 0.083, unequal budgets) is not claimed, faithfulness probes show only partial dependence on written steps, and the test slice leans toward repeated digits. `commit` stays `null`; no optional lesson is planned.

## Further reading

- Wei et al. (2022), the prompting method this decoder extends; see the `chain-of-thought` card.
- Nye et al., "Show Your Work: Scratchpads for Intermediate Computation with Language Models" (2021), arXiv:2112.00114, the lineage of the toy's trace arm.
