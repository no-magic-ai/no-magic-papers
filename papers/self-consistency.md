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

Instead of decoding one chain of thought greedily, sample several reasoning paths from the same prompted model, read each final answer, and return the answer most paths agree on. The vote marginalizes out the reasoning path and needs no extra training, verifier or reward model.

## Problem

Chain-of-thought prompting is decoded greedily, so a single wrong step in the one decoded path decides the answer. Sampling-and-ranking or beam search either need extra models or produce paths that are too similar. Can decoding exploit the fact that a hard problem usually admits several different ways of reasoning to one correct answer (Sec. 1)?

## Contribution

- A decoding strategy for chain-of-thought prompting: sample diverse reasoning paths r_i with answers a_i, then take the most consistent answer, argmax_a Σ_i 1(a_i = a) (Sec. 2).
- A comparison of aggregations (Table 1): majority vote, and weighted sums or averages using each path's probability, unnormalized or length-normalized by Eq. 1, P(r_i, a_i | prompt, question) = exp((1/K) Σ_k log P(t_k | prompt, question, t_1 … t_(k−1))).
- Large gains on arithmetic and commonsense benchmarks across four language models, with analyses of sampling schemes, number of paths, robustness and the role of diversity (Sec. 3).

## Method summary

- Prompt a frozen language model with few-shot chain-of-thought exemplars.
- Sample m outputs (temperature 0.5 or 0.7, top-k 40 where used; Sec. 3.1); a task-specific parser reads the answer, for arithmetic the first number after "The answer is" (footnote 1).
- Aggregate over a fixed answer set: the unweighted majority vote and the normalized weighted sum perform almost identically; the paper attributes the near-equal path probabilities to the model being "not well calibrated" (footnote 2). The method applies only where the final answer comes from a fixed answer set (Sec. 2).
- Diversity of the sampled paths is the key: beam search, which yields less diverse paths, helps less, and short equation-only paths leave "less opportunity" for diversity (Sec. 3.4–3.5).

## Key results

With PaLM 540B on GSM8K, accuracy rises from 56.5 (greedy chain of thought) to 74.4 with 40 sampled paths (+17.9; Tables 1–2); the abstract reports gains on SVAMP (+11.0), AQuA (+12.2), StrategyQA (+6.4) and ARC-challenge (+3.9). Results average 10 runs of 40 samples. Most of the gain arrives with 5 to 10 paths (Fig. 2, Sec. 5); the cost is one decode per path, and sampled paths can still be incorrect or nonsensical (Sec. 5).

## Relation to existing work

It replaces the greedy decoding of chain-of-thought prompting (Wei et al., 2022) with sampling and marginalization. Unlike sample-and-rank or verifier approaches (Cobbe et al., 2021), it trains nothing extra, and unlike beam search it seeks diverse paths. It is a decoding strategy, not a training method; supervised scratchpads (Nye et al., 2021) and rationale fine-tuning are separate lines of work.

## Implementation notes

`no-magic/03-systems/microcot.py` is a toy adaptation, not a reproduction. It trains two arms of one 17,345-parameter position-slot MLP (no attention) from the same initialization on four-digit sums modulo 10: a direct-answer arm and a trace arm trained Nye-style on written partial sums. It decodes the trace arm greedily and by self-consistency (40 samples at temperature 0.7, answer parsed after `#`, majority vote, ties by summed Eq. 1 weights then the smallest digit, abstention scored wrong). All five non-greedy Table 1 aggregations are printed as report-only.

Deviations: no pretrained model, no prompting and no top-k cut. Path diversity is built into the training data: three fixed operand orders announced by a route token the model must emit, not free-form sampled reasoning. Eq. 1 uses temperature-1 probabilities; the tie rule is the program's own; 40 samples come from one run, not 10.

At seed 42 the vote beats greedy by +0.104 (exact McNemar p = 8.5e-09), but greedy picks its route poorly; against the best single forced route, chosen after seeing the test slice, the margin is +0.068. Trace versus direct (+0.052, p = 0.083, unequal budgets) is not claimed, faithfulness probes show only partial dependence on written steps, and the held-out slice leans toward repeated digits. An earlier single-route design tied greedy and was not shipped. Results are bound to CPython 3.12.8. `commit` stays `null`; no optional lesson is planned.

## Further reading

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022), the prompting method this decoder extends; see the `chain-of-thought` card.
- Nye et al., "Show Your Work: Scratchpads for Intermediate Computation with Language Models" (2021), arXiv:2112.00114, the supervised-trace lineage the toy's trace arm follows.
