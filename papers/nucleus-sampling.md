---
slug: nucleus-sampling
title: "The Curious Case of Neural Text Degeneration"
authors:
  - Ari Holtzman
  - Jan Buys
  - Li Du
  - Maxwell Forbes
  - Yejin Choi
venue: ICLR
year: 2020
arxiv_id: "1904.09751"
doi: null
url: https://arxiv.org/abs/1904.09751
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - nucleus-sampling
  - top-p
  - decoding
  - text-degeneration
  - beam-search
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microbeam
  target_path: 03-systems/microbeam.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microbeam.py
    script_slug: microbeam
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The paper diagnoses why beam-search and likelihood-maximizing decoding produce repetitive, degenerate text from strong language models, and introduces nucleus (top-p) sampling: at each step, sample from the smallest set of tokens whose cumulative probability exceeds p, dynamically truncating the unreliable tail of the distribution.

## Problem

Likelihood-maximizing decoding strategies — greedy and beam search — were the default for sequence generation, inherited from machine translation where they had worked well. Applied to large open-ended language models like GPT-2, the same strategies produced text that was bland, repetitive, and frequently looped on a single phrase forever. The natural alternative — sampling proportional to the model's distribution — produced incoherent text because the long unreliable tail of the distribution accumulates substantial probability mass and occasionally produces gibberish tokens that derail subsequent generation. Existing sample-with-truncation methods (top-k sampling) capped the candidate set at a fixed k regardless of how peaked the distribution was, mistruncating both peaked and flat distributions.

## Contribution

The paper makes two contributions. First, a diagnosis: it shows that maximum-likelihood-like decoding lands in low-probability regions of the natural-language distribution because human text is itself surprising — the model assigns highest probability to text humans rarely produce. The metric is statistical: human text has a self-BLEU and entropy distribution that beam-search output does not match. Second, nucleus sampling (top-p): at each generation step, sort tokens by probability, take the smallest prefix whose cumulative probability ≥ p, renormalize over that prefix, and sample. The set size adapts to the distribution's shape: peaked distributions sample from very few tokens, flat ones from many. The paper shows nucleus sampling produces text whose statistical signatures match human text far better than greedy, beam, top-k, or full-distribution sampling.

## Method summary

- At each generation step compute the model's per-token probability distribution P over the vocabulary.
- Sort tokens by descending probability; let p_1 ≥ p_2 ≥ ... ≥ p_V be the sorted probabilities.
- Find the smallest k* such that Σ_{i=1}^{k*} p_i ≥ p (the nucleus threshold; the paper uses p = 0.9 to 0.95 for GPT-2 generation).
- Form the truncated distribution P' by zeroing out tokens beyond rank k* and renormalizing the remaining probabilities to sum to 1.
- Sample the next token from P'; append, repeat.
- The companion baselines compared in the paper: greedy (argmax), beam search at various widths, full-distribution sampling, and top-k sampling at various k.

## Key results

The paper measures human-likeness of generated text using statistical signatures (Zipfian rank-frequency distributions, repetition rates, perplexity under the source model, self-BLEU). Nucleus sampling matches human distributions far more closely than competing strategies on GPT-2-small and GPT-2-large generations. Beam search and greedy show characteristic looping and over-frequency of short common tokens; top-k sampling produces incoherence on flat distributions where the cap truncates too early. Human evaluation also rates nucleus-sampled continuations as more interesting and more like real text than the alternatives.

## Relation to existing work

Top-k sampling (Fan et al. 2018) is the immediate predecessor; nucleus sampling generalizes by making the truncation adaptive to the distribution's mass rather than its rank count. Beam search, the prior dominant method for translation and summarization, remains useful in those settings where the conditional distribution is more sharply peaked. Subsequent decoding work — typical sampling (Meister et al. 2022), eta sampling (Hewitt et al. 2022), min-p sampling (Minh et al. 2024), DRY repetition penalties — refines the truncation/penalty design. The paper's core diagnosis (human text is statistically high-entropy and likelihood-maximization undershoots that entropy) frames the entire post-2020 decoding literature.

## Implementation notes

A pedagogical decoding script can implement greedy, beam, top-k, and nucleus side by side on a small language model. Minimum viable nucleus implementation: get model logits, softmax, sort with stable indices, cumulative sum, find first index where cumsum ≥ p, mask out the rest, renormalize, sample with multinomial. Pitfalls: forgetting to renormalize after truncation (the post-truncation probabilities won't sum to 1); applying nucleus to logits instead of probabilities; using p too high (degenerates toward full-distribution sampling and reintroduces incoherence) or too low (degenerates toward greedy and reintroduces looping). Useful comparison: generate from the same prompt with each strategy and observe the qualitative differences — looping in greedy/beam, gibberish bursts in unrestricted sampling, coherent diverse text from nucleus.

## Open questions

The optimal p value is task-dependent and not predicted by the paper; subsequent work explored adaptive schemes. The paper does not address how decoding interacts with downstream metrics like factuality, where likelihood-maximizing strategies are sometimes preferable. Modern decoding for instruction-tuned chat models often combines lower temperatures with nucleus or with constrained-sampling techniques.
