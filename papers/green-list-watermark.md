---
slug: green-list-watermark
title: "A Watermark for Large Language Models"
authors:
  - John Kirchenbauer
  - Jonas Geiping
  - Yuxin Wen
  - Jonathan Katz
  - Ian Miers
  - Tom Goldstein
venue: ICML
year: 2023
arxiv_id: "2301.10226"
doi: null
url: https://arxiv.org/abs/2301.10226
discovered_via: maintainer
discovered_date: 2026-10-10
status: implemented
themes:
  primary: safety-robustness
  secondary: []
tags:
  - watermarking
  - logit-bias
  - hypothesis-testing
  - decoding
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microwatermark
  target_path: 03-systems/microwatermark.py
  target_tier: 03-systems
  batch_label: t3-gap-fill
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microwatermark.py
    script_slug: microwatermark
    commit: null
    release: null
    media_repo: no-magic-viz
    media_status: linked
    scene_path: scenes/scene_microwatermark.py
    preview_path: previews/microwatermark.gif
    media_note: null
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

A language model's output can be watermarked at sampling time: a hash of the previous token picks a pseudorandom "green" fraction γ of the vocabulary, green logits get a bonus δ, and anyone who knows the hashing rule can test text for excess green tokens with a one-proportion z-test, without the model.

## Problem

How can a proprietary model's text be marked so that it is detectable from a short span of tokens, with negligible quality cost, without model or API access, and with a stated false-positive rate?

## Contribution

- A hard rule (Alg. 1, sample only green tokens) and a soft rule (Alg. 2, add δ to green logits), keyed on a hash of the previous token.
- A one-proportion z-test for detection (Eq. 2, Eq. 3).
- Spike entropy (Def. 4.1), a lower bound on the expected green count (Theorem 4.2) and a perplexity bound (Theorem 4.3).
- A private mode on a keyed pseudorandom function (Sec. 5, Alg. 3), OPT-1.3B experiments and an attack catalogue (Sec. 7).

## Method summary

- A hash of s^(t−1) seeds a generator that splits the vocabulary into a green list G of size γ|V| and a red list (Alg. 2); private mode keys it with F_K over h tokens (Sec. 5).
- Soft rule: add δ to every green logit, then sample. Hard rule: sample only from G.
- Detection: under the null hypothesis of text written without knowledge of the rule (Eq. 1), z = (|s|_G − γT) / √(Tγ(1 − γ)) for |s|_G green tokens among T (Eq. 3). Text is flagged at z > 4, a one-sided Gaussian tail of about 3 × 10⁻⁵. The test uses neither the model nor δ.
- With α = e^δ and average spike entropy S*, the expected green count is at least γαT S* / (1 + (α − 1)γ) (Theorem 4.2): low-entropy text carries little watermark (Sec. 1.2, Sec. 4).
- Repeated n-grams break Eq. 3's independence assumption; the remedy is a larger hash window or not counting repeated n-grams, which "can also make the detector more sensitive" (Sec. 4.1).

## Key results

On OPT-1.3B with C4 RealNewsLike prompts and T = 200 ± 5 tokens (Table 2), multinomial sampling at temperature 0.7 with δ = 2.0 detects 0.994 of watermarked texts at γ = 0.25 and 0.984 at γ = 0.5 (z = 4), with at most one false positive per run of about 500 texts. γ = 0.1 is Pareto-optimal for strength versus perplexity, and beam search adds strength cheaply (Sec. 6). Replacing spans with T5 lowers detection as the edit budget grows; paraphrasing, tokenization, homoglyph and "emoji" attacks are discussed (Sec. 7).

## Relation to existing work

Earlier text watermarks rewrote finished text through parse trees or synonym tables (Atallah et al., 2001; Topkara et al., 2006) and degraded quality; this scheme biases the model's own sampling and needs no model to decode. Aaronson (2022) announced a related biasing approach. Model-parameter watermarks target model stealing, and post-hoc detectors such as Zellers et al. (2019) need no key (Sec. 8).

## Implementation notes

`no-magic/03-systems/microwatermark.py` is a pedagogical adaptation, not a reproduction. It trains a 1,992-parameter two-token-context MLP on a 48-word synthetic grammar, samples 200-token texts with the soft rule (γ = 0.25, δ = 2.0) and, for comparison, the hard rule (γ = 0.5), and detects them with Eq. 3 from tokens and key alone.

Deviations: green lists come from a keyed `blake2b` of the previous token (h = 1) seeding Python's `random`, not a vetted PRF; keys are public demo constants. The first generated token is unscored (its list depends on the unseen prompt), so 15 of 15 green at γ = 0.5 gives z = 3.873. Alg. 2 does not fix where δ enters relative to temperature, and the paper samples at 0.7 (App. C); the toy adds δ after temperature (t = 1 primary), the other order is report-only, and the released code's order was not checked. Perplexity uses the same toy model. The headline detector counts each distinct (previous, token) pair once; plain Eq. 3 is uncalibrated on this repetitive text. Controls use 200 null keys; "human" text is grammar text; shuffled text is not an exact null.

The gates were redesigned after a first look at the same seeded data, so passing them is a consistency check, not confirmation. A dated amendment after the design run retired the failed W8b resolution and W8c inflated-bound checks (never re-scored), gated W12 on DEDUP only and made W10 conditional on the δ order; the retained bound check is non-tight. No OPT, C4 or Table 2 result; no perfect-detection, robustness or security claim; no rate transfers to real text. Output is bound to CPython 3.12.8. `commit` stays `null`; no optional lesson is planned.
