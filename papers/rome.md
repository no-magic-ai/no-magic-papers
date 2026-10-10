---
slug: rome
title: "Locating and Editing Factual Associations in GPT"
authors:
  - Kevin Meng
  - David Bau
  - Alex Andonian
  - Yonatan Belinkov
venue: NeurIPS
year: 2022
arxiv_id: "2202.05262"
doi: null
url: https://arxiv.org/abs/2202.05262
discovered_via: maintainer
discovered_date: 2026-10-10
status: implemented
themes:
  primary: interpretability
  secondary:
    - alignment
tags:
  - knowledge-editing
  - causal-tracing
  - rank-one-update
  - associative-memory
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microrome
  target_path: 02-alignment/microrome.py
  target_tier: 02-alignment
  batch_label: t3-gap-fill
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microrome.py
    script_slug: microrome
    commit: null
    release: null
    media_repo: no-magic-viz
    media_status: linked
    scene_path: scenes/scene_microrome.py
    preview_path: previews/microrome.gif
    media_note: null
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

ROME (Rank-One Model Editing) changes one factual association in a trained GPT with a closed-form rank-one update of one mid-layer MLP output matrix, treated as a linear key-value memory: a new key-value pair is inserted while the least-squares fit to all other keys is kept.

## Problem

Where does a GPT-style transformer store a fact such as (Space Needle, located in, Seattle), and can that one association be changed directly, without fine-tuning and without disturbing related facts?

## Contribution

The paper introduces causal tracing (§2.1): run a prompt clean, then with noise on the subject's embeddings, then with noise while restoring one internal state to its clean value; the recovered probability of the correct object is that state's indirect effect. In GPT-2 XL, strongly causal states appear at mid-layer MLPs at the last subject token and at attention near the final token (§2.2). On that basis it proposes ROME (§3.1) and the CounterFact benchmark of difficult counterfactual edits (§3.3).

## Method summary

- An MLP output matrix W is viewed as a linear associative memory W K ≈ V (§3.1). Inserting a key k* with value v* while minimizing ‖ŴK − V‖ gives Ŵ = W + Λ (C⁻¹k*)ᵀ with C = K Kᵀ and Λ = (v* − W k*) / ((C⁻¹k*)ᵀ k*) (Eq. 2). App. A derives it from a constrained least-squares Lagrangian (Eq. 5–17).
- The key k* is the MLP key after the nonlinearity at the last subject token, averaged over N sampled text prefixes (Eq. 3).
- The value v* is found by optimizing a vector z that replaces the MLP output at that token, so that the new object becomes likely after each prefix, plus a KL "essence drift" term that keeps the prediction for a prompt like "{subject} is a" in place (Eq. 4).
- App. E.5 (GPT-2 XL): layer 18; C from 100,000 hidden-state samples over Wikipedia tokens; 20 prefixes; Adam lr 0.5, weight decay 1.5×10⁻³, KL factor 1×10², ≤ 20 steps, early stop at L(z) = 5×10⁻².

## Key results

On CounterFact with GPT-2 XL (7,500 records, Table 4), ROME reached Score 89.2 (harmonic mean of ES, PS and NS), with efficacy 100.0, paraphrase 96.4 and neighborhood 75.4 (unedited model: 78.1). Fine-tuning reached efficacy 100.0 and paraphrase 87.9, but neighborhood 40.4. On GPT-J, ROME's Score was 91.5. On zsRE (Table 1), ROME had 99.8 efficacy and 88.1 paraphrase accuracy. Editing each layer and token in turn (Fig. 5) worked best at mid-layer MLPs at the last subject token. Limits (§3.7): one fact per edit, directional associations, and an edited model may guess plausible new facts with no basis in evidence.

## Relation to existing work

The rank-one update follows the constrained least-squares rule of Bau et al. (2020) for editing image generators. Knowledge Editor (KE) and MEND learn hypernetworks that predict weight changes; Knowledge Neurons edits the MLP rows of attributed neurons; fine-tuning, with or without a norm constraint, is the baseline. MEMIT, a later paper, extends ROME to many edits at once.

## Implementation notes

`no-magic/02-alignment/microrome.py` is a pedagogical adaptation, not a reproduction. It trains a 3,936-parameter decoder from random weights on 24 invented subjects, each with one of 8 cities and one of 4 categories: one bias-free MLP (d = 16, H = 64), one attention head queried from the last position, no normalization, and a softmax restricted to the answer type. It then edits six subjects' cities one at a time on fresh copies of W_proj. k* averages the key over 20 prefixes of 0–3 filler tokens; v* comes from Eq. 4 with λ = 100, lr 0.1, L2 decay on z in the gradient only, at most 60 steps and early stop 0.05; C is the uncentered second moment of keys at every position of 1,500 sampled prompts. At runtime the program checks rank(ΔW) = 1, Ŵk* = v*, ΔW = Λuᵀ, Cu = k* and that no other tensor changes.

Deviations: single-token subjects, no attention before the MLP and no layer norm, so the causal trace and a wrong-token control show localization the architecture imposes; they illustrate the paper's finding and are not evidence for it. The authors' released code differs from the paper text (kl_factor 0.0625, a perturbation δ = z − z₀ with in-loss weight decay, a norm clamp); the program follows the paper. Evaluation uses small fixed prompt sets and ES/PS/NS comparisons without magnitudes, fluency or generation; a C = I control is reported only. Efficacy is seed-sensitive (in the design study the 0.95 threshold held on 6 of 17 seeds), and the reference output is bound to CPython 3.12.8. No GPT-2 XL, zsRE or CounterFact result is claimed. The `commit` field above is intentionally null. No release is assigned, and no optional lesson is planned.
