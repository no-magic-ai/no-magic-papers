---
slug: instructgpt
title: "Training language models to follow instructions with human feedback"
authors:
  - Long Ouyang
  - Jeff Wu
  - Xu Jiang
  - Diogo Almeida
  - Carroll L. Wainwright
  - Pamela Mishkin
  - Chong Zhang
  - Sandhini Agarwal
  - Katarina Slama
  - Alex Ray
  - John Schulman
  - Jacob Hilton
  - Fraser Kelton
  - Luke Miller
  - Maddie Simens
  - Amanda Askell
  - Peter Welinder
  - Paul Christiano
  - Jan Leike
  - Ryan Lowe
venue: NeurIPS
year: 2022
arxiv_id: "2203.02155"
doi: null
url: https://arxiv.org/abs/2203.02155
discovered_via: maintainer
discovered_date: 2026-10-09
status: implemented
themes:
  primary: alignment
  secondary: []
tags:
  - supervised-fine-tuning
  - instruction-following
  - rlhf
  - demonstrations
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microsft
  target_path: 02-alignment/microsft.py
  target_tier: 02-alignment
  batch_label: t3-gap-fill
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microsft.py
    script_slug: microsft
    commit: null
    release: null
    media_repo: no-magic-viz
    media_status: linked
    scene_path: scenes/scene_microsft.py
    preview_path: previews/microsft.gif
    media_note: null
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: ppo
---

## TL;DR

InstructGPT turns a pretrained GPT-3 into an instruction follower in three stages: supervised fine-tuning (SFT) on human demonstrations, a reward model trained on human rankings, and PPO against that reward. The `microsft` program implements only the first stage, at toy scale.

## Problem

Language models pretrained to predict Internet text are not trained to do what a user asks. They can produce untruthful, toxic or unhelpful output, because the next-token objective differs from following instructions helpfully and safely. The paper asks how far fine-tuning on human feedback can close that gap for prompts submitted to the OpenAI API.

## Contribution

The paper describes a full pipeline for fine-tuning GPT-3 models of 1.3B, 6B and 175B parameters with human demonstrations and comparisons, together with an evaluation by labelers on the same prompt distribution. It shows that a much smaller fine-tuned model can be preferred to a much larger model that was only pretrained.

## Method summary

- Step 1 (§3.1, Figure 2; §3.5 "Supervised fine-tuning"): labelers write demonstrations of the desired behavior for prompts, and a pretrained GPT-3 is fine-tuned on them with supervised learning. The SFT dataset has about 13k training prompts. The paper trains for 16 epochs with cosine learning-rate decay and residual dropout 0.2. It selects the final SFT model by reward-model score on the validation set, noting that validation loss overfits after one epoch.
- Step 2: labelers rank several model outputs per prompt, and a reward model is trained on these comparisons. The paper's Eq. 1 is this ranking loss; it is not an SFT loss.
- Step 3: the SFT model is optimized against the reward model with PPO, with a per-token KL penalty toward the SFT model. The PPO-ptx variant also mixes in pretraining gradients.

## Key results

On the paper's test prompts, labelers preferred outputs of the 1.3B InstructGPT model to those of the 175B GPT-3, despite over 100x fewer parameters. Outputs of the 175B InstructGPT were preferred to 175B GPT-3 outputs 85 ± 3% of the time. The paper also reports gains in truthfulness, less toxic output, and minimal performance regressions on public NLP datasets for the PPO-ptx models. These results belong to the paper's full pipeline at GPT-3 scale; the SFT stage alone does not account for them.

## Relation to existing work

The pipeline follows earlier work on learning from human preferences (Christiano et al., 2017; Stiennon et al., 2020) and uses PPO (Schulman et al., 2017) for step 3. DPO later replaces steps 2 and 3 with one supervised preference loss and uses an SFT model as its reference policy. In no-magic, `microppo.py` covers the reward-model and PPO stages, and `microdpo.py` covers DPO.

## Implementation notes

`no-magic/02-alignment/microsft.py` is a toy adaptation of step 1 only. It pretrains a 1,120-parameter one-layer, one-head character decoder (embedding 8, feed-forward 32, context 10, 17 tokens, no biases, parameter-free RMSNorm) on eight cyclic strings over `a`–`h`. Pretraining runs 200 Adam updates, each on one sampled string. The same learned weights then receive 300 full-parameter SFT updates on 14 synthetic demonstrations of the form `copy:a> → a` and `next:a> → b`. Each update averages the loss over all 14 pairs, and Adam's moments, not the weights, are reset between the stages. Two prompts, `copy:h>` and `next:h>`, are held out and reported only. Answers are decoded greedily from the model's logits; the program never looks up the demonstration function at inference time.

This response-only loss is a teaching choice, not prescribed by the paper; masked prompt tokens still receive gradients through attention. The demonstrations are synthetic, with no reward model, PPO or human evaluation. Full-batch SFT was selected after single-pair updates failed the seed-42 `next` check (3/7; threshold 4/7). A post-hoc training-prompt comparison paired the same learned base per seed over 0–15: 14 full-batch versus 4 single-pair passes; two full-batch seeds fail. The seed-42 full-batch outcome (copy 6/7, next 5/7) was known before selection. Full batching also changes Adam's moment normalization and raises SFT sequence-gradient exposure from 300 to 4,200 (14x), with pretraining unchanged. The comparison cannot isolate batching, establish necessity or generalization, or claim preregistration or paper fidelity.

The program's acceptance criteria are checked on training data only: at least 5% lower base-corpus loss after pretraining, at least 5% lower response loss after SFT than the learned base, changed parameters in both stages, and at least 4 of 7 exact training answers, including the end boundary, for each command. The `commit` field above is intentionally null. No release is assigned, and no optional lesson is planned.
