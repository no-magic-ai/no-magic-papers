---
slug: chain-of-thought
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
authors:
  - Jason Wei
  - Xuezhi Wang
  - Dale Schuurmans
  - Maarten Bosma
  - Brian Ichter
  - Fei Xia
  - Ed H. Chi
  - Quoc V. Le
  - Denny Zhou
venue: NeurIPS
year: 2022
arxiv_id: "2201.11903"
doi: null
url: https://arxiv.org/abs/2201.11903
discovered_via: maintainer
discovered_date: 2026-10-10
status: summarized
themes:
  primary: reasoning
  secondary: []
tags:
  - chain-of-thought
  - prompting
  - in-context-learning
  - emergence
routing:
  decision: reference-only
  target_repo: null
  target_script_slug: null
  target_path: null
  target_tier: null
  batch_label: t3-gap-fill
  review_date: null
implementations: []
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Writing a few worked examples with intermediate reasoning steps into the prompt makes sufficiently large language models produce their own step-by-step reasoning before answering, and that alone sharply improves multi-step arithmetic, commonsense and symbolic reasoning, without any fine-tuning.

## Problem

Scaling model size alone did not yield strong performance on multi-step reasoning, and teaching intermediate steps by training or fine-tuning needs costly rationale datasets per task (Sec. 1). Can in-context few-shot prompting elicit intermediate reasoning instead?

## Contribution

- Chain-of-thought prompting: few-shot exemplars of the form (input, chain of thought, output), where the chain is a series of natural-language intermediate steps (Sec. 2).
- Evidence on arithmetic, commonsense and symbolic reasoning across GPT-3, LaMDA, PaLM, UL2 and Codex that the benefit emerges with scale (Sec. 3–5).
- Ablations separating the effect of reasoning in language from equation-only output, extra tokens of computation and post-answer explanations (Sec. 3.3).

## Method summary

- Augment each of a handful of few-shot exemplars with a hand-written chain of thought leading to the answer; the same eight exemplars serve most arithmetic benchmarks.
- Decode the model's response greedily; it writes its own chain and then the answer.
- The paper lists four properties: problems decompose into steps, the chain gives an interpretable window into behavior, it applies to any task humans solve in language, and it needs only prompting (Sec. 2).

## Key results

With PaLM 540B, GSM8K accuracy rises from 17.9 with standard prompting to 56.9 with chain-of-thought prompting, above the prior best of 55 from finetuned GPT-3 with a verifier. The gains are an emergent ability of model scale: they appear only at about 100B parameters, and smaller models "produced fluent but illogical chains of thought" (Sec. 3.2). On GSM8K, equation-only prompts and variable-compute dots recover little of the gain, although equation-only output helps on easier one- or two-step datasets (Sec. 3.3). No language models were finetuned (Sec. 6).

## Relation to existing work

It extends rationale-augmented training (Ling et al., 2017; Cobbe et al., 2021) and scratchpads (Nye et al., 2021), which teach intermediate steps by training or fine-tuning, to few-shot prompting (Brown et al., 2020). Self-consistency (Wang et al., 2023) later replaces its greedy decoding with sampling and a majority vote over answers.

## Implementation notes

No no-magic program implements this paper. A stdlib toy cannot reproduce few-shot prompting of a frozen model of about 100B parameters, and the paper itself reports that small models fail at it.

`no-magic/03-systems/microcot.py`, owned by the `self-consistency` card, is related but different: it trains a 17,345-parameter position-slot MLP on supervised traces of written partial sums (the scratchpad lineage of Nye et al.), with three fixed reasoning routes built into the data, and compares that trace arm with a direct-answer arm before applying self-consistency decoding. Its traces are generated token by token by the trained model, not prompted, and it makes no emergence, faithful-reasoning or general-reasoning claim. This card is reference-only and has no planned lesson.

## Further reading

- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2023); see the `self-consistency` card.
- Nye et al., "Show Your Work: Scratchpads for Intermediate Computation with Language Models" (2021), arXiv:2112.00114.
