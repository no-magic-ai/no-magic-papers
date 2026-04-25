---
slug: react
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
authors:
  - Shunyu Yao
  - Jeffrey Zhao
  - Dian Yu
  - Nan Du
  - Izhak Shafran
  - Karthik Narasimhan
  - Yuan Cao
venue: ICLR
year: 2023
arxiv_id: "2210.03629"
doi: null
url: https://arxiv.org/abs/2210.03629
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: agents
  secondary:
    - reasoning
tags:
  - tool-use
  - prompting
  - action-loop
  - hotpotqa
  - alfworld
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microreact
  target_path: 04-agents/microreact.py
  target_tier: 04-agents
  batch_label: v3-backfill-agents
  review_date: null
implementations:
  - repo: no-magic
    path: 04-agents/microreact.py
    script_slug: microreact
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

ReAct interleaves chain-of-thought reasoning with explicit tool calls in a single decoding loop, letting one language model think, act, observe, and revise its plan inside one trajectory.

## Problem

Reasoning-only prompting such as chain of thought lets a model deliberate but not consult the world, so it confabulates facts that a quick lookup would settle. Action-only prompting lets a model call tools but offers no scratchpad for planning, error recovery, or strategy selection. Decision-making benchmarks like ALFWorld and WebShop showed that pure imitation learning and pure chain-of-thought both plateaued because neither thinking alone nor acting alone covers the full agent loop.

## Contribution

The paper proposes a prompting protocol where the model alternates between Thought, Action, and Observation steps. Thoughts are free-form natural language; actions are structured tool calls drawn from a small task-specific vocabulary; observations are the tool's responses, fed back into the prompt before the next thought. The same language model produces both thoughts and actions through ordinary next-token prediction; no separate planner or controller is introduced. The paper applies the loop to knowledge-intensive question answering with a Wikipedia search/lookup/finish action set and to interactive decision-making with environment-specific actions.

## Method summary

- Few-shot exemplars demonstrate trajectories that interleave Thought, Action, and Observation lines using a fixed surface grammar.
- Decoding generates until an Action line is emitted; the runner pauses generation, executes the action against the tool or environment, appends the resulting Observation, and resumes.
- Action vocabularies are kept small and task-specific: search, lookup, finish for Wikipedia question answering; navigate, pick, place, examine, and similar verbs for ALFWorld.
- A trajectory ends when the model emits a designated finish action carrying the answer or commit signal.
- For interactive tasks, ReAct-IM combines ReAct trajectories with imitation learning to improve sample efficiency without changing the loop.
- Baselines include Act-only (same prompt minus the Thought lines) and CoT-only (no actions, internal reasoning only) on the same backbone.

## Key results

On HotpotQA, ReAct with PaLM-540B reaches 35.1 exact-match versus 29.4 for CoT and 25.7 for Act-only; combining ReAct with self-consistent CoT lifts the result further. On FEVER, ReAct beats CoT by roughly 4.5 points. On ALFWorld, ReAct prompting outperforms BUTLER, an imitation-trained baseline, and outperforms CoT-only by a wide margin (around 71% vs 22% success). On WebShop, ReAct more than doubles Act-only success on the same backbone. The paper shows that the gains hold across PaLM and GPT-3-class models.

## Relation to existing work

ReAct sits between Chain-of-Thought (Wei et al. 2022) and tool-using language models such as WebGPT and SayCan. Where CoT keeps reasoning entirely internal, ReAct externalizes the parts of reasoning that need the world. Where prior tool-use work treated planning and acting as separate components, ReAct shows that a single language model trained for next-token prediction can host both as alternating spans in one trajectory. The paper foreshadows agent stacks that adopt the thought-action-observation loop as a primitive, including subsequent extensions like Reflexion, Tree-of-Thoughts, and most LangChain-style agent frameworks.

## Implementation notes

A pedagogical version needs only a thought-action-observation parser and a tiny tool registry. The toy script can use a deterministic local search function over a hand-coded knowledge base instead of Wikipedia, so the loop runs without network access. Hyperparameters that matter: maximum trajectory length to prevent loops, low decoding temperature for stable parsing, and the few-shot exemplar count (the paper uses six for question answering). Common pitfalls: parsing must reject malformed actions and feed an error back as the Observation rather than crashing — the recovery behavior is part of what ReAct demonstrates. Also instructive to include a CoT-only and Act-only ablation in the same script so the value of the alternation is visible side by side.

## Open questions

The paper does not isolate how much of the gain comes from the thought channel versus the action channel as a function of model scale. Subsequent work extends the loop with explicit self-critique (Reflexion) and with branching search over thought-action trees (Tree-of-Thoughts), suggesting that the linear loop in ReAct is one point on a wider design surface.
