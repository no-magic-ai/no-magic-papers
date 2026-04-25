---
slug: uct
title: "Bandit Based Monte-Carlo Planning"
authors:
  - Levente Kocsis
  - Csaba Szepesvári
venue: ECML
year: 2006
arxiv_id: null
doi: "10.1007/11871842_29"
url: https://link.springer.com/chapter/10.1007/11871842_29
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: agents
  secondary:
    - reasoning
tags:
  - mcts
  - uct
  - bandits
  - planning
  - tree-search
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micromcts
  target_path: 04-agents/micromcts.py
  target_tier: 04-agents
  batch_label: v3-backfill-agents
  review_date: null
implementations:
  - repo: no-magic
    path: 04-agents/micromcts.py
    script_slug: micromcts
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

UCT applies the UCB1 bandit rule recursively at every node of a search tree, turning Monte-Carlo rollouts into a planning algorithm whose value estimates converge to the minimax solution as the number of simulations grows.

## Problem

Classical game-tree planners such as alpha-beta search require an evaluation function and a tractable branching factor; both fail in domains like Go where the branching factor is large and no strong static evaluator exists. Earlier Monte-Carlo planners ran random rollouts and averaged returns, but they spent simulations uniformly across moves and converged slowly because they could not preferentially deepen promising lines. The paper asks how to allocate simulations adaptively while keeping the consistency guarantees of Monte-Carlo estimation.

## Contribution

The paper introduces UCT — Upper Confidence Bounds applied to Trees — a tree-search algorithm that treats the choice of child at each internal node as an independent multi-armed bandit problem and selects children using the UCB1 rule. The asymptotic analysis shows that the value estimate at the root converges to the optimal value as the number of simulations goes to infinity, and that the failure probability of selecting a suboptimal action at the root drops at a polynomial rate. The construction is general; it makes no assumption about the domain beyond the ability to simulate trajectories and observe a terminal reward.

## Method summary

- A search tree grows online from the current state. Each node stores a visit count and an estimated value.
- Each simulation starts at the root and proceeds in four phases: selection, expansion, rollout, and backup.
- Selection: at every internal node, the child maximizing the UCB1 score Q(s,a) + c · sqrt(ln N(s) / N(s,a)) is chosen, where Q is the average return through that child, N is the visit count, and c is the exploration constant.
- Expansion: when the selection reaches a node with unexplored children, one new child is added to the tree.
- Rollout: from the new node, a default policy (typically uniform random) plays out a trajectory until a terminal state and observes a reward.
- Backup: the observed reward is propagated back up the path, incrementing visit counts and updating average returns at every visited node.
- After a budget of simulations, the action at the root with the highest visit count (or highest average value) is returned.

## Key results

The paper proves that under standard assumptions on rewards, the value estimate at any node of the tree converges to the optimal value, and that the regret of the algorithm at the root is bounded. Empirically, it shows that UCT outperforms alpha-beta search with a fixed depth and uniform-rollout Monte-Carlo planning on Sailing, P-Game, and small Go boards. Subsequent independent work, most prominently the AlphaGo line, used UCT-style selection as the planning backbone, with neural networks replacing the random rollout and the action prior.

## Relation to existing work

UCT inherits its selection rule from UCB1 (Auer, Cesa-Bianchi, Fischer 2002), which solves the stationary multi-armed bandit problem. It generalizes earlier flat Monte-Carlo planners by replacing uniform sampling with bandit-guided sampling at every node, and it generalizes alpha-beta search by replacing exhaustive enumeration with selective deepening guided by sample returns. The algorithm is also closely related to sparse-sampling planners, but it does not require a fixed sampling depth or width.

## Implementation notes

A pedagogical version needs only a node class with visit count and value estimate, a recursive tree-policy selection that applies the UCB1 rule, and a random rollout. A small turn-based game such as Tic-Tac-Toe or Connect-Four is enough to make the convergence visible. The exploration constant c (the paper uses sqrt(2)) trades exploration against exploitation; sweeping c is a useful pedagogical exercise. Pitfalls: forgetting to negate values for adversarial games (each ply alternates the maximizing player), and using value-based action selection at the end instead of visit-count selection — the latter is more robust to noisy estimates. The script can also expose the four phases as separate functions to make the algorithm structure legible.

## Open questions

UCT's regret bounds depend on assumptions that are violated in many real domains, including unbounded variance and non-stationary returns. The paper does not address how the tree should be reused across moves, how to combine UCT with learned value functions, or how to handle simultaneous moves; later work (notably the AlphaGo line) addresses each of these.

## Further reading

- Auer, Cesa-Bianchi, Fischer (2002) — UCB1, the bandit rule UCT inherits.
- Coulom (2006) — concurrent introduction of Monte-Carlo Tree Search with a different selection rule.
