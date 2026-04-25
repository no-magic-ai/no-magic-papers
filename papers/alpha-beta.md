---
slug: alpha-beta
title: "An Analysis of Alpha-Beta Pruning"
authors:
  - Donald E. Knuth
  - Ronald W. Moore
venue: Artificial Intelligence
year: 1975
arxiv_id: null
doi: "10.1016/0004-3702(75)90019-3"
url: https://www.sciencedirect.com/science/article/pii/0004370275900193
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: agents
  secondary:
    - reasoning
tags:
  - game-tree
  - minimax
  - pruning
  - adversarial-search
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microminimax
  target_path: 04-agents/microminimax.py
  target_tier: 04-agents
  batch_label: v3-backfill-agents
  review_date: null
implementations:
  - repo: no-magic
    path: 04-agents/microminimax.py
    script_slug: microminimax
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Knuth and Moore give the first rigorous analysis of alpha-beta pruning, proving that with optimal move ordering it visits only O(b^(d/2)) nodes instead of the b^d that naive minimax would require, and characterizing exactly which subtrees the algorithm can safely skip.

## Problem

Minimax game-tree search dates to von Neumann and was used in the earliest chess programs (Shannon 1950, Samuel 1959), but the branching factor of any interesting game makes full enumeration impossible beyond very shallow depths. Alpha-beta pruning had been folklore in the AI community for over a decade — McCarthy described the idea around 1956 and several systems used it — but no one had proved exactly how much pruning was achievable, under what conditions, or which variants were correct. Programs were tuned without knowing whether their cutoffs respected the algorithm's invariants.

## Contribution

The paper formalizes alpha-beta as a recursive search algorithm parameterized by two bounds, alpha and beta, that represent the best score the maximizer can already guarantee and the best the minimizer can already guarantee at points along the path from the root. It proves three results that turned alpha-beta from a heuristic into an analyzed algorithm. First, alpha-beta returns the exact minimax value of the root, identical to plain minimax. Second, with perfect move ordering — the best move searched first at every node — alpha-beta examines O(b^(d/2)) leaves at depth d with branching factor b, an exponential improvement that doubles the searchable depth at fixed cost. Third, the paper categorizes the nodes of the game tree into types 1, 2, and 3 (PV, CUT, ALL) and proves that types 2 and 3 can be pruned with one move evaluated, while type 1 nodes (the principal variation) require all children. The paper also analyzes negamax, the symmetric reformulation that collapses min and max into a single recursion with sign flips.

## Method summary

- Recursive search with parameters (state, depth, α, β); on terminal or depth-zero, return the static evaluation.
- At a maximizing node, iterate children: recurse with (child, depth-1, α, β); update α to max(α, child_value); if α ≥ β, prune the remaining children (beta cutoff). Return α.
- At a minimizing node, mirror: update β to min(β, child_value); prune on β ≤ α (alpha cutoff). Return β.
- Negamax variant: every recursive call returns -search(child, depth-1, -β, -α); collapses the two cases into one.
- Move ordering heuristics — captures, killer moves, transposition-table best moves — push the strongest move to the front of the child list and approach the optimal-ordering O(b^(d/2)) bound.
- The paper analyzes both the worst case (no pruning) and the best case (perfect ordering), and gives bounds for random ordering as well.

## Key results

The principal theorem: with optimal move ordering, alpha-beta examines exactly the leaves of the type-1 and type-2 subtrees, which gives the O(b^(d/2)) leaf count. With random ordering, the expected leaf count is approximately O(b^(3d/4)). The paper gives precise constants and shows that the analysis matches earlier empirical observations from chess programs. A practical consequence: at fixed compute, perfect ordering doubles the search depth compared with plain minimax, which is the difference between a beginner and a strong tactical player in chess.

## Relation to existing work

Minimax itself dates to von Neumann's game theory and to Shannon's 1950 chess paper. Alpha-beta pruning was used in early AI systems by McCarthy and Samuel. Subsequent refinements include iterative deepening alpha-beta, principal-variation search, MTD(f), and Aspiration windows; all keep the same invariants Knuth and Moore proved here. Modern game-AI systems either combine alpha-beta with neural evaluation (Stockfish-NNUE) or replace it entirely with Monte-Carlo Tree Search (UCT and AlphaGo-line systems), but the analysis in this paper still defines what alpha-beta provably gives you.

## Implementation notes

A pedagogical script can use Tic-Tac-Toe or Connect-Four with depth-limited search. The minimal version is the negamax formulation: a single recursive function with sign flips at the recursive call. Pitfalls: alpha and beta must be passed by value, not mutated; the cutoff is α ≥ β. Compare plain minimax with alpha-beta on the same game and print node counts; the ratio at depth 4 or 5 shows the exponential gap. A useful extension is move-ordering — captures-first or last-best-move-first — to push toward the b^(d/2) limit. Transposition tables and iterative deepening are out of scope for a single-file version.

## Open questions

The paper assumes a static evaluation function exists at the depth horizon. Where it does not — Go and many imperfect-information games — alpha-beta degrades, motivating the Monte-Carlo line that culminates in UCT and AlphaGo.

## Further reading

- Shannon (1950) — "Programming a Computer for Playing Chess", the founding minimax paper.
- Kocsis & Szepesvári (2006) — UCT, the Monte-Carlo alternative when no static evaluator exists.
