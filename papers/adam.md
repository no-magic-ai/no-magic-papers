---
slug: adam
title: "Adam: A Method for Stochastic Optimization"
authors:
  - Diederik P. Kingma
  - Jimmy Ba
venue: ICLR
year: 2015
arxiv_id: "1412.6980"
doi: null
url: https://arxiv.org/abs/1412.6980
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: training-dynamics
  secondary: []
tags:
  - adam
  - adaptive-learning-rate
  - momentum
  - optimization
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microoptimizer
  target_path: 01-foundations/microoptimizer.py
  target_tier: 01-foundations
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microoptimizer.py
    script_slug: microoptimizer
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
  - repo: no-magic
    path: 02-alignment/adam_vs_sgd.py
    script_slug: adam_vs_sgd
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Adam combines momentum (an exponential moving average of gradients) with RMSProp-style per-parameter step sizes (an exponential moving average of squared gradients), adds bias correction for the early steps, and produces an optimizer that requires almost no per-task tuning while reaching strong performance across many objectives.

## Problem

Stochastic gradient descent with a single global learning rate works in principle but requires careful tuning per problem and per architecture. Earlier adaptive methods — AdaGrad accumulates squared gradients indefinitely and so its effective rate decays to zero; RMSProp uses an exponential moving average of squared gradients but is unpublished folklore at the time and lacks bias correction. Practitioners wanted a method that combined the variance-reduction benefits of momentum with per-parameter scaling, with reasonable defaults that did not require hand-tuning per task.

## Contribution

The paper integrates momentum and RMSProp into a single update rule, derives explicit bias-correction terms, proves convergence in convex settings under mild assumptions, and demonstrates strong empirical performance on logistic regression, multilayer networks, and convolutional networks for digit recognition. The bias correction is the small but critical piece: without it the early-step updates are scaled incorrectly, especially for β2 close to 1 where the second-moment estimate takes many steps to warm up. The paper also introduces AdaMax, a variant using the infinity norm instead of the L2 norm of the gradient.

## Method summary

- Maintain two exponential moving averages per parameter: m_t (first moment, gradient mean) and v_t (second moment, squared-gradient mean).
- Update at step t: m_t = β1·m_{t-1} + (1-β1)·g_t and v_t = β2·v_{t-1} + (1-β2)·g_t².
- Apply bias correction: m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t). Without this, the early steps are biased toward zero because the moving averages are initialized at zero.
- Update parameters: θ_t = θ_{t-1} - lr · m̂_t / (sqrt(v̂_t) + ε).
- Defaults that work surprisingly well across tasks: β1 = 0.9, β2 = 0.999, ε = 1e-8, lr = 1e-3.

## Key results

On logistic regression on IMDB and MNIST, Adam matches or outperforms AdaGrad and SGD with momentum across learning rates. On a feedforward network on MNIST and a convnet on CIFAR-10, Adam reaches the same loss in fewer iterations than the baselines. The empirical claim that propagated most widely was that Adam's defaults work well out-of-the-box on a broad range of problems, dramatically reducing the per-problem hyperparameter search.

## Relation to existing work

Adam composes ideas from momentum (Polyak 1964; Nesterov 1983), AdaGrad (Duchi et al. 2011), RMSProp (Tieleman & Hinton, unpublished course slides 2012), and AdaDelta (Zeiler 2012). Subsequent work refined Adam in several directions: AdamW (Loshchilov & Hutter 2019) decouples weight decay from the gradient term, fixing an interaction that had silently weakened L2 regularization; AMSGrad (Reddi et al. 2018) addresses a convergence-counterexample by maintaining a max of past v_t; LAMB and Adafactor scale Adam to very large batch sizes and parameter counts. AdamW is the dominant choice for modern transformer training.

## Implementation notes

A pedagogical optimizer-comparison script can implement SGD, momentum, RMSProp, and Adam side-by-side and run them on the same loss landscape. Minimum viable Adam: maintain m and v per parameter, apply the bias-correction divisors at each step, do the divide-by-sqrt update. Pitfalls: omitting bias correction (training appears slow for the first few hundred steps); applying weight decay as L2 inside the gradient rather than via AdamW's decoupled form (the effective regularization scales inverse-proportionally to gradient magnitude); using the same default ε on very small-magnitude problems where ε dominates the denominator. A useful diagnostic visualizes per-parameter step sizes — Adam's per-parameter scaling is what gives it robustness across heterogeneous gradient scales, and seeing this on a contrived heterogeneous loss makes the mechanism concrete.

## Open questions

Adam's convergence proof has known counterexamples in the original analysis (corrected by AMSGrad). Empirical performance versus carefully tuned SGD with momentum remains a debate, especially in vision: Adam is often the first choice but tuned SGD-momentum can win on some architectures. The choice between Adam and AdamW for any task that uses weight decay is not optional — they differ meaningfully — and many published experiments accidentally use the wrong variant.
