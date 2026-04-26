---
slug: vae
title: "Auto-Encoding Variational Bayes"
authors:
  - Diederik P. Kingma
  - Max Welling
venue: ICLR
year: 2014
arxiv_id: "1312.6114"
doi: null
url: https://arxiv.org/abs/1312.6114
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - vae
  - variational-inference
  - reparameterization-trick
  - elbo
  - latent-variable
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microvae
  target_path: 01-foundations/microvae.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microvae.py
    script_slug: microvae
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The Variational Autoencoder pairs an encoder network that outputs a Gaussian posterior over a latent code with a decoder that reconstructs data from the code, trained by maximizing the evidence lower bound using a reparameterization trick that lets gradients flow through the latent sample.

## Problem

Latent-variable generative models — z drawn from a prior, x drawn from a likelihood p(x|z) — are flexible but require either an intractable posterior p(z|x) for inference or a stochastic approximation that does not admit gradient-based learning. Mean-field variational inference fits a tractable q(z|x) by minimizing KL divergence, but the resulting bound's gradient with respect to the encoder requires sampling z from q, and the standard score-function gradient estimator has high variance. Earlier work (wake-sleep, Helmholtz machines) trained encoder and decoder networks but with separate updates that did not jointly optimize a single objective.

## Contribution

The paper introduces two ideas. First, the variational autoencoder framework: parameterize q(z|x) with an encoder neural network that outputs the mean and variance of a Gaussian, parameterize p(x|z) with a decoder, and maximize the evidence lower bound E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z)) jointly with respect to both networks. Second, the reparameterization trick: rewrite z = μ_φ(x) + σ_φ(x) ⊙ ε where ε ~ N(0, I), so the sampling becomes a deterministic function of x and an independent noise source. The gradient of the bound with respect to φ now flows through μ and σ via standard backpropagation, with much lower variance than the score-function estimator. The combination makes amortized variational inference practical for large datasets.

## Method summary

- Encoder network: x → (μ_φ(x), σ_φ(x)), parameterizing q_φ(z|x) = N(z; μ_φ(x), diag(σ_φ(x)²)).
- Reparameterized sample: z = μ_φ(x) + σ_φ(x) ⊙ ε, ε ~ N(0, I).
- Decoder network: z → distribution parameters for p_θ(x|z) (Bernoulli for binary pixels, Gaussian for continuous, etc.).
- Loss per example: L = -E_{ε}[log p_θ(x|z)] + KL(q_φ(z|x) || N(0, I)). The KL is closed-form for Gaussians.
- Train both networks by minimizing L over minibatches with standard SGD or Adam.
- At inference for generation, draw z ~ N(0, I) and run the decoder. For inference of latent given observed x, run the encoder.

## Key results

The paper demonstrates VAEs on MNIST and Frey Faces, showing that the reparameterized estimator produces lower-variance gradients than the score-function alternative, that the resulting model produces coherent samples by decoding random latents, and that the bound improves smoothly during training. The numerical scale is small by modern standards, but the framework's contribution is conceptual: amortized variational inference with backprop-friendly stochasticity. Subsequent VAEs scaled to high-resolution images (NVAE, VQ-VAE), discrete latents, hierarchical priors, and remain a backbone of large-scale generative pipelines (Stable Diffusion's encoder-decoder is a VAE).

## Relation to existing work

VAEs replaced or extended several earlier approaches: wake-sleep training of Helmholtz machines, Restricted Boltzmann Machines, and standard variational inference with mean-field approximations. They are closely related to autoencoders but differ in training objective (probabilistic ELBO vs. reconstruction-only) and sampling property (decoder produces a generative distribution vs. a single point). GANs (Goodfellow et al. 2014) are a contemporary alternative that produces sharper images but no tractable likelihood; VAEs and GANs were merged in many hybrid models. Diffusion models (Ho et al. 2020) can be viewed as a stack of denoising VAEs and now dominate image generation, but the latent-VAE pattern persists wherever a learned compressed representation is needed.

## Implementation notes

A pedagogical script can train a tiny VAE on MNIST with a small MLP encoder and decoder. Minimum viable trainer: encoder outputs μ and log-variance (more numerically stable than σ directly), reparameterized sample, decoder outputs Bernoulli logits, loss = binary-cross-entropy reconstruction + analytic KL term. Pitfalls: outputting σ instead of log σ² (negative values silently produce NaNs in the KL); forgetting that the KL term is per-element and should be summed across the latent dimension; weighting the reconstruction and KL terms differently without realizing that the relative scale changes the latent's information content (β-VAE makes this weight an explicit hyperparameter). Useful diagnostic: plot the per-example KL and reconstruction terms separately — if KL collapses to zero, the decoder is ignoring the latent (posterior collapse), a known failure mode.

## Open questions

The paper does not address posterior collapse (the decoder ignores the latent and the encoder collapses to the prior), which is a well-known failure mode in expressive autoregressive decoders. It also assumes a Gaussian variational family; richer families (normalizing flows, mixtures) appear in later work.
