---
slug: gan
title: "Generative Adversarial Networks"
authors:
  - Ian J. Goodfellow
  - Jean Pouget-Abadie
  - Mehdi Mirza
  - Bing Xu
  - David Warde-Farley
  - Sherjil Ozair
  - Aaron Courville
  - Yoshua Bengio
venue: NeurIPS
year: 2014
arxiv_id: "1406.2661"
doi: null
url: https://arxiv.org/abs/1406.2661
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - gan
  - adversarial
  - generative
  - minimax
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microgan
  target_path: 01-foundations/microgan.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microgan.py
    script_slug: microgan
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

GANs frame generative modeling as a two-player minimax game between a generator that maps noise to samples and a discriminator that classifies samples as real or fake; at equilibrium, the generator's distribution matches the data and the discriminator is at chance.

## Problem

Generative models before GANs faced a tradeoff between tractability and sample quality. Likelihood-based models (RBMs, autoregressive pixel models, variational autoencoders) optimized a tractable objective but produced blurry samples for natural images, because Gaussian or factorized observation models penalize sharp details that don't average correctly. Sampling-based models (Markov chain trainers) produced sharper images but required slow MCMC inference. The field needed a generative training procedure that could match a data distribution implicitly, without specifying the likelihood, while producing sharp samples.

## Contribution

The paper formalizes generative modeling as an adversarial game. A generator G maps a noise sample z (typically Gaussian) to a synthetic data point. A discriminator D maps a data point (real or synthetic) to a probability that it is real. The two are trained jointly: D maximizes the log probability of correctly classifying real and generated samples; G minimizes the log probability that D correctly identifies generated samples as fake. The paper proves that for fixed G the optimal D recovers a Jensen-Shannon-divergence-style score, and that the global optimum of the minimax game is reached when G's distribution equals the data distribution. The proof assumes infinite capacity and infinite data; the practical training procedure approximates it with neural networks and stochastic gradient steps.

## Method summary

- Generator G(z; θ_g): a feedforward network that maps a noise vector z ~ p_z (often N(0, I)) to a sample in data space.
- Discriminator D(x; θ_d): a feedforward network that maps a data point to a scalar probability of being real.
- Discriminator update: maximize E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))].
- Generator update: minimize E_{z ~ p_z}[log(1 - D(G(z)))], or equivalently maximize E_{z ~ p_z}[log D(G(z))] (the non-saturating form, which gives stronger gradients early when D wins easily).
- Alternate D and G updates per minibatch; the paper alternates k discriminator steps per generator step (k=1 is the common practical choice).
- At convergence the generator distribution matches the data distribution and D outputs 1/2 everywhere.

## Key results

The paper demonstrates GANs on MNIST, the Toronto Face Database, and CIFAR-10. Generated samples are crisper than contemporary VAE samples and qualitatively realistic. The paper does not claim quantitative state-of-the-art likelihoods (it cannot — GANs do not produce a tractable likelihood) but instead shows samples and a rough Parzen-window log-likelihood as a quality proxy. The contribution that propagated was the training-procedure framework, which subsequent work scaled (DCGAN, Progressive GAN, BigGAN, StyleGAN) into models capable of synthesizing photorealistic high-resolution images.

## Relation to existing work

GANs sit beside other generative families: variational autoencoders (Kingma & Welling 2014), autoregressive models (PixelRNN, PixelCNN), normalizing flows, and energy-based models. They differ in the training signal: a learned discriminator rather than a fixed likelihood. Subsequent refinements addressed three persistent problems. Mode collapse (G covers a few modes of p_data and ignores the rest) was attacked with minibatch discrimination, unrolled GANs, and Wasserstein objectives (Arjovsky et al. 2017). Training instability was addressed with spectral normalization, gradient penalties, and architectural choices like the DCGAN guidelines (Radford et al. 2016). Lack of an explicit latent inference path was addressed by BiGAN/ALI and by VAE-GAN hybrids. Diffusion models (Ho et al. 2020) eventually outperformed GANs on image generation quality and stability.

## Implementation notes

A pedagogical script can train a tiny GAN on MNIST or a 1-D Gaussian-mixture toy problem. Minimum viable trainer: define G and D as small MLPs, alternate updates with binary cross-entropy losses, sample noise per minibatch. Pitfalls: forgetting that D and G need separate optimizers and separate gradient updates; computing both losses inside one backward pass and contaminating gradients; using the saturating generator loss (log(1 - D(G(z)))) early in training when D dominates and gradients vanish — switch to the non-saturating form. Useful diagnostic: plot D(real) and D(fake) over training; both should hover near 0.5 at equilibrium. If D wins decisively (D(fake) → 0), G's gradients vanish and training stalls.

## Open questions

The paper leaves open the empirical question of how to make adversarial training stable at scale; it took years of follow-up work to find recipes that worked. It also does not give a satisfying way to evaluate sample quality quantitatively; later metrics (Inception Score, FID) partly fill this gap but each has known failure modes.
