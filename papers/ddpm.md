---
slug: ddpm
title: "Denoising Diffusion Probabilistic Models"
authors:
  - Jonathan Ho
  - Ajay Jain
  - Pieter Abbeel
venue: NeurIPS
year: 2020
arxiv_id: "2006.11239"
doi: null
url: https://arxiv.org/abs/2006.11239
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - diffusion
  - ddpm
  - score-matching
  - denoising
  - generative
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microdiffusion
  target_path: 01-foundations/microdiffusion.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microdiffusion.py
    script_slug: microdiffusion
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

DDPM trains a neural network to predict the noise added at each step of a fixed Gaussian forward diffusion process, then samples by reversing the process — iteratively denoising from pure Gaussian noise to a clean image — using a tractable parameterization that connects diffusion to denoising score matching.

## Problem

Diffusion models had been proposed in 2015 (Sohl-Dickstein et al.) as a Markov chain that gradually corrupts data with Gaussian noise and learns to reverse the process. But the original framing was awkward to train: the variational bound contained terms that were hard to estimate, and the resulting samples were not competitive with GANs or autoregressive models. Score-based generative models (Song & Ermon 2019) showed that learning the gradient of the data log-density at multiple noise scales could produce strong samples, but the connection between that view and the diffusion-Markov-chain view was implicit.

## Contribution

The paper unifies the two views and produces a practical recipe. It shows that the variational lower bound for the diffusion model simplifies dramatically under a specific parameterization: instead of predicting the previous-step mean directly, the network predicts the noise ε that was added at the current step. Under this parameterization the per-timestep loss becomes a simple weighted mean-squared error between the predicted and true noise, equivalent (up to weighting) to denoising score matching. The paper also fixes the variance schedule of the forward process so it does not need to be learned. The result is a training objective as simple as supervised regression and a sampling procedure that produces image-quality samples competitive with the best GANs at the time, on CIFAR-10 and LSUN.

## Method summary

- Forward process: q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) x_{t-1}, β_t I), with β_t a fixed linear or cosine schedule over T steps (the paper uses T = 1000).
- Closed-form marginals: x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε, where ᾱ_t is the cumulative product of (1 - β) and ε ~ N(0, I).
- Train a network ε_θ(x_t, t) to predict the noise: L = E_{t, x_0, ε}[||ε - ε_θ(x_t, t)||²]. The network sees a noised image and the timestep, outputs an ε estimate.
- Architecture: a U-Net with timestep embeddings injected into residual blocks.
- Sampling: start from x_T ~ N(0, I); for t = T, ..., 1, compute x_{t-1} from x_t using the predicted ε and the fixed schedule; the last step returns x_0.

## Key results

DDPM achieves an FID of 3.17 on unconditional CIFAR-10, beating prior likelihood-based models and matching contemporaneous GAN results. On LSUN bedroom, church, and cat at 256x256, samples are visually competitive with StyleGAN. The paper also reports that the bound's contribution simplifies to predicting noise, that the chosen parameterization gives much lower variance than alternatives, and that sampling quality remains stable across schedule choices.

## Relation to existing work

DDPM connects three lines: the original 2015 diffusion paper of Sohl-Dickstein et al.; score-based models (Song & Ermon 2019, NCSN), where the network predicts the score ∇_x log p_t(x) at multiple noise scales; and denoising autoencoders. The core insight — that predicting noise is equivalent to predicting the score and easier than predicting the previous-step mean — bridges the variational and score-matching views. Subsequent work compresses or accelerates the long sampling chain (DDIM, DPM-Solver), conditions the model on text or class for guided generation (classifier guidance, classifier-free guidance), and moves the diffusion to latent space (Stable Diffusion uses a VAE encoder before diffusion). The diffusion family largely displaced GANs for image generation by 2022.

## Implementation notes

A pedagogical script can train a tiny diffusion model on MNIST-scale images with a small CNN backbone. Minimum viable trainer: pick a β schedule, precompute ᾱ_t, sample t and ε per minibatch, form x_t in closed form, predict ε, MSE loss. The sampler runs T denoising steps with the explicit DDPM update equations. Pitfalls: forgetting that x_t can be sampled directly from x_0 without iterating the forward chain (a common implementation mistake that wastes compute and obscures the math); using a wrong constant in the sampling update (the σ_t² term has multiple valid choices — small β_t and the larger β̃_t both work); training with too few timesteps so the noise schedule jumps too quickly. Useful diagnostic: plot the average noise prediction error per timestep — it should be roughly constant across t after a few thousand steps; uneven curves indicate schedule miscalibration.

## Open questions

The paper does not address fast sampling — T = 1000 inference steps is impractical for many applications, and the field spent the next two years compressing this. It also leaves open the choice between learning the variance and fixing it; later work (Improved DDPM, Nichol & Dhariwal) showed learning a small parameter for the variance helps slightly.
