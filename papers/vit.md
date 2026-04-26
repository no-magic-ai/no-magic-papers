---
slug: vit
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors:
  - Alexey Dosovitskiy
  - Lucas Beyer
  - Alexander Kolesnikov
  - Dirk Weissenborn
  - Xiaohua Zhai
  - Thomas Unterthiner
  - Mostafa Dehghani
  - Matthias Minderer
  - Georg Heigold
  - Sylvain Gelly
  - Jakob Uszkoreit
  - Neil Houlsby
venue: ICLR
year: 2021
arxiv_id: "2010.11929"
doi: null
url: https://arxiv.org/abs/2010.11929
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary:
    - multimodal
tags:
  - vit
  - vision-transformer
  - patch-embedding
  - image-classification
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microvit
  target_path: 01-foundations/microvit.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microvit.py
    script_slug: microvit
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The Vision Transformer treats an image as a sequence of fixed-size patches, embeds each patch as a token, prepends a learnable [class] token, and feeds the sequence through a standard transformer encoder, matching or exceeding state-of-the-art convolutional networks when pretrained on enough data.

## Problem

Convolutional networks dominated computer vision for nearly a decade because their built-in inductive biases — locality, translation equivariance, and hierarchical receptive fields — match natural images well. Transformers had taken over NLP but applying them to images required tokenizing pixels somehow, and prior attempts (image-GPT, attention-augmented CNNs, axial attention) either scaled poorly or kept too much convolutional structure to test the underlying question. The paper asks whether a pure transformer, with the convolutional priors removed entirely, can compete with CNNs given enough training data.

## Contribution

The paper introduces ViT: an image is split into N non-overlapping patches of size P×P, each patch is flattened and projected linearly to a D-dimensional embedding, learnable position embeddings are added, a [class] token is prepended, and the resulting (N+1)-token sequence is processed by a standard pre-norm transformer encoder. The final representation of the [class] token is fed to a linear classifier head. The architecture has no convolutional or pooling layers — just patch embedding (a single conv-like linear projection at the input) and transformer blocks. The empirical claim is that ViT matches or beats the best CNNs when pretrained on a large enough dataset (JFT-300M); on smaller datasets like ImageNet alone, CNNs still win because the inductive bias matters more.

## Method summary

- Split image of size H×W into patches of size P×P, producing N = HW/P² patches; the paper uses P = 16 for 224×224 images, giving N = 196.
- Flatten each patch to a vector of length 3·P² (RGB) and project linearly to D dimensions — equivalently a stride-P convolution with D output channels.
- Prepend a learnable [class] token and add learnable 1-D position embeddings to all (N+1) positions.
- Run the sequence through L layers of standard pre-norm transformer encoder blocks (multi-head self-attention + MLP + residual connections + LayerNorm).
- A linear head on the final [class] token produces classification logits.
- Pretrain on a large image-classification dataset (ImageNet-21k or JFT-300M); fine-tune on the target task at higher resolution by interpolating the position embeddings.

## Key results

ViT-Large pretrained on JFT-300M reaches 88.55% top-1 accuracy on ImageNet, matching the best CNNs of the time at lower compute. ViT-Huge reaches 88.55% to 88.64% and beats EfficientNet on multiple downstream tasks. The paper's most important empirical finding is the data-scale crossover: with only ImageNet-1k pretraining, ViT underperforms ResNet-style baselines; with ImageNet-21k it matches them; with JFT-300M it surpasses them. The inductive bias of CNNs is a substitute for data, not a free advantage.

## Relation to existing work

ViT replaces the convolution-everywhere lineage (LeNet, AlexNet, VGG, ResNet, EfficientNet) with a pure transformer encoder, the same architecture used in BERT. Earlier hybrid approaches (DETR for detection, Image GPT for autoregressive pixel modeling, attention-augmented convnets) had used attention selectively; ViT shows the convolutional component is unnecessary if data is plentiful. Subsequent work refined the recipe: DeiT (Touvron et al. 2021) achieves ViT performance using only ImageNet-1k via stronger augmentation and distillation; Swin Transformer reintroduces local windowed attention to recover some convolutional inductive bias; MAE (He et al. 2022) pretrains ViT with masked-image modeling, parallel to BERT in NLP. ViT is also the visual encoder in most modern vision-language models (CLIP, Flamingo).

## Implementation notes

A pedagogical script can train a tiny ViT on CIFAR-10 or MNIST. Minimum viable model: a patch embedding module (a Conv2d with kernel = stride = P), a learnable [class] token, learnable position embeddings, and a small transformer encoder stack. Pitfalls: forgetting to add position embeddings (the model still trains but converges to a permutation-invariant solution that ignores spatial order); applying LayerNorm post-attention (the original Transformer convention) instead of pre-attention (modern ViT convention, more stable); using a patch size that doesn't divide the image (the paper handles this with padding or interpolation). Useful diagnostic: visualize the learned position embeddings as a 2D grid of cosine similarities — they should reveal a 2D structure even though the embeddings themselves are 1-D and unconstrained, demonstrating that the model learns spatial layout from data.

## Open questions

The paper does not analyze why ViT needs more data than CNNs to reach the same accuracy; subsequent theoretical work links this to the lack of locality bias. ViT is also computationally expensive at high resolution because attention is quadratic in the number of patches; later architectures (Swin, MViT) trade this off in different ways.
