---
slug: resnet
title: "Deep Residual Learning for Image Recognition"
authors:
  - Kaiming He
  - Xiangyu Zhang
  - Shaoqing Ren
  - Jian Sun
venue: CVPR
year: 2016
arxiv_id: "1512.03385"
doi: null
url: https://arxiv.org/abs/1512.03385
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - resnet
  - residual
  - skip-connection
  - deep-network
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microresnet
  target_path: 01-foundations/microresnet.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microresnet.py
    script_slug: microresnet
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

ResNet adds an identity-shortcut connection around each pair of convolutional layers so the layers learn a residual function F(x) added to x rather than the full mapping H(x); this lets very deep networks (50, 101, 152 layers) train as easily as shallow ones and consistently improves accuracy with depth.

## Problem

Plain stacked convolutional networks should improve with depth — a deeper network can in principle realize any function a shallower one can, plus more. In practice, networks deeper than about 20 layers degraded: training error went up, not down. The cause was not vanishing gradients (BatchNorm and careful initialization had largely solved that) but an optimization difficulty: the optimizer struggled to find good identity-or-near-identity mappings through stacks of nonlinear layers, so additional layers actively hurt. The paper asks whether reframing the layer's job from "learn H(x)" to "learn the residual H(x) - x" eliminates this degradation.

## Contribution

ResNet introduces residual blocks: a small stack of convolutional layers F(x) whose output is added to the block's input via an identity shortcut, producing y = F(x) + x. If the optimal block behavior is the identity, F can be driven to zero — much easier than reproducing the input through the convolutional stack. The paper proves this empirically by training networks of 18, 34, 50, 101, and 152 layers on ImageNet, showing depth now consistently improves accuracy, and reaching first place on ILSVRC 2015 classification, detection, and localization. The paper also introduces the bottleneck block (1×1, 3×3, 1×1 conv stack) that lets very deep networks share parameter budget efficiently.

## Method summary

- Building block: y = F(x, {W_i}) + W_s · x, where F is a stack of two or three convolutional layers with BatchNorm and ReLU, and W_s is identity (or a 1×1 projection if dimensions change).
- Two-layer block (used in ResNet-18, ResNet-34): two 3×3 convolutions; F has the same channel count as x throughout.
- Three-layer bottleneck block (used in ResNet-50, ResNet-101, ResNet-152): 1×1 conv (reduce channels) → 3×3 conv → 1×1 conv (restore channels); the bottleneck reduces FLOPs at the same depth.
- Down-sampling: stride-2 convolutions at block boundaries; the shortcut path uses a 1×1 stride-2 projection to match shape.
- Standard ImageNet recipe: SGD with momentum, weight decay 1e-4, batch size 256, learning rate decayed at fixed milestones, no dropout (BatchNorm provides regularization).

## Key results

ResNet-152 reaches 3.57% top-5 error on ImageNet, beating the previous year's winner (GoogLeNet, 6.67%) by a large margin. ResNet-34 outperforms a 34-layer plain network (which suffers degradation), confirming that residual connections, not depth alone, drive the improvement. The same architecture wins COCO detection and segmentation challenges with no task-specific changes. The paper also reports that increasing depth from 50 to 152 layers continues to improve accuracy, in contrast to plain networks where deeper is worse past 20 layers.

## Relation to existing work

ResNet builds on VGG-style networks (Simonyan & Zisserman 2014) and on Highway Networks (Srivastava et al. 2015), which proposed gated shortcut connections. Removing the gating — making the shortcut a fixed identity — is the simpler, more effective choice. ResNet's identity-shortcut idea propagates into nearly every subsequent deep architecture: DenseNet adds shortcuts from each layer to all subsequent layers; Transformers (Vaswani et al. 2017) include residual connections around every attention and MLP sublayer (which ViT inherits); modern very-deep networks rely on the same primitive. The pre-activation variant (He et al. 2016b, "Identity Mappings in Deep Residual Networks") cleans up the order of BN, ReLU, and addition for slightly better optimization.

## Implementation notes

A pedagogical script can train a small ResNet on CIFAR-10 with the paper's CIFAR variant (3 stages of 6n+2 layers each). Minimum viable block: two 3×3 convolutions with BatchNorm and ReLU, an identity shortcut, post-add ReLU. Pitfalls: forgetting the projection shortcut when channel count or spatial size changes (the 1×1 conv with matching stride); applying ReLU before the addition rather than after, which slightly hurts and breaks the identity-mapping intuition; using too high a learning rate without warmup on very deep variants. Useful diagnostic: compare a plain stack and a residual stack at the same depth — the plain version should train fine at depth 20 and degrade at depth 56; the residual version should improve through both.

## Open questions

The paper explains why residual connections help intuitively (easier identity mapping, gradient highway) but does not give a full theoretical account; subsequent analyses (loss-landscape visualization, ensemble interpretation of ResNets) offer partial answers. Pre-activation ResNets and modern variants (ResNeXt, Wide ResNet) refine the block design without changing the residual idea.
