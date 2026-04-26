---
slug: lenet-5
title: "Gradient-Based Learning Applied to Document Recognition"
authors:
  - Yann LeCun
  - Léon Bottou
  - Yoshua Bengio
  - Patrick Haffner
venue: Proceedings of the IEEE
year: 1998
arxiv_id: null
doi: "10.1109/5.726791"
url: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - cnn
  - lenet
  - convolution
  - pooling
  - mnist
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microconv
  target_path: 01-foundations/microconv.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microconv.py
    script_slug: microconv
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

LeNet-5 introduces the modern convolutional neural network: a stack of convolutional layers (local receptive fields with shared weights), subsampling layers (averaging then nonlinearity), and fully-connected layers, trained end-to-end by backpropagation, applied to handwritten digit and document recognition.

## Problem

Pattern-recognition systems in the 1990s relied on hand-engineered features followed by a separately trained classifier. The features were brittle, varied across domains, and absorbed most of the engineering effort. Multilayer perceptrons could in principle learn from raw pixels but had two practical problems on images: they used vastly more parameters than needed because they ignored 2D spatial structure, and they were not invariant to small shifts and distortions of the input. The paper asks whether a network with built-in spatial inductive biases — local connectivity, weight sharing across positions, and progressive subsampling — can learn directly from pixels and beat hand-engineered systems on practical document-recognition tasks.

## Contribution

The paper formalizes the convolutional neural network and trains LeNet-5 end-to-end on MNIST and on commercial check-reading. The architecture stacks three repeating motifs: a convolutional layer (each unit sees a small spatial window of the previous layer; weights are shared across all spatial positions of a feature map), a subsampling layer (pooling that averages and then applies a learned scaling and bias), and eventually fully-connected layers near the output. The combination gives shift invariance for free, dramatically reduces parameter count compared with a fully-connected network of similar receptive field, and trains with standard backpropagation and stochastic gradient descent. The same paper also introduces the Graph Transformer Network for combining segmentation and recognition, used in the deployed check-reading system, but the convolutional-network section is the lasting contribution.

## Method summary

- Input: 32×32 grayscale image (MNIST padded by 2 pixels per side).
- C1 layer: convolutional with 6 feature maps, 5×5 kernels.
- S2 layer: subsampling 2×2; for each 2×2 region, sum the four pixels, multiply by a learned coefficient, add a bias, apply tanh.
- C3 layer: convolutional with 16 feature maps, 5×5 kernels; not all C3 maps connect to all S2 maps (a hand-designed sparse connectivity that breaks symmetry and reduces parameters).
- S4 layer: subsampling 2×2 (same shape as S2).
- C5 layer: convolutional with 120 feature maps, 5×5 kernels; the receptive field is the full 5×5 from S4, so this is effectively fully connected.
- F6 layer: fully connected, 84 units, tanh activation.
- Output layer: 10 Euclidean radial-basis-function units (one per digit), distance from a prototype.
- Trained by stochastic gradient descent with a per-parameter learning rate, MSE loss against one-hot targets transformed by the RBF prototypes.

## Key results

LeNet-5 reaches 0.95% test error on MNIST in the original setup, and 0.7% with distortion-based data augmentation. At the time this was state of the art for digit recognition without per-image preprocessing. The paper also reports the deployed check-reading system reads roughly 10% of all checks in the United States as of the late 1990s, demonstrating that the network ran at industrial scale on commodity hardware. The convolutional architecture beats earlier MLPs on the same task using fewer parameters and shorter training time.

## Relation to existing work

LeNet-5 inherits the convolutional structure from Fukushima's Neocognitron (1980), which used unsupervised learning, and from the earlier supervised CNN of LeCun et al. (1989). It generalizes both by training the entire network end-to-end with backpropagation on a real-world task. The architecture lay relatively dormant at large scale until AlexNet (Krizhevsky, Sutskever, Hinton 2012) showed that the same ideas, scaled to ImageNet with ReLU activations, max pooling, dropout, and GPU training, beat hand-engineered features by a wide margin. VGG, GoogLeNet, ResNet, and modern CNNs all trace their core convolution-pool-classify structure to LeNet-5. ViT (Dosovitskiy et al. 2020) eventually showed transformers can match CNNs at scale, but the convolutional layer remains a foundational primitive.

## Implementation notes

A pedagogical script can implement LeNet-5-style convolutional and pooling layers from scratch and train on MNIST. Minimum viable trainer: a Conv2d-equivalent that does forward and backward over a batch with weight sharing across positions; an average-pool or max-pool layer; flatten and dense head; cross-entropy loss. Pitfalls: looping over spatial positions in pure Python is slow — for pedagogical clarity, loops are fine; for usable training time, vectorize. Forgetting the bias broadcast across spatial positions per feature map; using ReLU instead of tanh changes the dynamics (modern networks use ReLU; the LeNet paper used tanh and rescaled-tanh). Useful diagnostic: visualize the C1 filters after training — they should look like edge and stroke detectors, the canonical interpretability result for trained CNNs.

## Open questions

LeNet-5 is shallow by modern standards. The depth-degradation problem would not be resolved until ResNet (He et al. 2016).
