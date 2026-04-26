---
slug: word2vec
title: "Efficient Estimation of Word Representations in Vector Space"
authors:
  - Tomas Mikolov
  - Kai Chen
  - Greg Corrado
  - Jeffrey Dean
venue: arXiv
year: 2013
arxiv_id: "1301.3781"
doi: null
url: https://arxiv.org/abs/1301.3781
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary:
    - retrieval
tags:
  - word2vec
  - skip-gram
  - cbow
  - word-embeddings
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microembedding
  target_path: 01-foundations/microembedding.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microembedding.py
    script_slug: microembedding
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Word2Vec trains a shallow log-linear model to predict a word from its surrounding context (CBOW) or surrounding context from a word (Skip-gram), producing dense vector embeddings whose geometry encodes semantic and syntactic relationships at orders of magnitude lower compute than prior neural language models.

## Problem

Earlier methods for word representations were either sparse (one-hot, count-based) and gave no notion of similarity, or were learned by full neural language models (Bengio et al. 2003; Mnih and Hinton 2008) that were prohibitively expensive — training on a billion-word corpus could take weeks on a single CPU. The field needed embeddings that captured useful similarity structure but could be trained on web-scale corpora in hours.

## Contribution

The paper introduces two model families that strip away the hidden layer of prior neural language models and use only a single projection plus a softmax. CBOW (Continuous Bag-of-Words) sums or averages context-word vectors and predicts the center word; Skip-gram does the inverse, using the center word's vector to predict each context word independently. Both are trained on a sliding window over text. The architectural simplification — no hidden tanh layer — makes training one to two orders of magnitude faster than prior word-embedding models, allowing training on Google News (six billion tokens) in under a day. The paper also defines an analogy benchmark (king - man + woman ≈ queen) that captures relational structure and shows Word2Vec embeddings exhibit this structure better than Latent Semantic Analysis or distributed representations from prior work.

## Method summary

- Sliding window of size c (typically 5 or 10) over the corpus; for each center word w_t, the context is w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c}.
- CBOW: average the context vectors, project to vocabulary logits with W_out, softmax, predict w_t.
- Skip-gram: take w_t's vector, project to vocabulary logits with W_out, predict each context word independently.
- Loss: cross-entropy of the predicted distribution against the true word.
- Two embedding matrices: W_in (input/center vectors) and W_out (output/context vectors); only W_in is typically retained as the "word vectors" after training.
- The follow-up paper (Mikolov et al. 2013b, "Distributed Representations of Words and Phrases") introduces negative sampling and hierarchical softmax to replace the full softmax, which dominates compute for large vocabularies.

## Key results

Skip-gram trained on 6B Google News tokens, with 300-dimensional vectors and a window of 10, reaches 53.3% on the syntactic-analogy benchmark and 55.7% on the semantic-analogy benchmark, beating LSA, RNN-LM, and earlier neural embedding baselines while training in less than a day. The paper also demonstrates the famous analogies (king - man + woman → queen; Paris - France + Italy → Rome) qualitatively. CBOW is faster but slightly weaker on rare words; Skip-gram is slower but better at capturing fine-grained semantics.

## Relation to existing work

Word2Vec replaces the neural-network bottleneck of Bengio et al. (2003) with a log-linear model, trading some expressiveness for orders-of-magnitude speedup. It contrasts with count-based methods (LSA, HAL): GloVe (Pennington et al. 2014) later argued that the same kind of vectors arise from a global matrix factorization of co-occurrence counts and unified the two perspectives. fastText (Bojanowski et al. 2017) extends Word2Vec by averaging character n-gram embeddings, handling rare words and morphology better. Modern contextual embeddings (ELMo, BERT) supersede static embeddings by producing per-context vectors, but Word2Vec-style static embeddings remain a strong baseline for retrieval, classification, and any setting where one fixed vector per word suffices.

## Implementation notes

A pedagogical script can train Skip-gram on a small corpus (a single book, a Wikipedia subset). Minimum viable trainer: build a vocabulary, slide a window over tokens, for each center-context pair compute either full softmax (small vocab) or negative sampling (real-world vocab), update both embedding matrices. Pitfalls: sharing W_in and W_out as a single matrix degrades quality; using full softmax over a 100k+ vocabulary is infeasible — use negative sampling with k = 5 to 20 negatives per positive; subsampling frequent words (the paper drops common words like "the" with a probability proportional to their frequency) substantially improves quality. Useful diagnostic: after training, evaluate on a small analogy set or print nearest neighbors for a few seed words; meaningful structure should emerge within a few epochs on a moderate corpus.

## Open questions

The paper does not analyze why a log-linear model produces analogy-friendly geometry; later work (Levy & Goldberg 2014, Arora et al. 2016) gives partial theoretical accounts via implicit matrix factorization and isotropy. Word2Vec also assumes a single sense per word; multi-sense and contextual embeddings address this gap.
