---
slug: bpe
title: "Neural Machine Translation of Rare Words with Subword Units"
authors:
  - Rico Sennrich
  - Barry Haddow
  - Alexandra Birch
venue: ACL
year: 2016
arxiv_id: "1508.07909"
doi: null
url: https://arxiv.org/abs/1508.07909
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - bpe
  - tokenization
  - subword
  - machine-translation
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microtokenizer
  target_path: 01-foundations/microtokenizer.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microtokenizer.py
    script_slug: microtokenizer
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The paper adapts byte-pair encoding (originally a 1994 data-compression algorithm) to text tokenization for neural machine translation: greedily merge the most frequent adjacent symbol pair until the desired vocabulary size is reached, then use the resulting merge table to segment any input — including unseen rare words — into known subword units.

## Problem

Neural machine translation systems used a fixed word vocabulary. Words outside the vocabulary (rare nouns, named entities, morphological variants in languages like German or Russian) were replaced with an [UNK] token, making translation of these words impossible. Workarounds — copy mechanisms, character-level models, lookup tables — each had drawbacks: copy mechanisms only handled exact unknown-source-to-target alignment, character models were slow because sequences became much longer, and lookup tables required maintaining external resources. The paper asks whether segmenting words into a smaller, fixed vocabulary of subword units can give the model open-vocabulary coverage without character-level overhead.

## Contribution

The paper applies byte-pair encoding (Gage 1994), originally a compression scheme, to text tokenization. Start with each word represented as its sequence of characters plus an end-of-word marker. Count all adjacent symbol pairs across the corpus and merge the most frequent pair into a new symbol; add this merge to a table. Repeat until the vocabulary reaches a target size (typically 10k–60k merges). The merge table can then segment any input string — common words become single tokens, rare words split into known subword pieces, and truly novel sequences fall back to characters. Applied to WMT translation tasks, BPE eliminates [UNK] tokens, improves BLEU on rare-word translation, and allows the system to handle morphologically rich languages with the same architecture as English-French.

## Method summary

- Initialize: tokenize the training corpus into words; represent each word as a sequence of characters with a special end-of-word symbol (typically `</w>`).
- Count occurrences of each character-bigram across all words, weighted by word frequency.
- Find the most frequent pair (a, b); add merge rule "a b → ab" to the table.
- Apply the merge: in every word containing the bigram, replace adjacent (a, b) tokens with the new merged symbol.
- Repeat for K iterations, where K determines the final vocabulary size.
- At inference: greedily apply learned merges in order to any new input string; the result is a sequence of tokens drawn from the merge-table vocabulary plus characters.

## Key results

On WMT 2015 English-German and English-Russian, BPE-based NMT systems improve over word-level baselines by 1.1 to 1.3 BLEU on average and by much more on rare words specifically. The method dropped [UNK] rates from several percent to effectively zero. The paper also shows that BPE generalizes across languages without per-language tuning beyond choosing a vocabulary size.

## Relation to existing work

The merge algorithm itself comes from Gage (1994), "A New Algorithm for Data Compression" (C/C++ Users Journal). The contribution is the adaptation to tokenization: treating the merge table as an open-vocabulary segmenter rather than a compression dictionary. WordPiece (Schuster & Nakajima 2012, used in BERT) is a closely related variant that picks merges by likelihood rather than frequency. SentencePiece (Kudo & Richardson 2018) treats raw text as a stream of bytes (no whitespace-based pretokenization) and supports both BPE and unigram-LM segmentation; it is the dominant tokenizer for modern multilingual models. GPT and many LLMs use a byte-level BPE variant (Radford et al. 2019) that operates on bytes rather than Unicode code points, removing per-language preprocessing. Tokenizer choice has measurable downstream effects on multilingual fairness and on certain task accuracies.

## Implementation notes

A pedagogical script can implement BPE training and segmentation in pure Python. Minimum viable trainer: read a small corpus, build word-frequency dict, initialize each word as a character-tuple, iterate (count pairs, merge most-frequent, update vocabulary) for K iterations. Segmentation greedily applies learned merges to any input. Pitfalls: counting pairs naively is O(corpus × vocabulary) per iteration — use cached pair counts and only update affected words after each merge for tractable speed. Adding the end-of-word marker matters: without it, a token that spans a word boundary at training time will not match the same string at inference. Useful exercise: train BPE at vocabulary sizes 100, 1000, 10000 on the same corpus and observe how token granularity changes — small vocabularies are nearly character-level, large ones approach word-level for common tokens.

## Open questions

The paper does not analyze how vocabulary size interacts with downstream task performance; later work shows the choice has subtle effects (particularly cross-lingual transfer). The greedy-merge ordering is also not optimal in any formal sense; alternative segmentation algorithms (unigram language model, used in SentencePiece) make different trade-offs.
