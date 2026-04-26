---
slug: rag
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
authors:
  - Patrick Lewis
  - Ethan Perez
  - Aleksandra Piktus
  - Fabio Petroni
  - Vladimir Karpukhin
  - Naman Goyal
  - Heinrich Küttler
  - Mike Lewis
  - Wen-tau Yih
  - Tim Rocktäschel
  - Sebastian Riedel
  - Douwe Kiela
venue: NeurIPS
year: 2020
arxiv_id: "2005.11401"
doi: null
url: https://arxiv.org/abs/2005.11401
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: retrieval
  secondary:
    - architecture
tags:
  - rag
  - dense-retrieval
  - generation
  - knowledge
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microrag
  target_path: 01-foundations/microrag.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microrag.py
    script_slug: microrag
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

RAG combines a learned dense retriever with a sequence-to-sequence generator: for each input, the retriever returns the top-K passages from a non-parametric knowledge index, the generator conditions on each passage and produces a candidate output, and the final answer marginalizes over passages — letting a language model use external, swappable knowledge instead of memorizing everything in its parameters.

## Problem

Pretrained language models store knowledge in their parameters. This works for common facts but fails for long-tail knowledge, recent events, or any topic underrepresented in training data; the model has no way to update without retraining. Earlier extractive QA systems retrieved passages and selected spans, but the answer space was constrained to spans literally present in retrieved text. The paper asks whether retrieval can be coupled with free-form generation so the model can use retrieved evidence and still produce arbitrary natural-language answers, fact-check itself against the retrieved passages, and update its knowledge by swapping the index.

## Contribution

RAG defines a probabilistic model p(y|x) = Σ_z p(z|x) · p(y|x, z), where x is the input, z is a retrieved document, and the sum runs over the top-K documents returned by a learned retriever. The retriever is the pretrained Dense Passage Retriever (DPR): it encodes the query and each document into vectors, scores by inner product, and returns the top K via fast nearest-neighbor search over a precomputed Wikipedia index. The generator is BART, a pretrained seq2seq model. Two variants differ in granularity: RAG-Sequence uses the same retrieved document for all output tokens (marginalize per-output); RAG-Token allows a different document at each generation step (marginalize per-token). The retriever and generator are fine-tuned jointly: the retriever encoder is updated; the document index is treated as fixed during training but can be replaced at inference for knowledge updates.

## Method summary

- Retriever (DPR): two BERT-base encoders, one for queries and one for passages; passage encodings are precomputed for the Wikipedia corpus and indexed by FAISS for sublinear nearest-neighbor search.
- For input x, retrieve the top K passages by maximum inner product (paper uses K = 5 or 10).
- Generator (BART-large): for each retrieved passage z_k, concatenate (x, z_k) as a prefix and produce a conditional distribution p(y|x, z_k).
- RAG-Sequence: combine document-level posteriors p(y|x) = Σ_k p(z_k|x) · p(y|x, z_k); decode by per-document beam search and re-score by document posterior.
- RAG-Token: at each generation step, marginalize across documents: p(y_t|y_{<t}, x) = Σ_k p(z_k|x) · p(y_t|y_{<t}, x, z_k); decode with a token-level beam search across the marginal.
- Training: maximum likelihood with the retriever's query encoder fine-tuned by backprop through the marginalization; the document index is held fixed during training (refresh between epochs is optional).

## Key results

On open-domain QA (Natural Questions, TriviaQA, WebQuestions, CuratedTREC), RAG sets new state of the art at the time, beating both pure-extractive systems (DPR + reader) and pure-parametric systems (T5, closed-book BART). RAG-Sequence is generally stronger on factoid QA; RAG-Token wins on tasks where the answer requires combining facts from multiple documents (e.g. some Jeopardy questions). The paper also shows that swapping the underlying Wikipedia index for an updated snapshot lets RAG answer questions about new facts without any retraining — a key practical demonstration that knowledge can be edited by replacing the index.

## Relation to existing work

RAG combines two lines: dense retrieval (Karpukhin et al. 2020, DPR) and pretrained seq2seq generation (Lewis et al. 2020, BART). It contrasts with extractive QA (BERT-based reader on top of retrieved passages), which is constrained to literal spans, and with closed-book QA (a model answering from parameters alone), which cannot fact-check against retrievable evidence. Subsequent work generalized the recipe in many directions: REALM and FiD use the same retrieve-then-generate template with different fusion mechanisms; Atlas, RETRO, and InstructRetro pretrain end-to-end with retrieval; modern production RAG systems pair frozen embedding models, vector databases, and instruction-tuned LLMs without joint training. The conceptual frame from this paper — explicit non-parametric memory plus parametric generator — is the foundation of essentially every retrieval-augmented LLM system.

## Implementation notes

A pedagogical script can implement RAG with a tiny corpus, an embedding model, and a small generator. Minimum viable trainer: index passages by mean-pool of token embeddings; for each query, embed and retrieve top-K by cosine; concatenate and generate. Pitfalls: forgetting to detach the document encoder if not training it; using K too small (retrieval recall drops sharply at K = 1); decoding the RAG-Token marginal naively — easier to demonstrate RAG-Sequence first. Diagnostic: log top-K retrieval recall on a held-out QA set; low recall means the issue is the retriever, not generation.

## Open questions

RAG conditions on retrieved passages but the generated answer does not natively cite which passage supplied which fact. Citation-augmented and attribution-trained variants target this gap.
