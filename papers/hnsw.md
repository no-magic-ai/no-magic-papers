---
slug: hnsw
title: "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"
authors:
  - Yu A. Malkov
  - D. A. Yashunin
venue: arXiv
year: 2016
arxiv_id: "1603.09320"
doi: null
url: https://arxiv.org/abs/1603.09320
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: retrieval
  secondary: []
tags:
  - hnsw
  - ann
  - small-world-graph
  - vector-search
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microvectorsearch
  target_path: 03-systems/microvectorsearch.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microvectorsearch.py
    script_slug: microvectorsearch
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

HNSW builds a hierarchy of navigable small-world graphs over a vector dataset: search starts at the top sparse layer with long-range edges and descends through denser layers using greedy nearest-neighbor moves, achieving sublogarithmic query time and near-state-of-the-art recall for approximate nearest-neighbor search across dimensionalities and metrics.

## Problem

Approximate nearest-neighbor (ANN) search over high-dimensional vectors is the inner loop of vector retrieval, recommendation, and embedding-based search. Exact search costs O(N · d) per query and is infeasible for N in the millions. Earlier ANN methods — KD-trees, locality-sensitive hashing (LSH), product quantization with inverted files (IVF) — each had distinct weaknesses: KD-trees degrade in high dimensions; LSH requires tuning many hash functions and trades quality for speed; IVF needs careful coarse-quantizer training and rebuilds when the data distribution shifts. The paper asks whether a graph-based index — connecting each vector to its approximate neighbors — can deliver both high recall and high throughput across diverse data without per-collection tuning.

## Contribution

HNSW introduces a hierarchical multi-layer graph where each layer is a navigable small-world graph and the layer assignment of each point follows a geometric distribution. The bottom layer contains all points connected to local neighbors; each upper layer is an exponentially smaller subset, with longer-range edges that act as expressways. Search starts at a fixed entry point in the top layer, performs greedy nearest-neighbor moves until no closer neighbor exists, then descends one layer and repeats. The combination delivers logarithmic-or-better query time, recall above 0.95 at typical settings, and a single graph index that supports any distance metric (Euclidean, cosine, inner product) without rebuilding. The construction algorithm builds the graph incrementally by inserting points one at a time, performing the same hierarchical search to find candidate neighbors, and selecting the M best per layer for connection.

## Method summary

- Each point is assigned a maximum layer L drawn from the geometric distribution Geometric(1/log(M)); higher layers are exponentially less populated.
- The graph at layer ℓ has at most M neighbors per node (M_max in the paper, typically 16-48), chosen by a heuristic that picks diverse rather than purely closest neighbors.
- Insertion: starting at the top entry point, greedy-search down through layers L_top to L+1; at each layer from L down to 0, find ef_construction nearest candidates and connect to the M best.
- Query: same descent, but at layer 0 keep the ef nearest candidates (ef ≥ k) and return the top k.
- Heuristic neighbor selection (the paper's "select_neighbors_heuristic") picks diverse neighbors to avoid graph fragmentation; this is the key quality difference vs naive nearest-M.
- Distance metric is parameterized; the algorithm makes no assumptions beyond a pairwise distance function.

## Key results

The paper benchmarks HNSW on the SIFT1M, GIST1M, GloVe, and Deep1B datasets and shows it dominates earlier ANN methods (LSH variants, FLANN, NMSlib brute-force, IVF + product quantization) on the recall-vs-queries-per-second Pareto frontier. At the same query throughput HNSW achieves substantially higher recall; at the same recall it achieves several times higher throughput. The graph-based index is also robust to data distribution: it does not require coarse-quantizer training, and it handles streaming insertion natively.

## Relation to existing work

HNSW supersedes the small-world-graph line (NSW, Malkov et al. 2014) by adding the hierarchical layer structure, which fixes NSW's slow descent through dense regions. It contrasts with IVF (Jegou et al. 2011) and IVF+PQ (which dominates very large-scale settings where memory is the binding constraint), and with LSH families (which are simpler but consistently lower quality on the Pareto frontier). The FAISS library (Johnson et al. 2017) implements HNSW alongside IVF and other indices and is the standard implementation in production systems. Modern vector databases (Pinecone, Weaviate, Milvus, pgvector with HNSW extension, Qdrant) use HNSW or HNSW-derived indices as the default. Recent variants — DiskANN (Subramanya et al. 2019) for SSD-resident graphs, FreshDiskANN for streaming inserts at scale — extend the same graph-index family.

## Implementation notes

A pedagogical script can implement HNSW for a small dataset in pure Python. Minimum viable implementation: a dict from node id to {neighbor lists per layer, max layer}, a greedy-search routine, and an insertion routine that calls greedy search at each layer. The script can also include a brute-force baseline to confirm recall. Pitfalls: forgetting the diverse-neighbor heuristic (using nearest-M produces clusters that degrade graph navigability); using too small ef for queries (recall plummets); reseeding the entry point per query rather than reusing a stable one (search becomes noisy). Useful exercise: time and recall the same query across brute-force, IVF (with k-means coarse quantizer), and HNSW on SIFT-like vectors; the recall-throughput frontier makes the HNSW advantage visible.

## Open questions

The paper does not optimally tune M and ef for arbitrary datasets; later work (DiskANN, NSG) explores graph-construction variants with better worst-case behavior. Memory is HNSW's main weakness — every node stores M neighbor IDs at every layer it appears in — and is what motivates IVF + PQ for very large collections.
