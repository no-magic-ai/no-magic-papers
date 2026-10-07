# KV-Cache Lesson

Paper card: `papers/kv-cache.md`

Implementation: `no-magic/03-systems/microkv.py`

## Paper summary

Pope et al. (2022) study generative inference for very large transformers under tight latency targets and long contexts. They split inference into two phases: prefill, which runs the model over all input tokens of a batch in one parallel pass, and decode (generation), a sequential loop that produces one token per sequence per step. The attention keys and values of every layer — the KV cache — must stay in memory for the whole decode, and every decode step reloads the weights and the cache from high-bandwidth memory while the compute units mostly wait. Their example: for a 500B+ parameter model with multi-head attention, a batch of 512 at context length 2,048 needs a 3 TB KV cache, three times the size of the parameters.

The paper builds an analytical model of these costs and uses it to choose how to partition weights and activations across TPU v4 chips for each phase. Multi-query attention, in which all query heads share one key/value head, shrinks the cache by a factor of the number of heads; with a partitioning that shards that cache over the batch, they report support for up to 32× longer contexts. On PaLM 540B they report 29 ms per generated token at low batch size (with int8 weights) and 76% model FLOPS utilization when processing input tokens at large batch, with a 2,048-token context.

## Intuition

In causal decoding, the key and value of a token depend only on that token and the tokens before it. Once computed they never change, so recomputing them for the whole prefix at every step is pure waste. Caching them turns each step's projection and MLP work from proportional to the prefix length into constant, while attention still reads every cached position.

The price is memory that grows with every token: `2 × layers × KV heads × head size × tokens` values per sequence. At small scale that is trivial; at the paper's scale it outweighs the model, which is why cache size drives decisions such as multi-query attention, grouped-query attention, quantized caches and paged allocation. Those designs change what is stored and are not numerically identical to the full multi-head cache. Plain caching of standard causal attention, by contrast, changes nothing about the result: same weights, same mask, same outputs.

## Code walkthrough

`microkv.py` trains a small model (width 16, 2 heads, 1 layer, 300 steps) and copies its weights into plain-float lists. `generate_no_cache` re-runs every position at every step: it embeds the whole sequence, projects all queries, keys and values, attends causally for every position and runs the MLP on all of them, then keeps only the last position's logits. `generate_with_cache` embeds only the newest token, projects one query, key and value, appends the key and value to `kv_cache`, and attends with the new query over every cached position. Both decode greedily from the BOS token for 16 steps, and the script asserts that they produce identical tokens.

`linear_f` and the attention loops count scalar multiplications into a per-step counter. At step 1 both methods cost 3,536 (one position either way). At step 16 the uncached method costs 53,936 and the cached one 4,016, and over 16 steps the totals are 450,816 and 60,416 (7.5×); running the two generators with random weights reproduces these counts, which do not depend on the weight values. The memory table grows by `2 × 1 × 16 = 32` floats per position, reaching 512 floats (2,048 bytes as float32) after 16 positions. `simulate_paged_attention` then maps those positions to 4-slot blocks through a page table, the idea `micropaged.py` develops across many requests.

The closing signpost's "~5.2 GB" for LLaMA-2 70B at 4K context does not follow from the script's own formula: with 80 layers, width 8,192 and FP16 it gives about 10.7 GB, and LLaMA-2 70B's grouped-query attention (8 key/value heads of size 128) brings it to about 1.3 GB.

## Exercises

1. Derive the per-step counts: with the cache, step `n` costs `3,504 + 32n`; without it, `3,072n + 16n(n + 1) + 432`. Check both against the printed table.
2. Implement multi-query attention in the cached path: compute one 8-dimensional key and value per position and share them across both query heads. How many floats does the cache add per position? (16 instead of 32.) Are the generated tokens still identical to the multi-head version? (They need not be: it is a different model.)
3. Add a prefill phase: start from a 4-letter prompt, compute its keys and values in one pass, then decode. Count multiplications for the prefill separately from the decode steps.
4. Compute the cache for one sequence of a model with 32 layers, 32 heads of size 128 and a 4,096-token context in FP16 (`2 × 32 × 32 × 128 × 4,096 × 2` bytes ≈ 2.1 GB), then with 8 key/value heads (≈ 0.54 GB).
5. Store the cache in 8 bits per value with one absmax scale per vector (as in `microquant.py`) and measure how often greedy generation still matches the float cache.
