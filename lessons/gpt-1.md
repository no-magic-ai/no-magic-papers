# GPT-1 Lesson

Paper card: `papers/gpt-1.md`

Implementation: `no-magic/01-foundations/microgpt.py`

## Paper summary

Radford, Narasimhan, Salimans and Sutskever (2018) train a language model in two stages. First, unsupervised pre-training maximizes the likelihood of each token given the `k` tokens before it, `L1 = Σ log P(u_i | u_(i−k), …, u_(i−1))`, using a decoder-only transformer: token embeddings plus learned position embeddings (`h0 = U W_e + W_p`), a stack of masked self-attention blocks, and an output layer that reuses the token embedding matrix (`P(u) = softmax(h_n W_eᵀ)`). Second, supervised fine-tuning adds one linear layer `W_y` on the final block's activation, maximizes `L2 = Σ log P(y | x_1, …, x_m)` on a labelled task, and keeps the language-model loss as an auxiliary term, `L3 = L2 + λ·L1` with `λ = 0.5`. Structured inputs such as sentence pairs are turned into one token sequence with delimiter tokens, so the same network serves every task.

The model has 12 layers, 768-dimensional states, 12 heads and 3,072-dimensional feed-forward layers, uses GELU and LayerNorm, and reads a byte-pair vocabulary with 40,000 merges. It is pre-trained on BooksCorpus for 100 epochs of 64 sequences of 512 tokens with Adam (peak learning rate 2.5e-4, warmup then cosine decay); fine-tuning usually takes 3 epochs at 6.25e-5.

## Intuition

Next-token prediction is a training signal that needs no labels: every position of every text is an example. To predict well, the network has to model spelling, syntax and some meaning, and fine-tuning then reuses those features with very few new parameters. The masked (causal) attention is what makes the objective honest: position `i` may look only at positions before it, so it can never copy the answer.

Training this network means differentiating a long chain of simple operations. Reverse-mode automatic differentiation does this mechanically. Every operation records its inputs and its local derivative; the backward pass visits the operations from the loss back to the parameters and adds `upstream gradient × local derivative` into each input. When a value feeds several operations, its gradient is the sum of the contributions from each use. A GPU framework does the same thing on tensors; `microgpt.py` does it one scalar at a time, which is slow but leaves nothing hidden.

## Code walkthrough

`Value` stores `data`, `grad`, its `_children` and the `_local_grads` of the operation that produced it. `__mul__`, for example, records `(other.data, self.data)` as the derivatives of `a·b` with respect to `a` and `b`. `backward()` builds a topological order with a depth-first search, sets the loss gradient to 1 and walks the order in reverse, doing `child.grad += local_grad * v.grad`. The `+=` is the multivariable chain rule's sum over paths.

`init_parameters` creates the token table `wte` (27 × 16), the position table `wpe` (16 × 16), the query, key, value and output matrices (16 × 16 each), the MLP (`64 × 16` and `16 × 64`) and a separate output head `lm_head` (27 × 16): 4,192 parameters with no biases.

`gpt_forward` processes one position at a time. It adds the token and position embeddings, normalizes with RMSNorm, projects to queries, keys and values, and appends the key and value to per-layer lists. Each of the 4 heads scores its query against every key in the list, scaled by `1/sqrt(4)`, and mixes the values with the softmax weights. Because the lists only hold positions `0..t`, attention is causal without an explicit mask. A residual connection wraps the attention and the ReLU MLP, and `lm_head` produces logits.

`run_gpt` builds the vocabulary from the characters in `names.txt` (26 letters plus a BOS token that marks both ends of a name), trains for 1,000 steps on one shuffled name per step with the average next-character loss, calls `loss.backward()`, and applies Adam with bias correction and a linearly decaying learning rate. Sampling starts from BOS, divides the logits by a temperature of 0.5 and stops at the next BOS.

What differs from the paper: the vocabulary is characters, not BPE, and `microtokenizer.py` is not used; there is one layer of width 16 with RMSNorm instead of LayerNorm, ReLU instead of GELU and no biases or dropout; the output head is a separate matrix rather than `W_eᵀ`; and there is no fine-tuning stage, no `L2` or `L3`, and no warmup or cosine schedule. The source describes its layout as GPT-2's with these simplifications.

## Exercises

1. Build `z = x * y + x` with `x = Value(2.0)` and `y = Value(3.0)` and call `z.backward()`. Predict `x.grad` and `y.grad` before running. (`x.grad` is 4.0 — 3 from the product and 1 from the sum — and `y.grad` is 2.0.)
2. Before any training, the logits are near zero, so the first loss is close to `ln(27) ≈ 3.30`, the loss of a uniform guess. Print the loss of an untrained model on a few names and compare it with this value.
3. Tie the output head to the token embeddings, as the paper does (`P(u) = softmax(h_n W_eᵀ)`), by computing logits from `params['wte']` instead of `params['lm_head']`. How many parameters remain? (4,192 − 27 × 16 = 3,760.) Train both versions and compare their final losses.
4. Add the paper's auxiliary objective in a toy form: split names into two classes (for example, first letter before or after `m`), add a linear head on the last hidden state, and train with `L2 + 0.5 · L1`. Compare against training with `L2` alone.
5. Replace ReLU with a GELU approximation (`0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))`), deriving its local gradient for the `Value` class first.
