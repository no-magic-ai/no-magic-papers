# LoRA Lesson

Paper card: `papers/lora.md`

Implementation: `no-magic/02-alignment/microlora.py`

## Paper summary

Full fine-tuning updates every pretrained weight, so each task needs a complete copy of the model and optimizer state for all of it. Hu et al. (2021) instead freeze the pretrained matrix `W₀ ∈ R^(d×k)` and learn its update as a product of two thin matrices: `h = W₀x + ΔWx = W₀x + BAx`, with `B ∈ R^(d×r)`, `A ∈ R^(r×k)` and `r ≪ min(d, k)`. `A` starts from a random Gaussian and `B` starts at zero, so `ΔW = BA = 0` and training begins exactly at the pretrained model. The update is scaled by `α/r`; the authors set `α` to the first `r` they try and do not tune it. After training, `W = W₀ + BA` can be computed once, so inference has no extra latency, and switching tasks means subtracting one `BA` and adding another.

The paper applies LoRA only to attention weights and leaves the MLP frozen. With a fixed budget, adapting the query and value projections together worked best among the combinations it tried, and very small ranks were competitive. For GPT-3 175B it reports 10,000× fewer trainable parameters, training memory down from 1.2 TB to 350 GB, and task scores comparable to or better than full fine-tuning on its benchmarks: on WikiSQL, for example, full fine-tuning scored 73.8, LoRA with 4.7M parameters 73.4 and LoRA with 37.7M parameters 74.0. The authors also note that LoRA applied to every weight matrix with `r` equal to the full rank roughly recovers full fine-tuning's expressiveness.

## Intuition

Full fine-tuning can move a weight matrix anywhere; LoRA restricts the move to matrices of rank at most `r`, whose outputs change only along `r` directions. The paper's bet is that the change a task needs has low "intrinsic rank", so this restriction costs little while cutting trainable parameters and optimizer memory by orders of magnitude. It is an empirical claim about the tasks studied, not a theorem: a task that needs a high-rank change would expose the limit.

Initialization decides which factor learns first. With `y = W₀x + B(Ax)`, the gradient of `B` is `(∂L/∂y)(Ax)ᵀ` and the gradient of `A` is `Bᵀ(∂L/∂y)xᵀ`. With the paper's `B = 0` and random `A`, the first step changes `B`, the output-side factor, and leaves `A` alone. Either way the model starts at the pretrained function, because one of the two factors is zero.

## Code walkthrough

`microlora.py` first trains a base model on names A–M (800 steps), then freezes it and adapts to names N–Z (500 steps). `init_lora_adapters` creates, for the query and value projections, `A` with shape `[N_EMBD, LORA_RANK]` = 16 × 2 from `N(0, 0.02)` and `B` with shape 2 × 16 set to zero. `lora_linear` returns `W_frozen @ x + A @ (B @ x)`: `B` maps the 16-dimensional input down to 2 numbers and `A` maps them back up to 16.

So the script writes the update as `W + A·B` with the zero factor on the input side. This is not the paper's initialization with the names swapped: here the random factor is the output-side one. Running the script's own adaptation loop for one step on an untrained base, with real names, gave a summed absolute gradient of 0.0 for `A` and about 0.0105 for `B`; after the Adam update `B` had changed and `A` and the base weights had not. In the paper's arrangement the output-side factor would be the one to move first.

During adaptation `loss.backward()` still computes gradients for every base weight, because they are `Value` nodes in the graph. Lines that zero those gradients keep them from piling up, but the freezing itself comes from the optimizer loop, which updates only the adapter parameters. The results line reports `Full fine-tune: 4,192 | LoRA: 128 (3.1%)`; the first number is the base parameter count, not a trained full fine-tuning baseline, which the script never runs. It also omits the `α/r` scaling. The final losses on both splits compare the base and adapted models only.

## Exercises

1. Count adapter parameters for rank `r` in this script: each adapted projection adds `16r + 16r`, so two projections add `64r`. Check `r = 2` against the printed 128 and predict `r = 6` (384, about 9.2% of the base).
2. Implement the paper's initialization: make the output-side factor (16 × 2) zero and the input-side factor (2 × 16) random. Run one adaptation step and confirm which factor now receives the nonzero gradient.
3. Add a real full fine-tuning arm: copy the base model, train all 4,192 parameters on N–Z for 500 steps with the same optimizer, and compare N–Z and A–M loss with the LoRA run. Repeat with a few seeds before drawing a conclusion.
4. Merge the adapters: compute `W_q + A_q B_q` and `W_v + A_v B_v`, run the plain forward pass with the merged weights, and check that the logits match the adapter path to floating-point precision.
5. Add the `α/r` scaling with `α = r`, then vary `r` while keeping `α` fixed, and observe how the effective step size of the update changes.
