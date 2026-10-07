# RoPE Lesson

Paper card: `papers/rope.md`

Implementation: `no-magic/03-systems/microrope.py`

## Paper summary

Su et al. (2021) encode position by rotating queries and keys instead of adding a position vector to the input. Split a `d`-dimensional query or key into `d/2` pairs. At position `m`, pair `i` is rotated by the angle `m·θ_i`, with `θ_i = 10000^(−2(i−1)/d)` for `i = 1, …, d/2`; the full operation is a block-diagonal orthogonal matrix `R_(Θ,m)` (Eq. 15). Applied to attention:

`q_mᵀ k_n = (R_(Θ,m) W_q x_m)ᵀ (R_(Θ,n) W_k x_n) = x_mᵀ W_qᵀ R_(Θ,m)ᵀ R_(Θ,n) W_k x_n = x_mᵀ W_qᵀ R_(Θ,n−m) W_k x_n`,

because `R_(Θ,m)ᵀ R_(Θ,n) = R_(Θ,n−m)`. The paper's printed Eq. 16 writes `W_q` without the transpose; expanding the product shows the transpose belongs there, and the two forms agree only when `W_q` is symmetric. Each token is rotated by its absolute position, yet the score depends only on the offset `n − m`. The rotation preserves vector norms, adds no parameters, and can be applied with element-wise multiplications instead of a matrix product. The paper also shows that, with this choice of `θ_i`, an upper bound on the score decays as the relative distance grows, and that RoPE can be combined with linear attention. The resulting RoFormer is evaluated on translation, language-model pre-training and long Chinese text classification; the authors note that they lack a full explanation for its faster convergence and its gains on long texts.

## Intuition

Think of each pair of dimensions as a hand on a clock. A token at position `m` turns every hand by `m` times that hand's speed. When a query and a key meet in a dot product, only the angle between their hands matters, and that angle is the speed times the distance between the tokens. Fast hands (the first pairs) distinguish nearby positions; slow hands (the last pairs) keep turning slowly enough to separate distant ones.

Relative dependence is a property of each score, not a guarantee about lengths a model never saw. A trained model has only experienced the angle combinations that occur within its training length; beyond it, the fast pairs wrap around and the slow pairs reach angles never seen. Context extension methods such as NTK-aware scaling change the frequencies to keep angles within familiar ranges.

## Code walkthrough

`rope_frequencies(d)` returns `θ_i = 10000^(−2i/d)` for `i = 0, …, d/2 − 1` (the paper's formula with zero-based indexing). `apply_rope(vec, pos, freqs)` rotates each pair `(x_2i, x_2i+1)` by `pos·θ_i`, and `rope_attention_score` rotates a query and a key by their own positions and takes the scaled dot product.

`demonstrate_relative_position_property` scores the same query and key at several absolute positions with the same offset of 3. The RoPE scores agree to about `1e-17`, while additive sinusoidal encodings give scores that differ by about 0.41 — they are not purely relative. With a 16-dimensional head the first pair turns once every `2π ≈ 6.3` positions and the last pair, `θ_7 = 10000^(−14/16) ≈ 3.16 × 10⁻⁴`, once every 19,869 positions; the spectrum table prints both.

`demonstrate_length_extrapolation` compares scores beyond the 64-entry learned position table: the learned table simply has no entry there, while sinusoidal, RoPE and NTK-scaled RoPE scores can still be computed. These are untrained scores, so they show what each encoding can represent, not how a trained model behaves at those lengths.

`ntk_scaled_frequencies` raises the base to `10000 · s^(d/(d−2))` for a length factor `s`. Read the effect carefully: for `s = 4` and `d = 16` the first pair's speed is unchanged and the last pair is slowed by exactly 4×, with the pairs in between slowed by 1.22× to 3.28×. The source comment says high frequencies "get slowed down more than low frequencies", and the demo prints "Higher scale factors slow all frequencies proportionally"; both descriptions are inaccurate — the lowest frequencies change most and the highest not at all.

## Exercises

1. Rotate the unit vector `(1, 0)` in the first pair to position 1. Predict the result. (`(cos 1, sin 1) ≈ (0.540, 0.841)`.)
2. Prove `R(m)ᵀ R(n) = R(n − m)` for a single 2×2 rotation using the angle-difference identities, then confirm numerically that the scores for positions (5, 8) and (100, 103) agree. With a non-symmetric 2×2 `W_q`, also check that `x_mᵀ W_qᵀ R(n − m) W_k x_n` matches the rotated score and `x_mᵀ W_q R(n − m) W_k x_n` does not.
3. Show that `apply_rope` preserves the vector's length for any position, and explain why that matters for attention scores.
4. For `s = 4`, compute the ratio between standard and NTK-scaled frequencies for each pair. (They run from 1.0 for the first pair to 4.0 for the last.) Explain why the last pair is slowed by exactly `s`.
5. Add RoPE to `microgpt.py`: remove `wpe`, rotate each head's query and key by position inside `gpt_forward`, and compare the parameter count (4,192 − 256 = 3,936) and final loss with the learned-position version.
