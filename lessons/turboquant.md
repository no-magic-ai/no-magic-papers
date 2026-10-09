# TurboQuant Lesson

Paper card: `papers/turboquant.md`

Implementation: `no-magic/03-systems/microturboquant.py`

## Paper summary

TurboQuant is by Amir Zandieh, Majid Daliri, Majid Hadian and Vahab Mirrokni: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874, 2025). The header comment of `microturboquant.py` and the implementation guide (`no-magic/docs/implementation.md`) cite "Aamand et al." and the title "…with Optimal Bit Budget"; that attribution and title are incorrect.

The paper quantizes vectors online, without calibration data, and with guarantees that hold for every input vector rather than on average over a dataset. It targets two distortion measures: the reconstruction error `E‖x − x̃‖²` and the inner-product error `E|⟨y, x⟩ − ⟨y, x̃⟩|²` against a query `y` that stays at full precision, and for inner products it also requires the estimate to be unbiased, `E⟨y, x̃⟩ = ⟨y, x⟩`.

The MSE quantizer randomly rotates the unit vector `x`. After rotation each coordinate follows a known Beta distribution that depends only on the dimension, and distinct coordinates are nearly independent in high dimension, so each coordinate can be quantized on its own with a scalar codebook precomputed by the Lloyd–Max algorithm for that distribution; dequantization looks up the centroids and rotates back. MSE-optimal quantizers bias inner products — at one bit the estimate is scaled by `2/π` — so the inner-product quantizer runs the MSE quantizer with `b − 1` bits, applies a one-bit Quantized Johnson–Lindenstrauss (QJL) map to the residual `r = x − x̃_mse`, and stores `‖r‖`. QJL stores `sign(S·r)` for a Gaussian matrix `S ∈ R^(m×D)` with `m` projection rows and reconstructs `sqrt(π/2)/m · ‖r‖ · Sᵀ sign(S·r)`. Each Gaussian row `s` satisfies `E[s·sign(sᵀx)] = sqrt(2/π)·x/‖x‖`, so averaging `m` rows with that factor recovers `x` in expectation; for a full-precision query this gives an unbiased inner-product estimate with variance at most `π/(2m)·‖y‖²` for unit vectors. The paper takes `S` square, `m = D = d`, which is why its Definition 1 and Lemma 4 divide by the dimension `d`. The paper proves upper bounds for both quantizers within a small constant factor of information-theoretic lower bounds and reports, among other results, neutral long-context quality at 3.5 bits per channel for KV-cache compression.

## Intuition

A per-vector scalar quantizer struggles when one coordinate dominates: its scale is set by the largest value and the small coordinates round away. A shared random rotation makes every unit vector look statistically alike — after rotation the coordinates have the same distribution whatever the input — so one fixed codebook serves all vectors, and the worst case is no worse than the average case. Nothing is learned from data, which is what makes the method usable on vectors that arrive one at a time.

Reconstruction error and inner-product error are different objectives. A quantizer that is optimal for reconstruction tends to shrink vectors toward its centroids, which shrinks dot products on average. QJL fixes this for the small residual: the signs of random projections, scaled correctly and multiplied against the unquantized query, average to the true inner product.

## Code walkthrough

`random_rotation` fills a 32×32 matrix with Gaussian entries and orthonormalizes the columns with (modified) Gram–Schmidt; `main` asserts that `max|RᵀR − I| < 1e-10`. `absmax_quantize` is the baseline: one scale per vector, `2^(bits−1) − 1` integer levels per sign, and a single level when `bits` is 1, so the 1-bit and 2-bit settings both use the grid `{−1, 0, +1}`. `turboquant_encode` rotates and then applies the same absmax quantizer; `turboquant_decode` rescales and applies `Rᵀ`. That is the paper's rotate-then-quantize structure with a per-vector absmax grid in place of the precomputed Lloyd–Max codebooks.

`rate_distortion_table` compares inner-product error with and without rotation at 1, 2, 4 and 8 bits on 300 anisotropic synthetic vectors and 300 name embeddings. In one run, rotation helped the synthetic vectors at 4 and 8 bits (1.61× and 1.83× lower error) and hurt at 1 and 2 bits (0.50× and 0.49×), while the name embeddings, which are already dense, changed little.

`qjl_signs` and `qjl_estimate_inner_product` are where the script departs furthest from the paper. The demo signs both vectors — whole vectors, not residuals — and returns `(π/2) × mean(sign(S·a) · sign(S·b))`. The expected sign agreement for Gaussian projections is `1 − 2·arccos(ρ)/π`, so this estimate's expectation is `arcsin(ρ)`, not the cosine `ρ`: identical vectors give `π/2 ≈ 1.571`. At `ρ = 0.5` the expected agreement is exactly 1/3 and the expected estimate is `arcsin(0.5) = π/6 ≈ 0.5236`; a finite draw scatters around it with standard deviation `(π/2)·sqrt((1 − 1/9)/K)`, about 0.007 for `K = 40,000` projections, and one such draw with the script's functions returned 0.526. It is unbiased only at `ρ = 0`, and it ignores the vectors' norms. Its docstring calls it unbiased and says production QJL inverts the arccos; the paper instead keeps the query unquantized, as described above. No residual path, Lloyd–Max codebook or stored residual norm is implemented.

## Exercises

1. Replace the absmax quantizer in the TurboQuant path with fixed centroids for the rotated-coordinate distribution (approximately `N(0, 1/D)` at `D = 32`) at 2 and 4 bits, keep the encode/decode interface, and compare reconstruction error.
2. Implement the paper's QJL: store `sign(S·x)` for one vector, keep the other at full precision, and estimate `⟨y, sqrt(π/2)/m · ‖x‖ · Sᵀ sign(S·x)⟩`, where `S` has shape `m × D`. If you reuse the demo's `S` (256 rows over 32 dimensions), divide by 256, not 32; dividing by the dimension inflates every estimate eightfold. Measure the mean signed error over many pairs and compare it with the paired-sign demo.
3. Build the inner-product quantizer: quantize with `b − 1` bits, compute the residual and its norm, apply your QJL from exercise 2, and add the two estimates. Compare its inner-product error and bias with the MSE-only quantizer at the same total bits.
4. Compute the paired-sign estimator's expectation `arcsin(ρ)` for `ρ = 0.1, 0.5, 0.9` and confirm it empirically. Then apply `sin(π/2 · agreement)` and measure how much bias remains with 256 projections.
5. Swap the dense rotation for random sign flips followed by a Hadamard transform on a power-of-two dimension. Measure the encoding time and the inner-product error against the Gram–Schmidt rotation.
