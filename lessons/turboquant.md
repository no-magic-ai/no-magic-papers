# TurboQuant Lesson

Paper card: `papers/turboquant.md`

Implementation: `no-magic/03-systems/microturboquant.py`

## Paper summary

TurboQuant is a vector quantization method for settings where vectors arrive online and calibration is either unavailable or too expensive. The paper targets two distortion measures: reconstruction MSE and inner-product error. That distinction matters because a quantizer that reconstructs vectors well can still bias dot products, which are the actual primitive in attention and nearest-neighbor search.

The MSE path rotates each input vector with a shared random orthogonal matrix. Once a unit vector is randomly rotated, each coordinate follows a dimension-dependent distribution rather than a distribution tied to the original vector. That lets the quantizer use the same scalar codebook on every coordinate. The inner-product path then takes the MSE residual and stores one-bit QJL signs so the dot-product estimator is unbiased.

## Intuition

Quantization is hard when one coordinate dominates. A per-vector absmax quantizer must choose a scale large enough for the biggest coordinate, so small coordinates get rounded away. Random rotation spreads that directionality across coordinates. After rotation, the vector is dense and its coordinates have similar statistical scale.

That does not make information free. The rotation changes the shape of the quantization problem so a simple scalar quantizer becomes a reasonable approximation to the harder vector quantizer. TurboQuant's theoretical point is that this approximation is near the rate-distortion lower bound for the target setting, not just a convenient heuristic.

Inner products need another step. If the reconstructed vector is slightly shrunk toward centroids, dot products inherit that shrinkage as bias. QJL fixes the residual path by storing signs of random projections. Each sign is only one bit, but averaging many signs gives an unbiased angular estimate, so the residual can correct the dot product without storing another full vector.

## Code walkthrough

Start in `microturboquant.py` with `random_rotation`. The script samples a Gaussian matrix and orthonormalizes its columns with Gram-Schmidt. That is the pure-Python way to get a Haar-like random rotation small enough for a teaching script.

The baseline is `absmax_quantize`: pick a signed integer grid, scale by the largest absolute coordinate, round, and clip. This baseline is intentionally simple because it exposes the failure mode. On anisotropic vectors, one large coordinate controls the scale for every coordinate.

The TurboQuant path is `turboquant_encode` and `turboquant_decode`. Encoding computes `R @ x`, applies the scalar quantizer, and stores integer codes plus the scale. Decoding multiplies codes by the scale and applies `R.T`. The implementation uses absmax instead of precomputed Lloyd-Max centroids because the teaching goal is the rotation effect, not reproducing the paper's optimized scalar codebooks.

The residual path is split into `qjl_signs` and `qjl_estimate_inner_product`. The first projects a residual vector onto random Gaussian rows and stores only signs. The second averages paired signs and rescales the result. In production, the estimator uses the full angular inversion; the script keeps the linearized form so the mechanism is visible.

Read the script as three experiments: compare scalar quantization before and after rotation, check that the rotation is orthogonal, and measure how residual signs reduce inner-product error variance as the number of projections grows.

## Exercises

1. Replace the absmax scalar quantizer in the TurboQuant path with fixed centroids for a standard normal distribution. Keep the same encode/decode interface and compare MSE at 2 and 4 bits.
2. Swap the dense random rotation for a structured sign-flip plus Hadamard-style transform on power-of-two dimensions. Measure the runtime difference before comparing distortion.
3. Change the synthetic data anisotropy from mild to extreme. Identify the point where rotating stops helping because the input vectors become nearly one-sparse.
4. Implement the arccos inversion in the QJL estimator and compare bias against the linearized estimator on residual vectors with high cosine similarity.
