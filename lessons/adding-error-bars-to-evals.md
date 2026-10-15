# Adding Error Bars to Evals Lesson

Paper card: `papers/adding-error-bars-to-evals.md`

Implementation: `no-magic/01-foundations/microerrorbars.py`

## Paper summary

Miller (2024, arXiv 2411.00640) treats an eval score as a sample mean over questions drawn from a super-population. For a single model the standard error is `SE = sqrt(Var(s)/n)`, estimated with the sample variance (Eq. 1), and the 95% interval is the mean plus or minus `z·SE` (Eq. 3). For two models scored on the same questions, the unpaired difference adds the two variances (Eq. 5); the paired difference uses per-question differences: the variance of the paired estimate is `(Var(s_A) + Var(s_B) − 2Cov(s_A, s_B))/n` (Sec. 4.2, Eq. 7). Positive correlation between the models therefore shrinks the error bar.

The same variance answers a design question. With Type I rate `α`, Type II rate `β` and a paired analysis, `n = (z_{α/2} + z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ²` (Eq. 9), where `ω² = Var(x_A) + Var(x_B) − 2Cov(x_A, x_B)` is Miller's Sec. 5 shorthand for the conditional-mean part of the paired-difference variance, and the `σ²/K` terms add answer-sampling variance; in Part B the scores are deterministic predictions (`σ² = 0`), so the printed `omega^2` equals the pool `Var(s_A − s_B)`. Eq. 10 inverts it into a minimum detectable effect. Miller's worked example, with fictional parameters `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20`, gives `n ≈ 969`. Sec. 2.2 adds clustered standard errors for questions that share a passage; the Table 4 caption says that, analyzing the same data, clustered standard errors "can be over 3X larger" than naive ones on two evals. Miller calls bootstrapping unnecessary for a plain mean (Sec. 2.1) and applies the CLT to evals with "a large number of questions".

## Intuition

Three ideas stack. First, a score is an estimate: ask different questions and you would get a different number, so a gap smaller than the error bar is not a ranking. Second, comparing two models on the same questions cancels what both find hard or easy. Items that both models get right or both get wrong add nothing to the difference, so the paired error bar is narrower. Third, the error bar sets the price of detection: `n` scales with `1/δ²`, so a smaller effect costs disproportionately more questions.

The program adds a question Miller does not ask: do these formulas hold when `n` is small? Miller's CLT applies to large evals, so the small-`n` coverage study and the Wilson interval are this program's extension. The Wilson reference follows Brown, Cai and DasGupta (2001, doi 10.1214/ss/1009213286), who recommend the Wilson or equal-tailed Jeffreys prior interval for small `n`. The unpaired power rows also go beyond Eq. 9, which assumes a paired analysis.

Keep the scope in view: toy synthetic arms, one seed, binary scores with `K = 1`, no clustering. Nothing below is a measurement of a real LLM eval.

## Code walkthrough

Intervals. `clt_interval` (lines 170–181) is Eq. 1 plus Eq. 3: mean ± `z·sqrt(var/n)` with `z = 1.959964`, shared by every interval. It is symmetric and has zero width when all scores agree. `bootstrap_percentile_interval` (lines 184–200) resamples `n` scores with replacement `BOOTSTRAP_RESAMPLES` times and picks the lower and upper sorted resample means by integer index arithmetic (lines 198–199). `wilson_interval` (lines 203–219) is the Wilson score interval for `k` successes in `n`.

Part A. `draw_scores` (lines 225–236) draws Bernoulli scores at a known mean; `run_part_a` (line 340) counts coverage over `EVALS_PER_CELL` evals per cell. Look at two rows:

- `[A] mu=0.95 n= 30 CLT cov=0.797 ... zero_w=0.201`: `zero_w=0.201` is the share of evals whose interval had exactly zero width (all answers equal), so the CLT interval was a single point and missed.
- `[A] mu=0.95 n=100 CLT cov=0.863` against `bootstrap cov=0.938` and `Wilson cov=0.953`: here the bootstrap did much better than the CLT. At `mu=0.95 n=30` it did not (`bootstrap cov=0.789`, `Wilson cov=0.934`), because it inherits the same degenerate samples.

At `mu=0.50` all three methods sit between 0.955 and 0.970, and the printed `n=500` cells span 0.934–0.960. The Monte Carlo SE is 0.005–0.013, so small coverage differences are noise; compare rows with the `mcse` column in view.

Part B. `run_part_b` (line 418) trains two logistic regressions on synthetic data, A with 8 features and B with 7 (`ARM_A_FEATURES`, `ARM_B_FEATURES`), and scores both on a pool of `POOL_SIZE` items: `pool_acc=0.8936` for A and `0.8425` for B, a pool difference of `0.0511`. `pool_statistics` (line 384) computes `Var(s_A)=0.0951`, `Var(s_B)=0.1327`, `Cov=0.0509`, and `Corr(s_A, s_B)=0.4530`. `difference_intervals` (lines 400–415) builds the unpaired (Eq. 5) and paired (Eq. 7) intervals. The printed rows:

- `n= 30 unpaired cov=0.985 w=0.3391 | paired cov=0.919 w=0.2504 | ratio=0.738`
- `n=100 unpaired cov=0.990 w=0.1863 | paired cov=0.943 w=0.1374 | ratio=0.738`
- `n=500 unpaired cov=0.990 w=0.0836 | paired cov=0.963 w=0.0622 | ratio=0.744`

The paired interval is narrower at every `n`, by the printed ratios 0.738, 0.738 and 0.744. The unpaired interval over-covers, 0.985–0.990, because it ignores the covariance, so it is conservative here. The paired coverage at `n=30` is 0.919, below the 0.95 target: the paired interval is a CLT interval and inherits the small-`n` weakness.

Part C. `minimum_detectable_effect` (lines 485–490) is Eq. 10 and `required_sample_size` (lines 493–498) is Eq. 9; `run_part_c` (line 501) prints them. The paired variance is `omega^2 = 0.1260` and the unpaired variance is `Var(s_A)+Var(s_B) = 0.2278`. Rows include `MDE n= 100 paired mde=0.0995` against `unpaired mde=0.1337`, and `required n delta=0.05 paired n=396` against `unpaired n=716`. The line `Miller example omega^2=1/9 delta=0.03 n=968.997` reproduces his 969. The last row is a power check at the pool difference: `n=379 delta=0.0511 simulated=0.815+/-0.012 predicted=0.801`. Treat that as arithmetic agreement for one seed, not validation of the method on real evals.

Gates (`run_gates`, line 583) check code correctness only: training, arithmetic, and row counts. No effect size is ever tested.

## Exercises

1. At `mu=0.95 n=30` the table shows `zero_w=0.201` and CLT coverage 0.797. If an eval of 30 questions scores 30/30, what are the CLT standard error and interval, and what does the Wilson interval at `n=30` do differently? (CLT: the sample variance is zero, so SE is zero and the interval is a single point at the sample mean. Wilson never has zero width; its mean width in that row is 0.1736, and its coverage is 0.934.)
2. Using the printed `Var(s_A)=0.0951`, `Var(s_B)=0.1327`, `Cov=0.0509`, compute `Var(s_A − s_B)` and the unpaired variance. (`0.0951 + 0.1327 − 2·0.0509 = 0.1260`; unpaired `0.0951 + 0.1327 = 0.2278`, matching Part C.) Take the square root of the ratio of the two and compare it with the printed width ratios 0.738, 0.738 and 0.744. (Work it out; the answer should be close to those ratios.)
3. Eq. 9 has `n` proportional to `1/δ²`. Compare the printed paired required-n values 9893, 2474 and 396 at `delta=0.01`, `0.02` and `0.05`. Is the ratio between neighbouring rows what `(δ₁/δ₂)²` predicts? (Work it out from the printed values; each pair of rows should follow the square law.)
4. From Eq. 10, the minimum detectable effect scales with the square root of the variance. Predict the ratio of the printed unpaired to paired `mde` at `n=100` from the variances 0.2278 and 0.1260, then compare with `0.1337` and `0.0995`. (Work it out from the printed values.)
5. Run the script with a second seed, or change `TRUE_MEANS`, and report whether `mu=0.95 n=100` still shows bootstrap coverage clearly above CLT coverage. Compare the gap with the `mcse` column before you call it real. (No printed answer: this is an experiment; the program ran one seed.)
