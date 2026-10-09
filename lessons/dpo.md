# DPO Lesson

Paper card: `papers/dpo.md`

Implementation: `no-magic/02-alignment/microdpo.py`

## Paper summary

Reinforcement learning from human feedback (RLHF) fits a reward model to preference pairs and then optimizes the language model against it with an RL algorithm such as PPO, under a KL penalty that keeps it near a reference policy `π_ref` (usually the supervised fine-tuned model): maximize `E[r(x, y)] − β·KL(π_θ(·|x) ‖ π_ref(·|x))` (Eq. 3). Rafailov et al. (2023) observe that this objective has a closed-form optimum, `π_r(y|x) = π_ref(y|x)·exp(r(x, y)/β) / Z(x)` (Eq. 4), which can be solved for the reward: `r(x, y) = β log(π_r(y|x)/π_ref(y|x)) + β log Z(x)` (Eq. 5). Under the Bradley–Terry model, the probability that `y_w` is preferred to `y_l` depends only on the reward difference, so the intractable `Z(x)` cancels. Substituting gives a loss on the policy itself (Eq. 7):

`L_DPO = −E[ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]`.

Training is a classification-style loss on a fixed preference dataset: no separate reward model, no sampling from the policy during training, and no RL loop. The gradient raises the likelihood of `y_w`, lowers that of `y_l`, and weights each pair by `σ(r̂(x, y_l) − r̂(x, y_w))`, where `r̂ = β log(π_θ/π_ref)` is the implicit reward: pairs the model currently orders wrongly count most. In the paper's experiments on sentiment control, summarization and single-turn dialogue, DPO matched or exceeded PPO-based RLHF.

## Intuition

DPO and RLHF start from the same objective; they differ in how they reach its optimum. RLHF learns a reward, then searches for a policy by sampling and scoring its own outputs. DPO notices that the optimal policy and the reward determine each other, so it fits the policy directly to the comparisons. What it gives up is online exploration: it learns only from the pairs in its dataset, so preference quality and coverage still limit the result.

`β` sets how far the policy may move. It weights the KL penalty in Eq. 3, so a larger `β` keeps the optimum closer to `π_ref`; in the loss, a larger `β` saturates the sigmoid after a smaller change in log-probability ratios. The reference matters in a second way: DPO only rewards the policy for preferring `y_w` by more than `π_ref` already does.

## Code walkthrough

`microdpo.py` pretrains a `microgpt`-sized model on names (700 steps) and copies its weights into `ref_params` with `snapshot_weights`; the reference is evaluated with plain floats by `gpt_forward_float`, so it never enters the autograd graph. `create_preference_pairs` groups names by their first 2 or 3 letters (its default prompt lengths) and, within a group, pairs a whole name of 5 or more letters (chosen) with a whole name of at most 3 letters (rejected), keeping at most 150 pairs. A prefix must be shorter than the name, so a 3-letter name enters only its 2-letter group and shorter names enter none. With the defaults, every rejected name therefore has exactly 3 letters and shares a 2-letter prefix with its chosen partner: the rejected completion after the prefix is one letter and the chosen one at least three, and the 3-letter groups never yield a pair. On `names.txt` this gives 335 pairs before the cap, such as `nicolette` over `niv`. These preferences are a length rule, not human judgments.

`sequence_log_prob_policy` and `sequence_log_prob_reference` sum next-token log-probabilities over the whole sequence from the BOS token. The prompt prefix is scored too, but it is identical in both completions, so its terms cancel in the margin. `dpo_loss` forms `delta = β·(log_ratio_chosen − log_ratio_rejected)` and returns `log(1 + exp(−delta))`, switching to `−delta` itself when `−delta > 20` so that `math.exp` cannot overflow; it also returns the implicit rewards `β·log_ratio` for monitoring. The training loop averages the loss over a batch of pairs, backpropagates and takes an Adam step, for 60 steps at `β = 0.1`.

At step 1 the policy equals its snapshot, so every log-ratio is zero and every pair's loss is `ln 2 ≈ 0.6931`; calling `dpo_loss` on eight real pairs in that state returned 0.69314718 each. The final comparison prints the average generated length of the reference and aligned models. One thing to read critically: the comment beside `DPO_BETA` says a low `β` barely moves the policy and a high `β` reshapes it aggressively. The paper's derivation says the opposite, and the paper is the authority here.

What the script does not show: human or AI-labelled preferences, a reward model, a PPO baseline to compare against, or evidence that DPO beats RLHF in general. `microppo.py` covers the RL route on names.

## Exercises

1. With `β = 0.1`, suppose the policy has moved so that `log_ratio_chosen = 0.5` and `log_ratio_rejected = −0.5`. Compute `delta` and the loss. (`delta = 0.1`; loss = `log(1 + e^(−0.1)) ≈ 0.644`.) What is the gradient weight `σ(r̂_l − r̂_w)` for this pair? (`σ(−0.1) ≈ 0.475`.)
2. Show that the shared prefix cancels: write the chosen and rejected log-ratios as prefix terms plus completion terms and subtract.
3. Run the script with `DPO_BETA` set to 0.01, 0.1 and 1.0 and compare the reference and aligned average lengths. Which direction does the paper predict, and what do you observe?
4. Replace the loss with the unweighted objective `−(log π_θ(y_w) − log π_θ(y_l))`, which has neither the reference nor the `σ(·)` weight, and train again. Watch the generated names and the log-probabilities of the chosen completions; the paper reports that a naive version without the weighting coefficient can make the model degenerate.
5. Build a second preference rule (for example, prefer names that end in a vowel) and train DPO on a mixture of both rules. Measure how often generations satisfy each rule.
