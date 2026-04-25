# DeepSeek-R1 Lesson

Paper card: `papers/deepseek-r1.md`

Implementation status: `micror1.py` is not implemented yet. This draft uses `no-magic/02-alignment/microgrpo.py` as the partial code stand-in for the GRPO core.

## Paper summary

DeepSeek-R1 is less a single algorithm than a post-training recipe for reasoning. The clean experimental point is R1-Zero: start with a base model, skip supervised reasoning traces, train with GRPO on verifiable prompts, and let reasoning behavior emerge from reward pressure. The production-ready point is R1: add cold-start examples, another reasoning RL stage, rejection-sampled SFT, broad-domain SFT, and final RL for usefulness and safety.

The paper’s claim is not that SFT is useless. It shows that pure RL can discover reasoning behaviors, then shows why SFT still matters for readability, language consistency, and distillation.

## Intuition

GRPO turns a prompt into a small local tournament. For one question, the policy samples several completions. A deterministic reward scores each completion. The group mean becomes the baseline, so each response is judged relative to its siblings rather than against a learned value model.

That relative baseline is the lever. If a response finds a better strategy, gets the format right, or produces the correct final answer, it receives positive advantage inside the group. If it rambles or misses the answer, it receives negative advantage. Repeating this over many verifiable tasks pushes the model toward behaviors that improve reward: longer thinking, checking previous steps, trying alternatives, and formatting the answer predictably.

R1-Zero demonstrates emergence but also exposes the raw failure modes: mixed languages, awkward formatting, and hard-to-read traces. R1 adds human-shaped data and filtering after the RL discovery step so the final model is useful to people, not only optimized for verifiers.

## Code walkthrough

Read `microgrpo.py` as the minimum executable slice of the R1 training signal. The script samples multiple completions per prompt with `GROUP_SIZE`, scores them with a reward function, normalizes rewards inside the group, and applies a policy update with a KL penalty against a reference model.

The missing `micror1.py` layer should add three pieces around that core. First, a cold-start path: a tiny set of readable worked examples that initializes the policy before reasoning RL. Second, rejection sampling: generate several candidate traces after RL, keep the correct and readable ones, and use them as supervised data. Third, a distillation path: train a smaller policy on accepted traces from the stronger policy.

The paper’s reward design also matters. R1-Zero uses rule-based rewards for correctness and format because model-based process rewards can be hard to define and can be exploited. In a toy implementation, use arithmetic or string tasks where correctness is deterministic. Keep the format reward separate from the accuracy reward so readers can see how models learn both solving and packaging.

Do not treat the `<think>` tags as the lesson. The lesson is the optimization loop: verifiable rewards create a gradient toward better reasoning traces, and the model spends more test-time tokens because that behavior improves group-relative reward.

## Exercises

1. In `microgrpo.py`, split the reward into accuracy and format components. Log each component separately and find cases where format improves before accuracy.
2. Add a small cold-start dataset of worked arithmetic traces. Compare early reward curves with and without that warm start.
3. Implement rejection sampling after GRPO: generate four traces per prompt, keep only correct answers, and run a short supervised update on the accepted traces.
4. Train a smaller student policy on traces from the stronger policy. Compare direct RL on the student against distillation from the stronger policy.
