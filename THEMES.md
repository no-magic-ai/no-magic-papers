# Themes

Every paper card has one primary theme and zero to two secondary themes. These themes are navigation and batching keys, not free-form tags.

| Theme | Definition | Example papers |
|---|---|---|
| `efficient-inference` | Quantization, pruning, distillation, speculative decoding, and KV-cache optimization that reduce inference cost without changing the task. | TurboQuant, FlashAttention, speculative decoding |
| `long-context` | Attention variants, memory architectures, RoPE extensions, and sparse or hierarchical attention methods built to extend usable context length. | Titans, LongRoPE, sliding-window attention |
| `alignment` | RLHF, DPO, PPO, GRPO, KTO, SimPO, preference optimization, and reward modeling methods that shape model behavior after or during training. | DPO, GRPO, PPO |
| `reasoning` | Chain-of-thought, self-consistency, process rewards, test-time compute, self-play, and tree-search reasoning methods. | DeepSeek-R1, self-consistency |
| `architecture` | New model designs such as SSMs, MoE, hybrid attention, tokenization-free models, and differential or sparse attention. | Mamba-2, Native Sparse Attention, MoE |
| `training-dynamics` | Optimizers, scaling laws, curriculum learning, pretraining recipes, data ordering, and other papers about how training evolves. | scaling laws, curriculum learning |
| `parameter-efficient` | LoRA, DoRA, adapters, prefix tuning, prompt tuning, and related techniques that adapt models with few trainable parameters. | LoRA, QLoRA, prefix tuning |
| `interpretability` | Mechanistic interpretability, knowledge editing, probing, sparse autoencoders, and causal tracing methods. | ROME, MEMIT, sparse autoencoders |
| `retrieval` | RAG, vector search, reranking, dense retrieval, hybrid search, and retrieval-augmented model designs. | RAG, BM25, dense retrieval |
| `safety-robustness` | Jailbreaks, watermarking, red-teaming, bias, adversarial robustness, and methods that measure or improve safe behavior. | watermarking, jailbreak defense |
| `agents` | Planning, tool use, memory-augmented agents, multi-agent coordination, and computer-use agents. | ReAct, MCTS agents |
| `multimodal` | Vision-language, audio, video, and cross-modal fusion methods. | CLIP-style models, VLMs |
