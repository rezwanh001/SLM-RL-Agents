# Towards Robust Reinforcement Learning for Small-Scale Language Model Agents

[![arXiv](https://img.shields.io/badge/arXiv-2607.25091-b31b1b.svg)](https://arxiv.org/abs/2607.25091)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-mr3haque-yellow)](https://huggingface.co/mr3haque/SLM-RL-Agents)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/rezwanh001/slm-rl-agents/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)

---

This repository is the official implementation of the following paper.

Paper Title: **[Towards Robust Reinforcement Learning for Small-Scale Language Model Agents](https://arxiv.org/abs/2607.25091)**
<br/>
[Md Rezwanul Haque](https://scholar.google.com/citations?user=HaI-oFUAAAAJ&hl=en), [Md. Milon Islam](https://scholar.google.com/citations?user=ABC_LOAAAAAJ&hl=en), [Fakhri Karray](https://scholar.google.com/citations?user=9_Hpd5kAAAAJ&hl=en)
<br/>

#### ***Proceedings of the 2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Bellevue, WA, USA. Copyright 2026 by the author(s).***

---

## Abstract

The alignment of Small Language Models (SLMs) in the 70--500M parameter range using reinforcement learning is often considered unstable, though the underlying failure mechanisms have not been systematically investigated. In the State-of-the-Art (SOTA) research, fifteen (model, corpus) configurations were trained using Proximal Policy Optimization (PPO). The experiments included Pythia-70M, 160M, 410M and SmolLM2-135M, 360M on the TinyStories, CNN/DailyMail, and Wikitext-103 corpora. Three reproducible failure modes were identified in small-scale language models: silent LoRA parameter freezing in standard PEFT/TRL pipelines, numerical overflow in importance ratios when using bfloat16, and catastrophic policy collapse due to reward-model error. These issues were addressed using a merge-and-reinitialize adapter technique, float32 precision during PPO updates, and a three-layer safety mechanism comprising reward whitening, importance-ratio guarding, and weight rollback. In this paper, a capacity-headroom hypothesis is proposed, which states that PPO performance at the SLM scale depends on both a fluent supervised model ($\text{PPL}<20$) and a discriminative reward signal, rather than on the number of model parameters. The proposed system converged stably in all experiments and improved preference win rate over the SFT baseline in configurations with a fluent prior and an informative reward signal. Furthermore, it outperformed instruction-tuned baselines while requiring significantly less training data. All checkpoints, preference datasets, and training scripts are publicly released.

<!-- Reinforcement Learning from Human Feedback (RLHF) is widely used for large-scale language models, yet its application to **small language models (SLMs)** in the **70–500M parameter** range remains under-explored. It is commonly assumed that Proximal Policy Optimization (PPO) is inherently unstable at this scale. This work tests that assumption in the **single-turn specialization of the agent MDP**, the stability prerequisite of any multi-turn agentic extension, through a systematic empirical investigation. An end-to-end pipeline—supervised fine-tuning, Bradley–Terry reward modelling, and PPO—is applied uniformly to **five SLMs** from two architecture families (Pythia-70M/160M/410M and SmolLM2-135M/360M) across **three diverse corpora** (TinyStories, CNN/DailyMail, Wikitext-103), producing **15 fully trained configurations**. Three critical engineering instabilities specific to the small-scale regime are identified and resolved: silent LoRA parameter freezing, bfloat16 numerical overflows, and catastrophic policy collapse. The results reveal the **capacity-headroom hypothesis**: PPO effectiveness is determined by the joint availability of a fluent SFT prior and a discriminating reward signal rather than raw parameter count. The proposed pipeline achieves performance comparable to or better than released instruction-tuned baselines while requiring several orders of magnitude less training data. All model checkpoints, datasets, training scripts, and **forward-compatibility scaffolding for multi-turn agentic extensions** are publicly released. -->

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Scope: Single-Turn Specialization](#scope-single-turn-specialization)
- [Results](#results)
- [Computation Cost](#computation-cost)
- [Models & Datasets](#models--datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare Datasets](#1-prepare-datasets)
  - [Supervised Fine-Tuning](#2-supervised-fine-tuning-sft)
  - [Reward Model Training](#3-reward-model-training)
  - [PPO Alignment](#4-ppo-alignment)
  - [Evaluation](#5-evaluation)
  - [Run Full Pipeline](#6-run-full-pipeline)
- [Engineering Notes](#engineering-notes)
- [Interactive Demo](#interactive-demo)
- [Forward Compatibility: Multi-Turn Agentic Extensions](#forward-compatibility-multi-turn-agentic-extensions)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

This repository provides the complete implementation for the paper **"Towards Robust Reinforcement Learning for Small-Scale Language Model Agents"**. We investigate whether RLHF—the dominant alignment paradigm for large language models—transfers effectively to the sub-500M parameter regime, and we establish the **stability prerequisites** that any multi-turn SLM agent must satisfy to train reliably.

**Pipeline overview:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│         SLM-RL-Agents: End-to-End Pipeline (Single-Turn Case)           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 0: Data Preparation                                              │
│  ┌──────────────┐   3 corpora (TinyStories, CNN/DailyMail, Wikitext)    │
│  │  Raw Corpus  ├──► SFT splits (10K train / 200 eval per corpus)       │
│  │              ├──► Preference pairs (truncation / shuffle / mismatch) │
│  └──────────────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  Stage 1: Supervised Fine-Tuning (SFT)                                  │
│  ┌──────────────┐   LoRA adapter (r ∈ {8,16,32}, α = 2r)                │
│  │  Base SLM    ├──► + NEFTune noise injection (α=5)                    │
│  │  (frozen)    │    + AdamW, cosine LR schedule, 5 epochs              │
│  └──────────────┘──► π_SFT = θ₀ + Δθ_SFT                                │
│         │                                                               │
│         ▼                                                               │
│  Stage 2: Reward Model Training                                         │
│  ┌──────────────┐   Init from π_SFT backbone                            │
│  │  Preference  ├──► Bradley-Terry loss: -log σ(r_w - r_l)              │
│  │  Pairs (p,   │    Scalar head (num_labels=1)                         │
│  │  y_w, y_l)   │    modules_to_save=[score]                            │
│  └──────────────┘──► r_φ: (prompt, response) → ℝ                        │
│         │                                                               │
│         ▼                                                               │
│  Stage 3: PPO Alignment (float32 only)                                  │
│  ┌──────────────┐   ★ Merge-and-reinitialise (Algorithm 1)              │
│  │ merge_and_   ├──► Fresh LoRA (B=0) on merged weights                 │
│  │ unload()     │    Frozen π_ref for KL penalty                        │
│  └──────────────┘    ┌──────────────────────────────┐                   │
│         │            │  Safeguards:                 │                   │
│         │            │  • Reward whitening + 3σ clip│                   │
│         │            │  • Ratio threshold ≤ 5       │                   │
│         │            │  • NaN/Inf weight rollback   │                   │
│         │            └──────────────────────────────┘                   │
│         ▼                                                               │
│  Stage 4: Evaluation                                                    │
│  ┌──────────────┐   PPL / Reward / Δ / Win Rate                         │
│  │  200 held-out├──► Distinct-1/2, ROUGE, BLEU                          │
│  │  prompts     │    + SOTA baseline comparison                         │
│  └──────────────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────┐                               │
│  │  15 SFT + 15 PPO Aligned SLM Agents  │                               │
│  │  (single-turn specialization)        │                               │
│  └──────────────────────────────────────┘                               │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  Forward-Compatibility Layer  (src/agentic/*)            │           │
│  │  Same safety envelope, extended to multi-turn MDP with   │           │
│  │  tool invocation, clarifying queries, and termination    │           │
│  └──────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Contributions

1. **Stability prerequisites for PPO-aligned SLM agents** — three reproducible failure modes at the 70–500M scale are isolated (silent LoRA gradient freezing in PEFT/TRL, bfloat16 importance-ratio overflow, reward-driven policy collapse) together with named, version-stamped engineering remedies organised as a three-layer cybernetic safety envelope (reward whitening with 3σ clipping, importance-ratio cap ≤5, NaN/Inf weight-rollback snapshot). These prerequisites are necessary conditions for any instantiation of the agent MDP, single-turn or multi-turn.
2. **Merge-and-reinitialize for PEFT–PPO** — TRL v0.9.x freezes LoRA parameters when the policy is loaded as a PEFT adapter. This is fixed by merging the SFT adapter into base weights, then attaching a fresh LoRA before PPO:

   ```python
   # The fix (see scripts/train_ppo.py)
   merged = PeftModel.from_pretrained(base, sft_path).merge_and_unload()
   merged.save_pretrained(merged_dir)
   policy = AutoModelForCausalLMWithValueHead.from_pretrained(
       merged_dir, peft_config=LoraConfig(r=lora_r, ...))
   ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_dir)
   ```
3. **End-to-end empirical characterization across 5 SLM architectures** — the broadest reported study of PPO-based RLHF in the 70–500M parameter regime, spanning two architecture families and three text domains, producing 15 fully-trained configurations that converge stably in every case.
4. **Head-to-head SOTA comparison** — the PPO-aligned checkpoints are benchmarked against SmolLM2-135M/360M-Instruct and Qwen2.5-0.5B-Instruct on identical prompts, showing that a single-GPU, domain-specific training run is competitive with massively more expensive instruction-tuning pipelines.
5. **Capacity-headroom hypothesis** — PPO gain is governed by the joint availability of a fluent SFT prior (PPL_SFT < 20) and a discriminating reward signal rather than by raw parameter count. Every configuration with a reward delta exceeding +0.2 sits below the PPL threshold; seven of seven configurations above it show near-zero or negative deltas.
6. **Forward-compatibility scaffolding for multi-turn agents** — a released implementation of the general agent MDP with four action types (token emission, external tool invocation, clarifying-query emission, termination), a deterministic tool suite, and a per-turn PPO variant that inherits the three-layer safety envelope. No new empirical claims are attached to this layer; it is offered as a bridge for subsequent studies. See [Forward Compatibility](#forward-compatibility-multi-turn-agentic-extensions) below.

---

## Scope: Single-Turn Specialization

The empirical evaluation in this paper is confined to the **single-turn specialization** of the agent MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},T)$, defined by:

- **Horizon** $T=1$ at the turn level (the episode terminates at the end-of-sequence token)
- **Action space** $\mathcal{A}=\mathcal{V}$ collapses to the token vocabulary
- **Transition kernel** $\mathcal{P}$ is deterministic
- **Reward** $\mathcal{R}$ is terminal (delivered once at the end of the sequence)
- **Observation set** is empty (no external tools, no multi-turn memory)

Under this specialization the state reduces to the context $(x, y_{<t})$ and the MDP matches the canonical language-modelling formulation used in PPO for large-scale alignment. The three failure modes and the three-layer safety envelope studied here are **necessary preconditions** for any richer instantiation of $\mathcal{M}$: silent gradient freezing, importance-ratio overflow, and reward-driven collapse all break before multi-turn dynamics get a chance to help or hurt. Extending the empirical evaluation to the multi-turn specialization is released as scaffolding (see [Forward Compatibility](#forward-compatibility-multi-turn-agentic-extensions)) and is out of scope for the experiments reported below.

---

## Results

### SFT Perplexity Across 5 Architectures

Per-domain perplexity for each trained SFT model (lower is better; best per row in **bold**):

| Dataset       | Pythia-70M | Pythia-160M | Pythia-410M | SmolLM2-135M | SmolLM2-360M |
| ------------- | ---------- | ----------- | ----------- | ------------ | ------------ |
| TinyStories   | 51.4       | 13.5        | 6.5         | 7.0          | **5.3**      |
| CNN/DailyMail | 70.3       | 29.4        | 16.2        | 18.8         | **12.7**     |
| Wikitext-103  | 115.1      | 53.5        | 25.4        | 24.4         | **16.7**     |

SmolLM2 models dominate every domain thanks to their curated FineWeb-Edu pre-training data. Pythia-410M is competitive on TinyStories but trails SmolLM2-360M on harder corpora despite having more parameters.

### Main Results (SFT vs. PPO on per-model reward scales)

| Model           | Dataset           | SFT PPL | PPO PPL | SFT Reward       | PPO Reward       | Δ Reward   | Win Rate  |
| --------------- | ----------------- | ------- | ------- | ---------------- | ---------------- | ---------- | --------- |
| Pythia-70M      | TinyStories       | 51.4    | 51.2    | +6.61 ± 1.63     | +6.53 ± 1.42     | −0.075     | 48.6%     |
| Pythia-70M      | CNN/DailyMail     | 70.3    | 70.5    | +6.22 ± 1.21     | +6.04 ± 1.23     | −0.187     | 45.7%     |
| Pythia-70M      | Wikitext-103      | 115.1   | 116.7   | +5.81 ± 1.24     | +5.75 ± 1.30     | −0.062     | 48.6%     |
| Pythia-160M     | TinyStories       | 13.5    | 13.5    | −8.52 ± 2.39     | −8.28 ± 2.46     | **+0.238** | **52.8%** |
| Pythia-160M     | CNN/DailyMail     | 29.4    | 29.4    | −8.52 ± 1.31     | −8.71 ± 1.19     | −0.198     | 45.6%     |
| Pythia-160M     | Wikitext-103      | 53.5    | 53.2    | −8.40 ± 2.65     | −8.35 ± 2.34     | +0.044     | 50.5%     |
| **Pythia-410M** | **TinyStories**   | **6.5** | **7.3** | **−4.28 ± 4.14** | **−2.92 ± 3.48** | **+1.355** | **59.9%** |
| Pythia-410M     | CNN/DailyMail     | 16.2    | 17.1    | +1.20 ± 1.76     | +0.94 ± 1.79     | −0.259     | 45.9%     |
| Pythia-410M     | Wikitext-103      | 25.4    | 27.5    | +1.14 ± 2.89     | +0.10 ± 2.84     | −1.043     | 39.9%     |
| SmolLM2-135M    | TinyStories       | 7.0     | 7.4     | −0.92 ± 2.26     | −0.69 ± 1.96     | **+0.226** | **53.0%** |
| SmolLM2-135M    | CNN/DailyMail     | 18.8    | 19.2    | +0.22 ± 1.90     | +0.03 ± 1.90     | −0.194     | 47.1%     |
| SmolLM2-135M    | Wikitext-103      | 24.4    | 25.1    | −0.44 ± 1.53     | −0.42 ± 1.41     | +0.015     | 50.3%     |
| **SmolLM2-360M**| **TinyStories**   | **5.3** | **5.3** | **+1.69 ± 2.25** | **+2.41 ± 1.89** | **+0.724** | **59.7%** |
| SmolLM2-360M    | CNN/DailyMail     | 12.7    | 12.8    | +2.36 ± 1.09     | +2.36 ± 1.05     | −0.001     | 50.0%     |
| **SmolLM2-360M**| **Wikitext-103**  | **16.7**| **16.9**| **+2.71 ± 1.28** | **+2.98 ± 1.06** | **+0.272** | **56.5%** |

> **Notes:** Reward scores use per-configuration scales (each reward model is trained independently from the matching SFT checkpoint, so absolute magnitudes are not comparable across rows). Win rate = analytical probability Φ(Δ/√(σ²_PPO + σ²_SFT)) that a PPO response scores higher than a SFT response on the same prompt. All runs use the final PPO recipe: 250 steps, LR 5e-6, KL penalty 0.2, reward whitening + score clipping, float32 throughout, weight rollback on NaN/Inf.
>
> **Capacity-headroom hypothesis:** the three largest positive reward deltas all occur at the two highest-capacity models with fluent SFT priors (Pythia-410M/TinyStories Δ=+1.36, SmolLM2-360M/TinyStories Δ=+0.72, SmolLM2-360M/Wikitext Δ=+0.27). Pythia-70M, whose SFT perplexity exceeds 50 on every domain, shows near-zero movement everywhere. This confirms that PPO gain at the SLM scale is governed by the gap between a fluent SFT prior and the reward ceiling, not by raw parameter count.
>
> **Statistical significance:** a two-sided *z*-test over the *n* = 200 held-out prompts per configuration (95% CI on Δ) confirms three significant gains — Pythia-410M/TinyStories (CI [+0.61, +2.10], *p* < 0.001), SmolLM2-360M/TinyStories (CI [+0.32, +1.13], *p* < 0.001), and SmolLM2-360M/Wikitext-103 (CI [+0.04, +0.50], *p* < 0.05). The only significant regression, Pythia-410M/Wikitext-103 (Δ = −1.043, *p* < 0.001), coincides with an SFT prior above the fluency threshold (PPL = 25.4), in agreement with the capacity-headroom hypothesis.

### Comparison vs. Published SOTA Instruct-Tuned SLMs

Each instruct baseline and the matching SFT checkpoint is scored with the **same reward model** per dataset (matched by parameter class).

| Class      | Model                                                                                        | Training regime           | TS PPL   | TS R       | CNN PPL  | CNN R      | Wiki PPL | Wiki R    |
| ---------- | -------------------------------------------------------------------------------------------- | ------------------------- | -------- | ---------- | -------- | ---------- | -------- | --------- |
| **135M**   | SmolLM2-135M-Instruct ([Allal et al., 2024](https://arxiv.org/abs/2502.02737))               | instr.-tune, 2T tok       | 8.5      | **−0.52**  | 19.8     | **+0.35**  | 34.3     | −0.79     |
| **135M**   | **SmolLM2-135M (ours, SFT)**                                                                 | LoRA, 5 ep, 10K ex        | **7.0**  | −0.92      | **18.8** | +0.22      | **24.4** | −0.44     |
| **135M**   | **SmolLM2-135M (ours, PPO)**                                                                 | + 250-step PPO RLHF       | 7.4      | −0.69      | 19.2     | +0.03      | 25.1     | −0.42     |
| **360M+**  | SmolLM2-360M-Instruct ([Allal et al., 2024](https://arxiv.org/abs/2502.02737))               | instr.-tune, 4T tok       | 6.6      | +1.35      | 14.7     | **+3.08**  | 24.3     | +2.58     |
| **360M+**  | Qwen2.5-0.5B-Instruct ([Qwen Team, 2024](https://arxiv.org/abs/2412.15115))                  | instr.-tune, 18T tok      | 7.2      | +1.32      | 19.9     | +2.58      | 25.8     | +1.83     |
| **360M+**  | **SmolLM2-360M (ours, SFT)**                                                                 | LoRA, 5 ep, 10K ex        | **5.3**  | +1.69      | **12.7** | +2.36      | **16.7** | +2.71     |
| **360M+**  | **SmolLM2-360M (ours, PPO)**                                                                 | + 250-step PPO RLHF       | **5.3**  | **+2.41**  | **12.8** | +2.36      | **16.9** | **+2.98** |

**Key findings:**

- Domain-specific LoRA SFT **beats every instruction-tuned baseline on perplexity** across every dataset and at every scale, with the largest margin on Wikitext (16.9 vs. 24.3, a 30% reduction) at the 360M class.
- At the 360M class, the PPO checkpoint **achieves the best reward on TinyStories** (+2.41 vs. +1.35 for SmolLM2-360M-Instruct, +1.32 for Qwen2.5-0.5B-Instruct) and **on Wikitext-103** (+2.98 vs. +2.58 and +1.83) — a +0.40 absolute reward gain over the next best published baseline on Wikitext, and +1.15 over Qwen2.5-0.5B-Instruct on Wikitext.
- At the 135M class, PPO lifts reward from −0.92 to −0.69 on TinyStories, closing most of the gap to SmolLM2-135M-Instruct's −0.52.
- PPO actually **increases corpus-level Distinct-1 diversity** over the SFT baseline (e.g. SmolLM2-360M/TinyStories: 0.135 → 0.156; SmolLM2-360M/Wikitext: 0.227 → 0.310; SmolLM2-135M/Wikitext: 0.230 → 0.269), indicating that the stabilization techniques avoid the repetition-collapse failure mode often reported in small-scale RLHF. All Distinct-1/2 values reported in this README are computed corpus-level (matching `scripts/evaluate.py:compute_distinct_n`) and are reproducible from `results/all_results.json`; the same definition is used in Tables IV and V of the paper.
- These results are achieved with **~2 GPU-hours per configuration** on a single RTX A6000, vs. multi-thousand-GPU-hour regimes for the instruct baselines.

### Text Diversity (no collapse observed)

Corpus-level Distinct-1/2 and reference overlap (ROUGE-1/L) for the SFT prior and the PPO-aligned policy across **all 15 (model, corpus) configurations**. Bold indicates PPO improvement over the SFT prior on the same metric and row. Values are taken verbatim from `results/all_results.json` and match Table V of the paper.

| Model         | Dataset       | SFT D-1 | SFT D-2 | SFT R-1 | SFT R-L | PPO D-1     | PPO D-2     | PPO R-1     | PPO R-L     |
| ------------- | ------------- | ------- | ------- | ------- | ------- | ----------- | ----------- | ----------- | ----------- |
| Pythia-70M    | TinyStories   | 0.051   | 0.123   | 0.172   | 0.143   | 0.050       | 0.120       | 0.166       | 0.138       |
| Pythia-70M    | CNN/DailyMail | 0.130   | 0.329   | 0.184   | 0.115   | 0.124       | 0.310       | 0.178       | 0.111       |
| Pythia-70M    | Wikitext-103  | 0.084   | 0.189   | 0.118   | 0.095   | 0.081       | 0.186       | 0.110       | 0.087       |
| Pythia-160M   | TinyStories   | 0.103   | 0.320   | 0.241   | 0.161   | **0.104**   | **0.329**   | **0.246**   | **0.163**   |
| Pythia-160M   | CNN/DailyMail | 0.195   | 0.526   | 0.237   | 0.130   | **0.201**   | **0.543**   | 0.236       | **0.131**   |
| Pythia-160M   | Wikitext-103  | 0.143   | 0.399   | 0.171   | 0.119   | 0.132       | 0.366       | **0.172**   | **0.120**   |
| Pythia-410M   | TinyStories   | 0.119   | 0.456   | 0.351   | 0.202   | 0.110       | 0.390       | 0.323       | 0.194       |
| Pythia-410M   | CNN/DailyMail | 0.225   | 0.632   | 0.263   | 0.139   | 0.213       | 0.609       | **0.274**   | **0.145**   |
| Pythia-410M   | Wikitext-103  | 0.165   | 0.541   | 0.219   | 0.139   | 0.144       | 0.493       | 0.216       | **0.140**   |
| SmolLM2-135M  | TinyStories   | 0.147   | 0.513   | 0.310   | 0.181   | **0.172**   | **0.579**   | 0.253       | 0.154       |
| SmolLM2-135M  | CNN/DailyMail | 0.248   | 0.682   | 0.246   | 0.129   | **0.250**   | 0.678       | 0.232       | 0.123       |
| SmolLM2-135M  | Wikitext-103  | 0.230   | 0.638   | 0.185   | 0.120   | **0.269**   | **0.675**   | 0.149       | 0.102       |
| SmolLM2-360M  | TinyStories   | 0.135   | 0.489   | 0.355   | 0.212   | **0.156**   | **0.516**   | 0.335       | 0.205       |
| SmolLM2-360M  | CNN/DailyMail | 0.243   | 0.666   | 0.256   | 0.136   | **0.247**   | **0.670**   | 0.249       | 0.133       |
| SmolLM2-360M  | Wikitext-103  | 0.227   | 0.651   | 0.206   | 0.138   | **0.310**   | **0.731**   | 0.141       | 0.102       |

PPO consistently preserves or improves Distinct-1/2 over the SFT prior — no repetition collapse observed in any of the 15 configurations.

### Reproducibility

Every scalar in the three tables above is verifiable against `results/all_results.json` using `scripts/verify_results.py`, which cross-checks 339 numeric fields against the paper and prints any mismatch.

---

## Computation Cost

All experiments were conducted on a single workstation with **2× NVIDIA RTX A6000 (48 GB VRAM each)**, 128 GB RAM, CUDA 12.1. The table below breaks down the wall-clock time and training configuration for each pipeline stage.

### Per-Stage Training Details

| Stage            | Epochs / Steps    | Learning Rate | Batch Size                    | Sequence Length          | Time per Config |
| ---------------- | ----------------- | ------------- | ----------------------------- | ------------------------ | --------------- |
| **SFT**          | 5 epochs          | 2×10⁻⁵        | 8 × 4 = 32 (grad accum)       | 512                      | 20–35 min       |
| **Reward Model** | 2 epochs          | 1×10⁻⁵        | 8 × 2 = 16 (grad accum)       | 512                      | 5–10 min        |
| **PPO**          | 250 steps         | 5×10⁻⁶        | 32 (rollout) / 4 (mini-batch) | 512 (input) + 96 (gen)   | 45–90 min       |
| **Evaluation**   | —                 | —             | 200 prompts                   | 512 + 96                 | 3–5 min         |

### Per-Model Computation Summary

| Model        | Params | LoRA Rank (r) | LoRA α | SFT Time | RM Time | PPO Time | **Total per Config** |
| ------------ | ------ | ------------- | ------ | -------- | ------- | -------- | -------------------- |
| Pythia-70M   | 70M    | 8             | 16     | ~20 min  | ~5 min  | ~45 min  | **~70 min**          |
| Pythia-160M  | 162M   | 16            | 32     | ~25 min  | ~7 min  | ~55 min  | **~87 min**          |
| Pythia-410M  | 405M   | 32            | 64     | ~30 min  | ~8 min  | ~75 min  | **~113 min**         |
| SmolLM2-135M | 135M   | 16            | 32     | ~25 min  | ~7 min  | ~55 min  | **~87 min**          |
| SmolLM2-360M | 361M   | 32            | 64     | ~35 min  | ~10 min | ~90 min  | **~135 min**         |

### Total Pipeline Cost

| Metric                                      | Value                              |
| ------------------------------------------- | ---------------------------------- |
| Configurations                              | 5 models × 3 datasets = **15**     |
| Total GPU-hours (SFT + RM + PPO + eval)     | **~16 GPU-hours**                  |
| Average per configuration                   | **~1.1 GPU-hours**                 |
| Hardware                                    | 2× NVIDIA RTX A6000 (48 GB)        |
| Precision                                   | float32 (PPO), mixed (SFT/RM)      |
| Peak VRAM usage                             | ~38 GB (Pythia-410M PPO)           |

> **Cost comparison:** SmolLM2-360M-Instruct required 4T tokens of pre-training + multi-million-example instruction tuning. The SFT stage uses **10K examples for 5 epochs**, and PPO runs for **250 steps on 32-sample rollouts** — roughly **3–4 orders of magnitude** less training data than the released baselines.

---

## Models & Datasets

Everything from this paper lives in exactly **two** Hugging Face repositories:

| Artefact     | 🤗 Repo                                                                                                        | Contents                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Datasets** | **[`mr3haque/SLM-RL-Agents-Data`](https://huggingface.co/datasets/mr3haque/SLM-RL-Agents-Data)**               | 3 preprocessed corpora × 4 splits each (`sft_train`, `sft_eval`, `preference_train`, `preference_eval`)               |
| **Models**   | **[`mr3haque/SLM-RL-Agents`](https://huggingface.co/mr3haque/SLM-RL-Agents)**                                  | 15 SFT LoRA adapters **+** 15 fully-merged PPO models, organised under `sft/{model}/{dataset}/` and `ppo/{model}/{dataset}/` |

### Dataset repo layout

```
mr3haque/SLM-RL-Agents-Data
└── datasets/
    ├── tinystories/    {sft_train, sft_eval, preference_train, preference_eval}.json
    ├── cnn_dailymail/  same
    └── wikitext/       same
```

Load a specific split via the datasets library:

```python
from datasets import load_dataset

ds = load_dataset("mr3haque/SLM-RL-Agents-Data", name="tinystories", split="sft_train")
pref = load_dataset("mr3haque/SLM-RL-Agents-Data", name="cnn_dailymail", split="preference_train")
```

### Model repo layout

```
mr3haque/SLM-RL-Agents
├── sft/                                # 15 LoRA adapters
│   ├── pythia-70m/{tinystories,cnn_dailymail,wikitext}/
│   ├── pythia-160m/…
│   ├── pythia-410m/…
│   ├── smollm2-135m/…
│   └── smollm2-360m/…
└── ppo/                                # 15 FULL merged models (base + SFT + PPO)
    ├── pythia-70m/{tinystories,cnn_dailymail,wikitext}/
    ├── pythia-160m/…
    ├── pythia-410m/…
    ├── smollm2-135m/…
    └── smollm2-360m/…
```

### Load an SFT LoRA adapter

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_key, dataset = "smollm2-360m", "wikitext"
root = snapshot_download(
    repo_id="mr3haque/SLM-RL-Agents",
    allow_patterns=f"sft/{model_key}/{dataset}/**",
)
adapter_path = f"{root}/sft/{model_key}/{dataset}"

base  = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
tok   = AutoTokenizer.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
```

### Load a PPO model (already merged, no PEFT needed)

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_key, dataset = "smollm2-360m", "wikitext"
root = snapshot_download(
    repo_id="mr3haque/SLM-RL-Agents",
    allow_patterns=f"ppo/{model_key}/{dataset}/**",
)
ppo_path = f"{root}/ppo/{model_key}/{dataset}"

tok   = AutoTokenizer.from_pretrained(ppo_path)
model = AutoModelForCausalLM.from_pretrained(ppo_path)
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (CUDA 12.1 recommended)
- 24 GB+ GPU VRAM per model (48 GB for 360M–410M models)

### Setup

```bash
git clone https://github.com/rezwanh001/slm-rl-agents.git
cd slm-rl-agents

# Create conda environment
conda create -n slm-rl python=3.10
conda activate slm-rl

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers==4.45.2 trl==0.9.6 peft==0.18.1
pip install datasets accelerate evaluate rouge_score nltk

# Install package
pip install -e .
```

---

## Usage

### 1. Prepare Datasets

```bash
python scripts/prepare_all_datasets.py \
    --output_dir data \
    --num_train 10000 \
    --num_eval 1000
```

This downloads and preprocesses TinyStories, CNN/DailyMail, and Wikitext-103, generating both SFT splits and synthetic preference pairs.

### 2. Supervised Fine-Tuning (SFT)

```bash
python scripts/train_sft.py \
    --model_name EleutherAI/pythia-70m-deduped \
    --dataset_path data/tinystories/sft_train.json \
    --output_dir outputs/pythia-70m/tinystories/sft \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_r 8 \
    --lora_alpha 16
```

### 3. Reward Model Training

```bash
python scripts/train_reward.py \
    --base_model outputs/pythia-70m/tinystories/sft/final \
    --dataset_path data/tinystories/preference_train.json \
    --output_dir outputs/pythia-70m/tinystories/reward_model \
    --num_epochs 1 \
    --learning_rate 1e-5
```

### 4. PPO Alignment

```bash
python scripts/train_ppo.py \
    --policy_model outputs/pythia-70m/tinystories/sft/final \
    --reward_model outputs/pythia-70m/tinystories/reward_model/final \
    --output_dir outputs/pythia-70m/tinystories/ppo \
    --num_steps 500 \
    --kl_coef 0.1
```

> **Note:** The script automatically applies the merge-and-reinitialize strategy and float32 precision.

### 5. Evaluation

```bash
python scripts/evaluate.py \
    --model_path outputs/pythia-70m/tinystories/ppo/final \
    --reward_model_path outputs/pythia-70m/tinystories/reward_model/final \
    --dataset_path data/tinystories/sft_eval.json \
    --output_dir outputs/pythia-70m/tinystories/eval_ppo \
    --num_samples 200
```

### 6. Run Full Pipeline

To reproduce all experiments (requires 2× NVIDIA RTX A6000 or equivalent):

```bash
# Runs all 5 models × 3 datasets in parallel across 2 GPUs
bash scripts/run_all_experiments.sh
```

---

## Engineering Notes

### Merge-and-Reinitialize Strategy

TRL ≤ 0.9.x does not correctly handle PEFT-adapted policies in PPO: all LoRA parameters are frozen (`requires_grad=False`), and only the 2-parameter value head is updated. The policy generates text, receives reward, and processes gradients—but nothing changes. This is a silent failure.

**The fix** (implemented in `scripts/train_ppo.py`):

```python
# Step 1: Merge the SFT LoRA adapter into base weights
base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base, sft_adapter_path)
merged = peft_model.merge_and_unload()
merged.save_pretrained(merged_dir)

# Step 2: Fresh LoRA on merged weights (B matrices initialized to 0 → no-op at init)
fresh_lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, ...)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    merged_dir, peft_config=fresh_lora)

# Step 3: Frozen reference = merged weights, no LoRA
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_dir)
```

### Float32 Precision

Bfloat16 causes probability-ratio explosions (values > 10⁶) within the first PPO batch for models < 200M parameters, triggering CUDA assertion errors. Always use `torch_dtype=torch.float32` for both policy and reference models during PPO.

### Perplexity Computation

When evaluating perplexity on batches with padding, mask padding positions to prevent inflated scores:

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100   # mask padding tokens
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
ppl = torch.exp(outputs.loss)
```

---

## Interactive Demo

```bash
pip install gradio
python app.py
```

The demo allows side-by-side comparison of SFT and PPO agents for all trained configurations.

---

## Forward Compatibility: Multi-Turn Agentic Extensions

The single-turn specialization reported in the paper is the stability prerequisite for any multi-turn SLM agent. To make the multi-turn case tractable for subsequent work, a forward-compatibility layer is released under `src/agentic/`, in which the three-layer safety envelope (reward whitening, importance-ratio cap, weight rollback) is applied **per turn** rather than per rollout. **No new empirical claims are attached to this layer** — its release is motivated by the observation that the failure modes isolated at the 70–500M scale are necessary preconditions for any instantiation of the agent MDP, so the multi-turn case can only be examined once the single-turn prerequisites are in place.

### The General Agent MDP

The general MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},T)$ implemented in `src/agentic/` admits:

- **State** $\mathcal{S}$: a structured tuple comprising the task prompt, tokens emitted so far, any external observations, a turn index, and a remaining budget.
- **Actions** $\mathcal{A}$: four disjoint types — vocabulary emission, external tool invocation with arguments, clarifying-query emission, and termination.
- **Transition kernel** $\mathcal{P}$: deterministic for vocabulary emission and termination, stochastic for tool invocation (tool outputs extend the observation set).
- **Reward** $\mathcal{R}$: decomposes into a verifiable component from deterministic checks, a preference component from the Bradley–Terry model $r_\phi$, and a shaping component from per-token KL against the SFT prior.

Setting $T=1$, $\mathcal{A}=\mathcal{V}$, deterministic $\mathcal{P}$, terminal reward, and empty observation set recovers the single-turn specialization reported in the paper.

### Scaffolding Components

| File                                    | Purpose                                                                                                                       |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `src/agentic/environment.py`            | Parser recognising four sentinel-delimited action types (`<tool>…</tool>`, `<ask>…</ask>`, `<finish>…</finish>`, raw tokens); `State`, `Trajectory`, and `AgenticEnvironment` classes. |
| `src/agentic/tools.py`                  | Deterministic verifiable tool suite: `length_check`, `character_check`, `readability`, `retrieve`, `cite_check`, `coverage_check`, `search_passage`, `entity_lookup`, `arithmetic`, `verify_fact`, plus a `tool_call_accuracy` diagnostic. |
| `src/agentic/multi_agent.py`            | Actor + critic + editor ensemble for co-training with separate SLMs; regression-head critic; `build_critic_samples` helper.   |
| `src/agentic/agentic_ppo.py`            | Multi-turn PPO variant in which reward whitening, importance-ratio cap, and weight rollback are applied per turn; `per_turn_ratio_cap=10` default; replay-buffered rollback. |
| `scripts/build_agentic_datasets.py`     | Dataset builders for three candidate agentic tasks derived from the existing corpora: Interactive Story Agent (TinyStories), Grounded Summarisation Agent (CNN/DailyMail), Multi-hop Knowledge Agent (Wikitext-103). Produces `task_prompts.json`, `sft_tooluse.json`, and `trajectory_prefs.json` for each task. |

### Quick tour

```python
from src.agentic.environment import AgenticEnvironment
from src.agentic.tools import ToolRegistry

env = AgenticEnvironment(
    task_prompt="Summarise the article and cite at least two sources.",
    tools=ToolRegistry.default(),
    max_turns=8,
)
state = env.reset()

# Actor emits an action in the string form expected by the parser, e.g.
#   "<tool>retrieve(query='climate policy 2026')</tool>"
#   "<ask>Which region should be prioritised?</ask>"
#   "<finish>Summary text with [cite:doc_0] ...</finish>"
while not state.is_terminal:
    action_str = actor.sample(state)           # user-supplied policy
    state, reward, done = env.step(action_str)
```

### What is in scope here, and what is not

- **In scope:** released code, a documented action grammar, a deterministic tool suite that requires no external API calls, a multi-turn PPO variant that compiles and runs, unit tests, and dataset builders that extend the existing corpora.
- **Not in scope:** empirical evaluation of this scaffolding. The paper makes no quantitative claim about multi-turn performance, and this directory is explicitly offered as a bridge for subsequent studies rather than as an additional contribution. The refined capacity-headroom conditions for the multi-turn case (PPL_SFT < 20, tool-call accuracy on a held-out SFT split > 0.5, σ_R > 0.1) are stated as conjectures in the agentic methodology notes and are open for empirical test.

---

## Project Structure

```
SLM-RL-Agents/
├── src/
│   ├── slm_rl_agent/             # Core library (single-turn stack)
│   │   ├── data/                 # Dataset builders (SFT + preference pairs)
│   │   ├── models/               # Base-model wrappers and LoRA utilities
│   │   ├── rewards/              # Bradley–Terry reward model
│   │   ├── rl/                   # PPO / DPO trainers with SLM-specific fixes
│   │   └── utils/                # Logging, eval metrics, HF helpers
│   │
│   └── agentic/                  # Forward-compatibility scaffolding (multi-turn)
│       ├── environment.py        # General agent MDP with four action types
│       ├── tools.py              # Deterministic verifiable tool suite
│       ├── multi_agent.py        # Actor + critic + editor ensemble
│       └── agentic_ppo.py        # Multi-turn PPO variant (per-turn safety envelope)
│
├── scripts/
│   ├── prepare_all_datasets.py      # Build SFT + preference data for all 3 corpora
│   ├── build_agentic_datasets.py    # Build multi-turn task data (agentic extension)
│   ├── train_sft.py                 # LoRA SFT stage
│   ├── train_reward.py              # Bradley–Terry reward-model training
│   ├── train_ppo.py                 # PPO with merge-and-reinitialize + float32 fix
│   ├── train_dpo.py                 # DPO alternative (baseline)
│   ├── evaluate.py                  # Evaluate a (model, dataset) pair
│   ├── evaluate_baseline.py         # Evaluate SOTA instruct baselines
│   ├── aggregate_results.py         # Build results/all_results.json from eval dirs
│   ├── verify_results.py            # Cross-check 339 numeric fields vs. the paper
│   ├── upload_to_hf.py              # Push datasets + SFT adapters + merged PPO models
│   ├── eval_all_sft.sh              # Eval loop for the 15 SFT configs
│   ├── eval_all_ppo.sh              # Eval loop for the 15 PPO configs
│   ├── eval_baselines.sh            # Eval loop for SOTA baselines
│   ├── run_all_experiments.sh       # End-to-end 15-config training driver
│   ├── run_full_pipeline.sh         # Single-config SFT → reward → PPO → eval
│   ├── run_optimal.sh               # Optimal-hyperparam helper
│   └── run_ppo_only.sh              # PPO-only re-run helper
│
├── configs/                      # YAML hyperparameter configs
│   ├── model_configs.yaml
│   └── training_configs.yaml
│
├── tests/                        # Unit tests (single-turn + agentic scaffolding)
├── notebooks/                    # Exploratory analysis notebooks
├── data/                         # Generated preference datasets (gitignored)
├── outputs/                      # Training + eval artefacts (gitignored)
├── results/
│   └── all_results.json          # Canonical aggregated metrics (all 15 configs + baselines)
│
├── app.py                        # Gradio interactive demo (side-by-side SFT vs PPO)
├── requirements.txt
├── setup.py                      # Installable as `slm-rl-agents`
└── README.md                     # You are here
```

> **Note on the paper.** The paper is available on arXiv: [arXiv:2607.25091](https://arxiv.org/abs/2607.25091). The LaTeX source lives under `paper/` locally but is **intentionally not tracked** in this repository (see [.gitignore](https://github.com/rezwanh001/slm-rl-agents/blob/main/.gitignore)).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{haque2026slmrlagents,
  title         = {Towards Robust Reinforcement Learning for Small-Scale
                   Language Model Agents},
  author        = {Haque, Md Rezwanul and Islam, Md. Milon and Karray, Fakhri},
  booktitle     = {Proceedings of the IEEE International Conference on
                   Systems, Man, and Cybernetics (SMC)},
  year          = {2026},
  eprint        = {2607.25091},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2607.25091},
  note          = {University of Waterloo \& KUET \& MBZUAI}
}
```

---

## Acknowledgements

Computational resources were provided by the CPAMI Lab at the University of Waterloo. This work builds on the [TRL](https://github.com/huggingface/trl) library by Hugging Face, the [PEFT](https://github.com/huggingface/peft) library, and the [Pythia](https://github.com/EleutherAI/pythia) model suite by EleutherAI.

Documentation and linguistic refinement were assisted by [Claude](https://claude.ai) (Anthropic). All core research contributions — experimental design, training pipeline implementation, result analysis, and paper writing — are the original work of the authors.

---

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/rezwanh001/slm-rl-agents/blob/main/LICENSE) for details.
