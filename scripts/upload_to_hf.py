#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque

Upload SLM-RL-Agents datasets and models to HuggingFace Hub.

# ⚠️  DEPRECATED — DO NOT RUN ⚠️
# পূর্বে এই স্ক্রিপ্ট প্রতি (model, dataset) জোড়ার জন্য আলাদা HF repo বানাত
# (33টি repo: 3 dataset + 15 SFT + 15 PPO)। 2026-04-17 তারিখে সবগুলো
# per-config repo মুছে ফেলা হয়েছে এবং সব checkpoint একটিমাত্র consolidated
# repo `mr3haque/SLM-RL-Agents`-এ রাখা হয়েছে। এখন থেকে আপলোডের জন্য
# `scripts/consolidate_hf.py` ব্যবহার করতে হবে।
#
# এই ফাইলটি reference / template হিসেবে রাখা হয়েছে (model card text-গুলো
# এখনও কাজে লাগে); চালানো হলে এটি অস্তিত্বহীন repo-তে push করার চেষ্টা করবে।

Produces the following repos under the configured HF namespace:

Datasets (3):
  {NS}/slm-rl-tinystories-rlhf
  {NS}/slm-rl-cnn-dailymail-rlhf
  {NS}/slm-rl-wikitext-rlhf

Models (30):
  {NS}/slm-rl-{model_slug}-sft-{dataset}   (LoRA adapter, references public base)
  {NS}/slm-rl-{model_slug}-ppo-{dataset}   (merged full model: base + SFT LoRA + PPO LoRA)

Usage (DEPRECATED — use scripts/consolidate_hf.py instead):
  python scripts/upload_to_hf.py --namespace mr3haque               # upload everything
  python scripts/upload_to_hf.py --namespace mr3haque --only datasets
  python scripts/upload_to_hf.py --namespace mr3haque --only sft
  python scripts/upload_to_hf.py --namespace mr3haque --only ppo
  python scripts/upload_to_hf.py --namespace mr3haque --dry-run     # print plan only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "all_results.json"

MODELS: dict[str, dict[str, Any]] = {
    "pythia-70m":    {"hf_base": "EleutherAI/pythia-70m-deduped",    "params_m": 70,  "family": "Pythia"},
    "pythia-160m":   {"hf_base": "EleutherAI/pythia-160m-deduped",   "params_m": 162, "family": "Pythia"},
    "pythia-410m":   {"hf_base": "EleutherAI/pythia-410m-deduped",   "params_m": 410, "family": "Pythia"},
    "smollm2-135m":  {"hf_base": "HuggingFaceTB/SmolLM2-135M",       "params_m": 135, "family": "SmolLM2"},
    "smollm2-360m":  {"hf_base": "HuggingFaceTB/SmolLM2-360M",       "params_m": 361, "family": "SmolLM2"},
}

DATASETS = {
    "tinystories":    {"hf_slug": "slm-rl-tinystories-rlhf",     "source": "roneneldan/TinyStories",                 "domain": "short children's stories"},
    "cnn_dailymail":  {"hf_slug": "slm-rl-cnn-dailymail-rlhf",   "source": "abisee/cnn_dailymail",                   "domain": "news articles with summaries"},
    "wikitext":       {"hf_slug": "slm-rl-wikitext-rlhf",        "source": "Salesforce/wikitext (wikitext-103-raw-v1)", "domain": "encyclopedic prose"},
}


def slurp_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Dataset cards
# --------------------------------------------------------------------------

def dataset_card(dataset_key: str) -> str:
    info = DATASETS[dataset_key]
    dpath = DATA / dataset_key
    sft_train = slurp_json(dpath / "sft_train.json")
    sft_eval  = slurp_json(dpath / "sft_eval.json")
    pref_train = slurp_json(dpath / "preference_train.json")
    pref_eval  = slurp_json(dpath / "preference_eval.json")
    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
pretty_name: SLM-RL-Agents — {dataset_key.replace('_', '/').title()} RLHF split
tags:
  - rlhf
  - reward-modeling
  - sft
  - small-language-models
  - slm-rl-agents
size_categories:
  - 10K<n<100K
---

# SLM-RL-Agents: {dataset_key.replace('_', '/').title()} RLHF split

This dataset is a preprocessed slice of **{info['source']}** ({info['domain']}) prepared for the
[SLM-RL-Agents](https://github.com/rezwanh001/slm-rl-agents) framework. It is used to
(1) supervised-finetune small language models (SFT), (2) train a Bradley–Terry reward
model on preference pairs, and (3) further improve the SFT policy with PPO.

## Companion paper

> *Towards Robust Reinforcement Learning for Small-Scale Language Model Agents*
>
> Authors: Md Rezwanul Haque, Md. Milon Islam, Fakhri Karray
>
> Code: https://github.com/rezwanh001/slm-rl-agents

## Splits

| Split              | # examples |
|--------------------|------------|
| `sft_train`        | {len(sft_train):,} |
| `sft_eval`         | {len(sft_eval):,}  |
| `preference_train` | {len(pref_train):,} |
| `preference_eval`  | {len(pref_eval):,}  |

## Schema

`sft_*.json` — list of objects with keys:
- `prompt` (str)
- `response` (str)

`preference_*.json` — list of objects with keys:
- `prompt` (str)
- `chosen` (str) — higher-quality continuation
- `rejected` (str) — lower-quality continuation

Preference pairs are synthesised by ranking completions from candidate SLMs with a
length/coherence heuristic, following the SLM-RL-Agents data pipeline in
[`scripts/prepare_all_datasets.py`](https://github.com/rezwanh001/slm-rl-agents/blob/main/scripts/prepare_all_datasets.py).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{{NS}}/{info['hf_slug']}")
print(ds)
```

## Citation

```bibtex
@inproceedings{{haque2026slmrlagents,
  title         = {{Towards Robust Reinforcement Learning for Small-Scale Language Model Agents}},
  author        = {{Haque, Md Rezwanul and Islam, Md. Milon and Karray, Fakhri}},
  booktitle     = {{Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC)}},
  year          = {{2026}},
  eprint        = {{2607.25091}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.AI}},
  doi           = {{10.48550/arXiv.2607.25091}}
}}
```

## Source license

This dataset is a derivative of **{info['source']}**. Please consult the upstream
license before redistribution.
"""


# --------------------------------------------------------------------------
# Model cards
# --------------------------------------------------------------------------

def fmt_row(results: dict, key: str) -> str:
    v = results.get(key)
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def model_card_sft(model_key: str, dataset: str, results: dict, namespace: str) -> str:
    m = MODELS[model_key]
    ds_info = DATASETS[dataset]
    rd = results["our_models"][model_key]["datasets"][dataset]
    return f"""---
license: apache-2.0
base_model: {m['hf_base']}
tags:
  - rlhf
  - peft
  - lora
  - sft
  - small-language-model
  - slm-rl-agents
  - {m['family'].lower()}
language:
  - en
datasets:
  - {namespace}/{ds_info['hf_slug']}
library_name: peft
pipeline_tag: text-generation
---

# {m['family']} {m['params_m']}M — SFT on {dataset} (SLM-RL-Agents)

**Stage:** Supervised fine-tuning (SFT). LoRA adapter on top of `{m['hf_base']}`.

Part of [**SLM-RL-Agents**](https://github.com/rezwanh001/slm-rl-agents) —
a framework for efficient RLHF on Small Language Models (≤410M parameters).

## Results on {dataset} eval split (200 prompts)

| Metric           | Value |
|------------------|-------|
| Perplexity       | **{fmt_row(rd,'sft_perplexity')}** |
| Reward mean      | {fmt_row(rd,'sft_reward_mean')} ± {fmt_row(rd,'sft_reward_std')} |
| Distinct-1       | {fmt_row(rd,'sft_distinct1')} |
| Distinct-2       | {fmt_row(rd,'sft_distinct2')} |
| ROUGE-1 F1       | {fmt_row(rd,'sft_rouge1')} |
| ROUGE-2 F1       | {fmt_row(rd,'sft_rouge2')} |
| ROUGE-L F1       | {fmt_row(rd,'sft_rougeL')} |
| BLEU-4           | {fmt_row(rd,'sft_bleu4')} |

Reward mean is computed by the SLM-RL-Agents reward model trained on the same preference split.

## Training configuration

| Setting | Value |
|---|---|
| Base model | `{m['hf_base']}` |
| Trainable params | LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 3 |
| Batch size | 8 × grad accum 4 |
| Learning rate | 2e-4 (cosine) |
| Precision | bf16 |
| Hardware | 1× NVIDIA RTX A6000 (48 GB) |

## Usage

```python
# All 30 checkpoints live in the consolidated repo {namespace}/SLM-RL-Agents
# under sft/{{model}}/{{dataset}}/ and ppo/{{model}}/{{dataset}}/
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_root = snapshot_download(
    repo_id="{namespace}/SLM-RL-Agents",
    allow_patterns="sft/{model_key}/{dataset}/**",
)
adapter_path = f"{{adapter_root}}/sft/{model_key}/{dataset}"

base  = AutoModelForCausalLM.from_pretrained("{m['hf_base']}")
tok   = AutoTokenizer.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

prompt = "Once upon a time"
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=120, do_sample=True, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Citation

```bibtex
@inproceedings{{haque2026slmrlagents,
  title         = {{Towards Robust Reinforcement Learning for Small-Scale Language Model Agents}},
  author        = {{Haque, Md Rezwanul and Islam, Md. Milon and Karray, Fakhri}},
  booktitle     = {{Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC)}},
  year          = {{2026}},
  eprint        = {{2607.25091}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.AI}},
  doi           = {{10.48550/arXiv.2607.25091}}
}}
```
"""


def model_card_ppo(model_key: str, dataset: str, results: dict, namespace: str) -> str:
    m = MODELS[model_key]
    ds_info = DATASETS[dataset]
    rd = results["our_models"][model_key]["datasets"][dataset]
    delta = rd.get("reward_delta", 0.0)
    delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
    return f"""---
license: apache-2.0
base_model: {m['hf_base']}
tags:
  - rlhf
  - ppo
  - trl
  - small-language-model
  - slm-rl-agents
  - {m['family'].lower()}
language:
  - en
datasets:
  - {namespace}/{ds_info['hf_slug']}
pipeline_tag: text-generation
---

# {m['family']} {m['params_m']}M — PPO (RLHF) on {dataset} (SLM-RL-Agents)

**Stage:** PPO / RLHF. Full merged model — SFT LoRA and PPO LoRA have been merged
back into the base weights of `{m['hf_base']}`, so this repo can be loaded directly
with `AutoModelForCausalLM` (no PEFT needed).

Part of [**SLM-RL-Agents**](https://github.com/rezwanh001/slm-rl-agents) —
efficient RLHF for Small Language Models (≤410M parameters).

## Results on {dataset} eval split (200 prompts)

| Metric                          | SFT (previous stage) | **PPO (this model)** |
|---------------------------------|----------------------|----------------------|
| Perplexity (↓)                  | {fmt_row(rd,'sft_perplexity')} | **{fmt_row(rd,'ppo_perplexity')}** |
| Reward mean (↑)                 | {fmt_row(rd,'sft_reward_mean')} ± {fmt_row(rd,'sft_reward_std')} | **{fmt_row(rd,'ppo_reward_mean')} ± {fmt_row(rd,'ppo_reward_std')}** |
| **Reward Δ vs. SFT (↑)**        | —                    | **{delta_str}**     |
| Distinct-2                      | {fmt_row(rd,'sft_distinct2')} | {fmt_row(rd,'ppo_distinct2')} |
| ROUGE-L F1                      | {fmt_row(rd,'sft_rougeL')} | {fmt_row(rd,'ppo_rougeL')} |

Reward is given by the SLM-RL-Agents reward model (Bradley–Terry) trained on this dataset's
preference split.

## Training pipeline

1. **SFT** (LoRA r=16) on `{m['hf_base']}` with the `sft_train` split.
2. **Reward model** trained on `preference_train` (Bradley–Terry pairwise loss).
3. **PPO** (TRL) with the SFT model as both policy and reference; KL coefficient 0.05;
   LoRA policy head; float32 throughout; reward whitening and score clipping.
4. Final weights: base + SFT LoRA + PPO LoRA merged into a single full checkpoint.

Training uses a single NVIDIA RTX A6000 (48 GB). Per-config training budget: a few GPU hours.

## Usage

```python
# All 30 checkpoints live in the consolidated repo {namespace}/SLM-RL-Agents
# under sft/{{model}}/{{dataset}}/ and ppo/{{model}}/{{dataset}}/
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ppo_root = snapshot_download(
    repo_id="{namespace}/SLM-RL-Agents",
    allow_patterns="ppo/{model_key}/{dataset}/**",
)
ppo_path = f"{{ppo_root}}/ppo/{model_key}/{dataset}"

tok   = AutoTokenizer.from_pretrained(ppo_path)
model = AutoModelForCausalLM.from_pretrained(ppo_path)
model.eval()

prompt = "Once upon a time"
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=120, do_sample=True, temperature=0.8)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Related repositories

- Dataset: [`{namespace}/{ds_info['hf_slug']}`](https://huggingface.co/datasets/{namespace}/{ds_info['hf_slug']})
- SFT checkpoint (previous stage): [`{namespace}/SLM-RL-Agents`](https://huggingface.co/{namespace}/SLM-RL-Agents) — under `sft/{model_key}/{dataset}/`
- Code & paper: https://github.com/rezwanh001/slm-rl-agents

## Citation

```bibtex
@inproceedings{{haque2026slmrlagents,
  title         = {{Towards Robust Reinforcement Learning for Small-Scale Language Model Agents}},
  author        = {{Haque, Md Rezwanul and Islam, Md. Milon and Karray, Fakhri}},
  booktitle     = {{Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC)}},
  year          = {{2026}},
  eprint        = {{2607.25091}},
  archivePrefix = {{arXiv}},
  primaryClass  = {{cs.AI}},
  doi           = {{10.48550/arXiv.2607.25091}}
}}
```
"""


# --------------------------------------------------------------------------
# Model preparation (merge PPO stack into a standalone model)
# --------------------------------------------------------------------------

def merge_ppo_full(model_key: str, dataset: str, out_dir: Path) -> None:
    """Load base + SFT LoRA + PPO LoRA and save merged full model to out_dir."""
    m = MODELS[model_key]
    base_hf = m["hf_base"]
    sft_dir = OUTPUTS / model_key / dataset / "sft" / "final"
    ppo_dir = OUTPUTS / model_key / dataset / "ppo" / "final"

    print(f"    loading base   : {base_hf}")
    base = AutoModelForCausalLM.from_pretrained(base_hf, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(base_hf)

    print(f"    merging SFT    : {sft_dir}")
    m1 = PeftModel.from_pretrained(base, str(sft_dir))
    merged_sft = m1.merge_and_unload()

    print(f"    merging PPO    : {ppo_dir}")
    m2 = PeftModel.from_pretrained(merged_sft, str(ppo_dir))
    merged_final = m2.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_final.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"    saved merged   : {out_dir}")


# --------------------------------------------------------------------------
# Upload helpers
# --------------------------------------------------------------------------

def upload_folder_with_card(api: HfApi, repo_id: str, local_dir: Path, card: str,
                            repo_type: str = "model") -> None:
    print(f"  creating repo  : {repo_id}  ({repo_type})")
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)

    card_path = local_dir / "README.md"
    card_path.write_text(card)

    print(f"  uploading dir  : {local_dir}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        commit_message=f"Upload {repo_id}",
    )
    print(f"  done           : https://huggingface.co/{repo_id}")


def upload_dataset(api: HfApi, namespace: str, dataset: str, dry: bool) -> None:
    ds_info = DATASETS[dataset]
    repo_id = f"{namespace}/{ds_info['hf_slug']}"
    src = DATA / dataset
    print(f"\n[dataset] {repo_id}")
    if dry:
        print(f"  WOULD upload from {src}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "stage"
        staging.mkdir()
        for f in ["sft_train.json", "sft_eval.json", "preference_train.json", "preference_eval.json"]:
            shutil.copy2(src / f, staging / f)
        card = dataset_card(dataset).replace("{NS}", namespace)
        upload_folder_with_card(api, repo_id, staging, card, repo_type="dataset")


def upload_sft(api: HfApi, namespace: str, model_key: str, dataset: str, results: dict, dry: bool) -> None:
    repo_id = f"{namespace}/slm-rl-{model_key}-sft-{dataset}"
    src = OUTPUTS / model_key / dataset / "sft" / "final"
    print(f"\n[sft]     {repo_id}")
    if not src.exists():
        print(f"  SKIP — missing {src}")
        return
    if dry:
        print(f"  WOULD upload adapter from {src}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "stage"
        staging.mkdir()
        for f in src.iterdir():
            if f.name == "README.md":
                continue
            shutil.copy2(f, staging / f.name)
        card = model_card_sft(model_key, dataset, results, namespace)
        upload_folder_with_card(api, repo_id, staging, card, repo_type="model")


def upload_ppo(api: HfApi, namespace: str, model_key: str, dataset: str, results: dict, dry: bool) -> None:
    repo_id = f"{namespace}/slm-rl-{model_key}-ppo-{dataset}"
    print(f"\n[ppo]     {repo_id}")
    src_ppo = OUTPUTS / model_key / dataset / "ppo" / "final"
    if not src_ppo.exists():
        print(f"  SKIP — missing {src_ppo}")
        return
    if dry:
        print(f"  WOULD merge+upload PPO from {src_ppo}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp) / "merged"
        merge_ppo_full(model_key, dataset, merged)
        card = model_card_ppo(model_key, dataset, results, namespace)
        upload_folder_with_card(api, repo_id, merged, card, repo_type="model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True, help="HF user or org, e.g. mr3haque")
    parser.add_argument("--only", choices=["datasets", "sft", "ppo", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = parser.parse_args()

    results = slurp_json(RESULTS)
    api = HfApi()
    who = api.whoami()
    print(f"Logged in as: {who.get('name')}")
    print(f"Namespace   : {args.namespace}")
    print(f"Mode        : {args.only}  dry-run={args.dry_run}")
    print()

    if args.only in ("datasets", "all"):
        for ds in args.datasets:
            upload_dataset(api, args.namespace, ds, args.dry_run)

    if args.only in ("sft", "all"):
        for mk in args.models:
            for ds in args.datasets:
                upload_sft(api, args.namespace, mk, ds, results, args.dry_run)

    if args.only in ("ppo", "all"):
        for mk in args.models:
            for ds in args.datasets:
                upload_ppo(api, args.namespace, mk, ds, results, args.dry_run)

    print("\nDONE")


if __name__ == "__main__":
    main()
