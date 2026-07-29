#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque

Consolidate SLM-RL-Agents artefacts into exactly TWO Hugging Face repos:

    mr3haque/SLM-RL-Agents-Data   (dataset)
    mr3haque/SLM-RL-Agents        (model)

Actions (run individually with --stage):

    1. rename-data      Rename existing  mr3haque/SLM-RL-Agents (dataset)
                        to mr3haque/SLM-RL-Agents-Data.  Preserves downloads.
    2. clean-data       Wipe stale models/* from inside the data repo.
    3. upload-data      Upload 3 preprocessed datasets + polished README
                        with result tables into SLM-RL-Agents-Data.
    4. create-model     Create mr3haque/SLM-RL-Agents model repo.
    5. stage-model      Merge 15 PPO checkpoints into full models and
                        build the full staging tree on disk.
    6. upload-model     Upload staged model tree + polished README with
                        all 15 SFT/PPO result tables.
    7. cleanup-old      Delete the 33 per-config repos created earlier.
    all                 Run 1..7 in order.
"""
from __future__ import annotations

import argparse
import json
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

NAMESPACE = "mr3haque"
OLD_DATA_REPO = f"{NAMESPACE}/SLM-RL-Agents"           # current dataset repo
NEW_DATA_REPO = f"{NAMESPACE}/SLM-RL-Agents-Data"      # renamed dataset repo
MODEL_REPO    = f"{NAMESPACE}/SLM-RL-Agents"           # new model repo

STAGE_ROOT = Path("/tmp/slm-rl-agent-staging")
STAGE_DATA  = STAGE_ROOT / "data"
STAGE_MODEL = STAGE_ROOT / "model"

MODELS: dict[str, dict[str, Any]] = {
    "pythia-70m":    {"hf_base": "EleutherAI/pythia-70m-deduped",    "params_m": 70,  "family": "Pythia"},
    "pythia-160m":   {"hf_base": "EleutherAI/pythia-160m-deduped",   "params_m": 162, "family": "Pythia"},
    "pythia-410m":   {"hf_base": "EleutherAI/pythia-410m-deduped",   "params_m": 410, "family": "Pythia"},
    "smollm2-135m":  {"hf_base": "HuggingFaceTB/SmolLM2-135M",       "params_m": 135, "family": "SmolLM2"},
    "smollm2-360m":  {"hf_base": "HuggingFaceTB/SmolLM2-360M",       "params_m": 361, "family": "SmolLM2"},
}

DATASETS = {
    "tinystories":   {"source": "roneneldan/TinyStories",                      "domain": "Short children's stories (ages 3–5 vocabulary)",       "size": "15 MB"},
    "cnn_dailymail": {"source": "abisee/cnn_dailymail",                        "domain": "News articles with human-written summaries",            "size": "34 MB"},
    "wikitext":      {"source": "Salesforce/wikitext (wikitext-103-raw-v1)",   "domain": "Encyclopedic prose from good/featured Wikipedia pages", "size": "12 MB"},
}

# Per-config per-repo list built programmatically below.
PER_CONFIG_DATASET_REPOS = [
    f"{NAMESPACE}/slm-rl-tinystories-rlhf",
    f"{NAMESPACE}/slm-rl-cnn-dailymail-rlhf",
    f"{NAMESPACE}/slm-rl-wikitext-rlhf",
]
PER_CONFIG_MODEL_REPOS = [
    f"{NAMESPACE}/slm-rl-{m}-{stage}-{d}"
    for m in MODELS
    for stage in ("sft", "ppo")
    for d in DATASETS
]  # 15 * 2 = 30


def _load_results() -> dict:
    with open(RESULTS) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Stage 1: rename
# --------------------------------------------------------------------------

def stage_rename_data(api: HfApi) -> None:
    try:
        api.repo_info(NEW_DATA_REPO, repo_type="dataset")
        print(f"[rename] {NEW_DATA_REPO} already exists, skipping rename.")
        return
    except Exception:
        pass
    try:
        api.repo_info(OLD_DATA_REPO, repo_type="dataset")
    except Exception as e:
        print(f"[rename] old repo {OLD_DATA_REPO} not found: {e}")
        return
    print(f"[rename] {OLD_DATA_REPO}  ->  {NEW_DATA_REPO}  (preserves downloads)")
    api.move_repo(from_id=OLD_DATA_REPO, to_id=NEW_DATA_REPO, repo_type="dataset")
    print(f"[rename] done")


# --------------------------------------------------------------------------
# Stage 2: clean stale models/* from inside data repo
# --------------------------------------------------------------------------

def stage_clean_data(api: HfApi) -> None:
    files = api.list_repo_files(NEW_DATA_REPO, repo_type="dataset")
    stale = [f for f in files if f.startswith("models/")]
    if not stale:
        print(f"[clean-data] no stale models/* files in {NEW_DATA_REPO}")
        return
    print(f"[clean-data] deleting models/ folder ({len(stale)} stale files) from {NEW_DATA_REPO}")
    api.delete_folder(
        path_in_repo="models",
        repo_id=NEW_DATA_REPO,
        repo_type="dataset",
        commit_message="Remove stale models/ tree (models now live in mr3haque/SLM-RL-Agents)",
    )
    print("[clean-data] done")


# --------------------------------------------------------------------------
# Stage 3: build + upload polished data layout
# --------------------------------------------------------------------------

def _dataset_counts(dataset: str) -> dict[str, int]:
    dpath = DATA / dataset
    out = {}
    for f in ("sft_train", "sft_eval", "preference_train", "preference_eval"):
        with open(dpath / f"{f}.json") as fh:
            out[f] = len(json.load(fh))
    return out


def build_data_readme() -> str:
    results = _load_results()
    counts = {ds: _dataset_counts(ds) for ds in DATASETS}

    ds_table_lines = [
        "| Dataset | Source | Domain | Disk | SFT train | SFT eval | Pref. train | Pref. eval |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for ds, info in DATASETS.items():
        c = counts[ds]
        ds_table_lines.append(
            f"| **{ds}** | [`{info['source']}`](https://huggingface.co/datasets/{info['source'].split(' ')[0]}) | {info['domain']} | {info['size']} | {c['sft_train']:,} | {c['sft_eval']:,} | {c['preference_train']:,} | {c['preference_eval']:,} |"
        )
    ds_table = "\n".join(ds_table_lines)

    return f"""---
license: apache-2.0
task_categories:
  - text-generation
  - summarization
language:
  - en
pretty_name: SLM-RL-Agents Data — RLHF Splits for Small Language Models
tags:
  - rlhf
  - reward-modeling
  - preference-pairs
  - sft
  - small-language-models
  - slm-rl-agents
size_categories:
  - 10K<n<100K
configs:
  - config_name: tinystories
    data_files:
      - split: sft_train
        path: datasets/tinystories/sft_train.json
      - split: sft_eval
        path: datasets/tinystories/sft_eval.json
      - split: preference_train
        path: datasets/tinystories/preference_train.json
      - split: preference_eval
        path: datasets/tinystories/preference_eval.json
  - config_name: cnn_dailymail
    data_files:
      - split: sft_train
        path: datasets/cnn_dailymail/sft_train.json
      - split: sft_eval
        path: datasets/cnn_dailymail/sft_eval.json
      - split: preference_train
        path: datasets/cnn_dailymail/preference_train.json
      - split: preference_eval
        path: datasets/cnn_dailymail/preference_eval.json
  - config_name: wikitext
    data_files:
      - split: sft_train
        path: datasets/wikitext/sft_train.json
      - split: sft_eval
        path: datasets/wikitext/sft_eval.json
      - split: preference_train
        path: datasets/wikitext/preference_train.json
      - split: preference_eval
        path: datasets/wikitext/preference_eval.json
---

# SLM-RL-Agents-Data

**Companion datasets for the paper *Towards Robust Reinforcement Learning for Small-Scale Language Model Agents*.**

**Authors:** Md Rezwanul Haque, Md. Milon Islam, Fakhri Karray

| | |
|---|---|
| **Code**        | [github.com/rezwanh001/slm-rl-agents](https://github.com/rezwanh001/slm-rl-agents) |
| **Trained models** | [`mr3haque/SLM-RL-Agents`](https://huggingface.co/mr3haque/SLM-RL-Agents) |
| **License**     | Apache-2.0 (this processing); upstream corpora retain their own licenses |

This repository bundles the three preprocessed text corpora used to train the entire
SLM-RL-Agents framework — a complete three-stage RLHF pipeline (SFT → reward model → PPO)
applied to small language models in the **70M–410M parameter range**. Each corpus is
provided both as a **supervised fine-tuning** split (`sft_train`, `sft_eval`) and a
**preference-pair** split (`preference_train`, `preference_eval`) for Bradley–Terry
reward-model training.

---

## Dataset summary

{ds_table}

All splits have been deduplicated, prompt-normalized, and truncated to a uniform
`prompt + response ≤ 512 tokens` budget. Preference pairs are synthesised by
ranking completions from candidate SLMs with a length/coherence heuristic;
the exact pipeline is reproducible via
[`scripts/prepare_all_datasets.py`](https://github.com/rezwanh001/slm-rl-agents/blob/main/scripts/prepare_all_datasets.py).

---

## Quick start

```python
from datasets import load_dataset

# Load TinyStories SFT split
ds = load_dataset("mr3haque/SLM-RL-Agents-Data", name="tinystories", split="sft_train")
print(ds[0])

# Load CNN/DailyMail preference pairs
pref = load_dataset("mr3haque/SLM-RL-Agents-Data", name="cnn_dailymail", split="preference_train")
print(pref[0]["prompt"], "|", pref[0]["chosen"])
```

Or clone the raw JSON files directly:

```bash
huggingface-cli download mr3haque/SLM-RL-Agents-Data \\
    --repo-type dataset --local-dir ./slm-rl-data
```

---

## Schema

`sft_*` splits — list of objects:

```json
{{"prompt": "…", "response": "…"}}
```

`preference_*` splits — list of objects:

```json
{{"prompt": "…", "chosen": "…", "rejected": "…"}}
```

Higher-quality continuations are placed in `chosen`; lower-quality continuations in `rejected`.

---

## How the data is used in the paper

The three corpora are used to produce **15 fully trained RLHF configurations**
(5 SLM architectures × 3 domains). A reward model is trained on each
preference split and then used to PPO-align the corresponding SFT checkpoint.

| Stage | Input | Output |
|---|---|---|
| SFT            | `sft_train` | LoRA SFT checkpoint |
| Reward model   | `preference_train` | Bradley–Terry scalar reward model |
| PPO (RLHF)     | SFT checkpoint + reward model + `sft_eval` prompts | Aligned PPO checkpoint |
| Evaluation     | `sft_eval` | Perplexity, reward mean/std, ROUGE, BLEU, Distinct-n |

All 30 trained checkpoints (15 SFT + 15 PPO) are published in the companion repo
[`mr3haque/SLM-RL-Agents`](https://huggingface.co/mr3haque/SLM-RL-Agents).

---

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

## Licensing note

The preprocessing, preference-pair construction, and packaging are released under
Apache-2.0. The underlying text in each split is derived from an existing public
corpus and remains subject to that corpus's own license — TinyStories
(CDLA-Sharing-1.0), CNN/DailyMail (Apache-2.0), and Wikitext-103 (CC BY-SA 3.0).
Please consult those upstream licenses before redistribution.
"""


def stage_upload_data(api: HfApi) -> None:
    stage = STAGE_DATA
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # Copy the three datasets under datasets/{name}/*.json
    for ds in DATASETS:
        tgt = stage / "datasets" / ds
        tgt.mkdir(parents=True)
        for f in ("sft_train.json", "sft_eval.json", "preference_train.json", "preference_eval.json"):
            shutil.copy2(DATA / ds / f, tgt / f)

    # README
    (stage / "README.md").write_text(build_data_readme())

    print(f"[upload-data] uploading staging tree {stage} -> {NEW_DATA_REPO}")
    api.upload_folder(
        repo_id=NEW_DATA_REPO,
        repo_type="dataset",
        folder_path=str(stage),
        commit_message="Consolidate RLHF datasets + publish polished dataset card",
    )
    print("[upload-data] done")


# --------------------------------------------------------------------------
# Stage 4: create model repo
# --------------------------------------------------------------------------

def stage_create_model(api: HfApi) -> None:
    print(f"[create-model] ensuring {MODEL_REPO} exists")
    create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    print("[create-model] done")


# --------------------------------------------------------------------------
# Stage 5: merge + stage all 30 models locally
# --------------------------------------------------------------------------

def _merge_ppo(model_key: str, dataset: str, out_dir: Path) -> None:
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
    # Clean up the large in-memory state before the next iteration.
    del merged_final, m2, merged_sft, m1, base
    import gc; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def stage_stage_model() -> None:
    stage = STAGE_MODEL
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "sft").mkdir(parents=True)
    (stage / "ppo").mkdir(parents=True)

    # Copy SFT LoRA adapters
    for mk in MODELS:
        for ds in DATASETS:
            src = OUTPUTS / mk / ds / "sft" / "final"
            if not src.exists():
                print(f"[stage-model] SKIP SFT {mk}/{ds}: {src} missing")
                continue
            tgt = stage / "sft" / mk / ds
            tgt.mkdir(parents=True)
            for f in src.iterdir():
                if f.name == "README.md":
                    continue
                shutil.copy2(f, tgt / f.name)
            print(f"[stage-model] staged SFT {mk}/{ds}")

    # Merge PPO models into full checkpoints
    for mk in MODELS:
        for ds in DATASETS:
            ppo_src = OUTPUTS / mk / ds / "ppo" / "final"
            if not ppo_src.exists():
                print(f"[stage-model] SKIP PPO {mk}/{ds}: {ppo_src} missing")
                continue
            tgt = stage / "ppo" / mk / ds
            print(f"[stage-model] merging PPO {mk}/{ds}  ->  {tgt}")
            _merge_ppo(mk, ds, tgt)
    print("[stage-model] done")


# --------------------------------------------------------------------------
# Stage 6: build model README + upload
# --------------------------------------------------------------------------

def _fmt(v, prec=3, signed=False):
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return (f"{v:+.{prec}f}" if signed else f"{v:.{prec}f}" if abs(v) < 10 else f"{v:.2f}")
    return str(v)


def build_model_readme() -> str:
    r = _load_results()
    ours = r["our_models"]
    base = r["baselines"]

    # --- Main result table (15 rows) ---
    main_rows = [
        "| Model | Params | Dataset | SFT PPL ↓ | PPO PPL ↓ | SFT Reward ↑ | PPO Reward ↑ | Δ Reward | Win Rate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    import math
    def winrate(delta, s_sft, s_ppo):
        denom = math.sqrt((s_ppo or 0.0)**2 + (s_sft or 0.0)**2)
        if denom < 1e-9:
            return 50.0
        from math import erf, sqrt
        z = delta / denom
        return 50.0 * (1.0 + erf(z / sqrt(2.0)))

    for mk, mobj in ours.items():
        fam = MODELS[mk]["family"]
        pm  = mobj["params_m"]
        for ds in DATASETS:
            d = mobj["datasets"][ds]
            dlt = d.get("reward_delta", 0.0)
            wr  = winrate(dlt, d.get("sft_reward_std"), d.get("ppo_reward_std"))
            mark_l, mark_r = ("**", "**") if dlt > 0.15 else ("", "")
            main_rows.append(
                f"| {mark_l}{fam} {pm}M{mark_r} | {pm} | {ds} | "
                f"{_fmt(d['sft_perplexity'], 1)} | {_fmt(d['ppo_perplexity'], 1)} | "
                f"{_fmt(d['sft_reward_mean'], 2, signed=True)} ± {_fmt(d['sft_reward_std'], 2)} | "
                f"{_fmt(d['ppo_reward_mean'], 2, signed=True)} ± {_fmt(d['ppo_reward_std'], 2)} | "
                f"{mark_l}{_fmt(dlt, 3, signed=True)}{mark_r} | {wr:.1f}% |"
            )
    main_table = "\n".join(main_rows)

    # --- SOTA comparison ---
    def smol(key, ds, field):
        return base.get(key, {}).get(ds, {}).get(field, None)
    def ours_field(mk, ds, field):
        return ours[mk]["datasets"][ds].get(field, None)

    sota_rows = [
        "| Class | Model | Training regime | TS PPL | TS R | CNN PPL | CNN R | Wiki PPL | Wiki R |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        f"| 135M | SmolLM2-135M-Instruct | instruct-tune 2T tok | {_fmt(smol('smollm2-135m-instruct','tinystories','perplexity'),1)} | {_fmt(smol('smollm2-135m-instruct','tinystories','reward_mean'),2,signed=True)} | {_fmt(smol('smollm2-135m-instruct','cnn_dailymail','perplexity'),1)} | {_fmt(smol('smollm2-135m-instruct','cnn_dailymail','reward_mean'),2,signed=True)} | {_fmt(smol('smollm2-135m-instruct','wikitext','perplexity'),1)} | {_fmt(smol('smollm2-135m-instruct','wikitext','reward_mean'),2,signed=True)} |",
        f"| 135M | **SmolLM2-135M (ours, SFT)** | LoRA, 5 ep, 10K ex | **{_fmt(ours_field('smollm2-135m','tinystories','sft_perplexity'),1)}** | {_fmt(ours_field('smollm2-135m','tinystories','sft_reward_mean'),2,signed=True)} | **{_fmt(ours_field('smollm2-135m','cnn_dailymail','sft_perplexity'),1)}** | {_fmt(ours_field('smollm2-135m','cnn_dailymail','sft_reward_mean'),2,signed=True)} | **{_fmt(ours_field('smollm2-135m','wikitext','sft_perplexity'),1)}** | {_fmt(ours_field('smollm2-135m','wikitext','sft_reward_mean'),2,signed=True)} |",
        f"| 135M | **SmolLM2-135M (ours, PPO)** | + 250-step PPO RLHF | {_fmt(ours_field('smollm2-135m','tinystories','ppo_perplexity'),1)} | {_fmt(ours_field('smollm2-135m','tinystories','ppo_reward_mean'),2,signed=True)} | {_fmt(ours_field('smollm2-135m','cnn_dailymail','ppo_perplexity'),1)} | {_fmt(ours_field('smollm2-135m','cnn_dailymail','ppo_reward_mean'),2,signed=True)} | {_fmt(ours_field('smollm2-135m','wikitext','ppo_perplexity'),1)} | {_fmt(ours_field('smollm2-135m','wikitext','ppo_reward_mean'),2,signed=True)} |",
        f"| 360M+ | SmolLM2-360M-Instruct | instruct-tune 4T tok | {_fmt(smol('smollm2-360m-instruct','tinystories','perplexity'),1)} | {_fmt(smol('smollm2-360m-instruct','tinystories','reward_mean'),2,signed=True)} | {_fmt(smol('smollm2-360m-instruct','cnn_dailymail','perplexity'),1)} | **{_fmt(smol('smollm2-360m-instruct','cnn_dailymail','reward_mean'),2,signed=True)}** | {_fmt(smol('smollm2-360m-instruct','wikitext','perplexity'),1)} | {_fmt(smol('smollm2-360m-instruct','wikitext','reward_mean'),2,signed=True)} |",
        f"| 360M+ | Qwen2.5-0.5B-Instruct | instruct-tune 18T tok | {_fmt(smol('qwen25-05b-instruct','tinystories','perplexity'),1)} | {_fmt(smol('qwen25-05b-instruct','tinystories','reward_mean'),2,signed=True)} | {_fmt(smol('qwen25-05b-instruct','cnn_dailymail','perplexity'),1)} | {_fmt(smol('qwen25-05b-instruct','cnn_dailymail','reward_mean'),2,signed=True)} | {_fmt(smol('qwen25-05b-instruct','wikitext','perplexity'),1)} | {_fmt(smol('qwen25-05b-instruct','wikitext','reward_mean'),2,signed=True)} |",
        f"| 360M+ | **SmolLM2-360M (ours, SFT)** | LoRA, 5 ep, 10K ex | **{_fmt(ours_field('smollm2-360m','tinystories','sft_perplexity'),1)}** | {_fmt(ours_field('smollm2-360m','tinystories','sft_reward_mean'),2,signed=True)} | **{_fmt(ours_field('smollm2-360m','cnn_dailymail','sft_perplexity'),1)}** | {_fmt(ours_field('smollm2-360m','cnn_dailymail','sft_reward_mean'),2,signed=True)} | **{_fmt(ours_field('smollm2-360m','wikitext','sft_perplexity'),1)}** | {_fmt(ours_field('smollm2-360m','wikitext','sft_reward_mean'),2,signed=True)} |",
        f"| 360M+ | **SmolLM2-360M (ours, PPO)** | + 250-step PPO RLHF | **{_fmt(ours_field('smollm2-360m','tinystories','ppo_perplexity'),1)}** | **{_fmt(ours_field('smollm2-360m','tinystories','ppo_reward_mean'),2,signed=True)}** | **{_fmt(ours_field('smollm2-360m','cnn_dailymail','ppo_perplexity'),1)}** | {_fmt(ours_field('smollm2-360m','cnn_dailymail','ppo_reward_mean'),2,signed=True)} | **{_fmt(ours_field('smollm2-360m','wikitext','ppo_perplexity'),1)}** | **{_fmt(ours_field('smollm2-360m','wikitext','ppo_reward_mean'),2,signed=True)}** |",
    ]
    sota_table = "\n".join(sota_rows)

    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - rlhf
  - ppo
  - sft
  - lora
  - peft
  - trl
  - small-language-models
  - pythia
  - smollm2
  - slm-rl-agents
base_model:
  - EleutherAI/pythia-70m-deduped
  - EleutherAI/pythia-160m-deduped
  - EleutherAI/pythia-410m-deduped
  - HuggingFaceTB/SmolLM2-135M
  - HuggingFaceTB/SmolLM2-360M
datasets:
  - mr3haque/SLM-RL-Agents-Data
---

# SLM-RL-Agents — Models

**Companion model repository for the paper *Towards Robust Reinforcement Learning
for Small-Scale Language Model Agents*.**

**Authors:** Md Rezwanul Haque, Md. Milon Islam, Fakhri Karray

| | |
|---|---|
| **Code** | [github.com/rezwanh001/slm-rl-agents](https://github.com/rezwanh001/slm-rl-agents) |
| **Datasets** | [`mr3haque/SLM-RL-Agents-Data`](https://huggingface.co/datasets/mr3haque/SLM-RL-Agents-Data) |
| **License** | Apache-2.0 |
| **Hardware** | 1 × NVIDIA RTX A6000 (48 GB) |

This single repository hosts **all 30 trained checkpoints** from the SLM-RL-Agents
framework — 15 supervised-fine-tuned (SFT) small language models and 15 PPO-aligned
(RLHF) small language models — spanning **5 architectures × 3 text corpora**.

---

## Repository layout

```
SLM-RL-Agents/
├── sft/                               # 15 LoRA adapters
│   ├── pythia-70m/
│   │   ├── tinystories/               #  (adapter_model.safetensors + tokenizer)
│   │   ├── cnn_dailymail/
│   │   └── wikitext/
│   ├── pythia-160m/     ...
│   ├── pythia-410m/     ...
│   ├── smollm2-135m/    ...
│   └── smollm2-360m/    ...
│
└── ppo/                               # 15 FULL merged models (base + SFT + PPO)
    ├── pythia-70m/
    │   ├── tinystories/               #  (model.safetensors + tokenizer)
    │   ├── cnn_dailymail/
    │   └── wikitext/
    ├── ... (same structure for all 5 models)
```

Each **SFT** directory is a LoRA adapter that sits on top of the corresponding
public base model.  Each **PPO** directory is a fully merged model that already
contains the base weights + SFT LoRA + PPO LoRA collapsed into a single full
checkpoint — no PEFT installation required to load it.

---

## Main results — 15 configurations

Evaluated on the first **200 prompts** of each domain's held-out split
(`num_samples=200`, matching the raw `outputs/*/eval_*/evaluation_results.json`
files shipped with the [GitHub repo](https://github.com/rezwanh001/slm-rl-agents)).
Reward comes from the
SLM-RL-Agents Bradley–Terry reward model (per-configuration scale). Win rate is
the analytical probability that a PPO response scores higher than an SFT response
on the same prompt, Φ(Δ / √(σ²_PPO + σ²_SFT)).

{main_table}

**Key findings.**

- **Capacity-headroom hypothesis.** The three largest positive reward deltas occur
  at the two highest-capacity models: Pythia-410M / TinyStories (Δ = +1.36),
  SmolLM2-360M / TinyStories (Δ = +0.72), SmolLM2-360M / Wikitext-103 (Δ = +0.27).
  Models whose SFT baseline is already near-perfect see diminishing returns at this
  training budget — PPO gain is governed by the gap between a fluent SFT prior and
  the reward ceiling, not by raw parameter count.
- **Statistical significance.** A two-sided z-test over the n = 200 held-out prompts
  per configuration (95% CI on Δ) confirms three significant gains: Pythia-410M /
  TinyStories (CI [+0.61, +2.10], p < 0.001), SmolLM2-360M / TinyStories
  (CI [+0.32, +1.13], p < 0.001), and SmolLM2-360M / Wikitext-103
  (CI [+0.04, +0.50], p < 0.05). The only significant regression, Pythia-410M /
  Wikitext-103 (Δ = −1.043, p < 0.001), coincides with an SFT prior above the
  fluency threshold (PPL = 25.4).
- **No repetition collapse.** PPO consistently preserves or *improves* Distinct-2
  diversity over the SFT baseline — e.g. SmolLM2-360M / Wikitext goes from
  Distinct-1 = 0.23 → 0.31 and Distinct-2 = 0.65 → 0.73.
- **Efficiency.** Every configuration trains end-to-end (SFT → reward → PPO → eval)
  in a few GPU-hours on a single RTX A6000.

---

## Comparison vs. published SOTA instruct-tuned SLMs

Each instruct baseline is scored with the **same** SLM-RL-Agents reward model per
dataset. Lower perplexity = better; higher reward = better.

{sota_table}

**Highlights.**

- Our 360M-class SFT **beats every instruct baseline on perplexity** across
  every dataset — the largest margin is on Wikitext-103 (16.7 vs. 24.3, a 30 %
  reduction) at a single-GPU, domain-specific training budget.
- At the 360M class, our **PPO checkpoint is the best on TinyStories reward**
  (+2.41 vs. +1.35 for SmolLM2-360M-Instruct and +1.32 for Qwen2.5-0.5B-Instruct)
  and **best on Wikitext-103 reward** (+2.98 vs. +2.58 and +1.83).

---

## Usage

### Load an SFT LoRA adapter

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Pick one of the 15 (model, dataset) combinations
model_key, dataset = "smollm2-360m", "wikitext"
adapter_dir = snapshot_download(
    repo_id="mr3haque/SLM-RL-Agents",
    allow_patterns=f"sft/{{model_key}}/{{dataset}}/**",
)
adapter_path = f"{{adapter_dir}}/sft/{{model_key}}/{{dataset}}"

base  = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
tok   = AutoTokenizer.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
```

### Load a PPO model (already merged)

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_key, dataset = "smollm2-360m", "wikitext"
ppo_dir = snapshot_download(
    repo_id="mr3haque/SLM-RL-Agents",
    allow_patterns=f"ppo/{{model_key}}/{{dataset}}/**",
)
ppo_path = f"{{ppo_dir}}/ppo/{{model_key}}/{{dataset}}"

tok   = AutoTokenizer.from_pretrained(ppo_path)
model = AutoModelForCausalLM.from_pretrained(ppo_path)
```

---

## Training recipe (identical for all 15 configurations)

| Stage | Library | Key hyperparameters |
|---|---|---|
| **SFT**          | HuggingFace Trainer + PEFT | LoRA r=16, α=32, 3 epochs, bs 8×4, LR 2e-4, bf16 |
| **Reward model** | HuggingFace Trainer        | Bradley–Terry pairwise loss, 1 epoch, LR 1e-5 |
| **PPO**          | TRL 0.9.x                  | 250 steps, LR 5e-6, KL 0.05–0.2, score clip ±3σ, **float32**, weight rollback |

Three engineering fixes unique to the SLM regime — all implemented in
[`scripts/train_ppo.py`](https://github.com/rezwanh001/slm-rl-agents/blob/main/scripts/train_ppo.py):

1. **Merge-and-reinitialize for PEFT+PPO.** TRL ≤ 0.9.x silently freezes LoRA
   parameters when the policy is a PEFT adapter. We merge the SFT adapter into
   the base weights, then attach a fresh LoRA on top before PPO.
2. **Float32 throughout.** Bfloat16 causes probability-ratio explosions (> 10⁶)
   within the first PPO batch for models < 200M parameters. Float32 is required.
3. **Reward whitening + weight rollback.** Score-clipping at ±3σ and a
   per-step weight-rollback mechanism that reverts to the last healthy snapshot
   on NaN/Inf eliminate catastrophic collapse across all 15 runs.

---

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


def stage_upload_model(api: HfApi) -> None:
    stage = STAGE_MODEL
    if not stage.exists():
        raise RuntimeError(f"staging dir {stage} missing — run --stage stage-model first")
    (stage / "README.md").write_text(build_model_readme())
    print(f"[upload-model] uploading staging tree {stage}  ->  {MODEL_REPO}")
    api.upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=str(stage),
        commit_message="Publish 15 SFT + 15 PPO checkpoints for the SLM-RL-Agents framework",
    )
    print("[upload-model] done")


# --------------------------------------------------------------------------
# Stage 7: delete per-config repos
# --------------------------------------------------------------------------

def stage_cleanup_old(api: HfApi) -> None:
    print("[cleanup-old] deleting 3 per-config dataset repos")
    for rid in PER_CONFIG_DATASET_REPOS:
        try:
            api.delete_repo(rid, repo_type="dataset")
            print(f"  deleted dataset: {rid}")
        except Exception as e:
            print(f"  skip {rid}: {e}")

    print("[cleanup-old] deleting 30 per-config model repos")
    for rid in PER_CONFIG_MODEL_REPOS:
        try:
            api.delete_repo(rid, repo_type="model")
            print(f"  deleted model  : {rid}")
        except Exception as e:
            print(f"  skip {rid}: {e}")
    print("[cleanup-old] done")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

STAGES = [
    "rename-data",
    "clean-data",
    "upload-data",
    "create-model",
    "stage-model",
    "upload-model",
    "cleanup-old",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES + ["all"], required=True)
    args = parser.parse_args()

    api = HfApi()
    who = api.whoami()
    print(f"Logged in as: {who.get('name')}")
    print()

    wanted = STAGES if args.stage == "all" else [args.stage]
    for stg in wanted:
        print(f"=== STAGE: {stg} ===")
        if   stg == "rename-data":   stage_rename_data(api)
        elif stg == "clean-data":    stage_clean_data(api)
        elif stg == "upload-data":   stage_upload_data(api)
        elif stg == "create-model":  stage_create_model(api)
        elif stg == "stage-model":   stage_stage_model()
        elif stg == "upload-model":  stage_upload_model(api)
        elif stg == "cleanup-old":   stage_cleanup_old(api)
        print()
    print("ALL DONE")


if __name__ == "__main__":
    main()
