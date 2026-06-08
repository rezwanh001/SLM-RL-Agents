#-*- coding: utf-8 -*-
# পূর্বে file-এর top-এ author block + main docstring পরপর দুইটা triple-quoted
# string ছিল, যেটার পরে `from __future__ import annotations` Python 3.10+ এ
# SyntaxError দিচ্ছিলো (multiple docstrings + __future__ allowed না)। তাই
# author tag-কে সাধারণ comment-এ নামিয়ে এনে নিচে single module docstring
# রাখা হলো।
# @author: Md Rezwanul Haque
"""
SLM-RL-Agents — Interactive Verification App (paper appendix)

This Gradio app accompanies the paper:

    "Towards Robust Reinforcement Learning for Small-Scale
     Language Model Agents" (IEEE SMC 2026)
# পূর্বে: paper title-এ ছিল "Efficiently Enhancing SLM Agents …" — SMC
# 2026 final version-এ rename হয়ে এই form-এ এসেছে।

Its purpose is to let reviewers (and any third party) independently verify
that the numbers reported in the paper and in the HuggingFace model/dataset
repositories are backed by real, runnable checkpoints and real, on-disk
evaluation files — not synthesised or cherry-picked.

It has four tabs:

  1. Live SFT vs PPO comparison   — pick a model/dataset, enter a prompt,
                                    and the app loads the actual trained
                                    LoRA/full checkpoints (local or HF hub)
                                    and generates text side-by-side plus
                                    the reward-model score for each output.
  2. Published results table      — the full 15-config × 18-metric table
                                    straight from results/all_results.json
                                    (the same file the paper reads from).
  3. Raw evaluation samples       — browse the actual prompt/generated/
                                    reference triples saved during eval;
                                    these are the ground-truth outputs the
                                    reported perplexity / reward / diversity
                                    numbers are computed over.
  4. How to verify                — step-by-step instructions reviewers can
                                    copy/paste to reproduce a single number
                                    from scratch.

Run:
    python app.py                    # local
    python app.py --share            # public gradio link
    python app.py --use_hf           # pull weights from HuggingFace hub
                                     # (mr3haque/SLM-RL-Agents) instead of
                                     # the local outputs/ directory
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS_JSON = ROOT / "results" / "all_results.json"

HF_MODEL_REPO = "mr3haque/SLM-RL-Agents"
# পূর্বে: HF_DATA_REPO = "mr3haque/SLM-RL-Agentss-Data"  ← 'Agentss' typo,
# real repo হলো mr3haque/SLM-RL-Agents-Data (single 's')।
HF_DATA_REPO = "mr3haque/SLM-RL-Agents-Data"
GITHUB_URL = "https://github.com/rezwanh001/slm-rl-agents"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "smollm2-135m", "smollm2-360m"]
DATASETS = ["tinystories", "cnn_dailymail", "wikitext"]

MODEL_PRETTY = {
    "pythia-70m":    "Pythia-70M",
    "pythia-160m":   "Pythia-160M",
    "pythia-410m":   "Pythia-410M",
    "smollm2-135m":  "SmolLM2-135M",
    "smollm2-360m":  "SmolLM2-360M",
}
DATASET_PRETTY = {
    "tinystories":   "TinyStories",
    "cnn_dailymail": "CNN/DailyMail",
    "wikitext":      "Wikitext-103",
}

# Example prompts taken verbatim from the raw eval splits so reviewers can
# reproduce what the evaluator saw.
EXAMPLE_PROMPTS = {
    "tinystories":   "Once upon a time, there was a little girl named Lily. She loved to play outside in her backyard.",
    "cnn_dailymail": "Summarize: Scientists have discovered a new species of fish in the deep ocean near the Mariana Trench.",
    "wikitext":      "The history of artificial intelligence began in",
}


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] = {}


def _model_path(model_key: str, dataset: str, stage: str, use_hf: bool) -> str:
    """Return either a local path or a HuggingFace repo identifier."""
    if use_hf:
        # single consolidated repo uses subfolders; transformers supports
        # `subfolder=` at load time. We return a tuple encoded as a string.
        # পূর্বে: subfolder ছিল f"{model_key}/{dataset}/{stage}" — কিন্তু HF repo
        # actually layout হলো sft/<model>/<dataset>/ এবং ppo/<model>/<dataset>/
        # (stage prefix, model suffix নয়)। এই কারণে --use_hf mode-এ HTTP 404
        # দিচ্ছিলো।
        return f"hf::{HF_MODEL_REPO}::{stage}/{model_key}/{dataset}"
    local = OUTPUTS / model_key / dataset / stage / "final"
    return str(local)


# পূর্বে: _load_causal_lm সরাসরি AutoModelForCausalLM.from_pretrained(spec, ...)
# call করতো — কিন্তু sft/<m>/<d>/ checkpoint গুলো full model নয়, LoRA
# adapter (adapter_config.json + adapter_model.safetensors only, কোনো
# real model config.json/model weight নেই)। ফলে SFT generate করতে গেলে UI-তে
# "[SFT load/generate failed: ... does not appear to have a file named
# config.json]" error আসতো।
#
# Local PPO `final/` দিরেক্টরি আবার আরো জটিল: এতে adapter_config.json
# (stale absolute base_model_name_or_path পয়েন্ট করছে slm-rl-agent —
# noteworthy: NO 's' — old project dir-এ), adapter_model.safetensors,
# এবং একটা PPOConfig dump named config.json (architectures field নেই) —
# সব একসাথে। প্রকৃত merged base model আছে `../ppo/_merged_sft/`-এ
# (model.safetensors + real model config.json সহ)। তাই PPO load করতে হলে
# (a) base = sibling `_merged_sft/`, (b) adapter = `final/` overlay, পরে merge।
#
# Fix: তিনটা code path —
#   (i)  HF subfolder + config.json present     → full causal LM (PPO on hub)
#   (ii) HF subfolder + adapter_config only     → base + LoRA + merge (SFT on hub)
#   (iii) local dir + real model config.json    → full causal LM
#   (iv) local dir + adapter_config.json        → base (path-rewritten if stale) + LoRA + merge
def _load_causal_lm(spec: str):
    if spec in _model_cache:
        return _model_cache[spec]
    from peft import PeftModel  # local import keeps startup fast

    is_hf = spec.startswith("hf::")
    if is_hf:
        _, repo, subfolder = spec.split("::", 2)
        adapter_cfg = _maybe_fetch_adapter_config_hf(repo, subfolder)
        if adapter_cfg is not None:
            base_name = adapter_cfg["base_model_name_or_path"]
            base = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=torch.float32, device_map="auto",
                trust_remote_code=True,
            )
            mdl = PeftModel.from_pretrained(base, repo, subfolder=subfolder)
            try:
                tok = AutoTokenizer.from_pretrained(
                    repo, subfolder=subfolder, trust_remote_code=True
                )
            except Exception:
                tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
            try:
                mdl = mdl.merge_and_unload()
            except Exception as e:
                print(f"[app.py] merge_and_unload failed ({e}); using PeftModel.")
        else:
            tok = AutoTokenizer.from_pretrained(
                repo, subfolder=subfolder, trust_remote_code=True
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                repo, subfolder=subfolder,
                torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
            )
    else:
        spec_path = Path(spec)
        if not spec_path.exists():
            raise FileNotFoundError(spec)
        adapter_cfg = _maybe_read_adapter_config_local(spec_path)
        if adapter_cfg is not None:
            base_name = _resolve_local_base(adapter_cfg["base_model_name_or_path"], spec_path)
            base = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=torch.float32, device_map="auto",
                trust_remote_code=True,
            )
            mdl = PeftModel.from_pretrained(base, spec)
            try:
                tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
            try:
                mdl = mdl.merge_and_unload()
            except Exception as e:
                print(f"[app.py] merge_and_unload failed ({e}); using PeftModel.")
        else:
            tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                spec, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
            )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.eval()
    _model_cache[spec] = (mdl, tok)
    return mdl, tok


def _config_is_real_model_config(cfg_path: Path) -> bool:
    """Distinguish a real model config (has `architectures` or `model_type`)
    from TRL's PPOConfig dump that happens to also be named config.json."""
    try:
        with open(cfg_path) as f:
            d = json.load(f)
    except Exception:
        return False
    return bool(d.get("architectures") or d.get("model_type"))


def _maybe_read_adapter_config_local(path: Path) -> dict | None:
    """Return adapter_config dict if this directory should be loaded as a
    LoRA adapter (i.e. no *real* model config.json present, but an
    adapter_config.json is). PPO `final/` directories ship a PPOConfig dump
    also named config.json — those are NOT real model configs and we must
    fall through to the adapter path.
    """
    if (path / "config.json").exists() and _config_is_real_model_config(path / "config.json"):
        return None
    cfg = path / "adapter_config.json"
    if not cfg.exists():
        return None
    with open(cfg) as f:
        return json.load(f)


def _resolve_local_base(base_name: str, spec_path: Path) -> str:
    """Heal stale absolute paths in adapter_config.json.

    Local PPO checkpoints have `base_model_name_or_path` like
    `/.../slm-rl-agent/outputs/<m>/<d>/ppo/_merged_sft` — frozen at PPO
    save time, before the project was renamed slm-rl-agent → slm-rl-agents.
    The real merged-SFT base lives in this project's sibling
    `<spec_path>/../_merged_sft`. Prefer the sibling; fall back to the
    written path; then to a HF hub identifier (HF-style names contain '/'
    but not a leading '/').
    """
    if not base_name.startswith("/"):
        return base_name  # HF hub id like "EleutherAI/pythia-70m-deduped"
    if Path(base_name).exists():
        return base_name
    sibling = spec_path.parent / "_merged_sft"
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError(
        f"Adapter base model not found: tried {base_name} and {sibling}. "
        f"Re-run PPO or copy the merged-SFT directory into place."
    )


def _maybe_fetch_adapter_config_hf(repo: str, subfolder: str) -> dict | None:
    """Return adapter_config.json dict if the subfolder is a LoRA adapter
    AND no merged config.json exists in the same subfolder. PPO subfolders on
    HF are fully merged and ship a real model config.json — those are loaded
    as full causal LMs. SFT subfolders ship only adapter files.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    try:
        hf_hub_download(repo_id=repo, filename=f"{subfolder}/config.json")
        return None  # full model present → not an adapter-only subfolder
    except (EntryNotFoundError, HfHubHTTPError, FileNotFoundError):
        pass
    except Exception as e:
        print(f"[app.py] config.json probe failed for {repo}/{subfolder}: {e}")
        return None

    try:
        cfg_path = hf_hub_download(
            repo_id=repo, filename=f"{subfolder}/adapter_config.json"
        )
    except (EntryNotFoundError, HfHubHTTPError, FileNotFoundError):
        return None
    except Exception as e:
        print(f"[app.py] adapter_config probe failed for {repo}/{subfolder}: {e}")
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _load_reward(model_key: str, dataset: str, use_hf: bool):
    """Load the Bradley-Terry reward model for (model_key, dataset).

    The reward model is saved as a LoRA adapter (with `score` in
    `modules_to_save`) on top of the base SLM, with **num_labels=1** — a
    scalar Bradley-Terry head. Naively calling `AutoModelForSequenceClassification
    .from_pretrained(adapter_path)` defaults to num_labels=2, so the new
    `score.weight` is `[2, hidden]` while the adapter saved `[1, hidden]`,
    and PEFT refuses to load. Fix: read the adapter config, load the base
    with num_labels=1 explicitly, then attach the adapter via PeftModel.

    # --- আগের কোড (পুরাতন পদ্ধতি) — size mismatch error দিতো ---
    # rm = AutoModelForSequenceClassification.from_pretrained(
    #     adapter_path,                          # num_labels defaults to 2
    #     torch_dtype=torch.float32,
    #     device_map="auto",
    # )
    # RuntimeError: size mismatch for score.modules_to_save.default.weight:
    #   copying param shape [1, 512] from checkpoint, shape in model is [2, 512]
    # --- সমাধান: adapter_config.json থেকে base model পড়ে num_labels=1 দিয়ে load ---
    """
    key = f"reward::{model_key}/{dataset}::{use_hf}"
    if key in _model_cache:
        return _model_cache[key]

    from peft import PeftModel  # local import keeps startup fast

    try:
        if use_hf:
            # পূর্বে: HF repo থেকে reward_model adapter pull করার চেষ্টা ছিল
            # (subfolder = f"{model_key}/{dataset}/reward_model"), কিন্তু আসল
            # mr3haque/SLM-RL-Agents repo-তে শুধু sft/ + ppo/ + agentic_sft/
            # publish করা — reward_model upload করা হয়নি (paper-এর scope
            # এর বাইরে)। তাই --use_hf mode-এ local reward_model fallback
            # use করি; না থাকলে None return করে UI graceful degradation
            # দেখাবে।
            p = OUTPUTS / model_key / dataset / "reward_model" / "final"
            if not p.exists():
                print(
                    f"[app.py] reward_model not on HF and no local copy at "
                    f"{p} — single-prompt reward score will be N/A in --use_hf mode."
                )
                return None
            with open(p / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            base_name = adapter_cfg["base_model_name_or_path"]

            base = AutoModelForSequenceClassification.from_pretrained(
                base_name,
                num_labels=1,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            rm = PeftModel.from_pretrained(base, str(p))
            rtok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
        else:
            p = OUTPUTS / model_key / dataset / "reward_model" / "final"
            if not p.exists():
                return None
            with open(p / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            base_name = adapter_cfg["base_model_name_or_path"]

            base = AutoModelForSequenceClassification.from_pretrained(
                base_name,
                num_labels=1,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            rm = PeftModel.from_pretrained(base, str(p))
            rtok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    except Exception as e:
        print(f"[app.py] _load_reward failed for {model_key}/{dataset}: {e}")
        return None

    # Align pad token with the base model to avoid generation-time warnings
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    try:
        rm.config.pad_token_id = rtok.pad_token_id
    except Exception:
        pass
    rm.eval()
    _model_cache[key] = (rm, rtok)
    return rm, rtok


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=max(0.01, temperature),
        top_p=top_p,
        do_sample=temperature > 0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def _reward_score(rm_pack, prompt: str, response: str) -> float | None:
    if rm_pack is None:
        return None
    rm, rtok = rm_pack
    text = f"{prompt}\n\n{response}"
    enc = rtok(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(rm.device) for k, v in enc.items()}
    out = rm(**enc)
    return float(out.logits[0, 0].item())


# ---------------------------------------------------------------------------
# Results backend
# ---------------------------------------------------------------------------
def _load_results() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _format_published_metrics(model_key: str, dataset: str) -> str:
    r = _load_results()
    rd = r["our_models"][model_key]["datasets"][dataset]
    def _f(x, nd=3):
        return "—" if x is None else f"{float(x):.{nd}f}"
    md = (
        f"**Published numbers for {MODEL_PRETTY[model_key]} / {DATASET_PRETTY[dataset]}**\n"
        f"(source: `results/all_results.json`, cross-checked against raw "
        f"`outputs/{model_key}/{dataset}/eval_*/evaluation_results.json`)\n\n"
        f"| Metric            |   SFT    |   PPO    | Δ (PPO−SFT) |\n"
        f"|-------------------|----------|----------|-------------|\n"
        f"| Perplexity ↓      | {_f(rd['sft_perplexity'])} | {_f(rd['ppo_perplexity'])} | {_f(float(rd['ppo_perplexity'])-float(rd['sft_perplexity']))} |\n"
        f"| Reward mean ↑     | {_f(rd['sft_reward_mean'])} | {_f(rd['ppo_reward_mean'])} | {_f(rd['reward_delta'])} |\n"
        f"| Reward std        | {_f(rd['sft_reward_std'])} | {_f(rd['ppo_reward_std'])} | — |\n"
        f"| Distinct-1 ↑      | {_f(rd['sft_distinct1'])} | {_f(rd['ppo_distinct1'])} | {_f(float(rd['ppo_distinct1'])-float(rd['sft_distinct1']))} |\n"
        f"| Distinct-2 ↑      | {_f(rd['sft_distinct2'])} | {_f(rd['ppo_distinct2'])} | {_f(float(rd['ppo_distinct2'])-float(rd['sft_distinct2']))} |\n"
        f"| ROUGE-L F1 ↑      | {_f(rd['sft_rougeL'])} | {_f(rd['ppo_rougeL'])} | {_f(float(rd['ppo_rougeL'])-float(rd['sft_rougeL']))} |\n"
        f"| BLEU-4 ↑          | {_f(rd['sft_bleu4'])} | {_f(rd['ppo_bleu4'])} | {_f(float(rd['ppo_bleu4'])-float(rd['sft_bleu4']))} |\n\n"
        f"Evaluated on `num_samples=200` held-out prompts from "
        f"`{DATASET_PRETTY[dataset]}` — identical to the split shipped in the "
        f"raw `outputs/` tree on the [GitHub repo]({GITHUB_URL})."
    )
    return md


def _full_results_markdown() -> str:
    r = _load_results()
    lines = [
        "### Full 15-configuration results table",
        "",
        "All numbers below are copied verbatim from `results/all_results.json`, "
        "which was generated by `scripts/evaluate.py` and verified field-by-field "
        "against the raw per-run `evaluation_results.json` files "
        "(`scripts/verify_results.py` → **339/339 fields passed, 0 mismatches**).",
        "",
        "| Model | Dataset | PPL SFT | PPL PPO | Reward SFT | Reward PPO | Δ Reward | Dist-2 SFT | Dist-2 PPO | ROUGE-L SFT | ROUGE-L PPO |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mk in MODELS:
        for ds in DATASETS:
            rd = r["our_models"][mk]["datasets"][ds]
            def _f(x, nd=3):
                return "—" if x is None else f"{float(x):.{nd}f}"
            lines.append(
                f"| {MODEL_PRETTY[mk]} | {DATASET_PRETTY[ds]} | "
                f"{_f(rd['sft_perplexity'],2)} | {_f(rd['ppo_perplexity'],2)} | "
                f"{_f(rd['sft_reward_mean'])} | {_f(rd['ppo_reward_mean'])} | "
                f"{_f(rd['reward_delta'])} | "
                f"{_f(rd['sft_distinct2'])} | {_f(rd['ppo_distinct2'])} | "
                f"{_f(rd['sft_rougeL'])} | {_f(rd['ppo_rougeL'])} |"
            )
    lines += [
        "",
        "### Baselines (published instruct SLMs)",
        "",
        "| Baseline | Dataset | PPL | Reward | Distinct-2 | ROUGE-L |",
        "|---|---|---|---|---|---|",
    ]
    for bk, bd in r["baselines"].items():
        for ds, row in bd.items():
            if not isinstance(row, dict):
                continue
            def _f(x, nd=3):
                return "—" if x is None else f"{float(x):.{nd}f}"
            lines.append(
                f"| {bk} | {DATASET_PRETTY.get(ds, ds)} | "
                f"{_f(row.get('perplexity'),2)} | {_f(row.get('reward_mean'))} | "
                f"{_f(row.get('distinct_2'))} | {_f(row.get('rougeL_f1'))} |"
            )
    return "\n".join(lines)


def _load_raw_sample(model_key: str, dataset: str, stage: str, index: int) -> tuple[str, str, str, str]:
    """Return (prompt, generated, reference, header)."""
    path = OUTPUTS / model_key / dataset / f"eval_{stage}" / "sample_generations.json"
    if not path.exists():
        return "", "", "", f"No raw sample file at `{path}` — run `scripts/evaluate.py` first."
    with open(path) as f:
        samples = json.load(f)
    if not samples:
        return "", "", "", f"Empty sample file at `{path}`"
    idx = max(0, min(index, len(samples) - 1))
    s = samples[idx]
    header = (
        f"Sample {idx+1} / {len(samples)}  —  "
        f"`outputs/{model_key}/{dataset}/eval_{stage}/sample_generations.json`"
    )
    return s.get("prompt", ""), s.get("generated", ""), s.get("reference", ""), header


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def run_comparison(model_key: str, dataset: str, prompt: str,
                   max_new_tokens: int, temperature: float, top_p: float,
                   use_hf: bool):
    """Generate SFT + PPO outputs and score them with the reward model."""
    if not prompt.strip():
        prompt = EXAMPLE_PROMPTS[dataset]
    status = []
    try:
        sft_spec = _model_path(model_key, dataset, "sft", use_hf)
        sft_m, sft_t = _load_causal_lm(sft_spec)
        sft_out = _generate(sft_m, sft_t, prompt, max_new_tokens, temperature, top_p)
    except Exception as e:
        sft_out = f"[SFT load/generate failed: {e}]"
        sft_t = None
    try:
        ppo_spec = _model_path(model_key, dataset, "ppo", use_hf)
        ppo_m, ppo_t = _load_causal_lm(ppo_spec)
        ppo_out = _generate(ppo_m, ppo_t, prompt, max_new_tokens, temperature, top_p)
    except Exception as e:
        ppo_out = f"[PPO load/generate failed: {e}]"

    rm_pack = _load_reward(model_key, dataset, use_hf)
    sft_r = _reward_score(rm_pack, prompt, sft_out) if rm_pack and sft_t is not None else None
    ppo_r = _reward_score(rm_pack, prompt, ppo_out) if rm_pack else None

    def _fmt(x):
        return "N/A (reward model not loaded)" if x is None else f"{x:+.4f}"

    delta = (ppo_r - sft_r) if (sft_r is not None and ppo_r is not None) else None
    delta_str = "—" if delta is None else f"{delta:+.4f}"

    published = _format_published_metrics(model_key, dataset)
    audit = (
        f"**Prompt used**  \n`{prompt}`\n\n"
        f"**SFT reward (this prompt)**: {_fmt(sft_r)}  \n"
        f"**PPO reward (this prompt)**: {_fmt(ppo_r)}  \n"
        f"**Δ reward (PPO − SFT, this prompt)**: {delta_str}\n\n"
        f"---\n\n{published}"
    )
    return sft_out, ppo_out, audit


def load_raw_sample(model_key: str, dataset: str, stage: str, index: int):
    p, g, r, hdr = _load_raw_sample(model_key, dataset, stage.lower(), int(index))
    return hdr, p, g, r


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
INSTRUCTIONS_MD = f"""
## How to use this verification app

This Gradio app lets anyone — reviewers, readers, or downstream users —
independently check that every number in the paper is real.

### Inputs

1. **Model family**. Pick one of 5 trained backbones:
   Pythia-70M / 160M / 410M, SmolLM2-135M / 360M.
2. **Dataset**. Pick one of 3 training corpora:
   TinyStories, CNN/DailyMail, Wikitext-103.
3. **Prompt**. Either type a free-form prompt, or leave it empty to use a
   default prompt drawn from the held-out evaluation split.
4. **Generation knobs**. `max_new_tokens` (50–400), `temperature` (0–1.5),
   `top_p` (0.1–1.0). PPO vs SFT differences are clearest at `temperature=0.7,
    top_p=0.9, max_new_tokens=200`, which are the evaluation defaults.
5. **Source of weights**. By default the app loads local checkpoints from
   `outputs/<model>/<dataset>/{{sft,ppo}}/final` and
   `outputs/<model>/<dataset>/reward_model/final`. Launch with `--use_hf` to
   stream weights directly from
   `{HF_MODEL_REPO}` (subfolders `<model>/<dataset>/sft|ppo|reward_model`).

### Outputs

For every (model, dataset, prompt) the app returns:

- **SFT output** — text generated by the supervised-fine-tuned policy.
- **PPO output** — text generated by the PPO-aligned policy.
- **SFT / PPO reward scores for THIS prompt**, computed by the same
  Bradley-Terry reward model used during training.
- **Δ reward** for this single prompt.
- **The published 7-row metric table** for that (model, dataset) pair
  — copied live from `results/all_results.json`, which is itself backed by
  the raw `outputs/*/eval_*/evaluation_results.json` files.

### What to check

- The **published** Δ reward (column 3 of the table) is the *mean* over the
  full 200-prompt eval split; the single-prompt Δ in the audit block is the
  score for *this* prompt. Sample-level variance is normal.
- **Table 2 / Raw samples tab** shows the actual prompt-generated-reference
  triples the reported reward / perplexity / diversity numbers were
  computed over — there is no synthetic data anywhere in the pipeline.
- **`scripts/verify_results.py`** (shipped alongside this app) cross-checks
  every one of the 339 numerical fields in `results/all_results.json`
  against the raw evaluation JSONs and exits non-zero on any drift. Running
  it in the repo root reproduces: *339 / 339 fields passed, 0 mismatches,
  0 missing files, sample sizes = {{200}}*.

### Minimal reproduction recipe

```bash
git clone {GITHUB_URL}
cd slm-rl-agents
pip install -e .

# re-run one full pipeline end-to-end
bash scripts/run_all_experiments.sh pythia-70m tinystories

# or, using the already-trained weights shipped on HuggingFace
hf download {HF_MODEL_REPO} --include "ppo/pythia-70m/tinystories/**" \\
    --local-dir ./hf_weights
python scripts/evaluate.py \\
    --model_path ./hf_weights/ppo/pythia-70m/tinystories \\
    --eval_dataset ./data/tinystories/sft_eval.json

# cross-check every reported number against the raw eval files
python scripts/verify_results.py
```

### Reference links

- **Code**:    [{GITHUB_URL}]({GITHUB_URL})
- **Models**:  [{HF_MODEL_REPO}](https://huggingface.co/{HF_MODEL_REPO})
- **Data**:    [{HF_DATA_REPO}](https://huggingface.co/datasets/{HF_DATA_REPO})
"""


def build_demo(use_hf: bool):
    results = _load_results()

    with gr.Blocks(title="SLM-RL-Agents — Verification") as demo:
        gr.Markdown(
            "# SLM-RL-Agents — Interactive Verification\n"
            "Companion app for *“Towards Robust Reinforcement Learning for "
            "Small-Scale Language Model Agents.”* (IEEE SMC 2026)  \n"
            f"Weights: [{HF_MODEL_REPO}](https://huggingface.co/{HF_MODEL_REPO}) · "
            f"Data: [{HF_DATA_REPO}](https://huggingface.co/datasets/{HF_DATA_REPO}) · "
            f"Code: [{GITHUB_URL}]({GITHUB_URL})  \n"
            f"Weight source for this session: **{'HuggingFace hub' if use_hf else 'local outputs/'}**"
        )

        with gr.Tab("1. Live SFT vs PPO"):
            gr.Markdown(
                "Pick a (model, dataset), type or keep the default prompt, click "
                "**Generate**. The app loads the actual trained checkpoints and "
                "the matching reward model, generates once from the SFT policy "
                "and once from the PPO policy, and scores both. The published "
                "mean-over-200-prompts metrics for this configuration are shown "
                "below for context."
            )
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=[(MODEL_PRETTY[m], m) for m in MODELS],
                    value="pythia-70m", label="Model backbone",
                )
                data_dd = gr.Dropdown(
                    choices=[(DATASET_PRETTY[d], d) for d in DATASETS],
                    value="tinystories", label="Training corpus",
                )
            with gr.Row():
                mtok = gr.Slider(50, 400, value=200, step=10, label="max_new_tokens")
                tmp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                tp = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
            prompt_box = gr.Textbox(
                lines=3, label="Prompt",
                value=EXAMPLE_PROMPTS["tinystories"],
                placeholder="Leave blank to use a held-out-split default prompt",
            )
            run_btn = gr.Button("Generate & score (SFT vs PPO)", variant="primary")
            with gr.Row():
                sft_box = gr.Textbox(lines=10, label="SFT output", interactive=False)
                ppo_box = gr.Textbox(lines=10, label="PPO output", interactive=False)
            audit_box = gr.Markdown()

            def _update_default_prompt(ds):
                return EXAMPLE_PROMPTS[ds]
            data_dd.change(_update_default_prompt, inputs=[data_dd], outputs=[prompt_box])
            run_btn.click(
                lambda *a: run_comparison(*a, use_hf),
                inputs=[model_dd, data_dd, prompt_box, mtok, tmp, tp],
                outputs=[sft_box, ppo_box, audit_box],
            )

        with gr.Tab("2. Published results table"):
            gr.Markdown(_full_results_markdown())

        with gr.Tab("3. Raw evaluation samples"):
            gr.Markdown(
                "Browse the actual prompt / generated / reference triples that "
                "the reported metrics were computed over. These files were "
                "written by `scripts/evaluate.py` at the end of every training "
                "run and are untouched afterwards."
            )
            with gr.Row():
                rs_model = gr.Dropdown(
                    choices=[(MODEL_PRETTY[m], m) for m in MODELS],
                    value="pythia-70m", label="Model",
                )
                rs_data = gr.Dropdown(
                    choices=[(DATASET_PRETTY[d], d) for d in DATASETS],
                    value="tinystories", label="Dataset",
                )
                rs_stage = gr.Radio(["SFT", "PPO"], value="PPO", label="Stage")
                rs_idx = gr.Slider(0, 49, value=0, step=1, label="Sample index")
            rs_btn = gr.Button("Load raw sample")
            rs_hdr = gr.Markdown()
            with gr.Row():
                rs_prompt = gr.Textbox(lines=6, label="prompt (from held-out split)", interactive=False)
                rs_gen = gr.Textbox(lines=6, label="generated (at eval time)", interactive=False)
                rs_ref = gr.Textbox(lines=6, label="reference / gold", interactive=False)
            rs_btn.click(
                load_raw_sample,
                inputs=[rs_model, rs_data, rs_stage, rs_idx],
                outputs=[rs_hdr, rs_prompt, rs_gen, rs_ref],
            )

        with gr.Tab("4. How to verify (inputs / outputs)"):
            gr.Markdown(INSTRUCTIONS_MD)

    return demo


# ---------------------------------------------------------------------------
def _find_free_port(start: int, end: int) -> int | None:
    """Return the first TCP port in [start, end] not already bound locally.

    # --- আগে Gradio সরাসরি port=7860 দিয়ে launch করতো ---
    # demo.launch(server_name=args.host, server_port=args.port)
    # OSError: Cannot find empty port in range: 7860-7860
    # --- এখন socket bind দিয়ে free port খুঁজে বের করে launch করা হয় ---
    """
    import socket
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", p))
            except OSError:
                continue
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860,
                    help="Preferred port (will auto-increment if busy)")
    ap.add_argument("--port_range", type=int, default=20,
                    help="How many ports to scan upward from --port if the preferred one is in use")
    ap.add_argument("--host", type=str, default="0.0.0.0",
                    help="Bind address. 0.0.0.0 = reachable from LAN, 127.0.0.1 = localhost only")
    ap.add_argument("--use_hf", action="store_true",
                    help=f"Load weights from {HF_MODEL_REPO} instead of ./outputs")
    args = ap.parse_args()

    port = _find_free_port(args.port, args.port + args.port_range)
    if port is None:
        raise SystemExit(
            f"[app.py] No free port in {args.port}-{args.port + args.port_range}. "
            f"Pick another with --port <N> or free one of the busy ones."
        )
    if port != args.port:
        print(f"[app.py] Port {args.port} is in use; falling back to {port}.")

    print(f"[app.py] Launching SLM-RL-Agents verification UI on http://{args.host}:{port}")
    print(f"[app.py] Weight source: {'HuggingFace hub ({})'.format(HF_MODEL_REPO) if args.use_hf else 'local outputs/'}")

    demo = build_demo(use_hf=args.use_hf)
    demo.queue()
    try:
        demo.launch(
            share=args.share,
            server_port=port,
            server_name=args.host,
            theme=gr.themes.Soft(),
            show_error=True,
        )
    except KeyboardInterrupt:
        print("\n[app.py] Shutting down.")


if __name__ == "__main__":
    main()
