#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from models.config import TransformerConfig

# =========================
# data classes
# =========================


@dataclass
class TokenDecomp:
    token_id: int
    piece: str
    text: str
    logit: float
    prob: float
    weight_norm: float
    hidden_norm: float
    cosine: float
    dot: float
    rank: int


@dataclass
class SampleResult:
    index: int
    prompt: str
    target_text: str
    prompt_last_token_piece: str
    prompt_last_token_text: str
    target_token_id: int
    target_piece: str
    target_text_decoded: str
    top1_token_id: int
    top1_piece: str
    top1_text: str
    top1_correct: bool
    target_rank: int
    target_margin_vs_top1: float
    hidden_norm: float
    topk: list[TokenDecomp]
    target_decomp: TokenDecomp
    top1_decomp: TokenDecomp
    tied_top1_text: str | None = None
    tied_target_rank: int | None = None
    tied_target_margin_vs_top1: float | None = None


# =========================
# model loading
# =========================


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_config(model_path: Path) -> TransformerConfig:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    return TransformerConfig.model_validate_json(config_path.read_text(encoding="utf-8"))


def build_model(config: TransformerConfig, device: torch.device):
    os_name = platform.system()
    if os_name == "Linux":
        import models.liger.causal_lm as causal_lm_module
    else:
        import models.torch.causal_lm as causal_lm_module

    model = causal_lm_module.CausalLanguageModel(config=config).to(device)
    model.eval()
    return model


def load_model_and_tokenizer(
    model_path: str,
    checkpoint_path: str | None,
    device: torch.device,
):
    model_dir = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = load_config(model_dir)
    model = build_model(config, device=device)

    if checkpoint_path is None:
        ckpt = model_dir / "model.safetensors"
    else:
        ckpt = Path(checkpoint_path)

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    state_dict = load_file(str(ckpt), device=str(device))
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")

    return tokenizer, config, model


# =========================
# helpers
# =========================


def safe_piece(tokenizer, token_id: int) -> str:
    piece = tokenizer.convert_ids_to_tokens(token_id)
    if piece is None:
        return "<None>"
    return str(piece)


def safe_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return text.replace("\n", "\\n")


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def build_token_decomp(
    tokenizer,
    token_id: int,
    logits: torch.Tensor,
    probs: torch.Tensor,
    last_hidden: torch.Tensor,
    weight: torch.Tensor,
) -> TokenDecomp:
    w = weight[token_id]
    dot = torch.dot(last_hidden, w).item()

    hidden_norm = last_hidden.norm().item()
    weight_norm = w.norm().item()

    if hidden_norm == 0.0 or weight_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (hidden_norm * weight_norm)

    return TokenDecomp(
        token_id=int(token_id),
        piece=safe_piece(tokenizer, int(token_id)),
        text=safe_text(tokenizer, int(token_id)),
        logit=float(logits[token_id].item()),
        prob=float(probs[token_id].item()),
        weight_norm=float(weight_norm),
        hidden_norm=float(hidden_norm),
        cosine=float(cosine),
        dot=float(dot),
        rank=token_rank(logits, int(token_id)),
    )


def resolve_probe_token_ids(tokenizer, probe_tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in probe_tokens:
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            print(f"[warn] probe token {tok!r} is not single-token under current tokenizer, skip. ids={ids}")
            continue
        out[tok] = int(ids[0])
    return out


def offdiag_stats(cos_matrix: torch.Tensor) -> dict[str, float]:
    n = cos_matrix.size(0)
    if n <= 1:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}

    mask = ~torch.eye(n, dtype=torch.bool, device=cos_matrix.device)
    vals = cos_matrix[mask]
    return {
        "mean": float(vals.mean().item()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
    }


def pretty_token(token: str) -> str:
    return token.replace("\n", "\\n")


def load_cases(path: str) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "target" not in obj:
                raise ValueError(f"Line {i}: each json line must contain 'prompt' and 'target'")
            cases.append({"prompt": str(obj["prompt"]), "target": str(obj["target"])})
    if not cases:
        raise ValueError("No valid cases found.")
    return cases


# =========================
# main analysis
# =========================


@torch.no_grad()
def analyze_cases(
    model,
    tokenizer,
    cases: list[dict[str, str]],
    device: torch.device,
    topk: int,
    probe_tokens: list[str],
    compare_tied: bool,
) -> dict[str, Any]:
    lm_head_weight = model.lm_head.weight.detach().float()
    embedding_weight = model.decoder.embeddings.weight.detach().float()

    wrong_top1_counter: Counter[int] = Counter()
    wrong_topk_counter: Counter[int] = Counter()
    probe_stats: dict[str, dict[str, list[float] | int]] = {}
    probe_token_ids = resolve_probe_token_ids(tokenizer, probe_tokens)

    for tok in probe_token_ids:
        probe_stats[tok] = {
            "logits": [],
            "cosines": [],
            "ranks": [],
            "top1_count": 0,
        }

    hidden_vectors: list[torch.Tensor] = []
    logits_vectors: list[torch.Tensor] = []
    results: list[SampleResult] = []

    for idx, case in enumerate(cases):
        prompt = case["prompt"]
        target_text = case["target"]

        encoded_prompt = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded_prompt["input_ids"].to(device)

        if input_ids.numel() == 0:
            raise ValueError(f"Case {idx}: prompt tokenized to empty sequence: {prompt!r}")

        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if len(target_ids) == 0:
            raise ValueError(f"Case {idx}: target tokenized to empty sequence: {target_text!r}")

        target_token_id = int(target_ids[0])

        hidden_states = model(input_ids)
        last_hidden = hidden_states[0, -1, :].detach().float()
        logits = F.linear(last_hidden.unsqueeze(0), lm_head_weight)[0]
        probs = torch.softmax(logits, dim=-1)

        topk_vals, topk_ids = torch.topk(logits, k=min(topk, logits.size(0)))
        top1_id = int(topk_ids[0].item())

        target_decomp = build_token_decomp(
            tokenizer=tokenizer,
            token_id=target_token_id,
            logits=logits,
            probs=probs,
            last_hidden=last_hidden,
            weight=lm_head_weight,
        )
        top1_decomp = build_token_decomp(
            tokenizer=tokenizer,
            token_id=top1_id,
            logits=logits,
            probs=probs,
            last_hidden=last_hidden,
            weight=lm_head_weight,
        )

        topk_decomps: list[TokenDecomp] = []
        for token_id in topk_ids.tolist():
            topk_decomps.append(
                build_token_decomp(
                    tokenizer=tokenizer,
                    token_id=int(token_id),
                    logits=logits,
                    probs=probs,
                    last_hidden=last_hidden,
                    weight=lm_head_weight,
                )
            )

        top1_correct = top1_id == target_token_id
        target_margin_vs_top1 = float(target_decomp.logit - top1_decomp.logit)

        if not top1_correct:
            wrong_top1_counter[top1_id] += 1
            for tid in topk_ids.tolist():
                tid = int(tid)
                if tid != target_token_id:
                    wrong_topk_counter[tid] += 1

        # probe tokens
        for probe_text, probe_id in probe_token_ids.items():
            probe = build_token_decomp(
                tokenizer=tokenizer,
                token_id=probe_id,
                logits=logits,
                probs=probs,
                last_hidden=last_hidden,
                weight=lm_head_weight,
            )
            probe_stats[probe_text]["logits"].append(probe.logit)
            probe_stats[probe_text]["cosines"].append(probe.cosine)
            probe_stats[probe_text]["ranks"].append(probe.rank)
            if top1_id == probe_id:
                probe_stats[probe_text]["top1_count"] += 1

        prompt_last_id = int(input_ids[0, -1].item())

        tied_top1_text = None
        tied_target_rank = None
        tied_target_margin_vs_top1 = None

        if compare_tied:
            tied_logits = F.linear(last_hidden.unsqueeze(0), embedding_weight)[0]
            tied_top1_id = int(torch.argmax(tied_logits).item())
            tied_top1_text = safe_text(tokenizer, tied_top1_id)
            tied_target_rank = token_rank(tied_logits, target_token_id)
            tied_target_margin_vs_top1 = float(tied_logits[target_token_id].item() - tied_logits[tied_top1_id].item())

        results.append(
            SampleResult(
                index=idx,
                prompt=prompt,
                target_text=target_text,
                prompt_last_token_piece=safe_piece(tokenizer, prompt_last_id),
                prompt_last_token_text=safe_text(tokenizer, prompt_last_id),
                target_token_id=target_token_id,
                target_piece=safe_piece(tokenizer, target_token_id),
                target_text_decoded=safe_text(tokenizer, target_token_id),
                top1_token_id=top1_id,
                top1_piece=safe_piece(tokenizer, top1_id),
                top1_text=safe_text(tokenizer, top1_id),
                top1_correct=top1_correct,
                target_rank=target_decomp.rank,
                target_margin_vs_top1=target_margin_vs_top1,
                hidden_norm=float(last_hidden.norm().item()),
                topk=topk_decomps,
                target_decomp=target_decomp,
                top1_decomp=top1_decomp,
                tied_top1_text=tied_top1_text,
                tied_target_rank=tied_target_rank,
                tied_target_margin_vs_top1=tied_target_margin_vs_top1,
            )
        )

        hidden_vectors.append(last_hidden.cpu())
        logits_vectors.append(logits.cpu())

    hidden_stack = torch.stack(hidden_vectors, dim=0)
    logits_stack = torch.stack(logits_vectors, dim=0)

    hidden_cos = F.normalize(hidden_stack, dim=-1) @ F.normalize(hidden_stack, dim=-1).T
    logits_cos = F.normalize(logits_stack, dim=-1) @ F.normalize(logits_stack, dim=-1).T

    # aggregate probe stats
    probe_summary = {}
    for tok, stat in probe_stats.items():
        logits_list = stat["logits"]
        cos_list = stat["cosines"]
        ranks_list = stat["ranks"]
        probe_summary[tok] = {
            "token_id": probe_token_ids[tok],
            "piece": safe_piece(tokenizer, probe_token_ids[tok]),
            "text": safe_text(tokenizer, probe_token_ids[tok]),
            "avg_logit": float(sum(logits_list) / len(logits_list)) if logits_list else float("nan"),
            "avg_cosine": float(sum(cos_list) / len(cos_list)) if cos_list else float("nan"),
            "avg_rank": float(sum(ranks_list) / len(ranks_list)) if ranks_list else float("nan"),
            "top1_count": int(stat["top1_count"]),
        }

    # serialize counters
    def counter_to_list(counter: Counter[int], limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for token_id, count in counter.most_common(limit):
            out.append(
                {
                    "token_id": int(token_id),
                    "piece": safe_piece(tokenizer, int(token_id)),
                    "text": safe_text(tokenizer, int(token_id)),
                    "count": int(count),
                }
            )
        return out

    top1_acc = sum(1 for x in results if x.top1_correct) / len(results)
    mean_target_rank = sum(x.target_rank for x in results) / len(results)
    mean_target_margin = sum(x.target_margin_vs_top1 for x in results) / len(results)

    summary = {
        "num_cases": len(results),
        "top1_accuracy": top1_acc,
        "mean_target_rank": mean_target_rank,
        "mean_target_margin_vs_top1": mean_target_margin,
        "hidden_cosine_offdiag": offdiag_stats(hidden_cos),
        "logits_cosine_offdiag": offdiag_stats(logits_cos),
        "wrong_top1_tokens": counter_to_list(wrong_top1_counter, limit=20),
        "wrong_topk_tokens": counter_to_list(wrong_topk_counter, limit=20),
        "probe_summary": probe_summary,
    }

    if compare_tied:
        valid = [x for x in results if x.tied_target_rank is not None]
        summary["tied_compare"] = {
            "mean_tied_target_rank": float(sum(x.tied_target_rank for x in valid) / len(valid)),
            "mean_tied_target_margin_vs_top1": float(
                sum(x.tied_target_margin_vs_top1 for x in valid if x.tied_target_margin_vs_top1 is not None)
                / len(valid)
            ),
        }

    report = {
        "summary": summary,
        "samples": [asdict(x) for x in results],
    }
    return report


# =========================
# printing
# =========================


def print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]

    print("\n================ SUMMARY ================")
    print(f"num_cases: {s['num_cases']}")
    print(f"top1_accuracy: {s['top1_accuracy']:.4f}")
    print(f"mean_target_rank: {s['mean_target_rank']:.4f}")
    print(f"mean_target_margin_vs_top1: {s['mean_target_margin_vs_top1']:.6f}")

    print("\n[last hidden cosine offdiag]")
    print(json.dumps(s["hidden_cosine_offdiag"], ensure_ascii=False, indent=2))

    print("\n[last logits cosine offdiag]")
    print(json.dumps(s["logits_cosine_offdiag"], ensure_ascii=False, indent=2))

    print("\n[wrong top1 tokens]")
    for item in s["wrong_top1_tokens"]:
        print(
            f"count={item['count']:>4} | id={item['token_id']:>6} "
            f"| piece={item['piece']!r} | text={pretty_token(item['text'])!r}"
        )

    print("\n[wrong topk tokens]")
    for item in s["wrong_topk_tokens"]:
        print(
            f"count={item['count']:>4} | id={item['token_id']:>6} "
            f"| piece={item['piece']!r} | text={pretty_token(item['text'])!r}"
        )

    print("\n[probe summary]")
    for tok, item in s["probe_summary"].items():
        print(
            f"{tok!r} | id={item['token_id']:>6} | piece={item['piece']!r} "
            f"| avg_logit={item['avg_logit']:.6f} | avg_cosine={item['avg_cosine']:.6f} "
            f"| avg_rank={item['avg_rank']:.4f} | top1_count={item['top1_count']}"
        )

    if "tied_compare" in s:
        print("\n[tied compare]")
        print(json.dumps(s["tied_compare"], ensure_ascii=False, indent=2))


def print_case_details(report: dict[str, Any], max_cases: int) -> None:
    print("\n================ CASE DETAILS ================")
    for sample in report["samples"][:max_cases]:
        print(f"\n--- case {sample['index']} ---")
        print(f"prompt: {sample['prompt']}")
        print(f"target_text(raw): {sample['target_text']!r}")
        print(
            f"prompt_last_token: piece={sample['prompt_last_token_piece']!r} "
            f"text={pretty_token(sample['prompt_last_token_text'])!r}"
        )
        print(
            f"target_token: id={sample['target_token_id']} "
            f"piece={sample['target_piece']!r} text={pretty_token(sample['target_text_decoded'])!r}"
        )
        print(
            f"top1: id={sample['top1_token_id']} piece={sample['top1_piece']!r} "
            f"text={pretty_token(sample['top1_text'])!r} correct={sample['top1_correct']}"
        )
        print(f"target_rank: {sample['target_rank']}")
        print(f"target_margin_vs_top1: {sample['target_margin_vs_top1']:.6f}")

        print("[target decomp]")
        td = sample["target_decomp"]
        print(
            f"logit={td['logit']:.6f} prob={td['prob']:.6e} "
            f"weight_norm={td['weight_norm']:.6f} cosine={td['cosine']:.6f} "
            f"dot={td['dot']:.6f} rank={td['rank']}"
        )

        print("[top1 decomp]")
        td = sample["top1_decomp"]
        print(
            f"logit={td['logit']:.6f} prob={td['prob']:.6e} "
            f"weight_norm={td['weight_norm']:.6f} cosine={td['cosine']:.6f} "
            f"dot={td['dot']:.6f} rank={td['rank']}"
        )

        print("[topk]")
        for item in sample["topk"]:
            print(
                f"rank={item['rank']:>5} | id={item['token_id']:>6} "
                f"| piece={item['piece']!r} | text={pretty_token(item['text'])!r} "
                f"| logit={item['logit']:.6f} | prob={item['prob']:.6e} "
                f"| w_norm={item['weight_norm']:.6f} | cos={item['cosine']:.6f}"
            )

        if sample.get("tied_target_rank") is not None:
            print("[tied readout compare]")
            print(
                f"tied_top1_text={pretty_token(sample['tied_top1_text'])!r} "
                f"| tied_target_rank={sample['tied_target_rank']} "
                f"| tied_target_margin_vs_top1={sample['tied_target_margin_vs_top1']:.6f}"
            )


# =========================
# CLI
# =========================


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose readout collapse for decoder-only LM.")
    parser.add_argument("--model_path", type=str, required=True, help="Directory containing tokenizer + config.json")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model.safetensors. Default: <model_path>/model.safetensors",
    )
    parser.add_argument("--cases_path", type=str, required=True, help="JSONL file with fields: prompt, target")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    parser.add_argument("--topk", type=int, default=10, help="Top-k tokens to print per sample")
    parser.add_argument(
        "--probe_tokens",
        type=str,
        default="的,了,是,在,，,。",
        help="Comma-separated probe tokens",
    )
    parser.add_argument("--compare_tied", action="store_true", help="Also compare temporary tied readout")
    parser.add_argument("--max_print_cases", type=int, default=20, help="How many cases to print in detail")
    parser.add_argument(
        "--save_path",
        type=str,
        default="readout_diagnosis_report.json",
        help="Where to save the full JSON report",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    tokenizer, config, model = load_model_and_tokenizer(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    cases = load_cases(args.cases_path)
    probe_tokens = [x.strip() for x in args.probe_tokens.split(",") if x.strip()]

    print("loaded config:")
    print(json.dumps(config.model_dump(), ensure_ascii=False, indent=2))
    print(f"num_cases={len(cases)}")
    print(f"probe_tokens={probe_tokens}")
    print(f"compare_tied={args.compare_tied}")

    report = analyze_cases(
        model=model,
        tokenizer=tokenizer,
        cases=cases,
        device=device,
        topk=args.topk,
        probe_tokens=probe_tokens,
        compare_tied=args.compare_tied,
    )

    save_path = Path(args.save_path)
    save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(report)
    print_case_details(report, max_cases=args.max_print_cases)

    print(f"\nfull report saved to: {save_path}")


if __name__ == "__main__":
    main()
