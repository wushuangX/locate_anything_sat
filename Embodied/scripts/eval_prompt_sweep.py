#!/usr/bin/env python
"""Sweep detect prompt templates on the same DOTA val tiles."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

import importlib.util

_eval_path = Path(__file__).resolve().parent / "eval_baseline_dota.py"
_spec = importlib.util.spec_from_file_location("eval_baseline_dota", _eval_path)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)
DOTA_V1_CLASSES = _eval.DOTA_V1_CLASSES
NONE_RE = _eval.NONE_RE
extract_gt = _eval.extract_gt
load_jsonl = _eval.load_jsonl
match_boxes = _eval.match_boxes
parse_answer = _eval.parse_answer
TEMPLATES = [
    (
        "A",
        "Locate all the instances that matches the following description: {class}.",
        "train/detect control",
    ),
    (
        "B",
        "Locate all the instances that match the following description: {class}.",
        "grammar match",
    ),
    (
        "C",
        "Locate a single instance that matches the following description: {class}.",
        "T5/ground_single",
    ),
    (
        "D",
        "Locate all the {class}.",
        "short Locate",
    ),
    (
        "E",
        "Detect all {class}.",
        "Detect verb",
    ),
    (
        "F",
        "{class}",
        "class name only",
    ),
    (
        "G",
        "Find every object of category {class} in this aerial image.",
        "far natural language",
    ),
    (
        "H",
        "Locate all the instances that matches the following description: {class}</c>{class}.",
        "duplicated class",
    ),
]


def summarize(tp, fp, fn) -> dict:
    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())
    sum_fn = sum(fn.values())
    p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": sum_tp,
        "fp": sum_fp,
        "fn": sum_fn,
        "pred": sum_tp + sum_fp,
        "gt": sum_tp + sum_fn,
    }


def eval_template(worker, samples, template, data_root, max_new_tokens, generation_mode, iou_thr):
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_preds = 0
    total_gts = 0
    none_count = 0
    errors = 0
    for idx, sample in enumerate(samples):
        image = Image.open(Path(data_root) / sample["image"]).convert("RGB")
        gt_boxes = extract_gt(sample)
        total_gts += len(gt_boxes)
        all_pred_boxes = []
        for cls in DOTA_V1_CLASSES:
            prompt = template.replace("{class}", cls)
            try:
                result = worker.predict(
                    image,
                    prompt,
                    generation_mode=generation_mode,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                answer = result.get("answer", "")
            except Exception as exc:
                print(f"  [ERROR] tile {idx} class {cls}: {exc}", flush=True)
                errors += 1
                continue
            if NONE_RE.search(answer):
                none_count += 1
            all_pred_boxes.extend(parse_answer(answer))
        total_preds += len(all_pred_boxes)
        tp, fp, fn = match_boxes(all_pred_boxes, gt_boxes, iou_thr)
        for k in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
            total_tp[k] += tp[k]
            total_fp[k] += fp[k]
            total_fn[k] += fn[k]
        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            print(
                f"    [{idx + 1}/{len(samples)}] preds={total_preds} gts={total_gts} none={none_count}",
                flush=True,
            )
    overall = summarize(total_tp, total_fp, total_fn)
    overall["none_count"] = none_count
    overall["errors"] = errors
    overall["total_preds"] = total_preds
    overall["total_gts"] = total_gts
    overall["per_class"] = {
        cls: {"tp": total_tp[cls], "fp": total_fp[cls], "fn": total_fn[cls]}
        for cls in sorted(set(list(total_tp) + list(total_fp) + list(total_fn)))
    }
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-mode", default="hybrid")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = load_jsonl(Path(args.data_root) / args.annotation)
    if args.num_samples > 0 and len(samples) > args.num_samples:
        samples = random.sample(samples, args.num_samples)
    print(f"Loaded {len(samples)} samples", flush=True)

    import torch
    from locateanything_worker import LocateAnythingWorker

    print(f"Loading {args.model_path}", flush=True)
    worker = LocateAnythingWorker(
        args.model_path,
        device="cuda",
        dtype=torch.bfloat16,
        attn="sdpa",
    )
    print("Model loaded.", flush=True)

    results = []
    for tag, template, note in TEMPLATES:
        print(f"\n=== {tag}: {note} ===\n{template}", flush=True)
        overall = eval_template(
            worker,
            samples,
            template,
            args.data_root,
            args.max_new_tokens,
            args.generation_mode,
            args.iou_threshold,
        )
        row = {"id": tag, "note": note, "template": template, **overall}
        results.append(row)
        print(
            f"  {tag} P={overall['precision']:.3f} R={overall['recall']:.3f} "
            f"F1={overall['f1']:.3f} pred={overall['total_preds']} none={overall['none_count']}",
            flush=True,
        )

    out = {
        "model_path": args.model_path,
        "num_samples": len(samples),
        "iou_threshold": args.iou_threshold,
        "prompts": results,
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("\n" + "=" * 72)
    print(f"{'id':<3} {'P':>6} {'R':>6} {'F1':>6} {'pred':>6} {'none':>6}  note")
    print("-" * 72)
    for row in results:
        print(
            f"{row['id']:<3} {row['precision']:6.3f} {row['recall']:6.3f} {row['f1']:6.3f} "
            f"{row['total_preds']:6d} {row['none_count']:6d}  {row['note']}"
        )
    print("=" * 72)
    print(f"Report saved to {args.output_file}")


if __name__ == "__main__":
    main()
