#!/usr/bin/env python
"""Quick baseline evaluation of LocateAnything-3B on DOTA-v1.0 val tiles.

Samples N tiles, runs detection, parses boxes, and computes per-class
precision / recall / F1 at IoU=0.5.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

DOTA_V1_CLASSES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field",
    "roundabout", "harbor", "swimming-pool", "helicopter",
]

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
NONE_RE = re.compile(r"<box>[Nn]one</box>", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True, help="Output root from converter, e.g. data/dota_v1_hbb_512")
    p.add_argument("--annotation", required=True, help="JSONL file relative to data-root")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--generation-mode", default="hybrid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-file", default=None)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_answer(answer: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes = []
    for m in BOX_RE.finditer(answer):
        label = m.group(1).strip()
        x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        if x2 > x1 and y2 > y1:
            boxes.append((label, (x1, y1, x2, y2)))
    return boxes


def extract_gt(sample: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
    gpt_text = sample["conversations"][1]["value"]
    return parse_answer(gpt_text)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    preds: list[tuple[str, tuple]],
    gts: list[tuple[str, tuple]],
    iou_thr: float,
):
    """Greedy match by IoU, returns TP/FP/FN counts per class."""
    by_class_gt = defaultdict(list)
    by_class_pred = defaultdict(list)
    for i, (label, box) in enumerate(gts):
        by_class_gt[label].append((i, box))
    for i, (label, box) in enumerate(preds):
        by_class_pred[label].append((i, box))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    all_labels = set(by_class_gt.keys()) | set(by_class_pred.keys())
    for label in all_labels:
        gt_items = by_class_gt.get(label, [])
        pred_items = by_class_pred.get(label, [])
        matched_gt = set()
        matched_pred = set()

        scored = []
        for pi, (pidx, pbox) in enumerate(pred_items):
            for gi, (gidx, gbox) in enumerate(gt_items):
                sc = iou(pbox, gbox)
                if sc >= iou_thr:
                    scored.append((sc, pi, gi))
        scored.sort(reverse=True)

        for sc, pi, gi in scored:
            if pi not in matched_pred and gi not in matched_gt:
                matched_pred.add(pi)
                matched_gt.add(gi)
                tp[label] += 1

        fp[label] = len(pred_items) - len(matched_pred)
        fn[label] = len(gt_items) - len(matched_gt)

    return tp, fp, fn


def main():
    args = parse_args()
    random.seed(args.seed)

    ann_path = Path(args.data_root) / args.annotation
    samples = load_jsonl(ann_path)
    if args.num_samples > 0 and len(samples) > args.num_samples:
        samples = random.sample(samples, args.num_samples)
    print(f"Loaded {len(samples)} samples from {ann_path}")

    # Lazy import torch (heavy)
    import torch
    from locateanything_worker import LocateAnythingWorker

    print(f"Loading model from {args.model_path} ...")
    worker = LocateAnythingWorker(
        args.model_path,
        device="cuda",
        dtype=torch.bfloat16,
        attn="sdpa",
    )
    print("Model loaded.")

    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_preds = 0
    total_gts = 0
    none_count = 0
    errors = 0

    for idx, sample in enumerate(samples):
        image_rel = sample["image"]
        image_path = Path(args.data_root) / image_rel
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            errors += 1
            continue

        gt_boxes = extract_gt(sample)
        total_gts += len(gt_boxes)

        try:
            result = worker.detect(
                image,
                DOTA_V1_CLASSES,
                generation_mode=args.generation_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
            answer = result.get("answer", "")
        except Exception as e:
            print(f"  [ERROR] tile {idx}: {e}")
            errors += 1
            continue

        if NONE_RE.search(answer):
            none_count += 1

        pred_boxes = parse_answer(answer)
        total_preds += len(pred_boxes)

        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, args.iou_threshold)
        for k in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
            total_tp[k] += tp[k]
            total_fp[k] += fp[k]
            total_fn[k] += fn[k]

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(samples)}] preds={total_preds} gts={total_gts} none={none_count}")

    # Report
    print("\n" + "=" * 80)
    print(f"Baseline Evaluation: {args.model_path}")
    print(f"Samples: {len(samples)} | IoU threshold: {args.iou_threshold}")
    print(f"Total GT boxes: {total_gts} | Total predictions: {total_preds}")
    print(f"No-detection (<box>none</box>): {none_count} | Errors: {errors}")
    print("=" * 80)
    print(f"{'Class':<25} {'GT':>6} {'Pred':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 80)

    all_classes = sorted(set(list(total_tp.keys()) + list(total_fp.keys()) + list(total_fn.keys())))
    sum_tp = sum_fp = sum_fn = 0
    for cls in all_classes:
        tp = total_tp[cls]
        fp = total_fp[cls]
        fn = total_fn[cls]
        gt_count = tp + fn
        pred_count = tp + fp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{cls:<25} {gt_count:>6} {pred_count:>6} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    print("-" * 80)
    mp = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    mr = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    print(f"{'OVERALL':<25} {sum_tp+sum_fn:>6} {sum_tp+sum_fp:>6} {sum_tp:>5} {sum_fp:>5} {sum_fn:>5} {mp:>7.3f} {mr:>7.3f} {mf1:>7.3f}")
    print("=" * 80)

    report = {
        "model_path": args.model_path,
        "num_samples": len(samples),
        "iou_threshold": args.iou_threshold,
        "total_gt": total_gts,
        "total_preds": total_preds,
        "none_count": none_count,
        "overall": {"precision": mp, "recall": mr, "f1": mf1, "tp": sum_tp, "fp": sum_fp, "fn": sum_fn},
        "per_class": {cls: {"tp": total_tp[cls], "fp": total_fp[cls], "fn": total_fn[cls]} for cls in all_classes},
    }
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output_file}")


if __name__ == "__main__":
    main()
