#!/usr/bin/env python
"""COCO-style AP on DOTA LocateAnything OBB JSONL with rotated IoU.

Uses worker.detect_oriented and extra format/angle diagnostics.
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

from PIL import Image
from eaglevl.train.obb_geometry import dequantize_obb, rotated_iou


DOTA_V1_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]
OBB_RE = re.compile(r"<ref>(.*?)</ref><obb><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)></obb>")
NONE_RE = re.compile(r"<obb>[Nn]one</obb>", re.IGNORECASE)
LEGAL_BLOCK_RE = re.compile(
    r"(?:<ref>.*?</ref><obb><\d+><\d+><\d+><\d+><\d+></obb>|<obb>[Nn]one</obb>)",
    re.IGNORECASE,
)
IOU_THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50:0.05:0.95
AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--num-samples", type=int, default=0, help="0 = full split")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--generation-mode", default="hybrid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-file", required=True)
    p.add_argument("--pred-file", default=None, help="Reuse dumped detections JSON instead of running the model")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_answer(answer: str) -> list[tuple[str, tuple[int, int, int, int, int]]]:
    boxes = []
    for m in OBB_RE.finditer(answer or ""):
        boxes.append(
            (
                m.group(1),
                (
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                    int(m.group(6)),
                ),
            )
        )
    return boxes


def extract_gt(sample: dict) -> list[tuple[str, tuple[int, int, int, int, int]]]:
    return parse_answer(sample["conversations"][1]["value"])


def tok_area_px(box: tuple[int, int, int, int, int], width: int, height: int) -> float:
    _cx, _cy, w_q, h_q, _th = box
    return (w_q / 1000.0 * width) * (h_q / 1000.0 * height)


def to_pixel_obb(box: tuple[int, int, int, int, int], width: int, height: int):
    tile_size = max(int(width), int(height), 1)
    return dequantize_obb(box[0], box[1], box[2], box[3], box[4], tile_size, width, height)


def is_legal_obb_answer(answer: str) -> bool:
    stripped = LEGAL_BLOCK_RE.sub("", answer or "")
    return stripped.strip() == ""


def angle_abs_err_deg(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def iou(a, b) -> float:
    return rotated_iou(a, b)


def voc_ap(recalls: list[float], precisions: list[float]) -> float:
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def coco_ap_101(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    acc = 0.0
    for r in [i / 100.0 for i in range(101)]:
        p = 0.0
        for rec, prec in zip(recalls, precisions):
            if rec >= r:
                p = max(p, prec)
        acc += p
    return acc / 101.0


def evaluate_class(
    dets: list[tuple[int, tuple, float]],
    gts_by_image: dict[int, list[tuple]],
    iou_thr: float,
    angle_errs: list[float] | None = None,
) -> dict:
    npos = sum(len(v) for v in gts_by_image.values())
    if npos == 0:
        return {"ap_voc": float("nan"), "ap_coco": float("nan"), "npos": 0, "ndet": len(dets)}
    matched = {img: [False] * len(boxes) for img, boxes in gts_by_image.items()}
    tp = []
    fp = []
    for img_id, box, _score in dets:
        gt_boxes = gts_by_image.get(img_id, [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched[img_id][j]:
                continue
            ov = iou(box, gt)
            if ov > best_iou:
                best_iou = ov
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            matched[img_id][best_j] = True
            tp.append(1)
            fp.append(0)
            if angle_errs is not None:
                angle_errs.append(angle_abs_err_deg(float(box[4]), float(gt_boxes[best_j][4])))
        else:
            tp.append(0)
            fp.append(1)
    cum_tp = 0
    cum_fp = 0
    recalls = []
    precisions = []
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        recalls.append(cum_tp / npos)
        precisions.append(cum_tp / (cum_tp + cum_fp))
    return {
        "ap_voc": voc_ap(recalls, precisions),
        "ap_coco": coco_ap_101(recalls, precisions),
        "npos": npos,
        "ndet": len(dets),
        "recall": recalls[-1] if recalls else 0.0,
        "precision": precisions[-1] if precisions else 0.0,
    }


def mean_finite(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


def run_inference(args, samples: list[dict]) -> list[dict]:
    import torch
    from locateanything_worker import LocateAnythingWorker

    print(f"Loading model from {args.model_path} ...", flush=True)
    worker = LocateAnythingWorker(
        args.model_path,
        device="cuda",
        dtype=torch.bfloat16,
        attn="sdpa",
    )
    print("Model loaded.", flush=True)
    records = []
    none_count = 0
    for idx, sample in enumerate(samples):
        image_rel = sample["image"]
        image_path = Path(args.data_root) / image_rel
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        gt_boxes = extract_gt(sample)
        preds: list[dict] = []
        n_answers = 0
        n_legal = 0
        for cls in DOTA_V1_CLASSES:
            result = worker.detect_oriented(
                image,
                [cls],
                generation_mode=args.generation_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
            answer = result.get("answer", "")
            n_answers += 1
            if is_legal_obb_answer(answer):
                n_legal += 1
            if NONE_RE.search(answer):
                none_count += 1
            for rank, (label, box) in enumerate(parse_answer(answer)):
                preds.append(
                    {
                        "label": label,
                        "box": list(box),
                        "score": 1.0,
                        "rank": rank,
                        "prompt_class": cls,
                    }
                )
        records.append(
            {
                "image": image_rel,
                "width": width,
                "height": height,
                "gt": [{"label": lab, "box": list(box)} for lab, box in gt_boxes],
                "pred": preds,
                "n_answers": n_answers,
                "n_legal": n_legal,
            }
        )
        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            npred = sum(len(r["pred"]) for r in records)
            ngt = sum(len(r["gt"]) for r in records)
            print(f"  [{idx + 1}/{len(samples)}] preds={npred} gts={ngt} none={none_count}", flush=True)
    return records


def compute_metrics(records: list[dict]) -> dict:
    gts = []
    dets = []
    n_answers = 0
    n_legal = 0
    for img_id, rec in enumerate(records):
        w, h = rec["width"], rec["height"]
        n_answers += int(rec.get("n_answers", 0))
        n_legal += int(rec.get("n_legal", 0))
        for gt in rec["gt"]:
            box = tuple(gt["box"])
            gts.append(
                {
                    "img_id": img_id,
                    "label": gt["label"],
                    "box": to_pixel_obb(box, w, h),
                    "area": tok_area_px(box, w, h),
                }
            )
        for pred in rec["pred"]:
            box = tuple(pred["box"])
            dets.append(
                {
                    "img_id": img_id,
                    "label": pred["label"],
                    "box": to_pixel_obb(box, w, h),
                    "score": float(pred.get("score", 1.0)),
                    "rank": int(pred.get("rank", 0)),
                    "area": tok_area_px(box, w, h),
                }
            )

    def select(area_name: str, label: str | None):
        lo, hi = AREA_RANGES[area_name]
        gt_sel = [g for g in gts if lo <= g["area"] < hi and (label is None or g["label"] == label)]
        det_sel = [d for d in dets if lo <= d["area"] < hi and (label is None or d["label"] == label)]
        det_sel.sort(key=lambda d: (-d["score"], d["img_id"], d["rank"]))
        gts_by_image: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for g in gt_sel:
            gts_by_image[g["img_id"]].append(g["box"])
        packed = [(d["img_id"], d["box"], d["score"]) for d in det_sel]
        return packed, dict(gts_by_image)

    per_class_ap50 = {}
    angle_errs: list[float] = []
    for cls in DOTA_V1_CLASSES:
        packed, gts_by_image = select("all", cls)
        per_class_ap50[cls] = evaluate_class(packed, gts_by_image, 0.5, angle_errs)

    coco = {}
    for area_name in AREA_RANGES:
        aps_voc = []
        aps_coco = []
        ap50 = []
        ap75 = []
        for cls in DOTA_V1_CLASSES:
            packed, gts_by_image = select(area_name, cls)
            npos = sum(len(v) for v in gts_by_image.values())
            if npos == 0:
                continue
            row = {}
            for thr in IOU_THRS:
                row[thr] = evaluate_class(packed, gts_by_image, thr)
            aps_voc.append(mean_finite([row[t]["ap_voc"] for t in IOU_THRS]))
            aps_coco.append(mean_finite([row[t]["ap_coco"] for t in IOU_THRS]))
            ap50.append(row[0.5]["ap_coco"])
            ap75.append(row[0.75]["ap_coco"])
        coco[area_name] = {
            "AP": mean_finite(aps_coco),
            "AP_voc": mean_finite(aps_voc),
            "AP50": mean_finite(ap50),
            "AP75": mean_finite(ap75),
            "n_classes_with_gt": len(aps_coco),
        }

    return {
        "score_note": "all detection scores are 1.0; ranking is model emission order",
        "num_images": len(records),
        "num_gt": len(gts),
        "num_pred": len(dets),
        "format_legal_rate": (n_legal / n_answers) if n_answers else float("nan"),
        "mean_abs_angle_err_deg": mean_finite(angle_errs),
        "coco": coco,
        "per_class_AP50": {
            cls: {
                "AP50": row["ap_coco"],
                "AP50_voc": row["ap_voc"],
                "npos": row["npos"],
                "ndet": row["ndet"],
                "P": row.get("precision"),
                "R": row.get("recall"),
            }
            for cls, row in per_class_ap50.items()
        },
    }


def print_report(metrics: dict) -> None:
    print("DOTA OBB tile COCO/VOC AP  (rotated IoU, scores=1.0, emission order)")
    print(f"images={metrics['num_images']}  gt={metrics['num_gt']}  pred={metrics['num_pred']}")
    print(f"format_legal_rate={metrics.get('format_legal_rate')}  mean_abs_angle_err_deg={metrics.get('mean_abs_angle_err_deg')}")
    print("=" * 72)
    print(f"{'area':<8} {'AP':>8} {'AP50':>8} {'AP75':>8} {'n_cls':>6}")
    print("-" * 72)
    for area_name, row in metrics["coco"].items():
        print(
            f"{area_name:<8} {row['AP']:8.3f} {row['AP50']:8.3f} {row['AP75']:8.3f} {row['n_classes_with_gt']:6d}"
        )
    print("-" * 72)
    print(f"{'class':<22} {'GT':>5} {'Det':>5} {'AP50':>8} {'P':>7} {'R':>7}")
    print("-" * 72)
    for cls, row in metrics["per_class_AP50"].items():
        p = row["P"] if row["P"] is not None else float("nan")
        r = row["R"] if row["R"] is not None else float("nan")
        print(f"{cls:<22} {row['npos']:5d} {row['ndet']:5d} {row['AP50']:8.3f} {p:7.3f} {r:7.3f}")
    print("=" * 72)


def main():
    args = parse_args()
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.pred_file) if args.pred_file else out_path.with_suffix(".preds.json")

    if args.pred_file:
        records = json.loads(Path(args.pred_file).read_text())
        print(f"Loaded {len(records)} dumped images from {args.pred_file}")
    else:
        random.seed(args.seed)
        samples = load_jsonl(Path(args.data_root) / args.annotation)
        if args.num_samples > 0 and len(samples) > args.num_samples:
            samples = random.sample(samples, args.num_samples)
        print(f"Loaded {len(samples)} samples from {args.annotation}")
        records = run_inference(args, samples)
        pred_path.write_text(json.dumps(records))
        print(f"Wrote detections to {pred_path}")

    metrics = compute_metrics(records)
    metrics["model_path"] = args.model_path
    metrics["annotation"] = args.annotation
    metrics["pred_file"] = str(pred_path)
    print_report(metrics)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
