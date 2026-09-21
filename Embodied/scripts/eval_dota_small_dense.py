#!/usr/bin/env python
"""Small-dense DOTA tile eval (GT-class prompts only, long decode).

Filter test_t1 tiles with ≥min_small_gt COCO-small boxes, draw --num-samples
per seed, query only classes present in that tile's GT, max_new_tokens=2048.
Temperature is 0; seed only resamples which tiles. Not full val mAP.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_EMBODIED = _SCRIPTS.parent
sys.path.insert(0, str(_EMBODIED))
sys.path.insert(0, str(_SCRIPTS))

from PIL import Image  # noqa: E402

import eval_baseline_dota as ebase  # noqa: E402
import eval_dota_map as emap  # noqa: E402

SMALL_AREA = 32.0 ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--min-small-gt", type=int, default=10)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--generation-mode", default="hybrid")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", required=True, help="run tag, e.g. geom100k")
    return p.parse_args()


def n_small_gt(sample: dict, tile: int = 448) -> int:
    gts = emap.extract_gt(sample)
    return sum(1 for _, box in gts if emap.tok_area_px(box, tile, tile) < SMALL_AREA)


def f1_overall(records: list[dict], iou_thr: float = 0.5) -> dict:
    tp = fp = fn = 0
    for rec in records:
        preds = [(p["label"], tuple(p["box"])) for p in rec["pred"]]
        gts = [(g["label"], tuple(g["box"])) for g in rec["gt"]]
        t, f, n = ebase.match_boxes(preds, gts, iou_thr)
        tp += sum(t.values())
        fp += sum(f.values())
        fn += sum(n.values())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def cap_stats(records: list[dict]) -> dict:
    per_prompt = []
    rows = []
    for rec in records:
        w, h = rec["width"], rec["height"]
        n_gt = len(rec["gt"])
        n_pred = len(rec["pred"])
        n_small = sum(
            1 for g in rec["gt"] if emap.tok_area_px(tuple(g["box"]), w, h) < SMALL_AREA
        )
        n_sv_gt = sum(1 for g in rec["gt"] if g["label"] == "small-vehicle")
        n_sv_pred = sum(1 for p in rec["pred"] if p["label"] == "small-vehicle")
        by = defaultdict(int)
        for p in rec["pred"]:
            by[p.get("prompt_class", p["label"])] += 1
        mx = max(by.values()) if by else 0
        per_prompt.extend(by.values())
        rows.append(
            {
                "image": rec["image"],
                "n_gt": n_gt,
                "n_pred": n_pred,
                "n_small_gt": n_small,
                "n_sv_gt": n_sv_gt,
                "n_sv_pred": n_sv_pred,
                "max_one_prompt": mx,
                "n_classes_queried": rec.get("n_classes_queried", 0),
            }
        )
    return {
        "tiles_pred_ge20": sum(1 for r in rows if r["n_pred"] >= 20),
        "tiles_pred_ge30": sum(1 for r in rows if r["n_pred"] >= 30),
        "prompt_ge20": sum(1 for n in per_prompt if n >= 20),
        "prompt_ge30": sum(1 for n in per_prompt if n >= 30),
        "max_one_prompt": max(per_prompt) if per_prompt else 0,
        "mean_n_gt": statistics.fmean(r["n_gt"] for r in rows) if rows else 0.0,
        "mean_n_pred": statistics.fmean(r["n_pred"] for r in rows) if rows else 0.0,
        "mean_n_small_gt": statistics.fmean(r["n_small_gt"] for r in rows) if rows else 0.0,
        "per_tile": rows,
    }


def run_seed(worker, args, samples: list[dict], seed: int) -> dict:
    data_root = Path(args.data_root)
    records = []
    none_count = 0
    for idx, sample in enumerate(samples):
        image_rel = sample["image"]
        image = Image.open(data_root / image_rel).convert("RGB")
        width, height = image.size
        gt_boxes = emap.extract_gt(sample)
        classes = []
        seen = set()
        for lab, _ in gt_boxes:
            if lab in emap.DOTA_V1_CLASSES and lab not in seen:
                seen.add(lab)
                classes.append(lab)
        preds: list[dict] = []
        for cls in classes:
            result = worker.detect(
                image,
                [cls],
                generation_mode=args.generation_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
            answer = result.get("answer", "")
            if emap.NONE_RE.search(answer):
                none_count += 1
            for rank, (label, box) in enumerate(emap.parse_answer(answer)):
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
                "n_classes_queried": len(classes),
            }
        )
        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(
                f"  seed={seed} [{idx + 1}/{len(samples)}] "
                f"preds={sum(len(r['pred']) for r in records)} "
                f"gts={sum(len(r['gt']) for r in records)} none={none_count}",
                flush=True,
            )
    metrics = emap.compute_metrics(records)
    metrics["f1"] = f1_overall(records)
    metrics["cap"] = cap_stats(records)
    metrics["seed"] = seed
    metrics["images"] = [r["image"] for r in records]
    sv = metrics["per_class_AP50"].get("small-vehicle", {})
    metrics["small_vehicle"] = sv
    return metrics, records


def mean_std(vals: list[float]) -> dict:
    xs = [v for v in vals if v == v]
    if not xs:
        return {"mean": float("nan"), "std": float("nan")}
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0}
    return {"mean": statistics.fmean(xs), "std": statistics.pstdev(xs)}


def main() -> int:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = emap.load_jsonl(Path(args.data_root) / args.annotation)
    pool = [s for s in all_samples if n_small_gt(s) >= args.min_small_gt]
    print(
        f"[pool] {len(all_samples)} tiles -> {len(pool)} with smallGT>={args.min_small_gt}",
        flush=True,
    )
    if not pool:
        raise RuntimeError("empty small-dense pool")

    import torch
    from locateanything_worker import LocateAnythingWorker

    print(f"Loading model from {args.model_path} ...", flush=True)
    worker = LocateAnythingWorker(
        args.model_path, device="cuda", dtype=torch.bfloat16, attn="sdpa"
    )
    print("Model loaded.", flush=True)

    per_seed = []
    for seed in seeds:
        rng = random.Random(seed)
        k = min(args.num_samples, len(pool))
        chosen = rng.sample(pool, k)
        print(f"[seed={seed}] {k} tiles", flush=True)
        metrics, records = run_seed(worker, args, chosen, seed)
        stem = f"{args.name}_smalldense50_seed{seed}"
        (out_dir / f"{stem}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
        )
        (out_dir / f"{stem}.preds.json").write_text(json.dumps(records))
        coco = metrics["coco"]
        print(
            f"[seed={seed}] F1={metrics['f1']['f1']:.3f} "
            f"AP={coco['all']['AP']:.3f} AP50={coco['all']['AP50']:.3f} "
            f"smallAP50={coco['small']['AP50']:.3f} "
            f"SV_AP50={metrics['small_vehicle'].get('AP50')} "
            f"max_one_prompt={metrics['cap']['max_one_prompt']}",
            flush=True,
        )
        per_seed.append(metrics)

    def grab(path):
        cur = per_seed
        out = []
        for m in cur:
            x = m
            for k in path:
                x = x[k]
            out.append(float(x) if x is not None else float("nan"))
        return out

    summary = {
        "protocol": {
            "min_small_gt": args.min_small_gt,
            "num_samples": args.num_samples,
            "seeds": seeds,
            "gt_classes_only": True,
            "max_new_tokens": args.max_new_tokens,
            "generation_mode": args.generation_mode,
            "temperature": 0.0,
            "pool_size": len(pool),
            "model_path": args.model_path,
            "annotation": args.annotation,
        },
        "mean_std": {
            "f1": mean_std(grab(["f1", "f1"])),
            "AP": mean_std(grab(["coco", "all", "AP"])),
            "AP50": mean_std(grab(["coco", "all", "AP50"])),
            "small_AP": mean_std(grab(["coco", "small", "AP"])),
            "small_AP50": mean_std(grab(["coco", "small", "AP50"])),
            "SV_AP50": mean_std(grab(["small_vehicle", "AP50"])),
            "max_one_prompt": mean_std(
                [float(m["cap"]["max_one_prompt"]) for m in per_seed]
            ),
        },
        "per_seed_files": [
            f"{args.name}_smalldense50_seed{s}.json" for s in seeds
        ],
    }
    sum_path = out_dir / f"{args.name}_smalldense50_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("SUMMARY", json.dumps(summary["mean_std"], indent=2), flush=True)
    print(f"Wrote {sum_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
