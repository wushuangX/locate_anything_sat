#!/usr/bin/env python
"""Small-dense DOTA tile eval v2 — canonical unique-image protocol.

test_t1 的 517 行是 annotation chunk（371 unique images）；本脚本先用
merge_t1_chunks() 把同图多 chunk 合并为 canonical record（精确去重 (label, box)），
再在合并后过滤 smallGT>=min_small_gt，保证 GT 完整、图像唯一、顺序固定。

验收指标 fixed_iou_05（IoU=0.5 一对一贪心，与 reward.match_at_iou 同构）：
  all   : micro TP/FP/FN + P/R/F1（prompt_class 记 FP；emitted≠prompt 的框计
          label_mismatch_count，绝不记 TP）
  small : 匹配到面积 <32^2 GT 的 TP / 未匹配 small GT 的 FN / 未匹配且预测面积
          <32^2 的 FP（匹配到非 small GT 的预测不计 small FP）
  per_class + frequent_macro_f1（npos>=50）+ npos 加权 F1
score 恒为 1.0：tied-score AP 对输入顺序敏感，只作诊断，不参与 go/no-go。

验收门（compare_acceptance，仅 hybrid 是部署协议）：
  中间 checkpoint：small recall 或 overall micro F1 比 baseline 低 >0.01 →
  status=regression，exit 2。
  --final-check：small recall >= base+0.005、small F1 >= base、overall F1 >=
  base−0.005、平均预测框数 >= 0.9×baseline → pass exit 0 / fail exit 1。
slow 模式只诊断 decoder mismatch，不单独放行。
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--num-samples", type=int, default=0,
                   help="0 = 全部 canonical dense 图像（排序固定，不抽样）")
    p.add_argument("--min-small-gt", type=int, default=10)
    p.add_argument("--seeds", default="42",
                   help="抽样模式用；canonical(num-samples 0) 忽略 seed")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--generation-mode", default="hybrid", choices=["hybrid", "slow", "fast"])
    p.add_argument("--class-scope", default="all", choices=["all", "present"],
                   help="all=遍历 15 类（最终协议）；present=只问 GT 内类（快速 smoke）")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", required=True, help="run tag, e.g. geom100k / step250")
    p.add_argument("--baseline-metrics", default=None,
                   help="baseline 的 <stem>.json；提供时执行 compare_acceptance")
    p.add_argument("--final-check", action="store_true",
                   help="用最终阈值（而非中间 regression 阈值）判定")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# canonical merge
# --------------------------------------------------------------------------- #

def merge_t1_chunks(samples: list[dict]) -> list[dict]:
    """按 image 合并 test_t1 chunk 行；精确去重 (label, box)；排序 canonical。

    必须在合并后再计算 smallGT>=min_small_gt（否则 chunk 切分会低估每图 GT）。
    返回按 image/label/坐标排序的 canonical records。
    """
    by_image: dict[str, dict] = {}
    for s in samples:
        image = s["image"]
        rec = by_image.setdefault(image, {"image": image, "boxes": set(), "n_chunks": 0})
        rec["n_chunks"] += 1
        for label, box in emap.extract_gt(s):
            rec["boxes"].add((label, tuple(int(c) for c in box)))
    canonical = []
    for image in sorted(by_image):
        rec = by_image[image]
        gt_boxes = sorted(rec["boxes"], key=lambda lb: (lb[0], lb[1][0], lb[1][1], lb[1][2], lb[1][3]))
        canonical.append(
            {"image": image, "gt_boxes": gt_boxes,
             "n_chunks": rec["n_chunks"], "n_rows_merged": rec["n_chunks"] - 1}
        )
    return canonical


def canonical_small_gt(rec: dict, tile: int = 448) -> int:
    return sum(1 for _lab, box in rec["gt_boxes"]
               if emap.tok_area_px(box, tile, tile) < SMALL_AREA)


# --------------------------------------------------------------------------- #
# fixed-IoU metrics（验收）
# --------------------------------------------------------------------------- #

def _box_area_px(box, tile: int = 448) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) / 1000.0 * tile) * max(0.0, (y2 - y1) / 1000.0 * tile)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_fixed_iou_metrics(records: list[dict], iou_thr: float = 0.5) -> dict:
    """固定 IoU 一对一指标：all / small / per_class；label mismatch 单列。"""
    all_tp = all_fp = all_fn = 0
    small_tp = small_fp = small_fn = 0
    label_mismatch_count = 0
    per_class: dict[str, dict] = defaultdict(lambda: {"npos": 0, "tp": 0, "fp": 0, "fn": 0})

    for rec in records:
        gts = [(lab, tuple(box)) for lab, box in rec["gt_boxes"]]
        preds = [(p["prompt_class"], p["label"], tuple(p["box"])) for p in rec["pred"]]

        scored = []
        for pi, (pcls, elabel, pbox) in enumerate(preds):
            for gi, (glabel, gbox) in enumerate(gts):
                # 只有 prompt_class 与 emitted_label 相等时才允许匹配对应 GT
                if pcls != elabel or elabel != glabel:
                    continue
                sc = _iou(pbox, gbox)
                if sc >= iou_thr:
                    scored.append((sc, pi, gi))
        scored.sort(reverse=True)
        matched_pred, matched_gt = set(), set()
        for sc, pi, gi in scored:
            if pi not in matched_pred and gi not in matched_gt:
                matched_pred.add(pi)
                matched_gt.add(gi)

        for gi, (glabel, gbox) in enumerate(gts):
            per_class[glabel]["npos"] += 1
            g_small = _box_area_px(gbox) < SMALL_AREA
            if gi in matched_gt:
                all_tp += 1
                per_class[glabel]["tp"] += 1
                if g_small:
                    small_tp += 1
            else:
                all_fn += 1
                per_class[glabel]["fn"] += 1
                if g_small:
                    small_fn += 1
        for pi, (pcls, elabel, pbox) in enumerate(preds):
            p_small = _box_area_px(pbox) < SMALL_AREA
            if pi in matched_pred:
                continue
            all_fp += 1
            per_class[pcls]["fp"] += 1
            if elabel != pcls:
                label_mismatch_count += 1
            # 匹配到非 small GT 的预测不计 small FP（matched 已跳过）；未匹配小框计 FP
            if p_small:
                small_fp += 1

    metrics = {
        "all": _prf(all_tp, all_fp, all_fn),
        "small": _prf(small_tp, small_fp, small_fn),
        "per_class": {},
        "label_mismatch_count": label_mismatch_count,
        "iou_thr": iou_thr,
        "note": "all scores=1, AP order-sensitive, not acceptance metric",
    }
    for cls, c in sorted(per_class.items()):
        c = dict(c)
        c["f1"] = _prf(c["tp"], c["fp"], c["fn"])["f1"]
        metrics["per_class"][cls] = c
    frequent = [c["f1"] for c in metrics["per_class"].values() if c["npos"] >= 50 and c["f1"] == c["f1"]]
    metrics["frequent_macro_f1"] = statistics.fmean(frequent) if frequent else float("nan")
    num = den = 0.0
    for c in metrics["per_class"].values():
        if c["npos"] > 0 and c["f1"] == c["f1"]:
            num += c["npos"] * c["f1"]
            den += c["npos"]
    metrics["npos_weighted_f1"] = num / den if den else float("nan")
    return metrics


def mean_preds_per_image(records: list[dict]) -> float:
    return statistics.fmean(len(r["pred"]) for r in records) if records else float("nan")


# --------------------------------------------------------------------------- #
# acceptance（仅 hybrid 是部署验收协议）
# --------------------------------------------------------------------------- #

def compare_acceptance(candidate: dict, baseline: dict, final: bool) -> dict:
    cand = candidate["fixed_iou_05"]
    base = baseline["fixed_iou_05"]
    cand_small_recall = cand["small"]["recall"]
    cand_small_f1 = cand["small"]["f1"]
    cand_all_f1 = cand["all"]["f1"]
    base_small_recall = base["small"]["recall"]
    base_small_f1 = base["small"]["f1"]
    base_all_f1 = base["all"]["f1"]
    cand_npred = candidate["mean_preds_per_image"]
    base_npred = baseline["mean_preds_per_image"]
    out = {
        "final": bool(final),
        "candidate": {"small_recall": cand_small_recall, "small_f1": cand_small_f1,
                      "all_f1": cand_all_f1, "mean_preds_per_image": cand_npred},
        "baseline": {"small_recall": base_small_recall, "small_f1": base_small_f1,
                     "all_f1": base_all_f1, "mean_preds_per_image": base_npred},
    }
    if not final:
        regression = (cand_small_recall < base_small_recall - 0.01
                      or cand_all_f1 < base_all_f1 - 0.01)
        out["status"] = "regression" if regression else "ok"
        out["rule"] = "regression iff small_recall or all_f1 < baseline - 0.01"
        return out
    checks = {
        "small_recall_ge_base_plus_0.005": cand_small_recall >= base_small_recall + 0.005,
        "small_f1_ge_base": cand_small_f1 >= base_small_f1,
        "all_f1_ge_base_minus_0.005": cand_all_f1 >= base_all_f1 - 0.005,
        "mean_preds_ge_0.9x_base": cand_npred >= 0.9 * base_npred,
    }
    out["status"] = "pass" if all(checks.values()) else "fail"
    out["checks"] = checks
    return out


# --------------------------------------------------------------------------- #
# 生成 + 诊断（tied-score AP 只作历史对照）
# --------------------------------------------------------------------------- #

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


def run_seed(worker, args, samples: list[dict], seed: int) -> tuple[dict, list[dict]]:
    """samples: canonical records（已排序）；返回 (metrics, records)。"""
    data_root = Path(args.data_root)
    records = []
    none_count = 0
    for idx, rec in enumerate(samples):
        image_rel = rec["image"]
        image = Image.open(data_root / image_rel).convert("RGB")
        width, height = image.size
        gt_boxes = rec["gt_boxes"]
        if args.class_scope == "all":
            classes = list(emap.DOTA_V1_CLASSES)
        else:
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
    metrics = {
        "fixed_iou_05": compute_fixed_iou_metrics(records),
        "mean_preds_per_image": mean_preds_per_image(records),
        "fixed_small_vehicle": metrics_per_class(records),
    }
    # tied-score AP 与 legacy F1/cap 仅诊断
    legacy = emap.compute_metrics(records)
    legacy["f1"] = f1_overall(records)
    legacy["cap"] = cap_stats(records)
    metrics["diagnostics_ap_order_sensitive"] = legacy
    metrics["seed"] = seed
    metrics["images"] = [r["image"] for r in records]
    return metrics, records


def metrics_per_class(records: list[dict], cls: str = "small-vehicle") -> dict:
    tp = fp = fn = 0
    for rec in records:
        preds = [(p["prompt_class"], tuple(p["box"])) for p in rec["pred"]
                 if p["prompt_class"] == p["label"]]
        gts = [(g["label"], tuple(g["box"])) for g in rec["gt"]]
        t, f, n = ebase.match_boxes(preds, gts, 0.5)
        tp += t.get(cls, 0)
        fp += f.get(cls, 0)
        fn += n.get(cls, 0)
    return _prf(tp, fp, fn)


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
    canonical = merge_t1_chunks(all_samples)
    n_dup_rows = sum(r["n_rows_merged"] for r in canonical)
    pool = [r for r in canonical if canonical_small_gt(r) >= args.min_small_gt]
    print(
        f"[pool] {len(all_samples)} raw rows -> {len(canonical)} unique images "
        f"({n_dup_rows} merged chunk rows) -> {len(pool)} dense images with "
        f"smallGT>={args.min_small_gt} (computed AFTER merge)",
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

    canonical_mode = args.num_samples <= 0
    per_seed = []
    for seed in seeds:
        if canonical_mode:
            chosen = list(pool)  # 排序固定，不抽样
            print(f"[seed={seed}] canonical all {len(chosen)} images "
                  f"(order fixed by image)", flush=True)
        else:
            rng = random.Random(seed)
            k = min(args.num_samples, len(pool))
            chosen = sorted(rng.sample(pool, k), key=lambda r: r["image"])
            print(f"[seed={seed}] sampled {k} images (sorted after sampling)", flush=True)
        metrics, records = run_seed(worker, args, chosen, seed)
        metrics["protocol"] = {
            "min_small_gt": args.min_small_gt,
            "num_samples": args.num_samples,
            "canonical_mode": canonical_mode,
            "class_scope": args.class_scope,
            "unique_images": len(chosen),
            "raw_rows": len(all_samples),
            "canonical_images": len(canonical),
            "max_new_tokens": args.max_new_tokens,
            "generation_mode": args.generation_mode,
            "temperature": 0.0,
            "model_path": args.model_path,
            "annotation": args.annotation,
        }
        if canonical_mode:
            stem = f"{args.name}_dense_unique{len(pool)}_{args.generation_mode}"
        else:
            stem = f"{args.name}_smalldense{args.num_samples}_seed{seed}"
        (out_dir / f"{stem}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
        )
        (out_dir / f"{stem}.preds.json").write_text(json.dumps(records))
        fx = metrics["fixed_iou_05"]
        print(
            f"[seed={seed}] fixed05 all F1={fx['all']['f1']:.3f} "
            f"small R={fx['small']['recall']:.3f} small F1={fx['small']['f1']:.3f} "
            f"| diag AP50={legacy_ap50(metrics):.3f} smallAP50={legacy_small_ap50(metrics):.3f} "
            f"mean_preds={metrics['mean_preds_per_image']:.1f}",
            flush=True,
        )
        per_seed.append({"metrics": metrics, "stem": stem})

    # acceptance
    exit_code = 0
    if args.baseline_metrics:
        baseline = json.loads(Path(args.baseline_metrics).read_text())
        cand = per_seed[0]["metrics"]
        acc = compare_acceptance(cand, baseline, final=args.final_check)
        cand["acceptance"] = {
            "baseline_metrics": args.baseline_metrics,
            **acc,
        }
        stem = per_seed[0]["stem"]
        (out_dir / f"{stem}.json").write_text(
            json.dumps(cand, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"[acceptance] status={acc['status']} details={json.dumps(acc, ensure_ascii=False)}",
              flush=True)
        if not args.final_check and acc["status"] == "regression":
            exit_code = 2
        elif args.final_check:
            exit_code = 0 if acc["status"] == "pass" else 1

    if not canonical_mode:
        def grab(m, path):
            x = m
            for k in path:
                x = x[k]
            return float(x) if x is not None else float("nan")

        summary = {
            "protocol": per_seed[0]["metrics"]["protocol"] if per_seed else {},
            "seeds": seeds,
            "mean_std": {
                "fixed_all_f1": mean_std([grab(m["metrics"], ("fixed_iou_05", "all", "f1")) for m in per_seed]),
                "fixed_small_recall": mean_std([grab(m["metrics"], ("fixed_iou_05", "small", "recall")) for m in per_seed]),
                "fixed_small_f1": mean_std([grab(m["metrics"], ("fixed_iou_05", "small", "f1")) for m in per_seed]),
            },
            "per_seed_files": [m["stem"] + ".json" for m in per_seed],
        }
        sum_path = out_dir / f"{args.name}_smalldense{args.num_samples}_summary.json"
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print("SUMMARY", json.dumps(summary["mean_std"], indent=2), flush=True)
    return exit_code


def legacy_ap50(metrics: dict) -> float:
    try:
        return float(metrics["diagnostics_ap_order_sensitive"]["coco"]["all"]["AP50"])
    except Exception:
        return float("nan")


def legacy_small_ap50(metrics: dict) -> float:
    try:
        return float(metrics["diagnostics_ap_order_sensitive"]["coco"]["small"]["AP50"])
    except Exception:
        return float("nan")


if __name__ == "__main__":
    sys.exit(main())
