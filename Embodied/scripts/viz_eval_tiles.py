#!/usr/bin/env python
"""Draw GT vs model predictions on DOTA val tiles."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
NONE_RE = re.compile(r"<box>[Nn]one</box>", re.IGNORECASE)

PALETTE = {
    "plane": (255, 80, 80),
    "small-vehicle": (80, 180, 255),
    "large-vehicle": (40, 90, 220),
    "ship": (80, 220, 160),
    "harbor": (20, 140, 90),
    "storage-tank": (255, 180, 40),
    "bridge": (200, 80, 200),
    "tennis-court": (180, 255, 80),
    "basketball-court": (255, 120, 40),
    "soccer-ball-field": (120, 255, 120),
    "ground-track-field": (255, 80, 160),
    "roundabout": (180, 180, 180),
    "baseball-diamond": (255, 220, 80),
    "swimming-pool": (80, 200, 255),
    "helicopter": (255, 40, 120),
}
GT_COLOR = (40, 220, 70)
PRED_COLOR = (255, 60, 60)
FONT = ImageFont.load_default()

DATA_ROOT = Path("/data/locate_anything_sat/Embodied/data/dota_v1_hbb_448_mix_v1")
ANN = DATA_ROOT / "annotations/DOTA-v1.0_test_t1_hbb_448.jsonl"
OUT = Path("/data/locate_anything_sat/Embodied/data/dota_v1_hbb_448_mix_v1/_eval_viz")
LDA_CKPT = "/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1/checkpoint-15000"
MIX_CKPT = "/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_mix_v1_lora_2gpu_4k_run1/checkpoint-5000"
RUN1_CKPT = "/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_lora_2gpu_run1/checkpoint-5000"


def parse_answer(answer: str):
    boxes = []
    for m in BOX_RE.finditer(answer or ""):
        boxes.append(
            (
                m.group(1),
                (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))),
            )
        )
    return boxes


def extract_gt(sample):
    return parse_answer(sample["conversations"][1]["value"])


def tok_to_px(box, w, h):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 / 1000 * w)),
        int(round(y1 / 1000 * h)),
        int(round(x2 / 1000 * w)),
        int(round(y2 / 1000 * h)),
    )


def draw_boxes(img, boxes, *, color_mode="class", title="", subtitle=""):
    im = img.convert("RGB").copy()
    scale = 2
    im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    w, h = img.size
    for label, box in boxes:
        x1, y1, x2, y2 = tok_to_px(box, w, h)
        x1, x2 = sorted((x1 * scale, x2 * scale))
        y1, y2 = sorted((y1 * scale, y2 * scale))
        if x2 <= x1 or y2 <= y1:
            continue
        if color_mode == "gt":
            color = GT_COLOR
        elif color_mode == "pred":
            color = PRED_COLOR
        else:
            color = PALETTE.get(label, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = label[:18]
        tw = 6 * len(tag) + 4
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw, ty + 12], fill=color)
        draw.text((x1 + 2, ty), tag, fill=(0, 0, 0), font=FONT)
    header_h = 28
    canvas = Image.new("RGB", (im.width, im.height + header_h), (18, 18, 18))
    canvas.paste(im, (0, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), f"{title}  n={len(boxes)}  {subtitle}", fill=(240, 240, 240), font=FONT)
    return canvas


def concat_row(images):
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + 8 * (len(images) - 1)
    row = Image.new("RGB", (w, h), (8, 8, 8))
    x = 0
    for im in images:
        row.paste(im, (x, 0))
        x += im.width + 8
    return row


def load_samples(pick_seed: int, num: int, exclude: set[str], pool_mode: str):
    samples = []
    with ANN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if pool_mode == "eval50":
        random.seed(42)
        pool = random.sample(samples, 50)
    else:
        pool = samples
    candidates = []
    for s in pool:
        if s["image"] in exclude:
            continue
        gt = extract_gt(s)
        counts = Counter(lab for lab, _ in gt)
        candidates.append((s, gt, counts))
    rng = random.Random(pick_seed)
    if num < len(candidates):
        return rng.sample(candidates, num)
    rng.shuffle(candidates)
    return candidates


def detect_all(worker, image):
    preds = []
    none_n = 0
    raw = {}
    for cls in DOTA_V1_CLASSES:
        result = worker.detect(
            image,
            [cls],
            generation_mode="fast",
            max_new_tokens=512,
            temperature=0.0,
            verbose=False,
        )
        answer = result.get("answer", "")
        raw[cls] = answer
        if NONE_RE.search(answer):
            none_n += 1
        preds.extend(parse_answer(answer))
    return preds, none_n, raw


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pick-seed", type=int, default=7)
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--prefix", type=str, default="rand")
    p.add_argument("--exclude-json", type=Path, action="append", default=None)
    p.add_argument("--pool", choices=("eval50", "full"), default="eval50")
    p.add_argument(
        "--models",
        default="lda,mix4k",
        help="Comma-separated: lda, mix4k, run1",
    )
    return p.parse_args()


def main():
    import gc

    import torch
    from locateanything_worker import LocateAnythingWorker

    args = parse_cli()
    exclude: set[str] = set()
    for path in args.exclude_json or []:
        if path.exists():
            for rec in json.loads(path.read_text()):
                exclude.add(rec["image"])

    ckpt_map = {"lda": LDA_CKPT, "mix4k": MIX_CKPT, "run1": RUN1_CKPT}
    model_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    for tag in model_tags:
        if tag not in ckpt_map:
            raise ValueError(f"unknown model {tag}")

    OUT.mkdir(parents=True, exist_ok=True)
    picks = load_samples(args.pick_seed, args.num, exclude, args.pool)
    print(
        "picked",
        len(picks),
        "seed",
        args.pick_seed,
        "pool",
        args.pool,
        "exclude",
        len(exclude),
        "models",
        model_tags,
        flush=True,
    )
    for sample, gt, counts in picks:
        print(" ", sample["image"], "gt", len(gt), dict(counts), flush=True)

    results = {}
    for tag in model_tags:
        ckpt = ckpt_map[tag]
        print("loading", tag, ckpt, flush=True)
        worker = LocateAnythingWorker(ckpt, device="cuda", dtype=torch.bfloat16, attn="sdpa")
        for i, (sample, gt, counts) in enumerate(picks):
            img_path = DATA_ROOT / sample["image"]
            image = Image.open(img_path).convert("RGB")
            preds, none_n, raw = detect_all(worker, image)
            rec = results.setdefault(
                i,
                {
                    "sample": sample,
                    "gt": gt,
                    "counts": dict(counts),
                    "image": str(img_path),
                },
            )
            rec[tag] = {"preds": preds, "none": none_n, "raw": raw}
            print(f"  {tag} tile{i} preds={len(preds)} none={none_n}", flush=True)
        del worker
        gc.collect()
        torch.cuda.empty_cache()

    sheets = []
    meta = []
    for i, rec in results.items():
        image = Image.open(rec["image"]).convert("RGB")
        gt = rec["gt"]
        stem = Path(rec["sample"]["image"]).stem
        count_s = ",".join(f"{k}:{v}" for k, v in rec["counts"].items())[:60]
        panels = [
            draw_boxes(image, [], color_mode="pred", title=f"{args.prefix}{i} RGB", subtitle=stem[:40]),
            draw_boxes(image, gt, color_mode="gt", title="GT", subtitle=count_s),
        ]
        row_meta = {
            "index": i,
            "image": rec["sample"]["image"],
            "gt_n": len(gt),
            "gt_counts": rec["counts"],
            "file": f"{args.prefix}_{i:02d}_{stem}.png",
        }
        for tag in model_tags:
            pred = rec[tag]
            panels.append(
                draw_boxes(
                    image,
                    pred["preds"],
                    color_mode="pred",
                    title=f"{tag} pred",
                    subtitle=f"n={len(pred['preds'])} none={pred['none']}",
                )
            )
            row_meta[f"{tag}_n"] = len(pred["preds"])
            row_meta[f"{tag}_none"] = pred["none"]
        row = concat_row(panels)
        out_path = OUT / row_meta["file"]
        row.save(out_path, format="PNG")
        sheets.append(row)
        meta.append(row_meta)
        print("saved", out_path, flush=True)

    max_w = max(im.width for im in sheets)
    total_h = sum(im.height for im in sheets) + 8 * (len(sheets) - 1)
    contact = Image.new("RGB", (max_w, total_h), (8, 8, 8))
    y = 0
    for im in sheets:
        contact.paste(im, (0, y))
        y += im.height + 8
    contact.save(OUT / f"{args.prefix}_contact_sheet.png", format="PNG")
    (OUT / f"{args.prefix}_tiles.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print("VIZ_DONE", OUT, flush=True)



if __name__ == "__main__":
    main()
