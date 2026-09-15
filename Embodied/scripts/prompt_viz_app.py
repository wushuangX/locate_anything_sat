#!/usr/bin/env python
"""Interactive prompt + box visualization for LocateAnything (Streamlit).

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data/locate_anything_sat/Embodied \\
    streamlit run scripts/prompt_viz_app.py --server.port 6006 --server.address 0.0.0.0
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
PRED_COLOR = (255, 60, 60)
PRED_PALETTE = [(255, 60, 60), (255, 140, 0), (180, 80, 255), (0, 180, 200)]
GT_COLOR = (40, 220, 70)
FONT = ImageFont.load_default()

PRESETS = {
    "A train/detect": "Locate all the instances that matches the following description: {class}.",
    "B grammar match": "Locate all the instances that match the following description: {class}.",
    "C single": "Locate a single instance that matches the following description: {class}.",
    "D short Locate": "Locate all the {class}.",
    "E Detect": "Detect all {class}.",
    "F class only": "{class}",
    "G aerial NL": "Find every object of category {class} in this aerial image.",
    "H duplicated class": "Locate all the instances that matches the following description: {class}</c>{class}.",
    "free prompt (no {class})": "Locate all the instances that matches the following description: small-vehicle.",
}

DEFAULT_CKPT = "/data/locate_anything_sat/Embodied/work_dirs/dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1/checkpoint-15000"
DEFAULT_CKPT_ROOT = "/data/locate_anything_sat/Embodied/work_dirs"
BASE_MODEL_CANDIDATES = [
    "/data/LocateAnything-3B",
    "/root/autodl-tmp/LocateAnything-3B",
]
DEFAULT_COMPARE_CKPTS = [
    DEFAULT_CKPT,
    f"{DEFAULT_CKPT_ROOT}/dota_geom_lora_lda_2gpu_4k_100k_run1",
]
DEFAULT_ROOT = "/data/locate_anything_sat/Embodied/data/dota_v1_hbb_448_mix_v1"


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


def match_totals(preds, gts, iou_thr=0.5) -> tuple[int, int, int]:
    tp, fp, fn = match_boxes(preds, gts, iou_thr)
    return sum(tp.values()), sum(fp.values()), sum(fn.values())



def draw_boxes(img, boxes, title="", color=PRED_COLOR):
    im = img.convert("RGB").copy()
    scale = 2
    im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    w, h = img.size
    for label, box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = sorted((int(round(x1 / 1000 * w * scale)), int(round(x2 / 1000 * w * scale))))
        y1, y2 = sorted((int(round(y1 / 1000 * h * scale)), int(round(y2 / 1000 * h * scale))))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = (label or "")[:18]
        tw = 6 * len(tag) + 4
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw, ty + 12], fill=color)
        draw.text((x1 + 2, ty), tag, fill=(0, 0, 0), font=FONT)
    header_h = 24
    canvas = Image.new("RGB", (im.width, im.height + header_h), (18, 18, 18))
    canvas.paste(im, (0, header_h))
    ImageDraw.Draw(canvas).text((8, 6), f"{title} n={len(boxes)}", fill=(240, 240, 240), font=FONT)
    return canvas


@st.cache_resource(max_entries=4)
def load_worker(model_path: str, device: str = "cuda"):
    import torch
    from locateanything_worker import LocateAnythingWorker

    return LocateAnythingWorker(
        model_path,
        device=device,
        dtype=torch.bfloat16,
        attn="sdpa",
    )


def release_cached_workers() -> str:
    import gc
    import torch

    load_worker.clear()
    gc.collect()
    if not torch.cuda.is_available():
        return "CUDA not available"
    lines = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        lines.append(f"cuda:{i} allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB")
    gc.collect()
    return " | ".join(lines)



@st.cache_data
def load_gt_index(data_root: str) -> dict:
    """Index detection GT per image.

    Mix JSONL repeats the same tile (T1–T5). Keep unique boxes whose
    <ref> is a DOTA class name; ignore T4 none and T5 referring phrases
    unless the tile has no class-labeled boxes.
    """
    class_set = set(DOTA_V1_CLASSES)
    class_boxes: dict[str, list] = {}
    fallback: dict[str, list] = {}
    seen_class: dict[str, set] = {}
    seen_fallback: dict[str, set] = {}
    ann_dir = Path(data_root) / "annotations"
    if not ann_dir.is_dir():
        return class_boxes
    for path in sorted(ann_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                image = sample.get("image")
                if not image:
                    continue
                gpt = sample.get("conversations", [{}, {}])
                text = gpt[1]["value"] if len(gpt) > 1 else ""
                parsed = parse_answer(text)
                if not parsed:
                    continue
                det = [(lab, box) for lab, box in parsed if lab in class_set]
                if det:
                    bucket, seen_map = class_boxes, seen_class
                    items = det
                else:
                    bucket, seen_map = fallback, seen_fallback
                    items = parsed
                seen = seen_map.setdefault(image, set())
                lst = bucket.setdefault(image, [])
                for lab, box in items:
                    key = (lab, box)
                    if key in seen:
                        continue
                    seen.add(key)
                    lst.append((lab, box))
    out = dict(fallback)
    out.update(class_boxes)
    return out


def resolve_path(tile_path: str, data_root: str) -> Path | None:
    tile_path = (tile_path or "").strip()
    if not tile_path:
        return None
    p = Path(tile_path)
    if p.is_file():
        return p
    alt = Path(data_root) / tile_path
    if alt.is_file():
        return alt
    return None


def resolve_image(upload, tile_path: str, data_root: str):
    if upload is not None:
        return Image.open(upload).convert("RGB")
    p = resolve_path(tile_path, data_root)
    if p is None:
        return None
    return Image.open(p).convert("RGB")


def lookup_gt(upload, tile_path: str, data_root: str):
    if upload is not None:
        return []
    p = resolve_path(tile_path, data_root)
    if p is None:
        return []
    index = load_gt_index(data_root)
    key = (tile_path or "").strip().replace("\\", "/")
    if key in index:
        return index[key]
    root = Path(data_root).resolve()
    try:
        rel = str(p.resolve().relative_to(root))
    except ValueError:
        rel = p.name
    if rel in index:
        return index[rel]
    for image, boxes in index.items():
        if Path(image).name == p.name:
            return boxes
    return []


def run_detect(worker, image, prompts, generation_mode: str, max_new_tokens: int):
    import time
    logs, boxes = [], []
    none_n = 0
    t0 = time.perf_counter()
    for tag, prompt in prompts:
        result = worker.predict(
            image, prompt,
            generation_mode=generation_mode,
            max_new_tokens=int(max_new_tokens),
            temperature=0.0,
            verbose=False,
        )
        answer = result.get("answer", "")
        if NONE_RE.search(answer):
            none_n += 1
        parsed = parse_answer(answer)
        boxes.extend(parsed)
        logs.append(f"[{tag}]\nprompt: {prompt}\nanswer: {answer}\nparsed: {len(parsed)} boxes")
    return boxes, none_n, logs, time.perf_counter() - t0



@st.cache_data
def list_splits(data_root: str) -> list[str]:
    tiles = Path(data_root) / "tiles"
    if not tiles.is_dir():
        return []
    return sorted(p.name for p in tiles.iterdir() if p.is_dir())


@st.cache_data
def list_tile_names(data_root: str, split: str) -> list[str]:
    folder = Path(data_root) / "tiles" / split
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.glob("*.png"))


@st.cache_data
def list_checkpoints(ckpt_root: str) -> list[str]:
    root = Path(ckpt_root)
    if not root.is_dir():
        return []
    rows: list[tuple[str, int, str]] = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if (run / "config.json").is_file():
            rows.append((run.name, -1, str(run)))
        for child in run.iterdir():
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            if not (child / "config.json").is_file():
                continue
            try:
                step = int(child.name.split("-", 1)[1])
            except ValueError:
                continue
            rows.append((run.name, step, str(child)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [p for _, _, p in rows]


@st.cache_data
def find_base_model() -> str:
    """First existing base-model dir among BASE_MODEL_CANDIDATES."""
    for p in BASE_MODEL_CANDIDATES:
        if (Path(p) / "config.json").is_file():
            return p
    return ""


def _ckpt_label(path: str, ckpt_root: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(ckpt_root).resolve()))
    except ValueError:
        return Path(path).name



def main():
    st.set_page_config(page_title="LocateAnything prompt viz", layout="wide")
    st.title("LocateAnything prompt viz")
    st.caption("Green = GT from JSONL. Red = model. `{class}` is replaced per selected DOTA class.")

    with st.sidebar:
        mode = st.radio("mode", ["single", "compare"], index=0, horizontal=True)
        ckpt_root = st.text_input("checkpoint root", DEFAULT_CKPT_ROOT)
        ckpt_filter = st.text_input("filter checkpoint", placeholder="15000 / 100k / lda")
        ckpts = list_checkpoints(ckpt_root)
        base_model = find_base_model()
        if base_model and base_model not in ckpts:
            ckpts.insert(0, base_model)
        if not base_model:
            st.caption(f"base model not found in: {' / '.join(BASE_MODEL_CANDIDATES)}")
        if ckpt_filter.strip():
            q = ckpt_filter.strip().lower()
            ckpts = [p for p in ckpts if q in p.lower()]

        def fmt(path: str) -> str:
            return _ckpt_label(path, ckpt_root)

        model_path = DEFAULT_CKPT
        ckpt_paths: list[str] = []
        if mode == "single":
            if ckpts:
                idx = ckpts.index(DEFAULT_CKPT) if DEFAULT_CKPT in ckpts else 0
                model_path = st.selectbox(
                    "checkpoint", ckpts, index=idx, format_func=fmt
                )
            else:
                st.warning("No checkpoints under checkpoint root.")
                model_path = st.selectbox("checkpoint", [""], index=0, disabled=True)
        else:
            if ckpts:
                default_sel = (
                    ([base_model] if base_model else [])
                    + [p for p in DEFAULT_COMPARE_CKPTS if p in ckpts]
                )[:4]
                ckpt_paths = st.multiselect(
                    "checkpoints (2–4)",
                    ckpts,
                    default=default_sel,
                    max_selections=4,
                    format_func=fmt,
                )
            else:
                st.warning("No checkpoints under checkpoint root.")
        data_root = st.text_input("data root", DEFAULT_ROOT)
        generation_mode = st.selectbox("generation_mode", ["hybrid", "fast", "slow"], index=0)
        max_new_tokens = st.slider("max_new_tokens", 64, 1024, 512, 64)
        if st.button("Release GPU"):
            st.success(release_cached_workers())

    preset = st.selectbox("preset", list(PRESETS.keys()), index=0)
    template = st.text_area("prompt template", value=PRESETS[preset], height=80)
    classes = st.multiselect(
        "classes (used when template contains {class})",
        DOTA_V1_CLASSES,
        default=["small-vehicle", "ship", "plane", "storage-tank"],
    )

    upload = st.file_uploader("image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    splits = list_splits(data_root)
    default_split = "val" if "val" in splits else (splits[0] if splits else "")
    split = st.selectbox(
        "tiles split",
        splits,
        index=splits.index(default_split) if default_split in splits else 0,
        disabled=not splits,
    )
    query = st.text_input("filter filename", placeholder="P0104")
    names = list_tile_names(data_root, split) if split else []
    if query.strip():
        q = query.strip().lower()
        names = [n for n in names if q in n.lower()]
    max_shown = 500
    if len(names) > max_shown:
        st.caption(f"{len(names)} matches, showing first {max_shown}. Narrow the filter.")
        names = names[:max_shown]
    picked = st.selectbox("tile", names if names else [""], index=0)
    tile_path = f"tiles/{split}/{picked}" if split and picked else ""
    if tile_path:
        st.caption(f"selected `{tile_path}`")

    preview = resolve_image(upload, tile_path, data_root)
    gt_boxes = lookup_gt(upload, tile_path, data_root)

    if mode == "compare":
        col_in, col_gt = st.columns(2)
        col_pred = None
    else:
        col_in, col_gt, col_pred = st.columns(3)
    with col_in:
        if preview is not None:
            st.image(preview, caption="input", use_container_width=True)
        else:
            st.info("Upload an image or pick a server tile.")
    with col_gt:
        if preview is not None:
            st.image(
                draw_boxes(preview, gt_boxes, title="GT", color=GT_COLOR),
                caption=f"GT ({len(gt_boxes)} boxes)" if gt_boxes else "GT (none / not in JSONL)",
                use_container_width=True,
            )

    run = st.button("Detect", type="primary")
    if run:
        image = resolve_image(upload, tile_path, data_root)
        if image is None:
            st.error("Need an uploaded image or a valid tile path.")
            return
        if "{class}" in template:
            if not classes:
                st.error("Template has {class} but no classes selected.")
                return
            prompts = [(cls, template.replace("{class}", cls)) for cls in classes]
        else:
            if not template.strip():
                st.error("Empty prompt.")
                return
            prompts = [("free", template.strip())]

        if mode == "compare":
            if len(ckpt_paths) < 2:
                st.error("Compare mode needs at least 2 checkpoints.")
                return
            if len(ckpt_paths) > 4:
                st.warning("Compare mode uses at most 4 checkpoints; extra paths ignored.")
                ckpt_paths = ckpt_paths[:4]
            try:
                import torch
                n_gpu = max(torch.cuda.device_count(), 1)
                compare_rows = []
                for i, path in enumerate(ckpt_paths):
                    device = f"cuda:{i % n_gpu}"
                    worker = load_worker(path, device)
                    boxes, none_n, logs, elapsed = run_detect(
                        worker, image, prompts, generation_mode, max_new_tokens
                    )
                    if gt_boxes:
                        tp, fp, fn = match_totals(boxes, gt_boxes)
                    else:
                        tp, fp, fn = None, None, None
                    compare_rows.append((path, boxes, none_n, logs, elapsed, tp, fp, fn))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    st.error(
                        "GPU OOM: drop a checkpoint or restart Streamlit with both GPUs visible."
                    )
                else:
                    st.exception(e)
                return
            except Exception as e:
                st.exception(e)
                return

            pred_cols = st.columns(len(compare_rows))
            raw_parts = []
            for i, (path, boxes, none_n, logs, elapsed, tp, fp, fn) in enumerate(compare_rows):
                name = Path(path).name
                color = PRED_PALETTE[i % len(PRED_PALETTE)]
                caption = f"{name} | {len(boxes)} boxes | {elapsed:.1f}s"
                if tp is None:
                    caption += " | no GT"
                else:
                    caption += f" | TP/FP/FN={tp}/{fp}/{fn} @0.5"
                with pred_cols[i]:
                    st.image(
                        draw_boxes(
                            image,
                            boxes,
                            title=f"{name} none={none_n}/{len(prompts)}",
                            color=color,
                        ),
                        caption=caption,
                        use_container_width=True,
                    )
                raw_parts.append(f"===== {name} =====\n\n" + "\n\n".join(logs))
            st.text_area("raw model output", value="\n\n".join(raw_parts), height=320)
            return

        worker = load_worker(model_path)
        boxes, none_n, logs, _elapsed = run_detect(
            worker, image, prompts, generation_mode, max_new_tokens
        )
        with col_pred:
            st.image(
                draw_boxes(image, boxes, title=f"pred none={none_n}/{len(prompts)}", color=PRED_COLOR),
                caption=f"prediction ({len(boxes)} boxes)",
                use_container_width=True,
            )
        st.text_area("raw model output", value="\n\n".join(logs), height=320)


if __name__ == "__main__":
    main()
