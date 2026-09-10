#!/usr/bin/env python
"""Interactive prompt + box visualization for LocateAnything (Streamlit).

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data/locate_anything_sat/Embodied \\
    streamlit run scripts/prompt_viz_app.py --server.port 6006 --server.address 0.0.0.0
"""
from __future__ import annotations

import json
import re
import sys
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


@st.cache_resource
def load_worker(model_path: str):
    import torch
    from locateanything_worker import LocateAnythingWorker

    return LocateAnythingWorker(
        model_path,
        device="cuda",
        dtype=torch.bfloat16,
        attn="sdpa",
    )


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


def main():
    st.set_page_config(page_title="LocateAnything prompt viz", layout="wide")
    st.title("LocateAnything prompt viz")
    st.caption("Green = GT from JSONL. Red = model. `{class}` is replaced per selected DOTA class.")

    with st.sidebar:
        model_path = st.text_input("checkpoint", DEFAULT_CKPT)
        data_root = st.text_input("data root", DEFAULT_ROOT)
        generation_mode = st.selectbox("generation_mode", ["hybrid", "fast", "slow"], index=0)
        max_new_tokens = st.slider("max_new_tokens", 64, 1024, 512, 64)

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
        worker = load_worker(model_path)
        logs = []
        boxes = []
        none_n = 0
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
        for tag, prompt in prompts:
            result = worker.predict(
                image,
                prompt,
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
        with col_pred:
            st.image(
                draw_boxes(image, boxes, title=f"pred none={none_n}/{len(prompts)}", color=PRED_COLOR),
                caption=f"prediction ({len(boxes)} boxes)",
                use_container_width=True,
            )
        st.text_area("raw model output", value="\n\n".join(logs), height=320)


if __name__ == "__main__":
    main()
