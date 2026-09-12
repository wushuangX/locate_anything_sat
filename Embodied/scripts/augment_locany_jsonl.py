#!/usr/bin/env python
"""Emit 90-degree geometric variants of a LocateAnything JSONL.

Reads a 0-degree mix JSONL, rewrites ``<box>`` coordinates in ``[0, 1000]``
token space (rot90 / rot180 / rot270 / hflip), and writes one JSONL per
requested op next to the original pixels untouched. Pair each output file
with a recipe entry whose ``rotate`` / ``hflip`` fields make the training
loader apply the matching PIL transform after image load, e.g. five keys
over the same tiles::

    {
      "dota_train_r0":    {"annotation": "..._train_mix_....jsonl",  "root": "...", "rotate": 0,   "hflip": false, "color_jitter": true},
      "dota_train_r90":   {"annotation": "..._rot90.jsonl",          "root": "...", "rotate": 90,  "hflip": false, "color_jitter": true},
      "dota_train_r180":  {"annotation": "..._rot180.jsonl",         "root": "...", "rotate": 180, "hflip": false, "color_jitter": true},
      "dota_train_r270":  {"annotation": "..._rot270.jsonl",         "root": "...", "rotate": 270, "hflip": false, "color_jitter": true},
      "dota_train_hflip": {"annotation": "..._hflip.jsonl",          "root": "...", "rotate": 0,   "hflip": true,  "color_jitter": true}
    }

T5 referring samples (single-instance prompts, leftmost/rightmost/
topmost/bottommost phrases) are dropped: position words break under
rotation/flip. ``<box>none</box>`` answers pass through unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from eaglevl.train.augmentation import GEOM_OPS, geom_op_atoms, transform_box, transform_obb

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
NONE_RE = re.compile(r"<box>[Nn]one</box>")
OBB_RE = re.compile(r"<ref>(.*?)</ref><obb><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)></obb>")
OBB_NONE_RE = re.compile(r"<obb>[Nn]one</obb>", re.IGNORECASE)

HBB_SINGLE_PROMPT_PREFIX = "Locate a single instance that matches the following description:"
OBB_SINGLE_PROMPT_PREFIX = "Locate a single oriented instance that matches the following description:"
T5_PREDICATE_SUBSTRINGS = (
    "the leftmost ",
    "the rightmost ",
    "the topmost ",
    "the bottommost ",
)

VALID_OPS = tuple(GEOM_OPS.keys())


def is_t5(human: str) -> bool:
    """True for referring samples whose spatial wording breaks under geometry."""
    if human.startswith(HBB_SINGLE_PROMPT_PREFIX) or human.startswith(OBB_SINGLE_PROMPT_PREFIX):
        return True
    return any(s in human for s in T5_PREDICATE_SUBSTRINGS)


def rewrite_gpt(gpt: str, op: str) -> str:
    """Transform every box/obb block for a geometry op. Human text is unchanged."""
    if OBB_RE.search(gpt) or OBB_NONE_RE.search(gpt):
        if not OBB_RE.search(gpt) and OBB_NONE_RE.search(gpt):
            return gpt

        def _rebuild_obb(m: re.Match) -> str:
            cx, cy, w, h, th = transform_obb(
                int(m.group(2)), int(m.group(3)), int(m.group(4)),
                int(m.group(5)), int(m.group(6)), op,
            )
            return f"<ref>{m.group(1)}</ref><obb><{cx}><{cy}><{w}><{h}><{th}></obb>"

        return OBB_RE.sub(_rebuild_obb, gpt)

    if not BOX_RE.search(gpt) and NONE_RE.search(gpt):
        return gpt

    atoms = geom_op_atoms(op)

    def _rebuild_box(m: re.Match) -> str:
        x1, y1, x2, y2 = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        for atom in atoms:
            x1, y1, x2, y2 = transform_box(x1, y1, x2, y2, atom)
        return f"<ref>{m.group(1)}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"

    return BOX_RE.sub(_rebuild_box, gpt)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Source 0-degree JSONL (not modified)")
    p.add_argument("--output-dir", required=True, help="Directory for {stem}_{op}.jsonl outputs")
    p.add_argument("--stem", required=True, help="Output filename stem, e.g. DOTA-v1.0_train_mix_hbb_448")
    p.add_argument(
        "--ops",
        default="rot90,rot180,rot270,hflip",
        help="Comma-separated geom ops: rot90,rot180,rot270,hflip,hflip_rot90,hflip_rot180,hflip_rot270",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ops = [tok.strip() for tok in args.ops.split(",") if tok.strip()]
    if not ops:
        raise ValueError("--ops is empty")
    invalid = [op for op in ops if op not in VALID_OPS]
    if invalid:
        raise ValueError(f"Invalid --ops tokens {invalid}; expected subset of {list(VALID_OPS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    writers = {
        op: open(output_dir / f"{args.stem}_{op}.jsonl", "w", encoding="utf-8")
        for op in ops
    }
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if is_t5(sample["conversations"][0]["value"]):
                    dropped += 1
                    continue
                kept += 1
                gpt = sample["conversations"][1]["value"]
                for op in ops:
                    sample["conversations"][1]["value"] = rewrite_gpt(gpt, op)
                    writers[op].write(json.dumps(sample, ensure_ascii=False) + "\n")
    finally:
        for w in writers.values():
            w.close()

    for op in ops:
        print(f"[augment] wrote {kept} samples -> {output_dir / f'{args.stem}_{op}.jsonl'}")
    print(f"[augment] input={args.input} kept={kept} dropped_t5={dropped} ops={ops}")


if __name__ == "__main__":
    main()
