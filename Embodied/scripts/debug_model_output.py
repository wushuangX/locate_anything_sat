#!/usr/bin/env python
"""Quick debug: print raw model output on a single tile."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locateanything_worker import LocateAnythingWorker
from PIL import Image
import torch

w = LocateAnythingWorker("/data/LocateAnything-3B-merge1x1", device="cuda", dtype=torch.bfloat16, attn="sdpa")
img = Image.open("data/dota_v1_hbb_512/tiles/val/P0003__x0_y0_s512.png").convert("RGB")
cats = ["plane", "baseball-diamond", "bridge", "ground-track-field",
        "small-vehicle", "large-vehicle", "ship", "tennis-court",
        "basketball-court", "storage-tank", "soccer-ball-field",
        "roundabout", "harbor", "swimming-pool", "helicopter"]

r = w.detect(img, cats, generation_mode="hybrid", max_new_tokens=2048, temperature=0.0, verbose=False)
ans = r.get("answer", "")
print("=== HYBRID MODE ===")
print(repr(ans[:800]))
print()

r2 = w.detect(img, ["ship", "large-vehicle"], generation_mode="slow", max_new_tokens=512, temperature=0.0, verbose=False)
ans2 = r2.get("answer", "")
print("=== SLOW MODE ===")
print(repr(ans2[:800]))
