#!/usr/bin/env python
"""Patch a checkpoint's modeling_locateanything.py to use dynamic MLP construction."""

import sys
from pathlib import Path

path = Path(sys.argv[1])
code = path.read_text()

helpers = '''
def get_visual_projector_input_size(vision_config) -> int:
    merge_kernel = getattr(vision_config, "merge_kernel_size", (2, 2))
    if len(merge_kernel) != 2:
        raise ValueError("vision_config.merge_kernel_size must contain two positive integers")
    merge_h, merge_w = int(merge_kernel[0]), int(merge_kernel[1])
    if merge_h <= 0 or merge_w <= 0:
        raise ValueError("vision_config.merge_kernel_size must contain two positive integers")
    return int(vision_config.hidden_size) * merge_h * merge_w


def build_mlp_projector(vision_config, text_config) -> nn.Sequential:
    visual_hidden_size = get_visual_projector_input_size(vision_config)
    llm_hidden_size = int(text_config.hidden_size)
    return nn.Sequential(
        nn.LayerNorm(visual_hidden_size),
        nn.Linear(visual_hidden_size, llm_hidden_size),
        nn.GELU(),
        nn.Linear(llm_hidden_size, llm_hidden_size),
    )

'''

code = code.replace(
    "class LocateAnythingForConditionalGeneration",
    helpers + "class LocateAnythingForConditionalGeneration",
    1,
)

old_mlp = """        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        # MLP for moonvit (without pixel_shuffle_back, direct mapping)
        self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size*4),
                nn.Linear(vit_hidden_size*4, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )"""

new_mlp = """        self.mlp1 = build_mlp_projector(config.vision_config, config.text_config)
        self.visual_merge_kernel_size = tuple(int(x) for x in getattr(config.vision_config, "merge_kernel_size", (2, 2)))"""

if old_mlp not in code:
    print("ERROR: Could not find old MLP block to replace", file=sys.stderr)
    sys.exit(1)

code = code.replace(old_mlp, new_mlp)
path.write_text(code)
print(f"Patched {path}")
