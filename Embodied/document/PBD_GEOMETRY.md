# PBD geometry: HBB and OBB

LocateAnything Parallel Box Decoding is a **typed, self-delimited block**. Task type selects the MTP window **before** `generate()`; one MTP step cannot choose 6 vs 7 from the type token at `t0`.

## Atoms

| Schema | Length | Tokens |
|---|---|---|
| HBB (default) | `L=6` | `<box> <x1> <y1> <x2> <y2> </box>` |
| OBB (opt-in) | `L=7` | `<obb> <cx> <cy> <w> <h> <θ> </obb>` |

Empty answers: HBB `<box>none</box>`; OBB `<obb>none</obb>`. Pointing stays on the HBB `<box>` path.

`<quad>` is unused. OBB is not aliased onto `<quad>`.

## Inference window

`LocateAnythingForConditionalGeneration.generate(n_future_tokens=6)` is unchanged for HBB. `detect()` does not pass the kwarg, so the inner Qwen2 `block_size` used by `_create_attention_mask` / `build_magi_ranges` stays 6.

`LocateAnythingWorker.detect_oriented` sets `n_future_tokens=7` and uses the prompt

```
Locate all oriented instances that match the following description: {classes}.
```

`generate()` assigns `language_model.model.block_size = n_future_tokens` for the decode loop and restores the previous value in `finally`.

Hybrid AR returns to MTP on `</box>` **or** `</obb>` (`out_type` stays `box_end_ar`). `is_valid_box_frame` / `decode_bbox_avg` keep HBB index-5 logic. OBB uses greedy `decode_obb_greedy` (no circular mean).

## Training recipe

CLI `--block_size` remains 6 (existing shells/recipes unchanged).

| Recipe field | Effect |
|---|---|
| `geometry` absent or `"hbb"` | Tokenizer/vocab path identical to today; sample MTP length = CLI `block_size` |
| `"geometry": "obb"` | Add `<obb>` / `</obb>`; `config.obb_*`; dataset `block_size=7`; two-row embedding/`lm_head` grad hooks |

Invalid values other than `hbb` / `obb` / absent raise `ValueError` at dataset init.

HBB+OBB mixing is hand-edited recipe JSON with two keys (one without `geometry`, one with `"obb"`). Packed batches carry per-sample `pbd_block_sizes`. Magi/SDPA packing masks use that tensor; `sample_block_sizes=None` keeps the old scalar `block_size` (and would split a 7-token OBB atom).

Launch old HBB training with the existing shell + old recipe. Do not point HBB eval (`eval_baseline_dota.py` / `eval_dota_map.py`) at OBB JSONL.

OBB launcher: `Embodied/shell/train-dota-obb-lora.sh` (still `--block_size 6`; recipe `geometry` enables tokens). Converter: `Embodied/scripts/convert_dota_obb_to_locany.py`.

## DOTA-v2.0 OBB dataset (server)

Do not mix v1 JSONL. Generate on the training machine from `Embodied/`:

```bash
# Optional if v2 labels reference v1 pixels:
# export IMAGE_EXTRA_ROOT=/data/.../DOTA-v1.0
bash shell/generate-dota-v2-obb.sh
```

Writes `data/dota_v2_obb_448_mix_v1/`: 0° mix + official-val `test_t1`, then seven D4 JSONL files, then 8-key `*_train_only.json` (`geometry: obb`, `color_jitter: true`). Human/gpt strings stay the oriented templates. Eval uses 0° `test_t1` only.


## LE90 quantization

After `cv2.minAreaRect`, always `le90_canonicalize`: if `h > w`, swap `(w,h)` and `theta += 90`; wrap `theta` into `[-90, 90)` (`+90` → `-90`).

```
theta_q = clip(round((theta + 90) / 180 * 1000), 0, 1000)
theta_deg = 180 * theta_q / 1000 - 90
```

`cx,cy` are relative to the tile origin; `w,h` relative to `tile_size`; all linear dims `round(v / tile_size * 1000)` clipped to `[0,1000]`. Drop if `w_q==0` or `h_q==0`.
