#!/usr/bin/env bash
# Generate DOTA-v2.0 OBB LocateAnything tiles/JSONL/D4 recipe on the training server.
# Does not start training. Identity JSONL keeps T5; seven geom files drop T5.
# Eval stays 0° test_t1. Override paths via env; do not point this at the v1 HBB run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOTA_ROOT="${DOTA_ROOT:-/data/data/738c885b302947929603f33110544338/道路检测数据集/DOTA-v2.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/data/dota_v2_obb_448_mix_v1}"
RECIPE_NAME="${RECIPE_NAME:-dota_v2_obb_448_mix_v1}"
TILE_SIZE="${TILE_SIZE:-448}"
PYTHON="${PYTHON:-python}"
D4_OPS="rot90,rot180,rot270,hflip,hflip_rot90,hflip_rot180,hflip_rot270"

CONVERT_ARGS=(
  --dota-root "$DOTA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --version v2.0
  --splits train val
  --tile-size "$TILE_SIZE"
  --overlap 0.0
  --min-visibility 0.5
  --max-boxes-per-sample 30
  --recipe-name "$RECIPE_NAME"
)
if [[ -n "${IMAGE_EXTRA_ROOT:-}" ]]; then
  CONVERT_ARGS+=(--image-extra-roots "$IMAGE_EXTRA_ROOT")
fi

echo "[generate] convert $DOTA_ROOT -> $OUTPUT_ROOT"
"$PYTHON" scripts/convert_dota_obb_to_locany.py "${CONVERT_ARGS[@]}"

shopt -s nullglob
train_mix_files=("$OUTPUT_ROOT"/annotations/*_train_mix_obb_"${TILE_SIZE}".jsonl)
if [[ ${#train_mix_files[@]} -ne 1 ]]; then
  echo "[generate] expected one train_mix JSONL under $OUTPUT_ROOT/annotations, got: ${train_mix_files[*]:-none}" >&2
  exit 1
fi
TRAIN_MIX="${train_mix_files[0]}"
STEM="$(basename "$TRAIN_MIX" .jsonl)"

echo "[generate] D4 rewrite $STEM"
"$PYTHON" scripts/augment_locany_jsonl.py \
  --input "$TRAIN_MIX" \
  --output-dir "$OUTPUT_ROOT/annotations" \
  --stem "$STEM" \
  --ops "$D4_OPS"

ZERO_RECIPE="$OUTPUT_ROOT/recipes/${RECIPE_NAME}_train_only.json"
"$PYTHON" scripts/write_dota_obb_d4_recipe.py \
  --zero-recipe "$ZERO_RECIPE" \
  --output "$ZERO_RECIPE"

echo "[generate] train recipe: $ZERO_RECIPE (8 keys, geometry=obb, color_jitter)"
echo "[generate] eval JSONL: $OUTPUT_ROOT/annotations/*_test_t1_obb_${TILE_SIZE}.jsonl (0° only)"
echo "[generate] META_PATH=$ZERO_RECIPE"
