#!/usr/bin/env python
"""RL Phase A freeze-check gate（dryrun / timing / gate / summarize）。

三个门阈值是设计定值，不许实现时调：
  gate_rho     : 全部 parse_ok 样本上 Spearman(r_main, f1_anchor) ≥ 0.7
  gate_decile  : r_main 十分位桶，桶均值 f1_anchor 的 top − bottom ≥ 0.20
  gate_timing  : mean reward_seconds ≤ 0.25 × mean gen_seconds

exit 1 的唯一合法后续是修 reward.py 后 `--stage summarize --rescore`
（从 per_sample 的 answer 列重算奖励，不重跑生成），禁止进入 grpo_loop 实现。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Embodied 根（worker）
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/rl（reward/rollout）

import reward  # noqa: E402

TILE_SIZE = 448  # DOTA HBB mix v1 tile 尺寸（token→像素换算基准）

RHO_THR = 0.7
DECILE_THR = 0.20
TIMING_RATIO_THR = 0.25
GATE_WALLCLOCK_BUDGET_H = 12.0  # timing 阶段的预估上限（信息门）
FALLBACK_GEN_SECONDS = 10.0  # 单样本生成均值超此值 → 触发预决策回退 --tiles 384 --n 12


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["dryrun", "timing", "gate", "summarize"])
    p.add_argument("--model-path",
                   default="work_dirs/dota_v1_hbb_448_mix_v1_lora_lda_2gpu_4k_15k_run1")
    p.add_argument("--ann",
                   default="data/dota_v1_hbb_448_mix_v1/annotations/DOTA-v1.0_train_mix_hbb_448.jsonl")
    p.add_argument("--data-root", default="data/dota_v1_hbb_448_mix_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="work_dirs/rl_freeze_a")
    p.add_argument("--tiles", type=int, default=512)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--rescore", action="store_true",
                   help="仅 summarize：从 per_sample 的 answer 列重算奖励，不重跑生成")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def load_samples(ann_path: str) -> list:
    samples = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def dedupe_by_image(samples: list) -> list:
    """按 image 路径去重（同 tile 多任务行保留首行，保持文件顺序）。"""
    seen = {}
    for row in samples:
        if row["image"] not in seen:
            seen[row["image"]] = row
    return list(seen.values())


def select_tiles(args) -> list:
    samples = load_samples(args.ann)
    print(f"[data] loaded {len(samples)} rows from {args.ann}")
    print("[data] FIRST SAMPLE (字段核对, AGENTS §4.1):")
    print(json.dumps(samples[0], ensure_ascii=False, indent=2))
    population = dedupe_by_image(samples)
    print(f"[data] {len(population)} unique tiles after dedupe")
    k = min(args.tiles, len(population))
    chosen = random.Random(args.seed).sample(population, k)
    return chosen


def shard_split(items: list, shard: int, num_shards: int) -> list:
    return items[shard::num_shards]


def parse_gt(row: dict) -> list:
    # GT 由 converter 生成，单 tile 最多 30 框（--max-boxes-per-sample 30）；
    # 20 框上限只约束模型输出（格式失败阈值），GT 按 converter 实际上限解析
    return reward.parse_answer(row["conversations"][1]["value"], TILE_SIZE, max_boxes=30)["boxes"]


def load_image(args, rel_path: str):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # 防 DecompressionBomb 保护拦截
    return Image.open(Path(args.data_root) / rel_path).convert("RGB")


# --------------------------------------------------------------------------- #
# 逐样本评测（生成 + 奖励 + 评测锚点）
# --------------------------------------------------------------------------- #

def score_one(answer: str, gt: list, gen_seconds: float) -> dict:
    t0 = time.perf_counter()
    r = reward.compute_reward(answer, gt, tile_size=TILE_SIZE)
    pred_boxes = reward.parse_answer(answer, TILE_SIZE)["boxes"]
    f1_05 = reward.f1_at_05(pred_boxes, gt)
    f1_anc = reward.f1_anchor(pred_boxes, gt)
    reward_seconds = time.perf_counter() - t0
    return {
        "answer": answer,
        "r_main": r["r_main"],
        "r_eff": r["r_eff"],
        "f1_soft": r["f1_soft"],
        "f1_05": f1_05,
        "f1_anchor": f1_anc,
        "mean_max_iou": r["mean_max_iou"],
        "n_raw": r["n_raw"],
        "n_nms": r["n_nms"],
        "is_none": r["is_none"],
        "parse_ok": r["parse_ok"],
        "gen_seconds": gen_seconds,
        "reward_seconds": reward_seconds,
    }


def trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def run_dryrun(args) -> int:
    import rollout

    tiles = select_tiles(args)[:4]
    worker = rollout.make_worker(args.model_path)
    n = 2
    rows = []
    for row in tiles:
        gt = parse_gt(row)
        img = load_image(args, row["image"])
        for s in rollout.sample_tile(worker, img, row["conversations"][0]["value"], n):
            rec = score_one(s["answer"], gt, s["gen_seconds"])
            rec.update(tile=row["image"], shard=args.shard)
            rows.append(rec)
            print(f"[dryrun] {Path(row['image']).name} r_main={rec['r_main']:+.3f} "
                  f"f1_05={rec['f1_05']:.3f} parse_ok={rec['parse_ok']} "
                  f"n_raw={rec['n_raw']} none={rec['is_none']}")
            print(f"    prompt: {trunc(row['conversations'][0]['value'], 100)}")
            print(f"    answer: {trunc(rec['answer'], 140)}")
    rate = sum(r["parse_ok"] for r in rows) / len(rows)
    print(f"[dryrun] parse_ok rate = {rate:.2f} ({sum(r['parse_ok'] for r in rows)}/{len(rows)}); "
          f"放行阈值 >= 0.8")
    return 0 if rate >= 0.8 else 1


def run_timing(args) -> int:
    import rollout

    tiles = select_tiles(args)[:32]
    worker = rollout.make_worker(args.model_path)
    gen_s, reward_s = [], []
    t_start = time.perf_counter()
    for i, row in enumerate(tiles):
        gt = parse_gt(row)
        img = load_image(args, row["image"])
        for s in rollout.sample_tile(worker, img, row["conversations"][0]["value"], args.n):
            rec = score_one(s["answer"], gt, s["gen_seconds"])
            gen_s.append(rec["gen_seconds"])
            reward_s.append(rec["reward_seconds"])
        done = i + 1
        el = time.perf_counter() - t_start
        print(f"[timing] {done}/{len(tiles)} tiles, elapsed {el / 60:.1f} min, "
              f"ETA {(el / done) * (len(tiles) - done) / 60:.1f} min", flush=True)

    gen_s.sort()
    reward_s.sort()

    def pct(a, q):
        return a[min(len(a) - 1, max(0, math.ceil(q * len(a)) - 1))]

    mean_gen = sum(gen_s) / len(gen_s)
    mean_reward = sum(reward_s) / len(reward_s)
    total_samples = args.tiles * args.n
    wall_h = (mean_gen + mean_reward) * total_samples / args.num_shards / 3600.0
    print("[timing] ===== 计时表 =====")
    print(f"  gen_seconds  : mean={mean_gen:.3f} median={pct(gen_s, 0.5):.3f} "
          f"p90={pct(gen_s, 0.9):.3f} max={gen_s[-1]:.3f}")
    print(f"  reward_seconds: mean={mean_reward:.4f} median={pct(reward_s, 0.5):.4f} "
          f"max={reward_s[-1]:.4f}")
    print(f"  reward/gen ratio = {mean_reward / mean_gen:.4f} (gate_timing 阈值 0.25)")
    print(f"  gate 预估（{args.tiles} tiles × {args.n} × {args.num_shards} 卡）= {wall_h:.2f} h "
          f"(预算 {GATE_WALLCLOCK_BUDGET_H:.0f} h)")
    fallback = mean_gen > FALLBACK_GEN_SECONDS
    print(f"  单样本生成均值 {'>' if fallback else '<='} {FALLBACK_GEN_SECONDS}s → "
          + ("触发预决策回退：--tiles 384 --n 12（门阈值不变）" if fallback else "无需回退"))
    return 1 if fallback else 0


def run_gate(args) -> int:
    import rollout

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / f"per_sample_shard{args.shard}.jsonl"

    done = set()
    if per_sample_path.exists():
        with open(per_sample_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["tile"], r["k"]))
        print(f"[gate] resume: {len(done)} (tile,k) already done in {per_sample_path}")

    tiles = shard_split(select_tiles(args), args.shard, args.num_shards)
    print(f"[gate] shard {args.shard}/{args.num_shards}: {len(tiles)} tiles × n={args.n}")
    worker = rollout.make_worker(args.model_path)

    t_start = time.perf_counter()
    n_done = 0
    with open(per_sample_path, "a", encoding="utf-8") as fout:
        for i, row in enumerate(tiles):
            need = [k for k in range(args.n) if (row["image"], k) not in done]
            if not need:
                continue
            gt = parse_gt(row)
            img = load_image(args, row["image"])
            for k in need:
                s = rollout.sample_tile(worker, img, row["conversations"][0]["value"], 1)[0]
                rec = score_one(s["answer"], gt, s["gen_seconds"])
                rec.update(tile=row["image"], shard=args.shard, k=k)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            n_done_tiles = i + 1
            if n_done_tiles % 10 == 0 or n_done_tiles == len(tiles):
                el = time.perf_counter() - t_start
                eta = el / max(n_done, 1) * (len(tiles) * args.n - (len(done) + n_done)) \
                    if n_done else 0.0
                print(f"[gate] shard {args.shard}: {n_done_tiles}/{len(tiles)} tiles, "
                      f"{len(done) + n_done} samples, elapsed {el / 60:.1f} min, "
                      f"ETA {eta / 60:.1f} min", flush=True)
    print(f"[gate] shard {args.shard} finished: wrote {n_done} new samples to {per_sample_path}")
    return 0


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #

def _ranks(v: list) -> list:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 平均秩（1-based）
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r

def spearman(x: list, y: list) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        return float(spearmanr(x, y).correlation)
    except Exception:
        rx, ry = _ranks(x), _ranks(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        vx = sum((a - mx) ** 2 for a in rx)
        vy = sum((b - my) ** 2 for b in ry)
        if vx <= 0 or vy <= 0:
            return float("nan")
        return cov / math.sqrt(vx * vy)


def _pct(a: list, q: float) -> float:
    a = sorted(a)
    return a[min(len(a) - 1, max(0, math.ceil(q * len(a)) - 1))]


def _mean(a: list) -> float:
    return sum(a) / len(a) if a else float("nan")


def _std(a: list) -> float:
    if len(a) < 2:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1))


def load_per_sample_rows(out_dir: Path) -> list:
    rows = []
    paths = sorted(out_dir.glob("per_sample_shard*.jsonl"))
    if not paths:
        sys.exit(f"[summarize] no per_sample_shard*.jsonl under {out_dir}")
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def rescore_rows(args, rows: list) -> list:
    """从 answer 列重算奖励字段（修 reward.py 后使用；不重跑生成）。"""
    gt_map = {}
    for row in load_samples(args.ann):
        if row["image"] not in gt_map:
            gt_map[row["image"]] = parse_gt(row)
    for r in rows:
        gt = gt_map.get(r["tile"], [])
        rec = score_one(r["answer"], gt, r["gen_seconds"])
        r.update({k: rec[k] for k in rec if k != "answer"})
    # 按 shard 分组回写原文件
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    for shard, rs in by_shard.items():
        p = Path(args.out_dir) / f"per_sample_shard{shard}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[summarize] rescored {len(rs)} rows -> {p}")
    return rows


def run_summarize(args) -> int:
    out_dir = Path(args.out_dir)
    rows = load_per_sample_rows(out_dir)
    print(f"[summarize] loaded {len(rows)} samples from per_sample_shard*.jsonl")
    if args.rescore:
        rows = rescore_rows(args, rows)

    ok = [r for r in rows if r["parse_ok"]]
    n_ok = len(ok)

    def _anc(row):
        return float(row["f1_anchor"]) if "f1_anchor" in row else float(row["f1_05"])

    # --- gate_rho ---
    rho = spearman([r["r_main"] for r in ok], [_anc(r) for r in ok]) if n_ok >= 2 else float("nan")
    gate_rho = math.isfinite(rho) and rho >= RHO_THR

    # --- gate_decile ---
    decile_diff = float("nan")
    bucket_means = []
    if n_ok >= 10:
        srt = sorted(ok, key=lambda r: r["r_main"])
        n = len(srt)
        buckets = [srt[round(b * n / 10): round((b + 1) * n / 10)] for b in range(10)]
        buckets = [b for b in buckets if b]
        bucket_means = [_mean([_anc(r) for r in b]) for b in buckets]
        decile_diff = bucket_means[-1] - bucket_means[0]
    gate_decile = math.isfinite(decile_diff) and decile_diff >= DECILE_THR

    # --- gate_timing ---
    mean_reward = _mean([r["reward_seconds"] for r in rows])
    mean_gen = _mean([r["gen_seconds"] for r in rows])
    timing_ratio = mean_reward / mean_gen if mean_gen > 0 else float("inf")
    gate_timing = math.isfinite(timing_ratio) and timing_ratio <= TIMING_RATIO_THR

    # --- 诊断（不设门）---
    by_tile = {}
    for r in ok:
        by_tile.setdefault(r["tile"], []).append(r["r_main"])
    tile_stds = [_std(vs) for vs in by_tile.values() if len(vs) >= 2]
    r_eff_rho = (spearman([r["r_eff"] for r in ok], [_anc(r) for r in ok])
                 if n_ok >= 2 else float("nan"))
    rho_f105 = (spearman([r["r_main"] for r in ok], [r["f1_05"] for r in ok])
                if n_ok >= 2 else float("nan"))
    diag = {
        "n_total": len(rows),
        "n_parse_ok": n_ok,
        "none_rate": _mean([1.0 if r["is_none"] else 0.0 for r in ok]),
        "parse_fail_rate": 1.0 - n_ok / len(rows) if rows else float("nan"),
        "dup_rate": _mean([1.0 if r["n_raw"] > r["n_nms"] else 0.0 for r in ok]),
        "r_main_p10": _pct([r["r_main"] for r in ok], 0.10),
        "r_main_p50": _pct([r["r_main"] for r in ok], 0.50),
        "r_main_p90": _pct([r["r_main"] for r in ok], 0.90),
        "tau_ref_p10_tile_std_r": _pct(tile_stds, 0.10) if tile_stds else float("nan"),
        "spearman_r_eff_f1_anchor": r_eff_rho,
        "spearman_r_main_f1_05": rho_f105,
        "anchor": "mean(F1@0.3, F1@0.5, F1@0.7)",
    }

    summary = {
        "model_path": args.model_path,
        "ann": args.ann,
        "tiles": args.tiles,
        "n": args.n,
        "seed": args.seed,
        "rescored": bool(args.rescore),
        "gate_rho": {"value": rho, "threshold": RHO_THR, "pass": bool(gate_rho)},
        "gate_decile": {"diff": decile_diff, "threshold": DECILE_THR,
                        "bucket_means_f1_anchor": bucket_means, "pass": bool(gate_decile)},
        "gate_timing": {"mean_reward_seconds": mean_reward, "mean_gen_seconds": mean_gen,
                        "ratio": timing_ratio, "threshold": TIMING_RATIO_THR,
                        "pass": bool(gate_timing)},
        "diagnostics": diag,
        "all_pass": bool(gate_rho and gate_decile and gate_timing),
    }
    out_path = out_dir / "summary_gate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[summarize] ===== GATE =====")
    print(f"  gate_rho    : rho={rho:.4f} (thr >= {RHO_THR})  -> {'PASS' if gate_rho else 'FAIL'}")
    print(f"  gate_decile : top-bottom={decile_diff:.4f} (thr >= {DECILE_THR}) "
          f"-> {'PASS' if gate_decile else 'FAIL'}")
    if bucket_means:
        print(f"    decile bucket mean f1_anchor (bottom->top): "
              + " ".join(f"{m:.3f}" for m in bucket_means))
    print(f"  gate_timing : reward/gen={timing_ratio:.4f} (thr <= {TIMING_RATIO_THR}; "
          f"reward={mean_reward:.4f}s gen={mean_gen:.2f}s) -> "
          f"{'PASS' if gate_timing else 'FAIL'}")
    print(f"[summarize] diagnostics: {json.dumps(diag, ensure_ascii=False)}")
    print(f"[summarize] summary -> {out_path}")
    print(f"[summarize] ALL {'PASS' if summary['all_pass'] else 'FAIL'} "
          f"(exit {0 if summary['all_pass'] else 1})")
    return 0 if summary["all_pass"] else 1


def main() -> int:
    args = parse_args()
    random.seed(args.seed)  # 仅影响非采样逻辑；tile 抽样用独立 Random(seed)
    if args.stage == "dryrun":
        return run_dryrun(args)
    if args.stage == "timing":
        return run_timing(args)
    if args.stage == "gate":
        return run_gate(args)
    return run_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
