#!/usr/bin/env python
"""RL Phase A reward for LocateAnything-SAT HBB detection (design v3).

纯 CPU（torch/torchvision/scipy，无模型依赖）；HBB only（OBB 属 Phase C2）。

模型输出 token 格式（与 locateanything_worker / 训练 JSONL 一致）：
  框:     <ref>label</ref><box><x1><y1><x2><y2></box>   坐标为 [0,1000] 归一化整数
  无目标: <box>none</box>

v3 设计（替换 v2.1 的 20 框硬门 / NMS / many-to-one max-IoU / count·dup 项）：
  - 输出框上限 MAX_BOXES=256（21–256 框合法可解析）；GT 解析用独立上限
    GT_MAX_BOXES=4096，两者不复用。
  - 不做 NMS：重复预测作为未匹配 FP 自然压低 P_soft / F1_soft。
  - match_one_to_one(): scipy linear_sum_assignment 最大化 label-aware IoU，
    一对一；跨 label 配对质量固定 0；只返回 IoU>0 的 (pred_i, gt_j, iou)。
  - 正样本：
      S = sum(matched IoU)；P_soft = S/N_pred；R_soft = S/N_gt；
      F1_soft = 2S/(N_pred+N_gt)
      R_small = sum(matched IoU on area<32px^2 GT) / N_small_gt（448px 坐标面积）
      r_main  = 0.5*F1_soft + 0.5*R_small   （N_small_gt==0 时退化为 F1_soft）
  - parse invalid / 正样本答 <box>none</box> / 只有退化框 → r_main = -1.0
  - 空 GT：纯 none → 1.0；合法吐框 → -min(1, n_raw/10)；非法格式 → -1.0
  - 奖励总范围固定 [-1, 1]。

match_at_iou(): 阈值化贪心一对一（IoU 降序、同 label、IoU>=thr），f1_at() 与
freeze gate 的 hard anchor 共用；与 eval_baseline_dota.match_boxes 同构。
"""
from __future__ import annotations

import math
import re

import torch
from torchvision.ops import box_iou

MAX_BOXES = 256  # 模型输出框上限 = 格式失败阈值（21–256 合法）
GT_MAX_BOXES = 4096  # GT 解析独立上限，与输出上限不复用
SMALL_AREA = 32.0 ** 2  # COCO small 面积分桶（px^2，448 tile 像素坐标）

_RE_PAIR = re.compile(r"<ref>(.*?)</ref>\s*<box>(.*?)</box>", re.S)
_RE_BOX = re.compile(r"<box>(.*?)</box>", re.S)
_RE_REF = re.compile(r"<ref>(.*?)</ref>", re.S)
_RE_COORD = re.compile(r"<(-?\d+)>")


def parse_answer(text: str, tile_size: int = 448, max_boxes: int = MAX_BOXES) -> dict:
    """解析模型/GT 答案文本 → {valid, is_none, boxes, n_raw}。

    boxes: list[(label, x1, y1, x2, y2)]，坐标已 token→像素（c·tile_size/1000，clamp 到
    [0, tile_size]）；退化框（x2<=x1 或 y2<=y1）从 boxes 丢弃但仍计入 n_raw。
    n_raw: 解析出的框总数（丢弃前）。n_raw>max_boxes → valid=False。
    输出解析用默认 max_boxes=MAX_BOXES；解析 GT 必须显式传 GT_MAX_BOXES。
    """
    blocks = [m.group(1).strip() for m in _RE_BOX.finditer(text)]
    refs = _RE_REF.findall(text)
    pairs = _RE_PAIR.findall(text)

    # none 答案：全部 box 块均为 "none" 且无 ref（混排 none+实体框判非法，防博弈）
    if blocks and not refs and all(b == "none" for b in blocks):
        return {"valid": True, "is_none": True, "boxes": [], "n_raw": 0}

    invalid = {"valid": False, "is_none": False, "boxes": [], "n_raw": len(pairs)}
    if not blocks or not pairs:
        return invalid  # 无任何 box 块
    if len(refs) != len(pairs) or len(blocks) != len(pairs):
        return invalid  # 有 ref 无 box / 有 box 无 ref / none 混排

    boxes = []
    for label, coord_text in pairs:
        coords = [int(c) for c in _RE_COORD.findall(coord_text)]
        if len(coords) != 4:
            return invalid
        x1, y1, x2, y2 = (
            min(max(c * tile_size / 1000.0, 0.0), float(tile_size)) for c in coords
        )
        if x2 > x1 and y2 > y1:
            boxes.append((label, x1, y1, x2, y2))

    n_raw = len(pairs)
    valid = n_raw <= max_boxes
    return {
        "valid": valid,
        "is_none": False,
        "boxes": boxes if valid else [],
        "n_raw": n_raw,
    }


def _label_masked_iou(pred_boxes: list, gt_boxes: list) -> torch.Tensor:
    """[Np, Ng] IoU 矩阵；异 label 对置 0（label-aware：IoU 只在同 label 对上有效）。"""
    p = torch.tensor([[b[1], b[2], b[3], b[4]] for b in pred_boxes], dtype=torch.float64)
    g = torch.tensor([[b[1], b[2], b[3], b[4]] for b in gt_boxes], dtype=torch.float64)
    iou_mat = box_iou(p, g)
    mask = torch.tensor(
        [[pb[0] == gb[0] for gb in gt_boxes] for pb in pred_boxes], dtype=torch.bool
    )
    return iou_mat * mask


def match_one_to_one(pred_boxes: list, gt_boxes: list) -> list[tuple[int, int, float]]:
    """全局一对一最优匹配（soft reward 用）：匈牙利算法最大化 label-aware IoU。

    跨 label 配对质量固定 0；只返回 IoU>0 的 (pred_index, gt_index, iou)。
    """
    if not pred_boxes or not gt_boxes:
        return []
    iou_mat = _label_masked_iou(pred_boxes, gt_boxes)
    from scipy.optimize import linear_sum_assignment

    row_idx, col_idx = linear_sum_assignment(-iou_mat.numpy())  # maximize label-aware IoU
    return [
        (int(i), int(j), float(iou_mat[i, j]))
        for i, j in zip(row_idx, col_idx)
        if float(iou_mat[i, j]) > 0.0
    ]


def _iou_scalar(a, b) -> float:
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def match_at_iou(pred_boxes: list, gt_boxes: list, iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """阈值化贪心一对一（hard 指标用）：IoU 降序、同 label、IoU>=iou_thr。

    与 eval_baseline_dota.match_boxes 同构；返回 (pred_index, gt_index, iou)。
    """
    pred = [p for p in pred_boxes if p[3] > p[1] and p[4] > p[2]]
    gt = [g for g in gt_boxes if g[3] > g[1] and g[4] > g[2]]
    if not pred or not gt:
        return []
    scored = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p[0] != g[0]:
                continue
            sc = _iou_scalar(p, g)
            if sc >= iou_thr:
                scored.append((sc, i, j))
    scored.sort(reverse=True)
    matched_pred, matched_gt = set(), set()
    out = []
    for sc, i, j in scored:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            out.append((i, j, sc))
    return out


def _result(r_main, f1_soft, small_recall_soft, matched_iou_sum,
            n_raw, n_pred, n_gt, n_small_gt, is_none, parse_ok) -> dict:
    return {
        "r_main": r_main,
        "f1_soft": f1_soft,
        "small_recall_soft": small_recall_soft,
        "matched_iou_sum": matched_iou_sum,
        "n_raw": n_raw,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_small_gt": n_small_gt,
        "is_none": is_none,
        "parse_ok": parse_ok,
    }


def _n_small_gt(gt_boxes: list) -> int:
    return sum(1 for g in gt_boxes if (g[3] - g[1]) * (g[4] - g[2]) < SMALL_AREA)


def _reward_from_boxes(pred_boxes: list, gt_boxes: list, n_raw: int) -> dict:
    """常规分支：已解析（非 none、有非退化框）的框列表直接进奖励。selftest 用它做精确构造。"""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    matches = match_one_to_one(pred_boxes, gt_boxes)
    s = sum(iou for _, _, iou in matches)
    p_soft = s / n_pred if n_pred else 0.0
    r_soft = s / n_gt if n_gt else 0.0
    f1_soft = 2.0 * s / (n_pred + n_gt) if (n_pred + n_gt) else 0.0

    n_small = _n_small_gt(gt_boxes)
    if n_small:
        r_small = sum(iou for _, j, iou in matches
                      if (gt_boxes[j][3] - gt_boxes[j][1]) * (gt_boxes[j][4] - gt_boxes[j][2]) < SMALL_AREA) / n_small
        r_main = 0.5 * f1_soft + 0.5 * r_small
    else:
        r_small = 0.0
        r_main = f1_soft

    assert math.isfinite(r_main), f"reward not finite: {r_main}"
    assert -1.0 <= r_main <= 1.0, f"r_main out of range: {r_main}"
    return _result(r_main, f1_soft, r_small, s, n_raw, n_pred, n_gt, n_small, False, True)


def compute_reward(text: str, gt: list, *, tile_size: int = 448) -> dict:
    """主入口。gt: list[(label, x1, y1, x2, y2)]（像素坐标，与 parse_answer 同空间）。"""
    gt = [g for g in gt if g[3] > g[1] and g[4] > g[2]]  # 防御：丢弃不可匹配的退化 GT
    parse = parse_answer(text, tile_size)

    if not parse["valid"]:
        return _result(-1.0, 0.0, 0.0, 0.0, parse["n_raw"], 0, len(gt),
                       _n_small_gt(gt), False, False)

    n_raw = parse["n_raw"]

    if len(gt) == 0:  # 空 tile：纯 none 满分；合法吐框按数量衰减；不进 soft 匹配
        if parse["is_none"]:
            r = 1.0
        else:
            r = -min(1.0, n_raw / 10.0)
        return _result(r, 0.0, 0.0, 0.0, n_raw, 0, 0, 0, parse["is_none"], True)

    if parse["is_none"]:  # 有目标却答 none
        return _result(-1.0, 0.0, 0.0, 0.0, n_raw, 0, len(gt),
                       _n_small_gt(gt), True, True)

    if not parse["boxes"]:  # 合法但只有退化框（n_raw>0，无可匹配框）
        return _result(-1.0, 0.0, 0.0, 0.0, n_raw, 0, len(gt),
                       _n_small_gt(gt), False, True)

    return _reward_from_boxes(parse["boxes"], gt, n_raw)


def f1_at(pred: list, gt: list, iou_thr: float = 0.5) -> float:
    """同 label、贪心一对一、IoU>=iou_thr 的 F1（阈值化计数，非 soft）。

    空对空（TN）= 1.0；空 GT 却吐框 / 有 GT 却空 pred = 0.0。
    """
    pred = [p for p in pred if p[3] > p[1] and p[4] > p[2]]
    gt = [g for g in gt if g[3] > g[1] and g[4] > g[2]]
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    if len(pred) == 0:
        return 0.0
    tp = len(match_at_iou(pred, gt, iou_thr))
    p = tp / len(pred)
    r = tp / len(gt)
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


# --------------------------------------------------------------------------- #
# selftest：python scripts/rl/reward.py --selftest（无外部数据依赖）
# --------------------------------------------------------------------------- #

def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _ans(label: str, x1: int, y1: int, x2: int, y2: int) -> str:
    return f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"


def selftest() -> None:
    # token 空间用例统一 tile_size=1000 → token 数值 == 像素数值，消除量化舍入
    T = 1000

    # ---- parse_answer 单元 ----
    pa = parse_answer(_ans("a", 100, 200, 145, 225), tile_size=448)
    assert pa["valid"] and not pa["is_none"] and pa["n_raw"] == 1
    (lbl, x1, y1, x2, y2) = pa["boxes"][0]
    assert lbl == "a" and _approx(x1, 44.8) and _approx(y1, 89.6) \
        and _approx(x2, 64.96) and _approx(y2, 100.8), pa["boxes"]

    pa = parse_answer(_ans("a", -50, 200, 1200, 300), tile_size=448)  # clamp 到 [0, tile]
    (_, x1, y1, x2, y2) = pa["boxes"][0]
    assert (x1, y1, x2, y2) == (0.0, 89.6, 448.0, 134.4), pa["boxes"]

    pa = parse_answer(_ans("a", 500, 500, 100, 100), tile_size=T)  # 退化框：丢弃但计 n_raw
    assert pa["valid"] and pa["boxes"] == [] and pa["n_raw"] == 1

    assert parse_answer("<box>none</box>", T)["is_none"] is True
    assert parse_answer("<box> none </box>", T)["is_none"] is True
    assert parse_answer("locate nothing.", T)["valid"] is False  # 无 box 块
    assert parse_answer("<ref>a</ref>", T)["valid"] is False  # 有 ref 无 box
    assert parse_answer("<ref>a</ref> blah <box><1><2><3><4></box>", T)["valid"] is False
    assert parse_answer("<ref>a</ref><box>none</box>", T)["valid"] is False  # ref+none 混排
    assert parse_answer(_ans("a", 1, 1, 2, 2) + "<box>none</box>", T)["valid"] is False
    assert parse_answer("<ref>a</ref><box><1><2><3></box>", T)["valid"] is False  # 坐标数≠4

    # 21–256 框可解析；257 非法（v3：上限 256）
    assert parse_answer("".join(_ans("a", i, i, i + 5, i + 5) for i in range(21)), T)["valid"]
    assert parse_answer("".join(_ans("a", i, i, i + 5, i + 5) for i in range(256)), T)["valid"]
    assert parse_answer("".join(_ans("a", i, i, i + 5, i + 5) for i in range(257)), T)["valid"] is False
    # GT 解析独立上限 4096（不复用输出上限）
    assert parse_answer("".join(_ans("a", i, i, i + 5, i + 5) for i in range(300)),
                        T, max_boxes=GT_MAX_BOXES)["valid"]

    # ---- match_at_iou / match_one_to_one 单元 ----
    g2 = [("a", 0.0, 0.0, 10.0, 10.0), ("a", 20.0, 20.0, 30.0, 30.0)]
    m = match_at_iou([("a", 0.0, 0.0, 6.0, 10.0), ("a", 4.0, 0.0, 10.0, 10.0),
                      ("a", 20.0, 20.0, 30.0, 30.0)], g2, 0.5)
    # IoU 0.6 并列时按 (sc, i, j) 降序贪心：pred1 先占 g0，与 f1_at 历史行为一致
    assert [(i, j) for i, j, _ in m] == [(2, 1), (1, 0)], m
    m = match_one_to_one([("a", 0.0, 0.0, 10.0, 10.0), ("b", 0.0, 0.0, 10.0, 10.0)], g2)
    assert [(i, j) for i, j, _ in m] == [(0, 0)] and _approx(m[0][2], 1.0), m  # 错类不配

    # ---- compute_reward：30/30 完美 → 1.0 ----
    gt30 = [("a", float(11 * i), 11.0, float(11 * i + 9), 20.0) for i in range(30)]  # 全 small
    ans30 = "".join(_ans("a", round(b[1]), round(b[2]), round(b[3]), round(b[4])) for b in gt30)
    r = compute_reward(ans30, gt30, tile_size=T)
    assert _approx(r["r_main"], 1.0) and r["n_small_gt"] == 30 and r["n_pred"] == 30, r

    # ---- 20/30 完美子集（全 small）→ 0.5*(2*20/50) + 0.5*(20/30) ≈ 0.733 < 1.0 ----
    ans20 = "".join(_ans("a", round(b[1]), round(b[2]), round(b[3]), round(b[4])) for b in gt30[:20])
    r20 = compute_reward(ans20, gt30, tile_size=T)
    expect = 0.5 * (2.0 * 20 / 50) + 0.5 * (20.0 / 30)
    assert _approx(r20["r_main"], expect, 1e-9) and r20["r_main"] < r["r_main"], (r20, expect)

    # ---- 增加一个未匹配的正确框严格增益（补上漏检）----
    gt1 = [("a", 0.0, 0.0, 10.0, 10.0), ("a", 20.0, 20.0, 28.0, 28.0)]  # 全 small
    r_a = compute_reward(_ans("a", 0, 0, 10, 10), gt1, tile_size=T)
    r_ab = compute_reward(_ans("a", 0, 0, 10, 10) + _ans("a", 20, 20, 28, 28), gt1, tile_size=T)
    assert _approx(r_a["r_main"], 0.5 * (2 * 1 / 3) + 0.5 * 0.5), r_a
    assert _approx(r_ab["r_main"], 1.0) and r_ab["r_main"] > r_a["r_main"], (r_a, r_ab)

    # ---- 重复框严格降分（无 NMS：重复是未匹配 FP）----
    r_dup = compute_reward(_ans("a", 0, 0, 10, 10) * 2 + _ans("a", 20, 20, 28, 28), gt1, tile_size=T)
    assert r_dup["r_main"] < r_ab["r_main"] and _approx(r_dup["r_main"], 0.5 * (2 * 2 / 5) + 0.5 * 1.0), r_dup

    # ---- 错类不匹配 ----
    r = compute_reward(_ans("b", 0, 0, 10, 10), gt1, tile_size=T)
    assert _approx(r["r_main"], 0.0), r

    # ---- 正样本 none / 乱码 / 纯退化框 → -1 ----
    r = compute_reward("<box>none</box>", gt1, tile_size=T)
    assert _approx(r["r_main"], -1.0) and r["is_none"], r
    r = compute_reward("完全乱掉的输出 no boxes here", gt1, tile_size=T)
    assert _approx(r["r_main"], -1.0) and not r["parse_ok"], r
    r = compute_reward(_ans("a", 500, 500, 100, 100), gt1, tile_size=T)
    assert _approx(r["r_main"], -1.0) and r["parse_ok"] and r["n_pred"] == 0, r

    # ---- 21–256 框输出可解析进奖励（不再判 0）----
    r = compute_reward("".join(_ans("a", 400 + i, 400 + i, 400 + i + 5, 400 + i + 5)
                               for i in range(21)), gt1, tile_size=T)
    assert r["parse_ok"] and r["n_pred"] == 21 and r["n_raw"] == 21, r

    # ---- 空 GT：none(1.0) 优于任意吐框；非法 -1 ----
    r = compute_reward("<box>none</box>", [], tile_size=T)
    assert _approx(r["r_main"], 1.0) and r["is_none"], r
    r5 = compute_reward("".join(_ans("a", 10 * i, 10 * i, 10 * i + 9, 10 * i + 9) for i in range(5)),
                        [], tile_size=T)
    assert _approx(r5["r_main"], -0.5), r5
    r15 = compute_reward("".join(_ans("a", 10 * i, 10 * i, 10 * i + 9, 10 * i + 9) for i in range(15)),
                         [], tile_size=T)
    assert _approx(r15["r_main"], -1.0) and r15["r_main"] < r5["r_main"], r15
    r = compute_reward("完全乱掉的输出", [], tile_size=T)
    assert _approx(r["r_main"], -1.0), r

    # ---- R_small 只看 small GT：补检 large 不奖励 small 项 ----
    gt_mix = [("a", 0.0, 0.0, 10.0, 10.0), ("a", 100.0, 100.0, 200.0, 200.0)]  # 1 small + 1 large
    r_small_hit = compute_reward(_ans("a", 0, 0, 10, 10), gt_mix, tile_size=T)
    assert _approx(r_small_hit["r_main"], 0.5 * (2 * 1 / 3) + 0.5 * 1.0), r_small_hit
    r_large_hit = compute_reward(_ans("a", 100, 100, 200, 200), gt_mix, tile_size=T)
    assert _approx(r_large_hit["r_main"], 0.5 * (2 * 1 / 3) + 0.5 * 0.0), r_large_hit
    assert r_small_hit["r_main"] > r_large_hit["r_main"]
    # 无 small GT 时退化为纯 F1_soft
    gt_large = [("a", 100.0, 100.0, 200.0, 200.0)]
    r = compute_reward(_ans("a", 100, 100, 200, 200), gt_large, tile_size=T)
    assert _approx(r["r_main"], 1.0) and r["n_small_gt"] == 0, r

    # ---- f1_at（阈值化计数）----
    assert _approx(f1_at([("a", 0, 0, 10, 10)], [("a", 0, 0, 10, 10)]), 1.0)
    f = f1_at([("a", 0, 0, 6, 10), ("a", 4, 0, 10, 10)], [("a", 0, 0, 10, 10)], 0.5)
    assert _approx(f, 2.0 * 0.5 * 1.0 / 1.5), f
    assert _approx(f1_at([("b", 0, 0, 10, 10)], [("a", 0, 0, 10, 10)]), 0.0)
    assert _approx(f1_at([], [("a", 0, 0, 10, 10)]), 0.0)  # 漏检
    assert _approx(f1_at([("a", 0, 0, 10, 10)], []), 0.0)  # 虚警
    assert _approx(f1_at([], []), 1.0)  # 双空 TN

    print("reward.py selftest: all cases passed")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv[1:]:
        selftest()
    else:
        print(__doc__)
