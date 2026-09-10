# DOTA-v1.0 val 手工测试 tile（45 张）

官方 val 切成的 448 tile，路径相对 `data/dota_v1_hbb_448_mix_v1/`（Streamlit 侧栏 data root）。**15 类 × 3 张，互不重复**。优先框数适中、尽量单类，便于对照 GT。

标注来源：`annotations/DOTA-v1.0_test_t1_hbb_448.jsonl`。

Streamlit：split 选 `val`，filter 填 `Pxxxx`，只勾**该列那个类**才和「这类好不好看」对齐；15 类全勾才接近 mAP 穷举协议。

## plane（11–12）

- `tiles/val/P0217__x448_y448_s448.png`
- `tiles/val/P1738__x3552_y2240_s448.png`
- `tiles/val/P1583__x1792_y2240_s448.png`

## baseball-diamond（4–5）

- `tiles/val/P1732__x1792_y2240_s448.png`
- `tiles/val/P1583__x2688_y1792_s448.png`
- `tiles/val/P1584__x0_y3136_s448.png`

## bridge（5–6）

- `tiles/val/P1700__x3552_y0_s448.png`
- `tiles/val/P2645__x1344_y2994_s448.png`
- `tiles/val/P1356__x1344_y1344_s448.png`

## ground-track-field（1）

- `tiles/val/P1184__x1792_y3584_s448.png`
- `tiles/val/P1230__x4480_y1792_s448.png`
- `tiles/val/P1230__x4752_y1792_s448.png`

## small-vehicle（11–12）

- `tiles/val/P2610__x448_y896_s448.png`
- `tiles/val/P0749__x0_y376_s448.png`
- `tiles/val/P0791__x683_y0_s448.png`

## large-vehicle（12）

- `tiles/val/P0547__x445_y734_s448.png`
- `tiles/val/P1434__x1558_y0_s448.png`
- `tiles/val/P1452__x0_y0_s448.png`

## ship（12）

- `tiles/val/P0704__x448_y0_s448.png`
- `tiles/val/P0837__x1236_y2240_s448.png`
- `tiles/val/P0837__x1236_y2272_s448.png`

## tennis-court（12）

- `tiles/val/P1878__x0_y611_s448.png`
- `tiles/val/P2098__x0_y356_s448.png`
- `tiles/val/P2129__x393_y0_s448.png`

## basketball-court（4–5）

- `tiles/val/P2768__x448_y1344_s448.png`
- `tiles/val/P1432__x2688_y1792_s448.png`
- `tiles/val/P1440__x0_y896_s448.png`

## storage-tank（12）

- `tiles/val/P1242__x0_y1344_s448.png`
- `tiles/val/P2645__x4157_y896_s448.png`
- `tiles/val/P2695__x896_y0_s448.png`

## soccer-ball-field（2–4）

- `tiles/val/P2541__x1792_y1344_s448.png`
- `tiles/val/P0146__x1792_y896_s448.png`
- `tiles/val/P1184__x448_y5376_s448.png`

## roundabout（2–3）

- `tiles/val/P1380__x1344_y448_s448.png`
- `tiles/val/P1370__x2688_y896_s448.png`
- `tiles/val/P1380__x896_y448_s448.png`

## harbor（12）

- `tiles/val/P1009__x1344_y0_s448.png`
- `tiles/val/P1009__x0_y448_s448.png`
- `tiles/val/P1009__x448_y448_s448.png`

## swimming-pool（7–8）

- `tiles/val/P1377__x2688_y3136_s448.png`
- `tiles/val/P1384__x0_y2688_s448.png`
- `tiles/val/P1380__x0_y2688_s448.png`

## helicopter（9–12，图里还有 plane）

- `tiles/val/P1397__x1344_y2688_s448.png`
- `tiles/val/P0168__x1344_y896_s448.png`
- `tiles/val/P1390__x4032_y2650_s448.png`
