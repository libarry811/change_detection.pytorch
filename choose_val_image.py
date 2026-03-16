from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


SUPPORTED_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

TYPE_CLEAR = "清晰变化"
TYPE_FP = "碎点满天飞（假阳性FP）"
TYPE_FN = "建筑物空洞或断裂（假阴性FN）"
TYPE_EDGE = "边界像狗啃一样（边缘模糊）"
TYPE_UNKNOWN = "未归类"

SCENE_DESCRIPTION = {
    TYPE_CLEAR: "模型成功提取了主要变化特征，验证了改进版ResNet34骨干网络在主变化区域上的有效性。",
    TYPE_FP: "像素级模型易受光照与阴影干扰，产生孤立噪点和误检，说明需要面向对象的面积与形状约束来抑制假阳性。",
    TYPE_FN: "网络未完整保持建筑内部一致性，出现空洞、断裂或粘连，后续可通过区域生长与图斑融合进行修复。",
    TYPE_EDGE: "像素级分类缺乏几何拓扑约束，边界呈锯齿化，面向对象后处理可显著提升轮廓平滑性。",
    TYPE_UNKNOWN: "该样本不稳定匹配典型场景，建议人工复核。",
}


@dataclass
class MaskStats:
    tp: int
    fp: int
    fn: int
    tn: int
    iou: float
    precision: float
    recall: float
    f1: float
    fp_rate: float
    fn_rate: float
    pred_ratio: float
    gt_ratio: float
    edge_f1: float
    components: int
    fragmentation: float


def _choose_existing(candidates: List[str]) -> Path:
    for item in candidates:
        p = Path(item)
        if p.exists():
            return p
    return Path(candidates[0])


def _list_stems(folder: Path) -> Dict[str, Path]:
    items: Dict[str, Path] = {}
    if not folder.exists():
        return items
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            items[p.stem] = p
    return items


def _load_binary_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        max_value = float(np.max(img))
        if max_value > 0:
            img = (img / max_value) * 255.0
        img = img.astype(np.uint8)

    # Local validation outputs are usually 0/255; this also supports 0/1 masks.
    binary = img > (127 if int(img.max()) > 1 else 0)
    return binary


def _count_components(mask: np.ndarray) -> int:
    mask_u8 = mask.astype(np.uint8)
    n_labels, _labels = cv2.connectedComponents(mask_u8)
    return max(0, n_labels - 1)


def _edge_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    pred_edge = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    gt_edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)

    tp = int(np.logical_and(pred_edge, gt_edge).sum())
    fp = int(np.logical_and(pred_edge, np.logical_not(gt_edge)).sum())
    fn = int(np.logical_and(np.logical_not(pred_edge), gt_edge).sum())

    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


def _compute_stats(pred: np.ndarray, gt: np.ndarray) -> MaskStats:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    eps = 1e-9
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    gt_pos = int(gt.sum())
    bg_pos = int(np.logical_not(gt).sum())
    fp_rate = fp / (bg_pos + eps)
    fn_rate = fn / (gt_pos + eps)

    pred_ratio = float(pred.mean())
    gt_ratio = float(gt.mean())

    edge_f1 = _edge_f1(pred, gt)
    components = _count_components(pred)
    fragmentation = components / (1.0 + np.sqrt(max(1.0, float(pred.sum()))))

    return MaskStats(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fp_rate=float(fp_rate),
        fn_rate=float(fn_rate),
        pred_ratio=pred_ratio,
        gt_ratio=gt_ratio,
        edge_f1=float(edge_f1),
        components=components,
        fragmentation=float(fragmentation),
    )


def _score_types(name: str, u: MaskStats, s: MaskStats) -> Tuple[str, float, str]:
    gt_ratio = u.gt_ratio

    score_clear = -1.0
    if min(u.iou, s.iou) >= 0.70 and gt_ratio >= 0.002:
        score_clear = 0.7 * min(u.iou, s.iou) + 0.3 * min(u.edge_f1, s.edge_f1)

    score_fp = -1.0
    if gt_ratio <= 0.03 and max(u.fp_rate, s.fp_rate) >= 0.004:
        score_fp = 0.8 * max(u.fp_rate, s.fp_rate) + 0.2 * max(u.pred_ratio, s.pred_ratio)

    score_fn = -1.0
    if 0.002 <= gt_ratio <= 0.35 and max(u.fn_rate, s.fn_rate) >= 0.25:
        # Encourage cases with obvious misses but not complete collapse.
        score_fn = 0.65 * max(u.fn_rate, s.fn_rate) + 0.35 * max(u.fragmentation, s.fragmentation)

    score_edge = -1.0
    avg_edge = 0.5 * (u.edge_f1 + s.edge_f1)
    avg_frag = 0.5 * (u.fragmentation + s.fragmentation)
    if min(u.recall, s.recall) >= 0.35 and max(u.iou, s.iou) >= 0.35 and avg_edge <= 0.72:
        score_edge = 0.7 * (1.0 - avg_edge) + 0.3 * min(1.0, avg_frag)

    scores = {
        TYPE_CLEAR: score_clear,
        TYPE_FP: score_fp,
        TYPE_FN: score_fn,
        TYPE_EDGE: score_edge,
    }

    best_type = TYPE_UNKNOWN
    best_score = -1.0
    for t, sc in scores.items():
        if sc > best_score:
            best_score = sc
            best_type = t

    if best_score < 0:
        best_type = TYPE_UNKNOWN
        best_score = 0.0

    better_model = "equal"
    if u.iou > s.iou + 1e-6:
        better_model = "Unet"
    elif s.iou > u.iou + 1e-6:
        better_model = "STANet"

    return best_type, float(best_score), better_model


def _is_catastrophic_for_gt(gt_ratio: float, pred_ratio: float) -> bool:
    # Filter out extreme contradictory predictions:
    # - GT almost all background but prediction has large foreground flood.
    # - GT almost all foreground but prediction collapses to mostly background.
    if gt_ratio <= 0.01 and pred_ratio >= 0.20:
        return True
    if gt_ratio >= 0.99 and pred_ratio <= 0.80:
        return True
    return False


def _passes_basic_quality(u: MaskStats, s: MaskStats) -> bool:
    if _is_catastrophic_for_gt(u.gt_ratio, u.pred_ratio):
        return False
    if _is_catastrophic_for_gt(s.gt_ratio, s.pred_ratio):
        return False

    # Keep only samples with at least one model having usable overlap.
    if max(u.iou, s.iou) < 0.10:
        return False
    return True


def analyze(unet_dir: Path, stanet_dir: Path, label_dir: Path, topk: int) -> Tuple[List[dict], Dict[str, List[dict]]]:
    unet_map = _list_stems(unet_dir)
    stanet_map = _list_stems(stanet_dir)
    label_map = _list_stems(label_dir)

    common_names = sorted(set(unet_map) & set(stanet_map) & set(label_map))
    if not common_names:
        raise RuntimeError(
            "No paired files found. Make sure same stems exist in Unet/STANet/label folders."
        )

    all_rows: List[dict] = []
    filtered_out = 0
    for name in common_names:
        u_mask = _load_binary_mask(unet_map[name])
        s_mask = _load_binary_mask(stanet_map[name])
        g_mask = _load_binary_mask(label_map[name])

        if u_mask.shape != g_mask.shape or s_mask.shape != g_mask.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: Unet {u_mask.shape}, STANet {s_mask.shape}, GT {g_mask.shape}"
            )

        u_stats = _compute_stats(u_mask, g_mask)
        s_stats = _compute_stats(s_mask, g_mask)

        if not _passes_basic_quality(u_stats, s_stats):
            filtered_out += 1
            continue

        rep_type, score, better_model = _score_types(name, u_stats, s_stats)

        all_rows.append(
            {
                "name": name,
                "type": rep_type,
                "scene_desc": SCENE_DESCRIPTION[rep_type],
                "type_score": round(score, 6),
                "better_model": better_model,
                "unet_iou": round(u_stats.iou, 6),
                "stanet_iou": round(s_stats.iou, 6),
                "iou_gap": round(abs(u_stats.iou - s_stats.iou), 6),
                "unet_precision": round(u_stats.precision, 6),
                "stanet_precision": round(s_stats.precision, 6),
                "unet_recall": round(u_stats.recall, 6),
                "stanet_recall": round(s_stats.recall, 6),
                "unet_fp_rate": round(u_stats.fp_rate, 6),
                "stanet_fp_rate": round(s_stats.fp_rate, 6),
                "unet_fn_rate": round(u_stats.fn_rate, 6),
                "stanet_fn_rate": round(s_stats.fn_rate, 6),
                "gt_ratio": round(u_stats.gt_ratio, 6),
                "unet_edge_f1": round(u_stats.edge_f1, 6),
                "stanet_edge_f1": round(s_stats.edge_f1, 6),
            }
        )

    grouped: Dict[str, List[dict]] = {
        TYPE_CLEAR: [],
        TYPE_FP: [],
        TYPE_FN: [],
        TYPE_EDGE: [],
        TYPE_UNKNOWN: [],
    }
    for row in all_rows:
        grouped[row["type"]].append(row)

    # Keep the strongest candidates per type.
    for t in grouped:
        grouped[t].sort(key=lambda x: x["type_score"], reverse=True)
        grouped[t] = grouped[t][:topk]

    return all_rows, grouped, filtered_out


def _write_all_csv(rows: List[dict], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_group_csv(grouped: Dict[str, List[dict]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "type",
        "rank",
        "name",
        "scene_desc",
        "type_score",
        "better_model",
        "unet_iou",
        "stanet_iou",
        "iou_gap",
        "gt_ratio",
        "unet_fp_rate",
        "stanet_fp_rate",
        "unet_fn_rate",
        "stanet_fn_rate",
        "unet_edge_f1",
        "stanet_edge_f1",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, rows in grouped.items():
            for idx, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "type": t,
                        "rank": idx,
                        "name": row["name"],
                        "scene_desc": row["scene_desc"],
                        "type_score": row["type_score"],
                        "better_model": row["better_model"],
                        "unet_iou": row["unet_iou"],
                        "stanet_iou": row["stanet_iou"],
                        "iou_gap": row["iou_gap"],
                        "gt_ratio": row["gt_ratio"],
                        "unet_fp_rate": row["unet_fp_rate"],
                        "stanet_fp_rate": row["stanet_fp_rate"],
                        "unet_fn_rate": row["unet_fn_rate"],
                        "stanet_fn_rate": row["stanet_fn_rate"],
                        "unet_edge_f1": row["unet_edge_f1"],
                        "stanet_edge_f1": row["stanet_edge_f1"],
                    }
                )


def _write_group_json(grouped: Dict[str, List[dict]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)


def _print_preview(grouped: Dict[str, List[dict]], preview: int) -> None:
    for t in [TYPE_CLEAR, TYPE_FP, TYPE_FN, TYPE_EDGE, TYPE_UNKNOWN]:
        rows = grouped.get(t, [])
        print(f"\n[{t}] candidates: {len(rows)}")
        for row in rows[:preview]:
            print(
                f"  - {row['name']}: type_score={row['type_score']:.4f}, "
                f"UnetIoU={row['unet_iou']:.4f}, STANetIoU={row['stanet_iou']:.4f}, "
                f"better={row['better_model']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Semi-automatic representative validation sample selection: "
            "pair Unet/STANet/GT masks by same filename stem, classify into 4 scenes, output Top-K."
        )
    )
    parser.add_argument(
        "--unet-dir",
        type=Path,
        default=_choose_existing(["valUntres", "resUnet"]),
        help="Unet prediction folder (default: existing one among valUntres/resUnet)",
    )
    parser.add_argument(
        "--stanet-dir",
        type=Path,
        default=_choose_existing(["valSTANetres", "resSTANet"]),
        help="STANet prediction folder (default: existing one among valSTANetres/resSTANet)",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=_choose_existing(
            [
                "D:/DeepLearning/dataset/LEVIR/val_cropped/label",
                "val_cropped/label",
                "label",
            ]
        ),
        help="Validation GT label folder",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-K candidates per type")
    parser.add_argument(
        "--all-csv",
        type=Path,
        default=Path("choose_val_all_metrics.csv"),
        help="CSV path for metrics of all paired samples",
    )
    parser.add_argument(
        "--topk-csv",
        type=Path,
        default=Path("choose_val_topk.csv"),
        help="CSV path for Top-K candidates grouped by representative type",
    )
    parser.add_argument(
        "--topk-json",
        type=Path,
        default=Path("choose_val_topk.json"),
        help="JSON path for Top-K candidates grouped by representative type",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Print first N candidates per type in terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Unet dir:", args.unet_dir)
    print("STANet dir:", args.stanet_dir)
    print("Label dir:", args.label_dir)

    all_rows, grouped, filtered_out = analyze(
        unet_dir=args.unet_dir,
        stanet_dir=args.stanet_dir,
        label_dir=args.label_dir,
        topk=args.topk,
    )

    _write_all_csv(all_rows, args.all_csv)
    _write_group_csv(grouped, args.topk_csv)
    _write_group_json(grouped, args.topk_json)

    print(f"\nValid paired samples (after filtering): {len(all_rows)}")
    print(f"Filtered out catastrophic samples: {filtered_out}")
    print("All metrics CSV:", args.all_csv)
    print("Top-K CSV:", args.topk_csv)
    print("Top-K JSON:", args.topk_json)

    _print_preview(grouped, preview=args.preview)


if __name__ == "__main__":
    main()
