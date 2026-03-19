from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


SUPPORTED_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def _choose_existing(candidates: list[str]) -> Path:
	"""从候选路径中返回第一个存在的目录，否则返回第一个候选项。

	这样可以兼容不同机器上的目录命名差异。
	"""
	for item in candidates:
		p = Path(item)
		if p.exists():
			return p
	return Path(candidates[0])


def _list_stems(folder: Path) -> Dict[str, Path]:
	"""扫描文件夹中的图像文件，返回 {文件名主干: 完整路径} 映射。

	使用 stem 作为键，可以方便地和标签按同名文件配对。
	"""
	items: Dict[str, Path] = {}
	if not folder.exists():
		return items
	for p in folder.iterdir():
		if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
			items[p.stem] = p
	return items


def _load_binary_mask(path: Path) -> np.ndarray:
	"""读取掩膜并统一转换为布尔数组。

	支持以下输入格式：
	- 二值图 {0, 1}
	- 二值图 {0, 255}
	- 非 uint8 掩膜（先归一化到 0-255 再阈值化）
	"""
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

	# 同时兼容 {0,1} 与 {0,255} 两种二值标注格式。
	return img > (127 if int(img.max()) > 1 else 0)


def _build_confusion_color_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
	"""根据预测与真值构建混淆可视化彩色图。

	颜色约定（OpenCV 使用 BGR）：
	- TP：白色
	- TN：黑色
	- FP：红色
	- FN：绿色
	"""
	if pred.shape != gt.shape:
		raise RuntimeError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

	# OpenCV 写图使用 BGR 排列，下面按像素类别上色。
	color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

	tp = np.logical_and(pred, gt)
	tn = np.logical_and(~pred, ~gt)
	fp = np.logical_and(pred, ~gt)
	fn = np.logical_and(~pred, gt)

	color[tn] = (0, 0, 0)
	color[tp] = (255, 255, 255)
	color[fp] = (0, 0, 255)
	color[fn] = (0, 255, 0)
	return color


def _ensure_dirs(base_dir: Path) -> Path:
	"""确保输出目录存在，并返回单模型输出路径。

	当前脚本仅处理 UnetHybrid 结果，不再生成模型对比拼接图。
	"""
	hybrid_out = base_dir / "unet_hybrid"
	hybrid_out.mkdir(parents=True, exist_ok=True)
	return hybrid_out


def generate_all(unet_hybrid_dir: Path, label_dir: Path, out_dir: Path) -> None:
	"""仅针对 UnetHybrid 预测结果生成 TP/TN/FP/FN 彩图。

	预测图与标签图通过相同文件名主干（stem）进行配对。
	"""
	hybrid_map = _list_stems(unet_hybrid_dir)
	label_map = _list_stems(label_dir)

	common = sorted(set(hybrid_map) & set(label_map))
	if not common:
		raise RuntimeError("No paired names found across UnetHybrid/label folders")

	hybrid_out = _ensure_dirs(out_dir)

	total = len(common)
	print(f"Paired samples found: {total}")

	for idx, name in enumerate(common, start=1):
		# 读取 GT 与预测图，并统一成布尔掩膜。
		gt = _load_binary_mask(label_map[name])
		hybrid_pred = _load_binary_mask(hybrid_map[name])

		# 生成并保存该样本的混淆彩色图。
		hybrid_color = _build_confusion_color_map(hybrid_pred, gt)

		cv2.imwrite(str(hybrid_out / f"{name}.png"), hybrid_color)

		if idx % 100 == 0 or idx == total:
			print(f"Processed {idx}/{total}")

	print("Done.")
	print("UnetHybrid color maps:", hybrid_out)


def parse_args() -> argparse.Namespace:
	"""解析命令行参数（单模型 UnetHybrid 版本）。"""
	parser = argparse.ArgumentParser(
		description=(
			"针对 UnetHybrid 预测结果与验证集标签，生成 TP/TN/FP/FN 四色可视化图。"
		)
	)
	parser.add_argument(
		"--unet-hybrid-dir",
		type=Path,
		default=_choose_existing(["resUnetvalHybrid", "UnetHybrid"]),
		help="UnetHybrid 预测结果目录（默认优先 resUnetvalHybrid）",
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
		help="验证集标签目录",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path(__file__).resolve().parent / Path(__file__).stem,
		help="输出目录（默认：与当前脚本同名文件夹）",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print("UnetHybrid dir:", args.unet_hybrid_dir)
	print("Label dir:", args.label_dir)
	print("Output dir:", args.out_dir)

	generate_all(
		unet_hybrid_dir=args.unet_hybrid_dir,
		label_dir=args.label_dir,
		out_dir=args.out_dir,
	)


if __name__ == "__main__":
	main()
