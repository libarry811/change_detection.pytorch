from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


SUPPORTED_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def _choose_existing(candidates: list[str]) -> Path:
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

	# Supports both {0,1} and {0,255} masks.
	return img > (127 if int(img.max()) > 1 else 0)


def _build_confusion_color_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
	if pred.shape != gt.shape:
		raise RuntimeError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

	# BGR colors for OpenCV write:
	# TP: white, TN: black, FP: red, FN: green
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


def _stack_compare(unet_map: np.ndarray, stanet_map: np.ndarray) -> np.ndarray:
	if unet_map.shape != stanet_map.shape:
		raise RuntimeError("Unet and STANet map shape mismatch in compare view")

	sep = np.full((unet_map.shape[0], 12, 3), 30, dtype=np.uint8)
	return np.concatenate([unet_map, sep, stanet_map], axis=1)


def _ensure_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
	unet_out = base_dir / "unet"
	stanet_out = base_dir / "stanet"
	compare_out = base_dir / "compare"
	unet_out.mkdir(parents=True, exist_ok=True)
	stanet_out.mkdir(parents=True, exist_ok=True)
	compare_out.mkdir(parents=True, exist_ok=True)
	return unet_out, stanet_out, compare_out


def generate_all(unet_dir: Path, stanet_dir: Path, label_dir: Path, out_dir: Path) -> None:
	unet_map = _list_stems(unet_dir)
	stanet_map = _list_stems(stanet_dir)
	label_map = _list_stems(label_dir)

	common = sorted(set(unet_map) & set(stanet_map) & set(label_map))
	if not common:
		raise RuntimeError("No paired names found across Unet/STANet/label folders")

	unet_out, stanet_out, compare_out = _ensure_dirs(out_dir)

	total = len(common)
	print(f"Paired samples found: {total}")

	for idx, name in enumerate(common, start=1):
		gt = _load_binary_mask(label_map[name])
		unet_pred = _load_binary_mask(unet_map[name])
		stanet_pred = _load_binary_mask(stanet_map[name])

		unet_color = _build_confusion_color_map(unet_pred, gt)
		stanet_color = _build_confusion_color_map(stanet_pred, gt)
		compare = _stack_compare(unet_color, stanet_color)

		cv2.imwrite(str(unet_out / f"{name}.png"), unet_color)
		cv2.imwrite(str(stanet_out / f"{name}.png"), stanet_color)
		cv2.imwrite(str(compare_out / f"{name}.png"), compare)

		if idx % 100 == 0 or idx == total:
			print(f"Processed {idx}/{total}")

	print("Done.")
	print("Unet color maps:", unet_out)
	print("STANet color maps:", stanet_out)
	print("Compare maps:", compare_out)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate TP/TN/FP/FN four-color maps for Unet and STANet predictions against validation labels."
		)
	)
	parser.add_argument(
		"--unet-dir",
		type=Path,
		default=_choose_existing(["valUntres", "resUnet"]),
		help="Unet prediction directory",
	)
	parser.add_argument(
		"--stanet-dir",
		type=Path,
		default=_choose_existing(["valSTANetres", "resSTANet"]),
		help="STANet prediction directory",
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
		help="Validation GT label directory",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path(__file__).resolve().parent / Path(__file__).stem,
		help="Output directory (default: folder with same name as this file)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print("Unet dir:", args.unet_dir)
	print("STANet dir:", args.stanet_dir)
	print("Label dir:", args.label_dir)
	print("Output dir:", args.out_dir)

	generate_all(
		unet_dir=args.unet_dir,
		stanet_dir=args.stanet_dir,
		label_dir=args.label_dir,
		out_dir=args.out_dir,
	)


if __name__ == "__main__":
	main()
