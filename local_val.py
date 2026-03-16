from pathlib import Path

import torch
from torch.utils.data import DataLoader

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset
import os
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 验证集路径
VAL_ROOT = Path("D:/DeepLearning/dataset/LEVIR/val_cropped")
VAL_A_PATH = VAL_ROOT / "A"
VAL_B_PATH = VAL_ROOT / "B"
VAL_LABEL_PATH = VAL_ROOT / "label"

# 模型权重路径
UNET_WEIGHT_PATH = Path("D:/DeepLearning/change_detection.pytorch/best_model_Unet.pth")
STANET_WEIGHT_PATH = Path("D:/DeepLearning/change_detection.pytorch/best_model_STANet.pth")


def get_val_loader(batch_size=1, num_workers=0):
    val_dataset = LEVIR_CD_Dataset(
        str(VAL_ROOT),
        sub_dir_1="A",
        sub_dir_2="B",
        img_suffix=".png",
        ann_dir=str(VAL_LABEL_PATH),
        debug=False,
        test_mode=True,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _clean_state_dict(state_dict):
    # 兼容 DataParallel 保存时的 "module." 前缀
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_model(weight_path, arch):
    checkpoint = torch.load(str(weight_path), map_location=DEVICE)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model = cdp.create_model(
            arch,
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            siam_encoder=True,
            fusion_form="concat",
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _clean_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


def evaluate_model(model, dataloader, save_dir):
    metric_fns = [
        cdp.utils.metrics.Fscore(activation="argmax2d"),
        cdp.utils.metrics.Precision(activation="argmax2d"),
        cdp.utils.metrics.Recall(activation="argmax2d"),
        cdp.utils.metrics.IoU(activation="argmax2d"),
    ]
    results = {m.__name__: 0.0 for m in metric_fns}
    n = 0

    os.makedirs(save_dir, exist_ok=True)

    with torch.inference_mode():
        for x1, x2, y, _filename in dataloader:
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).long()

            pred = model(x1, x2)

            # Save predictions
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype("uint8")
            save_path = os.path.join(save_dir, f"{_filename[0]}")
            cv2.imwrite(save_path, pred_mask * 255)

            for metric_fn in metric_fns:
                value = metric_fn(pred, y).detach().cpu().item()
                results[metric_fn.__name__] += value
            n += 1

    if n > 0:
        for key in results:
            results[key] /= n
    return results


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"VAL A path: {VAL_A_PATH}")
    print(f"VAL B path: {VAL_B_PATH}")
    print(f"VAL label path: {VAL_LABEL_PATH}")

    val_loader = get_val_loader(batch_size=1, num_workers=0)

    print(f"\nLoading Unet weights: {UNET_WEIGHT_PATH}")
    unet_model = load_model(UNET_WEIGHT_PATH, arch="Unet")
    unet_save_dir = "valUntres"
    unet_metrics = evaluate_model(unet_model, val_loader, unet_save_dir)
    print("Unet metrics:", unet_metrics)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nLoading STANet weights: {STANET_WEIGHT_PATH}")
    stanet_model = load_model(STANET_WEIGHT_PATH, arch="STANet")
    stanet_save_dir = "valSTANetres"
    stanet_metrics = evaluate_model(stanet_model, val_loader, stanet_save_dir)
    print("STANet metrics:", stanet_metrics)