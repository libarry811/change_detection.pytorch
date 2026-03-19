from pathlib import Path

import torch
from torch.utils.data import DataLoader

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset
import os
import cv2

# 运行设备：优先使用 GPU，加速前向推理
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 验证集路径
VAL_ROOT = Path("D:/DeepLearning/dataset/LEVIR/val_cropped")
VAL_A_PATH = VAL_ROOT / "A"
VAL_B_PATH = VAL_ROOT / "B"
VAL_LABEL_PATH = VAL_ROOT / "label"

# 模型权重路径
# 1) 原始 Unet（交叉熵训练）
UNET_WEIGHT_BASELINE_PATH = Path("D:/DeepLearning/change_detection.pytorch/best_model_Unet.pth")
# 2) 新版 Unet（引入 HybridLoss: CrossEntropy + Dice）
UNET_WEIGHT_HYBRID_PATH = Path("D:/DeepLearning/change_detection.pytorch/best_model.pth")


def get_val_loader(batch_size=1, num_workers=0):
    # 这里使用 test_mode=True：表示评估/推理变换（无随机裁剪增强），
    # 不是“无标签测试集”含义；ann_dir 仍然会提供 GT，用于验证指标计算。
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
    # 兼容两类权重：
    # 1) 直接保存的完整 torch.nn.Module
    # 2) state_dict 或包含 state_dict 的 checkpoint 字典
    checkpoint = torch.load(str(weight_path), map_location=DEVICE)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # 按验证脚本固定配置重建模型结构，再加载参数
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


def count_model_params(model):
    """统计模型参数量（总参数量 + 可训练参数量）。"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _format_metrics(metrics):
    """统一格式化输出，便于对比阅读。

    这里约定 metrics 字典中使用以下标准键名：
    - precision
    - recall
    - fscore
    - iou_score
    """
    return {
        "Precision": metrics.get("precision", 0.0),
        "Recall": metrics.get("recall", 0.0),
        "F1-score": metrics.get("fscore", 0.0),
        "IoU": metrics.get("iou_score", 0.0),
    }


def _print_model_report(title, metrics, total_params, trainable_params, save_dir):
    """打印单模型评估报告。"""
    fm = _format_metrics(metrics)
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"Prediction folder : {save_dir}")
    print(f"Precision         : {fm['Precision']:.6f}")
    print(f"Recall            : {fm['Recall']:.6f}")
    print(f"F1-score          : {fm['F1-score']:.6f}")
    print(f"IoU               : {fm['IoU']:.6f}")
    print(f"Params (total)    : {total_params:,}")
    print(f"Params (trainable): {trainable_params:,}")
    # 同时打印基础统计量，方便后续复现实验核查
    if "tp" in metrics:
        print(f"TP / FP / FN / TN : {metrics['tp']} / {metrics['fp']} / {metrics['fn']} / {metrics['tn']}")
    if "num_images" in metrics:
        print(f"Num images        : {metrics['num_images']}")


def _print_compare_report(base_metrics, hybrid_metrics, base_params, hybrid_params):
    """打印两组结果差值（Hybrid - Baseline），正值表示提升。"""
    base_fm = _format_metrics(base_metrics)
    hybrid_fm = _format_metrics(hybrid_metrics)

    print(f"\n{'=' * 70}")
    print("Comparison: HybridLoss-Unet - Baseline-Unet")
    print(f"{'=' * 70}")
    print(f"Delta Precision : {hybrid_fm['Precision'] - base_fm['Precision']:+.6f}")
    print(f"Delta Recall    : {hybrid_fm['Recall'] - base_fm['Recall']:+.6f}")
    print(f"Delta F1-score  : {hybrid_fm['F1-score'] - base_fm['F1-score']:+.6f}")
    print(f"Delta IoU       : {hybrid_fm['IoU'] - base_fm['IoU']:+.6f}")
    print(f"Delta Params    : {hybrid_params[0] - base_params[0]:+,}")


def evaluate_model(model, dataloader, save_dir):
    """按“全验证集累计混淆矩阵”口径评估模型（第 2 种）。

    评估逻辑（推荐用于变化检测）：
    1) 遍历全部验证图像，把 TP/FP/FN/TN 在像素级全量累加。
    2) 在遍历结束后，再统一用累计值计算 Precision/Recall/F1/IoU。

    这种方式属于全局统计（micro/global），相比“逐图算分再平均”，
    对不同大小变化区域更稳健，也更常见于变化检测任务报告。
    """
    # 基础统计量初始化（全量累计）
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    num_images = 0

    # 防止除零
    eps = 1e-8

    os.makedirs(save_dir, exist_ok=True)

    with torch.inference_mode():
        for x1, x2, y, _filename in dataloader:
            # 将输入与标签搬到目标设备
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).long()

            # 前向推理得到 logits，随后取 argmax 得到类别标签
            pred = model(x1, x2)
            pred_cls = torch.argmax(pred, dim=1).long()  # [B, H, W]

            # 兼容不同标签维度：
            # 常见是 [B, H, W]；若为 [B, 1, H, W] 则压缩通道维。
            if y.dim() == 4 and y.size(1) == 1:
                gt_cls = y.squeeze(1)
            elif y.dim() == 4 and y.size(1) > 1:
                # 若标签是 one-hot，转成类别索引
                gt_cls = torch.argmax(y, dim=1)
            else:
                gt_cls = y

            # 按样本保存预测结果，文件名与数据集原图对齐
            # 变化检测任务默认前景（变化）类别为 1，保存为 0/255 二值图。
            batch_size = pred_cls.shape[0]
            for i in range(batch_size):
                pred_mask = pred_cls[i].detach().cpu().numpy().astype("uint8")
                save_path = os.path.join(save_dir, f"{_filename[i]}")
                cv2.imwrite(save_path, pred_mask * 255)

            # 变化类（正类）定义为类别 1，做像素级布尔统计
            pred_pos = (pred_cls == 1)
            gt_pos = (gt_cls == 1)

            # 全量累计 TP/FP/FN/TN
            tp += torch.logical_and(pred_pos, gt_pos).sum().item()
            fp += torch.logical_and(pred_pos, torch.logical_not(gt_pos)).sum().item()
            fn += torch.logical_and(torch.logical_not(pred_pos), gt_pos).sum().item()
            tn += torch.logical_and(torch.logical_not(pred_pos), torch.logical_not(gt_pos)).sum().item()
            num_images += batch_size

    # 统一由全局累计统计量计算指标（第 2 种口径）
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "fscore": f1,
        "iou_score": iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "num_images": int(num_images),
    }


if __name__ == "__main__":
    # 运行信息打印：用于确认路径和设备配置无误
    print(f"Using device: {DEVICE}")
    print(f"VAL A path: {VAL_A_PATH}")
    print(f"VAL B path: {VAL_B_PATH}")
    print(f"VAL label path: {VAL_LABEL_PATH}")

    # 构建验证集 DataLoader
    val_loader = get_val_loader(batch_size=1, num_workers=0)

    # 评估前先检查权重文件是否存在，避免中途报错
    if not UNET_WEIGHT_BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline weight not found: {UNET_WEIGHT_BASELINE_PATH}")
    if not UNET_WEIGHT_HYBRID_PATH.exists():
        raise FileNotFoundError(f"Hybrid weight not found: {UNET_WEIGHT_HYBRID_PATH}")

    # 评估 1：原始 Unet（best_model_Unet.pth）
    print(f"\nLoading baseline Unet weights: {UNET_WEIGHT_BASELINE_PATH}")
    unet_model = load_model(UNET_WEIGHT_BASELINE_PATH, arch="Unet")
    unet_total_params, unet_trainable_params = count_model_params(unet_model)
    unet_save_dir = "resUnetval"
    unet_metrics = evaluate_model(unet_model, val_loader, unet_save_dir)
    _print_model_report(
        title="Baseline Unet (best_model_Unet.pth)",
        metrics=unet_metrics,
        total_params=unet_total_params,
        trainable_params=unet_trainable_params,
        save_dir=unet_save_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 评估 2：HybridLoss Unet（best_model.pth）
    print(f"\nLoading hybrid-loss Unet weights: {UNET_WEIGHT_HYBRID_PATH}")
    unet_hybrid_model = load_model(UNET_WEIGHT_HYBRID_PATH, arch="Unet")
    hybrid_total_params, hybrid_trainable_params = count_model_params(unet_hybrid_model)
    unet_hybrid_save_dir = "resUnetvalHybrid"
    unet_hybrid_metrics = evaluate_model(unet_hybrid_model, val_loader, unet_hybrid_save_dir)
    _print_model_report(
        title="HybridLoss Unet (best_model.pth)",
        metrics=unet_hybrid_metrics,
        total_params=hybrid_total_params,
        trainable_params=hybrid_trainable_params,
        save_dir=unet_hybrid_save_dir,
    )

    # 汇总对比：帮助判断优化方向
    _print_compare_report(
        base_metrics=unet_metrics,
        hybrid_metrics=unet_hybrid_metrics,
        base_params=(unet_total_params, unet_trainable_params),
        hybrid_params=(hybrid_total_params, hybrid_trainable_params),
    )

    # print(f"\nLoading STANet weights: {STANET_WEIGHT_PATH}")
    # stanet_model = load_model(STANET_WEIGHT_PATH, arch="STANet")
    # stanet_save_dir = "valSTANetres"
    # stanet_metrics = evaluate_model(stanet_model, val_loader, stanet_save_dir)
    # print("STANet metrics:", stanet_metrics)