import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndi  # 终极抢救：孔洞填充
from tqdm import tqdm

def process_and_evaluate_v2():
    # ================= 1. 绝对路径核对区 =================
    dir_a = r"D:\DeepLearning\dataset\LEVIR\val_cropped\A"
    dir_b = r"D:\DeepLearning\dataset\LEVIR\val_cropped\B"
    dir_gt = r"D:\DeepLearning\dataset\LEVIR\val_cropped\label" # 真实标签
    dir_pred = r"D:\DeepLearning\change_detection.pytorch\valUntres" # 基线狗啃图
    dir_out = r"D:\DeepLearning\change_detection.pytorch\resval_refined" # 存新图
    
    os.makedirs(dir_out, exist_ok=True)
    # =====================================================

    files = [f for f in os.listdir(dir_pred) if f.endswith('.png')]
    print(f"检测到 {len(files)} 张图片，开始 [OBIA精炼 V2 (填洞版) + 全局算分] 终极流水线...")

    # 初始化全局指标累加器
    base_TP, base_FP, base_FN = 0, 0, 0
    obia_TP, obia_FP, obia_FN = 0, 0, 0

    for name in tqdm(files):
        path_a = os.path.join(dir_a, name)
        path_b = os.path.join(dir_b, name)
        path_pred = os.path.join(dir_pred, name)
        path_gt = os.path.join(dir_gt, name)

        if not os.path.exists(path_gt): 
            continue 

        # 读图与二值化
        img_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)
        pred_mask = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)

        pred_bin = (pred_mask > 127).astype(np.uint8)
        gt_bin = (gt_mask > 127).astype(np.uint8)

        # --------------------------------------------------
        # 环节 A：基线 (Baseline) 得分累加
        # --------------------------------------------------
        base_TP += np.sum((pred_bin == 1) & (gt_bin == 1))
        base_FP += np.sum((pred_bin == 1) & (gt_bin == 0))
        base_FN += np.sum((pred_bin == 0) & (gt_bin == 1))

        # --------------------------------------------------
        # 环节 B：进阶版 OBIA 超像素精炼 (收缩边缘 + 填补孔洞)
        # --------------------------------------------------
        img_concat = np.concatenate((img_a, img_b), axis=-1)
        # compactness=10：稍微增加规整度，防止边缘像融化的冰淇淋
        segments = slic(img_concat, n_segments=300, compactness=10, channel_axis=-1)
        
        refined_mask = np.zeros_like(pred_mask)
        for seg_id in np.unique(segments):
            block = (segments == seg_id)
            # 【止血刀】：阈值拉高到 50%，坚决不让背景沾光，保住 Precision
            if np.sum(pred_mask[block] == 255) / np.sum(block) > 0.5: 
                refined_mask[block] = 255
                
        # 转为布尔型供形态学使用
        bool_mask = (refined_mask == 255)

        # 【缝合刀】：拓扑孔洞填充！把建筑物内部的黑洞完美缝死，保住 Recall
        filled_mask = ndi.binary_fill_holes(bool_mask)
        
        # 【去噪刀】：抹除 20 像素以下的孤立噪点
        refined_bin = remove_small_objects(filled_mask, min_size=20)
        
        final_result = (refined_bin * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dir_out, name), final_result)

        # --------------------------------------------------
        # 环节 C：优化后 (OBIA) 得分累加
        # --------------------------------------------------
        obia_bin = refined_bin.astype(np.uint8)
        obia_TP += np.sum((obia_bin == 1) & (gt_bin == 1))
        obia_FP += np.sum((obia_bin == 1) & (gt_bin == 0))
        obia_FN += np.sum((obia_bin == 0) & (gt_bin == 1))

    # ================= 全局指标结算 =================
    eps = 1e-8
    bp = base_TP / (base_TP + base_FP + eps)
    br = base_TP / (base_TP + base_FN + eps)
    bf = 2 * (bp * br) / (bp + br + eps)
    
    op = obia_TP / (obia_TP + obia_FP + eps)
    ov = obia_TP / (obia_TP + obia_FN + eps)
    of = 2 * (op * ov) / (op + ov + eps)

    print("\n" + "="*50)
    print("🔥 1024 张全量数据终极战况 (V2 填洞版) 🔥")
    print("="*50)
    print(f"[原始基线] Precision: {bp*100:.2f}%, Recall: {br*100:.2f}%, F1-Score: {bf*100:.2f}%")
    print(f"[OBIA优化] Precision: {op*100:.2f}%, Recall: {ov*100:.2f}%, F1-Score: {of*100:.2f}%")
    print("-" * 50)
    
    f1_diff = (of - bf) * 100
    if f1_diff > 0:
        print(f"🎉 结论：抢救成功！全局 F1-Score 上涨了 {f1_diff:.2f}%！")
    else:
        print(f"⚠️ 结论：依然下跌了 {abs(f1_diff):.2f}%。模型原始输出可能已经触及像素级天花板。")

if __name__ == '__main__':
    process_and_evaluate_v2()