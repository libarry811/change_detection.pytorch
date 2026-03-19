import os
import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from tqdm import tqdm

def process_and_evaluate_morph():
    # ================= 1. 绝对路径核对区 =================
    dir_gt = r"D:\DeepLearning\dataset\LEVIR\val_cropped\label"
    dir_pred = r"D:\DeepLearning\change_detection.pytorch\valUntres"
    # 新建一个专属文件夹存放纯形态学精炼结果
    dir_out = r"D:\DeepLearning\change_detection.pytorch\resval_morph" 
    
    os.makedirs(dir_out, exist_ok=True)
    # =====================================================

    files = [f for f in os.listdir(dir_pred) if f.endswith('.png')]
    print(f"检测到 {len(files)} 张图片，开始执行 [纯形态学精炼] 稳如老狗版...")

    base_TP, base_FP, base_FN = 0, 0, 0
    morph_TP, morph_FP, morph_FN = 0, 0, 0

    # 【核心武器】：定义极其保守的 3x3 矩形结构元
    # 绝对不能用 5x5，否则会把靠得近的两栋楼强行粘连在一起！
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for name in tqdm(files):
        path_pred = os.path.join(dir_pred, name)
        path_gt = os.path.join(dir_gt, name)

        if not os.path.exists(path_gt): 
            continue 

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
        # 环节 B：纯形态学操作 (极简且致命)
        # --------------------------------------------------
        # 1. 闭运算：先膨胀再腐蚀，填补内部黑洞，平滑锯齿边缘
        closed_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. 面积过滤：干掉所有面积小于 20 像素的孤立白点噪点
        bool_mask = (closed_mask > 127)
        cleaned_mask = remove_small_objects(bool_mask, min_size=20)
        
        final_result = (cleaned_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dir_out, name), final_result)

        # --------------------------------------------------
        # 环节 C：优化后 (纯形态学版本) 得分累加
        # --------------------------------------------------
        morph_bin = (final_result > 127).astype(np.uint8)
        morph_TP += np.sum((morph_bin == 1) & (gt_bin == 1))
        morph_FP += np.sum((morph_bin == 1) & (gt_bin == 0))
        morph_FN += np.sum((morph_bin == 0) & (gt_bin == 1))

    # ================= 全局指标结算 =================
    eps = 1e-8
    bp = base_TP / (base_TP + base_FP + eps)
    br = base_TP / (base_TP + base_FN + eps)
    bf = 2 * (bp * br) / (bp + br + eps)
    
    mp = morph_TP / (morph_TP + morph_FP + eps)
    mr = morph_TP / (morph_TP + morph_FN + eps)
    mf = 2 * (mp * mr) / (mp + mr + eps)

    print("\n" + "="*50)
    print("🔥 1024 张全量数据：纯形态学战况 🔥")
    print("="*50)
    print(f"[原始基线] Precision: {bp*100:.2f}%, Recall: {br*100:.2f}%, F1-Score: {bf*100:.2f}%")
    print(f"[形态学版] Precision: {mp*100:.2f}%, Recall: {mr*100:.2f}%, F1-Score: {mf*100:.2f}%")
    print("-" * 50)
    
    f1_diff = (mf - bf) * 100
    if f1_diff > 0:
        print(f"🎉 终极胜利！极简主义干翻了超像素，F1-Score 微涨了 {f1_diff:.2f}%！")
    elif f1_diff == 0:
         print(f"⚖️ 打平了。说明基线极其完美，不需要任何画蛇添足。")
    else:
        print(f"⚠️ 依然微跌了 {abs(f1_diff):.2f}%。放弃后处理，直接去写论文！")

if __name__ == '__main__':
    process_and_evaluate_morph()