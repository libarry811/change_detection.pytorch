import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.morphology import remove_small_objects

def single_image_obia(path_a, path_b, path_pred, path_out):
    print("1. 正在读取图像...")
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    # 读取你深度学习跑出来的初版预测图 (必须是单通道灰度模式读取)
    pred_mask = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
    
    # 转 RGB 供 SLIC 使用
    img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    print("2. 正在进行 SLIC 超像素分割...")
    img_concat = np.concatenate((img_a_rgb, img_b_rgb), axis=-1)
    # 按照我们之前测试的完美参数：compactness=1 极其贴合边缘
    segments = slic(img_concat, n_segments=300, compactness=1, channel_axis=-1)

    print("3. 正在进行对象投票 (Object Voting)...")
    # 创建一张全黑的新画布，准备把合格的块涂白
    refined_mask = np.zeros_like(pred_mask)
    
    # 遍历那 300 个网格块
    for seg_id in np.unique(segments):
        # 找到属于当前这 1 块的所有像素位置 (布尔索引)
        block_idx = (segments == seg_id)
        
        # 核心：统计这个块里，初版预测图(pred_mask)给出的白点(255)占多少比例
        change_pixels = np.sum(pred_mask[block_idx] == 255)
        total_pixels = np.sum(block_idx)
        change_ratio = change_pixels / total_pixels
        
        # 判定：如果超过 30%，说明这个地块真变了，把新画布上对应的这整个块全部涂白！
        if change_ratio > 0.3:
            refined_mask[block_idx] = 255

    print("4. 正在进行形态学去噪 (消除孤立碎点)...")
    # remove_small_objects 需要输入 True/False 的布尔型数组
    bool_mask = (refined_mask == 255)
    # 直接干掉面积小于 20 像素的白点
    clean_bool_mask = remove_small_objects(bool_mask, min_size=30)
    
    # 把 True/False 还原回 255 和 0 的图像格式
    final_mask = (clean_bool_mask * 255).astype(np.uint8)

    print(f"5. 处理完成！保存精炼后的图像至: {path_out}")
    cv2.imwrite(path_out, final_mask)

if __name__ == '__main__':
    # --------- 替换为你挑出的一组图片的真实绝对路径 ---------
    # 原图 A 和 B
    test_A = r"D:\DeepLearning\dataset\LEVIR\val_cropped\A\val_25_0_768.png"
    test_B = r"D:\DeepLearning\dataset\LEVIR\val_cropped\B\val_25_0_768.png"
    
    # 深度学习跑出来的、带锯齿的初版预测图
    test_pred = r"D:\DeepLearning\change_detection.pytorch\valUntres\val_25_0_768.png"
    
    # 你想要保存精炼结果的位置
    test_out = r"D:\DeepLearning\change_detection.pytorch\test_refined_result.png"
    # --------------------------------------------------------
    
    single_image_obia(test_A, test_B, test_pred, test_out)