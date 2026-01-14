
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def compute_mean_std(data_dir, ext='.png'):
    """
    计算数据集图像通道的 mean 和 std
    """
    print(f"正在扫描目录: {data_dir} ...")
    img_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(ext)]
    
    if len(img_paths) == 0:
        print(f"错误: 在 {data_dir} 下未找到扩展名为 {ext} 的文件")
        return

    print(f"找到 {len(img_paths)} 张图片，开始计算...")

    # 初始化累加器
    pixel_sum = np.zeros(3) 
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    for path in tqdm(img_paths):
        try:
            img = Image.open(path).convert('RGB')
            # PIL 读取的是 RGB 顺序. 转换为float64防止溢出，尤其是平方计算
            pixels = np.array(img).reshape(-1, 3).astype(np.float64)
            
            pixel_count += pixels.shape[0]
            pixel_sum += np.sum(pixels, axis=0)
            pixel_sq_sum += np.sum(pixels ** 2, axis=0)
            
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

    # 计算全局 Mean
    mean = pixel_sum / pixel_count
    
    # 计算全局 Std
    var = (pixel_sq_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    print("\n" + "="*40)
    print(f"计算完成 (共 {len(img_paths)} 张图片)")
    print("注意: 结果顺序为 RGB")
    print("-" * 20)
    print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("="*40)
    
    return mean, std

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Image Mean and Std')
    parser.add_argument('--dir', type=str, required=True, help='SAR images directory path')
    parser.add_argument('--ext', type=str, default='.png', help='Image extension (e.g., .png, .jpg)')
    args = parser.parse_args()
    
    compute_mean_std(args.dir, args.ext)
