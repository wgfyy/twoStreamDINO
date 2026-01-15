
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def collect_images(data_dir: Path, exts) -> list[Path]:
    """递归收集指定后缀的图片路径."""
    exts = tuple(exts)
    return sorted([p for p in data_dir.rglob('*') if p.suffix.lower() in exts and p.is_file()])


def compute_mean_std(data_dir: Path, exts: list[str]):
    """
    计算数据集图像通道的 mean 和 std（递归子目录）
    exts: 允许的扩展名列表（含点）
    """
    print(f"正在扫描目录: {data_dir} ...")
    img_paths = collect_images(data_dir, exts)

    if len(img_paths) == 0:
        print(f"错误: 在 {data_dir} 下未找到扩展名为 {exts} 的文件")
        return

    print(f"找到 {len(img_paths)} 张图片，开始计算...")

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    for path in tqdm(img_paths):
        try:
            img = Image.open(path).convert('RGB')
            pixels = np.array(img).reshape(-1, 3).astype(np.float64)

            pixel_count += pixels.shape[0]
            pixel_sum += np.sum(pixels, axis=0)
            pixel_sq_sum += np.sum(pixels ** 2, axis=0)

        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

    mean = pixel_sum / pixel_count
    var = (pixel_sq_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    print("\n" + "=" * 40)
    print(f"计算完成 (共 {len(img_paths)} 张图片)")
    print("注意: 结果顺序为 RGB")
    print("-" * 20)
    print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("=" * 40)

    return mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Image Mean and Std')
    parser.add_argument('--dir', type=Path, required=True, help='图像根目录，递归遍历')
    parser.add_argument('--ext', type=str, default='.png,.jpg,.jpeg', help='逗号分隔扩展名列表，带点')
    args = parser.parse_args()

    ext_list = [e.strip().lower() for e in args.ext.split(',') if e.strip()]
    compute_mean_std(args.dir, ext_list)
