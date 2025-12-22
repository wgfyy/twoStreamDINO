#!/usr/bin/env python
"""
验证双模态数据集配置是否正确。
使用方法:
    python tools/verify_dual_modal_dataset.py configs/dino/dual_stream_dino_r50_dota5cls.py
"""

import argparse
import os
import os.path as osp
import json

from mmengine.config import Config
from mmengine.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='验证双模态数据集')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--show-samples', type=int, default=3, help='显示样本数量')
    return parser.parse_args()


def check_dataset_structure(data_root, ann_file, img_prefix, img2_prefix):
    """检查数据集目录结构"""
    print(f"\n{'='*60}")
    print("检查数据集目录结构")
    print('='*60)
    
    # 检查标注文件
    ann_path = osp.join(data_root, ann_file)
    if osp.exists(ann_path):
        print(f"✓ 标注文件存在: {ann_path}")
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        print(f"  - 图像数量: {len(ann_data.get('images', []))}")
        print(f"  - 标注数量: {len(ann_data.get('annotations', []))}")
        print(f"  - 类别数量: {len(ann_data.get('categories', []))}")
        categories = [c['name'] for c in ann_data.get('categories', [])]
        print(f"  - 类别列表: {categories}")
    else:
        print(f"✗ 标注文件不存在: {ann_path}")
        return False
    
    # 检查图像目录
    img_path = osp.join(data_root, img_prefix)
    if osp.exists(img_path):
        img_files = os.listdir(img_path)
        print(f"✓ 可见光图像目录存在: {img_path}")
        print(f"  - 图像数量: {len(img_files)}")
    else:
        print(f"✗ 可见光图像目录不存在: {img_path}")
        return False
    
    img2_path = osp.join(data_root, img2_prefix)
    if osp.exists(img2_path):
        img2_files = os.listdir(img2_path)
        print(f"✓ SAR图像目录存在: {img2_path}")
        print(f"  - 图像数量: {len(img2_files)}")
    else:
        print(f"✗ SAR图像目录不存在: {img2_path}")
        return False
    
    # 检查图像配对
    img_set = set(img_files)
    img2_set = set(img2_files)
    common = img_set & img2_set
    only_img = img_set - img2_set
    only_img2 = img2_set - img_set
    
    print(f"\n配对检查:")
    print(f"  - 配对图像数: {len(common)}")
    if only_img:
        print(f"  - 仅在可见光目录: {len(only_img)} 个 (如: {list(only_img)[:3]})")
    if only_img2:
        print(f"  - 仅在SAR目录: {len(only_img2)} 个 (如: {list(only_img2)[:3]})")
    
    return True


def test_dataset_loading(cfg, num_samples=3):
    """测试数据集加载"""
    print(f"\n{'='*60}")
    print("测试数据集加载")
    print('='*60)
    
    # 注册数据集
    from mmdet.datasets import DualModalCocoDataset
    
    dataset_cfg = cfg.train_dataloader.dataset
    print(f"数据集类型: {dataset_cfg['type']}")
    
    try:
        dataset = DATASETS.build(dataset_cfg)
        print(f"✓ 数据集构建成功")
        print(f"  - 样本数量: {len(dataset)}")
        
        # 显示几个样本
        print(f"\n前{num_samples}个样本:")
        for i in range(min(num_samples, len(dataset))):
            data = dataset[i]
            print(f"\n  样本 {i}:")
            if 'inputs' in data:
                inputs = data['inputs']
                print(f"    - inputs shape: {inputs.shape}")
            if 'data_samples' in data:
                ds = data['data_samples']
                print(f"    - img_path: {ds.metainfo.get('img_path', 'N/A')}")
                print(f"    - img_path2: {ds.metainfo.get('img_path2', 'N/A')}")
                if hasattr(ds, 'gt_instances'):
                    print(f"    - gt_bboxes: {len(ds.gt_instances.bboxes)} 个")
                    print(f"    - gt_labels: {ds.gt_instances.labels.tolist()}")
        
        return True
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    print(f"\n配置文件: {args.config}")
    
    # 获取数据集配置
    train_cfg = cfg.train_dataloader.dataset
    data_root = train_cfg.get('data_root', '')
    ann_file = train_cfg.get('ann_file', '')
    data_prefix = train_cfg.get('data_prefix', {})
    img_prefix = data_prefix.get('img', '')
    img2_prefix = data_prefix.get('img2', '')
    
    print(f"\n数据集配置:")
    print(f"  - data_root: {data_root}")
    print(f"  - ann_file: {ann_file}")
    print(f"  - img_prefix (可见光): {img_prefix}")
    print(f"  - img2_prefix (SAR): {img2_prefix}")
    
    # 检查目录结构
    structure_ok = check_dataset_structure(data_root, ann_file, img_prefix, img2_prefix)
    
    if structure_ok:
        # 测试数据集加载
        test_dataset_loading(cfg, args.show_samples)
    
    print(f"\n{'='*60}")
    print("验证完成")
    print('='*60)


if __name__ == '__main__':
    main()
