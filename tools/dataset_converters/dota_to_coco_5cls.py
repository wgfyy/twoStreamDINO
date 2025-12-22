#!/usr/bin/env python
"""
DOTA格式转COCO格式转换脚本 (5类: plane, ship, harbor, bridge, helicopter)

DOTA标注格式 (.txt):
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    188.0 1019.0 259.0 1023.0 254.0 1073.0 183.0 1069.0 plane 0

COCO标注格式 (.json):
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }

使用方法:
    # 基本用法
    python dota_to_coco_5cls.py data/train data/annotations/train.json
    
    # 指定图像后缀
    python dota_to_coco_5cls.py data/train data/annotations/train.json --img-suffix jpg
"""

import json
import os
import os.path as osp
from argparse import ArgumentParser
from glob import glob

import cv2
from tqdm import tqdm

# 5类目标
CLASSES_5 = ['plane', 'ship', 'harbor', 'bridge', 'helicopter']


def parse_args():
    parser = ArgumentParser(description='DOTA格式转COCO格式 (5类)')
    parser.add_argument('src_path', help='源数据路径，包含 images/ 和 labelTxt/ 子目录')
    parser.add_argument('dst_path', help='输出JSON文件路径')
    parser.add_argument('--img-suffix', default='png', help='图像后缀名，默认 png')
    parser.add_argument('--classes', nargs='+', default=CLASSES_5,
                        help=f'类别列表，默认: {CLASSES_5}')
    return parser.parse_args()


def parse_dota_annotation(ann_file, classes):
    """解析DOTA格式标注文件
    
    Args:
        ann_file: 标注文件路径 (.txt)
        classes: 类别列表
        
    Returns:
        list: 解析后的目标列表
    """
    objects = []
    
    if not osp.exists(ann_file):
        return objects
    
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        # 跳过不完整的行
        if len(parts) < 9:
            continue
        
        try:
            # 解析四边形坐标
            poly = [float(parts[i]) for i in range(8)]
            category = parts[8]
            difficulty = int(parts[9]) if len(parts) >= 10 else 0
            
            # 只保留指定类别
            if category not in classes:
                continue
            
            # 计算外接矩形 (水平框)
            xs = poly[0::2]  # x1, x2, x3, x4
            ys = poly[1::2]  # y1, y2, y3, y4
            
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)
            
            width = xmax - xmin
            height = ymax - ymin
            
            if width <= 0 or height <= 0:
                continue
            
            obj = {
                'category': category,
                'bbox': [xmin, ymin, width, height],  # COCO格式: [x, y, w, h]
                'poly': poly,
                'area': width * height,
                'difficulty': difficulty
            }
            objects.append(obj)
            
        except (ValueError, IndexError) as e:
            print(f"警告: 解析行失败 '{line}': {e}")
            continue
    
    return objects


def convert_dota_to_coco(src_path, dst_path, img_suffix='png', classes=CLASSES_5):
    """将DOTA格式数据集转换为COCO格式
    
    Args:
        src_path: 源数据路径，包含 images/ 和 labelTxt/ 子目录
        dst_path: 输出JSON文件路径
        img_suffix: 图像后缀名
        classes: 类别列表
    """
    # 路径设置
    img_dir = osp.join(src_path, 'images')
    ann_dir = osp.join(src_path, 'labelTxt')
    
    if not osp.exists(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not osp.exists(ann_dir):
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")
    
    # 创建输出目录
    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    
    # 构建COCO数据结构
    coco_data = {
        'info': {
            'description': 'DOTA Dataset (5 classes) converted to COCO format',
            'version': '1.0',
            'year': 2024,
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 添加类别
    for idx, cls_name in enumerate(classes):
        coco_data['categories'].append({
            'id': idx + 1,  # COCO类别ID从1开始
            'name': cls_name,
            'supercategory': 'object'
        })
    
    # 类别名到ID的映射
    cls_to_id = {cls_name: idx + 1 for idx, cls_name in enumerate(classes)}
    
    # 获取所有标注文件
    ann_files = glob(osp.join(ann_dir, '*.txt'))
    
    if not ann_files:
        print(f"警告: 未找到标注文件，尝试从图像目录获取文件列表")
        img_files = glob(osp.join(img_dir, f'*.{img_suffix}'))
        basenames = [osp.splitext(osp.basename(f))[0] for f in img_files]
    else:
        basenames = [osp.splitext(osp.basename(f))[0] for f in ann_files]
    
    print(f"找到 {len(basenames)} 个样本")
    print(f"类别: {classes}")
    
    image_id = 0
    ann_id = 0
    
    stats = {cls: 0 for cls in classes}
    
    for basename in tqdm(basenames, desc="转换中"):
        # 图像路径
        img_path = osp.join(img_dir, f'{basename}.{img_suffix}')
        
        if not osp.exists(img_path):
            print(f"警告: 图像不存在 {img_path}")
            continue
        
        # 获取图像尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
        
        height, width = img.shape[:2]
        
        # 添加图像信息
        coco_data['images'].append({
            'id': image_id,
            'file_name': f'{basename}.{img_suffix}',
            'width': width,
            'height': height
        })
        
        # 解析标注
        ann_path = osp.join(ann_dir, f'{basename}.txt')
        objects = parse_dota_annotation(ann_path, classes)
        
        # 添加标注
        for obj in objects:
            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cls_to_id[obj['category']],
                'bbox': obj['bbox'],
                'area': obj['area'],
                'segmentation': [obj['poly']],  # 可选：保留原始四边形
                'iscrowd': 0
            })
            stats[obj['category']] += 1
            ann_id += 1
        
        image_id += 1
    
    # 保存JSON
    with open(dst_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # 打印统计
    print(f"\n转换完成!")
    print(f"输出文件: {dst_path}")
    print(f"图像数量: {image_id}")
    print(f"标注数量: {ann_id}")
    print(f"各类别统计:")
    for cls, count in stats.items():
        print(f"  - {cls}: {count}")


def main():
    args = parse_args()
    convert_dota_to_coco(
        src_path=args.src_path,
        dst_path=args.dst_path,
        img_suffix=args.img_suffix,
        classes=args.classes
    )


if __name__ == '__main__':
    main()
