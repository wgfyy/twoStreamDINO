#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Prepare Dual-Modal DOTA Dataset for DualStreamDINO.

This script converts DOTA format annotations to COCO format and organizes
the dual-modal dataset (optical + SAR) for training.

Expected input structure:
data/
├── DOTA_optical/                 # Optical images
│   ├── train/
│   │   ├── images/
│   │   └── labelTxt/
│   ├── val/
│   │   ├── images/
│   │   └── labelTxt/
│   └── test/
│       └── images/
└── DOTA_sar/                     # SAR images (same filenames as optical)
    ├── train/
    │   └── images/
    ├── val/
    │   └── images/
    └── test/
        └── images/

Output structure:
data/
└── dual_modal_dota/
    ├── trainval/
    │   ├── optical/
    │   ├── sar/
    │   └── annotations/
    │       └── trainval.json
    ├── val/
    │   ├── optical/
    │   ├── sar/
    │   └── annotations/
    │       └── val.json
    └── test/
        ├── optical/
        ├── sar/
        └── annotations/
            └── test.json

Usage:
    python prepare_dual_modal_dota.py \
        --optical-root data/DOTA_optical \
        --sar-root data/DOTA_sar \
        --out-dir data/dual_modal_dota \
        --split trainval
"""

import argparse
import json
import os
import os.path as osp
import shutil
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm


# DOTA-v1.0 classes
DOTA_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
)

# DOTA-v1.5 classes
DOTA_V15_CLASSES = DOTA_CLASSES + ('container-crane',)

# DOTA-v2.0 classes
DOTA_V20_CLASSES = DOTA_V15_CLASSES + ('airport', 'helipad')


def parse_dota_annotation(ann_file: str, class_names: tuple, diff_thr: int = 100):
    """Parse DOTA format annotation file.
    
    Args:
        ann_file: Path to annotation file (.txt)
        class_names: Tuple of class names
        diff_thr: Difficulty threshold
        
    Returns:
        list: List of annotation dicts with keys:
            - bbox: [x1, y1, x2, y2, x3, y3, x4, y4] (quadrilateral)
            - bbox_hbox: [x_min, y_min, w, h] (horizontal bbox)
            - category: class name
            - difficult: difficulty flag
    """
    annotations = []
    
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        # Skip header lines
        if len(parts) < 9:
            continue
        if parts[0] in ['imagesource:', 'gsd:']:
            continue
        
        try:
            # Parse quadrilateral coordinates
            coords = [float(x) for x in parts[:8]]
            category = parts[8]
            difficult = int(parts[9]) if len(parts) > 9 else 0
            
            # Skip if class not in class_names
            if category not in class_names:
                continue
            
            # Skip difficult samples if above threshold
            if difficult > diff_thr:
                continue
            
            # Compute horizontal bounding box from quadrilateral
            xs = coords[0::2]  # x1, x2, x3, x4
            ys = coords[1::2]  # y1, y2, y3, y4
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            
            annotations.append({
                'bbox': coords,  # quadrilateral
                'bbox_hbox': [x_min, y_min, w, h],  # horizontal bbox
                'category': category,
                'difficult': difficult,
            })
        except (ValueError, IndexError):
            continue
    
    return annotations


def convert_to_coco_format(
    optical_img_dir: str,
    ann_dir: str,
    sar_img_dir: str,
    class_names: tuple,
    img_suffix: str = 'png'
):
    """Convert DOTA annotations to COCO format.
    
    Args:
        optical_img_dir: Directory containing optical images
        ann_dir: Directory containing annotation files
        sar_img_dir: Directory containing SAR images
        class_names: Tuple of class names
        img_suffix: Image file suffix
        
    Returns:
        dict: COCO format annotation dict
    """
    # Build category info
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            'id': i + 1,  # COCO category id starts from 1
            'name': name,
            'supercategory': 'object'
        })
    
    images = []
    annotations = []
    
    # Get all annotation files
    ann_files = sorted(glob(osp.join(ann_dir, '*.txt')))
    
    img_id = 0
    ann_id = 0
    
    for ann_file in tqdm(ann_files, desc='Converting annotations'):
        # Get image filename
        base_name = osp.splitext(osp.basename(ann_file))[0]
        img_name = f'{base_name}.{img_suffix}'
        
        # Check if optical image exists
        optical_img_path = osp.join(optical_img_dir, img_name)
        if not osp.exists(optical_img_path):
            print(f'Warning: Optical image not found: {optical_img_path}')
            continue
        
        # Check if SAR image exists
        sar_img_path = osp.join(sar_img_dir, img_name)
        if not osp.exists(sar_img_path):
            print(f'Warning: SAR image not found: {sar_img_path}')
            continue
        
        # Get image size
        with Image.open(optical_img_path) as img:
            width, height = img.size
        
        # Add image info
        images.append({
            'id': img_id,
            'file_name': img_name,
            'width': width,
            'height': height,
        })
        
        # Parse annotations
        anns = parse_dota_annotation(ann_file, class_names)
        
        for ann in anns:
            # Get category id
            cat_id = class_names.index(ann['category']) + 1
            
            # Get horizontal bbox [x, y, w, h]
            bbox = ann['bbox_hbox']
            area = bbox[2] * bbox[3]
            
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
            })
            ann_id += 1
        
        img_id += 1
    
    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    
    return coco_dict


def organize_dataset(
    optical_root: str,
    sar_root: str,
    out_dir: str,
    split: str,
    class_names: tuple,
    img_suffix: str = 'png',
    copy_images: bool = True
):
    """Organize dual-modal dataset.
    
    Args:
        optical_root: Root directory of optical images
        sar_root: Root directory of SAR images
        out_dir: Output directory
        split: Dataset split ('trainval', 'val', 'test')
        class_names: Tuple of class names
        img_suffix: Image file suffix
        copy_images: Whether to copy images or create symlinks
    """
    # Determine input directories
    if split == 'trainval':
        optical_img_dir = osp.join(optical_root, 'train', 'images')
        ann_dir = osp.join(optical_root, 'train', 'labelTxt')
        sar_img_dir = osp.join(sar_root, 'train', 'images')
    elif split == 'val':
        optical_img_dir = osp.join(optical_root, 'val', 'images')
        ann_dir = osp.join(optical_root, 'val', 'labelTxt')
        sar_img_dir = osp.join(sar_root, 'val', 'images')
    else:  # test
        optical_img_dir = osp.join(optical_root, 'test', 'images')
        ann_dir = None  # Test set has no annotations
        sar_img_dir = osp.join(sar_root, 'test', 'images')
    
    # Create output directories
    out_optical_dir = osp.join(out_dir, split, 'optical')
    out_sar_dir = osp.join(out_dir, split, 'sar')
    out_ann_dir = osp.join(out_dir, split, 'annotations')
    
    os.makedirs(out_optical_dir, exist_ok=True)
    os.makedirs(out_sar_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    
    print(f'Processing {split} split...')
    
    # Convert annotations to COCO format
    if ann_dir and osp.exists(ann_dir):
        coco_dict = convert_to_coco_format(
            optical_img_dir, ann_dir, sar_img_dir, class_names, img_suffix
        )
        
        # Save COCO annotation file
        ann_file = osp.join(out_ann_dir, f'{split}.json')
        with open(ann_file, 'w') as f:
            json.dump(coco_dict, f, indent=2)
        print(f'Saved annotations to {ann_file}')
        print(f'  - Images: {len(coco_dict["images"])}')
        print(f'  - Annotations: {len(coco_dict["annotations"])}')
        
        # Get valid image names
        valid_images = [img['file_name'] for img in coco_dict['images']]
    else:
        # For test set, use all images
        valid_images = [osp.basename(f) for f in glob(osp.join(optical_img_dir, f'*.{img_suffix}'))]
        
        # Create empty annotation file for test
        coco_dict = {
            'images': [{'id': i, 'file_name': name, 'width': 0, 'height': 0} 
                      for i, name in enumerate(valid_images)],
            'annotations': [],
            'categories': [{'id': i+1, 'name': n, 'supercategory': 'object'} 
                          for i, n in enumerate(class_names)],
        }
        ann_file = osp.join(out_ann_dir, f'{split}.json')
        with open(ann_file, 'w') as f:
            json.dump(coco_dict, f, indent=2)
    
    # Copy or link images
    print(f'Copying/linking images...')
    for img_name in tqdm(valid_images):
        src_optical = osp.join(optical_img_dir, img_name)
        src_sar = osp.join(sar_img_dir, img_name)
        dst_optical = osp.join(out_optical_dir, img_name)
        dst_sar = osp.join(out_sar_dir, img_name)
        
        if copy_images:
            if osp.exists(src_optical) and not osp.exists(dst_optical):
                shutil.copy2(src_optical, dst_optical)
            if osp.exists(src_sar) and not osp.exists(dst_sar):
                shutil.copy2(src_sar, dst_sar)
        else:
            if osp.exists(src_optical) and not osp.exists(dst_optical):
                os.symlink(osp.abspath(src_optical), dst_optical)
            if osp.exists(src_sar) and not osp.exists(dst_sar):
                os.symlink(osp.abspath(src_sar), dst_sar)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dual-modal DOTA dataset for DualStreamDINO'
    )
    parser.add_argument(
        '--optical-root',
        type=str,
        required=True,
        help='Root directory of optical DOTA images'
    )
    parser.add_argument(
        '--sar-root',
        type=str,
        required=True,
        help='Root directory of SAR images'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/dual_modal_dota',
        help='Output directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        nargs='+',
        default=['trainval', 'val', 'test'],
        choices=['trainval', 'val', 'test'],
        help='Dataset splits to process'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        choices=['v1', 'v1.5', 'v2'],
        help='DOTA version'
    )
    parser.add_argument(
        '--img-suffix',
        type=str,
        default='png',
        help='Image file suffix'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symbolic links instead of copying images'
    )
    
    args = parser.parse_args()
    
    # Select class names based on version
    if args.version == 'v1':
        class_names = DOTA_CLASSES
    elif args.version == 'v1.5':
        class_names = DOTA_V15_CLASSES
    else:
        class_names = DOTA_V20_CLASSES
    
    print(f'DOTA version: {args.version}')
    print(f'Number of classes: {len(class_names)}')
    print(f'Classes: {class_names}')
    
    # Process each split
    for split in args.split:
        organize_dataset(
            optical_root=args.optical_root,
            sar_root=args.sar_root,
            out_dir=args.out_dir,
            split=split,
            class_names=class_names,
            img_suffix=args.img_suffix,
            copy_images=not args.symlink
        )
    
    print('\nDone!')
    print(f'\nDataset structure created at: {args.out_dir}')
    print('\nTo train the model, run:')
    print('  python tools/train.py configs/dino/dual_stream_dino-4scale_r50_8xb2-12e_dota.py')


if __name__ == '__main__':
    main()
