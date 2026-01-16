import argparse
import json
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO OBB (8 points) to COCO-OBB format')
    parser.add_argument('img_path', help='Path to image directory')
    parser.add_argument('label_path', help='Path to label directory (txt files)')
    parser.add_argument('out_file', help='Output JSON file path')
    parser.add_argument('--classes', nargs='+', help='Class names list', required=True)
    return parser.parse_args()

def poly2obb(poly):
    """Convert 8-point polygon to (cx, cy, w, h, angle)."""
    if len(poly) != 8:
        return None
    pts = np.array(poly, dtype=np.float32).reshape(4, 2)
    # Use minAreaRect to get OBB
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    
    # cv2.minAreaRect returns angle in [-90, 0) usually.
    # We want angle in radians.
    # MMDetection 3.x OBB format varies, usually (cx, cy, w, h, angle_rad)
    # Check angle definition.
    # DOTA v1.5 uses angle in radians.
    
    return [cx, cy, w, h, angle * np.pi / 180.0]

def main():
    args = parse_args()
    
    # Define categories
    categories = []
    cat2id = {}
    for i, name in enumerate(args.classes):
        cat_id = i + 1
        categories.append({'id': cat_id, 'name': name, 'supercategory': 'object'})
        cat2id[name] = cat_id

    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    # List all txt files
    files = [f for f in os.listdir(args.label_path) if f.endswith('.txt')]
    
    print(f"Found {len(files)} label files.")

    for f in tqdm(files):
        base_name = osp.splitext(f)[0]
        # Check corresponding image (try likely extensions)
        found_img = False
        img_name = ""
        img_h, img_w = 0, 0
        
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            if osp.exists(osp.join(args.img_path, base_name + ext)):
                img_name = base_name + ext
                found_img = True
                
                # Get size
                img = cv2.imread(osp.join(args.img_path, img_name))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                break
        
        if not found_img:
            # Maybe image name is different logic?
            continue

        images.append({
            'id': img_id,
            'file_name': img_name,
            'height': img_h,
            'width': img_w
        })

        # Read Annotations
        with open(osp.join(args.label_path, f), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9: 
                    # Expect x1 y1 ... x4 y4 classname
                    # Or classname x1 ...
                    continue
                
                # Try to detect format
                # Format 1: x1 y1 ... x4 y4 class_name diff
                try:
                    coords = [float(x) for x in parts[:8]]
                    cls_name = parts[8]
                except ValueError:
                    # Maybe format: class_name x1 ...
                    # Try to parse
                    continue

                if cls_name not in cat2id:
                    continue
                
                poly = coords
                obb = poly2obb(poly)
                if obb is None:
                    continue
                
                cx, cy, w, h, angle = obb
                area = w * h
                
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat2id[cls_name],
                    'segmentation': [poly], # Save poly for reference
                    'area': area,
                    'bbox': [cx, cy, w, h, angle], # 5-dim OBB
                    'iscrowd': 0
                })
                ann_id += 1
        
        img_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    
    with open(args.out_file, 'w') as f:
        json.dump(coco_format, f)
    
    print(f"Converted {len(images)} images and {len(annotations)} annotations.")
    print(f"Saved to {args.out_file}")

if __name__ == '__main__':
    main()
