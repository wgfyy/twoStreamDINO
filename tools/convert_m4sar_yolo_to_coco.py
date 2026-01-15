#!/usr/bin/env python3
"""Convert the M4-SAR dual-modal YOLO labels to COCO format.

The script builds COCO json files that can be loaded with DualModalCocoDataset.
It assumes YOLO labels are quadrilateral polygons (class_id x1 y1 x2 y2 x3 y3 x4 y4)
with coordinates normalized to [0, 1].
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

# Category ids are 1-based in COCO; YOLO labels are 0-based.
CATEGORIES: List[Dict[str, object]] = [
    {"id": 1, "name": "bridge", "supercategory": "object"},
    {"id": 2, "name": "harbor", "supercategory": "object"},
    {"id": 3, "name": "oil_tank", "supercategory": "object"},
    {"id": 4, "name": "playground", "supercategory": "object"},
    {"id": 5, "name": "airport", "supercategory": "object"},
    {"id": 6, "name": "wind_turbine", "supercategory": "object"},
]

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def gather_images(img_dir: Path) -> List[Path]:
    """Collect image paths with supported extensions."""
    files: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(img_dir.glob(pattern))
    # Remove duplicates and sort for deterministic ids
    return sorted(set(files))


def polygon_area(points: Sequence[float]) -> float:
    """Compute polygon area using the shoelace formula."""
    n = len(points) // 2
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[2 * i], points[2 * i + 1]
        x2, y2 = points[2 * ((i + 1) % n)], points[2 * ((i + 1) % n) + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_yolo_line(
    line: str, width: int, height: int
) -> Tuple[int, List[float], List[float], float]:
    """
    Parse a single YOLO quadrilateral line.

    Returns (category_id, bbox[x,y,w,h], segmentation[list], area).
    """
    parts = line.strip().split()
    if len(parts) < 9:
        raise ValueError(f"Line has {len(parts)} fields (<9): {line}")

    cls_id = int(float(parts[0]))
    coords = list(map(float, parts[1:9]))

    # Decide whether the coordinates are normalized.
    normalized = all(0.0 <= c <= 1.5 for c in coords)

    xs = coords[0::2]
    ys = coords[1::2]

    if normalized:
        xs = [x * width for x in xs]
        ys = [y * height for y in ys]

    xs = [min(max(x, 0.0), float(width)) for x in xs]
    ys = [min(max(y, 0.0), float(height)) for y in ys]

    segmentation: List[float] = []
    for x, y in zip(xs, ys):
        segmentation.extend([x, y])

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError("Degenerate bbox after clipping")

    bbox = [x_min, y_min, bbox_w, bbox_h]
    area = bbox_w * bbox_h  # Align with existing DOTA COCO style

    return cls_id, bbox, segmentation, area


def convert_split(
    split: str,
    optical_img_dir: Path,
    optical_label_dir: Path,
    sar_img_dir: Path,
    out_json: Path,
) -> None:
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    img_id = 0
    ann_id = 0

    missing_labels: List[str] = []
    missing_sar: List[str] = []

    image_files = gather_images(optical_img_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {optical_img_dir}")

    for img_path in image_files:
        if not img_path.is_file():
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        if not (sar_img_dir / img_path.name).exists():
            missing_sar.append(img_path.name)

        label_path = optical_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(img_path.name)
            img_id += 1
            continue

        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cls_id, bbox, seg, area = parse_yolo_line(line, width, height)
            except ValueError:
                # Skip malformed entries while keeping the image.
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id + 1,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [seg],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    coco = {
        "info": {
            "description": "M4-SAR dual-modal dataset converted to COCO",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=True, indent=2)

    print(
        f"[{split}] images={len(images)} anns={len(annotations)} "
        f"missing_labels={len(missing_labels)} missing_sar={len(missing_sar)}"
    )
    if missing_labels:
        print(f"[{split}] Warning: {len(missing_labels)} images missing labels (kept without boxes)")
    if missing_sar:
        print(f"[{split}] Warning: {len(missing_sar)} SAR images not found")


def make_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    print(f"Linked {dst} -> {src}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert M4-SAR YOLO to COCO")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to M4-SAR root (containing optical/ and sar/)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write COCO annotations (and optional symlinks)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to convert",
    )
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Create symlinks under output-root/images/{optical,sar}/{split}",
    )
    args = parser.parse_args()

    optical_base = args.dataset_root / "optical"
    sar_base = args.dataset_root / "sar"

    for split in args.splits:
        optical_img_dir = optical_base / "images" / split
        optical_label_dir = optical_base / "labels" / split
        sar_img_dir = sar_base / "images" / split
        out_json = args.output_root / "annotations" / f"{split}.json"

        convert_split(split, optical_img_dir, optical_label_dir, sar_img_dir, out_json)

        if args.link_images:
            make_symlink(optical_img_dir, args.output_root / "images" / "optical" / split)
            make_symlink(sar_img_dir, args.output_root / "images" / "sar" / split)


if __name__ == "__main__":
    main()
