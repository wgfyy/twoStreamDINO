# Copyright (c) OpenMMLab. All rights reserved.
"""Dual-Modal DOTA Dataset for optical + SAR fusion detection."""
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DualModalDOTADataset(CocoDataset):
    """Dataset for dual-modal DOTA-style datasets (COCO format).
    
    This dataset handles paired optical and SAR images from DOTA dataset 
    that has been converted to COCO format. It supports multiple ways to 
    specify the second modality image path.
    
    The dataset structure should be:
    data/
    ├── dual_modal_dota/
    │   ├── trainval/
    │   │   ├── optical/          # First modality (optical/RGB)
    │   │   │   ├── P0001__1__0___0.png
    │   │   │   └── ...
    │   │   ├── sar/              # Second modality (SAR)
    │   │   │   ├── P0001__1__0___0.png
    │   │   │   └── ...
    │   │   └── annotations/
    │   │       └── trainval.json
    │   └── test/
    │       ├── optical/
    │       ├── sar/
    │       └── annotations/
    │           └── test.json
    
    Args:
        data_prefix (dict): Prefix for data path. Should contain:
            - img: Path to first modality images (optical)
            - img2: Path to second modality images (SAR)
        img2_suffix (str): Suffix to convert img path to img2 path.
            Defaults to None.
        img2_prefix_replace (tuple): Tuple of (old_prefix, new_prefix) to 
            replace in img path to get img2 path. Defaults to None.
    """
    
    # DOTA-v1.0 classes (15 classes)
    METAINFO = {
        'classes': (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field', 
            'roundabout', 'harbor', 'swimming-pool', 'helicopter'
        ),
        'palette': [
            (165, 42, 42),    # plane - brown
            (189, 183, 107), # baseball-diamond - khaki
            (0, 255, 0),     # bridge - green
            (255, 0, 0),     # ground-track-field - red
            (138, 43, 226),  # small-vehicle - purple
            (255, 128, 0),   # large-vehicle - orange
            (255, 0, 255),   # ship - magenta
            (0, 255, 255),   # tennis-court - cyan
            (255, 193, 193), # basketball-court - pink
            (0, 51, 153),    # storage-tank - dark blue
            (255, 250, 205), # soccer-ball-field - lemon
            (0, 139, 139),   # roundabout - dark cyan
            (255, 255, 0),   # harbor - yellow
            (147, 116, 116), # swimming-pool - gray
            (0, 0, 255)      # helicopter - blue
        ]
    }

    def __init__(
        self,
        *args,
        img2_suffix: str = None,
        img2_prefix_replace: tuple = None,
        **kwargs
    ):
        self.img2_suffix = img2_suffix
        self.img2_prefix_replace = img2_prefix_replace
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format with dual-modal paths.

        Args:
            raw_data_info (dict): Raw data information loaded from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # Path for first modality image (optical)
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        
        # Path for second modality image (SAR)
        if 'file_name2' in img_info:
            # Second modality path specified in annotation
            if self.data_prefix.get('img2', None):
                img_path2 = osp.join(self.data_prefix['img2'], img_info['file_name2'])
            else:
                img_path2 = osp.join(self.data_prefix['img'], img_info['file_name2'])
        elif self.data_prefix.get('img2', None):
            # Use img2 prefix with same filename
            img_path2 = osp.join(self.data_prefix['img2'], img_info['file_name'])
        elif self.img2_suffix is not None:
            # Add suffix to filename
            base_name, ext = osp.splitext(img_info['file_name'])
            img_path2 = osp.join(
                self.data_prefix['img'], 
                f"{base_name}{self.img2_suffix}{ext}"
            )
        elif self.img2_prefix_replace is not None:
            # Replace prefix in path
            old_prefix, new_prefix = self.img2_prefix_replace
            img_path2 = img_path.replace(old_prefix, new_prefix)
        else:
            raise ValueError(
                "Must specify one of: file_name2 in annotation, "
                "img2 in data_prefix, img2_suffix, or img2_prefix_replace"
            )
        
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        
        data_info['img_path'] = img_path
        data_info['img_path2'] = img_path2
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info


@DATASETS.register_module()
class DualModalDOTAv15Dataset(DualModalDOTADataset):
    """Dataset for dual-modal DOTA-v1.5 (16 classes)."""
    
    METAINFO = {
        'classes': (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field', 
            'roundabout', 'harbor', 'swimming-pool', 'helicopter',
            'container-crane'  # Additional class in DOTA-v1.5
        ),
        'palette': [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
            (255, 255, 0), (147, 116, 116), (0, 0, 255), (255, 99, 71)
        ]
    }


@DATASETS.register_module()
class DualModalDOTAv2Dataset(DualModalDOTADataset):
    """Dataset for dual-modal DOTA-v2.0 (18 classes)."""
    
    METAINFO = {
        'classes': (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field', 
            'roundabout', 'harbor', 'swimming-pool', 'helicopter',
            'container-crane', 'airport', 'helipad'  # Additional classes
        ),
        'palette': [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
            (255, 255, 0), (147, 116, 116), (0, 0, 255), (255, 99, 71),
            (100, 149, 237), (34, 139, 34)
        ]
    }
