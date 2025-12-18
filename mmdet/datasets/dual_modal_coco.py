# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .coco import CocoDataset


@DATASETS.register_module()
class DualModalCocoDataset(CocoDataset):
    """Dataset for dual-modal COCO-style datasets.
    
    This dataset extends CocoDataset to handle paired images from two modalities.
    It expects the annotation file to have an additional 'file_name2' field
    for the second modality image, or you can specify a second image directory.
    
    Args:
        data_prefix (dict): Prefix for data path. Should contain:
            - img: Path to first modality images
            - img2: Path to second modality images (optional, if not in annotation)
        img2_suffix (str): Suffix to convert img path to img2 path if img2 prefix
            is not specified. E.g., if img is 'xxx.jpg', img2 might be 'xxx_thermal.jpg'
            with suffix='_thermal'. Defaults to None.
        img2_prefix_replace (tuple): Tuple of (old_prefix, new_prefix) to replace
            in img path to get img2 path. Defaults to None.
    """
    
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
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # Path for first modality image
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        
        # Path for second modality image
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
