# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .dual_modal_dota import DualModalDOTADataset


@DATASETS.register_module()
class DualModalRotatedDOTADataset(DualModalDOTADataset):
    """Dataset for dual-modal OBB DOTA-style datasets.
    
    Expects annotation JSON to have 'bbox' in format [x, y, w, h, angle].
    """
    
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
        # Reuse logic from parent
        if 'file_name2' in img_info:
            if self.data_prefix.get('img2', None):
                img_path2 = osp.join(self.data_prefix['img2'], img_info['file_name2'])
            else:
                img_path2 = osp.join(self.data_prefix['img'], img_info['file_name2'])
        elif self.data_prefix.get('img2', None):
            img_path2 = osp.join(self.data_prefix['img2'], img_info['file_name'])
        elif self.img2_suffix is not None:
            base_name, ext = osp.splitext(img_info['file_name'])
            img_path2 = osp.join(
                self.data_prefix['img'], 
                f"{base_name}{self.img2_suffix}{ext}"
            )
        elif self.img2_prefix_replace is not None:
            old_prefix, new_prefix = self.img2_prefix_replace
            img_path2 = img_path.replace(old_prefix, new_prefix)
        else:
            img_path2 = img_path # Fallback or error? Parent raises error.
            if not self.test_mode:
               pass # Should not happen if parent logic matches

        
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
            
            # OBB Parsing
            # Case 1: bbox is [cx, cy, w, h, angle]
            if len(ann['bbox']) == 5:
                x, y, w, h, a = ann['bbox']
                bbox = [x, y, w, h, a]
            # Case 2: bbox is HBB [x, y, w, h] (maybe mixed?)
            elif len(ann['bbox']) == 4:
                # If loading HBB dataset into Rotated Model, assume angle=0
                # But this converts x,y (top-left) to cx, cy
                x1, y1, w, h = ann['bbox']
                bbox = [x1 + w/2, y1 + h/2, w, h, 0.0]
            else:
                continue

            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

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
