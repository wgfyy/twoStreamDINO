# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.match_costs.match_cost import BaseMatchCost

try:
    from mmcv.ops import box_iou_rotated
except ImportError:
    box_iou_rotated = None


@TASK_UTILS.register_module()
class RotatedIoUCost(BaseMatchCost):
    """RotatedIoUCost.

    Args:
        iou_mode (str): IoU mode. Currently only support 'iou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, iou_mode: str = 'iou', weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain "bboxes" (and "scores" if needed).
            gt_instances (:obj:`InstanceData`): Ground truth instances which
                must contain "bboxes".
            img_meta (Optional[dict]): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        if box_iou_rotated is None:
            raise ImportError('Please install mmcv-full to use RotatedIoUCost.')
            
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # Check if bboxes are normalized (common in DINO)
        # If max(pred_bboxes) <= 1.0, they might be normalized.
        # However, for Rotated boxes, angle can be > 1.
        # So we look at x, y, w, h.
        
        # We assume pred_bboxes and gt_bboxes are in the SAME scale.
        # DINO usually outputs normalized boxes (cx, cy, w, h, a) where cx,cy,w,h are [0,1].
        # But `box_iou_rotated` expects absolute coordinates for correct area calc?
        # Actually IoU is scale invariant, so normalized [0,1] is fine 
        # AS LONG AS both are normalized and width/height aspect ratio is preserved?
        # NO. If normalized to square [0,1]x[0,1] but image is non-square, 
        # then "angle" interpretation becomes skewed if we work in normalized space.
        # We MUST denormalize to absolute coordinates if the image is not square.
        
        if img_meta is not None:
            # Denormalize if they look normalized (heuristic) or if we force it.
            # Usually DINO Heads pass normalized boxes to Assigners.
            # But here `pred_instances` comes from `RunCodeSnippet` output logic or standard DINO.
            # Standard DINOHead puts normalized boxes in `.bboxes`.
            
            # For rotated IoU, we need absolute coordinates if we want the rotation to be meaningful
            # relative to the image aspect ratio.
            
            h, w = img_meta['img_shape'][:2]
            
            # Copy to avoid modifying original
            pred_bboxes_abs = pred_bboxes.clone()
            gt_bboxes_abs = gt_bboxes.clone()
            
            # Assumes cx, cy, w, h, a.
            # Multiply first 4 coords by scale.
            # Angle stays same.
            scale = torch.tensor([w, h, w, h], device=pred_bboxes.device)
            
            pred_bboxes_abs[:, :4] = pred_bboxes_abs[:, :4] * scale
            gt_bboxes_abs[:, :4] = gt_bboxes_abs[:, :4] * scale
            
            pred_bboxes = pred_bboxes_abs
            gt_bboxes = gt_bboxes_abs

        # Calculate Rotated IoU
        # mmcv.ops.box_iou_rotated expects (N, 5)
        # overlaps: (N, M)
        overlaps = box_iou_rotated(pred_bboxes, gt_bboxes)

        # The cost is -(IoU) * weight
        # Usually IoUCost is 1 - IoU, or just -IoU
        # Standard IoUCost in mmdet uses: iou_cost = - iou
        
        iou_cost = -overlaps
        return iou_cost * self.weight
