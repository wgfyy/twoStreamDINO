# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.losses.utils import weighted_loss

try:
    from mmcv.ops import diff_iou_rotated_2d
except ImportError:
    diff_iou_rotated_2d = None


@weighted_loss
def rotated_iou_loss(pred: Tensor,
                     target: Tensor,
                     linear: bool = False,
                     eps: float = 1e-6) -> Tensor:
    """Rotated IoU loss.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool): If True, use linear scale of loss else use log scale.
            Default: False.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: Loss tensor.
    """
    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full to use rotated_iou_loss.')
    
    # diff_iou_rotated_2d calculates IoU (differentiable)
    # It expects (x, y, w, h, a)
    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    # Clamp ious to avoid log(0) = -inf
    ious = torch.clamp(ious, min=eps, max=1.0)
    
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()
        
    return loss


@MODELS.register_module()
class RotatedIoULoss(nn.Module):
    """RotatedIoULoss.

    Args:
        linear (bool): If True, use linear scale of loss else use log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 linear: bool = False,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: int = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
                shape (n, 5).
            target (Tensor): Corresponding gt bboxes, shape (n, 5).
            weight (Tensor, optional): The weight of loss for each
                prediction.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
            
        loss = self.loss_weight * rotated_iou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
            
        return loss
