# Copyright (c) OpenMMLab. All rights reserved.
import copy
import functools
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from mmengine.structures import InstanceData
from mmengine.model import constant_init, bias_init_with_prob

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, reduce_mean, OptInstanceList
from .dino_head import DINOHead


@MODELS.register_module()
class RotatedDINOHead(DINOHead):
    """
    Rotated DINO Head for OBB detection.
    Predicts (cx, cy, w, h, angle).
    """
    
    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # Initialize classification branch (same as parent)
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        
        # Initialize regression branch with 5 outputs (cx, cy, w, h, angle)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 5))  # 5-dim for OBB
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
            
    def init_weights(self) -> None:
        """Initialize weights of the Rotated DINO head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        # Initialize w, h biases
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:4], -2.0)
        # Initialize angle bias to 0
        nn.init.constant_(self.reg_branches[0][-1].bias.data[4], 0.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:4], 0.0)
        
    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """
        Overridden to handle 5-dim boxes (OBB).
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        
        return self.loss_by_feat(
            outs[0],  # all_layers_outputs_classes
            outs[1],  # all_layers_outputs_coords
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas,
            dn_meta)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function for Rotated DINO Head (OBB).
        
        This overrides the parent class to handle 5-dim OBB boxes.
        """
        # Extract denoising and matching parts
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        
        loss_dict = dict()
        
        # Calculate losses for matching part
        if all_layers_matching_cls_scores is not None:
            matching_losses_cls, matching_losses_bbox, matching_losses_iou = \
                self._loss_dn_single(
                    all_layers_matching_cls_scores,
                    all_layers_matching_bbox_preds,
                    batch_gt_instances,
                    batch_img_metas,
                    batch_gt_instances_ignore)
            
            # NOTE: set num_decoder_layers - 1 as key to be same with parent
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    matching_losses_cls, matching_losses_bbox, matching_losses_iou):
                loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
                num_dec_layer += 1
        
        # Calculate losses for denoising part
        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = \
                self._loss_dn_single(
                    all_layers_denoising_cls_scores,
                    all_layers_denoising_bbox_preds,
                    batch_gt_instances,
                    batch_img_metas,
                    batch_gt_instances_ignore,
                    dn_meta=dn_meta)
            
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    dn_losses_cls, dn_losses_bbox, dn_losses_iou):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                num_dec_layer += 1
        
        # Calculate encoder losses if in two-stage mode
        if enc_cls_scores is not None:
            enc_loss_cls, enc_loss_bbox, enc_loss_iou = \
                self.loss_by_feat_single_obb(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances, batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_loss_bbox
            loss_dict['enc_loss_iou'] = enc_loss_iou
        
        return loss_dict
    
    def _loss_dn_single(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        dn_meta: Optional[Dict[str, int]] = None
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Calculate losses for all decoder layers.
        
        Args:
            all_layers_cls_scores: Shape (num_layers, bs, num_queries, num_classes)
            all_layers_bbox_preds: Shape (num_layers, bs, num_queries, 5)
            batch_gt_instances: GT instances for each image
            batch_img_metas: Meta info for each image
            dn_meta: Denoising meta info (if denoising part)
            
        Returns:
            Tuple of lists containing cls_loss, bbox_loss, iou_loss for each layer
        """
        num_dec_layers = all_layers_cls_scores.shape[0]
        
        losses_cls = []
        losses_bbox = []
        losses_iou = []
        
        for layer_id in range(num_dec_layers):
            loss_cls, loss_bbox, loss_iou = self.loss_by_feat_single_obb(
                all_layers_cls_scores[layer_id],
                all_layers_bbox_preds[layer_id],
                batch_gt_instances,
                batch_img_metas,
                dn_meta=dn_meta)
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)
            losses_iou.append(loss_iou)
        
        return losses_cls, losses_bbox, losses_iou
    
    def loss_by_feat_single_obb(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Optional[Dict[str, int]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculate loss for a single decoder layer with OBB.
        
        Args:
            cls_scores: Shape (bs, num_queries, num_classes)
            bbox_preds: Shape (bs, num_queries, 5) - (cx, cy, w, h, angle)
            batch_gt_instances: GT instances for each image
            batch_img_metas: Meta info for each image
            dn_meta: Denoising meta info
            
        Returns:
            Tuple of (cls_loss, bbox_loss, iou_loss)
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        
        # Get targets
        cls_reg_targets = self.get_targets_obb(
            cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas, dn_meta)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        # Classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        
        # Regression loss
        bbox_preds = bbox_preds.reshape(-1, 5)
        if num_total_pos > 0:
            bbox_weights_pos = bbox_weights.sum(-1) > 0
            
            # Normalize bbox predictions and targets
            loss_bbox = self.loss_bbox(
                bbox_preds[bbox_weights_pos],
                bbox_targets[bbox_weights_pos],
                weight=bbox_weights[bbox_weights_pos],
                avg_factor=num_total_pos)
            
            # IoU loss for rotated boxes
            loss_iou = self.loss_iou(
                bbox_preds[bbox_weights_pos],
                bbox_targets[bbox_weights_pos],
                weight=bbox_weights[bbox_weights_pos, :1].squeeze(-1),
                avg_factor=num_total_pos)
        else:
            loss_bbox = bbox_preds.sum() * 0
            loss_iou = bbox_preds.sum() * 0
        
        return loss_cls, loss_bbox, loss_iou
    
    def get_targets_obb(
        self,
        cls_scores_list: List[Tensor],
        bbox_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Optional[Dict[str, int]] = None
    ) -> tuple:
        """Compute regression and classification targets for all images.
        
        Args:
            cls_scores_list: List of cls scores for each image
            bbox_preds_list: List of bbox predictions for each image  
            batch_gt_instances: GT instances for each image
            batch_img_metas: Meta info for each image
            dn_meta: Denoising meta info
            
        Returns:
            Tuple containing targets and statistics
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single_obb,
            cls_scores_list,
            bbox_preds_list,
            batch_gt_instances,
            batch_img_metas,
            dn_meta=dn_meta)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    
    def _get_targets_single_obb(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        dn_meta: Optional[Dict[str, int]] = None
    ) -> tuple:
        """Compute targets for a single image with OBB.
        
        Args:
            cls_score: Shape (num_queries, num_classes)
            bbox_pred: Shape (num_queries, 5) - (cx, cy, w, h, angle)
            gt_instances: GT instances with 'bboxes' (N, 5) and 'labels' (N,)
            img_meta: Image meta info
            dn_meta: Denoising meta info
            
        Returns:
            Tuple of targets
        """
        img_h, img_w = img_meta['img_shape']
        num_bboxes = bbox_pred.size(0)
        
        gt_bboxes = gt_instances.bboxes  # (N, 5): cx, cy, w, h, angle
        gt_labels = gt_instances.labels
        
        # For OBB, we use the predicted boxes directly for assignment
        # The bbox_pred is already in normalized format
        # Unnormalize pred boxes for assignment (cx, cy, w, h denormalized, angle converted to radians)
        # Note: angle in bbox_pred is normalized [0, 1], representing [-pi/2, pi/2]
        PI_HALF = 1.5707963267948966  # pi/2
        
        # For assignment, we need to convert pred angle back to radians
        # Denormalize: [0, 1] -> [-pi/2, pi/2]
        bbox_pred_unnorm = bbox_pred.clone()
        bbox_pred_unnorm[..., :4] = bbox_pred[..., :4] * bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
        bbox_pred_unnorm[..., 4:5] = bbox_pred[..., 4:5] * (2 * PI_HALF) - PI_HALF  # convert to radians
        
        # Create pred instances for assignment
        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred_unnorm)
        
        # Assignment using OBB assigner
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
        
        # Label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        
        # BBox targets - normalize GT boxes
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        
        # Normalize GT boxes: cx, cy, w, h by image size, angle by pi to [0, 1]
        # Angle range is [-pi/2, pi/2], normalize to [0, 1]: (-pi/2 -> 0, pi/2 -> 1)
        # This matches the normalized prediction format
        pos_gt_bboxes_normalized = pos_gt_bboxes.clone()
        pos_gt_bboxes_normalized[..., :4] = pos_gt_bboxes[..., :4] / pos_gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])
        pos_gt_bboxes_normalized[..., 4:5] = (pos_gt_bboxes[..., 4:5] + PI_HALF) / (2 * PI_HALF)  # normalize angle to [0, 1]
        bbox_targets[pos_inds] = pos_gt_bboxes_normalized
        
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward function for Rotated DINO Head.
        
        Handles 5-dim OBB boxes (cx, cy, w, h, angle).
        References can be 4-dim or 5-dim.
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = references[layer_id]
            hidden_state = hidden_states[layer_id]
            
            # Classification
            outputs_class = self.cls_branches[layer_id](hidden_state)
            
            # Regression: outputs 5-dim (cx, cy, w, h, angle)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            
            # Handle reference points
            # reference shape: (bs, num_queries, 4 or 5)
            ref_dim = reference.shape[-1]
            
            # For the spatial dimensions (cx, cy, w, h), use standard DINO formulation
            # tmp_reg_preds[..., :4] += inverse_sigmoid(reference[..., :4])
            reference_xywh = inverse_sigmoid(reference[..., :4])
            tmp_reg_preds_xywh = tmp_reg_preds[..., :4] + reference_xywh
            
            # For angle dimension:
            # If reference has angle (5-dim), add it; otherwise just use predicted angle
            if ref_dim >= 5:
                # Reference has angle, refine it
                reference_angle = inverse_sigmoid(reference[..., 4:5])
                tmp_reg_preds_angle = tmp_reg_preds[..., 4:5] + reference_angle
            else:
                # Reference has no angle, predict from scratch
                # Apply sigmoid to bound the angle
                tmp_reg_preds_angle = tmp_reg_preds[..., 4:5]
            
            # Combine and apply sigmoid
            tmp_reg_preds_full = torch.cat([tmp_reg_preds_xywh, tmp_reg_preds_angle], dim=-1)
            outputs_coord = tmp_reg_preds_full.sigmoid()
            
            # outputs_coord is in [0, 1] for all dimensions including angle
            # Angle is normalized: [0, 1] represents [-pi/2, pi/2] radians
            # We keep it normalized here; will convert in loss calculation if needed
            
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse sigmoid function.
    
    Args:
        x (Tensor): Input tensor.
        eps (float): Minimum value to clamp.
        
    Returns:
        Tensor: Output tensor.
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    
    Note:
        This function applies ``func`` to multiple inputs and
        collate each output to a list.
    """
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
