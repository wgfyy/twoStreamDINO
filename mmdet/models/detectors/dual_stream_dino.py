# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .dino import DINO


@MODELS.register_module()
class FeatureFusionModule(nn.Module):
    """Feature Fusion Module for dual-stream backbone outputs.
    
    This module fuses features from two modalities by concatenation 
    followed by 1x1 convolution for channel reduction.
    
    Args:
        in_channels (list[int]): List of input channel numbers from each scale.
        out_channels (list[int]): List of output channel numbers for each scale.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Defaults to dict(type='ReLU').
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        norm_cfg: dict = dict(type='BN'),
        act_cfg: dict = dict(type='ReLU', inplace=True),
    ) -> None:
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.fusion_convs = nn.ModuleList()
        
        for in_c, out_c in zip(in_channels, out_channels):
            # Each fusion_conv takes concatenated features (2*in_c channels)
            # and outputs out_c channels
            fusion_conv = nn.Sequential(
                nn.Conv2d(in_c * 2, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c) if norm_cfg['type'] == 'BN' else nn.GroupNorm(32, out_c),
                nn.ReLU(inplace=True) if act_cfg['type'] == 'ReLU' else nn.Identity()
            )
            self.fusion_convs.append(fusion_conv)
    
    def forward(self, feats1: Tuple[Tensor], feats2: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function.
        
        Args:
            feats1 (tuple[Tensor]): Multi-scale features from backbone1.
            feats2 (tuple[Tensor]): Multi-scale features from backbone2.
            
        Returns:
            tuple[Tensor]: Fused multi-scale features.
        """
        assert len(feats1) == len(feats2) == len(self.fusion_convs)
        fused_feats = []
        for feat1, feat2, fusion_conv in zip(feats1, feats2, self.fusion_convs):
            # Concatenate features from two modalities along channel dimension
            concat_feat = torch.cat([feat1, feat2], dim=1)
            # Apply 1x1 conv for channel reduction
            fused_feat = fusion_conv(concat_feat)
            fused_feats.append(fused_feat)
        return tuple(fused_feats)


@MODELS.register_module()
class DualStreamDINO(DINO):
    """Dual-Stream DINO detector for multi-modal detection.
    
    This detector extends DINO to handle dual-modal inputs (e.g., RGB + Thermal,
    RGB + Depth, etc.). It uses two separate backbones to extract features from
    each modality, then fuses the features before the neck.
    
    Args:
        backbone (dict): Config of the first backbone (modality 1).
        backbone2 (dict): Config of the second backbone (modality 2).
        fusion_module (dict, optional): Config of feature fusion module.
            If None, will use default FeatureFusionModule.
        neck (dict, optional): Config of the neck. Defaults to None.
        encoder (dict, optional): Config of the Transformer encoder.
        decoder (dict, optional): Config of the Transformer decoder.
        bbox_head (dict, optional): Config for the bounding box head module.
        positional_encoding (dict, optional): Config of positional encoding.
        num_queries (int): Number of decoder query in Transformer.
        dn_cfg (dict, optional): Config of denoising query generator.
        with_box_refine (bool): Whether to refine the references in decoder.
        as_two_stage (bool): Whether to generate proposals from encoder output.
        num_feature_levels (int): Number of feature levels.
        train_cfg (dict, optional): Training config.
        test_cfg (dict, optional): Testing config.
        data_preprocessor (dict, optional): Config of data preprocessor.
        init_cfg (dict, optional): Initialization config.
    """
    
    def __init__(
        self,
        backbone: ConfigType,
        backbone2: ConfigType,
        fusion_module: OptConfigType = None,
        neck: OptConfigType = None,
        encoder: OptConfigType = None,
        decoder: OptConfigType = None,
        bbox_head: OptConfigType = None,
        positional_encoding: OptConfigType = None,
        num_queries: int = 900,
        dn_cfg: OptConfigType = None,
        with_box_refine: bool = True,
        as_two_stage: bool = True,
        num_feature_levels: int = 4,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        # Store backbone2 config before calling super().__init__
        self.backbone2_cfg = backbone2
        self.fusion_module_cfg = fusion_module
        
        super().__init__(
            backbone=backbone,
            neck=neck,
            encoder=encoder,
            decoder=decoder,
            bbox_head=bbox_head,
            positional_encoding=positional_encoding,
            num_queries=num_queries,
            dn_cfg=dn_cfg,
            with_box_refine=with_box_refine,
            as_two_stage=as_two_stage,
            num_feature_levels=num_feature_levels,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        
        # Build the second backbone
        self.backbone2 = MODELS.build(self.backbone2_cfg)
        
        # Build the feature fusion module
        if self.fusion_module_cfg is not None:
            self.fusion_module = MODELS.build(self.fusion_module_cfg)
        else:
            # Default fusion module using FeatureFusionModule
            # Get the output channels from neck config (in_channels)
            in_channels = neck['in_channels']  # e.g., [512, 1024, 2048]
            self.fusion_module = FeatureFusionModule(
                in_channels=in_channels,
                out_channels=in_channels,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True),
            )
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from dual-modal images.
        
        Args:
            batch_inputs (Tensor): Stacked dual-modal images with shape 
                (B, 6, H, W) where first 3 channels are modality1 (e.g., RGB)
                and last 3 channels are modality2 (e.g., Thermal/Depth).
                
        Returns:
            tuple[Tensor]: Multi-level features from neck.
        """
        # Split the input into two modalities
        # Assuming input shape is (B, 6, H, W) with 3+3 channels
        img1 = batch_inputs[:, :3, :, :]  # First modality (e.g., RGB)
        img2 = batch_inputs[:, 3:, :, :]  # Second modality (e.g., Thermal)
        
        # Extract features from both backbones
        feats1 = self.backbone(img1)
        feats2 = self.backbone2(img2)
        
        # Fuse features from both modalities
        fused_feats = self.fusion_module(feats1, feats2)
        
        # Pass through neck
        if self.with_neck:
            fused_feats = self.neck(fused_feats)
        
        return fused_feats
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input dual-modal images of shape 
                (bs, 6, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Inputs of shape (bs, 6, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples.
            rescale (bool): Whether to rescale the results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process for dual-modal input."""
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        return head_inputs_dict
