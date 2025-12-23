# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .dino import DINO


@MODELS.register_module()
class SimpleChannelFusion(nn.Module):
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


class SpatialCrossAttentionBlock(nn.Module):
    """Spatial Cross-Attention Block for bidirectional feature interaction.
    
    This block performs bidirectional cross-attention between two modalities:
    - Branch 1: RGB queries SAR (SAR as Key/Value, RGB as Query)
    - Branch 2: SAR queries RGB (RGB as Key/Value, SAR as Query)
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config for normalization layer.
        act_cfg (dict): Config for activation layer.
    改进方向：
        将交叉注意力机制与FPN的思想结合，在不同尺度上进行双向交叉注意力融合.
        可能的实现途径：
        1.先交叉注意力，再FPN
        2.直接在交叉的时候，就使用类似于模态1的C5层的Query去查询模态2的C4层的Key/Value，以此类推
        3.先FPN，然后两个模态的多尺度特征之间进行交叉注意力融合
    """
    
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        
        # Reduce dimension to reduce computation (usually 1/2 or 1/8)
        inter_channels = in_channels // 2
        
        # 1. Modality 1 (RGB) transformation layers
        self.conv_q1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 2. Modality 2 (SAR) transformation layers
        self.conv_q2 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_k2 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 3. Final fusion layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if norm_cfg['type'] == 'BN' else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True) if act_cfg['type'] == 'ReLU' else nn.Identity()
        )
        
        # Learnable parameters Gamma to control attention strength
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """Forward function.
        
        Args:
            x1 (Tensor): Visible feature [B, C, H, W]
            x2 (Tensor): SAR feature [B, C, H, W]
            
        Returns:
            Tensor: Fused feature [B, out_channels, H, W]
        """
        B, C, H, W = x1.size()
        
        # --- Branch 1: SAR assists RGB (SAR as Key/Value, RGB as Query) ---
        # Logic: RGB wants to see where SAR has strong responses
        q1 = self.conv_q1(x1).view(B, -1, H * W).permute(0, 2, 1)  # B, N, C'
        k2 = self.conv_k2(x2).view(B, -1, H * W)                   # B, C', N
        v2 = self.conv_v2(x2).view(B, -1, H * W)                   # B, C, N
        
        # Attention Map: RGB Query finds SAR Key
        attn12 = torch.bmm(q1, k2)  # B, N, N (spatial attention matrix)
        attn12 = F.softmax(attn12, dim=-1)
        
        # Aggregate SAR's Value to RGB
        out1 = torch.bmm(v2, attn12.permute(0, 2, 1)).view(B, C, H, W)
        x1_new = self.gamma1 * out1 + x1  # Residual connection
        
        # --- Branch 2: RGB assists SAR (RGB as Key/Value, SAR as Query) ---
        q2 = self.conv_q2(x2).view(B, -1, H * W).permute(0, 2, 1)
        k1 = self.conv_k1(x1).view(B, -1, H * W)
        v1 = self.conv_v1(x1).view(B, -1, H * W)
        
        attn21 = torch.bmm(q2, k1)
        attn21 = F.softmax(attn21, dim=-1)
        
        out2 = torch.bmm(v1, attn21.permute(0, 2, 1)).view(B, C, H, W)
        x2_new = self.gamma2 * out2 + x2
        
        # --- Final fusion ---
        # Concatenate the two enhanced features
        x_fused = torch.cat([x1_new, x2_new], dim=1)  # [B, 2C, H, W]
        x_fused = self.output_conv(x_fused)           # [B, out_channels, H, W]
        
        return x_fused


@MODELS.register_module()
class BiCrossAttentionFusion(nn.Module):
    """Bidirectional Cross-Modal Attention Fusion Module.
    
    This module applies bidirectional cross-attention between two modalities
    (e.g., Optical + SAR) at multiple feature scales.
    
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
        
        self.fusion_blocks = nn.ModuleList()
        
        for in_c, out_c in zip(in_channels, out_channels):
            self.fusion_blocks.append(
                SpatialCrossAttentionBlock(in_c, out_c, norm_cfg, act_cfg)
            )

    def forward(self, feats1: Tuple[Tensor], feats2: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function.
        
        Args:
            feats1 (tuple[Tensor]): Multi-scale Optical features (List of Tensors)
            feats2 (tuple[Tensor]): Multi-scale SAR features (List of Tensors)
            
        Returns:
            tuple[Tensor]: Fused multi-scale features.
        """
        assert len(feats1) == len(feats2) == len(self.fusion_blocks)
        
        fused_feats = []
        for feat1, feat2, block in zip(feats1, feats2, self.fusion_blocks):
            fused_feat = block(feat1, feat2)
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
        modality_drop_prob: float = 0.0,
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
        self.modality_drop_prob = modality_drop_prob
        
        # Build the feature fusion module
        if self.fusion_module_cfg is not None:
            self.fusion_module = MODELS.build(self.fusion_module_cfg)
        else:
            # Default fusion module using SimpleChannelFusion
            # Get the output channels from neck config (in_channels)
            in_channels = neck['in_channels']  # e.g., [512, 1024, 2048]
            self.fusion_module = SimpleChannelFusion(
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
        img2 = batch_inputs[:, 3:, :, :]  # Second modality (e.g., Thermal/Depth)
        
        # 2. 随机模态丢弃逻辑，用于提升在模态丢失情况下单模态推理的鲁棒性 (仅在训练模式下生效)
        if self.training and self.modality_drop_prob > 0:
            # 生成一个随机数
            rand_val = torch.rand(1).item()
            
            # 策略：
            # 0 ~ p: 丢弃 img1 (可见光丢失)
            # p ~ 2p: 丢弃 img2 (SAR丢失)
            # 2p ~ 1: 两个都保留
            # 注意：我们要避免同时丢弃两个模态，否则没法训练
            
            if rand_val < self.modality_drop_prob:
                # 丢弃modality1 (将 tensor 设为全0)
                # 使用 zeros_like 保持设备(device)和类型(dtype)一致
                img1 = torch.zeros_like(img1)
                
            elif rand_val < (2 * self.modality_drop_prob):
                # 丢弃 modality2
                img2 = torch.zeros_like(img2)

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
