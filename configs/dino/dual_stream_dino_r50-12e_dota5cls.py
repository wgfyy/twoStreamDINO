# 双流DINO配置 - 可见光+SAR融合检测 (DOTA 5类)
# 数据格式: COCO格式
# 目标类别: plane, ship, harbor, bridge, helicopter
# 
# 迁移学习策略:
#   1. 加载预训练的SimpleChannelFusion双流DINO权重 (backbone已适应光学+SAR)
#   2. 冻结两个backbone (frozen_stages=4)，专注学习新的注意力融合模块
#   3. 使用线性warmup让新模块稳定训练

_base_ = ['../_base_/default_runtime.py']

# ==================== 预训练权重 ====================
# 加载之前训练好的SimpleChannelFusion双流DINO模型
# 这会加载backbone, backbone2, neck, encoder, decoder, bbox_head等所有权重
# 新的fusion_module会随机初始化（因为结构不同）
# load_from = 'work_dirs/dual_stream_dino_r50-36e_dota5cls/epoch_36.pth'

# ==================== 数据集类别定义 ====================
# DOTA数据集的5个目标类别
CLASSES = ('plane', 'ship', 'harbor', 'bridge', 'helicopter')
num_classes = 5

# ==================== 模型配置 ====================
model = dict(
    type='DualStreamDINO',
    modality_drop_prob=0.0,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    
    # 双模态数据预处理器
    data_preprocessor=dict(
        type='DualModalDataPreprocessor',
        # 模态1 (可见光/RGB) 归一化参数
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # 模态2 (SAR) 归一化参数 - 如果SAR是灰度转3通道，可用相同参数
        mean2=[123.675, 116.28, 103.53],
        std2=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        bgr_to_rgb2=True,
        pad_size_divisor=1),
    
    # 第一个backbone (可见光)
    # 迁移学习策略: 使用预训练的双流DINO backbone权重，冻结全部4个stage
    # 让模型专注于学习融合模块，加速收敛
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,  
        norm_cfg=dict(type='BN', requires_grad=False),  # BN固定不训练
        norm_eval=True,   # 使用预训练的BN统计量
        style='pytorch',
        # 注意: init_cfg 会被 load_from 覆盖，这里保留作为fallback
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    
    # 第二个backbone (SAR)
    # 同样使用预训练的双流DINO backbone权重，冻结全部4个stage
    backbone2=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    # ==================== 融合模块配置 ====================
    # 可选方案:
    # 1. SimpleChannelFusion: 简单通道拼接+卷积 (Baseline, mAP=0.398)
    # 2. BiCrossAttentionFusion: 空间交叉注意力（4x下采样损失小目标）
    # 3. ChannelAttentionFusion: 通道交叉注意力（内存高效但效果一般）
    # 4. HybridAttentionFusion: 混合注意力（收敛慢，小目标损失）
    # 5. AdaptiveMultiScaleFusion: 【推荐】多尺度自适应融合+强残差
    # 6. ProtectedSpatialAttentionFusion: 保护小目标的局部空间融合
    
    # 方案1: 简单融合 Baseline（最快，mAP=0.398）
    # fusion_module=dict(
    #     type='SimpleChannelFusion',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[512, 1024, 2048],
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #     act_cfg=dict(type='ReLU', inplace=True)
    # ),
    
    # 方案5: 【推荐】多尺度自适应融合 + 强残差
    # - 低层(P3): 简单Cat保护小目标
    # - 高层(P4,P5): 轻量通道注意力
    # - 强残差: 保证不低于Baseline
    fusion_module=dict(
        type='AdaptiveMultiScaleFusion',
        in_channels=[512, 1024, 2048],
        out_channels=[512, 1024, 2048],
        low_level_indices=[0],       # 第一层(stride=8)用简单融合
        downsample_ratio=2,          # 减小下采样（如果用空间注意力）
        channel_reduction=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True)
    ),
    
    # 方案6: 保护小目标的局部空间注意力（无下采样）
    # fusion_module=dict(
    #     type='ProtectedSpatialAttentionFusion',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[512, 1024, 2048],
    #     kernel_size=7,               # 局部注意力窗口
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #     act_cfg=dict(type='ReLU', inplace=True)
    # ),
    
    # 旧方案（已验证效果不佳）
    # 方案2: 空间注意力（4x下采样导致mAP_s下降）
    # fusion_module=dict(
    #     type='BiCrossAttentionFusion',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[512, 1024, 2048],
    #     downsample_ratio=4,
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #     act_cfg=dict(type='ReLU', inplace=True)
    # ),
    
    # 方案3: 通道注意力（缺乏空间感知，效果不佳）
    # fusion_module=dict(
    #     type='ChannelAttentionFusion',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[512, 1024, 2048],
    #     reduction=4,
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #     act_cfg=dict(type='ReLU', inplace=True)
    # ),
    
    # 方案4: 混合注意力（收敛慢，前期落后Baseline）
    # fusion_module=dict(
    #     type='HybridAttentionFusion',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[512, 1024, 2048],
    #     downsample_ratio=4,
    #     channel_reduction=4,
    #     fusion_weight=0.5,
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    #     act_cfg=dict(type='ReLU', inplace=True)
    # ),
    
    # Neck
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    
    # Transformer Encoder
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    
    # Transformer Decoder
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    
    # 位置编码
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    
    # 检测头 - 5类
    bbox_head=dict(
        type='DINOHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    
    # 去噪配置
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    
    # 训练和测试设置
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# ==================== 数据集配置 ====================
# 直接使用 DualModalCocoDataset，无需额外的DOTA数据集类
dataset_type = 'DualModalCocoDataset'

# 数据集路径
data_root = '/home/wgfyy/datasets/DOTA-split-dualmodal-coco/'

backend_args = None

# 训练数据增强流水线
train_pipeline = [
    dict(type='LoadDualModalImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DualModalRandomFlip', prob=0.5),
    dict(
        type='DualModalRandomChoiceResize',
        scales=[(480, 800), (512, 800), (544, 800), (576, 800),
                (608, 800), (640, 800), (672, 800), (704, 800),
                (736, 800), (768, 800), (800, 800)],
        keep_ratio=True),
    dict(type='PackDualModalDetInputs')
]

# 测试数据流水线
test_pipeline = [
    dict(type='LoadDualModalImagesFromFile', backend_args=backend_args),
    dict(type='DualModalResize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDualModalDetInputs',
        meta_keys=('img_id', 'img_path', 'img_path2', 'ori_shape',
                   'img_shape', 'scale_factor'))
]

# 数据加载器配置
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(
            img='images/optical/',   # 可见光图像目录
            img2='images/sar/'       # SAR图像目录
        ),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(
            img='images/optical/',
            img2='images/sar/'
        ),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# ==================== 训练配置 ====================
# 使用FP32训练保证数值稳定性
# 差异化学习率策略:
#   - backbone (可见光): lr_mult=0.1，预训练特征已适配，轻微微调
#   - backbone2 (SAR): lr_mult=0.5，SAR特征差异大，需要更多学习
optim_wrapper = dict(
    type='OptimWrapper',  # 使用FP32训练
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 基础学习率
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    # 梯度累积: 每4个batch累积一次梯度更新
    # 有效batch_size = 2 × 4 = 8
    accumulative_counts=4,
    paramwise_cfg=dict(
        custom_keys={
            # backbone已完全冻结(frozen_stages=4)，这里的lr_mult实际上不会生效
            # 但保留以防将来解冻部分层
            'backbone': dict(lr_mult=0.0),   # 可见光backbone: 完全冻结
            'backbone2': dict(lr_mult=0.0)   # SAR backbone: 完全冻结
        }))

# 学习率调度
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    # 线性Warmup: 从0.1倍学习率开始，2000 iters内线性上升到基础学习率
    # 目的: 让新的融合模块在初期稳定探索参数空间
    dict(
        type='LinearLR',
        start_factor=0.1,  # 初始学习率 = 0.1 * base_lr = 0.00001
        by_epoch=False,
        begin=0,
        end=2000),  # warmup 2000 iters
    # 主学习率调度: 在epoch 8和11时衰减
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 自动学习率缩放
auto_scale_lr = dict(base_batch_size=16)

# 默认hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
