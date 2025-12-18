_base_ = [
    '../_base_/default_runtime.py'
]

# ==================== Model Config ====================
model = dict(
    type='DualStreamDINO',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    
    # Data preprocessor for dual-modal inputs
    data_preprocessor=dict(
        type='DualModalDataPreprocessor',
        # Modality 1 (RGB) normalization
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # Modality 2 (e.g., Thermal/Depth) normalization
        # If None, uses the same as modality 1
        mean2=[123.675, 116.28, 103.53],  # Adjust based on your second modality
        std2=[58.395, 57.12, 57.375],      # Adjust based on your second modality
        bgr_to_rgb=True,
        bgr_to_rgb2=True,  # Set to True if second modality is also BGR
        pad_size_divisor=1),
    
    # First backbone (Modality 1, e.g., RGB)
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    
    # Second backbone (Modality 2, e.g., Thermal/Depth)
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
    
    # Feature fusion module (optional, uses default if not specified)
    # fusion_module=dict(
    #     type='FeatureFusionModule',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[512, 1024, 2048],
    #     norm_cfg=dict(type='BN'),
    #     act_cfg=dict(type='ReLU', inplace=True)),
    
    # Neck (same as original DINO)
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    
    # Encoder (same as original DINO)
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),
    
    # Decoder (same as original DINO)
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    
    # Positional encoding (same as original DINO)
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    
    # Bbox head (same as original DINO)
    bbox_head=dict(
        type='DINOHead',
        num_classes=80,  # Change to your number of classes
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    
    # Denoising config (same as original DINO)
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),
    
    # Training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# ==================== Dataset Config ====================
dataset_type = 'DualModalCocoDataset'
data_root = 'data/dual_modal_coco/'  # Change to your data root

backend_args = None

# Training pipeline for dual-modal images
train_pipeline = [
    dict(type='LoadDualModalImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DualModalRandomFlip', prob=0.5),
    dict(
        type='DualModalRandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='PackDualModalDetInputs')
]

# Testing pipeline for dual-modal images
test_pipeline = [
    dict(type='LoadDualModalImagesFromFile', backend_args=backend_args),
    dict(type='DualModalResize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDualModalDetInputs',
        meta_keys=('img_id', 'img_path', 'img_path2', 'ori_shape',
                   'img_shape', 'scale_factor'))
]

# Dataloader config
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(
            img='train/rgb/',      # First modality images (e.g., RGB)
            img2='train/thermal/'  # Second modality images (e.g., Thermal)
        ),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(
            img='val/rgb/',
            img2='val/thermal/'
        ),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# ==================== Training Config ====================
# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone2': dict(lr_mult=0.1)  # Same learning rate for second backbone
        }))

# Learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# Auto scale learning rate
auto_scale_lr = dict(base_batch_size=16)
