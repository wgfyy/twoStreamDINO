# 双流DINO配置 - 可见光+SAR融合检测 (DOTA 5类)
# 数据格式: COCO格式
# 目标类别: plane, ship, harbor, bridge, helicopter
_base_ = './dual_stream_dino_r50-12e_M4SAR6cls_obb.py'
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    # 1. Warmup 保持不变 (非常重要，特别是对于 Transformer)
    dict(
        type='LinearLR',
        start_factor=0.001, # 建议从更低开始 (0.1 -> 0.001)，Transformer 对初始 LR 很敏感
        by_epoch=False,
        begin=0,
        end=1000),
    
    # 2. 切换为 CosineAnnealingLR
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,       # 对应 50 epoch
        by_epoch=True,
        begin=0,
        end=max_epochs,
        eta_min=1e-6,  # 最终学习率降为0 (或者 base_lr * 0.01)
        convert_to_iter_based=True # 推荐基于 iter 更新，更平滑
    )
]
