# 双流DINO配置 - 可见光+SAR融合检测 (DOTA 5类)
# 数据格式: COCO格式
# 目标类别: plane, ship, harbor, bridge, helicopter
_base_ = './dual_stream_dino_r50-12e_dota5cls.py'
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  # 初始学习率 = 0.1 * base_lr = 0.00001
        by_epoch=False,
        begin=0,
        end=2000),  # warmup 2000 iters
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]
