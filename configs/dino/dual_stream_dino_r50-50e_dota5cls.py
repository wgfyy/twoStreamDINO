# 双流DINO配置 - 可见光+SAR融合检测 (DOTA 5类)
# 继承自 12 epoch 的基础配置 (已改为 AdaptiveMultiScaleFusion3)
# 扩展为 50 epochs 以追求极致收敛

_base_ = './dual_stream_dino_r50-12e_dota5cls.py'

max_epochs = 50

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    # 保持 Warmup 不变 (注意: 多卡训练时可能需要在启动脚本中把 end=200 覆盖掉这里的 end=2000)
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=2000),
    # 学习率衰减策略
    # 按照 12e (8, 11) 和 36e (24, 33) 的比例推算
    # 50e 推荐: 40 (80%), 48 (96%)
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40, 48],
        gamma=0.1)
]
