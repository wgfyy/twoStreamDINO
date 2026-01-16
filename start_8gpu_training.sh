#!/bin/bash
# 8卡 RTX 5090 专用训练脚本
# 优化策略: Close Accumulation, Scale LR, Reduce Workers

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 基础配置
CONFIG="configs/dino/dual_stream_dino_r50-50e_M4SAR6cls.py"
WORK_DIR="work_dirs/dual_stream_dino_r50-50e_M4SAR6cls-adaptivemultiscale3-GN_8gpu"
GPUS=8
PORT=29600

echo "=========================================="
echo "启动 8卡 RTX 5090 全速训练"
echo "Effective Batch Size: 48 (8 GPUs * 6 Imgs)"
echo "------------------------------------------"
echo "参数调整:"
echo "1. Accumulative Counts: 1 (关闭累积)"
echo "2. Learning Rate: 0.0003 (微调减半)"
echo "3. Num Workers: 10 (降低 worker 防止验证死锁)"
echo "=========================================="

# 动态参数覆盖
# optim_wrapper.accumulative_counts=1 : 关闭梯度累积
# optim_wrapper.optimizer.lr=0.0008 : 线性缩放 (Batch8->32, Scaling=4x, 0.0002*4=0.0008)
# param_scheduler.0.end=200 : 【关键修复】Warmup步数调整。
#    单卡Batch=8时，1epoch=767step，2000step=2.6epoch。
#    8卡Batch=32时，1epoch=191step，原2000step=10.5epoch(整个训练过程都在预热！)。
#    故需降为 200 (约1个epoch)。
# train_dataloader.num_workers=10 : 适配128核CPU (10 workers * 8 GPUs = 80进程，保留48核给训练主进程和系统，避免抢占)
# default_hooks.logger.interval=10 : 提高日志频率
CFG_OPTIONS="--cfg-options optim_wrapper.accumulative_counts=1 optim_wrapper.optimizer.lr=0.0003 param_scheduler.0.end=500 train_dataloader.num_workers=10"

# 启动训练
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR $CFG_OPTIONS
