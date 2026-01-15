#!/bin/bash
# 8卡 RTX 5090 专用训练脚本
# 优化策略: Close Accumulation, Scale LR, Reduce Workers

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 基础配置
CONFIG="configs/dino/dual_stream_dino_r50-36e_dota5cls.py"
WORK_DIR="work_dirs/dual_stream_dino_r50-36e_dota5cls-adaptivemultiscale2-GN_8gpu"
GPUS=8
PORT=29600

echo "=========================================="
echo "启动 8卡 RTX 5090 全速训练"
echo "Effective Batch Size: 32 (8 GPUs * 4 Imgs)"
echo "------------------------------------------"
echo "参数调整:"
echo "1. Accumulative Counts: 1 (关闭累积)"
echo "2. Learning Rate: 0.0004 (线性缩放+微调)"
echo "3. Num Workers: 4 (避免 128 进程炸内存)"
echo "4. Log Interval: 10 (适应快速迭代)"
echo "=========================================="

# 动态参数覆盖
# optim_wrapper.accumulative_counts=1 : 关闭梯度累积
# optim_wrapper.optimizer.lr=0.00035 : 适配 Batch=32 的学习率
# train_dataloader.num_workers=4 : 降低CPU负载
# default_hooks.logger.interval=10 : 提高日志频率
CFG_OPTIONS="--cfg-options optim_wrapper.accumulative_counts=1 optim_wrapper.optimizer.lr=0.0004 train_dataloader.num_workers=4 default_hooks.logger.interval=10"

# 启动训练
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR $CFG_OPTIONS
