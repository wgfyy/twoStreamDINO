#!/bin/bash
# 双流DINO分布式训练启动脚本 (AutoDL多卡)

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet

# 默认GPU数量 (可以通过第一个参数传入，例如 ./start_dist_training.sh 4)
# 如果没有传入参数，默认使用 2 卡
GPUS=${1:-2}
PORT=${PORT:-29501} # 稍微改个端口避免冲突

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置文件 (与 start_training.sh 保持一致)
CONFIG="configs/dino/dual_stream_dino_r50-12e_M4SAR6cls.py"
WORK_DIR="work_dirs/dual_stream_dino_r50-12e_M4SAR6cls-adaptivemultiscale3-GN-2gpus"

echo "=========================================="
echo "双流DINO分布式训练 - 可见光+SAR融合检测"
echo "=========================================="
echo "使用GPU数量: $GPUS"
echo "配置文件: $CONFIG"
echo "工作目录: $WORK_DIR"
echo "Master Port: $PORT"
echo ""
echo "当前脚本会自动调用 tools/dist_train.sh"
echo "=========================================="

# 设置 MASTER_PORT 环境变量传给 tools/dist_train.sh (虽然该脚本内部也有默认值，但我们在外部控制更稳妥)
export PORT=$PORT

# 动态配置覆盖
# 场景: 2卡训练，目标 Effective Batch Size = 8
# 1. optim_wrapper.accumulative_counts=1 (4 batch * 2 gpus * 1 accum = 8 images)
# 2. optim_wrapper.optimizer.lr=0.0004 (相比单卡Accum=2，DDP平均了梯度，需翻倍LR补偿以保持更新力度)
CFG_OPTIONS="--cfg-options optim_wrapper.accumulative_counts=1 optim_wrapper.optimizer.lr=0.0004"  # 0.0002 * $GPUS

echo "已应用多卡适配参数: Accum=1, LR=0.0004"
echo "==========================开始训练=============================="

# 启动分布式训练
# tools/dist_train.sh 的参数格式: <CONFIG> <GPUS> [train.py arguments]
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR $CFG_OPTIONS
