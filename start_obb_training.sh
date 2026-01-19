#!/bin/bash
# OBB (Oriented Bounding Box) 训练脚本
# 配置文件: dual_stream_dino_r50-50e_M4SAR6cls_obb.py

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4  # 根据实际GPU数量调整

# 配置文件路径
CONFIG="configs/dino/dual_stream_dino_r50-50e_M4SAR6cls_obb.py"

# 工作目录（保存日志和检查点）
WORK_DIR="work_dirs/dual_stream_dino_r50-50e_M4SAR6cls_obb-5gpu"

# GPU数量（根据实际情况修改）
NUM_GPUS=5

# 创建工作目录
mkdir -p ${WORK_DIR}

echo "=========================================="
echo "OBB Training Script"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Work Dir: ${WORK_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "${CONFIG}" ]; then
    echo "Error: Config file not found: ${CONFIG}"
    exit 1
fi

# 开始训练
if [ ${NUM_GPUS} -eq 1 ]; then
    # 单GPU训练
    echo "Starting single GPU training..."
    python tools/train.py ${CONFIG} \
        --work-dir ${WORK_DIR} \
        2>&1 | tee ${WORK_DIR}/train_$(date +%Y%m%d_%H%M%S).log
else
    # 多GPU分布式训练
    echo "Starting distributed training with ${NUM_GPUS} GPUs..."
    bash tools/dist_train.sh ${CONFIG} ${NUM_GPUS} \
        --work-dir ${WORK_DIR} \
        2>&1 | tee ${WORK_DIR}/train_$(date +%Y%m%d_%H%M%S).log
fi

