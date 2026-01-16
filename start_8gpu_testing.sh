#!/bin/bash
# 8卡测试脚本：分布式推理，汇总 COCO mAP

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 基础配置（按需修改）
CONFIG="configs/dino/dual_stream_dino_r50-12e_M4SAR6cls.py"
WORK_DIR="work_dirs/dual_stream_dino_r50-12e_M4SAR6cls-adaptivemultiscale3-GN_8gpu"
CKPT="${WORK_DIR}/epoch_7.pth"          # 默认测试最后模型，可改成 epoch_*.pth 或 latest.pth
OUT_DIR="${WORK_DIR}_test_epoch7"       # 结果输出目录
GPUS=8
PORT=29600

echo "=========================================="
echo "启动 8 卡分布式测试"
echo "Config : ${CONFIG}"
echo "Checkpoint : ${CKPT}"
echo "Work Dir : ${OUT_DIR}"
echo "=========================================="

# 分布式测试
export PORT=${PORT}
bash tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS} --work-dir ${OUT_DIR}
# 注意：master端口由环境变量 PORT 控制，dist_test.sh 内部会读取 PORT 传给 torch.distributed.launch。
# 不要把 --master-port 传给 test.py（它不接受该参数），只需预先导出 PORT 即可。
