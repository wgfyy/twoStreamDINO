#!/bin/bash
# 训练监控脚本

WORK_DIR="work_dirs/dual_stream_dino_r50_dota5cls"

echo "=========================================="
echo "双流DINO训练监控"
echo "=========================================="
echo ""

# 检查训练是否在运行
if pgrep -f "train.py.*dual_stream_dino_r50_dota5cls" > /dev/null; then
    echo "✓ 训练进程正在运行"
else
    echo "✗ 训练进程未运行"
fi

echo ""
echo "工作目录: $WORK_DIR"
echo ""

# 查看最新日志
if [ -f "$WORK_DIR"/*.log ]; then
    echo "最近的训练日志:"
    echo "------------------------------------------"
    tail -30 "$WORK_DIR"/*.log
else
    echo "日志文件尚未生成..."
fi

echo ""
echo "=========================================="
echo "提示:"
echo "  - 实时监控: watch -n 2 ./monitor_training.sh"
echo "  - 查看完整日志: tail -f $WORK_DIR/*.log"
echo "  - TensorBoard: tensorboard --logdir $WORK_DIR"
echo "=========================================="
