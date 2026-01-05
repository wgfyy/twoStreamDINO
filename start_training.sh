#!/bin/bash
# 双流DINO训练启动脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG="configs/dino/dual_stream_dino_r50_dota5cls.py"
WORK_DIR="work_dirs/dual_stream_dino_r50_dota5cls"

echo "=========================================="
echo "双流DINO训练 - 可见光+SAR融合检测"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "工作目录: $WORK_DIR"
echo ""
echo "数据集信息:"
echo "  - 训练样本: 6136 张"
echo "  - 标注数量: 90329 个"
echo "  - 类别: plane, ship, harbor, bridge, helicopter"
echo ""
echo "开始训练..."
echo "=========================================="

# python tools/train.py $CONFIG --work-dir $WORK_DIR
# 以下为不同训练配置的命令示例
# 双流DINO cat+conv融合 12轮训练
# python tools/train.py configs/dino/dual_stream_dino_r50-12e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-12e_dota5cls
# 双流DINO 36轮训练
# python tools/train.py configs/dino/dual_stream_dino_r50-36e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-36e_dota5cls
# 单流DINO 可见光 12轮训练
# python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_dota5cls.py --work-dir work_dirs/dino-4scale_r50_8xb2-12e_dota5cls-optical
# 单流DINO 可见光 36轮训练
# python tools/train.py configs/dino/dino-4scale_r50_8xb2-36e_dota5cls.py --work-dir work_dirs/dino-4scale_r50_8xb2-36e_dota5cls-optical
# 双流DINO 可见光+SAR 12轮训练  空间、通道混合交叉注意力融合+AdaptiveGatedFusion+SelectiveFeatureFusion+梯度累积
# python tools/train.py configs/dino/dual_stream_dino_r50-12e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-12e_dota5cls-dualcrossattn
# 双流DINO 可见光+SAR 12轮训练  空间、通道混合交叉注意力融合+AdaptiveGatedFusion+SelectiveFeatureFusion+GN+梯度累积
# python tools/train.py configs/dino/dual_stream_dino_r50-12e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-12e_dota5cls-dualcrossattn-GN
# 双流DINO 可见光+SAR 36轮训练  空间、通道混合交叉注意力融合+AdaptiveGatedFusion+SelectiveFeatureFusion+GN+梯度累积
# python tools/train.py configs/dino/dual_stream_dino_r50-36e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-36e_dota5cls-dualcrossattn-GN
# 冻结4层backbone，只训练融合层和输出层，load_from = 'work_dirs/dual_stream_dino_r50-36e_dota5cls/epoch_36.pth'
# python tools/train.py configs/dino/dual_stream_dino_r50-12e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-12e_dota5cls-dualcrossattn-GN-backbonefreezed
# 冻结4层backbone，只训练融合层和输出层，load_from = 'work_dirs/dual_stream_dino_r50-36e_dota5cls/epoch_36.pth'
# python tools/train.py configs/dino/dual_stream_dino_r50-36e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-36e_dota5cls-dualcrossattn-GN-backbonefreezed
# 双流DINO 可见光+SAR 36轮训练  通道交叉注意力融合+AdaptiveGatedFusion+SelectiveFeatureFusion+GN+梯度累积  先前混合注意力效果不佳,用于排除混合注意力问题,冻结第一层,不load_from
python tools/train.py configs/dino/dual_stream_dino_r50-36e_dota5cls.py --work-dir work_dirs/dual_stream_dino_r50-36e_dota5cls-channelcrossattn-GN