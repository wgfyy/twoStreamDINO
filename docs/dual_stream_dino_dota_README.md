# Dual-Stream DINO 双模态目标检测

基于 MMDetection 的双流 DINO 目标检测器，用于融合可见光（Optical）和 SAR 图像进行目标检测。

## 📁 文件结构

```
mmdetection/
├── configs/dino/
│   └── dual_stream_dino_r50_dota5cls.py    # 5类DOTA数据集配置
├── mmdet/
│   ├── datasets/
│   │   ├── dual_modal_coco.py              # 双模态COCO数据集（核心）
│   │   └── transforms/
│   │       └── dual_modal_transforms.py    # 双模态数据增强
│   └── models/
│       ├── detectors/
│       │   └── dual_stream_dino.py         # 双流DINO检测器
│       └── data_preprocessors/
│           └── dual_modal_data_preprocessor.py
└── tools/
    ├── verify_dual_modal_dataset.py        # 数据集验证脚本
    ├── dual_stream_inference.py            # 推理脚本
    └── dataset_converters/
        └── dota_to_coco_5cls.py            # DOTA转COCO脚本（5类）
```

## 🗂️ 数据集准备

### 方案一：已有COCO格式数据（推荐）

如果你的数据已经是COCO格式，按如下结构组织：

```
data/dota_dual_modal/
├── annotations/
│   ├── train.json              # 训练集标注 (COCO格式)
│   └── val.json                # 验证集标注 (COCO格式)
└── images/
    ├── optical/                # 可见光图像
    │   ├── P0001__1__0___0.png
    │   └── ...
    └── sar/                    # SAR图像（与可见光同名配对）
        ├── P0001__1__0___0.png
        └── ...
```

**重要**: 可见光和SAR图像必须**同名**才能正确配对！

### 方案二：从DOTA格式转换

如果你的数据是DOTA原始格式，使用转换脚本：

#### 1. 原始DOTA数据结构

```
data/
├── optical/                    # 可见光图像及标注
│   ├── images/
│   │   ├── P0001__1__0___0.png
│   │   └── ...
│   └── annfiles/               # DOTA格式标注 (.txt)
│       ├── P0001__1__0___0.txt
│       └── ...
└── sar/                        # SAR图像（无需标注，共用optical的标注）
    └── images/
        ├── P0001__1__0___0.png
        └── ...
```

#### 2. DOTA标注文件格式 (.txt)

```
x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
188.0 1019.0 259.0 1023.0 254.0 1073.0 183.0 1069.0 plane 0
1059.0 1011.0 1115.0 975.0 1152.0 1050.0 1096.0 1086.0 ship 0
```

#### 3. 运行转换脚本

```bash
cd mmdetection

# 转换训练集
python tools/dataset_converters/dota_to_coco_5cls.py \
    data/optical/train \
    data/dota_dual_modal/annotations/train.json

# 转换验证集  
python tools/dataset_converters/dota_to_coco_5cls.py \
    data/optical/val \
    data/dota_dual_modal/annotations/val.json

# 然后创建符号链接或复制图像
mkdir -p data/dota_dual_modal/images
ln -s $(pwd)/data/optical/train/images data/dota_dual_modal/images/optical
ln -s $(pwd)/data/sar/train/images data/dota_dual_modal/images/sar
```

### 验证数据集

```bash
python tools/verify_dual_modal_dataset.py configs/dino/dual_stream_dino_r50_dota5cls.py
```

## 🚀 训练模型

### 单GPU训练

```bash
python tools/train.py configs/dino/dual_stream_dino_r50_dota5cls.py \
    --work-dir work_dirs/dual_stream_dino_r50_dota5cls
```

### 多GPU训练

```bash
# 4卡训练
bash tools/dist_train.sh configs/dino/dual_stream_dino_r50_dota5cls.py 4 \
    --work-dir work_dirs/dual_stream_dino_r50_dota5cls
```

### 恢复训练

```bash
python tools/train.py configs/dino/dual_stream_dino_r50_dota5cls.py \
    --resume work_dirs/dual_stream_dino_r50_dota5cls/latest.pth
```

## 📊 测试评估

```bash
python tools/test.py configs/dino/dual_stream_dino_r50_dota5cls.py \
    work_dirs/dual_stream_dino_r50_dota5cls/epoch_36.pth
```

## 🔍 推理

### 单对图像推理

```bash
python tools/dual_stream_inference.py \
    configs/dino/dual_stream_dino_r50_dota5cls.py \
    work_dirs/dual_stream_dino_r50_dota5cls/epoch_36.pth \
    --optical demo/optical.png \
    --sar demo/sar.png \
    --out-file results/result.png \
    --score-thr 0.3
```

### 批量推理

```bash
python tools/dual_stream_inference.py \
    configs/dino/dual_stream_dino_r50_dota5cls.py \
    work_dirs/dual_stream_dino_r50_dota5cls/epoch_36.pth \
    --optical-dir data/test/optical \
    --sar-dir data/test/sar \
    --out-dir results/
```

## ⚙️ 配置说明

### 修改数据路径

编辑 `configs/dino/dual_stream_dino_r50_dota5cls.py`:

```python
# 修改数据根目录
data_root = 'data/dota_dual_modal/'

# 修改标注文件和图像路径
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/train.json',
        data_prefix=dict(
            img='images/optical/',    # 可见光目录
            img2='images/sar/'        # SAR目录
        ),
    ))
```

### 支持的5个类别

```python
CLASSES = ('plane', 'ship', 'harbor', 'bridge', 'helicopter')
```

### 调整SAR图像归一化

根据你的SAR数据特性调整：

```python
data_preprocessor=dict(
    # SAR图像归一化参数（根据实际数据统计调整）
    mean2=[123.675, 116.28, 103.53],
    std2=[58.395, 57.12, 57.375],
)
```

### 调整训练参数

```python
# 批次大小
train_dataloader = dict(batch_size=2)

# 学习率
optim_wrapper = dict(optimizer=dict(lr=0.0001))

# 训练轮数
max_epochs = 36
```

## 📈 模型架构

```
Optical Image ──► Backbone1 (ResNet50) ──┐
                                          │
                                          ├──► Feature Fusion ──► Neck ──► DINO Transformer ──► Detection
                                          │
SAR Image ─────► Backbone2 (ResNet50) ──┘
```

## 🔧 故障排除

| 问题 | 解决方案 |
|------|----------|
| 内存不足 | 减小 `batch_size` 或图像尺寸 |
| 找不到对应SAR图像 | 确保SAR图像文件名与可见光相同 |
| SAR为灰度图 | 自动转换为3通道，无需处理 |
| 图像尺寸不匹配 | 自动调整SAR尺寸以匹配可见光 |

## 📝 引用

```bibtex
@article{zhang2022dino,
    title={DINO: DETR with Improved DeNoising Anchor Boxes},
    author={Zhang, Hao and others},
    journal={arXiv preprint arXiv:2203.03605},
    year={2022}
}

@inproceedings{xia2018dota,
    title={DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
    author={Xia, Gui-Song and others},
    booktitle={CVPR},
    year={2018}
}
```
