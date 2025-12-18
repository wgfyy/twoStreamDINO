from mmdet.apis import init_detector, inference_detector
import mmcv

# 指定配置文件和权重文件
config_file = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试一张图片 (使用官方 demo 图片)
img = 'demo/demo.jpg'
result = inference_detector(model, img)

print("Success! Inference completed.")
print(f"Result structure: {type(result)}")