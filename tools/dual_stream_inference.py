#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Inference script for Dual-Stream DINO on dual-modal images.

This script performs inference on paired optical and SAR images
using the DualStreamDINO detector.

Usage:
    # Single image pair inference
    python dual_stream_inference.py \
        configs/dino/dual_stream_dino-4scale_r50_8xb2-12e_dota.py \
        checkpoints/dual_stream_dino_dota.pth \
        --optical demo/optical.png \
        --sar demo/sar.png \
        --out-file demo/result.png
    
    # Batch inference on a directory
    python dual_stream_inference.py \
        configs/dino/dual_stream_dino-4scale_r50_8xb2-12e_dota.py \
        checkpoints/dual_stream_dino_dota.pth \
        --optical-dir data/test/optical \
        --sar-dir data/test/sar \
        --out-dir results/
"""

import argparse
import os
import os.path as osp
from glob import glob

import cv2
import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint

from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dual-Stream DINO Inference on Optical + SAR images')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    
    # Single image pair
    parser.add_argument('--optical', type=str, help='Path to optical image')
    parser.add_argument('--sar', type=str, help='Path to SAR image')
    parser.add_argument('--out-file', type=str, help='Output file path')
    
    # Batch inference
    parser.add_argument('--optical-dir', type=str, help='Directory of optical images')
    parser.add_argument('--sar-dir', type=str, help='Directory of SAR images')
    parser.add_argument('--out-dir', type=str, help='Output directory')
    
    # Inference options
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--wait-time', type=float, default=0, help='Display wait time')
    
    args = parser.parse_args()
    return args


def load_dual_modal_image(optical_path: str, sar_path: str):
    """Load and stack dual-modal images.
    
    Args:
        optical_path: Path to optical image
        sar_path: Path to SAR image
        
    Returns:
        np.ndarray: Stacked image (H, W, 6)
    """
    # Load optical image
    img_optical = mmcv.imread(optical_path, channel_order='bgr')
    
    # Load SAR image
    img_sar = mmcv.imread(sar_path, channel_order='bgr')
    
    # Handle grayscale SAR images
    if len(img_sar.shape) == 2:
        img_sar = np.stack([img_sar, img_sar, img_sar], axis=-1)
    elif img_sar.shape[2] == 1:
        img_sar = np.concatenate([img_sar, img_sar, img_sar], axis=-1)
    
    # Ensure same size
    if img_optical.shape[:2] != img_sar.shape[:2]:
        img_sar = mmcv.imresize(img_sar, (img_optical.shape[1], img_optical.shape[0]))
    
    # Stack along channel dimension
    img = np.concatenate([img_optical, img_sar], axis=-1)
    
    return img, img_optical


def preprocess_dual_modal(img: np.ndarray, cfg: Config, device: str):
    """Preprocess dual-modal image for inference.
    
    Args:
        img: Stacked dual-modal image (H, W, 6)
        cfg: Config
        device: Device string
        
    Returns:
        dict: Preprocessed data dict
    """
    # Get data preprocessor config
    data_preprocessor_cfg = cfg.model.data_preprocessor
    
    # Transpose to (6, H, W)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor


class DualStreamInferencer:
    """Inferencer for Dual-Stream DINO detector."""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        """Initialize the inferencer.
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to checkpoint file
            device: Device for inference
        """
        self.cfg = Config.fromfile(config_path)
        self.device = device
        
        # Build model
        self.model = MODELS.build(self.cfg.model)
        self.model.to(device)
        self.model.eval()
        
        # Load checkpoint
        load_checkpoint(self.model, checkpoint_path, map_location=device)
        
        # Get class names
        if hasattr(self.cfg, 'metainfo'):
            self.classes = self.cfg.metainfo.get('classes', None)
        else:
            self.classes = None
        
        # Initialize visualizer
        self.visualizer = None
        if 'visualizer' in self.cfg:
            self.visualizer = VISUALIZERS.build(self.cfg.visualizer)
            self.visualizer.dataset_meta = {'classes': self.classes}
    
    def inference(self, optical_path: str, sar_path: str, score_thr: float = 0.3):
        """Run inference on a pair of images.
        
        Args:
            optical_path: Path to optical image
            sar_path: Path to SAR image
            score_thr: Score threshold
            
        Returns:
            DetDataSample: Detection results
        """
        # Load images
        img, img_optical = load_dual_modal_image(optical_path, sar_path)
        ori_shape = img.shape[:2]
        
        # Preprocess
        with torch.no_grad():
            # Prepare data
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Create data sample
            data_sample = DetDataSample()
            data_sample.set_metainfo({
                'img_shape': ori_shape,
                'ori_shape': ori_shape,
                'scale_factor': (1.0, 1.0),
                'img_path': optical_path,
                'img_path2': sar_path,
            })
            
            # Run model forward
            data = {'inputs': img_tensor, 'data_samples': [data_sample]}
            data = self.model.data_preprocessor(data, training=False)
            
            # Forward
            results = self.model.predict(
                data['inputs'],
                data['data_samples'],
                rescale=True
            )
        
        result = results[0]
        
        # Filter by score threshold
        if score_thr > 0:
            pred_instances = result.pred_instances
            keep = pred_instances.scores > score_thr
            result.pred_instances = pred_instances[keep]
        
        return result, img_optical
    
    def visualize(
        self,
        img: np.ndarray,
        result: DetDataSample,
        out_file: str = None,
        show: bool = False,
        wait_time: float = 0
    ):
        """Visualize detection results.
        
        Args:
            img: Original optical image (for visualization)
            result: Detection results
            out_file: Output file path
            show: Whether to show the result
            wait_time: Display wait time
        """
        if self.visualizer is None:
            print('Warning: Visualizer not initialized')
            return
        
        self.visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=show,
            wait_time=wait_time,
            out_file=out_file
        )


def main():
    args = parse_args()
    
    # Initialize inferencer
    print(f'Loading model from {args.checkpoint}...')
    inferencer = DualStreamInferencer(
        args.config,
        args.checkpoint,
        device=args.device
    )
    print('Model loaded successfully!')
    
    # Single image pair inference
    if args.optical and args.sar:
        print(f'Processing: {args.optical} + {args.sar}')
        result, img = inferencer.inference(
            args.optical,
            args.sar,
            score_thr=args.score_thr
        )
        
        # Print results
        pred = result.pred_instances
        print(f'Detected {len(pred)} objects:')
        for i in range(len(pred)):
            bbox = pred.bboxes[i].cpu().numpy()
            score = pred.scores[i].cpu().item()
            label = pred.labels[i].cpu().item()
            print(f'  [{i}] Class: {label}, Score: {score:.3f}, '
                  f'BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]')
        
        # Visualize
        if args.out_file or args.show:
            inferencer.visualize(
                img,
                result,
                out_file=args.out_file,
                show=args.show,
                wait_time=args.wait_time
            )
            if args.out_file:
                print(f'Result saved to {args.out_file}')
    
    # Batch inference
    elif args.optical_dir and args.sar_dir:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
        
        # Get all optical images
        optical_files = sorted(glob(osp.join(args.optical_dir, '*.*')))
        
        for optical_path in optical_files:
            img_name = osp.basename(optical_path)
            sar_path = osp.join(args.sar_dir, img_name)
            
            if not osp.exists(sar_path):
                print(f'Warning: SAR image not found for {img_name}, skipping...')
                continue
            
            print(f'Processing: {img_name}')
            result, img = inferencer.inference(
                optical_path,
                sar_path,
                score_thr=args.score_thr
            )
            
            # Save result
            if args.out_dir:
                out_file = osp.join(args.out_dir, img_name)
                inferencer.visualize(img, result, out_file=out_file)
            
            # Show result
            if args.show:
                inferencer.visualize(
                    img, result, show=True, wait_time=args.wait_time
                )
        
        print(f'Done! Results saved to {args.out_dir}')
    
    else:
        print('Error: Please specify either:')
        print('  --optical and --sar for single image pair')
        print('  --optical-dir and --sar-dir for batch inference')


if __name__ == '__main__':
    main()
