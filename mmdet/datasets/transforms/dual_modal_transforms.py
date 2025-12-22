# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import get

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDualModalImagesFromFile(BaseTransform):
    """Load dual-modal images from file.
    
    This transform loads two images (from two modalities) and stacks them
    for dual-stream detection models.
    
    Required Keys:
        - img_path: Path to the first modality image
        - img_path2: Path to the second modality image
    
    Modified Keys:
        - img: Stacked image array with shape (H, W, 6) 
        - img_shape
        - ori_shape
    
    Added Keys:
        - img2_path: Path to second modality image (for reference)
    
    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments for file backend.
            Defaults to None.
        img2_color_type (str): Color type for second modality.
            Defaults to 'color'.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = 'color',
        imdecode_backend: str = 'cv2',
        backend_args: dict = None,
        img2_color_type: str = 'color',
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.img2_color_type = img2_color_type

    def transform(self, results: dict) -> dict:
        """Transform function to load dual-modal images.

        Args:
            results (dict): Result dict containing img_path and img_path2.

        Returns:
            dict: The dict contains loaded images and meta information.
        """
        # Load first modality image
        img_path = results['img_path']
        img_bytes = get(img_path, backend_args=self.backend_args)
        img1 = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        
        # Load second modality image
        img_path2 = results['img_path2']
        img_bytes2 = get(img_path2, backend_args=self.backend_args)
        img2 = mmcv.imfrombytes(
            img_bytes2, flag=self.img2_color_type, backend=self.imdecode_backend)
        
        # Handle grayscale images by converting to 3-channel
        if len(img2.shape) == 2:
            img2 = np.stack([img2, img2, img2], axis=-1)
        elif img2.shape[2] == 1:
            img2 = np.concatenate([img2, img2, img2], axis=-1)
        
        # Ensure both images have the same size
        if img1.shape[:2] != img2.shape[:2]:
            img2 = mmcv.imresize(img2, (img1.shape[1], img1.shape[0]))
        
        # Stack images along channel dimension: (H, W, 6)
        img = np.concatenate([img1, img2], axis=-1)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_path2'] = img_path2
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f"img2_color_type='{self.img2_color_type}', "
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class DualModalRandomFlip(BaseTransform):
    """Random flip for dual-modal images.
    
    This transform applies the same random flip to both modalities
    stored in a stacked image tensor.
    
    Required Keys:
        - img: Stacked dual-modal image (H, W, 6)
        - gt_bboxes (optional)
        - gt_masks (optional)
        
    Modified Keys:
        - img
        - gt_bboxes
        - gt_masks
        
    Added Keys:
        - flip
        - flip_direction
    
    Args:
        prob (float): Probability of flipping. Defaults to 0.5.
        direction (str): Flip direction. Options are 'horizontal',
            'vertical', 'diagonal'. Defaults to 'horizontal'.
    """

    def __init__(
        self,
        prob: float = 0.5,
        direction: str = 'horizontal',
    ) -> None:
        assert 0 <= prob <= 1
        assert direction in ['horizontal', 'vertical', 'diagonal']
        self.prob = prob
        self.direction = direction

    def transform(self, results: dict) -> dict:
        """Transform function to flip dual-modal images.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Flipped results.
        """
        if np.random.rand() < self.prob:
            img = results['img']
            
            # Split into two modalities
            img1 = img[..., :3]
            img2 = img[..., 3:]
            
            # Flip both modalities
            if self.direction == 'horizontal':
                img1 = mmcv.imflip(img1, direction='horizontal')
                img2 = mmcv.imflip(img2, direction='horizontal')
            elif self.direction == 'vertical':
                img1 = mmcv.imflip(img1, direction='vertical')
                img2 = mmcv.imflip(img2, direction='vertical')
            else:  # diagonal
                img1 = mmcv.imflip(img1, direction='horizontal')
                img1 = mmcv.imflip(img1, direction='vertical')
                img2 = mmcv.imflip(img2, direction='horizontal')
                img2 = mmcv.imflip(img2, direction='vertical')
            
            # Stack back
            results['img'] = np.concatenate([img1, img2], axis=-1)
            results['flip'] = True
            results['flip_direction'] = self.direction
            
            # Flip bboxes - use MMDetection 3.x Box API
            if 'gt_bboxes' in results:
                h, w = results['img_shape']
                bboxes = results['gt_bboxes']
                # MMDetection 3.x uses BaseBoxes, call flip_ method (in-place)
                bboxes.flip_(img_shape=(h, w), direction=self.direction)
                results['gt_bboxes'] = bboxes
        else:
            results['flip'] = False
            results['flip_direction'] = None
        
        return results


@TRANSFORMS.register_module()
class DualModalResize(BaseTransform):
    """Resize dual-modal images.
    
    This transform resizes both modalities consistently.
    
    Required Keys:
        - img: Stacked dual-modal image (H, W, 6)
        - gt_bboxes (optional)
        
    Modified Keys:
        - img
        - img_shape
        - gt_bboxes
        
    Added Keys:
        - scale
        - scale_factor
        - keep_ratio
    
    Args:
        scale (tuple): Target size (h, w).
        keep_ratio (bool): Whether to keep aspect ratio. Defaults to True.
        interpolation (str): Interpolation method. Defaults to 'bilinear'.
    """

    def __init__(
        self,
        scale: Tuple[int, int],
        keep_ratio: bool = True,
        interpolation: str = 'bilinear',
    ) -> None:
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transform function to resize dual-modal images.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Resized results.
        """
        img = results['img']
        h, w = img.shape[:2]
        
        # Split into two modalities
        img1 = img[..., :3]
        img2 = img[..., 3:]
        
        if self.keep_ratio:
            new_size, scale_factor = mmcv.rescale_size(
                (w, h), self.scale, return_scale=True)
            img1 = mmcv.imresize(
                img1, new_size, interpolation=self.interpolation)
            img2 = mmcv.imresize(
                img2, new_size, interpolation=self.interpolation)
            new_h, new_w = img1.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img1 = mmcv.imresize(
                img1, self.scale[::-1], interpolation=self.interpolation)
            img2 = mmcv.imresize(
                img2, self.scale[::-1], interpolation=self.interpolation)
            new_h, new_w = self.scale
            w_scale = self.scale[1] / w
            h_scale = self.scale[0] / h
        
        # Stack back
        results['img'] = np.concatenate([img1, img2], axis=-1)
        results['img_shape'] = (new_h, new_w)
        results['scale'] = (new_h, new_w)
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = self.keep_ratio
        
        # Scale bboxes - compatible with MMDetection 3.x Box types
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            # Use the rescale method if bboxes is a BaseBoxes instance
            if hasattr(bboxes, 'rescale_'):
                bboxes.rescale_([w_scale, h_scale])
                results['gt_bboxes'] = bboxes
            else:
                # Fallback for numpy array or tensor
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w_scale
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h_scale
                results['gt_bboxes'] = bboxes
        
        return results


@TRANSFORMS.register_module()
class PackDualModalDetInputs(BaseTransform):
    """Pack the inputs data for dual-modal detection.
    
    This transform packs the dual-modal image into the correct format
    for the data preprocessor.
    
    Required Keys:
        - img: Stacked dual-modal image (H, W, 6)
        - gt_bboxes (optional)
        - gt_bboxes_labels (optional)
        
    Added Keys:
        - inputs: Tensor of shape (6, H, W)
        - data_samples: DetDataSample
    
    Args:
        meta_keys (tuple): Meta keys to be collected. Defaults to common keys.
    """

    def __init__(
        self,
        meta_keys: tuple = ('img_id', 'img_path', 'img_path2', 'ori_shape',
                           'img_shape', 'scale_factor', 'flip',
                           'flip_direction'),
    ) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Transform function to pack dual-modal inputs.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Packed results with 'inputs' and 'data_samples'.
        """
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        packed_results = dict()
        
        # Pack image: (H, W, 6) -> (6, H, W)
        img = results['img']
        if len(img.shape) == 3:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
        packed_results['inputs'] = torch.from_numpy(img)
        
        # Create data sample
        data_sample = DetDataSample()
        
        # Pack instance data
        instance_data = InstanceData()
        
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            # Handle different bbox types (BaseBoxes, Tensor, ndarray)
            if hasattr(gt_bboxes, 'tensor'):
                # It's a BaseBoxes instance (e.g., HorizontalBoxes)
                instance_data.bboxes = gt_bboxes
            elif isinstance(gt_bboxes, torch.Tensor):
                instance_data.bboxes = gt_bboxes
            else:
                # Convert numpy array to tensor
                instance_data.bboxes = torch.from_numpy(gt_bboxes).float()
        
        if 'gt_bboxes_labels' in results:
            gt_labels = results['gt_bboxes_labels']
            if not isinstance(gt_labels, torch.Tensor):
                gt_labels = torch.from_numpy(np.array(gt_labels)).long()
            instance_data.labels = gt_labels
        
        if 'gt_ignore_flags' in results:
            ignore_flags = results['gt_ignore_flags']
            if not isinstance(ignore_flags, torch.Tensor):
                ignore_flags = torch.from_numpy(np.array(ignore_flags)).bool()
            instance_data.ignore_flags = ignore_flags
        
        data_sample.gt_instances = instance_data
        
        # Pack meta info
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        
        packed_results['data_samples'] = data_sample
        
        return packed_results


@TRANSFORMS.register_module()
class DualModalRandomChoiceResize(BaseTransform):
    """Random resize dual-modal images from a list of scales.
    
    Args:
        scales (list[tuple]): List of scales, each scale is (h, w).
        keep_ratio (bool): Whether to keep aspect ratio. Defaults to True.
    """

    def __init__(
        self,
        scales: list,
        keep_ratio: bool = True,
    ) -> None:
        self.scales = scales
        self.keep_ratio = keep_ratio

    def transform(self, results: dict) -> dict:
        """Transform function."""
        scale = self.scales[np.random.randint(len(self.scales))]
        
        resize_transform = DualModalResize(
            scale=scale,
            keep_ratio=self.keep_ratio,
        )
        return resize_transform(results)
