# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import ImgDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample


@MODELS.register_module()
class DualModalDataPreprocessor(ImgDataPreprocessor):
    """Data pre-processor for dual-modal detection tasks.
    
    This preprocessor handles two modal images simultaneously, stacking them
    along the channel dimension. Both modalities share the same normalization
    by default, but can have separate mean/std if specified.
    
    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels
            for modality 1. Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels for modality 1. Defaults to None.
        mean2 (Sequence[Number], optional): The pixel mean for modality 2.
            If None, uses the same as modality 1. Defaults to None.
        std2 (Sequence[Number], optional): The pixel std for modality 2.
            If None, uses the same as modality 1. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
        bgr_to_rgb2 (bool): whether to convert modality 2 from BGR to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        mean2: Sequence[Number] = None,
        std2: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        bgr_to_rgb2: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
    ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
        )
        
        # Mean and std for modality 2
        if mean2 is not None:
            assert std2 is not None, 'mean2 and std2 should be both None or both specified'
            self.register_buffer('mean2', torch.tensor(mean2).view(-1, 1, 1), False)
            self.register_buffer('std2', torch.tensor(std2).view(-1, 1, 1), False)
        else:
            self.register_buffer('mean2', self.mean.clone() if self.mean is not None else None, False)
            self.register_buffer('std2', self.std.clone() if self.std is not None else None, False)
        
        self.bgr_to_rgb2 = bgr_to_rgb2
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for dual-modal inputs.

        Args:
            data (dict): Data sampled from dataloader. Expected to contain
                'inputs' with dual-modal images, either as:
                - List of tensors with shape (6, H, W) 
                - Or dict with 'img' and 'img2' keys
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        batch_inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        
        # Handle dual-modal inputs
        # Expecting batch_inputs to be list of tensors with shape (6, H, W)
        # where first 3 channels are img1 and last 3 channels are img2
        if isinstance(batch_inputs, list):
            # Process each sample
            processed_inputs = []
            batch_pad_shape = []
            
            for inp in batch_inputs:
                # inp shape: (6, H, W) or could be dict
                if isinstance(inp, dict):
                    img1 = inp['img']
                    img2 = inp['img2']
                else:
                    # Assume stacked tensor (6, H, W)
                    img1 = inp[:3]
                    img2 = inp[3:]
                
                # Calculate pad shape
                h, w = img1.shape[-2:]
                pad_h = int(np.ceil(h / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(w / self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
                
                # BGR to RGB conversion for modality 1
                if self._channel_conversion:
                    img1 = img1[[2, 1, 0], ...]
                
                # BGR to RGB conversion for modality 2
                if self.bgr_to_rgb2:
                    img2 = img2[[2, 1, 0], ...]
                
                # Convert to float
                img1 = img1.float()
                img2 = img2.float()
                
                # Normalize modality 1
                if self.mean is not None:
                    img1 = (img1 - self.mean) / self.std
                
                # Normalize modality 2
                if self.mean2 is not None:
                    img2 = (img2 - self.mean2) / self.std2
                
                # Stack both modalities
                processed_inp = torch.cat([img1, img2], dim=0)  # (6, H, W)
                processed_inputs.append(processed_inp)
            
            # Pad and stack into batch
            max_h = max(s[0] for s in batch_pad_shape)
            max_w = max(s[1] for s in batch_pad_shape)
            
            padded_inputs = []
            for inp in processed_inputs:
                h, w = inp.shape[-2:]
                pad_h = max_h - h
                pad_w = max_w - w
                if pad_h > 0 or pad_w > 0:
                    inp = F.pad(inp, (0, pad_w, 0, pad_h), value=self.pad_value)
                padded_inputs.append(inp)
            
            batch_inputs = torch.stack(padded_inputs, dim=0)  # (B, 6, H, W)
        
        elif isinstance(batch_inputs, torch.Tensor):
            # Already batched tensor (B, 6, H, W)
            batch_size = batch_inputs.shape[0]
            img1 = batch_inputs[:, :3]  # (B, 3, H, W)
            img2 = batch_inputs[:, 3:]  # (B, 3, H, W)
            
            # BGR to RGB conversion
            if self._channel_conversion:
                img1 = img1[:, [2, 1, 0], ...]
            if self.bgr_to_rgb2:
                img2 = img2[:, [2, 1, 0], ...]
            
            # Convert to float
            img1 = img1.float()
            img2 = img2.float()
            
            # Normalize
            if self.mean is not None:
                img1 = (img1 - self.mean) / self.std
            if self.mean2 is not None:
                img2 = (img2 - self.mean2) / self.std2
            
            # Pad if necessary
            h, w = img1.shape[-2:]
            pad_h = int(np.ceil(h / self.pad_size_divisor)) * self.pad_size_divisor - h
            pad_w = int(np.ceil(w / self.pad_size_divisor)) * self.pad_size_divisor - w
            
            if pad_h > 0 or pad_w > 0:
                img1 = F.pad(img1, (0, pad_w, 0, pad_h), value=self.pad_value)
                img2 = F.pad(img2, (0, pad_w, 0, pad_h), value=self.pad_value)
            
            batch_inputs = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
            batch_pad_shape = [(img1.shape[-2], img1.shape[-1])] * batch_size
        
        # Update data samples with metadata
        if data_samples is not None:
            batch_input_shape = tuple(batch_inputs.shape[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })
            
            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

        return {'inputs': batch_inputs, 'data_samples': data_samples}
