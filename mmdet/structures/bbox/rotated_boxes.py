# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

from .base_boxes import BaseBoxes
from .box_type import register_box

T = TypeVar('T')
DeviceType = Union[str, torch.device]


@register_box(name='rbox')
class RotatedBoxes(BaseBoxes):
    """The rotated box class for OBB detection.

    The ``box_dim`` of ``RotatedBoxes`` is 5, which means the length of
    the last dimension of the data should be 5. The data format is
    (cx, cy, w, h, angle), where:
    - (cx, cy): center coordinates
    - (w, h): width and height
    - angle: rotation angle in radians, range [-pi/2, pi/2)

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape of
            (..., 5).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim: int = 5

    def __init__(self,
                 data: Union[Tensor, np.ndarray],
                 dtype: torch.dtype = None,
                 device: DeviceType = None,
                 clone: bool = True) -> None:
        super().__init__(data=data, dtype=dtype, device=device, clone=clone)

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        return self.tensor[..., 2] * self.tensor[..., 3]

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes."""
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes."""
        return self.tensor[..., 3]

    @property
    def angles(self) -> Tensor:
        """Return a tensor representing the angles of boxes."""
        return self.tensor[..., 4]

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        h, w = img_shape
        boxes = self.tensor
        
        if direction == 'horizontal':
            boxes[..., 0] = w - boxes[..., 0]
            boxes[..., 4] = -boxes[..., 4]  # Flip angle
        elif direction == 'vertical':
            boxes[..., 1] = h - boxes[..., 1]
            boxes[..., 4] = -boxes[..., 4]  # Flip angle
        else:  # diagonal
            boxes[..., 0] = w - boxes[..., 0]
            boxes[..., 1] = h - boxes[..., 1]
            # Angle unchanged for diagonal flip

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        boxes = self.tensor
        boxes[..., 0] += distances[0]
        boxes[..., 1] += distances[1]

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        For rotated boxes, we clip the center coordinates to be within
        the image boundaries.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        h, w = img_shape
        boxes = self.tensor
        boxes[..., 0] = boxes[..., 0].clamp(0, w)
        boxes[..., 1] = boxes[..., 1].clamp(0, h)

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean anti-clockwise rotation.
        """
        boxes = self.tensor
        angle_rad = angle * np.pi / 180.0
        
        # Rotate center coordinates
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        cx_old = boxes[..., 0] - center[0]
        cy_old = boxes[..., 1] - center[1]
        
        boxes[..., 0] = cx_old * cos_a - cy_old * sin_a + center[0]
        boxes[..., 1] = cx_old * sin_a + cy_old * cos_a + center[1]
        
        # Update angle
        boxes[..., 4] = boxes[..., 4] + angle_rad

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray):
                Shape (3, 3) for geometric transformation.
        """
        # For simplicity, only transform center coordinates
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = self.tensor.new_tensor(homography_matrix)
        
        boxes = self.tensor
        centers = boxes[..., :2]  # (N, 2)
        
        # Add homogeneous coordinate
        ones = centers.new_ones((*centers.shape[:-1], 1))
        centers_homo = torch.cat([centers, ones], dim=-1)  # (N, 3)
        
        # Apply transformation
        centers_transformed = torch.matmul(centers_homo, homography_matrix.T)
        
        # Normalize
        boxes[..., :2] = centers_transformed[..., :2] / centers_transformed[..., 2:3]

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        boxes = self.tensor
        boxes[..., 0] *= scale_factor[0]  # cx
        boxes[..., 1] *= scale_factor[1]  # cy
        boxes[..., 2] *= scale_factor[0]  # w
        boxes[..., 3] *= scale_factor[1]  # h
        # Note: angle remains unchanged

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        boxes = self.tensor
        boxes[..., 2] *= scale_factor[0]  # w
        boxes[..., 3] *= scale_factor[1]  # h

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> Tensor:
        """Find boxes inside the image.

        For rotated boxes, we check if the center is inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image by no
                more than ``allowed_border`` are considered "inside".
                Defaults to 0.

        Returns:
            Tensor: A BoolTensor indicating whether the box is inside
                the image. Assuming the original boxes have shape (m, n, 5),
                the output has shape (m, n).
        """
        h, w = img_shape
        boxes = self.tensor
        cx, cy = boxes[..., 0], boxes[..., 1]
        
        # Check if center is inside
        inside = (cx >= -allowed_border) & (cx < w + allowed_border) & \
                 (cy >= -allowed_border) & (cy < h + allowed_border)
        
        return inside

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False) -> Tensor:
        """Find inside box points.
        
        For rotated boxes, this is more complex. Here we provide a simplified
        version that checks if points are inside the axis-aligned bounding box.

        Args:
            points (Tensor): Points coordinates with shape of (m, 2) or (b, m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. Defaults to False.

        Returns:
            Tensor: A BoolTensor indicating whether a point is inside
                boxes. If ``is_aligned`` is False, the shape of output is
                (b, n, m). If ``is_aligned`` is True, the shape of output is
                (b, n).
        """
        # Simplified: use axis-aligned bounding box
        boxes = self.tensor
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        if is_aligned:
            return (points[..., 0] >= x1) & (points[..., 0] <= x2) & \
                   (points[..., 1] >= y1) & (points[..., 1] <= y2)
        else:
            # points: (m, 2), boxes: (n, 5)
            # output: (n, m)
            px = points[..., 0]  # (m,)
            py = points[..., 1]  # (m,)
            
            inside_x = (px.unsqueeze(0) >= x1.unsqueeze(-1)) & \
                       (px.unsqueeze(0) <= x2.unsqueeze(-1))
            inside_y = (py.unsqueeze(0) >= y1.unsqueeze(-1)) & \
                       (py.unsqueeze(0) <= y2.unsqueeze(-1))
            
            return inside_x & inside_y

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two sets of rotated boxes.
        
        This requires mmcv.ops.box_iou_rotated for accurate calculation.
        
        Args:
            boxes1 (BaseBoxes): RotatedBoxes with shape (N, 5) in
                (cx, cy, w, h, angle) format.
            boxes2 (BaseBoxes): RotatedBoxes with shape (M, 5) in
                (cx, cy, w, h, angle) format.
            mode (str): 'iou' (intersection over union). Defaults to 'iou'.
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (N, M) if ``is_aligned`` is False, else shape (N,).
        """
        try:
            from mmcv.ops import box_iou_rotated
        except ImportError:
            raise ImportError(
                'Please install mmcv-full to use rotated box IoU calculation.')
        
        boxes1_tensor = boxes1.tensor
        boxes2_tensor = boxes2.tensor
        
        if is_aligned:
            return box_iou_rotated(boxes1_tensor, boxes2_tensor, aligned=True)
        else:
            return box_iou_rotated(boxes1_tensor, boxes2_tensor, aligned=False)

    @staticmethod
    def from_instance_masks(masks) -> 'RotatedBoxes':
        """Create rotated boxes from instance masks.
        
        This extracts the minimum bounding rectangle from each mask.
        
        Args:
            masks: BitmapMasks or PolygonMasks.

        Returns:
            RotatedBoxes: Bounding rotated boxes.
        """
        # This is a placeholder - proper implementation would use cv2.minAreaRect
        raise NotImplementedError(
            'from_instance_masks is not implemented for RotatedBoxes')
