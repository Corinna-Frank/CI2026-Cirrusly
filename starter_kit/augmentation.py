#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit
# Data augmentation for climate image data

"""
Data augmentation module for climate image data.

Provides augmentation transforms that work with the TrainDataset format:
- input_level: (C, H, W) - 26 channels (13 pressure levels × 2 variables)
- input_auxiliary: (C, H, W) - 2 channels (land-sea mask, geopotential height)
- target: (1, H, W) - single channel target values (0-1 bounded)

All augmentations preserve the target value range [0, 1].
"""

import numpy as np
from typing import Dict, Callable, Optional, List
import torch


class AugmentationPipeline:
    """
    Composable augmentation pipeline that applies multiple transforms.
    
    Applies transforms sequentially to the data dictionary, preserving
    all keys (input_level, input_auxiliary, target).
    """
    
    def __init__(self, transforms: List[Callable]) -> None:
        """
        Initialize the pipeline with a list of transforms.
        
        Parameters
        ----------
        transforms : List[Callable]
            List of callable transforms. Each takes a dict and returns a dict.
        """
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply all transforms sequentially.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Input data dictionary with keys: input_level, input_auxiliary, target
            
        Returns
        -------
        Dict[str, np.ndarray]
            Augmented data dictionary
        """
        for transform in self.transforms:
            data = transform(data)
        return data


def random_flip_horizontal(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Randomly flip the data horizontally with 50% probability.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Input data dictionary
        
    Returns
    -------
    Dict[str, np.ndarray]
        Data with horizontal flip applied randomly
    """
    if np.random.random() > 0.5:
        for key in ['input_level', 'input_auxiliary', 'target']:
            if key in data:
                data[key] = np.flip(data[key], axis=-1).copy()
    return data


def random_flip_vertical(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Randomly flip the data vertically with 50% probability.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Input data dictionary
        
    Returns
    -------
    Dict[str, np.ndarray]
        Data with vertical flip applied randomly
    """
    if np.random.random() > 0.5:
        for key in ['input_level', 'input_auxiliary', 'target']:
            if key in data:
                data[key] = np.flip(data[key], axis=-2).copy()
    return data


def random_rotation_90(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Randomly rotate data by 0, 90, 180, or 270 degrees.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Input data dictionary
        
    Returns
    -------
    Dict[str, np.ndarray]
        Data with random 90-degree rotation applied
    """
    k = np.random.randint(0, 4)  # 0, 1, 2, or 3 times 90 degrees
    if k == 0:
        return data
    
    for key in ['input_level', 'input_auxiliary', 'target']:
        if key in data:
            # np.rot90 rotates counter-clockwise
            # k=1: 90°, k=2: 180°, k=3: 270°
            data[key] = np.rot90(data[key], k=k, axes=(-2, -1)).copy()
    return data


def random_crop(
    data: Dict[str, np.ndarray],
    crop_size: Optional[tuple] = None
) -> Dict[str, np.ndarray]:
    """
    Randomly crop the data to a smaller size.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Input data dictionary
    crop_size : tuple, optional
        (H, W) size to crop to. If None, uses 80% of original size.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Data with random crop applied
    """
    if crop_size is None:
        # Default to 80% crop
        h, w = data['input_level'].shape[-2:]
        crop_size = (int(h * 0.8), int(w * 0.8))
    
    h, w = data['input_level'].shape[-2:]
    crop_h, crop_w = crop_size
    
    # Random top-left corner
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    
    for key in ['input_level', 'input_auxiliary', 'target']:
        if key in data:
            data[key] = data[key][..., top:top+crop_h, left:left+crop_w].copy()
    
    return data


def random_brightness(
    data: Dict[str, np.ndarray],
    delta: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Randomly adjust brightness of input data.
    
    Note: Only applied to input_level, not target (to preserve target range).
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Input data dictionary
    delta : float
        Maximum absolute brightness change (applied to normalized data)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Data with brightness adjustment applied
    """
    if 'input_level' in data:
        brightness = np.random.uniform(-delta, delta)
        data['input_level'] = np.clip(
            data['input_level'] + brightness,
            -10, 10  # Reasonable range for normalized data
        ).astype(data['input_level'].dtype)
    return data


def random_contrast(
    data: Dict[str, np.ndarray],
    factor_range: tuple = (0.9, 1.1)
) -> Dict[str, np.ndarray]:
    """
    Randomly adjust contrast of input data.
    
    Note: Only applied to input_level, not target (to preserve target range).
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Input data dictionary
    factor_range : tuple
        (min, max) range for contrast multiplier
        
    Returns
    -------
    Dict[str, np.ndarray]
        Data with contrast adjustment applied
    """
    if 'input_level' in data:
        factor = np.random.uniform(*factor_range)
        # Apply contrast around the mean
        mean = data['input_level'].mean()
        data['input_level'] = np.clip(
            mean + factor * (data['input_level'] - mean),
            -10, 10
        ).astype(data['input_level'].dtype)
    return data


def build_augmentation_pipeline(
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotation: bool = True,
    brightness: bool = False,
    contrast: bool = False,
    crop: bool = False,
    crop_size: Optional[tuple] = None,
    seed: Optional[int] = None
) -> Callable:
    """
    Build an augmentation pipeline from configuration options.
    
    Parameters
    ----------
    horizontal_flip : bool
        Enable random horizontal flip
    vertical_flip : bool
        Enable random vertical flip
    rotation : bool
        Enable random 90-degree rotations
    brightness : bool
        Enable random brightness adjustment
    contrast : bool
        Enable random contrast adjustment
    crop : bool
        Enable random crop
    crop_size : tuple, optional
        (H, W) size for crop
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Callable
        Augmentation pipeline that can be passed to TrainDataset
    """
    if seed is not None:
        np.random.seed(seed)
    
    transforms = []
    
    if horizontal_flip:
        transforms.append(random_flip_horizontal)
    
    if vertical_flip:
        transforms.append(random_flip_vertical)
    
    if rotation:
        transforms.append(random_rotation_90)
    
    if brightness:
        transforms.append(random_brightness)
    
    if contrast:
        transforms.append(random_contrast)
    
    if crop:
        transforms.append(lambda d: random_crop(d, crop_size))
    
    if not transforms:
        # Return identity function if no augmentations enabled
        return lambda x: x
    
    return AugmentationPipeline(transforms)


# Default augmentation pipeline for climate data
def get_default_augmentation() -> Callable:
    """
    Get the default augmentation pipeline.
    
    Returns
    -------
    Callable
        Default augmentation with horizontal flip, vertical flip, and rotation
    """
    return build_augmentation_pipeline(
        horizontal_flip=True,
        vertical_flip=True,
        rotation=True,
        brightness=False,
        contrast=False,
        crop=False
    )