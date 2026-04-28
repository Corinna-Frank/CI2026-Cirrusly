#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

r'''
Training script for weather/climate baseline models.

Uses Hydra for configuration. Run with::

    python scripts/train.py

Override config values on the command line::

    python scripts/train.py model.n_epochs=20 device=cuda
'''

# System modules
import logging
import os
from typing import Tuple

from sympy import per

# External modules
import torch
import torch.nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

# Internal modules
from starter_kit.data import TrainDataset
from starter_kit.layers import InputNormalisation
from starter_kit.augmentation import build_augmentation_pipeline_with_shift


main_logger = logging.getLogger(__name__)


def _load_normalisation(
        path: str,
        device: torch.device
) -> InputNormalisation:
    r'''
    Load an InputNormalisation layer from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to a ``.pt`` file with ``mean`` and ``std`` tensors.
    device : torch.device
        Device to load tensors onto.

    Returns
    -------
    InputNormalisation
        Normalisation layer with buffers on ``device``.
    '''
    state = torch.load(path, map_location=device)
    return InputNormalisation(
        mean=state["mean"],
        std=state["std"]
    )


def _build_network(
        cfg: DictConfig,
        device: torch.device
) -> torch.nn.Module:
    r'''
    Instantiate the network from Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Network sub-config (``cfg.network``). Must contain ``_target_``.
    device : torch.device
        Device to move the network onto after construction.

    Returns
    -------
    torch.nn.Module
        Instantiated network on ``device``.
    '''
    network = hydra.utils.instantiate(cfg)
    return network.to(device)


def _build_loaders(
        cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    r'''
    Build training and validation data loaders.

    Parameters
    ----------
    cfg : DictConfig
        Data sub-config (``cfg.data``).

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training loader and validation loader.
    '''
    # Build augmentation pipeline if enabled
    augmentation = None
    if cfg.get('augmentation', {}).get('enabled', False):
        aug_cfg = cfg.augmentation
        augmentation = build_augmentation_pipeline_with_shift(
            horizontal_flip=aug_cfg.get('horizontal_flip', True),
            vertical_flip=aug_cfg.get('vertical_flip', True),
            rotation=aug_cfg.get('rotation', True),
            brightness=aug_cfg.get('brightness', False),
            contrast=aug_cfg.get('contrast', False),
            crop=aug_cfg.get('crop', False),
            crop_size=tuple(aug_cfg.get('crop_size', [])) if aug_cfg.get('crop_size') else None,
            horizontal_flip_probability=aug_cfg.get('horizontal_flip_probability', 0.5),
            vertical_flip_probability=aug_cfg.get('vertical_flip_probability', 0.5),
            rotation_probability=aug_cfg.get('rotation_probability', 1.0),
            brightness_probability=aug_cfg.get('brightness_probability', 1.0),
            contrast_probability=aug_cfg.get('contrast_probability', 1.0),
            crop_probability=aug_cfg.get('crop_probability', 1.0),
            target_spatial_shift=aug_cfg.get('target_spatial_shift', False),
            max_spatial_shift=aug_cfg.get('max_spatial_shift', 5),
            shift_probability=aug_cfg.get('shift_probability', 0.5),
            seed=aug_cfg.get('seed', None),
        )
    
    train_ds = TrainDataset(
        cfg.train_path,
        threads_limit=cfg.threads_limit,
        augmentation=augmentation,
    )
    val_ds = TrainDataset(
        cfg.val_path,
        threads_limit=cfg.threads_limit,
        augmentation=augmentation,
    )
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory if torch.cuda.is_available() else False,
        persistent_workers=cfg.persistent_workers
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


@hydra.main(
    config_path="../configs",
    config_name="train",
    version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    r'''
    Entry point: parse config, build model, and run training.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration tree.
    '''
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.store_path, exist_ok=True)

    network = _build_network(cfg.network, device)
    train_loader, val_loader = _build_loaders(cfg.data)

    model = hydra.utils.instantiate(
        cfg.model,
        network=network,
        train_loader=train_loader,
        val_loader=val_loader,
        store_path=cfg.store_path,
        device=device,
    )

    model.train()
    main_logger.info(
        "Training complete. Best model saved to %s",
        cfg.store_path
    )


if __name__ == "__main__":
    main()
