#!/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Any

import numpy as np
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from starter_kit.model import BaseModel


main_logger = logging.getLogger(__name__)


class DummyNetwork(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "DummyNetwork should never be called. "
            "RandomForestModel does not use a neural network."
        )


class RandomForestModel(BaseModel):
    """
    Random Forest regression model replacing the UNet.

    Operates on flattened per-pixel features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 20,
        random_state: int = 42,
    ):
        super().__init__()

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
        )

        self._is_fitted = False
        
        self.network = None   # will be injected by trainer


    def _prepare_features(
        self,
        input_level: torch.Tensor,
        input_auxiliary: torch.Tensor,
        target: torch.Tensor | None = None,
    ):
        """
        Convert tensors to flat NumPy arrays suitable for sklearn.
        """

        # Collapse pressure levels into channels
        x_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )

        # Use first two auxiliary fields
        x_aux = input_auxiliary[:, :2]

        # Concatenate channels
        x = torch.cat([x_level, x_aux], dim=1)
        # x: (B, C, H, W)

        B, C, H, W = x.shape

        # Move to (B*H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = x.cpu().numpy()

        if target is None:
            return x, None, (B, H, W)

        # Target: (B, 1, H, W) or (B, H, W)
        y = target.view(-1).cpu().numpy()

        return x, y, (B, H, W)

    def estimate_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Train or evaluate the Random Forest and compute loss.
        """

        x, y, shape = self._prepare_features(
            batch["input_level"],
            batch["input_auxiliary"],
            batch["target"],
        )

        # Train once
        if not self._is_fitted:
            main_logger.info("Fitting Random Forest...")
            self.rf.fit(x, y)
            self._is_fitted = True

        # Predict
        y_pred = self.rf.predict(x)

        # Compute loss (MAE)
        loss_value = mean_absolute_error(y, y_pred)

        B, H, W = shape

        # Reshape prediction back to grid
        prediction = (
            torch.from_numpy(y_pred)
            .view(B, 1, H, W)
            .clamp(0.0, 1.0)
        )

        loss = torch.tensor(loss_value, dtype=torch.float32)

        return {
            "loss": loss,
            "prediction": prediction,
        }

    def estimate_auxiliary_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Auxiliary metrics.
        """

        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()

        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()

        return {"mse": mse, "accuracy": accuracy}