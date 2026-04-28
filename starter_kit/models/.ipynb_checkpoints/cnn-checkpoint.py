#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules
import logging
from typing import Dict, Any

# External modules
import torch
import torch.nn

# Internal modules
from starter_kit.model import BaseModel
from starter_kit.layers import InputNormalisation


main_logger = logging.getLogger(__name__)
loss_fn='huber'
log_loss=False
LOG_EPS = 1e-6 # for log_loss

TARGET_NOISE_STD = 0.3        # strength of noise (logit space)
TARGET_NOISE_EPS = 1e-4       # numerical safety


r'''
The normalisation mean and std values are pre-computed from the training data.
As in the MLP, all pressure levels are collapsed into the channels dimension
and only the first two auxiliary fields (land sea mask and geopotential) are
used. For each of these 30 input features we compute the mean and std across
all spatial locations, weighted by the latitude weights, and averaged across
all time steps in the training set. These values are stored in the lists below
and used to initialise the InputNormalisation layer in the MLPNetwork.
'''

_normalisation_mean = [
    294.531359,287.010605,278.507482,262.805241,227.580722,201.364517,
    209.719502,0.010667,0.006922,0.003784,0.001229,0.000088,0.000003,
    0.000003,-1.412110,-0.914917,0.431349,3.504875,11.699176,6.758849,
    -1.214763,0.167424,-0.105374,-0.172138,-0.022648,0.030789,0.281048,
    -0.094608,0.410844,2129.684371
]
_normalisation_std = [
    62.864550,61.180621,58.938862,56.016099,47.532073,32.281805,38.084321,
    0.006102,0.004648,0.003013,0.001266,0.000080,0.000001,0.000000,4.661358,
    6.159993,7.763541,9.877940,16.068963,11.681901,10.705570,4.119853,4.318767,
    4.810067,6.209760,10.585627,5.680168,2.978756,0.498762,3602.712270
]



class CNNNetwork(torch.nn.Module):
    """
    Simple fully convolutional network for gridded data.
    """

    def __init__(
        self,
        input_channels: int = 30,
        hidden_channels: int = 64,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        self.normalisation = InputNormalisation(
            mean=torch.tensor(_normalisation_mean),
            std=torch.tensor(_normalisation_std)
        )

        layers = []

        # First layer
        layers.append(torch.nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        ))
        layers.append(torch.nn.SiLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1
            ))
            layers.append(torch.nn.SiLU())

        # Output layer
        output_layer = torch.nn.Conv2d(
            hidden_channels, 1, kernel_size=1
        )
        torch.nn.init.normal_(output_layer.weight, std=1e-6)
        torch.nn.init.constant_(output_layer.bias, 0.5)

        layers.append(output_layer)

        self.net = torch.nn.Sequential(*layers)

    def forward(
        self,
        input_level: torch.Tensor,
        input_auxiliary: torch.Tensor
    ) -> torch.Tensor:
        # Collapse pressure levels into channels
        flattened_input_level = input_level.reshape(
            input_level.shape[0], -1, *input_level.shape[-2:]
        )

        # Use first two auxiliary fields
        sliced_auxiliary = input_auxiliary[:, :2]

        # Concatenate along channel dimension
        x = torch.cat(
            [flattened_input_level, sliced_auxiliary],
            dim=1
        )

        # Apply normalisation per channel
        x = x.movedim(1, -1)
        x = self.normalisation(x)
        x = x.movedim(-1, 1)

        return self.net(x)

class CNNModel(BaseModel):
    r'''
    Model wrapper for an MLP network with standard loss outputs.

    This class delegates forward execution to a hidden MLP network and
    computes a mean absolute error loss together with auxiliary metrics.
    '''

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        r'''
        Compute the primary training loss and prediction output.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing ``input_level``,
            ``input_auxiliary``, and ``target`` tensors.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``loss`` and ``prediction``.
            ``loss`` is the mean absolute error and ``prediction`` is the
            model output clamped to ``[0, 1]``.
        '''
        # Forward pass
        prediction = self.network(
            input_level=batch["input_level"],
            input_auxiliary=batch["input_auxiliary"]
        )
    
        # Clamp prediction in original space (safety)
        prediction = prediction.clamp(0.,1.)
        target = batch["target"]

        
        # --- TARGET NOISE (training only) ---
        if self.network.training and TARGET_NOISE_STD>0:
            target = add_target_noise(
                target,
                std=TARGET_NOISE_STD,
                eps=1e-4,
            )

        # --- LOG TRANSFORM ---
        if log_loss:
            prediction = torch.log1p(prediction + LOG_EPS)
            target = torch.log1p(target + LOG_EPS)
    
        if loss_fn == "huber":
            loss_map = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
                beta=0.02,
            )
        elif loss_fn=='bce_huber':
            eps = 1e-4
            prediction = prediction.clamp(eps, 1. - eps)
            bce_map = torch.nn.functional.binary_cross_entropy(
                prediction,
                target,
                reduction="none",
            )
            # ----- Huber (Smooth L1) loss map -----
            huber_map = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
                beta=0.05,   # tolerance in probability space
            )
            alpha = 0.2  # weight for BCE (boundary mass)
            loss_map = alpha * bce_map + (1.0 - alpha) * huber_map

        else:
            error = prediction - target
            loss_map = torch.log(torch.cosh(error))
    
        loss = (loss_map * self.lat_weights).mean()
    
        # --- INVERSE LOG TRANSFORM FOR OUTPUT ---
        if log_loss:
            prediction = torch.expm1(prediction)
            prediction = prediction.clamp(0., 1.)
    
        return {
            "loss": loss,
            "prediction": prediction,
        }

    
    def estimate_auxiliary_loss(
            self,
            batch: Dict[str, torch.Tensor],
            outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r'''
        Compute auxiliary regression and classification metrics.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing the ground-truth ``target`` tensor.
        outputs : Dict[str, Any]
            Model outputs from ``estimate_loss`` containing ``prediction``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``mse`` and ``accuracy``.
            ``mse`` is the mean squared error and ``accuracy`` is the
            thresholded classification accuracy at 0.5.
        '''
        mse = (outputs["prediction"] - batch["target"]).pow(2)
        mse = (mse * self.lat_weights).mean()
        prediction_bool = (outputs["prediction"] > 0.5).float()
        target_bool = (batch["target"] > 0.5).float()
        accuracy = (prediction_bool == target_bool).float()
        accuracy = (accuracy * self.lat_weights).mean()
        return {"mse": mse, "accuracy": accuracy}

def add_target_noise(target, std=0.15, eps=1e-4):
    """
    Add Gaussian noise to targets in logit space.
    Keeps result strictly in (0,1).
    """
    # Avoid exact 0 or 1
    target = target.clamp(eps, 1.0 - eps)

    # Convert to logit
    logit = torch.log(target) - torch.log1p(-target)

    # Add Gaussian noise
    noisy_logit = logit + torch.randn_like(logit) * std

    # Back to probability space
    noisy_target = torch.sigmoid(noisy_logit)
    return noisy_target
