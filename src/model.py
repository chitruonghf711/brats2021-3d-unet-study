"""
Model architecture module for BraTS 2021 segmentation.
Integrates standard 3D UNet architectures from MONAI.
"""

from typing import Tuple, Union, Optional, List, Sequence
import torch
import torch.nn as nn
from monai.networks.nets import UNet, BasicUNet, DynUNet
from monai.networks.layers import Norm


def build_unet_model(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 3,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str = "batch",
    dropout: float = 0.1,
    model_type: str = "unet"
) -> nn.Module:
    """
    Instantiate a 3D UNet model from MONAI.

    Args:
        spatial_dims: Spatial dimensions (3 for 3D MRI volumes).
        in_channels: Number of input channels (4 for BraTS: FLAIR, T1, T1ce, T2).
        out_channels: Number of output channels (3 for BraTS targets: TC, WT, ET).
        channels: Feature map channels at each resolution level.
        strides: Downsampling strides at each level.
        num_res_units: Number of residual units per block.
        norm: Normalization layer type ('batch', 'instance', etc.).
        dropout: Dropout probability.
        model_type: 'unet' for MONAI UNet, 'basic_unet' for MONAI BasicUNet.

    Returns:
        torch.nn.Module: Instantiated 3D UNet model.
    """
    norm_type = Norm.BATCH if norm.lower() == "batch" else Norm.INSTANCE

    if model_type.lower() == "basic_unet":
        model = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(16, 16, 32, 64, 128, 16),
            dropout=dropout,
            norm=norm_type
        )
    else:
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm_type,
            dropout=dropout
        )

    return model


def build_nnunet_model(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 3,
    filters: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[Sequence[int]] = (
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    ),
    kernel_size: Optional[Sequence[Sequence[int]]] = None,
    upsample_kernel_size: Optional[Sequence[Sequence[int]]] = None,
    norm_name: str = "instance",
    act_name: Union[str, Tuple[str, dict]] = ("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
    deep_supervision: bool = False,
    res_block: bool = False,
    dropout: Optional[float] = None
) -> nn.Module:
    """
    Instantiate an nnU-Net baseline architecture using MONAI's DynUNet.

    nnU-Net architectural characteristics:
    - Multi-stage dynamic encoder-decoder with anisotropic/isotropic kernels.
    - Instance Normalization instead of Batch Normalization.
    - LeakyReLU activation (negative_slope = 0.01).
    - Deep supervision support (optional).
    - Residual encoder blocks (optional, ResEnc variant).

    Args:
        spatial_dims: Spatial dimensionality (3 for 3D volumes).
        in_channels: 4 input modalities for BraTS (FLAIR, T1, T1ce, T2).
        out_channels: 3 output target regions (TC, WT, ET).
        filters: Feature channels progression across stages.
        strides: Convolution downsampling strides per stage.
        kernel_size: Convolution kernel sizes per stage. Defaults to (3, 3, 3).
        upsample_kernel_size: Transposed convolution kernel sizes for decoder.
        norm_name: 'instance' (standard nnU-Net) or 'batch'.
        act_name: Activation layer. Defaults to LeakyReLU.
        deep_supervision: Whether to output intermediate multi-scale predictions.
        res_block: Whether to use residual units (nnU-Net ResEnc variant).
        dropout: Optional dropout rate.

    Returns:
        torch.nn.Module: Instantiated DynUNet model.
    """
    strides_list = [list(s) for s in strides]
    num_stages = len(strides_list)

    if kernel_size is None:
        kernel_size_list = [[3, 3, 3] for _ in range(num_stages)]
    else:
        kernel_size_list = [list(k) for k in kernel_size]

    if upsample_kernel_size is None:
        upsample_kernel_list = strides_list[1:]
    else:
        upsample_kernel_list = [list(u) for u in upsample_kernel_size]

    if isinstance(act_name, str):
        if act_name.lower() == "leakyrelu":
            act_config = ("leakyrelu", {"negative_slope": 0.01, "inplace": True})
        else:
            act_config = act_name
    else:
        act_config = act_name

    model = DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size_list,
        strides=strides_list,
        upsample_kernel_size=upsample_kernel_list,
        filters=list(filters),
        dropout=dropout,
        norm_name=norm_name,
        act_name=act_config,
        deep_supervision=deep_supervision,
        res_block=res_block
    )

    return model


def build_model(config) -> nn.Module:
    """
    Unified model factory building either standard UNet or nnU-Net (DynUNet)
    based on config.model_name.
    """
    model_name = getattr(config, "model_name", "unet").lower()
    if model_name in ("nnunet", "dynunet"):
        return build_nnunet_model(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            filters=config.nnunet_filters,
            strides=config.nnunet_strides,
            kernel_size=config.nnunet_kernel_size,
            norm_name=config.nnunet_norm,
            act_name=config.nnunet_act,
            deep_supervision=config.nnunet_deep_supervision,
            res_block=config.nnunet_res_block,
            dropout=config.dropout if config.dropout > 0 else None
        )
    else:
        return build_unet_model(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels,
            strides=config.strides,
            num_res_units=config.num_res_units,
            norm=config.norm,
            dropout=config.dropout
        )


def get_model_summary(model: nn.Module) -> dict:
    """
    Calculate total and trainable parameters of the model.

    Returns:
        Dictionary containing parameter counts and estimated model size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)

    return {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2)
    }

