# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from mmdet.models.losses.utils import weight_reduce_loss
from mmyolo.registry import MODELS
from .iou_loss import bbox_overlaps  # reuse your bbox_overlaps

@MODELS.register_module()
class PIOULoss(nn.Module):
    """PIoU Loss (Perceptual IoU, corner-based for occlusion robustness).

    Args:
        bbox_format (str): "xywh" or "xyxy". Defaults to "xywh".
        eps (float): Numerical stability epsilon.
        reduction (str): "none", "mean", or "sum".
        loss_weight (float): Weight factor.
        return_iou (bool): Return IoU tensor if True.
    """

    def __init__(self,
                 iou_mode: str = 'piou',
                 bbox_format: str = 'xywh',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 return_iou: bool = True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:

        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        iou = bbox_overlaps(
            pred,
            target,
            iou_mode='piou',  # 👈 use new PIoU mode
            bbox_format=self.bbox_format,
            eps=self.eps)

        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight,
                                                     reduction, avg_factor)

        if self.return_iou:
            return loss, iou
        else:
            return loss
