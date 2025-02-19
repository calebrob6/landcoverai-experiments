from typing import Any

import torch
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    FBetaScore,
    JaccardIndex,
    Precision,
    Recall,
)

class CustomSemanticSegmentationTask(SemanticSegmentationTask):

    def __init__(
        self, *args: Any, cosine_lr_cycle: int = 50, lr_min: float = 1e-6, scheduler: str= "plateau", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)  # pass args and kwargs to the parent class
        self.model = torch.compile(self.model)


    def configure_metrics(self) -> None:
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = self.hparams["ignore_index"]

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallPrecision": Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallRecall": Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "OverallIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "AverageAccuracy": Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AveragePrecision": Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AverageRecall": Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AverageF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="macro",
                    multidim_average="global",
                    ignore_index=ignore_index,
                ),
                "AverageIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_optimizers(
        self,
    ) -> 'lightning.pytorch.utilities.types.OptimizerLRScheduler':
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        scheduler: str = self.hparams['scheduler']
        cosine_lr_cycle: int = self.hparams['cosine_lr_cycle']
        lr_min: float = self.hparams['lr_min']
        patience: int = self.hparams['patience']

        optimizer = AdamW(self.parameters(), lr=self.hparams['lr'])
        if scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=cosine_lr_cycle, eta_min=lr_min)
        elif scheduler == 'plateau':
             scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        else:
            raise ValueError(f"Scheduler type '{scheduler}' is not valid. Currently, only supports 'cosine' and 'plateau'.")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor},
        }

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams['model']
        backbone: str = self.hparams['backbone']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']

        if model == 'unet':
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == 'upernet':
            self.model = smp.UPerNet(
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )