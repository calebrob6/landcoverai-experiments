from typing import Any

import torch
import kornia.augmentation as K

from torchgeo.datasets import LandCoverAI
from torchgeo.datamodules.geo import NonGeoDataModule


class LandCoverAIDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LandCover.ai dataset.

    Uses the train/val/test splits from the dataset.
    """

    mean = torch.tensor(0)
    std = torch.tensor(255)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new LandCoverAIDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LandCoverAI`.
        """
        super().__init__(LandCoverAI, batch_size, num_workers, **kwargs)

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=None, keepdim=True
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )
