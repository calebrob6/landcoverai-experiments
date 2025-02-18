import argparse
import os

import lightning
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")


def main(args):
    lightning.pytorch.seed_everything(args.seed)
    config = OmegaConf.load(args.config)
    module = instantiate(config.module)

    if args.root is None:
        datamodule = instantiate(config.datamodule)
    else:
        datamodule = instantiate(config.datamodule, root=args.root)

    logger = TensorBoardLogger(
        save_dir="lightning_logs", name=config.experiment_name
    )

    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    devices = [args.device] if args.device is not None else config.trainer.devices
    trainer = instantiate(
        config.trainer, logger=logger, callbacks=callbacks, devices=devices
    )
    trainer.fit(module, datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        required=False,
        help="Optionally override root from config",
    )
    parser.add_argument(
        "--logger", type=str, choices=["tensorboard"], default=None
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)