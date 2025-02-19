import argparse
import os

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")


def main(args):
    # Load config
    config = OmegaConf.load(args.config)

    if args.batch_size:
        config.datamodule.batch_size = args.batch_size
    if args.lr:
        config.module.lr = args.lr
    if args.epochs:
        config.trainer.max_epochs = args.epochs
    if args.model:
        config.module.model = args.model
    if args.backbone:
        config.module.backbone = args.backbone

    if args.root is None:
        datamodule = instantiate(config.datamodule)
    else:
        datamodule = instantiate(config.datamodule, root=args.root)

    if args.experiment_root:
        log_root = os.path.join("lightning_logs", args.experiment_root)
        os.makedirs(log_root, exist_ok=True)
    else:
        log_root = "lightning_logs"

    params = [
        config.module.model,
        config.module.backbone,
        str(config.module.lr),
        str(config.datamodule.batch_size),
        str(config.trainer.max_epochs),
        str(args.seed),
    ]
    experiment_name = "_".join(params)
    print(f"Experiment name: {experiment_name}")

    # Start experiment
    pl.seed_everything(args.seed)

    module = instantiate(config.module)
    logger = TensorBoardLogger(
        save_dir=log_root, name=experiment_name
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(save_last=True, save_top_k=3, mode="max", monitor="val_AverageIoU")
    ]
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
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--model", type=str, help="Override model")
    parser.add_argument("--backbone", type=str, help="Override backbone")
    parser.add_argument("--experiment_root", type=str, help="Experiment root")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)