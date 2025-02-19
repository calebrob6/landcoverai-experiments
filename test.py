import argparse
import os
import torch
import pandas as pd
import lightning as pl
from src.trainers import CustomSemanticSegmentationTask
from torchgeo.datamodules import LandCoverAIDataModule

CSV_FILENAME = "landcoverai_test_results.csv"

def load_model(checkpoint_path):
    """Load the model checkpoint."""
    model = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model

def test_model(model, gpu=0):
    """Evaluate the model on the LandCoverAI test set."""
    datamodule = LandCoverAIDataModule(root="data/LandCoverAI", batch_size=32, num_workers=6)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="gpu", devices=[gpu])

    results = trainer.test(model, datamodule=datamodule)
    return results[0] if results else {}

def save_results(checkpoint_path, results):
    """Append test results to CSV."""
    data = {"checkpoint": checkpoint_path}
    data.update(results)

    df = pd.DataFrame([data])

    if os.path.exists(CSV_FILENAME):
        df.to_csv(CSV_FILENAME, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_FILENAME, mode="w", header=True, index=False)

    print(f"Results saved to {CSV_FILENAME}")

def main(args):
    model = load_model(args.checkpoint)
    results = test_model(model, gpu=args.gpu)
    save_results(args.checkpoint, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()
    main(args)
