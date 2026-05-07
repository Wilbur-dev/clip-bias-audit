from pathlib import Path

import yaml
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.fairface_dataset import (
    FairFaceDataset,
    build_balanced_fairface_subset,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_yaml("configs/experiment.yaml")
    occ_config = load_yaml("data/occupations.yaml")

    dataset_cfg = config["dataset"]
    exp_cfg = config["experiment"]
    analysis_sets = config["analysis_sets"]

    transform = transforms.Compose(
        [
            transforms.Resize((exp_cfg["image_size"], exp_cfg["image_size"])),
            transforms.ToTensor(),
        ]
    )

    print("========== Occupation / Prompt Check ==========")
    print(f"num occupations: {len(occ_config['occupations'])}")
    print(f"num templates: {len(occ_config['templates'])}")
    print("example prompt:")
    print(occ_config["templates"][0].format(occupation=occ_config["occupations"][0]))
    
    for analysis_name, analysis_cfg in analysis_sets.items():
        group_col = analysis_cfg["group_col"]
        sample_size = analysis_cfg["group_sample_size"]

        print(f"\n========== Checking analysis set: {analysis_name} ==========")
        print(f"group_col: {group_col}")
        print(f"group_sample_size: {sample_size}")

        sampled_df = build_balanced_fairface_subset(
            label_csv = dataset_cfg["label_csv"],
            group_col = group_col,
            sample_size = sample_size,
            seed = exp_cfg["seed"],
        )

        print("\nGroup counts:")
        print(sampled_df[group_col].value_counts())

        dataset = FairFaceDataset(
            dataframe = sampled_df,
            image_root = dataset_cfg["root"],
            transform = transform,
        )

        loader = DataLoader(
            dataset,
            batch_size = exp_cfg["batch_size"],
            shuffle = True,
            num_workers = exp_cfg["num_workers"],
        )

        batch = next(iter(loader))

        print(f"\ntotal samples: {len(dataset)}")
        print(f"image batch shape: {batch['image'].shape}")
        print(f"gender examples: {batch['gender'][:5]}")
        print(f"race examples: {batch['race'][:5]}")
        print(f"age examples: {batch['age'][:5]}")

    print("\nDay1 dataset check passed.")


if __name__ == "__main__":
    main()