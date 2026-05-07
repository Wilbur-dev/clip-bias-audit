from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FairFaceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: str, transform = None):
        self.df = dataframe.reset_index(drop = True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = self.image_root / row["file"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_path": str(image_path),
            "file": row["file"],
            "gender": row["gender"],
            "race": row["race"],
            "age": row["age"],
        }


def load_fairface_labels(label_csv: str) -> pd.DataFrame:
    df = pd.read_csv(label_csv)

    required_cols = {"file", "gender", "race", "age"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def balanced_sample_by_group(
    df: pd.DataFrame,
    group_col: str,
    sample_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    sampled_parts = []

    for group_name, group_df in df.groupby(group_col):
        n = min(len(group_df), sample_size)
        sampled = group_df.sample(n = n, random_state = seed, replace = False,)
        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, axis = 0)
    sampled_df = sampled_df.sample(frac = 1.0, random_state = seed).reset_index(drop = True)

    return sampled_df


def build_balanced_fairface_subset(
    label_csv: str,
    group_col: str,
    sample_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    df = load_fairface_labels(label_csv)
    return balanced_sample_by_group(
        df = df,
        group_col = group_col,
        sample_size = sample_size,
        seed = seed,
    )