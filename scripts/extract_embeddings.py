import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from src.datasets.fairface_dataset import (
    FairFaceDataset,
    build_balanced_fairface_subset,
)
from src.models.clip_encoder import CLIPEncoder


def load_yaml(path):
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def main():
    config = load_yaml("configs/experiment.yaml")
    occ_config = load_yaml("data/occupations.yaml")

    dataset_cfg = config["dataset"]
    exp_cfg = config["experiment"]
    analysis_sets = config["analysis_sets"]

    occupations = occ_config["occupations"]
    templates = occ_config["templates"]

    texts = [
        template.format(occupation = occupation)
        for occupation in occupations
        for template in templates
    ]

    transform = transforms.Compose([
        transforms.Resize((exp_cfg["image_size"], exp_cfg["image_size"])),
        transforms.ToTensor(),
    ])

    analysis_dfs = {}

    for analysis_name, analysis_cfg in analysis_sets.items():
        group_col = analysis_cfg["group_col"]
        sample_size = analysis_cfg["group_sample_size"]

        print(f"\n========== Building analysis set: {analysis_name} ==========")

        df = build_balanced_fairface_subset(
            label_csv = dataset_cfg["label_csv"],
            group_col = group_col,
            sample_size = sample_size,
            seed = exp_cfg["seed"],
        ).reset_index(drop = True)

        analysis_dfs[analysis_name] = df

        print(df[group_col].value_counts())

    shared_df = (
        pd.concat(analysis_dfs.values(), axis = 0, ignore_index = True)
        .drop_duplicates(subset = ["file"])
        .reset_index(drop = True)
    )

    file_to_shared_idx = {
        file_name: idx for idx, file_name in enumerate(shared_df["file"])
    }

    shared_out_dir = "results/embeddings/shared"
    os.makedirs(shared_out_dir, exist_ok = True)

    for analysis_name, df in analysis_dfs.items():
        analysis_out_dir = f"results/embeddings/{analysis_name}"
        os.makedirs(analysis_out_dir, exist_ok = True)

        analysis_metadata = df.copy().reset_index(drop = True)
        analysis_metadata["shared_idx"] = analysis_metadata["file"].map(
            file_to_shared_idx
        )

        if analysis_metadata["shared_idx"].isna().any():
            missing = analysis_metadata[
                analysis_metadata["shared_idx"].isna()
            ]["file"].tolist()
            raise ValueError(
                f"Some files in {analysis_name} are missing from shared_df: "
                f"{missing[:5]}"
            )

        analysis_metadata["shared_idx"] = analysis_metadata["shared_idx"].astype(int)

        analysis_metadata.to_csv(
            f"{analysis_out_dir}/metadata.csv",
            index = False,
        )

        print(
            f"Saved analysis metadata: {analysis_out_dir}/metadata.csv "
            f"({len(analysis_metadata)} rows)"
        )

    dataset = FairFaceDataset(
        dataframe = shared_df,
        image_root = dataset_cfg["root"],
        transform = transform,
    )

    loader = DataLoader(
        dataset,
        batch_size = exp_cfg["batch_size"],
        shuffle = False,
        num_workers = exp_cfg["num_workers"],
    )

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        pretrained = model_cfg["pretrained"]
        safe_name = safe_model_name(model_name)

        print(f"\n========== Encoding shared embeddings with {model_name} ==========")

        encoder = CLIPEncoder(
            model_name = model_name,
            pretrained = pretrained,
        )

        all_image_embeddings = []
        all_meta = []

        for batch in tqdm(loader, desc = f"shared-{model_name}"):
            images = batch["image"]

            image_emb = encoder.encode_images(images)
            all_image_embeddings.append(image_emb)

            batch_size = images.shape[0]

            for i in range(batch_size):
                all_meta.append(
                    {
                        "image_path": batch["image_path"][i],
                        "file": batch["file"][i],
                        "gender": batch["gender"][i],
                        "race": batch["race"][i],
                        "age": batch["age"][i],
                    }
                )

        image_embeddings = torch.cat(all_image_embeddings, dim = 0)
        text_embeddings = encoder.encode_texts(texts)

        image_path = f"{shared_out_dir}/image_embeddings_{safe_name}.pt"
        text_path = f"{shared_out_dir}/text_embeddings_{safe_name}.pt"
        meta_path = f"{shared_out_dir}/metadata_{safe_name}.csv"

        torch.save(image_embeddings, image_path)
        torch.save(text_embeddings, text_path)
        pd.DataFrame(all_meta).to_csv(meta_path, index = False)

        print(f"Saved shared image embeddings: {image_path}")
        print(f"Saved shared text embeddings: {text_path}")
        print(f"Saved shared metadata: {meta_path}")
        print("Image embeddings shape:", image_embeddings.shape)
        print("Text embeddings shape:", text_embeddings.shape)

    print("\nEmbedding extraction finished.")


if __name__ == "__main__":
    main()