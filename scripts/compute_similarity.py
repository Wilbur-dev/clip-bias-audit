import os
import yaml
import torch
import pandas as pd
import numpy as np
from itertools import combinations

from src.metrics.bias_metrics import mean_gap, cohens_d, wasserstein_1d


def load_yaml(path):
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def build_text_metadata(occupations, templates):
    rows = []

    for occupation in occupations:
        for prompt_idx, template in enumerate(templates):
            rows.append(
                {
                    "occupation": occupation,
                    "prompt_idx": prompt_idx,
                    "prompt": template.format(occupation = occupation),
                }
            )

    return pd.DataFrame(rows)


def compute_group_pair_metrics(
    similarity,
    metadata,
    text_metadata,
    group_col,
):
    results = []

    group_values = sorted(metadata[group_col].dropna().unique())

    for group_a, group_b in combinations(group_values, 2):
        idx_a = metadata[metadata[group_col] == group_a].index.to_numpy()
        idx_b = metadata[metadata[group_col] == group_b].index.to_numpy()

        for text_idx, text_row in text_metadata.iterrows():
            scores_a = similarity[idx_a, text_idx]
            scores_b = similarity[idx_b, text_idx]

            results.append(
                {
                    "group_type": group_col,
                    "group_a": group_a,
                    "group_b": group_b,
                    "occupation": text_row["occupation"],
                    "prompt_idx": text_row["prompt_idx"],
                    "prompt": text_row["prompt"],
                    "mean_a": float(np.mean(scores_a)),
                    "mean_b": float(np.mean(scores_b)),
                    "mean_gap": float(mean_gap(scores_a, scores_b)),
                    "cohens_d": float(cohens_d(scores_a, scores_b)),
                    "wasserstein": float(wasserstein_1d(scores_a, scores_b)),
                    "n_a": int(len(scores_a)),
                    "n_b": int(len(scores_b)),
                }
            )

    return results


def main():
    config = load_yaml("configs/experiment.yaml")
    occ_config = load_yaml("data/occupations.yaml")

    text_metadata = build_text_metadata(
        occ_config["occupations"],
        occ_config["templates"],
    )

    all_results = []

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        safe_name = safe_model_name(model_name)

        shared_emb_dir = "results/embeddings/shared"

        print(f"\n========== Loading shared embeddings: {model_name} ==========")

        image_embeddings = torch.load(
            f"{shared_emb_dir}/image_embeddings_{safe_name}.pt"
        )
        text_embeddings = torch.load(
            f"{shared_emb_dir}/text_embeddings_{safe_name}.pt"
        )

        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: "
                f"{image_embeddings.shape[1]} vs {text_embeddings.shape[1]}"
            )

        shared_similarity = image_embeddings @ text_embeddings.T
        shared_similarity = shared_similarity.numpy()

        for analysis_name, analysis_cfg in config["analysis_sets"].items():
            group_col = analysis_cfg["group_col"]

            print(
                f"\n========== Computing similarity: "
                f"{analysis_name} | {model_name} =========="
            )

            analysis_meta_path = f"results/embeddings/{analysis_name}/metadata.csv"
            metadata = pd.read_csv(analysis_meta_path)

            if "shared_idx" not in metadata.columns:
                raise ValueError(
                    f"Missing shared_idx in {analysis_meta_path}. "
                    f"Please rerun scripts/extract_embeddings.py first."
                )

            similarity = shared_similarity[
                metadata["shared_idx"].to_numpy()
            ]

            rows = compute_group_pair_metrics(
                similarity = similarity,
                metadata = metadata,
                text_metadata = text_metadata,
                group_col = group_col,
            )

            for row in rows:
                row["analysis_name"] = analysis_name
                row["group_type"] = group_col
                row["model"] = model_name

            all_results.extend(rows)

    os.makedirs("results/metrics", exist_ok = True)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv("results/metrics/basic_metrics.csv", index = False)

    print("\nSaved: results/metrics/basic_metrics.csv")
    print(result_df.head())


if __name__ == "__main__":
    main()