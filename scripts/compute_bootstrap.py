import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import yaml

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


def ci_range(values, ci = 95):
    alpha = (100 - ci) / 2
    return (
        float(np.percentile(values, alpha)),
        float(np.percentile(values, 100 - alpha)),
    )


def bootstrap_all_metric_ci(
    scores_a,
    scores_b,
    n_bootstrap = 500,
    ci = 95,
    seed = 42,
):
    """
    Image-level bootstrap.

    For each bootstrap round:
    1. resample image-level similarity scores within group A
    2. resample image-level similarity scores within group B
    3. recompute mean_gap, Cohen's d, and Wasserstein distance
    """
    rng = np.random.default_rng(seed)

    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    n_a = len(scores_a)
    n_b = len(scores_b)

    gap_values = []
    d_values = []
    w_values = []

    for _ in range(n_bootstrap):
        sample_a = rng.choice(scores_a, size = n_a, replace = True)
        sample_b = rng.choice(scores_b, size = n_b, replace = True)

        gap_values.append(mean_gap(sample_a, sample_b))
        d_values.append(cohens_d(sample_a, sample_b))
        w_values.append(wasserstein_1d(sample_a, sample_b))

    return {
        "mean_gap": ci_range(gap_values, ci),
        "cohens_d": ci_range(d_values, ci),
        "wasserstein": ci_range(w_values, ci),
    }


def main():
    config = load_yaml("configs/experiment.yaml")
    occ_config = load_yaml("data/occupations.yaml")

    text_metadata = build_text_metadata(
        occ_config["occupations"],
        occ_config["templates"],
    )

    n_bootstrap = config.get("bootstrap", {}).get("n_bootstrap", 500)
    ci = config.get("bootstrap", {}).get("ci", 95)
    seed = config.get("bootstrap", {}).get("seed", 42)

    rows = []
    shared_emb_dir = "results/embeddings/shared"

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        safe_name = safe_model_name(model_name)

        print(f"\n========== Bootstrap CI: {model_name} ==========")

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
            metadata_path = f"results/embeddings/{analysis_name}/metadata.csv"

            print(f"Processing: {analysis_name} | group_col={group_col}")

            metadata = pd.read_csv(metadata_path)

            if "shared_idx" not in metadata.columns:
                raise ValueError(
                    f"Missing shared_idx in {metadata_path}. "
                    f"Please rerun scripts/extract_embeddings.py first."
                )

            similarity = shared_similarity[
                metadata["shared_idx"].to_numpy()
            ]

            group_values = sorted(metadata[group_col].dropna().unique())

            for group_a, group_b in combinations(group_values, 2):
                idx_a = metadata[metadata[group_col] == group_a].index.to_numpy()
                idx_b = metadata[metadata[group_col] == group_b].index.to_numpy()

                for text_idx, text_row in text_metadata.iterrows():
                    scores_a = similarity[idx_a, text_idx]
                    scores_b = similarity[idx_b, text_idx]

                    gap = float(mean_gap(scores_a, scores_b))
                    d = float(cohens_d(scores_a, scores_b))
                    w = float(wasserstein_1d(scores_a, scores_b))

                    ci_result = bootstrap_all_metric_ci(
                        scores_a,
                        scores_b,
                        n_bootstrap = n_bootstrap,
                        ci = ci,
                        seed = seed,
                    )

                    gap_ci = ci_result["mean_gap"]
                    d_ci = ci_result["cohens_d"]
                    w_ci = ci_result["wasserstein"]

                    rows.append(
                        {
                            "analysis_name": analysis_name,
                            "model": model_name,
                            "group_type": group_col,
                            "group_a": group_a,
                            "group_b": group_b,
                            "occupation": text_row["occupation"],
                            "prompt_idx": text_row["prompt_idx"],
                            "prompt": text_row["prompt"],
                            "mean_a": float(np.mean(scores_a)),
                            "mean_b": float(np.mean(scores_b)),
                            "mean_gap": gap,
                            "mean_gap_ci_lower": gap_ci[0],
                            "mean_gap_ci_upper": gap_ci[1],
                            "cohens_d": d,
                            "cohens_d_ci_lower": d_ci[0],
                            "cohens_d_ci_upper": d_ci[1],
                            "wasserstein": w,
                            "wasserstein_ci_lower": w_ci[0],
                            "wasserstein_ci_upper": w_ci[1],
                            "n_a": int(len(scores_a)),
                            "n_b": int(len(scores_b)),
                            "n_bootstrap": int(n_bootstrap),
                            "ci": int(ci),
                        }
                    )

    out_df = pd.DataFrame(rows)

    os.makedirs("results/metrics", exist_ok = True)
    out_path = "results/metrics/bootstrap_ci.csv"
    out_df.to_csv(out_path, index = False)

    print(f"\nSaved: {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()