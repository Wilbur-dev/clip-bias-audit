import os
import pandas as pd

from src.metrics.stability_metrics import (
    prompt_directional_consistency,
    prompt_sensitivity,
)


def main():
    basic_df = pd.read_csv("results/metrics/basic_metrics.csv")

    required_cols = {
        "analysis_name",
        "model",
        "group_type",
        "group_a",
        "group_b",
        "occupation",
        "prompt_idx",
        "mean_gap",
        "cohens_d",
        "wasserstein",
    }

    missing = required_cols - set(basic_df.columns)
    if missing:
        raise ValueError(f"Missing columns in basic_metrics.csv: {missing}")

    rows = []

    group_keys = [
        "analysis_name",
        "model",
        "group_type",
        "group_a",
        "group_b",
        "occupation",
    ]

    
    for keys, group_df in basic_df.groupby(group_keys):
        analysis_name, model, group_type, group_a, group_b, occupation = keys

        gaps = group_df["mean_gap"].to_numpy()
        cohens_ds = group_df["cohens_d"].to_numpy()
        wassersteins = group_df["wasserstein"].to_numpy()

        rows.append(
            {
                "analysis_name": analysis_name,
                "model": model,
                "group_type": group_type,
                "group_a": group_a,
                "group_b": group_b,
                "occupation": occupation,
                "num_prompts": int(group_df["prompt_idx"].nunique()),
                "avg_mean_gap": float(gaps.mean()),
                "avg_abs_mean_gap": float(abs(gaps).mean()),
                "avg_cohens_d": float(cohens_ds.mean()),
                "avg_abs_cohens_d": float(abs(cohens_ds).mean()),
                "avg_wasserstein": float(wassersteins.mean()),
                "prompt_consistency": prompt_directional_consistency(gaps),
                "prompt_sensitivity": prompt_sensitivity(gaps),
            }
        )

    out_df = pd.DataFrame(rows)

    os.makedirs("results/metrics", exist_ok = True)
    out_path = "results/metrics/prompt_stability.csv"
    out_df.to_csv(out_path, index = False)

    print(f"Saved: {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()