import os

import numpy as np
import pandas as pd


def thresholded_sign(x: float, eps: float = 1e-4) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def main():
    df = pd.read_csv("results/metrics/prompt_stability.csv")

    required_cols = {
        "analysis_name",
        "model",
        "group_type",
        "group_a",
        "group_b",
        "occupation",
        "avg_mean_gap",
        "avg_abs_cohens_d",
        "prompt_consistency",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = []

    group_keys = [
        "analysis_name",
        "group_type",
        "group_a",
        "group_b",
        "occupation",
    ]

    for keys, group_df in df.groupby(group_keys):
        analysis_name, group_type, group_a, group_b, occupation = keys

        model_signs = [
            thresholded_sign(x)
            for x in group_df["avg_mean_gap"].to_numpy()
        ]

        model_agreement = abs(np.mean(model_signs))

        mean_abs_d = group_df["avg_abs_cohens_d"].mean()
        mean_prompt_consistency = group_df["prompt_consistency"].mean()

        stability_adjusted_score = (
            mean_abs_d
            * mean_prompt_consistency
            * model_agreement
        )

        rows.append(
            {
                "analysis_name": analysis_name,
                "group_type": group_type,
                "group_a": group_a,
                "group_b": group_b,
                "occupation": occupation,
                "num_models": int(group_df["model"].nunique()),
                "model_agreement": float(model_agreement),
                "mean_abs_d": float(mean_abs_d),
                "mean_prompt_consistency": float(mean_prompt_consistency),
                "stability_adjusted_score": float(stability_adjusted_score),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        "stability_adjusted_score",
        ascending=False,
    ).reset_index(drop=True)

    out_df["rank"] = out_df.index + 1

    os.makedirs("results/metrics", exist_ok = True)

    out_df.to_csv("results/metrics/model_agreement.csv", index = False)
    out_df.to_csv("results/metrics/final_ranking.csv", index = False)

    print("Saved: results/metrics/model_agreement.csv")
    print("Saved: results/metrics/final_ranking.csv")
    print(out_df.head(20))


if __name__ == "__main__":
    main()