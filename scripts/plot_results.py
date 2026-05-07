import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_top_k(df, analysis_name, k = 10):
    sub = df[df["analysis_name"] == analysis_name].head(k)

    plt.figure(figsize = (10, 6))

    plt.barh(
        sub["occupation"],
        sub["stability_adjusted_score"]
    )

    plt.gca().invert_yaxis()
    plt.title(f"Top {k} occupations ({analysis_name})")
    plt.xlabel("stability_adjusted_score")

    os.makedirs("results/figures", exist_ok = True)
    plt.savefig(f"results/figures/top_{analysis_name}.png")
    plt.close()


def main():
    df = pd.read_csv("results/metrics/final_ranking.csv")

    for analysis_name in df["analysis_name"].unique():
        plot_top_k(df, analysis_name)

    print("Saved figures")


if __name__ == "__main__":
    main()