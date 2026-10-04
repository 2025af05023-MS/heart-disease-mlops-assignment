"""Create classroom-friendly EDA figures from the cleaned data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from heartlab.data import FEATURES, load_data


def make_plots(output: Path = Path("artifacts/eda")) -> list[Path]:
    frame = load_data()
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, feature in zip(axes, ["age", "chol", "thalach"], strict=True):
        sns.histplot(data=frame, x=feature, hue="target", bins=16, ax=axis, element="step")
        axis.set_title(feature)
    fig.suptitle("Selected Cleveland feature distributions by outcome")
    fig.tight_layout()
    path = output / "distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(4.5, 3.5))
    counts = frame["target"].value_counts().reindex([0, 1], fill_value=0)
    sns.barplot(x=["Absent", "Present"], y=counts.values, ax=axis)
    axis.set(ylabel="Records", title="Binary target balance")
    for index, value in enumerate(counts.values):
        axis.text(index, value + 2, str(value), ha="center")
    fig.tight_layout()
    path = output / "class_balance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(frame[FEATURES + ["target"]].corr(numeric_only=True), cmap="vlag", center=0,
                vmin=-1, vmax=1, ax=axis)
    axis.set_title("Pearson correlations (coded categories require cautious interpretation)")
    fig.tight_layout()
    path = output / "correlation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


if __name__ == "__main__":
    for plot in make_plots():
        print(plot)
