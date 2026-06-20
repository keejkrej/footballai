"""Plot training metrics from PyTorch Lightning CSV logs.

Usage:
    python plot_metrics.py --log_dir ./runs/footballai_csv

Saves `training_curves.png` in the log directory.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(log_dir: Path) -> pd.DataFrame:
    """Load the metrics.csv written by CSVLogger."""
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv found at {csv_path}")
    return pd.read_csv(csv_path)


def plot_metric(df: pd.DataFrame, metric: str, ax: plt.Axes, smooth: int = 1) -> None:
    """Plot train and validation curves for one metric."""
    train_key = f"train/{metric}_step"
    val_key = f"val/{metric}"
    epoch_train_key = f"train/{metric}_epoch"

    if train_key in df.columns:
        train_df = df.dropna(subset=[train_key])
        if len(train_df) > 0:
            x = train_df["step"].values
            y = train_df[train_key].values
            if smooth > 1 and len(y) > smooth:
                y_smooth = pd.Series(y).rolling(window=smooth, min_periods=1).mean().values
            else:
                y_smooth = y
            ax.plot(x, y_smooth, alpha=0.35, color="C0", label="train step")

    if epoch_train_key in df.columns:
        epoch_df = df.dropna(subset=[epoch_train_key])
        if len(epoch_df) > 0:
            ax.plot(
                epoch_df["step"].values,
                epoch_df[epoch_train_key].values,
                marker="o",
                color="C0",
                label="train epoch",
            )

    if val_key in df.columns:
        val_df = df.dropna(subset=[val_key])
        if len(val_df) > 0:
            ax.plot(
                val_df["step"].values,
                val_df[val_key].values,
                marker="s",
                color="C1",
                label="val epoch",
            )

    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs")
    parser.add_argument("--log_dir", type=str, default="./runs/footballai_csv")
    parser.add_argument("--metrics", type=str, default="total,pass_xy,pass_slot,shot_prob,shot_xg,turnover")
    parser.add_argument("--smooth", type=int, default=20, help="Rolling average window for step curves")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    df = load_metrics(log_dir)
    print(f"Loaded {len(df)} rows from {log_dir / 'metrics.csv'}")

    metrics = [m.strip() for m in args.metrics.split(",")]
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for metric, ax in zip(metrics, axes):
        plot_metric(df, metric, ax, smooth=args.smooth)

    fig.tight_layout()
    out_path = Path(args.out) if args.out else (log_dir / "training_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
