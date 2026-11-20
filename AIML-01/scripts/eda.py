"""Exploratory data analysis -> docs/EDA.md

Usage:
    python scripts/eda.py --dataset synthetic
    python scripts/eda.py --dataset paysim --path data/PS_20174392719_1491204439457_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.data import load_dataset


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [f"{v:,.4g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                 for v in row.tolist()]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def run(df: pd.DataFrame) -> str:
    n, fraud = len(df), int(df["is_fraud"].sum())
    rate = fraud / n
    out = ["# Exploratory Data Analysis", ""]
    out += [f"* Transactions: **{n:,}**",
            f"* Fraudulent: **{fraud:,}** ({rate:.3%}) -> imbalance ratio 1:{(n - fraud) / max(fraud, 1):,.0f}",
            f"* Users: {df['user_id'].nunique():,} | Devices: {df['device_id'].nunique():,} | "
            f"Cards: {df['card_id'].nunique():,}",
            f"* Time span: {df['timestamp'].min()} -> {df['timestamp'].max()}", ""]
    out += ["## Why accuracy is the wrong metric", "",
            f"A model that predicts *legit* for everything scores **{1 - rate:.2%} accuracy** "
            "while catching zero fraud. We therefore optimise and report **PR-AUC** "
            "(average precision), whose no-skill baseline equals the fraud rate "
            f"(**{rate:.4f}**).", ""]
    amt = df.groupby("is_fraud")["amount"].describe()[["mean", "50%", "max"]]
    amt.index = amt.index.map({0: "legit", 1: "fraud"})
    amt.index.name = "class"
    out += ["## Amount by class", "", _md_table(amt), ""]
    ch = pd.crosstab(df["channel"], df["is_fraud"], normalize="index")
    ch.columns = ["legit_share", "fraud_rate"][: len(ch.columns)]
    out += ["## Fraud rate by channel", "", _md_table(ch.sort_values(ch.columns[-1], ascending=False)), ""]
    dev_users = df.groupby("device_id")["user_id"].nunique()
    shared = df[df["device_id"].isin(dev_users[dev_users > 1].index)]
    out += ["## Shared devices (graph signal)", "",
            f"* Devices used by >1 account: {int((dev_users > 1).sum()):,}",
            f"* Fraud rate on shared devices: **{shared['is_fraud'].mean():.2%}** vs "
            f"{df['is_fraud'].mean():.2%} overall", "",
            "Shared infrastructure is the motivation for the graph neural network stage.", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--path")
    p.add_argument("--out", default="docs/EDA.md")
    a = p.parse_args()
    report = run(load_dataset(a.dataset, a.path))
    Path(a.out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
