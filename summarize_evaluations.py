#!/usr/bin/env python3
"""Summarize transfer evaluation CSVs by algorithm and domain."""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="evaluation/results.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    summary = data.groupby(["algorithm", "domain"], dropna=False).agg(
        episodes=("episode", "count"),
        completion_rate=("completion", "mean"),
        median_completion_seconds=("completion_seconds", "median"),
        mean_food_per_minute=("food_per_minute", "mean"),
        mean_collisions=("collisions", "mean"),
        mean_safety_overrides=("safety_overrides", "mean"),
        mean_action_change=("mean_action_change", "mean"),
        mean_action_saturation=("action_saturation_rate", "mean"),
    )
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
