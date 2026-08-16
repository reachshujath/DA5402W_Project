from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_mlops.config import FEATURE_COLUMNS


def create_dataset(path: Path, rows: int = 2000, fraud_rows: int = 80) -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = rng.lognormal(mean=3.0, sigma=1.0, size=rows)
    frame["Class"] = 0
    fraud_indexes = rng.choice(rows, size=fraud_rows, replace=False)
    frame.loc[fraud_indexes, "Class"] = 1
    frame.loc[fraud_indexes, ["V1", "V2", "V3"]] += 4.0
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("creditcard_ci.csv"))
    args = parser.parse_args()
    create_dataset(args.path)
