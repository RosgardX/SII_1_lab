from pathlib import Path

import numpy as np

from part_1_a.data_loader import load_dataset
from part_1_a.models.train import split_data as split_arrays
from part_1_b.data_loader import load_data
from part_1_b.models.train import split_data as split_frame
from part_1_b.preprocessing.prepare_data import prepare_dataset, fit_preprocessor, transform


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = PROJECT_DIR / "plots"

RANDOM_STATE = 13
TEST_SIZE = 0.2
VAL_SIZE = 0.25


def scale(X_train: np.ndarray, *others: np.ndarray) -> list:
    """Стандартизирует все выборки по среднему и std train."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    return [(X - mean) / std for X in (X_train, *others)]


def load_bike_data() -> dict:
    df = load_dataset(processed=True)

    feature_columns = [
        column for column in df.columns
        if column != "cnt" and not column.startswith("season_")
    ]
    X = df[feature_columns].to_numpy(dtype=float)
    y = df["cnt"].to_numpy(dtype=float)

    X_rest, X_test, y_rest, y_test = split_arrays(X, y, TEST_SIZE, RANDOM_STATE)
    X_train, X_val, y_train, y_val = split_arrays(X_rest, y_rest, VAL_SIZE, RANDOM_STATE)
    X_train, X_val, X_test = scale(X_train, X_val, X_test)

    return {
        "features": feature_columns,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def load_spaceship_data() -> dict:
    df = prepare_dataset(load_data())

    rest_df, test_df = split_frame(df, TEST_SIZE, RANDOM_STATE)
    train_df, val_df = split_frame(rest_df, VAL_SIZE, RANDOM_STATE)

    params = fit_preprocessor(train_df)
    frames = [transform(frame, params) for frame in (train_df, val_df, test_df)]

    feature_columns = [c for c in frames[0].columns if c != "Transported"]
    X_train, X_val, X_test = scale(
        *[frame[feature_columns].to_numpy(dtype=float) for frame in frames]
    )
    y_train, y_val, y_test = [frame["Transported"].to_numpy(dtype=int) for frame in frames]

    return {
        "features": feature_columns,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }