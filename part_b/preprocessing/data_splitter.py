
import numpy as np
import pandas as pd


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет данные на обучающую и тестовую выборки со стратификацией."""

    if not 0 < test_size < 1:
        raise ValueError("test_size должен находиться между 0 и 1.")

    if "Transported" not in data.columns:
        raise ValueError("В датасете отсутствует целевая переменная Transported.")

    if data["Transported"].isna().any():
        raise ValueError("В целевой переменной Transported есть пропуски.")

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for class_value in data["Transported"].unique():
        class_indices = np.flatnonzero(
            data["Transported"].to_numpy() == class_value
        )
        class_indices = rng.permutation(class_indices)

        test_count = round(len(class_indices) * test_size)

        test_indices.extend(class_indices[:test_count])
        train_indices.extend(class_indices[test_count:])

    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    X_train = train_data.drop(columns=["Transported"])
    X_test = test_data.drop(columns=["Transported"])

    y_train = train_data["Transported"].astype(int)
    y_test = test_data["Transported"].astype(int)

    return X_train, X_test, y_train, y_test
