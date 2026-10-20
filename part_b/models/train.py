import numpy as np
import pandas as pd

from part_b.data_loader import load_data
from part_b.preprocessing.prepare_data import (
    PROJECT_DIR, prepare_dataset, fit_preprocessor, transform
)
from part_b.models.logistic_regression import LogisticRegression


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Стратифицированно делит датасет: доля классов в train и test одинакова."""
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []

    for _, group in df.groupby("Transported"):
        indices = rng.permutation(group.index.to_numpy())
        test_count = int(len(indices) * test_size)
        test_idx.extend(indices[:test_count])
        train_idx.extend(indices[test_count:])

    return df.loc[rng.permutation(train_idx)], df.loc[rng.permutation(test_idx)]


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Стандартизирует признаки по среднему и std обучающей выборки."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    return (X_train - mean) / std, (X_test - mean) / std


if __name__ == "__main__":
    df = prepare_dataset(load_data())

    # Сначала split, потом всё, что «учится» на данных — защита от data leakage
    train_df, test_df = split_data(df)

    params = fit_preprocessor(train_df)
    train_df = transform(train_df, params)
    test_df = transform(test_df, params)

    feature_columns = [c for c in train_df.columns if c != "Transported"]
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["Transported"].to_numpy(dtype=int)
    y_test = test_df["Transported"].to_numpy(dtype=int)

    X_train, X_test = scale_features(X_train, X_test)

    model = LogisticRegression(learning_rate=0.1, n_iterations=3000)
    model.fit(X_train, y_train)

    # Majority baseline: всегда предсказывает самый частый класс train
    majority_class = int(np.bincount(y_train).argmax())
    majority_proba = y_train.mean()

    results_dir = PROJECT_DIR / "data" / "processed"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "actual": y_test,
        "probability": model.predict_proba(X_test),
        "predicted": model.predict(X_test),
        "baseline": majority_class,
        "baseline_probability": majority_proba,
    }).to_csv(results_dir / "test_predictions.csv", index=False)

    pd.DataFrame({
        "iteration": np.arange(1, len(model.loss_history) + 1),
        "logloss": model.loss_history,
    }).to_csv(results_dir / "loss_history.csv", index=False)

    pd.DataFrame({"feature": feature_columns, "weight": model.weights}) \
        .to_csv(results_dir / "weights.csv", index=False)

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    print(f"Доля Transported=1 в train / test: {y_train.mean():.3f} / {y_test.mean():.3f}")
    print(f"Количество признаков: {X_train.shape[1]}")
    print(f"Начальный LogLoss: {model.loss_history[0]:.4f}")
    print(f"Финальный LogLoss: {model.loss_history[-1]:.4f}")
    print(f"Результаты сохранены в: {results_dir}")
