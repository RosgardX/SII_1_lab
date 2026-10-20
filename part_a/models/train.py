
import numpy as np
import pandas as pd

from part_a.data_loader import load_dataset, PROJECT_DIR
from part_a.models.linear_regression import LinearRegression


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Разделяет данные на обучающую и тестовую выборки."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> tuple:
    """Масштабирует признаки по средним и отклонениям обучающей выборки."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    df = load_dataset(processed=True)

    feature_columns = [
        column for column in df.columns
        if column != "cnt" and not column.startswith("season_")
    ]

    X = df[feature_columns].to_numpy(dtype=float)
    y = df["cnt"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = scale_features(X_train, X_test)

    model = LinearRegression(learning_rate=0.01, n_iterations=5000)
    model.fit(X_train, y_train)

    baseline_predictions = np.full(len(y_test), y_train.mean())
    predictions = model.predict(X_test)

    results_dir = PROJECT_DIR / "data" / "processed"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "actual": y_test,
        "predicted": predictions,
        "baseline": baseline_predictions
    }).to_csv(results_dir / "test_predictions.csv", index=False)

    pd.DataFrame({
        "iteration": np.arange(1, len(model.loss_history) + 1),
        "mse": model.loss_history
    }).to_csv(results_dir / "loss_history.csv", index=False)

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    print(f"Количество признаков: {X_train.shape[1]}")
    print(f"Начальная MSE: {model.loss_history[0]:.2f}")
    print(f"Финальная MSE: {model.loss_history[-1]:.2f}")
    print(f"Предсказания сохранены: {results_dir / 'test_predictions.csv'}")
    print(f"История обучения сохранена: {results_dir / 'loss_history.csv'}")
