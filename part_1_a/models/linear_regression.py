
import numpy as np


class LinearRegression:
    """Линейная регрессия с обучением методом градиентного спуска."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисляет среднеквадратичную ошибку предсказаний."""
        predictions = self.predict(X)
        return float(np.mean((predictions - y) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучает модель, обновляя коэффициенты градиентным спуском."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = float(np.mean(y))
        self.loss_history = []

        for _ in range(self.n_iterations):
            predictions = np.sum(X * self.weights, axis=1) + self.bias
            errors = predictions - y

            dw = (2 / n_samples) * (X.T @ errors)
            db = 2 * np.mean(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.loss_history.append(self.compute_loss(X, y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает целевые значения по признакам."""
        X = np.asarray(X, dtype=float)
        return np.sum(X * self.weights, axis=1) + self.bias
