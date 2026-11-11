import numpy as np

from part_1_a.models.linear_regression import LinearRegression
from part_2_a.regularization import Regularization


class RegularizedLinearRegression(LinearRegression):
    """Линейная регрессия с L1/L2/ElasticNet регуляризацией."""

    def __init__(
        self,
        penalty: str = "none",
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.01,
        n_iterations: int = 5000,
    ):
        super().__init__(learning_rate=learning_rate, n_iterations=n_iterations)
        self.regularization = Regularization(penalty, alpha, l1_ratio)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """MSE + штраф за веса."""
        return super().compute_loss(X, y) + self.regularization.loss(self.weights)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Градиентный спуск с учётом штрафа (bias не штрафуется)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = float(np.mean(y))
        self.loss_history = []

        for _ in range(self.n_iterations):
            predictions = X @ self.weights + self.bias
            errors = predictions - y

            dw = (2 / n_samples) * (X.T @ errors)
            dw += self.regularization.gradient(self.weights)
            db = 2 * np.mean(errors)

            self.weights -= self.learning_rate * dw
            self.weights = self.regularization.l1_step(self.weights, self.learning_rate)
            self.bias -= self.learning_rate * db

            self.loss_history.append(self.compute_loss(X, y))

        return self