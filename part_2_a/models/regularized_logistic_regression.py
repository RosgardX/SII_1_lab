import numpy as np

from part_1_b.models.logistic_regression import LogisticRegression
from part_2_a.regularization import Regularization


class RegularizedLogisticRegression(LogisticRegression):
    """Логистическая регрессия с L1/L2/ElasticNet регуляризацией"""

    def __init__(
        self,
        penalty: str = "none",
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.1,
        n_iterations: int = 3000,
        threshold: float = 0.5,
    ):
        super().__init__(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            threshold=threshold,
        )
        self.regularization = Regularization(penalty, alpha, l1_ratio)

    def compute_loss(self, y: np.ndarray, probabilities: np.ndarray) -> float:
        """LogLoss + штраф за веса."""
        return super().compute_loss(y, probabilities) + self.regularization.loss(self.weights)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Градиентный спуск с учётом штрафа"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            probabilities = self.sigmoid(X @ self.weights + self.bias)
            self.loss_history.append(self.compute_loss(y, probabilities))

            errors = probabilities - y
            dw = (X.T @ errors) / n_samples
            dw += self.regularization.gradient(self.weights)
            db = np.mean(errors)

            self.weights -= self.learning_rate * dw
            self.weights = self.regularization.l1_step(self.weights, self.learning_rate)
            self.bias -= self.learning_rate * db

        return self