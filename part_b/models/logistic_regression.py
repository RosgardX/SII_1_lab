import numpy as np


class LogisticRegression:
    """Логистическая регрессия с обучением методом градиентного спуска."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 3000,
        threshold: float = 0.5,
        eps: float = 1e-15,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.eps = eps
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Переводит линейный выход в вероятность от 0 до 1."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y: np.ndarray, probabilities: np.ndarray) -> float:
        """Binary Cross-Entropy (LogLoss)."""
        p = np.clip(probabilities, self.eps, 1 - self.eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучает модель, обновляя веса градиентным спуском."""
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
            db = np.mean(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятность класса 1 (Transported = True)."""
        X = np.asarray(X, dtype=float)
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает метки классов 0/1 по порогу threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)
