import numpy as np


PENALTIES = ("none", "l1", "l2", "elasticnet")


class Regularization:
    """Штраф за величину весов: none, L1, L2 или ElasticNet."""

    def __init__(self, penalty: str = "none", alpha: float = 0.0, l1_ratio: float = 0.5):
        if penalty not in PENALTIES:
            raise ValueError(f"penalty должен быть одним из {PENALTIES}")
        if alpha < 0:
            raise ValueError("alpha должен быть неотрицательным")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio должен находиться между 0 и 1")

        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    @property
    def l1(self) -> float:
        """Сила L1-части штрафа."""
        if self.penalty == "l1":
            return self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * self.l1_ratio
        return 0.0

    @property
    def l2(self) -> float:
        """Сила L2-части штрафа"""
        if self.penalty == "l2":
            return self.alpha
        if self.penalty == "elasticnet":
            return self.alpha * (1 - self.l1_ratio)
        return 0.0

    def loss(self, weights: np.ndarray) -> float:
        """Штраф, который прибавляется к loss"""
        return float(self.l1 * np.sum(np.abs(weights)) + self.l2 * np.sum(weights ** 2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Градиент L2-части штрафа"""
        return 2 * self.l2 * weights

    def l1_step(self, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """L1-шаг"""
        if self.l1 == 0:
            return weights

        shifted = weights - learning_rate * self.l1 * np.sign(weights)
        # Вес, который перескочил через ноль, становится ровно нулём
        shifted[np.sign(shifted) != np.sign(weights)] = 0.0
        return shifted
