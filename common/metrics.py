import numpy as np


#Регрессия

def mae(y_true, y_pred) -> float:
    """Средняя абсолютная ошибка"""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred) -> float:
    """Средняя квадратичная ошибка"""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """Корень из MSE — ошибка в единицах таргета"""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    """Коэффициент детерминации"""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


#Классификация

def accuracy(y_true, y_pred) -> float:
    """Доля правильных ответов"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Матрица ошибок: строка — истинный класс, столбец — предсказанный"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[index[true], index[pred]] += 1

    return matrix


def per_class_scores(matrix: np.ndarray) -> dict:
    """Precision, recall и F1 для каждого класса по матрице ошибок"""
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp   # предсказали класс, а он не тот
    fn = matrix.sum(axis=1) - tp   # класс был, но модель скипнула в другой

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)

    return {"precision": precision, "recall": recall, "f1": f1}


def binary_scores(y_true, y_pred, positive=1) -> dict:
    """Precision, recall и F1 для положительного класса"""
    labels = np.array([1 - positive, positive])
    scores = per_class_scores(confusion_matrix(y_true, y_pred, labels))
    return {name: float(values[1]) for name, values in scores.items()}


def macro_f1(y_true, y_pred) -> float:
    """Среднее F1 по классам"""
    return float(per_class_scores(confusion_matrix(y_true, y_pred))["f1"].mean())


def log_loss(y_true, probabilities, eps: float = 1e-15) -> float:
    """Binary Cross-Entropy: −(1/n)·Σ[y·log p + (1−y)·log(1−p)]."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
