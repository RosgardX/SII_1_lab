import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix
)

from part_b.preprocessing.prepare_data import PROJECT_DIR


RESULTS_DIR = PROJECT_DIR / "data" / "processed"
PLOTS_DIR = PROJECT_DIR / "plots"


def evaluate_predictions(actual, predicted, probability) -> dict:
    """Вычисляет метрики качества бинарной классификации."""
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "LogLoss": log_loss(actual, probability, labels=[0, 1]),
    }


def show_confusion(actual, predicted, name: str) -> None:
    """Выводит TP / FP / FN / TN."""
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    print(f"\n{name}:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")


if __name__ == "__main__":
    results = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    history = pd.read_csv(RESULTS_DIR / "loss_history.csv")
    weights = pd.read_csv(RESULTS_DIR / "weights.csv")

    model_metrics = evaluate_predictions(
        results["actual"], results["predicted"], results["probability"]
    )
    baseline_metrics = evaluate_predictions(
        results["actual"], results["baseline"], results["baseline_probability"]
    )

    print("Качество на тестовой выборке")
    print(pd.DataFrame({
        "Логистическая регрессия": model_metrics,
        "Majority baseline": baseline_metrics,
    }).round(4))

    show_confusion(results["actual"], results["predicted"], "Логистическая регрессия")
    show_confusion(results["actual"], results["baseline"], "Majority baseline")

    print("\nСамые сильные признаки (по |весу|)")
    top = weights.reindex(weights["weight"].abs().sort_values(ascending=False).index)
    print(top.head(10).round(3).to_string(index=False))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history["iteration"], history["logloss"])
    plt.xlabel("Итерация")
    plt.ylabel("LogLoss (BCE)")
    plt.title("Изменение ошибки в процессе обучения")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_history.png")
    plt.close()

    cm = confusion_matrix(results["actual"], results["predicted"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["False", "True"])
    ax.set_yticks([0, 1], ["False", "True"])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Факт")
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    weights.sort_values("weight").plot(
        kind="barh", x="feature", y="weight", legend=False, grid=True,
        title="Веса логистической регрессии", figsize=(9, 9)
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "weights.png")
    plt.close()

    print(f"\nГрафики сохранены в папке: {PLOTS_DIR}")
