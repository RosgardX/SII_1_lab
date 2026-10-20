import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from part_a.data_loader import PROJECT_DIR, PLOTS_DIR


RESULTS_DIR = PROJECT_DIR / "data" / "processed"


def evaluate_predictions(actual, predicted) -> dict:
    """Вычисляет основные метрики качества регрессии."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R²": r2_score(actual, predicted)
    }


if __name__ == "__main__":
    results = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    history = pd.read_csv(RESULTS_DIR / "loss_history.csv")

    model_metrics = evaluate_predictions(
        results["actual"], results["predicted"]
    )
    baseline_metrics = evaluate_predictions(
        results["actual"], results["baseline"]
    )

    print("Качество на тестовой выборке")
    print(pd.DataFrame({
        "Линейная регрессия": model_metrics,
        "Базовая модель": baseline_metrics
    }).round(2))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history["iteration"], history["mse"])
    plt.xlabel("Итерация")
    plt.ylabel("MSE")
    plt.title("Изменение ошибки в процессе обучения")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_history.png")
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(results["actual"], results["predicted"], alpha=0.6)

    min_value = min(results["actual"].min(), results["predicted"].min())
    max_value = max(results["actual"].max(), results["predicted"].max())

    plt.plot([min_value, max_value], [min_value, max_value], "--")
    plt.xlabel("Фактическое количество аренд")
    plt.ylabel("Предсказанное количество аренд")
    plt.title("Фактические и предсказанные значения")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png")
    plt.close()

    print(f"\nГрафики сохранены в папке: {PLOTS_DIR}")