import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.metrics import rmse, log_loss
from part_2_a.data import load_bike_data, load_spaceship_data, RESULTS_DIR, PLOTS_DIR
from part_2_a.models.regularized_linear_regression import RegularizedLinearRegression
from part_2_a.models.regularized_logistic_regression import RegularizedLogisticRegression


TOLERANCE = 0.001

SETUPS = {
    "linear": {
        "title": "Bike Sharing (линейная регрессия)",
        "load": load_bike_data,
        "model": RegularizedLinearRegression,
        "metric": lambda model, X, y: rmse(y, model.predict(X)),
        "metric_name": "RMSE",
        "loss_name": "MSE",
        "learning_rate": 0.01,
        "iterations": [100, 300, 1000, 3000, 10000, 30000],
    },
    "logistic": {
        "title": "Spaceship Titanic (логистическая регрессия)",
        "load": load_spaceship_data,
        "model": RegularizedLogisticRegression,
        "metric": lambda model, X, y: log_loss(y, model.predict_proba(X)),
        "metric_name": "LogLoss",
        "loss_name": "LogLoss",
        "learning_rate": 0.1,
        "iterations": [100, 300, 1000, 3000, 10000],
    },
}


def run_attempts(setup: dict, data: dict) -> tuple[pd.DataFrame, list]:
    """Обучает модель без штрафа с разным числом итераций."""
    rows = []

    for n_iterations in setup["iterations"]:
        model = setup["model"](learning_rate=setup["learning_rate"], n_iterations=n_iterations)
        model.fit(data["X_train"], data["y_train"])
        rows.append({
            "n_iterations": n_iterations,
            "train": setup["metric"](model, data["X_train"], data["y_train"]),
        })

    return pd.DataFrame(rows), model.loss_history   # кривая самого длинного запуска


def choose_iterations(results: pd.DataFrame) -> int:
    """Минимальное число итераций, после которого train меняется меньше чем на TOLERANCE."""
    final = results["train"].iloc[-1]
    converged = results[np.abs(results["train"] - final) <= abs(final) * TOLERANCE]
    return int(converged["n_iterations"].iloc[0])


def plot_train_vs_iterations(results: pd.DataFrame, setup: dict, chosen: int, path):
    """Метрика на train для каждого запуска."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["n_iterations"], results["train"], color="#2a78d6", linewidth=2,
            marker="o", markersize=8, label="запуски")

    chosen_train = results.loc[results["n_iterations"] == chosen, "train"]
    ax.scatter(chosen, chosen_train, s=220, facecolors="none", edgecolors="black", linewidths=1.5,
               zorder=5, label=f"выбрано: {chosen} итераций")

    ax.set_xscale("log")
    ax.set_xlabel("Число итераций (лог. шкала)")
    ax.set_ylabel(f"{setup['metric_name']} на train")
    ax.set_title(f"{setup['title']}: запуски с разным числом итераций (lr = {setup['learning_rate']:g})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curve(history: list, setup: dict, chosen: int, path):
    """Кривая loss на train самого длинного запуска."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(history) + 1), history, color="#2a78d6", linewidth=2, label="loss на train")
    ax.axvline(chosen, color="black", linestyle="--", linewidth=1.5, label=f"выбрано: {chosen} итераций")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Итерация (лог. шкала)")
    ax.set_ylabel(f"{setup['loss_name']} на train (лог. шкала)")
    ax.set_title(f"{setup['title']}: кривая обучения (lr = {setup['learning_rate']:g})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, setup in SETUPS.items():
        print(f"\n{setup['title']}, lr = {setup['learning_rate']}")
        data = setup["load"]()
        results, history = run_attempts(setup, data)
        chosen = choose_iterations(results)

        results.to_csv(RESULTS_DIR / f"{name}_iterations.csv", index=False)
        plot_train_vs_iterations(results, setup, chosen, PLOTS_DIR / f"{name}_iterations_train.png")
        plot_loss_curve(history, setup, chosen, PLOTS_DIR / f"{name}_iterations_loss.png")

        print(results.rename(columns={"train": f"{setup['metric_name']} train"})
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"Выбрано: {chosen} итераций")

    print(f"\nРезультаты: {RESULTS_DIR}\nГрафики: {PLOTS_DIR}")