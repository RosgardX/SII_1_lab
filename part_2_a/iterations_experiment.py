import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.metrics import rmse, log_loss
from part_2_a.data import load_bike_data, load_spaceship_data, RESULTS_DIR, PLOTS_DIR
from part_2_a.models.regularized_linear_regression import RegularizedLinearRegression
from part_2_a.models.regularized_logistic_regression import RegularizedLogisticRegression


TOLERANCE = 0.001
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

SETUPS = {
    "linear": {
        "title": "Bike Sharing (линейная регрессия)",
        "load": load_bike_data,
        "model": RegularizedLinearRegression,
        "metric": lambda model, X, y: rmse(y, model.predict(X)),
        "metric_name": "RMSE",
        "loss_name": "MSE",
        "learning_rates": [0.001, 0.003, 0.01, 0.03, 0.1],
        "chosen_lr": 0.01,
        "iterations": [100, 300, 1000, 3000, 10000, 30000],
    },
    "logistic": {
        "title": "Spaceship Titanic (логистическая регрессия)",
        "load": load_spaceship_data,
        "model": RegularizedLogisticRegression,
        "metric": lambda model, X, y: log_loss(y, model.predict_proba(X)),
        "metric_name": "LogLoss",
        "loss_name": "LogLoss",
        "learning_rates": [0.01, 0.03, 0.1, 0.3, 1.0],
        "chosen_lr": 0.1,
        "iterations": [100, 300, 1000, 3000, 10000],
    },
}


def run_attempts(setup: dict, data: dict) -> tuple[pd.DataFrame, dict]:
    """Обучает модель без штрафа для каждой пары (lr, число итераций)."""
    rows, histories = [], {}

    for lr in setup["learning_rates"]:
        for n_iterations in setup["iterations"]:
            model = setup["model"](learning_rate=lr, n_iterations=n_iterations)
            model.fit(data["X_train"], data["y_train"])
            rows.append({
                "learning_rate": lr,
                "n_iterations": n_iterations,
                "train": setup["metric"](model, data["X_train"], data["y_train"]),
            })
        histories[lr] = model.loss_history

    return pd.DataFrame(rows), histories


def choose_iterations(results: pd.DataFrame, lr: float) -> int:
    """Минимальное число итераций, после которого train меняется меньше чем на TOLERANCE."""
    runs = results[results["learning_rate"] == lr].sort_values("n_iterations")
    final = runs["train"].iloc[-1]
    converged = runs[np.abs(runs["train"] - final) <= abs(final) * TOLERANCE]
    return int(converged["n_iterations"].iloc[0])


def plot_train_vs_iterations(results: pd.DataFrame, setup: dict, chosen: int, path):
    """Метрика на train от числа итераций — по линии на каждый learning rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for color, lr in zip(COLORS, setup["learning_rates"]):
        runs = results[results["learning_rate"] == lr]
        width = 3 if lr == setup["chosen_lr"] else 1.5
        ax.plot(runs["n_iterations"], runs["train"], color=color, linewidth=width,
                marker="o", markersize=7, label=f"lr = {lr:g}")

    chosen_train = results.query("learning_rate == @setup['chosen_lr'] and n_iterations == @chosen")["train"]
    ax.scatter(chosen, chosen_train, s=200, facecolors="none", edgecolors="black", linewidths=1.5,
               zorder=5, label=f"выбрано: lr = {setup['chosen_lr']:g}, {chosen} итераций")

    ax.set_xscale("log")
    ax.set_xlabel("Число итераций (лог. шкала)")
    ax.set_ylabel(f"{setup['metric_name']} на train")
    ax.set_title(f"{setup['title']}: качество на train при разном числе итераций")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curves(histories: dict, setup: dict, chosen: int, path):
    """Кривые loss на train для самого длинного запуска каждого learning rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for color, (lr, history) in zip(COLORS, histories.items()):
        width = 3 if lr == setup["chosen_lr"] else 1.5
        ax.plot(np.arange(1, len(history) + 1), history, color=color, linewidth=width, label=f"lr = {lr:g}")

    ax.axvline(chosen, color="black", linestyle="--", linewidth=1.5, label=f"выбрано: {chosen} итераций")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Итерация (лог. шкала)")
    ax.set_ylabel(f"{setup['loss_name']} на train (лог. шкала)")
    ax.set_title(f"{setup['title']}: кривые обучения")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, setup in SETUPS.items():
        print(f"\n{setup['title']}")
        data = setup["load"]()
        results, histories = run_attempts(setup, data)
        chosen = choose_iterations(results, setup["chosen_lr"])

        results.to_csv(RESULTS_DIR / f"{name}_iterations.csv", index=False)
        plot_train_vs_iterations(results, setup, chosen, PLOTS_DIR / f"{name}_iterations_train.png")
        plot_loss_curves(histories, setup, chosen, PLOTS_DIR / f"{name}_iterations_loss.png")

        table = results.pivot(index="n_iterations", columns="learning_rate", values="train")
        print(f"{setup['metric_name']} на train (строки — итерации, столбцы — learning rate):")
        print(table.to_string(float_format=lambda v: f"{v:.4g}"))
        print(f"Выбрано: lr = {setup['chosen_lr']}, {chosen} итераций")

    print(f"\nРезультаты: {RESULTS_DIR}\nГрафики: {PLOTS_DIR}")