from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "day.csv"
PLOTS_DIR = PROJECT_DIR / "plots"


def load_dataset(path: Path) -> pd.DataFrame:
    """Загружает исходный датасет."""
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")

    return pd.read_csv(path)


def print_statistics(df: pd.DataFrame) -> None:
    """Выводит основную статистику и проверяет качество данных."""
    print("Общая информация")
    print(f"Размер датасета: {df.shape}")
    print(f"Объём в памяти: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    print("\nПервые пять строк")
    print(df.head().to_string(index=False))

    print("\nТипы данных ")
    print(df.dtypes.to_string())

    print("\n Пропуски")
    print(df.isna().sum().to_string())

    print(f"\nПолные дубликаты: {df.duplicated().sum()}")

    print("\nОписательная статистика")
    print(df.describe().T.round(2).to_string())

    print("\nУникальные значения категориальных признаков")
    for column in ["season", "yr", "mnth", "holiday",
                   "weekday", "workingday", "weathersit"]:
        print(f"\n{column}:")
        print(df[column].value_counts().sort_index().to_string())

    print("\nЦелевая переменная: cnt")
    print(df["cnt"].describe().round(2).to_string())

    print(
        "\nПроверка casual + registered = cnt: "
        f"{(df['casual'] + df['registered']).equals(df['cnt'])}"
    )


def save_plots(df: pd.DataFrame) -> None:
    """Сохраняет графики распределения и динамики аренд."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Распределение количества аренд за день
    df["cnt"].plot(
        kind="hist",
        bins=25,
        title="Распределение количества аренд за день",
        xlabel="Количество аренд за день",
        ylabel="Количество дней",
        grid=True,
        figsize=(9, 5),
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cnt_distribution_before.png")
    plt.close()

    dates = df.copy()
    dates["dteday"] = pd.to_datetime(dates["dteday"])
    dates = dates.sort_values("dteday")

    ax = dates.plot(
        x="dteday",
        y="cnt",
        kind="line",
        title="Количество аренд по дням",
        xlabel="Дата",
        ylabel="Количество аренд",
        legend=False,
        grid=True,
        figsize=(12, 5),
    )

    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m.%Y"))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cnt_over_time_before.png")
    plt.close()


if __name__ == "__main__":
    dataset = load_dataset(DATA_PATH)
    print_statistics(dataset)
    save_plots(dataset)