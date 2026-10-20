
import pandas as pd
import matplotlib.pyplot as plt

from part_a.data_loader import load_dataset, PLOTS_DIR


def print_statistics(df: pd.DataFrame) -> None:
    """Проверяет обработанный датасет."""
    print("Общая информация")
    print(f"Размер датасета: {df.shape}")
    print(f"Объём в памяти: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    print("\nПервые пять строк")
    print(df.head().to_string(index=False))

    print("\nПропуски:", df.isna().sum().sum())
    print("Полные дубликаты:", df.duplicated().sum())

    print("\nСтатистика признаков")
    print(df.describe().T.round(2).to_string())

    dummy_columns = [
        column for column in df.columns
        if any(
            column.startswith(prefix)
            for prefix in ("season_", "mnth_", "weekday_", "weathersit_")
        )
    ]

    invalid_dummy_columns = [
        column for column in dummy_columns
        if not df[column].isin([0, 1]).all()
    ]

    print("\nСтолбцы с некорректными значениями:", len(invalid_dummy_columns))

    print("\nКорреляция признаков с cnt")
    correlations = df.corr(numeric_only=True)["cnt"].drop("cnt")
    print(correlations.sort_values(ascending=False).round(3).to_string())


def save_plots(df: pd.DataFrame) -> None:
    """Сохраняет корреляции cnt с признаками."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    correlations = (
        df.corr(numeric_only=True)["cnt"]
        .drop("cnt")
        .sort_values()
    )

    correlations.plot(
        kind="barh",
        title="Корреляция признаков с количеством аренд",
        xlabel="Коэффициент корреляции",
        figsize=(9, max(5, len(correlations) * 0.35)),
        grid=True
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlations_after.png")
    plt.close()


if __name__ == "__main__":
    dataset = load_dataset(processed=True)
    print_statistics(dataset)
    save_plots(dataset)
