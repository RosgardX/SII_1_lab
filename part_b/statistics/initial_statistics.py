
import pandas as pd

from part_b.data_loader import load_data

def show_initial_statistics(data: pd.DataFrame) -> None:
    """Выводит общую начальную статистику датасета."""

    print("Размер датасета")
    print(f"Количество строк: {data.shape[0]}")
    print(f"Количество столбцов: {data.shape[1]}")

    print("\nТипы данных и количество непустых значений")
    data.info()

    print("\nПервые пять строк")
    print(data.head().to_string(index=False))

    show_missing_values(data)
    show_duplicates(data)
    show_numeric_statistics(data)
    show_unique_values(data)
    show_outliers(data)
    show_target_distribution(data)

def show_duplicates(data: pd.DataFrame) -> None:
    """Проверяет дубликаты без учёта PassengerId."""

    comparison_data = data.drop(columns=["PassengerId"], errors="ignore")
    duplicated_mask = comparison_data.duplicated(keep=False)
    duplicated_rows = data[duplicated_mask]

    print("\nПроверка дубликатов без PassengerId")
    print(
        "Количество повторяющихся строк:",
        comparison_data.duplicated().sum()
    )

    if not duplicated_rows.empty:
        print(duplicated_rows.to_string(index=False))
    else:
        print("Дубликаты не обнаружены.")


def show_missing_values(data: pd.DataFrame) -> None:
    """Выводит количество и долю пропущенных значений."""

    print("\nПропущенные значения")

    missing = pd.DataFrame({
        "Количество": data.isna().sum(),
        "Доля, %": (data.isna().mean() * 100).round(2),
    })

    missing = missing[missing["Количество"] > 0]

    if missing.empty:
        print("Пропущенные значения отсутствуют.")
    else:
        print(missing.to_string())


def show_numeric_statistics(data: pd.DataFrame) -> None:
    """Выводит описательную статистику числовых признаков."""

    print("\nСтатистика числовых признаков")
    print(data.describe().round(2).to_string())


def show_unique_values(data: pd.DataFrame) -> None:
    """Выводит количество уникальных значений по столбцам."""

    print("\nКоличество уникальных значений")
    print(data.nunique(dropna=True).to_string())


def show_outliers(data: pd.DataFrame) -> None:
    """Выявляет потенциальные выбросы методом межквартильного размаха."""

    print("\nПотенциальные выбросы по методу IQR")

    numeric_data = data.select_dtypes(include="number")
    outlier_rows = []

    for column in numeric_data.columns:
        if column == "PassengerId":
            continue

        values = numeric_data[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = values[
            (values < lower_bound) | (values > upper_bound)
        ]

        outlier_rows.append({
            "Признак": column,
            "Нижняя граница": round(lower_bound, 2),
            "Верхняя граница": round(upper_bound, 2),
            "Количество выбросов": len(outliers),
            "Доля, %": round(len(outliers) / len(values) * 100, 2),
        })

    if outlier_rows:
        print(pd.DataFrame(outlier_rows).to_string(index=False))
    else:
        print("Числовые признаки для проверки отсутствуют.")


def show_target_distribution(data: pd.DataFrame) -> None:
    """Выводит распределение целевой переменной Transported."""

    target = "Transported"

    if target not in data.columns:
        return

    print("\nРаспределение целевой переменной Transported")

    counts = data[target].value_counts(dropna=False)
    percentages = (
        data[target].value_counts(normalize=True, dropna=False) * 100
    )

    target_statistics = pd.DataFrame({
        "Количество": counts,
        "Доля, %": percentages.round(2),
    })

    print(target_statistics.to_string())

if __name__ == "__main__":
    dataset = load_data()
    show_initial_statistics(dataset)
