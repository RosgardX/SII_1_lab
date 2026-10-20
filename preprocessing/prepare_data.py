from data_loader import load_dataset, PROCESSED_DATA_PATH

import pandas as pd


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает признаки для линейной регрессии."""
    df = df.copy()

    df = df.drop(
        columns=["instant", "dteday", "casual", "registered"]
    )

    categorical_columns = [
        "season",
        "mnth",
        "weekday",
        "weathersit",
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    return df


if __name__ == "__main__":
    df = load_dataset()
    processed = prepare_dataset(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Размер исходного датасета: {df.shape}")
    print(f"Размер обработанного датасета: {processed.shape}")
    print(f"Обработанный датасет сохранён: {PROCESSED_DATA_PATH}")