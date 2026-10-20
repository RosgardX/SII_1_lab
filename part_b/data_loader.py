from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "train.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Загружает датасет из CSV-файла."""
    if not path.exists():
        raise FileNotFoundError(f"Датасет не найден: {path}")

    return pd.read_csv(path)