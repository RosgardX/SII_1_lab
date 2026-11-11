from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "day.csv"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "day_processed.csv"
PLOTS_DIR = PROJECT_DIR / "plots"

def load_dataset(processed: bool = False) -> pd.DataFrame:
    """Загружает исходный или обработанный датасет."""
    path = PROCESSED_DATA_PATH if processed else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    return pd.read_csv(path)