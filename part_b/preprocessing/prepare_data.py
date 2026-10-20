from pathlib import Path

import numpy as np
import pandas as pd

from part_b.data_loader import load_data


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "train_processed.csv"

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLUMNS = ["HomePlanet", "Destination", "Deck", "Side"]
BOOL_COLUMNS = ["CryoSleep", "VIP"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт новые признаки из PassengerId и Cabin."""
    df = df.copy()

    # PassengerId = gggg_pp, где gggg — номер группы (семья / компания)
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    return df


def fill_by_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_spend = df[SPEND_COLUMNS].sum(axis=1, min_count=1)

    group_planet = df.groupby("Group")["HomePlanet"].transform(
        lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(group_planet)

    deck_planet = df["Deck"].map(
        {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    )
    df["HomePlanet"] = df["HomePlanet"].fillna(deck_planet)

    df.loc[df["CryoSleep"].isna() & (total_spend > 0), "CryoSleep"] = False

    for column in SPEND_COLUMNS:
        df.loc[df[column].isna() & (df["CryoSleep"] == True), column] = 0

    for column in SPEND_COLUMNS:
        df.loc[df[column].isna() & (df["Age"] < 13), column] = 0

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Детерминированная подготовка: признаки + правила заполнения."""
    df = add_features(df)
    df = fill_by_rules(df)
    df["Transported"] = df["Transported"].astype(int)

    return df.drop(columns=["PassengerId", "Name", "Cabin", "Group"])

def fit_preprocessor(train: pd.DataFrame) -> dict:
    """Запоминает медианы, моды и категории по обучающей выборке."""
    params = {"median": {}, "mode": {}, "categories": {}}

    for column in ["Age", "CabinNum"] + SPEND_COLUMNS:
        params["median"][column] = train[column].median()

    for column in CATEGORICAL_COLUMNS + BOOL_COLUMNS:
        params["mode"][column] = train[column].mode()[0]

    for column in CATEGORICAL_COLUMNS:
        params["categories"][column] = sorted(
            train[column].fillna(params["mode"][column]).unique()
        )

    return params


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Заполняет оставшиеся пропуски и кодирует признаки по параметрам train."""
    df = df.copy()

    for column, value in params["median"].items():
        df[column] = df[column].fillna(value)

    for column, value in params["mode"].items():
        df[column] = df[column].fillna(value)

    for column in BOOL_COLUMNS:
        df[column] = df[column].astype(bool).astype(int)

    df["TotalSpend"] = df[SPEND_COLUMNS].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    for column in SPEND_COLUMNS + ["TotalSpend"]:
        df[column] = np.log1p(df[column])

    df["IsChild"] = (df["Age"] < 13).astype(int)

    # One-hot с фиксированным набором категорий из train
    for column in CATEGORICAL_COLUMNS:
        df[column] = pd.Categorical(df[column], categories=params["categories"][column])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)

    return df


if __name__ == "__main__":
    raw = load_data()
    processed = prepare_dataset(raw)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Размер исходного датасета: {raw.shape}")
    print(f"Размер после подготовки: {processed.shape}")
    print("\nПропуски до / после правил заполнения:")
    before = add_features(raw).isna().sum()
    after = processed.isna().sum()
    print(pd.DataFrame({"До": before, "После правил": after}).dropna()
          .astype(int).query("`До` > 0").to_string())
    print(f"\nОбработанный датасет сохранён: {PROCESSED_DATA_PATH}")
    print("Оставшиеся пропуски заполняются медианой/модой train в train.py")
