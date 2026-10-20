import pandas as pd

from config import DATA_PATH, TIME_COL, VAL_START, TEST_START


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    return df.sort_values([TIME_COL, "transaction_id"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    t = df[TIME_COL]
    train = df[t < VAL_START]
    val = df[(t >= VAL_START) & (t < TEST_START)]
    test = df[t >= TEST_START]
    return train, val, test
