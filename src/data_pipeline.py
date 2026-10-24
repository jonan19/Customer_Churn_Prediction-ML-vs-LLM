"""
Data loading + cleaning.

The raw CSV is committed to this repo (data/raw/telco_churn_raw.csv) so
the pipeline is reproducible even if the upstream mirror disappears; if
that file is ever missing, `load_raw()` re-downloads it from the public
GitHub mirror of IBM's sample Telco Customer Churn dataset.
"""
import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    if config.RAW_CSV_PATH.exists():
        logger.info(f"Loading cached raw data from {config.RAW_CSV_PATH}")
        return pd.read_csv(config.RAW_CSV_PATH)

    logger.info(f"No local cache found; downloading from {config.DATA_URL}")
    df = pd.read_csv(config.DATA_URL)
    config.RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CSV_PATH, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in this dataset.

    `TotalCharges` is loaded as an object dtype because 11 rows hold a
    blank string instead of a number. Every one of those 11 rows has
    `tenure == 0` -- i.e. a brand-new customer who has not been billed
    yet -- so the correct fix is to treat the blank as 0, not to drop the
    row or impute a mean (which would fabricate billing history that
    doesn't exist).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    blank_mask = df["TotalCharges"].isna()
    assert (df.loc[blank_mask, "tenure"] == 0).all(), (
        "Found a blank TotalCharges row with nonzero tenure -- the "
        "'brand new customer' explanation doesn't hold, investigate before filling with 0."
    )
    df.loc[blank_mask, "TotalCharges"] = 0.0

    df[config.TARGET_COL] = (df[config.TARGET_COL] == "Yes").astype(int)
    return df


if __name__ == "__main__":
    raw = load_raw()
    clean_df = clean(raw)
    print(clean_df.shape)
    print(clean_df[config.TARGET_COL].value_counts(normalize=True))
