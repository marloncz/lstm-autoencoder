import logging
import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_test_val_split(
    df: pd.DataFrame,
    params: dict | None = None,
    split_type: str = "index",
    ts_format: str = "%d.%m.%Y",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates train-test-val-split.

    Args:
        df (pd.DataFrame): Data to be split.
        params (dict): Data splitting parameters (ggf modified by get_filter_range())
        split_type (str): Type of split. It can be "index" for classic split or "config" for config based split. Default is "index".
        ts_data (bool): If True, time series data is split based on config parameters.
        ts_format (str): Format of the timestamp column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Train, test and validation data.
    """
    if split_type == "config":
        assert params is not None, "Provide 'params' for config based split."
        # TODO: add assert check for index based split via timestamp
        train_start = pd.to_datetime(params["train_start"], format=ts_format)
        logger.info("Train start: %s", train_start)
        train_end = pd.to_datetime(params["train_end"], format=ts_format)
        logger.info("Train end: %s", train_end)
        val_start = pd.to_datetime(params["val_start"], format=ts_format)
        logger.info("Val start: %s", val_start)
        val_end = pd.to_datetime(params["val_end"], format=ts_format)
        logger.info("Val end: %s", val_end)
        test_start = pd.to_datetime(params["test_start"], format=ts_format)
        logger.info("Test start: %s", test_start)
        test_end = pd.to_datetime(params["test_end"], format=ts_format)
        logger.info("Test end: %s", test_end)

        train = df[train_start:train_end]
        val = df[val_start:val_end]
        test = df[test_start:test_end]
    elif split_type == "index":
        # define sizes for train, test and val based on fixed percentages
        # TODO: add params for split sizes
        train_size = int(len(df) * 0.6)
        val_size = int(len(df) * 0.3)
        test_size = len(df) - train_size - val_size

        logger.info("Train size: %s, Val size: %s, Test size: %s", train_size, val_size, test_size)

        # splitting data based on defined sizes
        train = df.iloc[:train_size]
        val = df.iloc[train_size : train_size + val_size]
        test = df.iloc[train_size + val_size :]
    else:
        raise ValueError("Invalid split type. Choose 'index' or 'config'")

    return train, val, test


def scale_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
    scaler_path: str | None = None,
) -> tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """Standardizes features by removing the mean and scaling to unit variance.

    Args:
        train (pd.DataFrame): Train data to be scaled.
        test (pd.DataFrame): Test data to be scaled.
        val (pd.DataFrame): Validation data to be scaled.
        scaler_path (Optional[str]): Path to the saved scaler.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            train, val and test data.

    """
    sc = StandardScaler()
    cols = train.columns.tolist()

    # exclude binary cols from scaling
    binary_cols = ["holiday", "day_sin", "day_cos", "year_sin", "year_cos"]
    for binary_col in binary_cols:
        if binary_col in cols:
            cols.remove(binary_col)

    logger.info("Fitting StandardScaler on columns: %s", cols)

    sc.fit(train[cols])

    # saving scaler
    if scaler_path is None:
        logger.info(
            "'scaler_path' not provided. Saving scaler to default path ('data/02_intermediate')."
        )
        scaler_path = os.path.join("data/02_intermediate", "scaler.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(sc, f)

    train[cols] = pd.DataFrame(
        sc.transform(train[cols]),
        columns=cols,
        index=train.index,
    )

    val[cols] = pd.DataFrame(
        sc.transform(val[cols]),
        columns=cols,
        index=val.index,
    )

    test[cols] = pd.DataFrame(
        sc.transform(test[cols]),
        columns=cols,
        index=test.index,
    )

    return train, val, test


def apply_scaler(
    df: pd.DataFrame,
    inverse: bool = False,
    scaler_path: str | None = None,
) -> pd.DataFrame:
    """Standardize data by using saved scaler.

    Args:
        df (pd.DataFrame): Data to be scaled.
        inverse (bool): If True, data is inverse transformed.
        scaler_path (Optional[str]): Path to the saved scaler.

    Returns:
        pd.DataFrame: Scaled data.
    """
    if scaler_path is None:
        logger.info(
            "'scaler_path' not provided. Loading scaler from default path ('data/02_intermediate')."
        )
        scaler_path = os.path.join("data/02_intermediate", "scaler.pkl")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    cols = df.columns.tolist()

    # exclude binary cols from scaling
    binary_cols = ["holiday", "day_sin", "day_cos", "year_sin", "year_cos"]
    for binary_col in binary_cols:
        if binary_col in cols:
            cols.remove(binary_col)

    logger.info("Applying scaler to columns %s", cols)

    if inverse:
        logger.info("Inverse transform data with StandardScaler")
        df[cols] = pd.DataFrame(
            scaler.inverse_transform(df[cols]),
            columns=cols,
            index=df.index,
        )
    else:
        logger.info("Transform data with StandardScaler")
        df[cols] = pd.DataFrame(
            scaler.transform(df[cols]),
            columns=cols,
            index=df.index,
        )

    return df
