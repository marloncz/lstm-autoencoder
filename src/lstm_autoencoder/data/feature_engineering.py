import logging
from datetime import datetime

import holidays
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    fe_params: dict,
):
    """Add additional (time, holiday, weather) features to consumption data.

    Args:
        df (pd.DataFrame): Consumption data.
        fe_params (dict): Feature engineering parameters.
        df_weather (Optional[pd.DataFrame]): Weather data with timestamp index.

    Raises:
        ValueError: If weather data or weather features are not given, but add_time is set to true.

    Returns:
        pd.DataFrame: Consumption data with added features.
    """
    add_time = fe_params["add_time"]  # Whether to add time features.
    add_holiday = fe_params["add_holiday"]  # Whether to add holiday feature.

    if add_time:
        df = add_time_features(df)

    if add_holiday:
        df = add_holiday_feature(df)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time of day and time of year signals by using sine and cosine transforms.

    Args:
        df (pd.DataFrame): Data with timestamp as index.

    Returns:
        pd.DataFrame: Data including the time signals.
    """
    day = 24 * 60 * 60

    # Get timestamp in seconds
    timestamp_s = df.index.map(pd.Timestamp.timestamp)

    # Get usable sine and cosine transforms to clear time of day and time of year signals
    df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    return df


def add_holiday_feature(df: pd.DataFrame, country: str = "Germany") -> pd.DataFrame:
    """Creates a holiday feature.

    Args:
        df (pd.DataFrame): Data with timestamp as index.
        country (str): Country for which the holidays should be added.

    Returns:
        pd.DataFrame: Data including the holiday feature.
    """
    years = df.index.year.unique().to_list()
    country_holidays_class = getattr(holidays, country)
    holiday_list = []
    for holiday_date, _ in country_holidays_class(years=years).items():
        holiday_list.append(holiday_date)

    # add 26.12. as holiday
    for year in years:
        holiday_list.append(datetime.strptime(f"{year}-12-26", "%Y-%m-%d").date())

    df["holiday"] = df.apply(lambda row: 1 if row.name.date() in holiday_list else 0, axis=1)

    return df
