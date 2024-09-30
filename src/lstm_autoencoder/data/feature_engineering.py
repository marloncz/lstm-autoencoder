import logging
from datetime import datetime

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    fe_params: dict,
    holiday_country: str = "Germany",
):
    """Add additional (time, holiday, weather) features to consumption data.

    Args:
        df (pd.DataFrame): Consumption data.
        fe_params (dict): Feature engineering parameters.
        country (str): Country for which the holidays should be added. Default is 'Germany'.

    Raises:
        ValueError: If weather data or weather features are not given, but add_time is set to true.

    Returns:
        pd.DataFrame: Consumption data with added features.
    """
    add_time = fe_params["add_time"]  # Whether to add time features.
    add_holiday = fe_params["add_holiday"]  # Whether to add holiday feature.

    if add_time:
        logger.info("Adding time features.")
        df = add_time_features(df)

    if add_holiday:
        logger.info("Adding holidays for country %s", holiday_country)
        df = add_holiday_feature(df)

    return df


def add_time_features(
    df: pd.DataFrame, period: str = "day", custom_interval: int | float | None = None
) -> pd.DataFrame:
    """Creates time signals (sine and cosine) for a specified time period, supporting both timestamp and numeric indices.

    Args:
        df (pd.DataFrame): Data with timestamp or numeric values as index.
        period (str): The period to create time features for. Options are:
            'hour', 'day', 'week', 'month', 'year', or 'custom' for a user-defined time period.
        custom_interval (int): Custom time interval (used if period is 'custom').

    Returns:
        pd.DataFrame: Data including the time signals.
    """
    # mapping dict for period to seconds
    period_seconds = {
        "hour": 60 * 60,
        "day": 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
        "month": 30.44 * 24 * 60 * 60,  # apprx. average month length
        "year": 365.25 * 24 * 60 * 60,  # account for leap years
    }

    # if index is numeric, it will be used directly
    if pd.api.types.is_numeric_dtype(df.index):
        timestamp_s = df.index.values
        period = "custom"
        assert (
            period == "custom" and custom_interval is not None
        ), "If index is numeric, period must be 'custom' with custom_interval!"
    # if index is datetime, convert to timestamp in seconds
    else:
        timestamp_s = df.index.map(pd.Timestamp.timestamp)

    # Determine the period in seconds to use for sine and cosine transforms
    if period == "custom" and custom_interval is not None:
        period_s = custom_interval
    elif period in period_seconds:
        period_s = period_seconds[period]
    else:
        raise ValueError(
            f"Invalid period: {period}. Choose from 'hour', 'day', 'week', 'month', 'year', or use 'custom' with custom_interval."
        )

    # Apply sine and cosine transforms based on the selected period
    df[f"{period}_sin"] = np.sin(timestamp_s * (2 * np.pi / period_s))
    df[f"{period}_cos"] = np.cos(timestamp_s * (2 * np.pi / period_s))

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
