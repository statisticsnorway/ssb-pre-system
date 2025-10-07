import warnings
from typing import cast

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)


def additive_benchmark(
    df_indicator: pd.DataFrame,
    df_target: pd.DataFrame,
    liste_km: list[str] | str,
    startyear: int,
    endyear: int,
) -> pd.DataFrame:
    """Adjust values in df_target to match df_indicator with additive quota adjustment.

    This function adjusts the values in df_target to match the values in df_indicator
    by first aggregating df_target to the same frequency as df_indicator.
    Then calculating the difference between the two, spreading the difference evenly
    across the corresponding periods in df_target, and adding the difference to df_target.
    The adjusted df_indicator is then returned, with non-overlapping columns left untreated.

    Author: Benedikt Goodman, National accounts Department and
    Vemund Rundberget, Macroeconomics Department, Research Division, SSB

    Args:
        df_indicator: DataFrame containing the indicator values that will be benchmarked.
            Must have a PeriodIndex.
        df_target: DataFrame containing the values to benchmark additively against.
            Must have a PeriodIndex.
        liste_km: List of series names (columns) to benchmark. If a single string is
            provided, it is automatically wrapped into a list.
        startyear: The starting year (inclusive) of the benchmarking period.
        endyear: The ending year (inclusive) of the benchmarking period.
            Must be <= 2099.

    Returns:
        pd.DataFrame: An adjusted DataFrame with the same frequency as the original df_target,
            with non-overlapping columns left untreated.

    Raises:
        TypeError: If inputs are not DataFrames, indices are not PeriodIndex,
            parameters are of incorrect type, or required columns are missing.
        AssertionError: If the chosen start or end years are not available in either
            the indicator or target DataFrames.

    Notes:
        - Any series containing only zeros, non-numeric values, or NaNs will be
          excluded from benchmarking.
        - The method ensures consistency: the sum of the adjusted indicator series
          over each target period will exactly match the corresponding target value.
        - This corresponds to an additive Denton-style temporal disaggregation.

    Examples:
        >>> import pandas as pd
        >>> from pre_system.additive_benchmark import additive_benchmark
        >>> idx_monthly = pd.period_range("2022-01", periods=6, freq="M")
        >>> idx_quarterly = pd.period_range("2022Q1", periods=2, freq="Q")
        >>> df_indicator = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=idx_monthly)
        >>> df_target = pd.DataFrame({"value": [21, 15]}, index=idx_quarterly)
        >>> additive_benchmark(df_indicator, df_target, ["value"], 2022, 2022).sum()
        value    36.0
        dtype: float64
    """
    # Checking object types.
    if not isinstance(df_indicator, pd.DataFrame):
        raise TypeError("The indicator dataframe is not a pd.DataFrame.")
    if not isinstance(df_target, pd.DataFrame):
        raise TypeError("The target dataframe is not a pd.DataFrame.")
    if not isinstance(liste_km, (list, str)):
        raise TypeError(
            "You need to create a list of all the series you wish to benchmark, and it must be in the form of a list or string."
        )
    if isinstance(liste_km, str):
        liste_km = [liste_km]
    if not isinstance(startyear, int):
        raise TypeError("The start year must be an integer.")
    if not isinstance(endyear, int):
        raise TypeError("The end year must be an integer.")
    if endyear > 2099:
        raise TypeError(
            "The final year must be less than 2099. Are you sure you entered it correctly?."
        )

    # Checking indeces.
    if not isinstance(df_indicator.index, pd.PeriodIndex):
        raise TypeError("Index must be a pd.PeriodIndex in the indicator DataFrame.")
    if not isinstance(df_target.index, pd.PeriodIndex):
        raise TypeError("Index must be a pd.PeriodIndex in the target DataFrame.")

    # Checking columns.
    if not pd.Series(liste_km).isin(df_indicator.columns).all():
        raise TypeError(
            f"{np.setdiff1d(liste_km, df_indicator.columns).tolist()} are missing in the indicator dataframe."
        )
    if not pd.Series(liste_km).isin(df_target.columns).all():
        raise TypeError(
            f"{np.setdiff1d(liste_km, df_target.columns).tolist()} are missing in the target dataframe."
        )

    # Filters out series not sent to chaining.
    df_indicator_of_concern = df_indicator[
        df_indicator.columns[df_indicator.columns.isin(liste_km)]
    ]
    mask_indicator = (df_indicator_of_concern.index.year <= endyear) & (  # type: ignore[attr-defined]
        df_indicator_of_concern.index.year >= startyear  # type: ignore[attr-defined]
    )
    df_indicator_of_concern = df_indicator_of_concern.loc[mask_indicator, :]

    df_target_of_concern = df_target[
        df_target.columns[df_target.columns.isin(liste_km)]
    ]
    # Filters out series not sent to chaining.
    mask_target = (df_target_of_concern.index.year <= endyear) & (  # type: ignore[attr-defined]
        df_target_of_concern.index.year >= startyear  # type: ignore[attr-defined]
    )
    df_target_of_concern = df_target_of_concern.loc[mask_target, :]

    # Value checks for df_indicator.
    indicatorzerowarnlist = []  # Zeroes checks.
    for col in df_indicator_of_concern.columns:
        if (df_indicator_of_concern[col] == 0).all():
            indicatorzerowarnlist.append(f"{col}")
    if len(indicatorzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {indicatorzerowarnlist} in the indicator dataframe.",
            UserWarning,
            stacklevel=2,
        )
    indicatorintwarnlist = []  # Non-int check.
    for col in df_indicator_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(df_indicator_of_concern[col])
            and col
            not in df_indicator_of_concern.columns[
                df_indicator_of_concern.isna().any()
            ].to_list()
        ):
            indicatorintwarnlist.append(f"{col}")
    if len(indicatorintwarnlist) > 0:
        warnings.warn(
            f"There are values in {indicatorintwarnlist} in the indicator dataframe that are not real numbers. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )
    if df_indicator_of_concern.isna().any().any() is np.True_:  # NaN check.
        warnings.warn(
            f"There are NaN-values in {df_indicator_of_concern.columns[df_indicator_of_concern.isna().any()].to_list()} in the indicator dataframe. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )
    # Value checks for df_target.
    targetzerowarnlist = []  # Zeroes checks.
    for col in df_target_of_concern.columns:
        if (df_target_of_concern[col] == 0).all():
            targetzerowarnlist.append(f"{col}")
    if len(targetzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {targetzerowarnlist} in the target dataframe.",
            UserWarning,
            stacklevel=2,
        )
    targetintwarnlist = []  # Non-int check.
    for col in df_target_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(df_target_of_concern[col])
            and col
            not in df_target_of_concern.columns[
                df_target_of_concern.isna().any()
            ].to_list()
        ):
            targetintwarnlist.append(f"{col}")
    if len(targetintwarnlist) > 0:
        warnings.warn(
            f"There are values in {targetintwarnlist} in the target dataframe that are not real numbers. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )
    if df_target_of_concern.isna().any().any() is np.True_:  # NaN check.
        warnings.warn(
            f"There are NaN-values in {df_target_of_concern.columns[df_target_of_concern.isna().any()].to_list()} in the target dataframe. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )

    # Filters out valintwarnlist
    df_target_of_concern = df_target_of_concern[
        [col for col in df_target_of_concern.columns if col not in targetintwarnlist]
    ]
    df_indicator_of_concern = df_indicator_of_concern[
        [col for col in df_indicator_of_concern.columns if col not in targetintwarnlist]
    ]
    # Filters out mnrintwarnlist
    df_target_of_concern = df_target_of_concern[
        [col for col in df_target_of_concern.columns if col not in indicatorintwarnlist]
    ]
    df_indicator_of_concern = df_indicator_of_concern[
        [
            col
            for col in df_indicator_of_concern.columns
            if col not in indicatorintwarnlist
        ]
    ]
    liste_km = [
        serie
        for serie in liste_km
        if serie
        not in (
            indicatorintwarnlist
            + targetintwarnlist
            + df_target_of_concern.columns[df_target_of_concern.isna().any()].to_list()
            + df_indicator_of_concern.columns[
                df_indicator_of_concern.isna().any()
            ].to_list()
        )
    ]

    # Logical checks.
    # Checking that start and end years are in range.
    if pd.Period(str(startyear), freq="Y") not in df_indicator_of_concern.index.asfreq(
        "Y"
    ):
        raise AssertionError("Selected start year not in the indicator dataframe.")
    if pd.Period(str(endyear), freq="Y") not in df_indicator_of_concern.index.asfreq(
        "Y"
    ):
        raise AssertionError("Selected end year not in the indicator dataframe.")
    if pd.Period(str(startyear), freq="Y") not in df_target_of_concern.index.asfreq(
        "Y"
    ):
        raise AssertionError("Selected start year not in the target dataframe.")
    if pd.Period(str(endyear), freq="Y") not in df_target_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected end year not in the target dataframe.")

    # Aggregate df_indicator_of_concern to the frequency of df_target_of_concern
    df_indicator_agg = df_indicator_of_concern.resample(
        df_target_of_concern.index.freqstr
    ).sum()

    # Calculates amount of months in a quarter or year and so forth
    num_periods = len(
        pd.period_range(
            start=pd.Period(
                pd.Period("1900", df_target_of_concern.index.freqstr).start_time,
                df_indicator_of_concern.index.freqstr,
            ),
            end=pd.Period(
                pd.Period("1900", df_target_of_concern.index.freqstr).end_time,
                df_indicator_of_concern.index.freqstr,
            ),
            freq=df_indicator_of_concern.index.freqstr,
        )
    )

    # Calculate the difference between df_indicator_of_concern and the aggregated df_target_of_concern
    diff = (df_target_of_concern - df_indicator_agg) / num_periods

    # Disaggregate diff
    diff = diff.resample(df_indicator.index.freqstr).ffill()

    # Add difference
    df_indicator_of_concern = df_indicator_of_concern + diff

    return cast(pd.DataFrame, df_indicator_of_concern)
