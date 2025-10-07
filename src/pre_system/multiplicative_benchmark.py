import warnings
from typing import cast

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)


def multiplicative_benchmark(
    df_indicator: pd.DataFrame,
    df_target: pd.DataFrame,
    liste_km: list[str] | str,
    startyear: int,
    endyear: int,
) -> pd.DataFrame:
    """Perform multiplicative benchmarking of high-frequency indicator data against low-frequency target data over a given time range.

    This method adjusts a high-frequency indicator series (e.g., monthly data)
    so that its aggregated values match a lower-frequency target series
    (e.g., annual data). The adjustment is multiplicative: indicators are
    divided by a ratio of their aggregated sums to the target values, and
    then interpolated back to the high-frequency index.

    Author: Magnus Helliesen Kvåle, National accounts Department and
    Vemund Rundberget, Macroeconomics Department, Research Division, SSB

    Args:
      df_indicator: DataFrame with the high-frequency indicator series.
          Must have a PeriodIndex at a higher frequency (e.g., monthly).
      df_target: DataFrame with the low-frequency target (benchmark) series.
          Must have a PeriodIndex at a lower frequency (e.g., yearly).
      liste_km: Column names to benchmark. If a single string is provided,
          it will be wrapped in a list.
      startyear: The starting year (inclusive) of the benchmarking period.
      endyear: The ending year (inclusive) of the benchmarking period.
          Must be <= 2099.

    Returns:
      pd.DataFrame: A DataFrame with the benchmarked indicator series, indexed by the
          same frequency as `df_indicator`.

    Raises:
      TypeError: If inputs are not DataFrames, indices are not PeriodIndex, or
        parameters have incorrect types.
      AssertionError: If the selected start or end year is not present in either
        the indicator or target DataFrames.

    Warns:
      UserWarning: If zero-only series, non-numeric values, or NaNs are detected.
        Such series are excluded from benchmarking.

    Notes:
      - Series containing only zeros, non-numeric values, or NaNs are excluded.
      - Consistency is ensured: the sum of the adjusted indicator over a target
        period exactly matches the corresponding target value.
      - Equivalent to the multiplicative Denton method used in temporal
        disaggregation of time series.

    Examples:
      >>> import pandas as pd
      >>> from pre_system.multiplicative_benchmark import multiplicative_benchmark
      >>> idx_monthly = pd.period_range("2018-01", "2019-12", freq="M")
      >>> idx_yearly = pd.period_range("2018", "2019", freq="Y")
      >>> df_indicator = pd.DataFrame({"A": range(len(idx_monthly))}, index=idx_monthly)
      >>> df_target = pd.DataFrame({"A": [66, 210]}, index=idx_yearly)
      >>> multiplicative_benchmark(df_indicator, df_target, "A", 2018, 2019).head()
                 A
      2018-01  0.0
      2018-02  1.0
      2018-03  2.0
      2018-04  3.0
      2018-05  4.0
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

    df_ratio = (
        df_indicator_of_concern.groupby(
            pd.PeriodIndex(
                df_indicator_of_concern.index, freq=df_target_of_concern.index.freq
            )
        )
        .sum()
        .div(df_target_of_concern)
        .resample(df_indicator_of_concern.index.freq)
        .ffill()
    )

    res_df = df_indicator_of_concern.div(
        df_ratio.fillna(1).reindex(df_indicator_of_concern.index)
    )[df_indicator_of_concern.columns]

    return cast(pd.DataFrame, res_df)
