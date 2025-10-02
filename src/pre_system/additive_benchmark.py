# +
import warnings

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
    """TLDR: Adjust values in df_target to match df_indicator with additive quota adjustment.

    Adjust values in df_target to match df_indicator using additive quota adjustment.

    This function adjusts the values in df_target to match the values in df_indicator by first aggregating df_target to the same
    frequency as df_indicator, then calculating the difference between the two, spreading the difference evenly across the
    corresponding periods in df_target, and adding the difference to df_target. The adjusted df_indicator is then returned,
    with non-overlapping columns left untreated.

    Author: Benedikt Goodman, Seksjon for Nasjonalregnskap and Vemund Rundberget, Seksjon for makroøkonomi, Forksningsavdelingen, SSB

    Parameters
    ----------
    df_indicator : pd.DataFrame
        The DataFrame containing the indicator values that will be benchmarked. Must have a period index.
    df_target : pd.DataFrame
        The DataFrame containing the values to benchmark additively against. Must have a period index.
    strict_mode : bool, optional
        If True, raise an error if the time intervals are not of equal length.
        If False, issue a warning instead of raising an error. Default is True.

    Returns:
    -------
    pd.DataFrame
        The adjusted df_indicator with the same frequency as the original df_target, with non-overlapping columns left untreated.

    Raises:
    ------
    ValueError
        If either of the input DataFrames does not have a datetime or period index, or if df_target has a finer frequency
        than df_indicator (i.e. df_indicator has quarterly data and df_target has monthly data)
    KeyError
        If no overlapping column names between df_indicator and df_target exist.
    IndexError
        If strict_mode is True and the time intervals are not of equal length.
    UserWarning
        If strict_mode is False and the time intervals are not of equal length.

    Examples:
    --------
    >>> import pandas as pd
    >>> df_target = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6]}, index=pd.date_range(start='2022-01-01', periods=6, freq='M'))
    >>> df_indicator = pd.DataFrame({'value': [7, 12], 'other_col': [10, 20]}, index=pd.date_range(start='2022-01-01', periods=2, freq='Q'))
    >>> adjusted_df = additive_benchmark(df_target, df_indicator)
    >>> print(adjusted_df)
                value  other_col
        2022-01-31   3.5        10
        2022-02-28   4.5        10
        2022-03-31   5.5        10
        2022-04-30   6.5        20
        2022-05-31   7.5        20
        2022-06-30   8.5        20

    """
    # Checking object types.
    if not isinstance(df_indicator, pd.DataFrame):
        raise TypeError("The indicator dataframe is not a pd.DataFrame.")
    else:
        pass
    if not isinstance(df_target, pd.DataFrame):
        raise TypeError("The target dataframe is not a pd.DataFrame.")
    else:
        pass
    if not isinstance(liste_km, list) or isinstance(liste_km, str):
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
    df_indicator_of_concern = df_indicator_of_concern[
        (df_indicator_of_concern.index.year <= endyear)
        & (df_indicator_of_concern.index.year >= startyear)
    ]

    df_target_of_concern = df_target[
        df_target.columns[df_target.columns.isin(liste_km)]
    ]  # Filters out series not sent to chaining.
    df_target_of_concern = df_target_of_concern[
        (df_target_of_concern.index.year <= endyear)
        & (df_target_of_concern.index.year >= startyear)
    ]

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
    if pd.Period(startyear, freq="Y") not in df_indicator_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected start year not in the indicator dataframe.")
    if pd.Period(endyear, freq="Y") not in df_indicator_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected end year not in the indicator dataframe.")
    if pd.Period(startyear, freq="Y") not in df_target_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected start year not in the target dataframe.")
    if pd.Period(endyear, freq="Y") not in df_target_of_concern.index.asfreq("Y"):
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

    return df_indicator_of_concern
# +
import warnings

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
    """TLDR: Adjust values in df_target to match df_indicator with additive quota adjustment.

    Adjust values in df_target to match df_indicator using additive quota adjustment.

    This function adjusts the values in df_target to match the values in df_indicator by first aggregating df_target to the same
    frequency as df_indicator, then calculating the difference between the two, spreading the difference evenly across the
    corresponding periods in df_target, and adding the difference to df_target. The adjusted df_indicator is then returned,
    with non-overlapping columns left untreated.

    Author: Benedikt Goodman, Seksjon for Nasjonalregnskap and Vemund Rundberget, Seksjon for makroøkonomi, Forksningsavdelingen, SSB

    Parameters
    ----------
    df_indicator : pd.DataFrame
        The DataFrame containing the indicator values that will be benchmarked. Must have a period index.
    df_target : pd.DataFrame
        The DataFrame containing the values to benchmark additively against. Must have a period index.
    strict_mode : bool, optional
        If True, raise an error if the time intervals are not of equal length.
        If False, issue a warning instead of raising an error. Default is True.

    Returns:
    -------
    pd.DataFrame
        The adjusted df_indicator with the same frequency as the original df_target, with non-overlapping columns left untreated.

    Raises:
    ------
    ValueError
        If either of the input DataFrames does not have a datetime or period index, or if df_target has a finer frequency
        than df_indicator (i.e. df_indicator has quarterly data and df_target has monthly data)
    KeyError
        If no overlapping column names between df_indicator and df_target exist.
    IndexError
        If strict_mode is True and the time intervals are not of equal length.
    UserWarning
        If strict_mode is False and the time intervals are not of equal length.

    Examples:
    --------
    >>> import pandas as pd
    >>> df_target = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6]}, index=pd.date_range(start='2022-01-01', periods=6, freq='M'))
    >>> df_indicator = pd.DataFrame({'value': [7, 12], 'other_col': [10, 20]}, index=pd.date_range(start='2022-01-01', periods=2, freq='Q'))
    >>> adjusted_df = additive_benchmark(df_target, df_indicator)
    >>> print(adjusted_df)
                value  other_col
        2022-01-31   3.5        10
        2022-02-28   4.5        10
        2022-03-31   5.5        10
        2022-04-30   6.5        20
        2022-05-31   7.5        20
        2022-06-30   8.5        20

    """
    # Checking object types.
    if not isinstance(df_indicator, pd.DataFrame):
        raise TypeError("The indicator dataframe is not a pd.DataFrame.")
    else:
        pass
    if not isinstance(df_target, pd.DataFrame):
        raise TypeError("The target dataframe is not a pd.DataFrame.")
    else:
        pass
    if not isinstance(liste_km, list) or isinstance(liste_km, str):
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
    df_indicator_of_concern = df_indicator_of_concern[
        (df_indicator_of_concern.index.year <= endyear)
        & (df_indicator_of_concern.index.year >= startyear)
    ]

    df_target_of_concern = df_target[
        df_target.columns[df_target.columns.isin(liste_km)]
    ]  # Filters out series not sent to chaining.
    df_target_of_concern = df_target_of_concern[
        (df_target_of_concern.index.year <= endyear)
        & (df_target_of_concern.index.year >= startyear)
    ]

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
    if pd.Period(startyear, freq="Y") not in df_indicator_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected start year not in the indicator dataframe.")
    if pd.Period(endyear, freq="Y") not in df_indicator_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected end year not in the indicator dataframe.")
    if pd.Period(startyear, freq="Y") not in df_target_of_concern.index.asfreq("Y"):
        raise AssertionError("Selected start year not in the target dataframe.")
    if pd.Period(endyear, freq="Y") not in df_target_of_concern.index.asfreq("Y"):
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

    return df_indicator_of_concern
