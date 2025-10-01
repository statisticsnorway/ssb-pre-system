# +
import pandas as pd
import numpy as np


def additive_benchmark(
    df_indicator: pd.DataFrame,
    df_target: pd.DataFrame,
    liste_km: list[str] | str,
    startyear: int,
    endyear: int
) -> pd.DataFrame:
    """
    TLDR: Adjust values in df_target to match df_indicator with additive quota adjustment.

    Adjust values in df_target to match df_indicator using additive quota adjustment.

    This function adjusts the values in df_target to match the values in df_indicator by first aggregating df_target to the same
    frequency as df_indicator, then calculating the difference between the two, spreading the difference evenly across the
    corresponding periods in df_target, and adding the difference to df_target. The adjusted df_indicator is then returned,
    with non-overlapping columns left untreated.

    Author: Benedikt Goodman, Seksjon for Nasjonalregnskap

    Parameters
    ----------
    df_indicator : pd.DataFrame
        The DataFrame containing the indicator values that will be benchmarked. Must have a period index.
    df_target : pd.DataFrame
        The DataFrame containing the values to benchmark additively against. Must have a period index.
    strict_mode : bool, optional
        If True, raise an error if the time intervals are not of equal length.
        If False, issue a warning instead of raising an error. Default is True.

    Returns
    -------
    pd.DataFrame
        The adjusted df_indicator with the same frequency as the original df_target, with non-overlapping columns left untreated.

    Raises
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

    Examples
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
    # Aggregate df_target_of_concern to the frequency of df_indicator_of_concern
    df_indicator_agg = (
        df_indicator_of_concern[overlapping_cols].resample(df_target_of_concern.index.freqstr).sum()
    )

    # Calculates amount of months in a quarter or year and so forth
    num_periods = calculate_subperiods(
        df_indicator_of_concern.index.freqstr, df_target_of_concern.index.freqstr
    )

    # Calculate the difference between df_indicator_of_concern and the aggregated df_target_of_concern
    diff = (df_target_of_concern[overlapping_cols] - df_indicator_agg) / num_periods

    # Disaggregate diff
    diff = diff.resample(df_indicator_of_concern.index.freqstr).ffill()

    # Add difference
    df_indicator_of_concern[overlapping_cols] = df_indicator_of_concern[overlapping_cols] + diff

    return df_indicator_of_concern
# -


