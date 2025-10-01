def additive_benchmark(
    df_indicator: pd.DataFrame, target_df: pd.DataFrame, strict_mode=True
) -> pd.DataFrame:
    """
    TLDR: Adjust values in target_df to match df_indicator with additive quota adjustment.

    Adjust values in target_df to match df_indicator using additive quota adjustment.

    This function adjusts the values in target_df to match the values in df_indicator by first aggregating target_df to the same
    frequency as df_indicator, then calculating the difference between the two, spreading the difference evenly across the
    corresponding periods in target_df, and adding the difference to target_df. The adjusted df_indicator is then returned,
    with non-overlapping columns left untreated.

    Author: Benedikt Goodman, Seksjon for Nasjonalregnskap

    Parameters
    ----------
    df_indicator : pd.DataFrame
        The DataFrame containing the indicator values that will be benchmarked. Must have a period index.
    target_df : pd.DataFrame
        The DataFrame containing the values to benchmark additively against. Must have a period index.
    strict_mode : bool, optional
        If True, raise an error if the time intervals are not of equal length.
        If False, issue a warning instead of raising an error. Default is True.

    Returns
    -------
    pd.DataFrame
        The adjusted df_indicator with the same frequency as the original target_df, with non-overlapping columns left untreated.

    Raises
    ------
    ValueError
        If either of the input DataFrames does not have a datetime or period index, or if target_df has a finer frequency
        than df_indicator (i.e. df_indicator has quarterly data and target_df has monthly data)
    KeyError
        If no overlapping column names between df_indicator and target_df exist.
    IndexError
        If strict_mode is True and the time intervals are not of equal length.
    UserWarning
        If strict_mode is False and the time intervals are not of equal length.

    Examples
    --------
    >>> import pandas as pd
    >>> target_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6]}, index=pd.date_range(start='2022-01-01', periods=6, freq='M'))
    >>> df_indicator = pd.DataFrame({'value': [7, 12], 'other_col': [10, 20]}, index=pd.date_range(start='2022-01-01', periods=2, freq='Q'))
    >>> adjusted_df = additive_benchmark(target_df, df_indicator)
    >>> print(adjusted_df)
                value  other_col
        2022-01-31   3.5        10
        2022-02-28   4.5        10
        2022-03-31   5.5        10
        2022-04-30   6.5        20
        2022-05-31   7.5        20
        2022-06-30   8.5        20

    """
    # Validate input DataFrame indexes
    if not (
        isinstance(df_indicator.index, pd.PeriodIndex)
        and isinstance(target_df.index, pd.PeriodIndex)
    ):
        raise ValueError("Both DataFrames must have a datetime or a period index.")

    # Raise issue if for example df_indicator has year index while df_target has months
    if get_duration(df_indicator.index) >= get_duration(target_df.index):
        raise ValueError(
            "Period index in df_indicator must be of finer frequency than df_target."
        )

    # Checks if the input dataframes cover the same time-intervall.
    # Raises error if intervals are note the same and strict_mode is True
    check_time_interval(df_indicator, target_df, strict_mode=strict_mode)


    # Find overlapping columns
    overlapping_cols = df_indicator.columns.intersection(target_df.columns)
    if overlapping_cols.empty:
        raise KeyError(
            "There are no overlapping columns between the two input dataframes."
        )

    # Aggregate target_df to the frequency of df_indicator
    df_indicator_agg = (
        df_indicator[overlapping_cols].resample(target_df.index.freqstr).sum()
    )

    # Calculates amount of months in a quarter or year and so forth
    num_periods = calculate_subperiods(
        df_indicator.index.freqstr, target_df.index.freqstr
    )

    # Calculate the difference between df_indicator and the aggregated target_df
    diff = (target_df[overlapping_cols] - df_indicator_agg) / num_periods

    # Disaggregate diff
    diff = diff.resample(df_indicator.index.freqstr).ffill()

    # Add difference
    df_indicator[overlapping_cols] = df_indicator[overlapping_cols] + diff

    return df_indicator
