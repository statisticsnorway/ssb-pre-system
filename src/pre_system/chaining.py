import warnings

import numpy as np
import pandas as pd


def chain_df(
    val_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    serieslist: list[str] | str | None = None,
    baseyear: int | None = None,
    startyear: int | None = None,
    endyear: int | None = None,
    appendvlname: bool = False,
) -> pd.DataFrame:
    """Chaining economic time series data.

    Processes and validates data for chaining economic time series data, ensuring proper
    formats, types, and constraints across two dataframes. Also, performs warnings for
    data issues like NaN values or missing data columns while preparing time series for
    year-over-year chaining.

    This function validates the input dataframes, extracts overlapping column series based on
    user input or intersection of dataframe columns, and ensures proper filtering to only include
    non-problematic data for all chaining operations. Columns with issues such as
    non-numeric data, NaN values, or zero-only values are warned about, and will be excluded
    from chaining processing. The function also checks for start, end, and base year constraints,
    ensuring valid time ranges for chaining.

    Args:
        val_df: Input dataframe containing current price values data used for chaining.
        fp_df: Input dataframe containing fixed price values data used for chaining.
        serieslist: List of column names, a single string, or None specifying
            the series to chain. Uses the intersection of columns from the dataframes if None.
        baseyear: An integer specifying the base year for chaining operations.
            Must be within valid constraints.
        startyear: Specifies the start year for limiting the chaining range. Computes
            automatically based on data ranges if not provided.
        endyear: Specifies the end year for limiting the chaining range. Computes
            automatically based on data ranges if not provided.
        appendvlname: Whether to append suffix/prefix to chained series. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe containing the chained series for all specified or detected
        valid columns in the input dataframes. Columns with detected issues are excluded from the output.

    Raises:
        TypeError: If invalid types or indices are encountered in inputs, or if year constraints fail.
        AssertionError: If the selected start or end year is not present in either
            the indicator or target DataFrames
    """
    # Checking dfs object type.
    if not isinstance(val_df, pd.DataFrame):
        raise TypeError("The value dataframe is not a pd.DataFrame.")
    if not isinstance(fp_df, pd.DataFrame):
        raise TypeError("The fixed price dataframe is not a pd.DataFrame.")

    # Checking index.
    if not isinstance(val_df.index, pd.PeriodIndex):
        raise TypeError(
            "Index must be a pd.PeriodIndex in the current prices DataFrame."
        )
    if not isinstance(fp_df.index, pd.PeriodIndex):
        raise TypeError("Index must be a pd.PeriodIndex in the fixed prices DataFrame.")

    if (
        startyear is None
    ):  # Sets start and end year as the greatest range possible if not otherwise specified.
        startyear = max(val_df.index.min().year, fp_df.index.min().year)
    if endyear is None:
        endyear = min(val_df.index.max().year, fp_df.index.max().year)
    if startyear > endyear:
        raise TypeError("The start year cannot be greater than the end year.")

    # Checking list object type.
    if serieslist is None:
        serieslist = np.intersect1d(val_df.columns, fp_df.columns).tolist()

    if not isinstance(serieslist, (list, str)):
        raise TypeError(
            "You need to create a list of all the series you wish to chain, and it must be in the form of a list or string."
        )
    if isinstance(serieslist, str):
        serieslist = [serieslist]

    # Checking columns.
    if not pd.Series(serieslist).isin(val_df.columns).all():
        raise TypeError(
            f"{np.setdiff1d(serieslist, val_df.columns).tolist()} are missing in the value dataframe."
        )
    if not pd.Series(serieslist).isin(fp_df.columns).all():
        raise TypeError(
            f"{np.setdiff1d(serieslist, fp_df.columns).tolist()} are missing in the fixed price dataframe."
        )

    # Filters out series not sent to chaining.
    val_df_of_concern = val_df[val_df.columns[val_df.columns.isin(serieslist)]]

    # Ensure index is PeriodIndex for .year
    if not isinstance(val_df_of_concern.index, pd.PeriodIndex):
        raise TypeError("val_df_of_concern index must be a PeriodIndex.")
    val_df_of_concern = val_df_of_concern[
        (val_df_of_concern.index.year <= endyear)
        & (val_df_of_concern.index.year >= startyear)
    ]

    fp_df_of_concern = fp_df[fp_df.columns[fp_df.columns.isin(serieslist)]]

    # Ensure index is PeriodIndex for .year
    if not isinstance(fp_df_of_concern.index, pd.PeriodIndex):
        raise TypeError("val_df_of_concern index must be a PeriodIndex.")
    # Filters out series not sent to chaining.
    fp_df_of_concern = fp_df_of_concern[
        (fp_df_of_concern.index.year <= endyear)
        & (fp_df_of_concern.index.year >= startyear)
    ]

    # Value checks for val_df.
    valzerowarnlist = []  # Zeroes checks.
    for col in val_df_of_concern.columns:
        if (val_df_of_concern[col] == 0).all():
            valzerowarnlist.append(f"{col}")
    if len(valzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {valzerowarnlist} in the value dataframe.",
            UserWarning,
            stacklevel=2,
        )
    valintwarnlist = []  # Non-int check.
    for col in val_df_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(val_df_of_concern[col])
            and col
            not in val_df_of_concern.columns[val_df_of_concern.isna().any()].to_list()
        ):
            valintwarnlist.append(f"{col}")
    if len(valintwarnlist) > 0:
        warnings.warn(
            f"There are values in {valintwarnlist} in the value dataframe that are not real numbers. Skipping sending these to chaining.",
            UserWarning,
            stacklevel=2,
        )
    if val_df_of_concern.isna().any().any() is np.True_:  # NaN check.
        warnings.warn(
            f"There are NaN-values in {val_df_of_concern.columns[val_df_of_concern.isna().any()].to_list()} in the value dataframe. Skipping sending these to chaining.",
            UserWarning,
            stacklevel=2,
        )

    # Value checks for fp_df.
    fpzerowarnlist = []  # Zeroes checks.
    for col in fp_df_of_concern.columns:
        if (fp_df_of_concern[col] == 0).all():
            fpzerowarnlist.append(f"{col}")
    if len(fpzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {fpzerowarnlist} in the fixed price dataframe.",
            UserWarning,
            stacklevel=2,
        )

    fpintwarnlist = []  # Non-int check.
    for col in fp_df_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(fp_df_of_concern[col])
            and col
            not in fp_df_of_concern.columns[fp_df_of_concern.isna().any()].to_list()
        ):
            fpintwarnlist.append(f"{col}")
    if len(fpintwarnlist) > 0:
        warnings.warn(
            f"There are values in {fpintwarnlist} in the fixed price dataframe that are not real numbers. Skipping sending these to chaining.",
            UserWarning,
            stacklevel=2,
        )
    if fp_df_of_concern.isna().any().any() is np.True_:  # NaN check.
        warnings.warn(
            f"There are NaN-values in {fp_df_of_concern.columns[fp_df_of_concern.isna().any()].to_list()} in the fixed price dataframe. Skipping sending these to chaining.",
            UserWarning,
            stacklevel=2,
        )

    # Filters out valintwarnlist
    val_df_of_concern = val_df_of_concern[
        [col for col in val_df_of_concern.columns if col not in valintwarnlist]
    ]
    fp_df_of_concern = fp_df_of_concern[
        [col for col in fp_df_of_concern.columns if col not in valintwarnlist]
    ]
    # Filters out fpintwarnlist
    val_df_of_concern = val_df_of_concern[
        [col for col in val_df_of_concern.columns if col not in fpintwarnlist]
    ]
    fp_df_of_concern = fp_df_of_concern[
        [col for col in fp_df_of_concern.columns if col not in fpintwarnlist]
    ]
    serieslist = [
        serie
        for serie in serieslist
        if serie
        not in (
            valintwarnlist
            + fpintwarnlist
            + fp_df_of_concern.columns[fp_df_of_concern.isna().any()].to_list()
            + val_df_of_concern.columns[val_df_of_concern.isna().any()].to_list()
        )
    ]

    # Check years are not None and are int
    if startyear is None or endyear is None or baseyear is None:
        raise TypeError("startyear, endyear, and baseyear must be provided.")
    if not isinstance(startyear, int):
        raise TypeError("The start year must be an integer.")
    if not isinstance(endyear, int):
        raise TypeError("The end year must be an integer.")
    if not isinstance(baseyear, int):
        raise TypeError("The base year must be an integer.")
    if endyear > 2099:
        raise TypeError(
            "The final year must be less than 2099. Are you sure you entered it correctly?."
        )
    if baseyear > 2099:
        raise TypeError(
            "The final year must be less than 2099. Are you sure you entered it correctly?."
        )

    # Checking that start and end years are in range
    if pd.Period(str(startyear), freq="Y") not in val_df_of_concern.index:
        raise AssertionError("Selected start year not in the value dataframe.")
    if pd.Period(str(endyear), freq="Y") not in val_df_of_concern.index:
        raise AssertionError("Selected end year not in the value dataframe.")
    if pd.Period(str(startyear), freq="Y") not in fp_df_of_concern.index:
        raise AssertionError("Selected start year not in the fixed price dataframe.")
    if pd.Period(str(endyear), freq="Y") not in fp_df_of_concern.index:
        raise AssertionError("Selected end year not in the fixed price dataframe.")
    # Checking base year is in year range.
    if not (baseyear <= endyear and baseyear >= startyear):
        raise AssertionError("Base year is not between end year and start year.")
    if pd.Period(str(baseyear), freq="Y") not in val_df_of_concern.index:
        raise AssertionError("Selected base year not in the value dataframe.")
    if pd.Period(str(baseyear), freq="Y") not in fp_df_of_concern.index:
        raise AssertionError("Selected base year not in the fixed price dataframe.")

    # ratio
    ratio_df = (fp_df_of_concern / val_df_of_concern.shift(1)).fillna(1)

    # accumulated ratio
    cum_product_df = ratio_df.cumprod()

    # Volume values
    if not isinstance(val_df_of_concern.index, pd.PeriodIndex):
        raise TypeError("val_df_of_concern index must be a PeriodIndex.")
    vl_df: pd.DataFrame = (
        cum_product_df
        * val_df_of_concern.loc[val_df_of_concern.index.year == baseyear].values[0]
        / cum_product_df.loc[val_df_of_concern.index.year == baseyear].values[0]
    )

    if appendvlname:
        vl_df.columns = vl_df.columns.astype(str) + ".vl"

    return vl_df
