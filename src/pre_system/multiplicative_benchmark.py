# +
import pandas as pd
import numpy as np

def multiplicative_benchmark(
    df_indicator: pd.DataFrame,
    df_target: pd.DataFrame,
    liste_km: list[str] | str,
    startyear: int,
    endyear: int
) -> pd.DataFrame:
    # Checking object types.
    if not isinstance(df_indicator, pd.DataFrame):
        raise TypeError("The indicator dataframe is not a pd.DataFrame.")
    if not isinstance(df_target, pd.DataFrame):
        raise TypeError("The target dataframe is not a pd.DataFrame.")
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
        raise TypeError(
            "Index must be a pd.PeriodIndex in the indicator DataFrame."
        )
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
    df_indicator_of_concern = df_indicator[df_indicator.columns[df_indicator.columns.isin(liste_km)]]
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
            not in df_indicator_of_concern.columns[df_indicator_of_concern.isna().any()].to_list()
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
            f"There are NaN-values in {df_indicator_of_concern.columns[df_indicator_of_concern.isna().any()].to_list()} in the indicator dataframe.",
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
            not in df_target_of_concern.columns[df_target_of_concern.isna().any()].to_list()
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
            f"There are NaN-values in {df_target_of_concern.columns[df_target_of_concern.isna().any()].to_list()} in the target dataframe.",
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
        [col for col in df_indicator_of_concern.columns if col not in indicatorintwarnlist]
    ]
    liste_km = [
        serie
        for serie in liste_km
        if serie not in targetintwarnlist and serie not in indicatorintwarnlist
    ]

    # Logical checks.
    # Checking that start and end years are in range.
    if pd.Period(startyear, freq="Y") not in df_indicator_of_concern.index:
        raise AssertionError("Selected start year not in the indicator dataframe.")
    if pd.Period(endyear, freq="Y") not in df_indicator_of_concern.index:
        raise AssertionError("Selected end year not in the indicator dataframe.")
    if pd.Period(startyear, freq="Y") not in df_target_of_concern.index:
        raise AssertionError("Selected start year not in the target dataframe.")
    if pd.Period(endyear, freq="Y") not in df_target_of_concern.index:
        raise AssertionError("Selected end year not in the target dataframe.")
    
    df_ratio = (
        df_indicator.groupby(
            pd.PeriodIndex(df_indicator.index, freq=df_target.index.freq)
        )
        .sum()
        .div(df_target)
        .resample(df_indicator.index.freq)
        .ffill()
    )
    
    res_df = df_indicator.div(df_ratio.fillna(1).reindex(df_indicator.index))[
        df_indicator.columns
    ]

    return res_df

# +
# def multiplicative_benchmark(df_target: pd.DataFrame, df_indicator: pd.DataFrame):    
#     df_ratio = (
#         df_indicator.groupby(
#             pd.PeriodIndex(df_indicator.index, freq=df_target.index.freq)
#         )
#         .sum()
#         .div(df_target)
#         .resample(df_indicator.index.freq)
#         .ffill()
#     )
    
#     res_df = df_indicator.div(df_ratio.fillna(1).reindex(df_indicator.index))[
#         df_indicator.columns
#     ]

#     return res_df
# -




