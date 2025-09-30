# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import warnings

import numpy as np
import pandas as pd


def chain_df(
    val_df: pd.DataFrame, 
    fp_df: pd.DataFrame, 
    serieslist: list[str] | str=None, 
    baseyear: int=None, 
    startyear: int=None, 
    endyear: int=None, 
    appendvlname: bool = False
):
    # Checking list object type.
    if not isinstance(serieslist, list) or isinstance(serieslist, str):
        raise TypeError("You need to create a list of all the series you wish to chain, and it must be in the form of a list or string.")
    if isinstance(serieslist, str):
        serieslist = [serieslist]
    # Checking dfs object type.
    if not isinstance(val_df, pd.DataFrame):
        raise TypeError("The value dataframe is not a pd.DataFrame.")
    if not isinstance(fp_df, pd.DataFrame):
        raise TypeError("The fixed price dataframe is not a pd.DataFrame.")

    # Checking index.
    if not isinstance(val_df.index, pd.PeriodIndex):
        raise TypeError('Index must be a pd.PeriodIndex in the current prices DataFrame.')
    if not isinstance(fp_df.index, pd.PeriodIndex):
        raise TypeError('Index must be a pd.PeriodIndex in the fixed prices DataFrame.') 
    # Checking columns.
    if not pd.Series(serieslist).isin(val_df.columns).all():
        raise TypeError(
            f"{np.setdiff1d(serieslist, val_df.columns).tolist()} are missing in the value dataframe."
        )
    if not pd.Series(serieslist).isin(fp_df.columns).all():
        raise TypeError(
            f"{np.setdiff1d(serieslist, fp_df.columns).tolist()} are missing in the fixed price dataframe."
        )

    if startyear is None:
        startyear = val_df.index[0].year
    if endyear is None:
        endyear   = val_df.index[-1].year
    if startyear > endyear:
        raise TypeError("The start year cannot be greater than the end year.")

    # Filters out series not sent to chaining.
    val_df_of_concern = val_df[
        val_df.columns[val_df.columns.isin(serieslist)]
    ]  
    val_df_of_concern = val_df_of_concern[
        (val_df_of_concern.index.year <= endyear)
        & (val_df_of_concern.index.year >= startyear)
    ]
    
    fp_df_of_concern = fp_df[
        fp_df.columns[fp_df.columns.isin(serieslist)]
    ]  # Filters out series not sent to chaining.
    fp_df_of_concern = fp_df_of_concern[
        (fp_df_of_concern.index.year <= endyear)
        & (fp_df_of_concern.index.year >= startyear)
    ]
    

    # Value checks for val_df.
    valzerowarnlist = [] # Zeroes checks.
    for col in val_df_of_concern.columns:
        if (val_df_of_concern[col] == 0).all():
            valzerowarnlist.append(f"{col}")
    if len(valzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {valzerowarnlist} in the value dataframe.",
            UserWarning,
            stacklevel=2,
        )
    valintwarnlist = [] # Non-int check.
    for col in val_df_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(val_df_of_concern[col])
            and col not in val_df_of_concern.columns[val_df_of_concern.isna().any()].to_list()
        ):
            valintwarnlist.append(f"{col}")
    if len(valintwarnlist) > 0:
        valintwarnlist.warn(
            f"There are values in {valintwarnlist} in the value dataframe that are not real numbers. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )
    if val_df_of_concern.isna().any().any() is np.True_: # NaN check.
        warnings.warn(
            f"There are NaN-values in {val_df_of_concern.columns[val_df_of_concern.isna().any()].to_list()} in the value dataframe.",
            UserWarning,
            stacklevel=2,)
    
    # Value checks for fp_df.
    fpzerowarnlist = [] # Zeroes checks.
    for col in fp_df_of_concern.columns:
        if (fp_df_of_concern[col] == 0).all():
            fpzerowarnlist.append(f"{col}")
    if len(fpzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {fpzerowarnlist} in the fixed price dataframe.",
            UserWarning,
            stacklevel=2,
        )

    fpintwarnlist = [] # Non-int check.
    for col in fp_df_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(fp_df_of_concern[col])
            and col not in fp_df_of_concern.columns[fp_df_of_concern.isna().any()].to_list()
        ):
            fpintwarnlist.append(f"{col}")
    if len(fpintwarnlist) > 0:
        fpintwarnlist.warn(
            f"There are values in {fpintwarnlist} in the fixed price dataframe that are not real numbers. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )
    if fp_df_of_concern.isna().any().any() is np.True_: # NaN check.
        warnings.warn(
            f"There are NaN-values in {fp_df_of_concern.columns[fp_df_of_concern.isna().any()].to_list()} in the fixed price dataframe.",
            UserWarning,
            stacklevel=2,)

    # valintwarnlist
    val_df_of_concern = val_df_of_concern[
        [col for col in val_df_of_concern.columns if col not in valintwarnlist]
    ]
    fp_df_of_concern = fp_df_of_concern[
        [col for col in fp_df_of_concern.columns if col not in valintwarnlist]
    ]
    # mnrintwarnlist
    val_df_of_concern = val_df_of_concern[
        [col for col in val_df_of_concern.columns if col not in fpintwarnlist]
    ]
    fp_df_of_concern = fp_df_of_concern[
        [col for col in fp_df_of_concern.columns if col not in fpintwarnlist]
    ]
    serieslist = [
        serie
        for serie in serieslist
        if serie not in valintwarnlist and serie not in fpintwarnlist
    ]

    # Checking that start and end years are in range.
    if pd.Period(startyear, freq="Y") not in val_df_of_concern.index:
        raise AssertionError('Selected start year not in the value dataframe.')
    if pd.Period(endyear, freq="Y") not in val_df_of_concern.index:
        raise AssertionError('Selected end year not in the value dataframe.')        
    if pd.Period(startyear, freq="Y") not in fp_df_of_concern.index:
        raise AssertionError('Selected start year not in the fixed price dataframe.')
    if pd.Period(endyear, freq="Y") not in fp_df_of_concern.index:
        raise AssertionError('Selected end year not in the value dataframe.')

    # Checking base year is in year range.
    if not baseyear <= endyear or not baseyear > startyear:
        raise AssertionError('Base year is not between end year and start year.')
    if pd.Period(baseyear, freq="Y") not in val_df_of_concern.index:
        raise AssertionError('Selected base year not in the value dataframe.')
    if pd.Period(baseyear, freq="Y") not in fp_df_of_concern.index:
        raise AssertionError('Selected base year not in the fixed price dataframe.')
        
    if not isinstance(startyear, int):
        raise TypeError("The start year must be an integer.")
    if not isinstance(endyear, int):
        raise TypeError("The end year must be an integer.")
    if endyear > 2099:
        raise TypeError(
            "The final year must be less than 2099. Are you sure you entered it correctly?."
        )
    if not isinstance(baseyear, int):
        raise TypeError("The base year must be an integer.")
    if baseyear > 2099:
        raise TypeError(
            "The final year must be less than 2099. Are you sure you entered it correctly?."
        )

    # ratio
    ratio_df = (fp_df_of_concern / val_df_of_concern.shift(1)).fillna(1)
    
    # accumulated ratio
    cum_product_df = ratio_df.cumprod()

    # Volume values
    vl_df = (
        cum_product_df*val_df_of_concern.loc[val_df_of_concern.index.year == baseyear].values[0]/
        cum_product_df.loc[val_df_of_concern.index.year == baseyear].values[0]
    )
    
    if appendvlname:
        vl_df.columns = vl_df.columns.astype(str) + ".vl"
    
    return vl_df
