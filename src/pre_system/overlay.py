##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

from __future__ import annotations

import pandas as pd


def overlay(*dfs: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Combines multiple Pandas DataFrames or Series by overlaying their values based on index alignment.

    Parameters:
    -----------
    ``*dfs`` : pandas.DataFrame or pandas.Series
        Multiple DataFrames or Series to be combined.

    Returns:
    --------
    pandas.DataFrame or pandas.Series
    Combined DataFrame or Series with overlaid values.

    Raises:
    -------
    TypeError
        If the input is a mixture of DataFrames and Series.
    AttributeError
        If not all DataFrames/Series have Pandas.PeriodIndex or if they don't share the same frequency.

    Notes:
    ------
    This function overlays values from multiple DataFrames or Series, aligning them based on their indices.
    It creates a new DataFrame or Series by combining the input objects.
    The index of the returned object is based on the union of indices from all input DataFrames or Series.
    """
    if len(dfs) == 1:
        return dfs[0]

    if not all(isinstance(x.index, pd.PeriodIndex) for x in dfs):
        raise AttributeError("all DataFrames/Series must have have Pandas.PeriodIndex")

    if not all(x.index.freq == y.index.freq for x, y in zip(dfs[:-1], dfs[1:])):  # type: ignore
        raise AttributeError("all DataFrames/Series must have same freq")

    if all(isinstance(x, pd.DataFrame) for x in dfs):
        return _overlay_dataframe(dfs)  # type: ignore[arg-type]
    elif all(isinstance(x, pd.Series) for x in dfs):
        return _overlay_series(dfs)  # type: ignore[arg-type]
    else:
        raise TypeError("input must be all DataFrames or all Series")


def _overlay_dataframe(dfs: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
    output = pd.DataFrame(dtype=float)
    for df in dfs:
        output = output.combine_first(df)
    return output


def _overlay_series(dfs: tuple[pd.Series, ...]) -> pd.Series:
    output = pd.Series(dtype=float)
    for df in dfs:
        output = output.combine_first(df)
    return output
