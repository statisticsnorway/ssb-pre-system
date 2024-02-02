##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

import pandas as pd

def overlay(*dfs):
    """
    Combines multiple Pandas DataFrames or Series by overlaying their values based on index alignment.

    Parameters:
    -----------
    *dfs : pandas.DataFrame or pandas.Series
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

    if all(isinstance(x, pd.DataFrame) for x in dfs):
        output = pd.DataFrame(dtype=float)
    elif all(isinstance(x, pd.Series) for x in dfs):
        output = pd.Series(dtype=float)
    else:
        raise TypeError('input must be all DataFrames or all Series')

    if all(isinstance(x.index, pd.PeriodIndex) for x in dfs) is False:
        raise AttributeError('all DataFrames/Series must have have Pandas.PeriodIndex')

    if len(dfs) == 1:
        return dfs[0]

    if all(x.index.freq==y.index.freq for x, y in zip(dfs[:-1], dfs[1:])) is False:
        raise AttributeError('all DataFrames/Series must have same freq')

    for df in dfs:
        output = output.combine_first(df)

    return output
