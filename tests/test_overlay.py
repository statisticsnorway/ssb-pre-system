import numpy as np
import pandas as pd

from pre_system.overlay import overlay


def test_overlay(quarterly_df: pd.DataFrame) -> None:
    df0 = quarterly_df.copy()
    df1 = quarterly_df.copy()
    df2 = quarterly_df.copy()

    # Lets remove some values
    df0.iloc[0:5, :] = np.nan
    df1.iloc[3:9, :] = np.nan
    df2.iloc[10:12, :] = np.nan

    # If we overlay the three DataFrames, the result should contains no NaN's
    result_df = overlay(df0, df1, df2)
    assert not result_df.isna().any().any()  # type: ignore [union-attr]

    # If we leave out df2 there are still some, since the NaN's in df0 and df1 overlap
    result_df2 = overlay(df0, df1)
    assert result_df2.isna().any().any()  # type: ignore [union-attr]
