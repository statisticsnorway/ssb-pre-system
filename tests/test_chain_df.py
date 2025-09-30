# +
import pandas as pd
import pytest

from pre_system.chaining import chain_df


# +
def make_test_dfs():
    idx = pd.period_range("2015", "2019", freq="Y")
    val_df = pd.DataFrame({"serie1": [1, 2, 3, 4, 5]}, index=idx)
    fp_df  = pd.DataFrame({"serie1": [2, 3, 4, 5, 6]}, index=idx)
    return val_df, fp_df

def test_chain_df_valid_case():
    val_df, fp_df = make_test_dfs()
    result = chain_df(
        val_df=val_df,
        fp_df=fp_df,
        serieslist=["serie1"],
        baseyear=2017,
        startyear=2015,
        endyear=2019,
        appendvlname=True,
    )
    # Should return a DataFrame with same index
    assert isinstance(result, pd.DataFrame)
    assert "serie1.vl" in result.columns
    assert all(result.index == val_df.index)

def test_chain_df_missing_column_raises():
    val_df, fp_df = make_test_dfs()
    with pytest.raises(TypeError) as excinfo:
        chain_df(
            val_df=val_df,
            fp_df=fp_df,
            serieslist=["nonexistent"],
            baseyear=2017,
            startyear=2015,
            endyear=2019,
        )
    assert "missing" in str(excinfo.value).lower()

def test_chain_df_invalid_baseyear_raises():
    val_df, fp_df = make_test_dfs()
    with pytest.raises(AssertionError):
        chain_df(
            val_df=val_df,
            fp_df=fp_df,
            serieslist=["serie1"],
            baseyear=2014,   # outside of index
            startyear=2015,
            endyear=2019,
        )
