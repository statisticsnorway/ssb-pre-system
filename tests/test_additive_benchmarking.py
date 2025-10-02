# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import pytest

from pre_system.src.pre_system.additive_benchmark import additive_benchmark


def test_basic_functionality():
    # Monthly target
    df_target = pd.DataFrame(
        {"value": [1, 2, 3, 4, 5, 6]},
        index=pd.period_range("2022-01", periods=6, freq="M"),
    )
    # Quarterly indicator
    df_indicator = pd.DataFrame(
        {"value": [12, 15]},
        index=pd.period_range("2022Q1", periods=2, freq="Q"),
    )

    result = additive_benchmark(df_indicator, df_target, ["value"], 2022, 2022)

    # Check that index is still monthly
    assert result.index.freqstr == "M"
    # Check number of rows
    assert len(result) == 6
    # Check adjustment preserved total sums
    assert round(result["value"].sum(), 6) == df_target["value"].sum()


def test_invalid_index_type():
    # Using DatetimeIndex instead of PeriodIndex
    df_target = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=pd.date_range("2022-01-01", periods=3, freq="M"),
    )
    df_indicator = pd.DataFrame(
        {"value": [6]},
        index=pd.period_range("2022Q1", periods=1, freq="Q"),
    )

    with pytest.raises(TypeError, match="Index must be a pd.PeriodIndex"):
        additive_benchmark(df_indicator, df_target, ["value"], 2022, 2022)


def test_missing_column_in_target():
    df_target = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.period_range("2022-01", periods=3, freq="M"),
    )
    df_indicator = pd.DataFrame(
        {"value": [6]},
        index=pd.period_range("2022Q1", periods=1, freq="Q"),
    )

    with pytest.raises(TypeError, match="missing in the target dataframe"):
        additive_benchmark(df_indicator, df_target, ["value"], 2022, 2022)
