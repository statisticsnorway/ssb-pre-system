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
#     display_name: ssb-pre-system
#     language: python
#     name: ssb-pre-system
# ---

# %%
import pandas as pd
import pytest

from pre_system.src.pre_system.multiplicative_benchmark import multiplicative_benchmark


def test_multiplicative_benchmark_happy_path():
    # Monthly indicator (high frequency)
    idx_monthly = pd.period_range("2020-01", "2020-12", freq="M")
    df_indicator = pd.DataFrame({"A": [1.0] * 12}, index=idx_monthly)

    # Yearly target (low frequency)
    idx_yearly = pd.period_range("2020", "2020", freq="Y")
    df_target = pd.DataFrame({"A": [24.0]}, index=idx_yearly)

    result = multiplicative_benchmark(df_indicator, df_target, ["A"], 2020, 2020)

    # Should still be monthly
    assert result.index.freqstr == "M"
    # Aggregated value must equal target
    assert result["A"].sum() == pytest.approx(24.0)


def test_multiplicative_benchmark_invalid_index_type():
    # Indicator with DatetimeIndex instead of PeriodIndex
    idx_monthly = pd.date_range("2020-01-01", periods=12, freq="M")
    df_indicator = pd.DataFrame({"A": range(12)}, index=idx_monthly)

    idx_yearly = pd.period_range("2020", "2020", freq="Y")
    df_target = pd.DataFrame({"A": [66]}, index=idx_yearly)

    with pytest.raises(TypeError, match="Index must be a pd.PeriodIndex"):
        multiplicative_benchmark(df_indicator, df_target, ["A"], 2020, 2020)


def test_multiplicative_benchmark_startyear_not_in_data():
    idx_monthly = pd.period_range("2020-01", "2020-12", freq="M")
    df_indicator = pd.DataFrame({"A": [1.0] * 12}, index=idx_monthly)

    idx_yearly = pd.period_range("2020", "2020", freq="Y")
    df_target = pd.DataFrame({"A": [12.0]}, index=idx_yearly)

    with pytest.raises(AssertionError, match="start year not in the indicator dataframe"):
        multiplicative_benchmark(df_indicator, df_target, ["A"], 2019, 2020)
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
#     display_name: ssb-pre-system
#     language: python
#     name: ssb-pre-system
# ---

# %%
import pandas as pd
import pytest

from pre_system.src.pre_system.multiplicative_benchmark import multiplicative_benchmark


def test_multiplicative_benchmark_happy_path():
    # Monthly indicator (high frequency)
    idx_monthly = pd.period_range("2020-01", "2020-12", freq="M")
    df_indicator = pd.DataFrame({"A": [1.0] * 12}, index=idx_monthly)

    # Yearly target (low frequency)
    idx_yearly = pd.period_range("2020", "2020", freq="Y")
    df_target = pd.DataFrame({"A": [24.0]}, index=idx_yearly)

    result = multiplicative_benchmark(df_indicator, df_target, ["A"], 2020, 2020)

    # Should still be monthly
    assert result.index.freqstr == "M"
    # Aggregated value must equal target
    assert result["A"].sum() == pytest.approx(24.0)


def test_multiplicative_benchmark_invalid_index_type():
    # Indicator with DatetimeIndex instead of PeriodIndex
    idx_monthly = pd.date_range("2020-01-01", periods=12, freq="M")
    df_indicator = pd.DataFrame({"A": range(12)}, index=idx_monthly)

    idx_yearly = pd.period_range("2020", "2020", freq="Y")
    df_target = pd.DataFrame({"A": [66]}, index=idx_yearly)

    with pytest.raises(TypeError, match="Index must be a pd.PeriodIndex"):
        multiplicative_benchmark(df_indicator, df_target, ["A"], 2020, 2020)


def test_multiplicative_benchmark_startyear_not_in_data():
    idx_monthly = pd.period_range("2020-01", "2020-12", freq="M")
    df_indicator = pd.DataFrame({"A": [1.0] * 12}, index=idx_monthly)

    idx_yearly = pd.period_range("2020", "2020", freq="Y")
    df_target = pd.DataFrame({"A": [12.0]}, index=idx_yearly)

    with pytest.raises(AssertionError, match="start year not in the indicator dataframe"):
        multiplicative_benchmark(df_indicator, df_target, ["A"], 2019, 2020)
