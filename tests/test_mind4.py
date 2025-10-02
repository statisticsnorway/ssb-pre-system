import numpy as np
import pandas as pd
import pytest

from pre_system.mind4 import mind4


def _monthly_df(
    start: str, periods: int, value: float = 1.0, col: str = "x"
) -> pd.DataFrame:
    idx = pd.period_range(start=start, periods=periods, freq="M")
    return pd.DataFrame({col: np.full(len(idx), value, dtype=float)}, index=idx)


def _quarterly_df(
    start: str, periods: int, value: float = 1.0, col: str = "x"
) -> pd.DataFrame:
    idx = pd.period_range(start=start, periods=periods, freq="Q")
    return pd.DataFrame({col: np.full(len(idx), value, dtype=float)}, index=idx)


def _annual_df(
    start_year: int, years: int, total_per_year: float = 12.0, col: str = "x"
) -> pd.DataFrame:
    idx = pd.period_range(start=str(start_year), periods=years, freq="Y")
    return pd.DataFrame(
        {col: np.full(len(idx), total_per_year, dtype=float)}, index=idx
    )


def test_list_d4_accepts_string_and_runs() -> None:
    mnr = _monthly_df("2019-01", 12, 1.0, "x")
    rea = _annual_df(2019, 1, 12.0, "x")

    result = mind4(mnr, rea, "x", basisaar=2019, startaar=2019, freq="M")

    assert list(result.columns) == ["x"]
    assert isinstance(result.index, pd.PeriodIndex) and result.index.freqstr == "M"
    assert len(result) == 12
    pd.testing.assert_frame_equal(result.resample("Y").sum(), rea)


def test_mnr_index_must_be_periodindex() -> None:
    # Build DataFrame with DatetimeIndex to trigger the index type check
    idx = pd.date_range(start="2019-01-01", periods=12, freq="MS")
    mnr = pd.DataFrame({"x": np.ones(len(idx))}, index=idx)
    rea = _annual_df(2019, 1, 12.0, "x")
    with pytest.raises(
        TypeError, match=r"monthly dataframe does not have a pd\.PeriodIndex"
    ):
        mind4(mnr, rea, ["x"], basisaar=2019, startaar=2019, freq="M")


def test_missing_columns_in_monthly_dataframe_raises() -> None:
    mnr = _monthly_df("2019-01", 12, 1.0, "y")  # column 'x' is missing
    rea = _annual_df(2019, 1, 12.0, "x")
    with pytest.raises(
        TypeError, match=r"\['x'\] are missing in the monthly dataframe."
    ):
        mind4(mnr, rea, ["x"], basisaar=2019, startaar=2019, freq="M")


def test_year_overlap_required_raises() -> None:
    # mnr covers 2019, rea covers 2020 only; expect a mismatch error
    mnr = _monthly_df("2019-01", 12, 1.0, "x")
    rea = _annual_df(2020, 1, 12.0, "x")
    with pytest.raises(
        TypeError, match=r"There aren't values in both series for \{2019\}"
    ):
        mind4(mnr, rea, ["x"], basisaar=2020, startaar=2019, freq="M")


def test_monthly_happy_path_preserves_annual_totals_and_shape() -> None:
    # Two full years monthly with constant values summing to annual totals
    mnr = _monthly_df("2019-01", 24, 1.0, "x")  # monthly ones
    rea = _annual_df(2019, 2, 12.0, "x")  # annual total = 12 each year

    result = mind4(mnr, rea, ["x"], basisaar=2020, startaar=2019, freq="M")

    # Shape and index
    assert list(result.columns) == ["x"]
    assert isinstance(result.index, pd.PeriodIndex) and result.index.freqstr == "M"
    assert len(result) == 24

    # Totals should match the annual constraints exactly for this simple case
    annual_totals = result.resample("Y").sum()
    pd.testing.assert_frame_equal(annual_totals, rea)


def test_quarterly_happy_path_preserves_annual_totals_and_shape() -> None:
    # Two full years quarterly with constant values summing to annual totals
    mnr = _quarterly_df("2019Q1", 8, 1.0, "x")  # quarterly ones
    rea = _annual_df(2019, 2, 4.0, "x")  # annual total = 4 each year

    result = mind4(mnr, rea, ["x"], basisaar=2020, startaar=2019, freq="Q")

    # Shape and index
    assert list(result.columns) == ["x"]
    assert isinstance(result.index, pd.PeriodIndex) and result.index.freqstr == "Q-DEC"
    assert len(result) == 8

    # Totals should match the annual constraints
    annual_totals = result.resample("Y").sum()
    pd.testing.assert_frame_equal(annual_totals, rea)


def test_warns_on_nan_in_monthly_input() -> None:
    mnr = _monthly_df("2019-01", 12, 1.0, "x")
    mnr.iloc[5, 0] = np.nan  # Introduce NaN
    rea = _annual_df(2019, 1, 12.0, "x")

    with pytest.warns(UserWarning, match="There are NaN-values.*monthly dataframe"):
        _ = mind4(mnr, rea, ["x"], basisaar=2019, startaar=2019, freq="M")
