import numpy as np
import pandas as pd
import pytest

from pre_system.mind4 import mind4


def _mk_period_index(start, end, freq):
    # Inclusive endpoints
    return pd.period_range(start=start, end=end, freq=freq)


def test_mind4_monthly_happy_path_yearly_sums_match():
    # Build monthly data for 2 years with simple patterns
    idx_m = _mk_period_index("2020-01", "2021-12", "M")
    # Two series: a constant one and a ramp one
    mnr = pd.DataFrame(
        {
            "A": np.r_[np.repeat(10.0, 12), np.repeat(20.0, 12)],
            "B": np.r_[np.arange(1, 13, dtype=float), np.arange(1, 13, dtype=float)],
        },
        index=idx_m,
    )

    # Annual totals equal to the sum of monthly raw values
    idx_y = _mk_period_index("2020", "2021", "Y")
    rea = pd.DataFrame(
        {
            "A": [
                mnr.loc[mnr.index.year == 2020, "A"].sum(),
                mnr.loc[mnr.index.year == 2021, "A"].sum(),
            ],
            "B": [
                mnr.loc[mnr.index.year == 2020, "B"].sum(),
                mnr.loc[mnr.index.year == 2021, "B"].sum(),
            ],
        },
        index=idx_y,
    )

    res = mind4(
        mnr=mnr, rea=rea, liste_d4=["A", "B"], basisaar=2021, startaar=2020, freq="M"
    )

    # Index and columns preserved
    assert isinstance(res.index, pd.PeriodIndex)
    assert res.index.equals(idx_m)
    assert list(res.columns) == ["A", "B"]

    # Yearly sums of result equal to the annual targets exactly (or within tight tolerance)
    res_y = res.resample("Y").sum()
    pd.testing.assert_index_equal(res_y.index, idx_y)
    pd.testing.assert_frame_equal(
        res_y.sort_index(), rea.sort_index(), check_dtype=False, rtol=1e-10, atol=1e-8
    )


def test_mind4_quarterly_happy_path_yearly_sums_match():
    # Quarterly variant
    idx_q = _mk_period_index("2020Q1", "2021Q4", "Q")
    mnr_q = pd.DataFrame(
        {
            "A": np.r_[np.repeat(30.0, 4), np.repeat(40.0, 4)],
        },
        index=idx_q,
    )

    idx_y = _mk_period_index("2020", "2021", "Y")
    rea = pd.DataFrame(
        {
            "A": [
                mnr_q.loc[mnr_q.index.year == 2020, "A"].sum(),
                mnr_q.loc[mnr_q.index.year == 2021, "A"].sum(),
            ],
        },
        index=idx_y,
    )

    res = mind4(
        mnr=mnr_q, rea=rea, liste_d4=["A"], basisaar=2021, startaar=2020, freq="Q"
    )
    res_y = res.resample("Y").sum()
    pd.testing.assert_frame_equal(
        res_y.sort_index(), rea.sort_index(), check_dtype=False
    )


def test_mind4_input_validation_errors():
    idx_m = _mk_period_index("2020-01", "2020-12", "M")
    mnr = pd.DataFrame({"A": np.repeat(1.0, 12)}, index=idx_m)
    idx_y = _mk_period_index("2020", "2020", "Y")
    rea = pd.DataFrame({"A": [12.0]}, index=idx_y)

    with pytest.raises(TypeError):
        mind4(
            mnr=mnr.reset_index(), rea=rea, liste_d4=["A"], basisaar=2020, startaar=2020
        )
    with pytest.raises(TypeError):
        bad_index_df = mnr.copy()
        bad_index_df.index = pd.Index(range(len(bad_index_df)))
        mind4(mnr=bad_index_df, rea=rea, liste_d4=["A"], basisaar=2020, startaar=2020)
    with pytest.raises(TypeError):
        mind4(mnr=mnr, rea=rea, liste_d4=["A"], basisaar=2051, startaar=2020)
    with pytest.raises(TypeError):
        mind4(mnr=mnr, rea=rea, liste_d4=["A"], basisaar=2019, startaar=2020)
    with pytest.raises(TypeError):
        mind4(
            mnr=mnr,
            rea=rea[["A"]].rename(columns={"A": "B"}),
            liste_d4=["A"],
            basisaar=2020,
            startaar=2020,
        )
    with pytest.raises(TypeError):
        mind4(
            mnr=mnr[["A"]].rename(columns={"A": "B"}),
            rea=rea,
            liste_d4=["A"],
            basisaar=2020,
            startaar=2020,
        )
    with pytest.raises(TypeError):
        mind4(mnr=mnr, rea=rea, liste_d4=["A"], basisaar=2020, startaar=2020, freq="W")


def test_mind4_emits_warnings_and_skips_non_numeric():
    idx_m = _mk_period_index("2020-01", "2020-12", "M")
    mnr = pd.DataFrame(
        {
            "A": np.repeat(1.0, 12),
            "B": np.repeat(0.0, 12),  # all zeros -> warning
            "C": [np.nan] * 12,  # NaNs -> warning
            "D": ["x"] * 12,  # non-numeric -> warning and skipped
        },
        index=idx_m,
    )

    idx_y = _mk_period_index("2020", "2020", "Y")
    rea = pd.DataFrame(
        {
            "A": [12.0],
            "B": [0.0],
            "C": [0.0],
            "D": [0.0],
        },
        index=idx_y,
    )

    with pytest.warns(UserWarning):
        res = mind4(
            mnr=mnr,
            rea=rea,
            liste_d4=["A", "B", "C", "D"],
            basisaar=2020,
            startaar=2020,
            freq="M",
        )

    # Non-numeric 'D' should be dropped from the results
    assert "D" not in res.columns
    # Still produce a result for A and possibly B/C if solvable; check yearly sums for A
    res_y = res.resample("Y").sum()
    if "A" in res.columns:
        assert (
            pytest.approx(res_y.loc[idx_y[0], "A"], rel=1e-10, abs=1e-8)
            == rea.loc[idx_y[0], "A"]
        )
