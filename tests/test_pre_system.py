from pathlib import Path

import pandas as pd
from pandas import testing as tm
from pytest import CaptureFixture

from pre_system.pre_system import PreSystem
from tests.conftest import Formulas


def _assert_is_none(x: object) -> None:
    assert x is None


def test_formulas(
    formulas: Formulas,
    annual_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> None:
    pre_system = PreSystem("Test PreSystem")
    for formula in formulas:
        pre_system.add_formula(formula)

    assert len(pre_system.formulae) == 8
    _assert_is_none(pre_system.baseyear)
    _assert_is_none(pre_system.annuals_df)
    _assert_is_none(pre_system.indicators_df)
    _assert_is_none(pre_system.weights_df)
    _assert_is_none(pre_system.corrections_df)

    pre_system.baseyear = 2020
    assert pre_system.baseyear is not None

    pre_system.annuals_df = annual_df
    pre_system.indicators_df = indicator_df
    pre_system.weights_df = weight_df
    assert pre_system.annuals_df is not None
    assert pre_system.indicators_df is not None
    assert pre_system.weights_df is not None

    result_df = pre_system.evaluate

    write_new_facit_file = False
    file = Path(__file__).parent / "testdata" / "facit_pre_system_evaluate.parquet"
    if write_new_facit_file:
        result_df.to_parquet(file)

    facit_df = pd.read_parquet(file)
    tm.assert_frame_equal(result_df, facit_df)


def test_info(capsys: CaptureFixture[str], formulas: Formulas) -> None:
    pre_system = PreSystem("Test PreSystem")
    for formula in formulas:
        pre_system.add_formula(formula)
    pre_system.info()

    captured = capsys.readouterr()
    lines = captured.out.split("\n")
    assert lines[2] == "Baseyear is None."
    assert lines[4] == "DataFrames updated:"
    assert lines[5] == "annuals_df       None"
    assert lines[6] == "indicators_df    None"
    assert lines[7] == "weights_df       None (optional)"
    assert lines[8] == "corrections_df   None (optional)"
