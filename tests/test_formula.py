from pathlib import Path

import pandas as pd
from pandas import testing as tm

from pre_system.formula import Formula


def test_formula_info(capsys, formulas) -> None:
    formulas.xa.info()
    captured = capsys.readouterr()
    lines = captured.out.split("\n")

    assert lines[0] == "xa = xa1 if year>=2017 else xa2"
    assert lines[1] == " xa1 = xa*<date None>*(x0+x1+x2)/sum((x0+x1+x2)<date None>)"
    assert lines[2] == " xa2 = xa*<date None>*(x0+x1)/sum((x0+x1)<date None>)"

    formulas.vxa.info()
    captured = capsys.readouterr()
    lines = captured.out.split("\n")
    assert (
        lines[0]
        == "vxa = sum(xa<date None>)*(xa/(w0*p0+w1*p1+w2*p2))/sum(xa/(w0*p0+w1*p1+w2*p2)<date None>)"
    )
    assert lines[1] == " xa = xa1 if year>=2017 else xa2"
    assert lines[2] == "  xa1 = xa*<date None>*(x0+x1+x2)/sum((x0+x1+x2)<date None>)"
    assert lines[3] == "  xa2 = xa*<date None>*(x0+x1)/sum((x0+x1)<date None>)"


def test_formula_print(capsys, formulas) -> None:
    print(formulas.xa)
    print(formulas.xb)
    print(formulas.vxa)
    print(formulas.x)
    captured = capsys.readouterr()
    lines = captured.out.split("\n")
    assert lines[0] == "Formula: xa = xa1 if year>=2017 else xa2"
    assert lines[1] == "Formula: xb = xb*<date None>*(x3+x4)/sum((x3+x4)<date None>)"
    assert lines[2] == (
        "Formula: vxa = sum(xa<date None>)*(xa/(w0*p0+w1*p1+w2*p2))/"
        "sum(xa/(w0*p0+w1*p1+w2*p2)<date None>)"
    )
    assert lines[3] == "Formula: xy = xa+xb"


def test_formula_indicator_weights(formulas) -> None:
    result_with_trace = formulas.vxa.indicators_weights()
    facit_with_trace = [
        ("p0", "w0"),
        ("p1", "w1"),
        ("p2", "w2"),
        ("x0", 1),
        ("x1", 1),
        ("x2", 1),
        ("x0", 1),
        ("x1", 1),
    ]
    assert str(result_with_trace) == str(facit_with_trace)

    result_without_trace = formulas.vxa.indicators_weights(trace=False)
    facit_without_trace = [("p0", "w0"), ("p1", "w1"), ("p2", "w2")]
    assert str(result_without_trace) == str(facit_without_trace)


def test_formula_evaluate(formulas, annual_df, indicator_df, weight_df) -> None:
    Formula.baseyear = 2020  # TODO: Understand why not formulas.x.baseyear = 2020
    result_x = formulas.x.evaluate(annual_df, indicator_df, weight_df)
    result_vx = formulas.vx.evaluate(annual_df, indicator_df, weight_df)

    write_new_facit_file = False
    file_x = Path(__file__).parent / "testdata" / "facit_formula_evaluate_x.parquet"
    file_vx = Path(__file__).parent / "testdata" / "facit_formula_evaluate_vx.parquet"
    if write_new_facit_file:
        result_df_x = result_x.to_frame()
        result_df_x.to_parquet(file_x)
        result_df_vx = result_vx.to_frame()
        result_df_vx.to_parquet(file_vx)

    facit_df_x = pd.read_parquet(file_x)
    facit_x = facit_df_x.iloc[:, 0]
    tm.assert_series_equal(result_x, facit_x, check_names=False)

    facit_df_vx = pd.read_parquet(file_vx)
    facit_vx = facit_df_vx.iloc[:, 0]
    tm.assert_series_equal(result_vx, facit_vx, check_names=False)
