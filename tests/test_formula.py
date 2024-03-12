from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from pre_system.formula import FDeflate
from pre_system.formula import FInflate
from pre_system.formula import FJoin
from pre_system.formula import Formula
from pre_system.formula import FSum
from pre_system.formula import Indicator


class Formulas(NamedTuple):
    """Class for storing a list of test formulas."""

    xa1: Indicator
    xa2: Indicator
    xa: FJoin
    xb: Indicator
    vxa: FDeflate
    vxb: FInflate
    x: FSum
    vx: FSum


@pytest.fixture
def formulas() -> Formulas:
    # Let's make a formula that extrapolates x using x1, x2 and x3
    xa1 = Indicator("xa1", "xa", ["x0", "x1", "x2"])
    xa2 = Indicator("xa2", "xa", ["x0", "x1"])
    xa = FJoin("xa", xa1, xa2, 2017)
    xb = Indicator("xb", "xb", ["x3", "x4"])

    # Let's deflate them with a bunch of weighted price indices
    vxa = FDeflate("vxa", xa, ["p0", "p1", "p2"], ["w0", "w1", "w2"])
    vxb = FInflate("vxb", xb, ["p3", "p4"], ["w3", "w4"])

    # Finally, let's sum them up
    x = FSum("xy", xa, xb)
    vx = FSum("vxy", vxa, vxb)

    return Formulas(xa1, xa2, xa, xb, vxa, vxb, x, vx)


@pytest.fixture(scope="session")
def generator():
    return np.random.default_rng(0)


@pytest.fixture
def annual_df(generator) -> pd.DataFrame:
    years = 13
    return pd.DataFrame(
        np.exp(0.02 + generator.normal(0, 0.01, (years, 10)).cumsum(axis=0)),
        columns=[f"x{i}" for i in "abcdefghij"],
        index=pd.period_range(start="2010", periods=years, freq="Y"),
    )


@pytest.fixture
def indicator_df(generator) -> pd.DataFrame:
    years = 13
    return pd.DataFrame(
        np.exp(0.02 + generator.normal(0, 0.01, (years * 12, 10)).cumsum(axis=0)),
        columns=[f"x{i}" for i in range(5)] + [f"p{i}" for i in range(5)],
        index=pd.period_range(start="2010-01", periods=years * 12, freq="M"),
    )


@pytest.fixture
def weight_df(generator) -> pd.DataFrame:
    years = 13
    weight_df = pd.DataFrame(
        10 + generator.normal(0, 1, (years, 5)).cumsum(axis=0),
        columns=[f"w{i}" for i in range(5)],
        index=pd.period_range(start="2010", periods=years, freq="Y"),
    )
    weight_df[["w0", "w1", "w2"]] = weight_df[["w0", "w1", "w2"]].divide(
        weight_df[["w0", "w1", "w2"]].sum(axis=1), axis=0
    )
    weight_df[["w3", "w4"]] = weight_df[["w3", "w4"]].divide(
        weight_df[["w3", "w4"]].sum(axis=1), axis=0
    )
    return weight_df


def test_formula_info(capsys, formulas) -> None:
    formulas.xa.info()
    captured = capsys.readouterr()
    lines = captured.out.split("\n")

    assert lines[0] == "xa = xa1 if year>=2017 else xa2"
    assert lines[1] == " xa1 = xa*<date None>*(x0+x1+x2)/sum((x0+x1+x2)<date None>)"
    assert lines[2] == " xa2 = xa*<date None>*(x0+x1)/sum((x0+x1)<date None>)"


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
    Formula.baseyear = 2020
    result = formulas.x.evaluate(annual_df, indicator_df, weight_df)
    print(result)
