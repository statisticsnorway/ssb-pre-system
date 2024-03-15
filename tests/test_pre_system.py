from pre_system.pre_system import PreSystem


def test_formulas(formulas, annual_df, indicator_df, weight_df) -> None:
    pre_system = PreSystem("Test PreSystem")
    for formula in formulas:
        pre_system.add_formula(formula)

    assert len(pre_system.formulae) == 8
    assert pre_system.baseyear is None
    assert pre_system.annuals_df is None
    assert pre_system.indicators_df is None
    assert pre_system.weights_df is None
    assert pre_system.corrections_df is None

    pre_system.baseyear = 2020
    assert pre_system.baseyear is not None

    pre_system.annuals_df = annual_df
    pre_system.indicators_df = indicator_df
    pre_system.weights_df = weight_df
    assert pre_system.annuals_df is not None
    assert pre_system.indicators_df is not None
    assert pre_system.weights_df is not None

    pre_system.info()
