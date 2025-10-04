##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

from __future__ import annotations

import numpy as np
import pandas as pd


class Formula:
    """Abstract base class for all pre-system formulas.

    Provides a common interface and input validation for computing time
    series from annual levels, indicator series, optional weights, and
    optional corrections. Subclasses supply the concrete computation
    in the what property, indicators_weights, and evaluate.

    Attributes:
        _name (str): Lower-cased identifier of the formula.
        _baseyear (int | None): Base year used for normalisation and alignment.
        _calls_on (dict[str, Formula]): Dependency formulas used by this formula.
    """

    # TODO: Check why this class uses _baseyear as both a class variable and
    # an instance variable. The instance variable hides the class variable and
    # causes confusion. Check if one of them can be renamed.
    _baseyear: int | None = None

    def __init__(self, name: str) -> None:
        """Initialize a Formula instance.

        Parameters
        ----------
        name : str
            The name of the formula.

        Raises:
        ------
        TypeError
            If `name` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("name must be str")
        self._name = name.lower()
        self._baseyear: int | None = None
        self._calls_on: dict[str, Formula] = {}

    @property
    def name(self) -> str:
        """Formula name.

        Returns:
            str: Lower-cased unique name of the formula.
        """
        return self._name

    @property
    def baseyear(self) -> int | None:
        """Base year used by the formula.

        Returns:
            int | None: The base year if set; otherwise None.
        """
        return self._baseyear

    @baseyear.setter
    def baseyear(self, baseyear: int) -> None:
        """Set the base year for this formula.

        Args:
            baseyear: Year used for normalisation and alignment.

        Raises:
            TypeError: If baseyear is not an int.
        """
        if not isinstance(baseyear, int):
            raise TypeError("baseyear must be int")
        self._baseyear = baseyear

    @property
    def what(self) -> str:
        """Algebraic representation of the formula.

        Returns:
            str: A human-readable expression describing how the series is computed.
        """
        return ""

    @property
    def calls_on(self) -> dict[str, Formula]:
        """Dependencies this formula uses.

        Returns:
            dict[str, Formula]: Mapping of dependency names to Formula instances.
        """
        return self._calls_on

    @property
    def indicators(self) -> list[str]:
        """Indicator column names referenced by this formula.

        Returns:
            list[str]: Indicator identifiers expected in indicators_df.
        """
        return []

    @property
    def weights(self) -> list[str] | list[float]:
        """Weights used by the formula.

        Returns:
            list[str] | list[float]: Either names of weight columns (to be
            read from weights_df) or constant numeric weights.
        """
        return []

    def __repr__(self) -> str:
        return f"Formula: {self.name} = {self.what}"

    def __call__(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the formula by calling the instance.

        This is shorthand for calling evaluate(...).

        Args:
            annual_df: Annual level series.
            indicators_df: Indicator series.
            weights_df: Optional weights.
            correction_df: Optional corrections.
            test_dfs: Whether to validate input DataFrames.

        Returns:
            pd.Series: The evaluated time series.
        """
        return self.evaluate(
            annual_df, indicators_df, weights_df, correction_df, test_dfs=test_dfs
        )

    def __add__(self, other: Formula) -> FSum:
        """Create a summed formula.

        Args:
            other: The right-hand formula to add.

        Returns:
            FSum: A formula representing self + other.
        """
        return FSum(f"{self.name}+{other.name}", self, other)

    def __mul__(self, other: Formula) -> FMult:
        """Create a product formula.

        Args:
            other: The right-hand formula to multiply by.

        Returns:
            FMult: A formula representing self * other.
        """
        return FMult(f"{self.name}*{other.name}", self, other)

    def __truediv__(self, other: Formula) -> FDiv:
        """Create a division formula.

        Args:
            other: The right-hand formula to divide by.

        Returns:
            FDiv: A formula representing self / other.
        """
        return FDiv(f"{self.name}/{other.name}", self, other)

    def info(self, i: int = 0) -> None:
        """Print a tree view of this formula and its dependencies.

        Args:
            i: Indentation level used internally for recursion.
        """
        what = self.what if len(self.what) <= 100 else "..."
        print(f'{" "*i}{self.name} = {what}')
        for _, val in self.calls_on.items():
            val.info(i + 1)

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """List indicator-weight pairs contributing to this formula.

        Args:
            trace: If True, include pairs from dependencies as well.

        Returns:
            list[tuple[str, float]]: List of (indicator, weight) pairs. The
            weight may be a float or a name that resolves to a weight series.
        """
        return []

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the formula using the provided data.

        This function is only used by subclasses to check preconditions. In this
        baseclass it returns a dummy pd.Series object which is not used.

        Args:
            annual_df: The annual data used for evaluation.
            indicators_df: The indicator data used for evaluation.
            weights_df: The weight data used for evaluation. Optional and defaults to None.
            correction_df: The correction data used for evaluation. Ootional and defaults to None.
            test_dfs: If dataframes should be tested or not.

        Returns:
            A dummy pd.Series object. The return value is only valid for subclasses.

        Raises:
            ValueError: If the base year is not set or is out of range for the provided data.
            AttributeError: If the index of any input DataFrame is not a Pandas
                PeriodIndex or if the frequency is incorrect.
        """
        if self.baseyear is None:
            raise ValueError("baseyear is None")

        if test_dfs:
            self._check_df("annual_df", annual_df, self.baseyear, "YE")
            self._check_df("indicators_df", indicators_df, self.baseyear)

            if weights_df is not None:
                self._check_df("weights_df", weights_df, self.baseyear, "YE")

            if not isinstance(indicators_df.index, pd.PeriodIndex):
                raise AttributeError("indicators_df.index must be Pandas.PeriodIndex")

            if correction_df is not None:
                self._check_df(
                    "correction_df",
                    correction_df,
                    self.baseyear,
                    indicators_df.index.freqstr,
                )
        return pd.Series()

    # Method that checks that conditions are met for DataFrame to be valid input
    @staticmethod
    def _check_df(
        df_name: str, df: pd.DataFrame, baseyear: int, frequency: str | None = None
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df_name} must be a Pandas.DataFrame")
        if not isinstance(df.index, pd.PeriodIndex):
            raise AttributeError(f"{df_name}.index must be Pandas.PeriodIndex")
        if frequency and df.index.freq != frequency:
            raise AttributeError(f"{df_name} must have frequency {frequency}")
        if baseyear not in df.index.year:
            raise IndexError(f"baseyear {baseyear} is out of range for annual_df")
        if not all(np.issubdtype(df[x].dtype, np.number) for x in df.columns):  # type: ignore [arg-type]
            raise TypeError(f"All columns in {df_name} must be numeric")


class Indicator(Formula):
    """Indicator-based disaggregation formula.

    Uses one or more indicator series, optionally with weights and an
    optional correction factor, to distribute an annual level across
    sub-annual periods. See __init__ for parameter details.
    """

    def __init__(
        self,
        name: str,
        annual: str,
        indicators: list[str],
        weights: list[str] | list[float] | None = None,
        correction: str | None = None,
        normalise: bool = False,
        aggregation: str = "sum",
    ) -> None:
        """Initialize an Indicator object.

        Parameters
        ----------
        name : str
            The name of the indicator.
        annual : str
            The name of the annual data.
        indicator_names : list[str]
            The list of indicator names.
        weight_names : list[str], optional
            The list of weight names, by default None.
        correction_name : str, optional
            The name of the correction data, by default None.

        Raises:
        ------
        IndexError
            If `weight_names` is provided and has a different length than `indicator_names`.
        """
        super().__init__(name)
        if not isinstance(annual, str):
            raise TypeError("annual must be str")
        if not isinstance(indicators, list):
            raise TypeError("indicator_names must be a list")
        if not all(isinstance(x, str) for x in indicators):
            raise TypeError("indicator_names must containt str")
        if weights and len(weights) != len(indicators):
            raise IndexError("weight_names must have same length as indicator_names")
        if weights and not all(isinstance(x, type(weights[0])) for x in weights):
            raise TypeError("all weights must be of same type")
        if aggregation.lower() not in ["sum", "avg"]:
            raise NameError("aggregation must be sum or avg")
        self._annual = annual
        self._indicators = [x.strip() for x in indicators]
        self._weights = [] if weights is None else weights
        self._correction = correction
        self._normalise = normalise
        self._aggregation = aggregation.lower()

    @property
    def indicators(self) -> list[str]:
        """Indicator column names used by this formula.

        Returns:
            list[str]: Indicator identifiers expected in indicators_df.
        """
        return self._indicators

    @property
    def weights(self) -> list[str] | list[float]:
        """Weights to apply to indicators.

        Returns:
            list[str] | list[float]: Either names of weight columns (from
            weights_df) or constant numeric weights. Defaults to 1.0 per
            indicator when not provided.
        """
        if self._weights:
            return self._weights
        return [1.0 for _ in self.indicators]

    @property
    def what(self) -> str:
        """Textual representation of this indicator formula.

        Returns:
            str: Expression showing how the annual level is distributed by
            indicators (optionally weighted, normalised, and corrected).
        """
        correction = f"{self._correction}*" if self._correction else ""

        if self._normalise:
            indicators = [
                f"{x}/sum({x}<date {self.baseyear}>)" for x in self._indicators
            ]
        else:
            indicators = self._indicators

        if self._weights:
            aggregated_indicators = "+".join(
                [
                    "*".join([str(x).lower(), y.lower()])
                    for x, y in zip(self._weights, indicators, strict=True)
                ]
            )
        else:
            aggregated_indicators = "+".join([x.lower() for x in indicators])

        numerator = f"{correction}({aggregated_indicators})"
        denominator = f"{self._aggregation}({numerator}<date {self.baseyear}>)"
        fraction = f"{numerator}/{denominator}"

        return f"{self._annual.lower()}*<date {self.baseyear}>*{fraction}"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        Args:
            trace: Ignored for this class; included for API symmetry.

        Returns:
            list[tuple[str, float]]: Pairs of indicators with their numeric
            weights. Raises TypeError if any weight is not a float.

        Raises:
            TypeError: If any weight is not a float.
        """
        if not all(isinstance(x, float) for x in self.weights):
            raise TypeError("all weights must be of type float")
        return [(x, y) for x, y in zip(self.indicators, self.weights, strict=True)]  # type: ignore

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If the baseyear is not set.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.
        NameError
            If the required column names are not present in the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        super().evaluate(
            annual_df, indicators_df, weights_df, correction_df, test_dfs=test_dfs
        )
        if not isinstance(annual_df.index, pd.PeriodIndex):
            raise AttributeError("annual_df.index must be Pandas.PeriodIndex")

        if self._annual not in annual_df.columns:
            raise NameError(f"Cannot find {self._annual} in annual_df")

        if any(x not in indicators_df.columns for x in self._indicators):
            missing = [x for x in self._indicators if x not in indicators_df.columns]
            raise NameError(f'Cannot find {",".join(missing)} in indicators_df')

        indicator_matrix = indicators_df.loc[:, self._indicators]
        if not isinstance(indicator_matrix.index, pd.PeriodIndex):
            raise AttributeError("indicator_matrix.index must be Pandas.PeriodIndex")

        if self._normalise:
            indicator_matrix = indicator_matrix.div(
                indicator_matrix.loc[indicator_matrix.index.year == self.baseyear].sum()
            )

        if self._weights:
            if all(isinstance(x, str) for x in self._weights):
                if weights_df is None:
                    raise NameError(f"{self.name} expects weights_df")
                if any(x not in weights_df.columns for x in self._weights):
                    missing = [x for x in self._weights if x not in weights_df.columns]  # type: ignore
                    raise NameError(f'Cannot find {",".join(missing)} in weights_df')
                if not isinstance(weights_df.index, pd.PeriodIndex):
                    raise AttributeError("weights_df.index must be Pandas.PeriodIndex")
                weight_vector = weights_df.loc[
                    weights_df.index.year == self.baseyear, self._weights
                ].to_numpy()  # type: ignore [misc]

            if all(isinstance(x, float) for x in self._weights):
                weight_vector = np.array([self._weights])

            weighted_indicators = pd.Series(
                indicator_matrix.to_numpy().dot(weight_vector.transpose())[:, 0],
                index=indicators_df.index,
            )
        else:
            weighted_indicators = indicator_matrix.sum(axis=1, skipna=False)

        if self._correction:
            if correction_df is None:
                raise NameError(f"{self.name} expects correction_df")
            if self._correction not in correction_df.columns:
                raise NameError(f"{self._correction} is not in correction_df")
            corrected_indicators = (
                weighted_indicators * correction_df.loc[:, self._correction]
            )
        else:
            corrected_indicators = weighted_indicators

        evaluated_series = annual_df.loc[
            annual_df.index.year == self.baseyear, self._annual
        ].to_numpy() * corrected_indicators.div(
            corrected_indicators.loc[
                corrected_indicators.index.year == self.baseyear
            ].sum()
            if self._aggregation == "sum"
            else corrected_indicators.loc[
                corrected_indicators.index.year == self.baseyear
            ].mean()
        )

        return evaluated_series  # type: ignore [no-any-return]


class FDeflate(Formula):
    """Deflate a base formula by indicator(s).

    Computes a series proportional to a base formula divided by (possibly
    weighted) indicator(s), optionally normalised and corrected. See
    `__init__` for parameter details.
    """

    def __init__(
        self,
        name: str,
        formula: Formula,
        indicators: list[str],
        weights: list[str] | list[float] | None = None,
        correction: str | None = None,
        normalise: bool = False,
    ) -> None:
        """Initialize an FDeflate object.

        Parameters
        ----------
        name : str
            The name of the FDeflate formula.
        formula : Formula
            The base formula to be used.
        indicator_names : list[str]
            List of indicator names used in the formula.
        weight_names : list[str], optional
            List of weight names corresponding to the indicator names. Defaults to None.
        correction_name : str, optional
            The name of the correction factor. Defaults to None.

        Raises:
        ------
        TypeError
            If `formula` is not of type Formula.
        IndexError
            If `weight_names` is provided and has a different length than `indicator_names`.
        """
        super().__init__(name)
        if not isinstance(formula, Formula):
            raise TypeError("formula must be of type Formula")
        if weights and len(weights) != len(indicators):
            raise IndexError("weight_names must have same length as indicator_names")
        if weights and not all(isinstance(x, type(weights[0])) for x in weights):
            raise TypeError("all weights must be of same type")
        self._formula = formula
        self._indicators = [x.strip() for x in indicators]
        self._weights = [] if weights is None else weights
        self._correction = correction
        self._normalise = normalise
        self._calls_on = {formula.name: formula}

    @property
    def indicators(self) -> list[str]:
        """Indicator column names used by this deflation formula.

        Returns:
            list[str]: Indicator identifiers expected in indicators_df.
        """
        return self._indicators

    @property
    def weights(self) -> list[str] | list[float]:
        """Weights to apply to indicators in deflation.

        Returns:
            list[str] | list[float]: Either names of weight columns (from
            weights_df) or constant numeric weights. Defaults to 1.0 per
            indicator when not provided.
        """
        if self._weights:
            return self._weights
        return [1.0 for _ in self.indicators]

    @property
    def what(self) -> str:
        """Textual representation of this deflation formula.

        Returns:
            str: Expression showing base formula divided by weighted indicators
            (optionally normalised and corrected) and scaled to the base year.
        """
        correction = f"{self._correction}*" if self._correction else ""

        if self._normalise:
            indicators = [
                f"{x}/sum({x}<date {self.baseyear}>)" for x in self._indicators
            ]
        else:
            indicators = self._indicators

        if self._weights:
            aggregated_indicators = "+".join(
                [
                    "*".join([str(x).lower(), y.lower()])
                    for x, y in zip(self._weights, indicators, strict=True)
                ]
            )
        else:
            aggregated_indicators = "+".join([x.lower() for x in indicators])

        numerator = f"{correction}{self._formula.name}/({aggregated_indicators})"
        denominator = f"sum({numerator}<date {self.baseyear}>)"
        fraction = f"({numerator})/{denominator}"

        return f"sum({self._formula.name}<date {self.baseyear}>)*{fraction}"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        Args:
            trace: Ignored for this class; included for API symmetry.

        Returns:
            list[tuple[str, float]]: Pairs of indicators with their numeric
            weights. Raises TypeError if any weight is not a float.
        """
        return [(x, y) for x, y in zip(self.indicators, self.weights, strict=True)] + (
            self._formula.indicators_weights(trace=trace) if trace else []
        )  # type: ignore [return-value]

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the deflation formula.

        Args:
            annual_df: Annual level series.
            indicators_df: Indicator series.
            weights_df: Optional weights used to combine indicators.
            correction_df: Optional correction series applied after deflation.
            test_dfs: Whether to validate inputs (recommended).

        Returns:
            pd.Series: The deflated series aligned to the indicator frequency.

        Raises:
            NameError: If a required column name is missing from one of the DataFrames.
            AttributeError: If the index is not of type pd.PeriodIndex or has an incorrect frequency.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if any(x not in indicators_df.columns for x in self._indicators):
            raise NameError(
                f'All of {",".join(self._indicators)} is not in indicators_df'
            )

        indicator_matrix = indicators_df.loc[:, self._indicators]
        if not isinstance(indicator_matrix.index, pd.PeriodIndex):
            raise AttributeError("indicator_matrix.index must be Pandas.PeriodIndex")

        if self._normalise:
            indicator_matrix = indicator_matrix.div(
                indicator_matrix.loc[indicator_matrix.index.year == self.baseyear].sum()
            )

        if self._weights:
            if all(isinstance(x, str) for x in self._weights):
                if weights_df is None:
                    raise NameError(f"{self.name} expects weights_df")
                if any(x not in weights_df.columns for x in self._weights):
                    missing = [x for x in self._weights if x not in weights_df.columns]
                    raise NameError(f'Cannot find {",".join(missing)} in weights_df')  # type: ignore [arg-type]
                if not isinstance(weights_df.index, pd.PeriodIndex):
                    raise AttributeError("weights_df.index must be Pandas.PeriodIndex")
                weight_vector = weights_df.loc[
                    weights_df.index.year == self.baseyear, self._weights
                ].to_numpy()  # type: ignore [misc]

            if all(isinstance(x, float) for x in self._weights):
                weight_vector = np.array([self._weights])

            weighted_indicators = pd.Series(
                indicator_matrix.to_numpy().dot(weight_vector.transpose())[:, 0],
                index=indicators_df.index,
            )
        else:
            weighted_indicators = indicator_matrix.sum(axis=1, skipna=False)

        evaluated_formula = self._formula.evaluate(*all_dfs, test_dfs=test_dfs)
        if not isinstance(evaluated_formula.index, pd.PeriodIndex):
            raise AttributeError("evaluated_formula.index must be Pandas.PeriodIndex")

        formula_divided = evaluated_formula.div(weighted_indicators)

        if self._correction:
            if correction_df is None:
                raise NameError(f"{self.name} expects correction_df")
            if self._correction not in correction_df.columns:
                raise NameError(f"{self._correction} is not in correction_df")
            formula_corrected = formula_divided * correction_df.loc[:, self._correction]
        else:
            formula_corrected = formula_divided
        if not isinstance(formula_corrected.index, pd.PeriodIndex):
            raise AttributeError("formula_corrected.index must be Pandas.PeriodIndex")

        evaluated_series = evaluated_formula.loc[
            evaluated_formula.index.year == self.baseyear
        ].sum() * formula_corrected.div(
            formula_corrected.loc[formula_corrected.index.year == self.baseyear].sum()
        )

        return evaluated_series  # type: ignore [no-any-return]


class FInflate(Formula):
    """Inflate a base formula by indicator(s).

    Computes a series proportional to a base formula multiplied by
    (possibly weighted) indicator(s), optionally normalised and corrected.
    See __init__ for parameter details.
    """

    def __init__(
        self,
        name: str,
        formula: Formula,
        indicators: list[str],
        weights: list[str] | list[float] | None = None,
        correction: str | None = None,
        normalise: bool = False,
    ) -> None:
        """Initialize an FInflate object.

        Parameters
        ----------
        name : str
            The name of the FDeflate formula.
        formula : Formula
            The base formula to be used.
        indicator_names : list[str]
            List of indicator names used in the formula.
        weight_names : list[str], optional
            List of weight names corresponding to the indicator names. Defaults to None.
        correction_name : str, optional
            The name of the correction factor. Defaults to None.

        Raises:
        ------
        TypeError
            If `formula` is not of type Formula.
        IndexError
            If `weight_names` is provided and has a different length than `indicator_names`.
        """
        super().__init__(name)
        if not isinstance(formula, Formula):
            raise TypeError("formula must be of type Formula")
        if weights and len(weights) != len(indicators):
            raise IndexError("weight_names must have same length as indicator_names")
        if weights and not all(isinstance(x, type(weights[0])) for x in weights):
            raise TypeError("all weights must be of same type")
        self._formula = formula
        self._indicators = [x.strip() for x in indicators]
        self._weights = [] if weights is None else weights
        self._correction = correction
        self._normalise = normalise
        self._calls_on = {formula.name: formula}

    @property
    def indicators(self) -> list[str]:
        """Indicator column names used by this inflation formula.

        Returns:
            list[str]: Indicator identifiers expected in indicators_df.
        """
        return self._indicators

    @property
    def weights(self) -> list[str] | list[float]:
        """Weights to apply to indicators in inflation.

        Returns:
            list[str] | list[float]: Either names of weight columns (from
            weights_df) or constant numeric weights. Defaults to 1.0 per
            indicator when not provided.
        """
        if self._weights:
            return self._weights
        return [1.0 for _ in self.indicators]

    @property
    def what(self) -> str:
        """Textual representation of this inflation formula.

        Returns:
            str: Expression showing base formula multiplied by weighted
            indicators (optionally normalised and corrected) and scaled to
            the base year.
        """
        correction = f"{self._correction}*" if self._correction else ""

        if self._normalise:
            indicators = [
                f"{x}/sum({x}<date {self.baseyear}>)" for x in self._indicators
            ]
        else:
            indicators = self._indicators

        if self._weights:
            aggregated_indicators = "+".join(
                [
                    "*".join([str(x).lower(), y.lower()])
                    for x, y in zip(self._weights, indicators, strict=True)
                ]
            )
        else:
            aggregated_indicators = "+".join([x.lower() for x in indicators])

        numerator = f"{correction}{self._formula.name}*({aggregated_indicators})"
        denominator = f"sum({numerator}<date {self.baseyear}>)"
        fraction = f"({numerator})/{denominator}"

        return f"sum({self._formula.name}<date {self.baseyear}>)*{fraction}"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        Args:
            trace: Ignored for this class; included for API symmetry.

        Returns:
            list[tuple[str, float]]: Pairs of indicators with their numeric
            weights. Raises TypeError if any weight is not a float.
        """
        return [(x, y) for x, y in zip(self.indicators, self.weights, strict=True)] + (
            self._formula.indicators_weights(trace=trace) if trace else []
        )  # type: ignore [return-value]

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the inflation formula.

        Args:
            annual_df: Annual level series.
            indicators_df: Indicator series.
            weights_df: Optional weights used to combine indicators.
            correction_df: Optional correction series applied after inflation.
            test_dfs: Whether to validate inputs (recommended).

        Returns:
            pd.Series: The inflated series aligned to the indicator frequency.

        Raises:
            NameError: If a required column name is missing from one of the DataFrames.
            AttributeError: If the index is not of type pd.PeriodIndex or has an incorrect frequency.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if any(x not in indicators_df.columns for x in self._indicators):
            raise NameError(
                f'All of {",".join(self._indicators)} is not in indicators_df'
            )

        indicator_matrix = indicators_df.loc[:, self._indicators]
        if not isinstance(indicator_matrix.index, pd.PeriodIndex):
            raise AttributeError("indicator_matrix.index must be Pandas.PeriodIndex")

        if self._normalise:
            indicator_matrix = indicator_matrix.div(
                indicator_matrix.loc[indicator_matrix.index.year == self.baseyear].sum()
            )

        if self._weights:
            if all(isinstance(x, str) for x in self._weights):
                if weights_df is None:
                    raise NameError(f"{self.name} expects weights_df")
                if any(x not in weights_df.columns for x in self._weights):
                    missing = [x for x in self._weights if x not in weights_df.columns]
                    raise NameError(f'Cannot find {",".join(missing)} in weights_df')  # type: ignore [arg-type]
                if not isinstance(weights_df.index, pd.PeriodIndex):
                    raise AttributeError("weights_df.index must be Pandas.PeriodIndex")
                weight_vector = weights_df.loc[
                    weights_df.index.year == self.baseyear, self._weights
                ].to_numpy()  # type: ignore [misc]

            if all(isinstance(x, float) for x in self._weights):
                weight_vector = np.array([self._weights])

            weighted_indicators = pd.Series(
                indicator_matrix.to_numpy().dot(weight_vector.transpose())[:, 0],
                index=indicators_df.index,
            )
        else:
            weighted_indicators = indicator_matrix.sum(axis=1, skipna=False)

        evaluated_formula = self._formula.evaluate(*all_dfs, test_dfs=test_dfs)
        if not isinstance(evaluated_formula.index, pd.PeriodIndex):
            raise AttributeError("evaluated_formula.index must be Pandas.PeriodIndex")

        formula_divided = evaluated_formula * weighted_indicators

        if self._correction:
            if correction_df is None:
                raise NameError(f"{self.name} expects correction_df")
            if self._correction not in correction_df.columns:
                raise NameError(f"{self._correction} is not in correction_df")
            formula_corrected = formula_divided * correction_df.loc[:, self._correction]
        else:
            formula_corrected = formula_divided
        if not isinstance(formula_corrected.index, pd.PeriodIndex):
            raise AttributeError("formula_corrected.index must be Pandas.PeriodIndex")

        evaluated_series = evaluated_formula.loc[
            evaluated_formula.index.year == self.baseyear
        ].sum() * formula_corrected.div(
            formula_corrected.loc[formula_corrected.index.year == self.baseyear].sum()
        )

        return evaluated_series  # type: ignore [no-any-return]


class FSum(Formula):
    """Sum of multiple formulas.

    Produces a series that is the element-wise sum of its operand formulas.
    """

    def __init__(self, name: str, *formulae: Formula) -> None:
        """Initialize an FSum object.

        Parameters
        ----------
        name : str
            The name of the FSum object.
        *formulae : Formula
            Variable number of Formula objects.

        Raises:
        ------
        TypeError
            If any of the *formulae is not of type Formula.
        """
        super().__init__(name)
        if not all(isinstance(x, Formula) for x in formulae):
            raise TypeError("*formulae must be of type Formula")
        self._formulae = formulae
        self._calls_on = {x.name: x for x in formulae}

    @property
    def what(self) -> str:
        """Textual representation of this sum formula.

        Returns:
            str: Expression showing the sum of operand formulas.
        """
        return "+".join([x.name for x in self._formulae])

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method uses the `_formula.indicators_weights` to compute the indicator
        weights when `trace` is set to `True`. If `trace` is `False`, it returns an
        empty list.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float. Returns an empty list if `trace` is False.
        """
        indicators_weights = []
        if trace:
            for formula in self._formulae:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If any of the formulae do not evaluate.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.
        NameError
            If the required column names are not present in the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if any(x.evaluate(*all_dfs, test_dfs=test_dfs) is None for x in self._formulae):
            raise ValueError("some of the formulae do not evaluate")

        # The sum function returns an int if there are no elements to sum.
        # Make it return an empty pd.Series instead, if this is the case.
        formulae_values = [
            x.evaluate(*all_dfs, test_dfs=test_dfs) for x in self._formulae
        ]
        return sum(formulae_values) if formulae_values else pd.Series()  # type: ignore


class FSumProd(Formula):
    """Weighted sum of products of formulas.

    Computes a linear combination of operand formulas with either numeric
    coefficients or weight column names.
    """

    def __init__(
        self, name: str, formulae: list[Formula], weights: list[float] | list[str]
    ) -> None:
        """Initialize an FSumProd object.

        Parameters
        ----------
        name : str
            The name of the FSum object.
        formulae : list[Formula]
            ...
        coefficients : list[float]
            ...

        Raises:
        ------
        TypeError
            If any of the *formulae is not of type Formula.
        """
        super().__init__(name)
        if not all(isinstance(x, Formula) for x in formulae):
            raise TypeError("*formulae must be of type Formula")
        if weights and not all(isinstance(x, type(weights[0])) for x in weights):
            raise TypeError("all weights must be of same type")
        self._formulae = formulae
        self._weights = weights
        self._calls_on = {x.name: x for x in formulae}

    @property
    def what(self) -> str:
        """Textual representation of this weighted sum-product formula.

        Returns:
            str: Expression showing the weighted sum of products of operand formulas.
        """
        return "+".join(
            [
                "*".join([x.name, str(y).lower()])
                for x, y in zip(self._formulae, self._weights, strict=True)
            ]
        )

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method aggregates the weights of indicators from a collection of
        formulas. If trace is set to True, the weights from each formula's
        indicators are retrieved and combined.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float.
        """
        indicators_weights = []
        if trace:
            for formula in self._formulae:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If any of the formulae do not evaluate.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.
        NameError
            If the required column names are not present in the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if any(x.evaluate(*all_dfs, test_dfs=test_dfs) is None for x in self._formulae):
            raise ValueError("some of the formulae do not evaluate")

        if all(isinstance(x, str) for x in self._weights):
            if weights_df is None:
                raise NameError(f"{self.name} expects weights_df")
            if any(x not in weights_df.columns for x in self._weights):
                missing = [x for x in self._weights if x not in weights_df.columns]
                raise NameError(f'Cannot find {",".join(missing)} in weights_df')  # type: ignore [arg-type]
            if not isinstance(weights_df.index, pd.PeriodIndex):
                raise AttributeError("weights_df.index must be Pandas.PeriodIndex")
            weight_vector = (
                weights_df[weights_df.index.year == self.baseyear][self._weights]
                .sum()
                .tolist()
            )
            return sum(  # type: ignore
                x.evaluate(*all_dfs, test_dfs=test_dfs) * y
                for x, y in zip(self._formulae, weight_vector, strict=True)
            )
        if all(isinstance(x, float) for x in self._weights):
            # The sum function returns an int if there are no elements to sum.
            # Make it return an empty pd.Series instead, if this is the case.
            formulae_values = [
                x.evaluate(*all_dfs, test_dfs=test_dfs) * y
                for x, y in zip(self._formulae, self._weights, strict=True)
            ]
            return sum(formulae_values) if formulae_values else pd.Series()
        raise TypeError("All weights must be str or float")


class FMult(Formula):
    """Element-wise product of two formulas.

    Multiplies two formula series with matching indices.
    """

    def __init__(self, name: str, formula1: Formula, formula2: Formula) -> None:
        """Initialize an FMult object.

        Args:
            name: The name of the FMult object.
            formula1: The first formula.
            formula2: The second formula.

        Raises:
            TypeError: If any of the input formulas are not of type Formula.
        """
        super().__init__(name)
        if not (isinstance(formula1, Formula) and isinstance(formula2, Formula)):
            raise TypeError("formula1 and formula2 must be of type Formula")
        self._formula1 = formula1
        self._formula2 = formula2
        self._calls_on = {formula1.name: formula1, formula2.name: formula2}

    @property
    def what(self) -> str:
        """Textual representation of this product formula.

        Returns:
            str: Expression showing the product of two operand formulas.
        """
        return f"{self._formula1.name}*{self._formula2.name}"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method aggregates the weights of indicators from a collection of
        formulas. If trace is set to True, the weights from each formula's
        indicators are retrieved and combined.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float.
        """
        indicators_weights = []
        if trace:
            for formula in [self._formula1, self._formula2]:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If the baseyear is not set.
            If formula1 does not evaluate.
            If formula2 does not evaluate.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if self._formula1.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError("formula1 does not evaluate")
        if self._formula2.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError("formula2 does not evaluate")

        return self._formula1.evaluate(
            *all_dfs, test_dfs=test_dfs
        ) * self._formula2.evaluate(*all_dfs, test_dfs=test_dfs)


class FDiv(Formula):
    """Element-wise division of two formulas.

    Divides one formula series by another with matching indices.
    """

    def __init__(self, name: str, formula1: Formula, formula2: Formula) -> None:
        """Initialize an FDiv object.

        Args:
            name: The name of the FDiv object.
            formula1: The numerator formula.
            formula2: The denominator formula.

        Raises:
            TypeError: If any of the input formulas are not of type Formula.
        """
        super().__init__(name)
        if not (isinstance(formula1, Formula) and isinstance(formula2, Formula)):
            raise TypeError("formula1 and formula2 must be of type Formula")
        self._formula1 = formula1
        self._formula2 = formula2
        self._calls_on = {formula1.name: formula1, formula2.name: formula2}

    @property
    def what(self) -> str:
        """Textual representation of this division formula.

        Returns:
            str: Expression showing the division of two operand formulas.
        """
        return f"{self._formula1.name}/{self._formula2.name}"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method aggregates the weights of indicators from a collection of
        formulas. If trace is set to True, the weights from each formula's
        indicators are retrieved and combined.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float.
        """
        indicators_weights = []
        if trace:
            for formula in [self._formula1, self._formula2]:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If the baseyear is not set.
            If formula1 does not evaluate.
            If formula2 does not evaluate.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if self._formula1.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError("formula1 does not evaluate")
        if self._formula2.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError("formula2 does not evaluate")

        return self._formula1.evaluate(*all_dfs, test_dfs=test_dfs).div(
            self._formula2.evaluate(*all_dfs, test_dfs=test_dfs)
        )


class MultCorr(Formula):
    """Apply a multiplicative correction to a formula.

    Multiplies the evaluated series by a named correction series.
    """

    def __init__(self, formula: Formula, correction_name: str) -> None:
        """Initialize a MultCorr object.

        Parameters
        ----------
        formula : Formula
            The Formula object to be multiplied by the correction factor.
        correction_name : str
            The name of the correction factor.

        Raises:
        ------
        TypeError
            If formula is not of type Formula.
        """
        super().__init__(formula.name)
        if not isinstance(formula, Formula):
            raise TypeError("formula must be of type Formula")
        self._formula = formula
        self._correction_name = correction_name
        self._calls_on = formula._calls_on

    @property
    def baseyear(self) -> int | None:
        """Base year used for the correction formula.

        Returns:
            int | None: The base year if set, otherwise None.
        """
        return self._baseyear

    @baseyear.setter
    def baseyear(self, baseyear: int) -> None:
        if not isinstance(baseyear, int):
            raise TypeError("baseyear must be int")
        self._baseyear = baseyear
        # Pass baseyear to formula that goes into correction
        self._formula.baseyear = baseyear

    @property
    def what(self) -> str:
        """Textual representation of this multiplicative correction formula.

        Returns:
            str: Expression showing the multiplicative correction applied to the formula.
        """
        return (
            f"sum(({self._formula.what})<date {self.baseyear}>)*"
            f"{self._correction_name}*({self._formula.what})/"
            f"sum({self._correction_name}*({self._formula.what})<date {self.baseyear}>)"
        )

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method uses the `_formula.indicators_weights` to compute the indicator
        weights when `trace` is set to `True`. If `trace` is `False`, it returns an
        empty list.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float. Returns an empty list if `trace` is False.
        """
        return self._formula.indicators_weights(trace=trace) if trace else []

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If the baseyear is not set.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.
        NameError
            If the required column names are not present in the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if correction_df is None:
            raise TypeError("correction_df must be defined when evaluating MultCorr")

        evaluated_formula = self._formula.evaluate(*all_dfs, test_dfs=test_dfs)
        if not isinstance(evaluated_formula.index, pd.PeriodIndex):
            raise AttributeError("evaluated_formula.index must be Pandas.PeriodIndex")

        formula_corrected = evaluated_formula * correction_df[self._correction_name]
        if not isinstance(formula_corrected.index, pd.PeriodIndex):
            raise AttributeError("formula_corrected.index must be Pandas.PeriodIndex")

        return evaluated_formula[  # type: ignore
            evaluated_formula.index.year == self.baseyear
        ].sum() * formula_corrected.div(
            formula_corrected[formula_corrected.index.year == self.baseyear].sum()
        )


class AddCorr(Formula):
    """Apply an additive correction to a formula.

    Adds a named correction series to the evaluated series.
    """

    def __init__(self, formula: Formula, correction_name: str) -> None:
        """Initialize an AddCorr object.

        Parameters
        ----------
        formula : Formula
            The Formula object to be added with the correction factor.
        correction_name : str
            The name of the correction factor.

        Raises:
        ------
        TypeError
            If formula is not of type Formula.
        """
        super().__init__(formula.name)
        if not isinstance(formula, Formula):
            raise TypeError("formula must be of type Formula")
        self._formula = formula
        self._correction_name = correction_name
        self._calls_on = formula.calls_on

    @property
    def baseyear(self) -> int | None:
        """Base year used for the additive correction formula.

        Returns:
            int | None: The base year if set, otherwise None.
        """
        return self._baseyear

    @baseyear.setter
    def baseyear(self, baseyear: int) -> None:
        if not isinstance(baseyear, int):
            raise TypeError("baseyear must be int")
        self._baseyear = baseyear
        # Pass baseyear to formula that goes into correction
        self._formula.baseyear = baseyear

    @property
    def what(self) -> str:
        """Textual representation of this additive correction formula.

        Returns:
            str: Expression showing the additive correction applied to the formula.
        """
        return f"{self._correction_name}+({self._formula.what})-avg({self._correction_name}<date {self.baseyear})"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method uses the `_formula.indicators_weights` to compute the indicator
        weights when `trace` is set to `True`. If `trace` is `False`, it returns an
        empty list.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float. Returns an empty list if `trace` is False.
        """
        return self._formula.indicators_weights(trace=trace) if trace else []

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If the baseyear is not set.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.
        NameError
            If the required column names are not present in the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if correction_df is None:
            raise TypeError("correction_df must be defined when evaluating AddCorr")
        if not isinstance(correction_df.index, pd.PeriodIndex):
            raise AttributeError("correction_df.index must be Pandas.PeriodIndex")

        return (
            correction_df[self._correction_name]
            + self._formula.evaluate(*all_dfs, test_dfs=test_dfs)
            - correction_df[correction_df.index.year == self.baseyear][
                self._correction_name
            ].mean()
        )


class FJoin(Formula):
    """Join two formulas at a given year.

    Uses one formula up to (from_year - 1) and another from from_year onward.
    """

    def __init__(
        self, name: str, formula1: Formula, formula0: Formula, from_year: int
    ) -> None:
        """Initialize an FJoin object.

        Parameters
        ----------
        name : str
            The name of the FJoin formula.
        formula1 : Formula
            The Formula object to be joined starting in from_year.
        formula0 : Formula
            The Formula object to be joined ending before from_year.
        from_year : str
            The year to join formula1 and formula0

        Raises:
        ------
        TypeError
            If formula1 is not of type Formula.
        TypeError
            If formula0 is not of type Formula.
        TypeError
            If from year is not of type str.
        """
        super().__init__(name)
        if not (isinstance(formula1, Formula) and isinstance(formula0, Formula)):
            raise TypeError("formula1 and formula0 must be of type Formula")
        if not isinstance(from_year, int):
            raise TypeError("from_year must must be of type int")
        self._formula1 = formula1
        self._formula0 = formula0
        self._from_year = from_year
        self._calls_on = {formula1.name: formula1, formula0.name: formula0}

    @property
    def indicators(self) -> list[str]:
        """All indicator names used by both joined formulas.

        Returns:
            list[str]: Unique indicator names from both formulas.
        """
        return list(set(self._formula1.indicators).union(self._formula0.indicators))

    @property
    def what(self) -> str:
        """Textual representation of this join formula.

        Returns:
            str: Expression showing which formula is used for each year.
        """
        return f"{self._formula1.name} if year>={self._from_year} else {self._formula0.name}"

    def indicators_weights(self, trace: bool = True) -> list[tuple[str, float]]:
        """Indicator-weight pairs for this indicator formula.

        This method uses the `_formula.indicators_weights` to compute the indicator
        weights when `trace` is set to `True`. If `trace` is `False`, it returns an
        empty list.

        Args:
            trace: A flag to indicate whether to calculate the weights with or without tracing.

        Returns:
            list[tuple[str, float]]: A list of tuples where each tuple contains
                the indicator name as a string and its corresponding weight
                as a float. Returns an empty list if `trace` is False.
        """
        indicators_weights = []
        if trace:
            for formula in [self._formula1, self._formula0]:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(
        self,
        annual_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        weights_df: pd.DataFrame | None = None,
        correction_df: pd.DataFrame | None = None,
        test_dfs: bool = True,
    ) -> pd.Series:
        """Evaluate the data using the provided DataFrames and return the evaluated series.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.
        indicators_df : pd.DataFrame
            The DataFrame containing indicator data.
        weights_df : pd.DataFrame, optional
            The DataFrame containing weight data. Defaults to None.
        correction_df : pd.DataFrame, optional
            The DataFrame containing correction data. Defaults to None.

        Raises:
        ------
        ValueError
            If the baseyear is not set.
        TypeError
            If any of the input DataFrames is not of type pd.DataFrame.
        AttributeError
            If the index of any DataFrame is not of type pd.PeriodIndex or has incorrect frequency.
        IndexError
            If the baseyear is out of range for any of the DataFrames.
        NameError
            If the required column names are not present in the DataFrames.

        Returns:
        -------
        pd.Series
            The evaluated series.
        """
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        evaluated_formula1 = self._formula1.evaluate(*all_dfs, test_dfs=test_dfs)
        evaluated_formula0 = self._formula0.evaluate(*all_dfs, test_dfs=test_dfs)
        if not isinstance(evaluated_formula1.index, pd.PeriodIndex):
            raise AttributeError("evaluated_formula.index must be Pandas.PeriodIndex")
        if not isinstance(evaluated_formula0.index, pd.PeriodIndex):
            raise AttributeError("evaluated_formula.index must be Pandas.PeriodIndex")

        return pd.concat(  # type: ignore
            [
                evaluated_formula0[evaluated_formula0.index.year < self._from_year],
                evaluated_formula1[evaluated_formula1.index.year >= self._from_year],
            ]
        )
