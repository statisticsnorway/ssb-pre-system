##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

from __future__ import annotations

import datetime

import pandas as pd

from .formula import Formula


class PreSystem:
    def __init__(self, name: str) -> None:
        self._name = name
        self._baseyear: int | None = None
        self._formulae: dict[str, Formula] = {}
        self._annuals_df: pd.DataFrame | None = None
        self._indicators_df: pd.DataFrame | None = None
        self._weights_df: pd.DataFrame | None = None
        self._corrections_df: pd.DataFrame | None = None
        self._annual_df_updated: str | None = None
        self._indicator_df_updated: str | None = None
        self._weight_df_updated: str | None = None
        self._correction_df_updated: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def baseyear(self) -> int | None:
        return self._baseyear

    @baseyear.setter
    def baseyear(self, baseyear: int) -> None:
        if not isinstance(baseyear, int):
            raise TypeError("baseyear must be int")
        for _, formula in self.formulae.items():
            formula.baseyear = baseyear
        self._baseyear = baseyear

    @property
    def formulae(self) -> dict[str, Formula]:
        return self._formulae

    @property
    def indicators(self) -> list[str]:
        indicator_set: set[str] = set()
        for _, formula in self.formulae.items():
            indicator_set = indicator_set.union(formula.indicators)
        return list(indicator_set)

    @property
    def annuals_df(self) -> pd.DataFrame | None:
        return self._annuals_df

    @annuals_df.setter
    def annuals_df(self, annuals_df: pd.DataFrame) -> None:
        """Set the DataFrame containing annual data.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.

        Raises:
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex or has incorrect frequency.
        """
        if not isinstance(annuals_df, pd.DataFrame):
            raise TypeError("annual_df must be a Pandas.DataFrame")
        if isinstance(annuals_df.index, pd.PeriodIndex):
            if annuals_df.index.freq != "YE":
                raise AttributeError("annual_df must have annual frequency")
            self._check_missing(annuals_df)
            self._annuals_df = annuals_df
            self._annual_df_updated = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            raise AttributeError("annual_df.index must be Pandas.PeriodIndex")

    @property
    def indicators_df(self) -> pd.DataFrame | None:
        return self._indicators_df

    @indicators_df.setter
    def indicators_df(self, indicators_df: pd.DataFrame) -> None:
        """Set the DataFrame containing indicator data.

        Parameters
        ----------
        indicator_df : pd.DataFrame
            The DataFrame containing indicator data.

        Raises:
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex.
        """
        if not isinstance(indicators_df, pd.DataFrame):
            raise TypeError("indicator_df must be a Pandas.DataFrame")
        if not isinstance(indicators_df.index, pd.PeriodIndex):
            raise AttributeError("indicators_df.index must be Pandas.PeriodIndex")
        self._check_missing(indicators_df)
        self._indicators_df = indicators_df
        self._indicator_df_updated = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @property
    def weights_df(self) -> pd.DataFrame | None:
        return self._weights_df

    @weights_df.setter
    def weights_df(self, weights_df: pd.DataFrame) -> None:
        """Set the DataFrame containing weight data.

        Parameters
        ----------
        weight_df : pd.DataFrame
            The DataFrame containing weight data.

        Raises:
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex or has incorrect frequency.
        """
        if not isinstance(weights_df, pd.DataFrame):
            raise TypeError("weight_df must be a Pandas.DataFrame")
        if isinstance(weights_df.index, pd.PeriodIndex):
            if weights_df.index.freq != "YE":
                raise AttributeError("weights_df must have annual frequency")
            self._check_missing(weights_df)
            self._weights_df = weights_df
            self._weight_df_updated = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            raise AttributeError("weights_df.index must be Pandas.PeriodIndex")

    @property
    def corrections_df(self) -> pd.DataFrame | None:
        return self._corrections_df

    @corrections_df.setter
    def corrections_df(self, corrections_df: pd.DataFrame) -> None:
        """Set the DataFrame containing correction data.

        Parameters
        ----------
        correction_df : pd.DataFrame
            The DataFrame containing correction data.

        Raises:
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex.
        """
        if not isinstance(corrections_df, pd.DataFrame):
            raise TypeError("correction_df must be a Pandas.DataFrame")
        if isinstance(corrections_df.index, pd.PeriodIndex):
            self._check_missing(corrections_df)
            self._corrections_df = corrections_df
            self._correction_df_updated = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            raise AttributeError("correction_df.index must be Pandas.PeriodIndex")

    @staticmethod
    def _check_missing(df: pd.DataFrame) -> None:
        count_missing = df.isna().sum()
        if count_missing.sum() > 0:
            print(
                f'WARNING: there are NaN values in {",".join([y for x, y in zip(count_missing, count_missing.index) if x > 0])}'
            )

    def __repr__(self) -> str:
        return f"PreSystem: {self.name}"

    def info(self) -> None:
        print(
            "\n".join(
                [
                    f"PreSystem {self.name} consists of {len(self.formulae)} formulae.",
                    "",
                    f"Baseyear is {self._baseyear}.",
                    "",
                    "DataFrames updated:",
                    f"annuals_df       {self._annual_df_updated}",
                    f"indicators_df    {self._indicator_df_updated}",
                    f"weights_df       {self._weight_df_updated} (optional)",
                    f"corrections_df   {self._correction_df_updated} (optional)",
                ]
            )
        )

    def add_formula(self, formula: Formula) -> None:
        """Add a formula to the PreSystem.

        Parameters
        ----------
        formula : Formula
            The Formula object to be added.

        Raises:
        ------
        TypeError
            If formula is not of type Formula.
        KeyError
            If any of the dependencies of the formula are not registered.
        NameError
            If a formula with the same name already exists and points to a different formula.
        """
        if not isinstance(formula, Formula):
            raise TypeError("formula must be of type Formula")

        if not all(self.formulae.get(x) for x in formula.calls_on):
            raise KeyError(f'Register all dependencies: {",".join(formula.calls_on)}')

        existing_formula = self.formulae.get(formula.name)

        if existing_formula is None:
            self.formulae[formula.name] = formula
        elif formula is not existing_formula:
            raise NameError(
                f"Formula name {formula.name} already exists and points to a different formula"
            )

    def formula(self, name: str) -> Formula | None:
        """Get a formula from the PreSystem.

        Parameters
        ----------
        name : str
            The name of the formula to retrieve.

        Returns:
        -------
        Formula or None
            The requested formula, or None if it doesn't exist.
        """
        return self.formulae.get(name)

    @property
    def evaluate(self) -> pd.DataFrame:
        """Evaluate all registered formulas using the provided data.

        Returns:
        --------
        pd.DataFrame
            The evaluated formulas as a DataFrame.
        """
        self._check_dfs()
        if self.annuals_df is None:
            raise AttributeError("annuals_df can not be None when evaluating")
        if self.indicators_df is None:
            raise AttributeError("indicators_df can not be None when evaluating")

        for _, formula in self.formulae.items():
            if formula.baseyear != self.baseyear:
                raise AttributeError(
                    f"baseyear for formula {formula.name} is not {self.baseyear}. Try setting baseyear"
                )

        evaluated = {}
        for name, formula in self.formulae.items():
            evaluated[name] = formula.evaluate(
                self.annuals_df,
                self.indicators_df,
                self.weights_df,
                self.corrections_df,
                test_dfs=False,
            )

        return pd.concat(evaluated, axis=1)

    def evaluate_formula(self, name: str) -> pd.Series:
        """Evaluate a specific formula using the provided data.

        Parameters
        ----------
        name : str
            The name of the formula to evaluate.

        Returns:
        -------
        pd.Series
            The evaluated formula as a Series.
        """
        self._check_dfs()
        if self.annuals_df is None:
            raise AttributeError("annuals_df can not be None when evaluating")
        if self.indicators_df is None:
            raise AttributeError("indicators_df can not be None when evaluating")

        formula = self.formulae.get(name)
        if not isinstance(formula, Formula):
            raise NameError(f"formula {name} is not in PreSystem {self.name}")

        return formula.evaluate(
            self.annuals_df,
            self.indicators_df,
            self.weights_df,
            self.corrections_df,
            test_dfs=False,
        )

    def evaluate_formulae(self, *names: str) -> pd.DataFrame:
        """Evaluate specific formulae using the provided data.

        Parameters
        ----------
        formula_name : str
            The names of the formulae to evaluate.

        Returns:
        -------
        pd.DataFrame
            The evaluated formulae as a DataFrame.
        """
        self._check_dfs()
        if self.annuals_df is None:
            raise AttributeError("annuals_df can not be None when evaluating")
        if self.indicators_df is None:
            raise AttributeError("indicators_df can not be None when evaluating")

        evaluated = {}
        for name in names:
            formula = self.formulae.get(name)
            if not isinstance(formula, Formula):
                raise NameError(f"formula {name} is not in PreSystem {self.name}")

            evaluated[name] = formula.evaluate(
                self.annuals_df,
                self.indicators_df,
                self.weights_df,
                self.corrections_df,
                test_dfs=False,
            )

        return pd.concat(evaluated, axis=1)

    def _check_dfs(self) -> None:
        if self.annuals_df is None:
            raise AttributeError("annuals_df can not be None when evaluating")
        if not isinstance(self.annuals_df.index, pd.PeriodIndex):
            raise AttributeError("annual_df.index must be Pandas.PeriodIndex")
        if self.indicators_df is None:
            raise AttributeError("indicators_df can not be None when evaluating")
        if not isinstance(self.indicators_df.index, pd.PeriodIndex):
            raise AttributeError("indicators_df.index must be Pandas.PeriodIndex")

        if self.baseyear not in self.annuals_df.index.year:
            raise IndexError(f"baseyear {self.baseyear} is out of range for annuals_df")
        if self.baseyear not in self.indicators_df.index.year:
            raise IndexError(
                f"baseyear {self.baseyear} is out of range for indicators_df"
            )
        if self.weights_df is not None:
            if not isinstance(self.weights_df.index, pd.PeriodIndex):
                raise AttributeError("weights_df.index must be Pandas.PeriodIndex")
            if self.baseyear not in self.weights_df.index.year:
                raise IndexError(
                    f"baseyear {self.baseyear} is out of range for weights_df"
                )
        if self.corrections_df is not None:
            if not isinstance(self.corrections_df.index, pd.PeriodIndex):
                raise AttributeError("corrections_df.index must be Pandas.PeriodIndex")
            if self.baseyear not in self.corrections_df.index.year:
                raise IndexError(
                    f"baseyear {self.baseyear} is out of range for corrections_df"
                )
            if self.corrections_df.index.freq != self.indicators_df.index.freq:
                raise IndexError(
                    "corrections_df does not have the same frequency as indicators_df"
                )
