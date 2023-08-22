##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

import pandas as pd
import datetime
from pre_system.formula import Formula


class PreSystem:
    def __init__(self, name):
        self._name = name
        self._baseyear = None
        self._formulae = {}
        self._annual_df = None
        self._indicator_df = None
        self._weight_df = None
        self._correction_df = None
        self._annual_df_updated = None
        self._indicator_df_updated = None
        self._weight_df_updated = None
        self._correction_df_updated = None

    @property
    def name(self):
        return self._name

    @property
    def baseyear(self):
        return self._baseyear

    @property
    def formulae(self):
        return self._formulae

    @property
    def indicators(self):
        indicator_set = set()
        for _, formula in self.formulae.items():
            indicator_set = indicator_set.union(formula.indicators)
        return list(indicator_set)

    @property
    def annual_df(self):
        return self._annual_df

    @property
    def indicator_df(self):
        return self._indicator_df

    @property
    def weight_df(self):
        return self._weight_df

    @property
    def correction_df(self):
        return self._correction_df

    @baseyear.setter
    def baseyear(self, baseyear):
        if isinstance(baseyear, int) is False:
            raise TypeError('baseyear must be int')
        for _, formula in self.formulae.items():
            formula.baseyear = baseyear
        self._baseyear = baseyear

    @annual_df.setter
    def annual_df(self, annual_df):
        """
        Set the DataFrame containing annual data.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The DataFrame containing annual data.

        Raises
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex or has incorrect frequency.
        """
        if isinstance(annual_df, pd.DataFrame) is False:
            raise TypeError('annual_df must be a Pandas.DataFrame')
        if isinstance(annual_df.index, pd.PeriodIndex) is False:
            raise AttributeError('annual_df.index must be Pandas.PeriodIndex')
        if annual_df.index.freq != 'a':
            raise AttributeError('annual_df must have annual frequency')
        self._annual_df = annual_df
        self._annual_df_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @indicator_df.setter
    def indicator_df(self, indicator_df):
        """
        Set the DataFrame containing indicator data.

        Parameters
        ----------
        indicator_df : pd.DataFrame
            The DataFrame containing indicator data.

        Raises
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex.
        """
        if isinstance(indicator_df, pd.DataFrame) is False:
            raise TypeError('indicator_df must be a Pandas.DataFrame')
        if isinstance(indicator_df.index, pd.PeriodIndex) is False:
            raise AttributeError('indicators_df.index must be Pandas.PeriodIndex')
        self._indicator_df = indicator_df
        self._indicator_df_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @weight_df.setter
    def weight_df(self, weight_df):
        """
        Set the DataFrame containing weight data.

        Parameters
        ----------
        weight_df : pd.DataFrame
            The DataFrame containing weight data.

        Raises
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex or has incorrect frequency.
        """
        if isinstance(weight_df, pd.DataFrame) is False:
            raise TypeError('weight_df must be a Pandas.DataFrame')
        if isinstance(weight_df.index, pd.PeriodIndex) is False:
            raise AttributeError('weights_df.index must be Pandas.PeriodIndex')
        if weight_df.index.freq != 'a':
            raise AttributeError('weights_df must have annual frequency')
        self._weight_df = weight_df
        self._weight_df_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @correction_df.setter
    def correction_df(self, correction_df):
        """
        Set the DataFrame containing correction data.

        Parameters
        ----------
        correction_df : pd.DataFrame
            The DataFrame containing correction data.

        Raises
        ------
        TypeError
            If the assigned value is not a Pandas DataFrame.
        AttributeError
            If the index of the DataFrame is not a PeriodIndex.
        """
        if isinstance(correction_df, pd.DataFrame) is False:
            raise TypeError('correction_df must be a Pandas.DataFrame')
        if isinstance(correction_df.index, pd.PeriodIndex) is False:
            raise AttributeError('correction_df.index must be Pandas.PeriodIndex')
        self._correction_df = correction_df
        self._correction_df_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def __repr__(self):
        return f'PreSystem: {self.name}'

    def info(self):
        print(
            '\n'.join([
            f'PreSystem {self.name} consists of {len(self.formulae)} formulae.',
            '',
            f'Baseyear is {self._baseyear}.',
            '',
            'DataFrames updated:',
            f'annual_df       {self._annual_df_updated}',
            f'indicator_df    {self._indicator_df_updated}',
            f'weight_df       {self._weight_df_updated} (optional)',
            f'correction_df   {self._correction_df_updated} (optional)'])
        )

    def add_formula(self, formula):
        """
        Add a formula to the PreSystem.

        Parameters
        ----------
        formula : Formula
            The Formula object to be added.

        Raises
        ------
        TypeError
            If formula is not of type Formula.
        KeyError
            If any of the dependencies of the formula are not registered.
        NameError
            If a formula with the same name already exists and points to a different formula.
        """
        if isinstance(formula, Formula) is False:
            raise TypeError('formula must be of type Formula')

        if all(self.formulae.get(x) for x in formula.calls_on) is False:
            raise KeyError(
                f'Register all dependencies: {",".join(formula.calls_on)}'
            )

        existing_formula = self.formulae.get(formula.name)

        if existing_formula is None:
            self.formulae[formula.name] = formula
        else:
            if formula is not existing_formula:
                raise NameError(
                    f'Formula name {formula.name} already exists and points to a different formula'
                )

    def formula(self, name):
        """
        Get a formula from the PreSystem.

        Parameters
        ----------
        formula_name : str
            The name of the formula to retrieve.

        Returns
        -------
        Formula or None
            The requested formula, or None if it doesn't exist.
        """
        return self.formulae.get(name)

    @property
    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate all registered formulas using the provided data.

        Returns
        -------
        pd.DataFrame
            The evaluated formulas as a DataFrame.
        """
        evaluate_df = pd.DataFrame()

        for _, formula in self.formulae.items():
            if formula.baseyear != self.baseyear:
                raise AttributeError(f'baseyear for formula {formula.name} is not {self.baseyear}')

        evaluate_df = pd.DataFrame()

        for name, formula in self.formulae.items():
            evaluate_df = pd.concat(
                [
                    evaluate_df,
                    pd.DataFrame(
                        formula
                        .evaluate(
                            self._annual_df,
                            self._indicator_df,
                            self._weight_df,
                            self._correction_df),
                        columns=[name]
                    )
                ],
                axis=1
            )

        return evaluate_df

    def evaluate_formula(self, name: str) -> pd.Series:
        """
        Evaluate a specific formula using the provided data.

        Parameters
        ----------
        formula_name : str
            The name of the formula to evaluate.

        Returns
        -------
        pd.Series
            The evaluated formula as a Series.
        """
        formula = self.formulae.get(name)

        if formula is not None:
            return (
                self
                ._formulae.get(name)
                .evaluate(
                    self._annual_df,
                    self._indicator_df,
                    self._weight_df,
                    self._correction_df
                    )
            )
        else:
            raise NameError(f'formula {name} is not in PreSystem {self.name}')
