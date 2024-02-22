##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

import pandas as pd
import numpy as np


class Formula:
    _baseyear = None

    def __init__(self, name):
        """
        Initialize a Formula instance.

        Parameters
        ----------
        name : str
            The name of the formula.

        Raises
        ------
        TypeError
            If `name` is not a string.
        """

        if isinstance(name, str) is False:
            raise TypeError('name must be str')
        self._name = name.lower()
        self._baseyear = None
        self._calls_on = None

    @property
    def name(self):
        return self._name

    @property
    def baseyear(self):
        return self._baseyear

    @property
    def what(self):
        return None

    @property
    def calls_on(self):
        return self._calls_on

    @property
    def indicators(self):
        return []

    @property
    def weights(self):
        return []

    @baseyear.setter
    def baseyear(self, baseyear):
        if isinstance(baseyear, int) is False:
            raise TypeError('baseyear must be int')
        self._baseyear = baseyear

    def __repr__(self):
        return f'Formula: {self.name} = {self.what}'

    def __call__(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None):
        return self.evaluate(annual_df,
                             indicators_df,
                             weights_df,
                             correction_df)

    def __add__(self, other):
        return FSum(f'{self.name}+{other.name}', self, other)

    def __mul__(self, other):
        return FMult(f'{self.name}*{other.name}', self, other)

    def __truediv__(self, other):
        return FDiv(f'{self.name}/{other.name}', self, other)

    def info(self, i=0):
        what = self.what if len(self.what) <= 100 else '...'
        print(f'{" "*i}{self.name} = {what}')
        for _, val in self.calls_on.items():
            val.info(i+1)

    def indicators_weights(self, trace=True):
        return []

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the formula using the provided data.

        Parameters
        ----------
        annual_df : pd.DataFrame
            The annual data used for evaluation.
        indicators_df : pd.DataFrame
            The indicator data used for evaluation.
        weights_df : pd.DataFrame, optional
            The weight data used for evaluation. Defaults to None.
        correction_df : pd.DataFrame, optional
            The correction data used for evaluation. Defaults to None.

        Raises
        ------
        ValueError
            If the base year is not set or is out of range for the provided data.
        TypeError
            If any of the input data is not a Pandas DataFrame.
        AttributeError
            If the index of any input DataFrame is not a Pandas PeriodIndex or if the frequency is incorrect.
        IndexError
            If the base year is out of range for any input DataFrame.
        """

        if self.baseyear is None:
            raise ValueError('baseyear is None')

        if test_dfs:
            self._check_df('annual_df', annual_df, self.baseyear, 'a')
            self._check_df('indicators_df', indicators_df, self.baseyear)

            if weights_df is not None:
                self._check_df('weights_df', weights_df, self.baseyear, 'a')

            if correction_df is not None:
                self._check_df('correction_df', correction_df, self.baseyear, indicators_df.index.freq)

    # Method that checks that conditions are met for DataFrame to be valid input
    @staticmethod
    def _check_df(df_name, df, baseyear, frequency=None):
        if isinstance(df, pd.DataFrame) is False:
            raise TypeError(f'{df_name} must be a Pandas.DataFrame')
        if isinstance(df.index, pd.PeriodIndex) is False:
            raise AttributeError(f'{df_name}.index must be Pandas.PeriodIndex')
        if frequency and df.index.freq != frequency:
            raise AttributeError(f'{df_name} must have frequency {frequency}')
        if df[df.index.year == baseyear].shape[0] == 0:
            raise IndexError(f'baseyear {baseyear} is out of range for annual_df')
        if all(np.issubdtype(df[x].dtype, np.number) for x in df.columns) is False:
            raise TypeError(f'All columns in {df_name} must be numeric')


class Indicator(Formula):
    def __init__(self,
                 name: str,
                 annual: str,
                 indicators: list[str],
                 weights: list[str] | list[float] = None,
                 correction: str = None,
                 normalise=False,
                 aggregation: str = 'sum'):
        """
        Initialize an Indicator object.

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

        Raises
        ------
        IndexError
            If `weight_names` is provided and has a different length than `indicator_names`.
        """

        super().__init__(name)
        if isinstance(annual, str) is False:
            raise TypeError('annual must be str')
        if isinstance(indicators, list) is False:
            raise TypeError('indicator_names must be a list')
        if all(isinstance(x, str) for x in indicators) is False:
            raise TypeError('indicator_names must containt str')
        if weights and len(weights) != len(indicators):
            raise IndexError('weight_names must have same length as indicator_names')
        self._annual = annual
        self._indicators = [x.strip() for x in indicators]
        if weights:
            if all(isinstance(x, type(weights[0])) for x in weights) is False:
                raise TypeError('all weights must be of same type')
        self._weights = weights
        self._correction = correction
        self._normalise = normalise
        if aggregation.lower() not in ['sum', 'avg']:
            raise NameError('aggregation must be sum or avg')
        self._aggregation = aggregation.lower()
        self._calls_on = {}

    @property
    def indicators(self):
        return self._indicators

    @property
    def weights(self):
        if self._weights:
            return self._weights
        return [1 for _ in self.indicators]

    @property
    def what(self):
        correction = f'{self._correction}*' if self._correction else ''

        if self._normalise:
            indicators = [f'{x}/sum({x}<date {self.baseyear}>)' for x in self._indicators]
        else:
            indicators = self._indicators

        if self._weights:
            aggregated_indicators = (
                '+'.join(['*'.join([str(x).lower(), y.lower()]) for x, y in
                          zip(self._weights, indicators)])
            )
        else:
            aggregated_indicators = (
                '+'.join([x.lower() for x in indicators])
            )

        numerator = f'{correction}({aggregated_indicators})'
        denominator = f'{self._aggregation}({numerator}<date {self.baseyear}>)'
        fraction = f'{numerator}/{denominator}'

        return (
            f'{self._annual.lower()}*<date {self.baseyear}>*{fraction}'
        )

    def indicators_weights(self, trace=True):
        return [(x, y) for x, y in zip(self.indicators, self.weights)]

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        super().evaluate(annual_df,
                         indicators_df,
                         weights_df,
                         correction_df,
                         test_dfs=test_dfs)

        if (self._annual in annual_df.columns) is False:
            raise NameError(f'Cannot find {self._annual} in annual_df')

        if all(x in indicators_df.columns for x in self._indicators) is False:
            missing = [x for x in self._indicators if x not in indicators_df.columns]
            raise NameError(f'Cannot find {",".join(missing)} in indicators_df')

        indicator_matrix = indicators_df.loc[:, self._indicators]
        if self._normalise:
            indicator_matrix = (
                indicator_matrix.div(
                    indicator_matrix
                    .loc[
                        indicator_matrix.index.year == self.baseyear
                    ].sum()
                )
            )

        if self._weights:
            if all(isinstance(x, str) for x in self._weights):
                if weights_df is None:
                    raise NameError(f'{self.name} expects weights_df')
                if all(x in weights_df.columns for x in self._weights) is False:
                    missing = [x for x in self._weights if x not in weights_df.columns]
                    raise NameError(f'Cannot find {",".join(missing)} in weights_df')

            indicator_matrix = indicator_matrix.to_numpy()

            if all(isinstance(x, str) for x in self._weights):
                weight_vector = (
                    weights_df
                    .loc[weights_df.index.year == self.baseyear, self._weights]
                    .to_numpy()
                )
            if all(isinstance(x, float) for x in self._weights):
                weight_vector = np.array([self._weights])

            weighted_indicators = pd.Series(
                indicator_matrix.dot(weight_vector.transpose())[:, 0],
                index=indicators_df.index
            )
        else:
            weighted_indicators = indicator_matrix.sum(axis=1, skipna=False)

        if self._correction:
            if correction_df is None:
                raise NameError(f'{self.name} expects correction_df')
            if (self._correction in correction_df.columns) is False:
                raise NameError(f'{self._correction} is not in correction_df')
            corrected_indicators = weighted_indicators*correction_df.loc[:, self._correction]
        else:
            corrected_indicators = weighted_indicators

        evaluated_series = (
            annual_df.loc[annual_df.index.year == self.baseyear, self._annual].to_numpy()
            * corrected_indicators.div(
                corrected_indicators
                .loc[
                    corrected_indicators.index.year == self.baseyear
                ].sum()
                if self._aggregation == 'sum' else
                corrected_indicators
                .loc[
                    corrected_indicators.index.year == self.baseyear
                ].mean()
            )
        )

        return evaluated_series


class FDeflate(Formula):
    def __init__(self,
                 name: str,
                 formula: Formula,
                 indicators: list[str],
                 weights: list[str] | list[float] = None,
                 correction: str = None,
                 normalise=False):
        """
        Initialize an FDeflate object.

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

        Raises
        ------
        TypeError
            If `formula` is not of type Formula.
        IndexError
            If `weight_names` is provided and has a different length than `indicator_names`.
        """

        super().__init__(name)
        if isinstance(formula, Formula) is False:
            raise TypeError('formula must be of type Formula')
        if weights and len(weights) != len(indicators):
            raise IndexError('weight_names must have same length as indicator_names')
        self._formula = formula
        self._indicators = [x.strip() for x in indicators]
        if weights:
            if all(isinstance(x, type(weights[0])) for x in weights) is False:
                raise TypeError('all weights must be of same type')
        self._weights = weights
        self._correction = correction
        self._normalise = normalise
        self._calls_on = {formula.name: formula}

    @property
    def indicators(self):
        return self._indicators

    @property
    def weights(self):
        if self._weights:
            return self._weights
        return [1 for _ in self.indicators]

    @property
    def what(self):
        correction = f'{self._correction}*' if self._correction else ''

        if self._normalise:
            indicators = [f'{x}/sum({x}<date {self.baseyear}>)' for x in self._indicators]
        else:
            indicators = self._indicators

        if self._weights:
            aggregated_indicators = (
                '+'.join(['*'.join([str(x).lower(), y.lower()]) for x, y in
                          zip(self._weights, indicators)])
            )
        else:
            aggregated_indicators = (
                '+'.join([x.lower() for x in indicators])
            )

        numerator = f'{correction}{self._formula.name}/({aggregated_indicators})'
        denominator = f'sum({numerator}<date {self.baseyear}>)'
        fraction = f'({numerator})/{denominator}'

        return (
            f'sum({self._formula.name}<date {self.baseyear}>)*{fraction}'
        )

    def indicators_weights(self, trace=True):
        return (
            [(x, y) for x, y in zip(self.indicators, self.weights)]
            +(self._formula.indicators_weights(trace=trace) if trace else [])
        )

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if all(x in indicators_df.columns for x in self._indicators) is False:
            raise NameError(f'All of {",".join(self._indicators)} is not in indicators_df')

        indicator_matrix = indicators_df.loc[:, self._indicators]
        if self._normalise:
            indicator_matrix = (
                indicator_matrix.div(
                    indicator_matrix
                    .loc[
                        indicator_matrix.index.year == self.baseyear
                    ].sum()
                )
            )

        if self._weights:
            if all(isinstance(x, str) for x in self._weights):
                if weights_df is None:
                    raise NameError(f'{self.name} expects weights_df')
                if all(x in weights_df.columns for x in self._weights) is False:
                    missing = [x for x in self._weights if x not in weights_df.columns]
                    raise NameError(f'Cannot find {",".join(missing)} in weights_df')

            indicator_matrix = indicator_matrix.to_numpy()

            if all(isinstance(x, str) for x in self._weights):
                weight_vector = (
                    weights_df
                    .loc[weights_df.index.year == self.baseyear, self._weights]
                    .to_numpy()
                )
            if all(isinstance(x, float) for x in self._weights):
                weight_vector = np.array([self._weights])

            weighted_indicators = pd.Series(
                indicator_matrix.dot(weight_vector.transpose())[:, 0],
                index=indicators_df.index
            )
        else:
            weighted_indicators = indicator_matrix.sum(axis=1, skipna=False)

        evaluated_formula = self._formula.evaluate(*all_dfs, test_dfs=test_dfs)

        formula_divided = evaluated_formula.div(weighted_indicators)

        if self._correction:
            if correction_df is None:
                raise NameError(f'{self.name} expects correction_df')
            if (self._correction in correction_df.columns) is False:
                raise NameError(f'{self._correction} is not in correction_df')
            formula_corrected = formula_divided*correction_df.loc[:, self._correction]
        else:
            formula_corrected = formula_divided

        evaluated_series = (
            evaluated_formula
            .loc[evaluated_formula.index.year == self.baseyear]
            .sum()
            * formula_corrected.div(
                formula_corrected
                .loc[
                    formula_corrected.index.year == self.baseyear
                ].sum()
            )
        )

        return evaluated_series


class FInflate(Formula):
    def __init__(self,
                 name: str,
                 formula: Formula,
                 indicators: list[str],
                 weights: list[str] | list[float] = None,
                 correction: str = None,
                 normalise=False):
        """
        Initialize an FInflate object.

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

        Raises
        ------
        TypeError
            If `formula` is not of type Formula.
        IndexError
            If `weight_names` is provided and has a different length than `indicator_names`.
        """

        super().__init__(name)
        if isinstance(formula, Formula) is False:
            raise TypeError('formula must be of type Formula')
        if weights and len(weights) != len(indicators):
            raise IndexError('weight_names must have same length as indicator_names')
        self._formula = formula
        self._indicators = [x.strip() for x in indicators]
        if weights:
            if all(isinstance(x, type(weights[0])) for x in weights) is False:
                raise TypeError('all weights must be of same type')
        self._weights = weights
        self._correction = correction
        self._normalise = normalise
        self._calls_on = {formula.name: formula}

    @property
    def indicators(self):
        return self._indicators

    @property
    def weights(self):
        if self._weights:
            return self._weights
        return [1 for _ in self.indicators]

    @property
    def what(self):
        correction = f'{self._correction}*' if self._correction else ''

        if self._normalise:
            indicators = [f'{x}/sum({x}<date {self.baseyear}>)' for x in self._indicators]
        else:
            indicators = self._indicators

        if self._weights:
            aggregated_indicators = (
                '+'.join(['*'.join([str(x).lower(), y.lower()]) for x, y in
                          zip(self._weights, indicators)])
            )
        else:
            aggregated_indicators = (
                '+'.join([x.lower() for x in indicators])
            )

        numerator = f'{correction}{self._formula.name}*({aggregated_indicators})'
        denominator = f'sum({numerator}<date {self.baseyear}>)'
        fraction = f'({numerator})/{denominator}'

        return (
            f'sum({self._formula.name}<date {self.baseyear}>)*{fraction}'
        )

    def indicators_weights(self, trace=True):
        return (
            [(x, y) for x, y in zip(self.indicators, self.weights)]
            +(self._formula.indicators_weights(trace=trace) if trace else [])
        )

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if all(x in indicators_df.columns for x in self._indicators) is False:
            raise NameError(f'All of {",".join(self._indicators)} is not in indicators_df')

        indicator_matrix = indicators_df.loc[:, self._indicators]
        if self._normalise:
            indicator_matrix = (
                indicator_matrix.div(
                    indicator_matrix
                    .loc[
                        indicator_matrix.index.year == self.baseyear
                    ].sum()
                )
            )

        if self._weights:
            if all(isinstance(x, str) for x in self._weights):
                if weights_df is None:
                    raise NameError(f'{self.name} expects weights_df')
                if all(x in weights_df.columns for x in self._weights) is False:
                    missing = [x for x in self._weights if x not in weights_df.columns]
                    raise NameError(f'Cannot find {",".join(missing)} in weights_df')

            indicator_matrix = indicator_matrix.to_numpy()

            if all(isinstance(x, str) for x in self._weights):
                weight_vector = (
                    weights_df
                    .loc[weights_df.index.year == self.baseyear, self._weights]
                    .to_numpy()
                )
            if all(isinstance(x, float) for x in self._weights):
                weight_vector = np.array([self._weights])

            weighted_indicators = pd.Series(
                indicator_matrix.dot(weight_vector.transpose())[:, 0],
                index=indicators_df.index
            )
        else:
            weighted_indicators = indicator_matrix.sum(axis=1, skipna=False)

        evaluated_formula = self._formula.evaluate(*all_dfs, test_dfs=test_dfs)

        formula_divided = evaluated_formula*weighted_indicators

        if self._correction:
            if correction_df is None:
                raise NameError(f'{self.name} expects correction_df')
            if (self._correction in correction_df.columns) is False:
                raise NameError(f'{self._correction} is not in correction_df')
            formula_corrected = formula_divided*correction_df.loc[:, self._correction]
        else:
            formula_corrected = formula_divided

        evaluated_series = (
            evaluated_formula
            .loc[evaluated_formula.index.year == self.baseyear]
            .sum()
            * formula_corrected.div(
                formula_corrected
                .loc[
                    formula_corrected.index.year == self.baseyear
                ].sum()
            )
        )

        return evaluated_series


class FSum(Formula):
    def __init__(self,
                 name,
                 *formulae: Formula):
        """
        Initialize an FSum object.

        Parameters
        ----------
        name : str
            The name of the FSum object.
        *formulae : Formula
            Variable number of Formula objects.

        Raises
        ------
        TypeError
            If any of the *formulae is not of type Formula.
        """

        super().__init__(name)
        if all(isinstance(x, Formula) for x in formulae) is False:
            raise TypeError('*formulae must be of type Formula')
        self._formulae = formulae
        self._calls_on = {x.name: x for x in formulae}

    @property
    def what(self):
        return '+'.join([x.name for x in self._formulae])

    def indicators_weights(self, trace=True):
        indicators_weights = []
        if trace:
            for formula in self._formulae:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if any(x.evaluate(*all_dfs, test_dfs=test_dfs) is None for x in self._formulae):
            raise ValueError('some of the formulae do not evaluate')

        return sum(
            x.evaluate(*all_dfs, test_dfs=test_dfs)
            for x in self._formulae
        )


class FSumProd(Formula):
    def __init__(self,
                 name,
                 formulae: list[Formula],
                 weights: list[float] | list[str]):
        """
        Initialize an FSumProd object.

        Parameters
        ----------
        name : str
            The name of the FSum object.
        formulae : list[Formula]
            ...
        coefficients : list[float]
            ...

        Raises
        ------
        TypeError
            If any of the *formulae is not of type Formula.
        """

        super().__init__(name)
        if all(isinstance(x, Formula) for x in formulae) is False:
            raise TypeError('*formulae must be of type Formula')
        self._formulae = formulae
        if weights:
            if all(isinstance(x, type(weights[0])) for x in weights) is False:
                raise TypeError('all weights must be of same type')
        self._weights = weights
        self._calls_on = {x.name: x for x in formulae}

    @property
    def what(self):
        return '+'.join(['*'.join([x.name, str(y).lower()]) for x, y in
                         zip(self._formulae, self._weights)])

    def indicators_weights(self, trace=True):
        indicators_weights = []
        if trace:
            for formula in self._formulae:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if any(x.evaluate(*all_dfs, test_dfs=test_dfs) is None for x in self._formulae):
            raise ValueError('some of the formulae do not evaluate')

        if all(isinstance(x, str) for x in self._weights):
            if weights_df is None:
                raise NameError(f'{self.name} expects weights_df')
            if all(x in weights_df.columns for x in self._weights) is False:
                missing = [x for x in self._weights if x not in weights_df.columns]
                raise NameError(f'Cannot find {",".join(missing)} in weights_df')
            weight_vector = (
                weights_df[weights_df.index.year == self.baseyear][self._weights].sum().tolist()
            )
            return sum(
                x.evaluate(*all_dfs, test_dfs=test_dfs)*y
                for x, y in zip(self._formulae, weight_vector)
            )
        if all(isinstance(x, float) for x in self._weights):
            return sum(
                x.evaluate(*all_dfs, test_dfs=test_dfs)*y
                for x, y in zip(self._formulae, self._weights)
            )
        raise TypeError('All weights must be str or float')


class FMult(Formula):
    def __init__(self,
                 name,
                 formula1: Formula,
                 formula2: Formula):
        super().__init__(name)
        if isinstance(formula1, Formula) and isinstance(formula1, Formula) is False:
            raise TypeError('formula1 and formula2 must be of type Formula')
        self._formula1 = formula1
        self._formula2 = formula2
        self._calls_on = {formula1.name: formula1, formula2.name: formula2}

    @property
    def what(self):
        return f'{self._formula1.name}*{self._formula2.name}'

    def indicators_weights(self, trace=True):
        indicators_weights = []
        if trace:
            for formula in [self._formula1, self._formula2]:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if self._formula1.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError(f'formula1 does not evaluate')
        if self._formula2.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError(f'formula2 does not evaluate')

        return (
            self._formula1.evaluate(*all_dfs, test_dfs=test_dfs)
            * self._formula2.evaluate(*all_dfs, test_dfs=test_dfs)
            )


class FDiv(Formula):
    def __init__(self,
                 name,
                 formula1: Formula,
                 formula2: Formula):
        super().__init__(name)
        if isinstance(formula1, Formula) and isinstance(formula1, Formula) is False:
            raise TypeError('formula1 and formula2 must be of type Formula')
        self._formula1 = formula1
        self._formula2 = formula2
        self._calls_on = {formula1.name: formula1, formula2.name: formula2}

    @property
    def what(self):
        return f'{self._formula1.name}/{self._formula2.name}'

    def indicators_weights(self, trace=True):
        indicators_weights = []
        if trace:
            for formula in [self._formula1, self._formula2]:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        if self._formula1.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError(f'formula1 does not evaluate')
        if self._formula2.evaluate(*all_dfs, test_dfs=test_dfs) is None:
            raise ValueError(f'formula2 does not evaluate')

        return (
            self._formula1.evaluate(*all_dfs, test_dfs=test_dfs)
            .div(self._formula2.evaluate(*all_dfs, test_dfs=test_dfs))
            )


class MultCorr(Formula):
    def __init__(self, formula: Formula, correction_name):
        """
        Initialize a MultCorr object.

        Parameters
        ----------
        formula : Formula
            The Formula object to be multiplied by the correction factor.
        correction_name : str
            The name of the correction factor.

        Raises
        ------
        TypeError
            If formula is not of type Formula.
        """

        super().__init__(formula.name)
        if isinstance(formula, Formula) is False:
            raise TypeError('formula must be of type Formula')
        self._formula = formula
        self._correction_name = correction_name
        self._calls_on = formula._calls_on

    @property
    def baseyear(self):
        return self._baseyear

    @baseyear.setter
    def baseyear(self, baseyear):
        if isinstance(baseyear, int) is False:
            raise TypeError('baseyear must be int')
        self._baseyear = baseyear
        # Pass baseyear to formula that goes into correction
        self._formula.baseyear = baseyear

    @property
    def what(self):
        return (
            f'sum(({self._formula.what})<date {self.baseyear}>)*'
            f'{self._correction_name}*({self._formula.what})/'
            f'sum({self._correction_name}*({self._formula.what})<date {self.baseyear}>)'
        )

    def indicators_weights(self, trace=True):
        return self._formula.indicators_weights(trace=trace) if trace else []

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        evaluated_formula = self._formula.evaluate(*all_dfs, test_dfs=test_dfs)

        formula_corrected = evaluated_formula*correction_df[self._correction_name]

        return (
            evaluated_formula[evaluated_formula.index.year == self.baseyear].sum()
            *formula_corrected.div(
                formula_corrected[formula_corrected.index.year == self.baseyear].sum()
            )
        )


class AddCorr(Formula):
    def __init__(self, formula: Formula, correction_name):
        """
        Initialize an AddCorr object.

        Parameters
        ----------
        formula : Formula
            The Formula object to be added with the correction factor.
        correction_name : str
            The name of the correction factor.

        Raises
        ------
        TypeError
            If formula is not of type Formula.
        """

        super().__init__(formula.name)
        if isinstance(formula, Formula) is False:
            raise TypeError('formula must be of type Formula')
        self._formula = formula
        self._correction_name = correction_name
        self._calls_on = formula.calls_on

    @property
    def baseyear(self):
        return self._baseyear

    @baseyear.setter
    def baseyear(self, baseyear):
        if isinstance(baseyear, int) is False:
            raise TypeError('baseyear must be int')
        self._baseyear = baseyear
        # Pass baseyear to formula that goes into correction
        self._formula.baseyear = baseyear

    @property
    def what(self):
        return f'{self._correction_name}+({self._formula.what})-avg({self._correction_name}<date {self.baseyear})'

    def indicators_weights(self, trace=True):
        return self._formula.indicators_weights(trace=trace) if trace else []

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        return (
            correction_df[self._correction_name]
            + self._formula.evaluate(*all_dfs, test_dfs=test_dfs)
            - correction_df[correction_df.index.year == self.baseyear][self._correction_name].mean()
        )


class FJoin(Formula):
    def __init__(self,
                 name: str,
                 formula1: Formula,
                 formula0: Formula,
                 from_year: int):
        """
        Initialize an FJoin object.

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

        Raises
        ------
        TypeError
            If formula1 is not of type Formula.
        TypeError
            If formula0 is not of type Formula.
        TypeError
            If from year is not of type str.
        """

        super().__init__(name)
        if isinstance(formula1, Formula) and isinstance(formula0, Formula) is False:
            raise TypeError('formula1 and formula0 must be of type Formula')
        if isinstance(from_year, int) is False:
            raise TypeError('from_year must must be of type int')
        self._formula1 = formula1
        self._formula0 = formula0
        self._from_year = from_year
        self._calls_on = {formula1.name: formula1, formula0.name: formula0}

    @property
    def indicators(self):
        return list(set(self._formula1.indicators).union(self._formula0.indicators))

    @property
    def what(self):
        return f'{self._formula1.name} if year>={self._from_year} else {self._formula0.name}'

    def indicators_weights(self, trace=True):
        indicators_weights = []
        if trace:
            for formula in [self._formula1, self._formula0]:
                indicators_weights.extend(formula.indicators_weights(trace=trace))
        return indicators_weights

    def evaluate(self,
                 annual_df: pd.DataFrame,
                 indicators_df: pd.DataFrame,
                 weights_df: pd.DataFrame = None,
                 correction_df: pd.DataFrame = None,
                 test_dfs: bool=True
                ) -> pd.Series:
        """
        Evaluate the data using the provided DataFrames and return the evaluated series.

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

        Raises
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

        Returns
        -------
        pd.Series
            The evaluated series.
        """

        all_dfs = (annual_df, indicators_df, weights_df, correction_df)
        super().evaluate(*all_dfs, test_dfs=test_dfs)

        evaluated_formula1 = self._formula1.evaluate(*all_dfs, test_dfs=test_dfs)
        evaluated_formula0 = self._formula0.evaluate(*all_dfs, test_dfs=test_dfs)

        return pd.concat(
            [
                evaluated_formula0[evaluated_formula0.index.year < self._from_year],
                evaluated_formula1[evaluated_formula1.index.year >= self._from_year]
            ]
        )
