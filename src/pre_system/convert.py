##################################
# Author: Magnus Kvåle Helliesen #
# mkh@ssb.no                     #
##################################

import numpy as np
import pandas as pd


def convert(input_df: pd.DataFrame, to_freq: str) -> pd.DataFrame:
    """Upsamples or downsamples a DataFrame with a PeriodIndex to the specified frequency.

    Parameters
    ----------
    input_df : pd.DataFrame
        The input DataFrame with a PeriodIndex to be converted.
    to_freq : str
        The target frequency to which the DataFrame will be converted.
        Valid options: 'A' (annual), 'Q' (quarterly), or 'M' (monthly).

    Returns:
    -------
    pd.DataFrame
        The converted DataFrame with the specified frequency.

    Raises:
    ------
    TypeError
        If the DataFrame does not have a PeriodIndex.
    ValueError
        If the conversion is not possible for the given input frequency.

    Notes:
    -----
    - The conversion is performed by resampling the input DataFrame using the sum aggregation function.
    - If the target frequency is lower or equal to the input frequency, the conversion is done using resampling only.
    - If the target frequency is higher, a first-order conditions matrix is constructed and solved to fill missing values.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a DataFrame")

    if not isinstance(input_df.index, pd.PeriodIndex):
        raise TypeError("DataFrame must have PeriodIndex")

    if not all(np.issubdtype(input_df[x].dtype, np.number) for x in input_df.columns):  # type: ignore [arg-type]
        raise TypeError("All columns in input_df must be numeric")

    freq_to_periods_per_year = {"a": 1, "q": 4, "m": 12}
    input_periods_per_year = freq_to_periods_per_year.get(
        input_df.index.freqstr[0].lower()
    )
    output_periods_per_year = freq_to_periods_per_year.get(to_freq.lower())

    if input_periods_per_year is None:
        raise ValueError(
            f"cannot convert from frequency {input_df.index.freqstr[0].lower()}"
        )

    if output_periods_per_year is None:
        raise ValueError(f"cannot convert to frequency {to_freq.lower()}")

    output_df = input_df.resample(to_freq).sum()

    # If output_df is being downsampled, return output_df as is
    if output_periods_per_year <= input_periods_per_year:
        return output_df

    # If output_df is being upsampled, setup to solve the first order conditions
    input_matrix = input_df.to_numpy()
    output_matrix = output_df.to_numpy()

    # Determine number of periods in input_df and output_df and number of series
    oi_ratio = output_periods_per_year / input_periods_per_year
    (ni, ns) = input_matrix.shape
    no = int(ni * oi_ratio)

    # Construct first order condition matrix (as a block matrix of four bocks)
    # See doc TBA for documentation of matrix
    A11 = np.zeros((no, no))
    A12 = np.zeros((no, ni))
    A21 = A12.T
    A22 = np.zeros((ni, ni))

    # Build sub matrix A11
    for i in range(2, no - 2):
        A11[i, i - 2 : i + 3] = [1, -4, 6, -4, 1]

    A11[0, :3] = [1, -2, 1]
    A11[1, :4] = [-2, 5, -4, 1]
    A11[-1, -3:] = [1, -2, 1]
    A11[-2, -4:] = [1, -4, 5, -2]

    # Build submatrix A12 (A21 follows as the transpose of A12)
    for i in range(no):
        A12[i, int(i / oi_ratio)] = 1

    # Stack matrices and vectors
    A = np.vstack((np.hstack((A11, A12)), np.hstack((A21, A22))))
    X = np.vstack((np.zeros((no, ns)), input_matrix))

    # Solve for solutions to the first order conditons
    solution = np.linalg.solve(A, X)

    # Store the relevant elements of solution matrix to output_matrix
    output_matrix[:no, :] = solution[:-ni, :]

    return output_df


def convert_step(input_df: pd.DataFrame, to_freq: str) -> pd.DataFrame:
    if not isinstance(input_df.index, pd.PeriodIndex):
        raise TypeError("DataFrame must have PeriodIndex")

    if not all(np.issubdtype(input_df[x].dtype, np.number) for x in input_df.columns):  # type: ignore [arg-type]
        raise TypeError("All columns in input_df must be numeric")

    freq_to_periods_per_year = {"a": 1, "q": 4, "m": 12}
    input_periods_per_year = freq_to_periods_per_year.get(
        input_df.index.freqstr[0].lower()
    )
    output_periods_per_year = freq_to_periods_per_year.get(to_freq.lower())

    if input_periods_per_year is None:
        raise ValueError(
            f"cannot convert from frequency {input_df.index.freqstr[0].lower()}"
        )

    if output_periods_per_year is None:
        raise ValueError(f"cannot convert to frequency {to_freq.lower()}")

    output_df = input_df.resample(to_freq).sum()

    # If output_df is being downsampled, return output_df as is
    if output_periods_per_year <= input_periods_per_year:
        return output_df

    input_matrix = input_df.to_numpy()
    output_matrix = output_df.to_numpy()

    # Determine number of periods in input_df and output_df and number of series
    oi_ratio = output_periods_per_year / input_periods_per_year
    ni = input_matrix.shape[0]
    no = int(ni * oi_ratio)

    A = np.zeros((no, ni))

    for i in range(ni):
        A[int(i * oi_ratio) : int((i + 1) * oi_ratio), i] = 1 / oi_ratio

    output_matrix[:, :] = np.matmul(A, input_matrix)

    return output_df
