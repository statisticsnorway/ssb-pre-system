import warnings
from typing import Literal

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)
np.set_printoptions(suppress=True)


def minm4(
    mnr: pd.DataFrame,
    rea: pd.DataFrame,
    liste_m4: list[str] | str,
    basisaar: int,
    startaar: int,
    freq: Literal["M", "Q"] = "M",
) -> pd.DataFrame:
    """Perform benchmarking of monthly or quarterly data against yearly data using MinD4 method.

    This function takes two sets of data (monthly/quarterly and yearly), validates them,
    and applies the MinD4 method to "benchmark" or adjust the input data accordingly.
    It ensures consistency between the series in terms of data structure, formats,
    and numerical properties.

    Author: Vemund Rundberget, Seksjon for makroøkonomi, Forksningsavdelingen, SSB

    Args:
        mnr: DataFrame containing monthly or quarterly data to be benchmarked.
            The index must be a pandas PeriodIndex.
        rea: DataFrame containing yearly data for benchmarking.
            The index must be a pandas PeriodIndex.
        liste_m4: A list of series (column names from the DataFrame)
            to be benchmarked, or a single series as a string.
        basisaar: The final year for benchmarking (benchmarking range end).
        startaar: The start year for benchmarking (benchmarking range start).
        freq: Frequency of the data. Use "M" for monthly
            and "Q" for quarterly. Defaults to "M".

    Returns:
        A DataFrame with the benchmarked values for the specified series in `liste_m4`.

    Raises:
        TypeError: If the input parameters or dataframes do not meet the expected types,
            structures, or contents.
    """
    res_dict = {}

    if not (freq == "M" or freq == "Q"):
        raise TypeError('The frequency setting must me either "M" or "Q".')
    if freq == "M":
        freq_ = 12
        periodely = "monthly"
    if freq == "Q":
        freq_ = 4
        periodely = "quarterly"

    # CHECKS.
    # Checking object type.
    if not isinstance(liste_m4, (list, str)):
        raise TypeError(
            "You need to create a list of all the series you wish to benchmark, and it must be in the form of a list or string."
        )
    if isinstance(liste_m4, str):
        liste_m4 = [liste_m4]

    # Checking the monthly DF.
    if not isinstance(mnr, pd.DataFrame):
        raise TypeError(f"The {periodely} dataframe is not a DataFrame.")
    if not isinstance(mnr.index, pd.PeriodIndex):
        raise TypeError(f"The {periodely} dataframe does not have a pd.PeriodIndex.")
    if not pd.Series(liste_m4).isin(mnr.columns).all():
        raise TypeError(
            f"{np.setdiff1d(liste_m4, mnr.columns).tolist()} are missing in the {periodely} dataframe."
        )

    # Filters out series not sent to benchmarking.
    mnr_of_concern = mnr[mnr.columns[mnr.columns.isin(liste_m4)]]
    mask_mnr = (mnr_of_concern.index.year <= basisaar) & (  # type: ignore[attr-defined]
        mnr_of_concern.index.year >= startaar  # type: ignore[attr-defined]
    )
    mnr_of_concern = mnr_of_concern.loc[mask_mnr, :]

    if mnr_of_concern.isna().any().any() is np.True_:
        warnings.warn(
            f"There are NaN-values in {mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list()} in the {periodely} dataframe. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )

    mnrzerowarnlist = []
    for col in mnr_of_concern.columns:
        if (mnr_of_concern[col] == 0).all():
            mnrzerowarnlist.append(f"{col}")
    if len(mnrzerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {mnrzerowarnlist} in the monthly dataframe.",
            UserWarning,
            stacklevel=2,
        )

    mnrintwarnlist = []
    for col in mnr_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(mnr_of_concern[col])
            and col not in mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list()
        ):
            mnrintwarnlist.append(f"{col}")
    if len(mnrintwarnlist) > 0:
        warnings.warn(
            f"There are values in {mnrintwarnlist} in the {periodely} dataframe that are not real numbers. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )

    # Checking the yearly DF.
    if not isinstance(rea, pd.DataFrame):
        raise TypeError("The yearly dataframe is not a DataFrame.")
    if not isinstance(rea.index, pd.PeriodIndex):
        raise TypeError("The yearly dataframe does not have a pd.PeriodIndex.")
    if not pd.Series(liste_m4).isin(rea.columns).all():
        raise TypeError(
            f"{np.setdiff1d(liste_m4, rea.columns).tolist()} are missing in the yearly dataframe."
        )

    # Filters out series not sent to benchmarking.
    rea_of_concern = rea[rea.columns[rea.columns.isin(liste_m4)]]
    mask_rea = (rea_of_concern.index.year <= basisaar) & (  # type: ignore[attr-defined]
        rea_of_concern.index.year >= startaar  # type: ignore[attr-defined]
    )
    rea_of_concern = rea_of_concern.loc[mask_rea, :]

    if rea_of_concern.isna().any().any() is np.True_:
        warnings.warn(
            f"There are NaN-values in {rea_of_concern.columns[rea_of_concern.isna().any()].to_list()} in the yearly dataframe. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )

    reazerowarnlist = []
    for col in rea_of_concern.columns:
        if (rea_of_concern[col] == 0).all():
            reazerowarnlist.append(f"{col}")
    if len(reazerowarnlist) > 0:
        warnings.warn(
            f"There are only zeroes in {reazerowarnlist} in the yearly dataframe.",
            UserWarning,
            stacklevel=2,
        )

    reaintwarnlist = []
    for col in rea_of_concern.columns:
        if (
            not pd.api.types.is_any_real_numeric_dtype(rea_of_concern[col])
            and col not in rea_of_concern.columns[rea_of_concern.isna().any()].to_list()
        ):
            reaintwarnlist.append(f"{col}")
    if len(reaintwarnlist) > 0:
        warnings.warn(
            f"There are values in {reaintwarnlist} in the yearly dataframe that are not real numbers. Skipping sending these to benchmarking.",
            UserWarning,
            stacklevel=2,
        )
    # reaintwarnlist
    rea_of_concern = rea_of_concern[
        [col for col in rea_of_concern.columns if col not in reaintwarnlist]
    ]
    mnr_of_concern = mnr_of_concern[
        [col for col in mnr_of_concern.columns if col not in reaintwarnlist]
    ]
    # mnrintwarnlist
    mnr_of_concern = mnr_of_concern[
        [col for col in mnr_of_concern.columns if col not in mnrintwarnlist]
    ]
    rea_of_concern = rea_of_concern[
        [col for col in rea_of_concern.columns if col not in mnrintwarnlist]
    ]
    liste_m4 = [
        serie
        for serie in liste_m4
        if serie
        not in (
            reaintwarnlist
            + mnrintwarnlist
            + mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list()
            + rea_of_concern.columns[rea_of_concern.isna().any()].to_list()
        )
    ]

    if (
        not set(mnr_of_concern.index.year.unique()).difference(
            set(rea_of_concern.index.year)
        )
        == set()
    ):
        raise TypeError(
            f"There aren't values in both series for {set(mnr_of_concern.index.year.unique()).difference(set(rea_of_concern.index.year))}."
        )

    if not isinstance(startaar, int):
        raise TypeError("The start year must be an integer.")
    if not isinstance(basisaar, int):
        raise TypeError("The final year must be an integer.")
    if basisaar >= 2050:
        raise TypeError(
            "The final year must be less than 2050. Are you sure you entered it correctly?."
        )
    if basisaar < startaar:
        raise TypeError("The start year cannot be greater than the final year.")

    printlist = []

    for elem in liste_m4:
        printlist.append(f"{elem}")
        print(
            f"Benchmarking {printlist} with MinD4 from {startaar} to {basisaar}.",
            end="\r",
            flush=True,
        )

        # Defines the monthly and yearly values.
        datam_ = mnr_of_concern[elem].values
        datay_ = rea_of_concern[elem].values
        datam = np.hstack([np.asarray(datam_)])
        datay = np.hstack([np.asarray(datay_)])

        # Counting months and years.
        nm = datam.shape[0]
        ny = datay.shape[0]

        # Setting up submatrices a for A and -vectors x for X consisting of zeros.
        a1 = np.zeros((nm, nm), dtype=np.float64)
        a2 = np.zeros((nm, ny), dtype=np.float64)
        a3 = np.zeros((ny, nm), dtype=np.float64)
        a4 = np.zeros((ny, ny), dtype=np.float64)
        x1 = np.zeros((nm, 1), dtype=np.float64)
        x2 = np.zeros((ny, 1), dtype=np.float64)

        # Fills in submatrices a into A and -vectors x into X according to Skjæveland 1985, pages 18 and 19 (modified to benchmark by month instead of quarter)
        for i in range(0, nm):
            a1[i, i] = (1 + 1 * (i > 0 and i < nm - 1)) / datam[i] ** 2
            if i > 0:
                a1[i, i - 1] = -1 / (datam[i] * datam[i - 1])
            if i < nm - 1:
                a1[i, i + 1] = -1 / (datam[i] * datam[i + 1])

        for i in range(0, nm):
            for j in range(0, ny):
                if i / freq_ >= j and i / freq_ < j + 1:
                    a2[i, j] = 1
                    a3[j, i] = 1
                x2[j] = datay[j]

        # Combines submatrices a into A and -vectors x into X according to Skjæveland 1985, pages 18 and 19 (modified to reconcile by month instead of quarter)
        A = np.vstack((np.hstack((a1, a2)), np.hstack((a3, a4))))
        X = np.vstack((x1, x2))

        # Solves the equation system AY=X for Y if possible.
        try:
            Y = np.linalg.solve(A, X)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Wasn't able to benchmark {elem}.", UserWarning, stacklevel=2
            )

        res_dict[elem] = Y[0:nm].flatten()

    res = pd.DataFrame(res_dict, index=mnr_of_concern.index)

    # Removing resulting NaN series from the res-DataFrame.
    res = res[res.columns[~res.isna().any()]]

    # Checks that the deviation on the totals after benchmarking is zero for the series in liste_m4.
    # Skips the series that already trigger error messages in the input.
    skippe = (
        mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list()
        + rea_of_concern.columns[rea_of_concern.isna().any()].to_list()
    )
    for elem in set(liste_m4) - set(skippe):
        if (
            ((res.resample("Y").sum() - rea_of_concern) >= -1)
            & ((res.resample("Y").sum() - rea_of_concern) <= 1)
        ).all()[elem] is not np.True_:
            warnings.warn(
                f"There are deviations on the benchmarked totals in {elem} so something did not go well.",
                UserWarning,
                stacklevel=2,
            )

    print("\nBenchmarking with MinM4 done!")

    return res
