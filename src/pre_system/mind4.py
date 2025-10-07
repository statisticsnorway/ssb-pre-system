import warnings
from typing import Literal

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)
np.set_printoptions(suppress=True)


def mind4(
    mnr: pd.DataFrame,
    rea: pd.DataFrame,
    liste_d4: list[str] | str,
    basisaar: int,
    startaar: int,
    freq: Literal["M", "Q"] = "M",
) -> pd.DataFrame:
    """Executes benchmarking of monthly or quarterly data against annual data for specific series, using the MinD4 method.

    The function validates input data types, checks for consistency,
    and addresses edge cases such as missing values or invalid contents. The function
    allows scaling adjustments for leading and trailing periods using specified start and
    basis years, which determine the timeframe of analysis.

    Author: Vemund Rundberget, Seksjon for makroøkonomi, Forksningsavdelingen, SSB

    Args:
        mnr: DataFrame with monthly or quarterly data having a pd.PeriodIndex.
        rea: DataFrame with annual data having a pd.PeriodIndex.
        liste_d4: List or single string of series names to benchmark.
        basisaar: Final year to include in the analysis.
        startaar: Initial year to include in the analysis.
        freq: Frequency of the time series data ('M' for monthly, 'Q' for quarterly).
            Default is "M".

    Returns:
        pd.DataFrame: Dataframe containing the benchmarking results, after MinD4 adjustments.

    Raises:
        TypeError: If freq is not "M" or "Q"; if liste_d4 is not a list or string;
            if mnr/rea are not DataFrames with a PeriodIndex; if required series are
            missing in mnr/rea; if startaar/basisaar are not integers, if basisaar >= 2050,
            if basisaar < startaar, or if there are years present in the monthly/quarterly
            data that are missing in the yearly data.
    """
    res_dict = {}

    if freq not in ["M", "Q"]:
        raise TypeError('The frequency setting must me either "M" or "Q".')
    if freq == "M":
        freq_ = 12
        periodely = "monthly"
    if freq == "Q":
        freq_ = 4
        periodely = "quarterly"

    # CHECKS.
    # Checking object type.
    if not isinstance(liste_d4, (list, str)):
        raise TypeError(
            "You need to create a list of all the series you wish to benchmark, and it must be in the form of a list or string."
        )

    if isinstance(liste_d4, str):
        liste_d4 = [liste_d4]

    # Checking the monthly/quarterly DF.
    if not isinstance(mnr, pd.DataFrame):
        raise TypeError(f"The {periodely} dataframe is not a DataFrame.")
    if not isinstance(mnr.index, pd.PeriodIndex):
        raise TypeError(f"The {periodely} dataframe does not have a pd.PeriodIndex.")
    if not pd.Series(liste_d4).isin(mnr.columns).all():
        raise TypeError(
            f"{np.setdiff1d(liste_d4, mnr.columns).tolist()} are missing in the {periodely} dataframe."
        )

    # Filters out series not sent to benchmarking.
    mnr_of_concern = mnr[mnr.columns[mnr.columns.isin(liste_d4)]]
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
    if not pd.Series(liste_d4).isin(rea.columns).all():
        raise TypeError(
            f"{np.setdiff1d(liste_d4, rea.columns).tolist()} are missing in the yearly dataframe."
        )

    # Filters out series not sent to benchmarking.
    rea_of_concern = rea[rea.columns[rea.columns.isin(liste_d4)]]
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
    liste_d4 = [
        serie
        for serie in liste_d4
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

    # Preperation of input data.

    # Scaling of the leading and following values.
    avs = rea_of_concern / mnr_of_concern.resample("Y").sum()

    printlist = []

    for elem in liste_d4:
        printlist.append(f"{elem}")
        print(
            f"Benchmarking {printlist} with MinD4 from {startaar} to {basisaar}.",
            end="\r",
            flush=True,
        )

        # Loads monthly and yearly values
        datam_ = mnr_of_concern[elem].values
        datay_ = rea_of_concern[elem].values

        # Different scaling factors for the leading and following values to ensure the volume figure is correct in the start year.
        avstemming1 = avs[avs.index.year == startaar][elem].values
        avstemming2 = avs[avs.index.year == basisaar][elem].values

        # The leading value.
        if freq == "M":
            datamf = mnr[(mnr.index.year == startaar) & (mnr.index.month == 1)][
                elem
            ].values
        if freq == "Q":
            datamf = mnr[(mnr.index.year == startaar) & (mnr.index.quarter == 1)][
                elem
            ].values

        # Scaling of the leading value.
        datayf = avstemming1 * datamf

        # The following value.
        if freq == "M":
            datame = mnr[(mnr.index.year == basisaar) & (mnr.index.month == freq_)][
                elem
            ].values
        if freq == "Q":
            datame = mnr[(mnr.index.year == startaar) & (mnr.index.quarter == freq_)][
                elem
            ].values

        # Scaling of the following value.
        dataye = avstemming2 * datame

        datam = np.hstack((np.asarray(datamf), np.asarray(datam_), np.asarray(datame)))
        datay = np.hstack((np.asarray(datayf), np.asarray(datay_), np.asarray(dataye)))

        # Counting months/quarters and years.
        nm = datam.shape[0]
        ny = datay.shape[0]

        # Setting up submatrices a for A and -vectors x for X consisting of zeros.
        a1 = np.zeros((nm, nm), dtype=np.float64)
        a2 = np.zeros((nm, ny), dtype=np.float64)
        a3 = np.zeros((ny, nm), dtype=np.float64)
        a4 = np.zeros((ny, ny), dtype=np.float64)
        x1 = np.zeros((nm, 1), dtype=np.float64)
        x2 = np.zeros((ny, 1), dtype=np.float64)

        # Fills in submatrices a into A and -vectors x into X according to Skjæveland 1985, pages 18 and 19.
        for i in range(0, nm):
            a1[i, i] = (1 + 1 * (i > 0 and i < nm - 1)) / datam[i] ** 2
            if i > 0:
                a1[i, i - 1] = -1 / (datam[i] * datam[i - 1])
            if i < nm - 1:
                a1[i, i + 1] = -1 / (datam[i] * datam[i + 1])

        for i in range(0, nm):
            for j in range(0, ny):
                if (i - 1) / freq_ >= j - 1 and (i - 1) / freq_ < j:
                    a2[i, j] = 1
                    a3[j, i] = 1
                x2[j] = datay[j]

        # Combines submatrices a into A and -vectors x into X according to Skjæveland 1985, pages 18 and 19.
        A = np.vstack((np.hstack((a1, a2)), np.hstack((a3, a4))))
        X = np.vstack((x1, x2))

        # Solves the equation system AY=X for Y if possible. Otherwise, returns null.
        try:
            Y = np.linalg.solve(A, X)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Wasn't able to benchmark {elem}.", UserWarning, stacklevel=2
            )
            # Y=np.zeros((nm,1),dtype=np.float64)

        res_dict[elem] = Y[1 : nm - 1].flatten()

    res = pd.DataFrame(res_dict, index=mnr_of_concern.index)

    # Removing resulting NaN series from the res-DataFrame.
    res = res[res.columns[~res.isna().any()]]

    # Checks that the deviation on the totals after benchmarking is zero for the series in liste_d4.
    # Skips the series that already trigger error messages in the input.
    skippe = (
        mnr_of_concern.columns[mnr_of_concern.isna().any()].to_list()
        + rea_of_concern.columns[rea_of_concern.isna().any()].to_list()
    )
    for elem in set(liste_d4) - set(skippe):
        if (
            ((res.resample("Y").sum() - rea_of_concern) >= -1)
            & ((res.resample("Y").sum() - rea_of_concern) <= 1)
        ).all()[elem] is not np.True_:
            warnings.warn(
                f"There are deviations on the benchmarked totals in {elem} so something did not go well.",
                UserWarning,
                stacklevel=2,
            )

    print("\nBenchmarking with MinD4 done!")

    return res
