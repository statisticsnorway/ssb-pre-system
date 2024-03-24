from pathlib import Path

import pandas as pd
from pandas import testing as tm

from pre_system.convert import convert
from pre_system.convert import convert_step


def test_convert(quarterly_df) -> None:
    result_df = convert(quarterly_df, "M")

    write_new_facit_file = False
    file = Path(__file__).parent / "testdata" / "facit_convert_monthly.parquet"
    if write_new_facit_file:
        result_df.to_parquet(file)

    facit_df = pd.read_parquet(file)
    tm.assert_frame_equal(result_df, facit_df)


def test_convert_step(quarterly_df) -> None:
    result_df = convert_step(quarterly_df, "M")

    write_new_facit_file = False
    file = Path(__file__).parent / "testdata" / "facit_convert_step_monthly.parquet"
    if write_new_facit_file:
        result_df.to_parquet(file)

    facit_df = pd.read_parquet(file)
    tm.assert_frame_equal(result_df, facit_df)
