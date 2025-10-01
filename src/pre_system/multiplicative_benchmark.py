def benchmark(df_target: pd.DataFrame, df_indicator: pd.DataFrame):
    
    
    
    df_ratio = (
        df_indicator.groupby(
            pd.PeriodIndex(df_indicator.index, freq=df_target.index.freq)
        )
        .sum()
        .div(df_target)
        .resample(df_indicator.index.freq)
        .ffill()
    )

    return df_indicator.div(df_ratio.fillna(1).reindex(df_indicator.index))[
        df_indicator.columns
    ]
