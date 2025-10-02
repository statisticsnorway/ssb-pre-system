# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ssb-pre-system
#     language: python
#     name: ssb-pre-system
# ---

# %%
import pandas as pd
from examples.example_m_y_numbers import cols
from examples.example_m_y_numbers import m_array
from examples.example_m_y_numbers import m_index
from examples.example_m_y_numbers import y_array
from examples.example_m_y_numbers import y_index
from src.pre_system.additive_benchmark import additive_benchmark
from src.pre_system.multiplicative_benchmark import multiplicative_benchmark

# %%
# %run -i 'examples/example_m_y_numbers.py'

m_df = pd.DataFrame(data=m_array, columns=cols)
y_df = pd.DataFrame(data=y_array, columns=cols)

m_df.index = m_index
y_df.index = y_index

# %%
q_df = m_df.resample("Q").sum()

# %%
collist = ["serieA", "serieB"]

(additive_benchmark(m_df, y_df, collist, 2016, 2022) - m_df).plot()

# %%
(multiplicative_benchmark(m_df, y_df, collist, 2016, 2022) / m_df).plot()

# %%
(
    multiplicative_benchmark(m_df, y_df, collist, 2016, 2022).resample("Y").sum()
    - additive_benchmark(m_df, y_df, collist, 2016, 2022).resample("Y").sum()
).round(9)

# %%
multiplicative_benchmark(m_df, q_df, collist, 2016, 2022).resample("Q").sum() - q_df

# %%
additive_benchmark(m_df, q_df, collist, 2016, 2022).resample("Q").sum() - q_df

# %%
