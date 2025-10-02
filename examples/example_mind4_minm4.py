# ---
# jupyter:
#   jupytext:
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
# Necessary packages.
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
pd.set_option("display.float_format", "{:.0f}".format)

# Importing the functions from the .py-files with the same name respectively.
from src.pre_system.mind4 import mind4
from src.pre_system.minm4 import minm4

# %% [markdown]
# ## Input Data

# %% [markdown]
# Loading input data.

# %%
# %run -i 'examples/example_m_y_numbers.py'

# %%
m_df = pd.DataFrame(data=m_array, columns=cols)
y_df = pd.DataFrame(data=y_array, columns=cols)

m_df.index = m_index
y_df.index = y_index



# %% jupyter={"outputs_hidden": true}
q_df = m_df.resample("Q").sum()

# %% [markdown]
# ## Value to be Benchmarked

# %% jupyter={"outputs_hidden": true}
(m_df.resample("Y").sum() - y_df)

# %% [markdown]
# ### Defining Parameters

# %%
list_to_benchmarking = m_df.columns.to_list()

baseyear = 2022
firstyear = 2016

# list_to_benchmarking1 = list_to_benchmarking
# list_to_benchmarking1.remove('serieA')
# list_to_benchmarking1

# %% [markdown]
# # Benchmarking

# %% [markdown]
# ### Monthly frequency

# %%
result_d4 = mind4(m_df, y_df, list_to_benchmarking, baseyear, firstyear)

# result

# %%
result_m4 = minm4(m_df, y_df, list_to_benchmarking, baseyear, firstyear)

# result_m4

# %% [markdown]
# ### Quarterly frequency

# %%
result_q_m4 = minm4(q_df, y_df, list_to_benchmarking, baseyear, firstyear, freq="Q")

# result_m4

# %%
result_q_d4 = mind4(q_df, y_df, list_to_benchmarking, baseyear, firstyear, freq="Q")

# result_q_d4

# %% [markdown]
# ## Effect of Benchmarking

# %% [markdown]
# ### Monthly frequency

# %%
(result_d4 / m_df).plot()

# %%
(result_m4 / m_df).plot()

# %% [markdown]
# ### Quarterly frequency

# %%
(result_q_d4 / q_df).plot()

# %%
(result_q_m4 / q_df).plot()

# %% [markdown]
# # Deviations on the Total Post Benchmarking.

# %%
# With MinD4.
(result_d4.resample("Y").sum() - y_df)

# %%
# With MinD4.
(result_m4.resample("Y").sum() - y_df)

# %%
# With MinD4.
(result_q_d4.resample("Y").sum() - y_df)

# %%
# With MinD4.
(result_q_m4.resample("Y").sum() - y_df)

# %%

# %%
