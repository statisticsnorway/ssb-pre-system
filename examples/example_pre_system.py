# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Demo of pre-system Formula and PreSystem classes
# By: Magnus Kvåle Helliesen
#
# This notebook is entirely selfcontained, in the sense that it doesn not use any existing data, but creates random data for illustrational purposes.
#
# We import `pandas`, `numpy`, `matplotlib` and the `Formula` class and sub-classes, and `PreSystem`.

# %%
# ruff: noqa: B018
import os

# Change directory until find project root
notebook_path = os.getcwd()
for _ in range(50):
    if "pyproject.toml" in os.listdir():
        break
    os.chdir("../")

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.pre_system.formula import FDeflate
from src.pre_system.formula import FInflate
from src.pre_system.formula import FJoin
from src.pre_system.formula import Formula
from src.pre_system.formula import FSum
from src.pre_system.formula import Indicator
from src.pre_system.pre_system import PreSystem

# %% [markdown]
# ## Defining formulae
# There are a bunch of different Formula *child*-classes, some of which are put to use below.

# %%
# Let's make a formula that extrapolates x using x1, x2 and x3
xa1 = Indicator("xa1", "xa", ["x0", "x1", "x2"])
xa2 = Indicator("xa2", "xa", ["x0", "x1"])
xa = FJoin("xa", xa1, xa2, 2017)
xb = Indicator("xb", "xb", ["x3", "x4"])

# Let's deflate them with a bunch of weighted price indices
vxa = FDeflate("vxa", xa, ["p0", "p1", "p2"], ["w0", "w1", "w2"])
vxb = FInflate("vxb", xb, ["p3", "p4"], ["w3", "w4"])

# Finally, let's sum them up
x = FSum("xy", xa, xb)
vx = FSum("vxy", vxa, vxb)

# %% [markdown]
# ## Looking at the textual representation of the formulae
# We haven't any data yet, but we can look at the textual representation of the formulae.

# %%
xa.info()

# %%
xb

# %%
vxa

# %%
vxb

# %%
x

# %%
vx

# %% [markdown]
# We can also trace back to any formulae that go into some formula.

# %%
vxa.info()

# %% [markdown]
# We can also look at pairwise indicators and weights

# %%
# With trace
vxa.indicators_weights()

# %%
# Without trace
vxa.indicators_weights(trace=False)

# %% [markdown]
# ## Subjecting formulae to data (evaluating formulae with respect to data)
# Let's make some random data and store them in Pandas DataFrames. Importantly, the data *must* be indexed by Pandas PeriodIndices.

# %%
years = 13

# %%
annual_df = pd.DataFrame(
    np.exp(0.02 + np.random.normal(0, 0.01, (years, 10)).cumsum(axis=0)),
    columns=[f"x{i}" for i in "abcdefghij"],
    index=pd.period_range(start="2010", periods=years, freq="Y"),
)

annual_df.plot(figsize=(15, 2.5))
plt.title("Annual values")
plt.show()

# %%
indicator_df = pd.DataFrame(
    np.exp(0.02 + np.random.normal(0, 0.01, (years * 12, 10)).cumsum(axis=0)),
    columns=[f"x{i}" for i in range(5)] + [f"p{i}" for i in range(5)],
    index=pd.period_range(start="2010-01", periods=years * 12, freq="M"),
)

indicator_df.plot(figsize=(15, 2.5))
plt.title("Indicators")
plt.show()

# %%
weight_df = pd.DataFrame(
    10 + np.random.normal(0, 1, (years, 5)).cumsum(axis=0),
    columns=[f"w{i}" for i in range(5)],
    index=pd.period_range(start="2010", periods=years, freq="Y"),
)

weight_df[["w0", "w1", "w2"]] = weight_df[["w0", "w1", "w2"]].divide(
    weight_df[["w0", "w1", "w2"]].sum(axis=1), axis=0
)
weight_df[["w3", "w4"]] = weight_df[["w3", "w4"]].divide(
    weight_df[["w3", "w4"]].sum(axis=1), axis=0
)

weight_df.plot(figsize=(15, 2.5))
plt.title("Weights")
plt.show()

# %% [markdown]
# Before we can evaluate the formulae, we need to set a baseyear.

# %%
Formula.baseyear = 2020

# %%
fig = x.evaluate(annual_df, indicator_df, weight_df).plot(figsize=(15, 2.5))
plt.title("x")
plt.show()

# %%
fig = vx.evaluate(annual_df, indicator_df, weight_df).plot(figsize=(15, 2.5))
plt.title("vx")
plt.show()

# %% [markdown]
# ## Organizing the formulae in the PreSystem class
# The PreSystem class is written to contain formulae, and allow the user to easily evaluate them, subject to data (contained by PreSystem).

# %%
# Let's create a PreSystem instance
pre_system = PreSystem("Test PreSystem")
pre_system

# %% [markdown]
# The PreSystem is now ready to accept formulae and data.

# %%
pre_system.add_formula(xa1)
pre_system.add_formula(xa2)
pre_system.add_formula(xa)
pre_system.add_formula(xb)
pre_system.add_formula(vxa)
pre_system.add_formula(vxb)
pre_system.add_formula(x)
pre_system.add_formula(vx)
pre_system.info()

# %% [markdown]
# In order to evaluate the PreSystem, we need to put data in it, and set the baseyear.

# %%
pre_system.baseyear = 2020
pre_system.annuals_df = annual_df
pre_system.indicators_df = indicator_df
pre_system.weights_df = weight_df
pre_system.info()

# %% [markdown]
# Now we can evaluate the PreSystem (horay!).

# %%
pre_system.evaluate

# %% [markdown]
# ## Convert and overlay functions
# `pre_system` also contain three functions, `convert`, `convert_step` and `overlay`. Let's import these.
#
# ### Convert

# %%
from src.pre_system.convert import convert
from src.pre_system.convert import convert_step
from src.pre_system.overlay import overlay

# %% [markdown]
# We can now "upsample" time series from, say, quarterly to monthly frequency.

# %%
quarterly_df = pd.DataFrame(
    np.exp(0.02 + np.random.normal(0, 0.01, (16, 3)).cumsum(axis=0)),
    columns=[f"x{i}" for i in range(3)],
    index=pd.period_range(start="2019q1", periods=16, freq="q"),
)

quarterly_df.plot(figsize=(15, 2.5))
plt.title("Quarterly data")
plt.show()

# %%
convert(quarterly_df, "m").plot(figsize=(15, 2.5))
plt.title("Quarterly data to montlhy using smooth method")
plt.show()

# %%
convert_step(quarterly_df, "m").plot(figsize=(15, 2.5))
plt.title("Quarterly data to montlhy using step method")
plt.show()

# %% [markdown]
# ### Overlay
# Sometimes we have missing data, and we want to use and "overlay" function that takes any number of DataFrames or Series as input, and outputs the data from the first wherever they are present, the second if any from the first is missing, the third if any from the first and second are missing, and so on (you get the point). The function `overlay` does exactly this.

# %%
# Lets make some DataFrames and remove some values
df0 = quarterly_df.copy()
df1 = quarterly_df.copy()
df2 = quarterly_df.copy()

df0.iloc[0:5, :] = np.nan
df1.iloc[3:9, :] = np.nan
df2.iloc[10:12, :] = np.nan

# %%
df0

# %%
df1

# %%
df2

# %%
# If we overlay the three DataFrames, the result contains no NaN's
overlay(df0, df1, df2)

# %%
# If we leave out df2 there are still some, since the NaN's in df0 and df1 overlap
overlay(df0, df1)

# %%
pre_system.baseyear = 2012
pre_system.evaluate_formulae("xa", "xb").plot()
