from preprocessing import contract_dict, df_dict, equities_labels, energies_labels, currencies_labels, metals_labels, gold_labels, softs_labels, bonds_labels
from lin_reg_analysis import rolling_lr, PC_proj_ts, PC_proj_df
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings("ignore")


def lead_lag_cont_regression(returns_data, diff, lookback, shift):

    # Creating diff returns data
    PC_proj_weekly = returns_data.copy().cumsum().diff(periods = diff)

    # Linear regression on weekly returns with shifted predictors
    PC_proj_weekly_ll_one = PC_proj_weekly.copy()
    PC_proj_weekly_ll_one.iloc[:,0] = PC_proj_weekly_ll_one.iloc[:,0].shift(periods=shift)
    return PC_proj_weekly_ll_one
    # rolling_lr(PC_proj_weekly_ll_one.iloc[:,0], PC_proj_weekly_ll_one.iloc[:,1:], lookback, intercept = False)

test = lead_lag_cont_regression(PC_proj_df, diff = 1, lookback = 150, shift = 0)