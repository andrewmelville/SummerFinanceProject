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


def lead_lag_disc_regression(returns_data, period, lookback, shift):
    # Creating frequent returns data
    week_chunks = np.array_split(returns_data.copy().cumsum(), returns_data.shape[0]/period)

    PC_proj_weekly = pd.DataFrame([]*returns_data.shape[0], columns = returns_data.columns, index = [week.index[0] for week in week_chunks])
    for week in week_chunks:
        PC_proj_weekly.loc[week.index[0]] = week.sum()
        
    # Linear regression on weekly returns with 1 weeks shifted predictors
    PC_proj_weekly_ll_one = PC_proj_weekly.copy()
    PC_proj_weekly_ll_one.iloc[:,0] = PC_proj_weekly_ll_one.iloc[:,0].shift(periods=shift)
    rolling_lr(PC_proj_weekly_ll_one.iloc[:,0], PC_proj_weekly_ll_one.iloc[:,1:], lookback, intercept = False)

test = lead_lag_disc_regression(PC_proj_df, period = 7, lookback = 150, shift = 4)