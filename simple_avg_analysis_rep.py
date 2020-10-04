#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:43:52 2020

@author: andrewmelville
"""

from preprocessing import contract_dict, df_dict, equities_labels, energies_labels, currencies_labels, metals_labels, gold_labels, softs_labels, bonds_labels
from lin_reg_analysis import rolling_lr, rolling_lasso, rolling_ridge, PC_proj_ts, PC_proj_df, commodities_2013
from sklearn.decomposition import PCA
from PlottingFunctions import series_plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings("ignore")


# Create percentage returns data shifted to be positive
simp_avg_df = PC_proj_df.copy()
simp_avg_df['Commods Avg'] = commodities_2013.mean(axis=1)
simp_avg_daily_percent = simp_avg_df.cumsum()
for col in simp_avg_daily_percent.columns:

    simp_avg_daily_percent[col] = simp_avg_daily_percent[col] - min([val for val in simp_avg_df.cumsum()[col] if val < 0])+0.05
    simp_avg_daily_percent[col] = simp_avg_daily_percent[col].shift(periods = 1) / simp_avg_daily_percent[col]

week_chunks = np.array_split(simp_avg_daily_percent.copy(), simp_avg_daily_percent.shape[0]/5)

simp_avg_daily_pc_week_sum = pd.DataFrame([]*simp_avg_daily_percent.shape[0], columns = simp_avg_daily_percent.columns, index = [week.index[0] for week in week_chunks])
for week in week_chunks:
    simp_avg_daily_pc_week_sum.loc[week.index[0]] = week.sum()
    
# Linear regression on weekly returns with 1 weeks shifted predictors
simp_avg_daily_pc_weekly_ll = simp_avg_daily_pc_week_sum.copy()
simp_avg_daily_pc_weekly_ll.iloc[:,0] = simp_avg_daily_pc_week_sum.iloc[:,0].shift(periods=1)
test = rolling_lr(simp_avg_daily_pc_weekly_ll.iloc[:,0], simp_avg_daily_pc_weekly_ll.iloc[:,1:], 150, intercept = False)
# test = rolling_lasso(simp_avg_daily_pc_weekly_ll.iloc[:,0], simp_avg_daily_pc_weekly_ll.iloc[:,1:], 150, intercept = False, alph = 0)
# test = rolling_ridge(simp_avg_daily_pc_weekly_ll.iloc[:,0], simp_avg_daily_pc_weekly_ll.iloc[:,1:], 150, intercept = False, alph = 2)

# series_plot([simp_avg_daily_pc_weekly_ll.iloc[:,0], simp_avg_daily_pc_weekly_ll['WTI Crude Oil']],'Oil Price against commodities')
#%%
# # series_plot([commodities_2013.mean(axis=1) - trimmed_dates['NYMEX WTI Crude Oil']],'Oil Price against commodities')
# # series_plot([commodities_2013.mean(axis=1) - trimmed_dates['NYMEX WTI Crude Oil']],'Oil Price against commodities')
# series_plot([commodities_2013.mean(axis=1) - trimmed_dates['ICE US Dollar Index']],'Oil Price against commodities')
# series_plot([commodities_2013.mean(axis=1) - trimmed_dates['NYMEX WTI Crude Oil']],'Oil Price against commodities')
