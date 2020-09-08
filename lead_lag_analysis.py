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

PC_proj_weekly = PC_proj_df.copy()

plt.plot(PC_proj_weekly['1st PC Projection'].cumsum())