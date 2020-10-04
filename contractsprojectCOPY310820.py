#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:54:18 2020

@author: andrewmelville
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist


import seaborn as sns
sns.set_style('whitegrid')
import itertools
import sklearn as skl
import datetime
import warnings
warnings.filterwarnings("ignore")


#%% Preliminary Printing
   
    
# # Printing lengths of each data frame
# for contract in contract_names:
#     print(contract, len(contract_dict[contract]))
    
# # Printing date ranges of each contract
# for contract in contract_names:
#     print(contract, contract_dict[contract].index.min(),contract_dict[contract].index.max())
    
#%% Daily Returns for each market
# contract_dict_rolling = contract_dict.copy()


# # Creating daily returns column
# for contract in contract_names:
#     contract_dict_rolling[contract]['Returns'] = contract_dict_rolling[contract]['Close'].diff()

# # Computing rolling std of daily returns
# for contract in contract_names:
#     contract_dict_rolling[contract]['Rolling Std'] = contract_dict_rolling[contract]['Returns'].rolling(20, min_periods = 15).std()

# # Computing standardised daily returns
# for contract in contract_names:
#     contract_dict_rolling[contract]['Std Daily Returns'] = contract_dict_rolling[contract]['Returns'] / contract_dict_rolling[contract]['Rolling Std']
    
# # Computing rolling std of standardised daily returns
# for contract in contract_names:
#     contract_dict_rolling[contract]['Rolling Std of Std Daily Returns'] = contract_dict_rolling[contract]['Std Daily Returns'].rolling(20, min_periods = 15).std()
    
    
#  #%% Variable Dictionary Creation
    
    
# # Creating dictionary of dataframes for each variable and each contract
# df_dict = dict(zip([],[]))
# for var in [col_name  for col_name in contract_dict['SHFE Zinc'].columns if col_name != 'Symbol']:
#     df_dict[var] = pd.DataFrame([], index = pd.bdate_range('1/1/1980', end='31/7/2020'))
#     for contract in contract_names:
#         df_dict[var][contract] = contract_dict[contract][var]
#%% Summary stats for each contract
        
        
# summary_dict = dict(zip([],[]))
# for contract in contract_names:
#     summary_dict[contract] = contract_dict[contract].describe()

#%%
    
# Checking plots of close prices for each exchange group
plt.figure(figsize=(20,10))

for j, sector in enumerate(sector_col_dict.keys()):
    for i, contract in enumerate(sectors_dict_inv[sector]):
        plt.plot(df_dict['Close'][contract], label = sectors_dict[contract] if i == 0 else "", color = sector_col_dict[sectors_dict[contract]], lw = 0.5)

        
plt.xlabel('Year')
plt.ylabel('Close Price (US Dollars)')
plt.title('Available Data for all Contracts')
plt.legend()
plt.show()

# for contract in contract_names:
#     if contract[0:3] == 'CME' and contract != 'CME Nikkei 225' and contract != 'CME E-mini Dow Jones':
#         plt.plot(df_dict['Close'][contract],label=contract)
# plt.legend(loc='upper center')
# plt.show()

# for contract in contract_names:
#     if contract[0:3] == 'EUR' and contract != 'EUREX DAX' and contract != 'EUREX EURO STOXX 50 Index':
#         plt.plot(df_dict['Close'][contract],label=contract)
# plt.legend(loc='upper center')
# plt.show()

# for contract in contract_names:
#     if contract[0:3] == 'ICE' and contract != 'ICE WTI Crude Oil' and contract != 'ICE Cocoa':
#         plt.plot(df_dict['Close'][contract],label=contract)
# # plt.legend(loc='upper center')
# plt.show()
        
# for contract in contract_names:
#     if contract[0:3] == 'LIF' and contract != 'LIFFE FTSE 100 Index' and contract != 'LIFFE London Cocoa':
#         plt.plot(df_dict['Close'][contract],label=contract)
# plt.legend(loc='upper center')
# plt.show()
        
# for contract in contract_names:
#     if contract[0:3] == 'SHF':
#         plt.plot(df_dict['Close'][contract],label=contract)
# # plt.legend(loc='upper center')
# plt.show()
        
# for contract in contract_names:
#     plt.plot(df_dict['Close'][contract],label=contract)
#     plt.title(contract)
#     plt.show()
    

#%% Function for returning df of ewm daily returns
    
    
# def EW_STD(alpha):
#     contract_dict_ew = contract_dict.copy()
#     # Computing ew std
#     for contract in contract_names:
#         contract_dict_ew[contract]['EW Std {}'.format(alpha)] = contract_dict_ew[contract]['Returns'].ewm(halflife=alpha).std()
          
#     # Computing EW standardised daily returns
#     for contract in contract_names:
#         contract_dict_ew[contract]['EW Std Daily Returns {}'.format(alpha)] = contract_dict_ew[contract]['Returns'] / contract_dict_ew[contract]['EW Std {}'.format(alpha)]
    
#     # Computing rolling std of EW standardised daily returns
#     for contract in contract_names:
#         contract_dict_ew[contract]['Rolling Std of EW Std Daily Returns {}'.format(alpha)] = contract_dict_ew[contract]['EW Std Daily Returns {}'.format(alpha)].ewm(halflife=alpha).std()
        
#     return contract_dict_ew

# ew_5 = EW_STD(5)
# ew_50 = EW_STD(50)


#%% Creation of 5-year returns column and edgelist for graph plotting
        

# Taking dates after 1/1/2020 and excluding DJIA
# mask = df_dict['Std 2 Day Returns'].index > '2000-01-01'
returns_full = df_dict['Std 2 Day Returns']
# returns_full = returns_full.drop('CBOT Dow Jones Ind Avg (DJIA)',axis=1)

# Creating matrix of correlation values
twoday_corr_mat = returns_full.corr()

#convert matrix to list of edges and rename the columns
corr_list = twoday_corr_mat.stack().reset_index()
corr_list.columns = ['asset_1','asset_2','correlation']

#remove self correlations
corr_list = corr_list.loc[corr_list['asset_1'] != corr_list['asset_2']].copy()

#%% Seaborn Clustermap Plot



# Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

cluster_map = sns.clustermap(twoday_corr_mat, xticklabels = contract_names, yticklabels = contract_names, cmap="RdYlGn", center=0,
            square=True)

cluster_map.fig.suptitle('Correlation of 2 Day Standardised Returns')

hm = cluster_map.ax_heatmap.get_position()
plt.setp(cluster_map.ax_heatmap.yaxis.get_majorticklabels(), fontsize=6)
plt.setp(cluster_map.ax_heatmap.xaxis.get_majorticklabels(), fontsize=6)
cluster_map.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])
col = cluster_map.ax_col_dendrogram.get_position()
cluster_map.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height])


#%% Static 5-Year lookback graph

roll_corr_df = contract_dict.copy()

# Taking dates after 1/1/2020 and excluding DJIA
mask = df_dict['Std 2 Day Returns'].index > '2015-01-01'
returns_2015 = df_dict['Std 2 Day Returns'][mask]
returns_2015 = returns_2015.drop('CBOT Dow Jones Ind Avg (DJIA)',axis=1)

# Creating matrix of correlation values
twoday_corr_mat_2015 = returns_2015.corr()

#convert matrix to list of edges and rename the columns
corr_list_2015 = twoday_corr_mat_2015.stack().reset_index()
corr_list_2015.columns = ['asset_1','asset_2','correlation']

#remove self correlations
corr_list_2015 = corr_list_2015.loc[corr_list_2015['asset_1'] != corr_list_2015['asset_2']].copy()



Gx = nx.from_pandas_edgelist(corr_list_2015, 'asset_1', 'asset_2', edge_attr=['correlation'])

threshold = 0

# # list to store edges to remove
# remove = []
# # loop through edges in Gx and find correlations which are below the threshold
# for asset_1, asset_2 in Gx.edges():
#     corr = Gx[asset_1][asset_2]['correlation']
#     #add to remove node list if abs(corr) < threshold
#     if abs(corr) < threshold:
#         remove.append((asset_1, asset_2))

# remove edges contained in the remove list
# Gx.remove_edges_from(remove)

def assign_colour(correlation):
    if correlation <= 0:
        return "#ffa09b"  # red
    else:
        return "#9eccb7"  # green


def assign_thickness(correlation, benchmark_thickness=2, scaling_factor=3):
    return benchmark_thickness * abs(correlation)**scaling_factor


def assign_node_size(degree, scaling_factor=5):
    return degree * scaling_factor

#%%
# assign colours to edges depending on positive or negative correlation
# assign edge thickness depending on magnitude of correlation
edge_colours = []
edge_width = []
for key, value in nx.get_edge_attributes(Gx, 'correlation').items():
    edge_colours.append(assign_colour(value))
    edge_width.append(assign_thickness(value))

# # assign node size depending on number of connections (degree)
# node_size = []
# for key, value in dict(Gx.degree).items():
#     node_size.append(assign_node_size(value))
    
    
# draw improved graph
sns.set(rc={'figure.figsize': (9, 9)})
font_dict = {'fontsize': 18}
#%%
fixed_pos = nx.kamada_kawai_layout(Gx)

# draw improved graph
nx.draw(Gx, pos=nx.spring_layout(Gx), with_labels=True,
        node_size= 5 , node_color="#e1575c", edge_color=edge_colours,
       width = edge_width)
plt.title("Contract 2 Day Returns Correlations - Fruchterman-Reingold layout",fontdict=font_dict)
plt.show()

#%%
from PlottingFunctions import PCA_loadings_visualiser
PCA_loadings_visualiser(equities_pca, equities_labels, [-1,-1,-1], 'Equities')
PCA_loadings_visualiser(energies_pca, energies_labels, [-1,-1,-1], 'Energies')
PCA_loadings_visualiser(currencies_pca, currencies_labels, [1,1,-1], 'Currencies')
PCA_loadings_visualiser(metals_pca, metals_labels, [-1,1,1], 'Metals')
PCA_loadings_visualiser(gold_pca, gold_labels, [1,1,-1], 'Precious Metals')
PCA_loadings_visualiser(softs_pca, softs_labels, [1,1,1,1], 'Softs')
PCA_loadings_visualiser(bonds_pca, bonds_labels, [-1,-1,-1,1,1], 'Bonds')
#%% Stability of PC through time

cluster_keys = [equities_labels, energies_labels, currencies_labels, metals_labels, gold_labels, softs_labels, bonds_labels]
cluster_names = ['Equities', 'Energies', 'Currencies', 'Metals', 'Precious Metals','Softs', 'Bonds']                                        
#%% 
    
for i, cluster in enumerate(cluster_keys): 
    # print(cluster)
    prop_of_var(trimmed_dates[cluster], cluster_names[i], 80)                                                                                                                                   
#%% Comparing PC projection time series to simple average of contracts


# commodities_2013['1st PC Projection'] = PC_proj_df
# commodities_2013['Simp Avg'] = commodities_2013.drop('1st PC Projection',axis=1).mean(axis=1)
# commodities_2013['Difference'] = commodities_2013['1st PC Projection'] - commodities_2013['Simp Avg']

# plt.figure(figsize=(20,10))
# plt.plot(commodities_2013['Simp Avg'].iloc[179:], lw=1)
# plt.xlabel('Year')
# plt.ylabel('Simple Average')
# plt.title('Simple Average of Commodities')
# # plt.legend()
# plt.show()
# #%% Comparing cumsums of PC projection time series to simple average of contracts


# plt.figure(figsize=(20,10))
# # plt.plot(commodities_2013['Simp Avg'].cumsum().iloc[179:] - commodities_2013['1st PC Projection'].cumsum().iloc[179:], lw=1)
# plt.plot(commodities_2013['Simp Avg'].cumsum().iloc[179:], lw=1, label = 'Simple Average')
# plt.plot(commodities_2013['1st PC Projection'].cumsum().iloc[179:], lw=1, label = '1st PC Projection')

# plt.xlabel('Year')
# plt.ylabel('Simple Average')
# plt.title('Simple Average of Commodities')
# plt.legend()
# plt.show()
#%% Plotting NaN days to visualise missing data


# na_dates = commodities_2013.copy()

# plt.figure(figsize=(20,10))
# for i, contract in enumerate(na_dates.columns):

#     hold_copycopy = na_dates[contract].copy()    
    
#     na_dates[contract][pd.isnull(hold_copy) == False] = np.nan
#     na_dates[contract][pd.isnull(hold_copy)] = i
    
#     plt.plot(na_dates[contract],ds='steps-pre',lw=2,label = contract.split(' ')[0], color = sector_col_dict[sectors_dict[contract]])

# plt.title('NaN days in Commodities Data')
# plt.xlabel('Date')
# plt.ylabel('False Height')
# plt.legend(loc=4)
# plt.show()
#%% Fixed PCA Projection


# proportion_dict = dict(zip([],[]))
# def PC_proj_ts(returns_data, lookback):

#     # Dropping nan rows
#     returns_data = returns_data.dropna()
#     proj_col = [0]*returns_data.shape[0]

#     fix_pca = PCA(n_components = returns_data.shape[1])
#     fix_pca.fit(returns_data)
    
#     for i in range(returns_data.shape[0]-lookback):
        
#         # Splice data frame
#         returns_window = returns_data.iloc[i:i+lookback,:]
#         # print(returns_window)
        
#         # Project day t's returns onto PC space and take first component
#         day_t_proj = fix_pca.transform(np.array(returns_data.iloc[i+lookback,:]).reshape(1,-1))
#         proj_col[i+lookback] = day_t_proj[0][0]
#         proj_df = pd.DataFrame([], index = returns_data.index)
#         proj_df['1st PC Projection'] = proj_col
    
#     plt.figure(figsize=(20,10))    
#     plt.plot(proj_df['1st PC Projection'].iloc[60:], lw=1)
#     plt.xlabel('Year')
#     plt.ylabel('Projection onto 1st PC')
#     plt.title('Projection of Commodities Returns onto 1st PC')
#     # plt.legend()
#     plt.show()
    
#     return proj_df
        
# test = PC_proj_ts(trimmed_dates[energies_labels+metals_labels+softs_labels+gold_labels], 180)