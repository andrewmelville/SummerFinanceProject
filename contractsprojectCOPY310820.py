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



#%% Data Loading

# Reading in contract list
contract_list = pd.read_csv("/Users/andrewmelville/Documents/SummerFinanceProject/Future Contract List.csv")
contract_names = contract_list['NAME']

# Importing from file names and creating dctionary of dataframes for each variable and each contract
contract_dict = dict(zip([],[]))
for i, contract in enumerate(contract_names):
    contract_dict[contract]  = pd.DataFrame([], index = pd.bdate_range(start = '1/1/1980', end = '7/31/2020'))

    current = pd.read_csv("/Users/andrewmelville/Documents/SummerFinanceProject/ContinuousSeries/{}.csv".format(contract), index_col = 0, skiprows = 0, skipfooter = 1, header = 1)
    current.index = pd.to_datetime(current.index)
    
    contract_dict[contract] = contract_dict[contract].join(current)

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
    
    
 #%% Variable Dictionary Creation
    
    
# Creating dictionary of dataframes for each variable and each contract
df_dict = dict(zip([],[]))
for var in [col_name  for col_name in contract_dict['SHFE Zinc'].columns if col_name != 'Symbol']:
    df_dict[var] = pd.DataFrame([], index = pd.bdate_range('1/1/1980', end='31/7/2020'))
    for contract in contract_names:
        df_dict[var][contract] = contract_dict[contract][var]
#%% Summary stats for each contract
        
        
summary_dict = dict(zip([],[]))
for contract in contract_names:
    summary_dict[contract] = contract_dict[contract].describe()


 #%% Preliminary Plotting


energies_labels = ['ICE Heating Oil',
                                                 'ICE WTI Crude Oil',
                                                 'NYMEX Gasoline', 
                                                 'ICE Gasoil',
                                                 'NYMEX Heating Oil',
                                                 'ICE Brent Crude Oil',
                                                 'NYMEX WTI Crude Oil']

equities_labels = ['CME Nikkei 225',
                                                 'LIFFE FTSE 100 Index',
                                                 'EUREX DAX', 
                                                 'EUREX EURO STOXX 50 Index',
                                                 'MX Montreal SPTSX 60 Index',
                                                 'CME S&P 500 Index',
                                                 'CME NASDAQ 100 Index Mini',
                                                 'CME S&P 400 Midcap Index',
                                                 'CME Russell 2000 Index Mini Futures',
                                                 'CME E-mini Dow Jones',
                                                 'CME S&P 500 Index E-Mini']

currencies_labels = ['CME Japanese Yen JPY',
                                                 'CME British Pound GBP', 
                                                 'CME Euro FX',
                                                 'CME Swiss Franc CHF']

metals_labels = ['COMEX Copper',
                                                 'SHFE Natural Rubber',
                                                 'SHFE Rebar', 
                                                 'SHFE Lead',
                                                 'SHFE Aluminium',
                                                 'SHFE Copper',
                                                 'SHFE Zinc']

gold_labels = ['NYMEX Gold',
                                                 'NYMEX Palladium',
                                                 'NYMEX Silver', 
                                                 'NYMEX Platinum']


bonds_labels = ['LIFFE EURIBOR',
                                              'LIFFE Short Sterling',
                                              'EUREX EuroOAT',
                                              'LIFFE Long Gilt',
                                              'EUREX EuroSchatz',
                                              'EUREX EuroBobl',
                                              'EUREX EuroBund',
                                              'CME Eurodollar',
                                              'CBOT 2-year US Treasury Note',
                                              'CBOT 30-year US Treasury Bond',
                                              'CBOT 10-year US Treasury Note',
                                              'CBOT 5-year US Treasury Note']

softs_labels = ['CBOT Soybean Oil',
                                              'CBOT Soybean Meal',
                                              'CBOT Soybeans',
                                              'MGEX Hard Red Spring Wheat',
                                              'CBOT Wheat',
                                              'CME Kansas City Wheat',
                                              'CBOT Corn',
                                              'CBOT Oats']

full_sector_labels = equities_labels + energies_labels + currencies_labels + metals_labels + gold_labels + softs_labels + bonds_labels

equities_dict = {i: 'Equities' for i in equities_labels}
energies_dict = {i: 'Energies' for i in energies_labels}
currencies_dict = {i: 'Currencies' for i in currencies_labels}
metals_dict = {i: 'Metals' for i in metals_labels}
gold_dict = {i: 'Precious Metals' for i in gold_labels}
bonds_dict = {i: 'Bonds' for i in bonds_labels}
softs_dict = {i: 'Softs' for i in softs_labels}
misc_dict = {name:'Other' for name in contract_names if name not in full_sector_labels}

sectors_dict = equities_dict.copy()
sectors_dict.update(energies_dict)
sectors_dict.update(currencies_dict)
sectors_dict.update(metals_dict)
sectors_dict.update(gold_dict)
sectors_dict.update(bonds_dict)
sectors_dict.update(softs_dict)
sectors_dict.update(misc_dict)

sector_col_dict = {'Equities':'#D81B60',
                   'Energies':'#1E88E5',
                   'Currencies':'#FFC107',
                   'Metals':'#1E88E5',
                   'Precious Metals':'#000000',
                   'Bonds':'#F5B1FF',
                   'Softs':'#112CFF',
                   'Other':'#35FF11'}
    

sectors_dict_inv = {}
for k, v in sectors_dict.items():
    sectors_dict_inv[v] = sectors_dict_inv.get(v, [])
    sectors_dict_inv[v].append(k)
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
    

#%% Daily Returns for each market
    
    
def EW_STD(alpha):
    contract_dict_ew = contract_dict.copy()
    # Computing ew std
    for contract in contract_names:
        contract_dict_ew[contract]['EW Std {}'.format(alpha)] = contract_dict_ew[contract]['Returns'].ewm(halflife=alpha).std()
          
    # Computing EW standardised daily returns
    for contract in contract_names:
        contract_dict_ew[contract]['EW Std Daily Returns {}'.format(alpha)] = contract_dict_ew[contract]['Returns'] / contract_dict_ew[contract]['EW Std {}'.format(alpha)]
    
    # Computing rolling std of EW standardised daily returns
    for contract in contract_names:
        contract_dict_ew[contract]['Rolling Std of EW Std Daily Returns {}'.format(alpha)] = contract_dict_ew[contract]['EW Std Daily Returns {}'.format(alpha)].ewm(halflife=alpha).std()
        
    return contract_dict_ew

ew_5 = EW_STD(5)
ew_50 = EW_STD(50)

#%% Creating Returns Columns
    

# Creating daily returns column
for contract in contract_names:
    contract_dict[contract]['Returns'] = contract_dict[contract]['Close'].diff()

# Creating bi-daily returns column
for contract in contract_names:
    contract_dict[contract]['2 Day Returns'] = contract_dict[contract]['Close'].diff(periods=2)

# # Computing rolling std of daily returns
# for contract in contract_names:
    contract_dict[contract]['Rolling Std'] = contract_dict[contract]['Returns'].rolling(20, min_periods = 15).std()

# Computing standardised daily returns
for contract in contract_names:
    contract_dict[contract]['Std 2 Day Returns'] = contract_dict[contract]['2 Day Returns'] / (contract_dict[contract]['Rolling Std']*np.sqrt(2))


# Creating dictionary of variable dataframes
df_dict = dict(zip([],[]))
for var in [col_name  for col_name in contract_dict['SHFE Zinc'].columns if col_name != 'Symbol']:
    df_dict[var] = pd.DataFrame([], index = pd.bdate_range('1/1/1980', end='31/7/2020'))
    for contract in contract_names:
        df_dict[var][contract] = contract_dict[contract][var]
#%% Creation of 5-yereturns column and edgelist for graph plotting
        

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


#%% Principal Component Analysis 


from sklearn.decomposition import PCA
#%% Equities Cluster
# Creating sub dataframes for obvious clusters


equities_labels = ['CME Nikkei 225',
                                                 'LIFFE FTSE 100 Index',
                                                 'EUREX DAX', 
                                                 'EUREX EURO STOXX 50 Index',
                                                 'MX Montreal SPTSX 60 Index',
                                                 'CME S&P 500 Index',
                                                 'CME NASDAQ 100 Index Mini',
                                                 'CME S&P 400 Midcap Index',
                                                 'CME Russell 2000 Index Mini Futures',
                                                 'CME E-mini Dow Jones',
                                                 'CME S&P 500 Index E-Mini']
# Indicies Cluster
equities_returns = df_dict['Std 2 Day Returns'][equities_labels].copy()

# Removing nan dates
equities_returns = equities_returns.dropna()


#'CME Russell 1000 Index Mini Futures' is removed as it only has 1024 available dates


equities_pca = PCA(n_components=equities_returns.shape[1])
equities_pca.fit(equities_returns)

print(equities_pca.explained_variance_ratio_)

# equities_transformed_returns = equities_pca.transform(equities_returns)

# # Getting transformed unit basis vectors in PC space
# equities_basis_transformed = [equities_pca.transform(a.reshape(1,-1)) for a in np.diag([1]*len(equities_labels))]
# equities_basis_transformed = [a / np.linalg.norm(a) for a in equities_basis_transformed]
# # equities_basis_transformed = np.array(equities_basis_transformed)

# for vector in equities_basis_transformed:
#     plt.quiver(0,0,vector[0][0],vector[0][1],scale=3)
#     # plt.quiver([0]*len(equities_labels),equities_basis_transformed[0],equities_basis_transformed[1])
# plt.scatter(equities_transformed[:,0], equities_transformed[:,1], s = 0.5)

#%% Oils Cluster

energies_labels = ['ICE Heating Oil',
                                                 'ICE WTI Crude Oil',
                                                 'ICE Gasoil',
                                                 'ICE Brent Crude Oil',
                                                 'NYMEX Gasoline',
                                                 'NYMEX Heating Oil',
                                                 'NYMEX WTI Crude Oil']
energies_returns = df_dict['Std 2 Day Returns'][energies_labels].copy()

# Removing nan dates
energies_returns = energies_returns.dropna()




energies_pca = PCA(n_components=energies_returns.shape[1])
energies_pca.fit(energies_returns)

print(energies_pca.explained_variance_ratio_)


#%% Currencies Cluster

currencies_labels = ['CME Japanese Yen JPY',
                                                 'CME British Pound GBP', 
                                                 'CME Euro FX',
                                                 'CME Swiss Franc CHF']

## 'ICE British Pound GBP dropped due to correlation with CME and less data

# Indicies Cluster
currencies_returns = df_dict['Std 2 Day Returns'][currencies_labels].copy()

# Removing nan dates
currencies_returns = currencies_returns.dropna()



currencies_pca = PCA(n_components=currencies_returns.shape[1])
currencies_pca.fit(currencies_returns)

print(currencies_pca.explained_variance_ratio_)

#%% Metals Cluster


metals_labels = ['COMEX Copper',
                                                 'SHFE Natural Rubber',
                                                 'SHFE Rebar', 
                                                 'SHFE Lead',
                                                 'SHFE Aluminium',
                                                 'SHFE Copper',
                                                 'SHFE Zinc']
# Indicies Cluster
metals_returns = df_dict['Std 2 Day Returns'][metals_labels].copy()

# Removing nan dates
metals_returns = metals_returns.dropna()


metals_pca = PCA(n_components=metals_returns.shape[1])
metals_pca.fit(metals_returns)

print(metals_pca.explained_variance_ratio_)

#%% Gold Cluster

gold_labels = ['NYMEX Gold',
                                                 'NYMEX Palladium',
                                                 'NYMEX Silver', 
                                                 'NYMEX Platinum']
gold_returns = df_dict['Std 2 Day Returns'][gold_labels].copy()

# Removing nan dates
gold_returns = gold_returns.dropna()


gold_pca = PCA(n_components=gold_returns.shape[1])
gold_pca.fit(gold_returns)

print(gold_pca.explained_variance_ratio_)

#%% Softs Cluster

softs_labels = ['CBOT Soybean Oil',
                                              'CBOT Soybean Meal',
                                              'CBOT Soybeans',
                                              'CBOT Wheat',
                                              'CBOT Corn',
                                              'CBOT Oats',
                                              'CME Kansas City Wheat',
                                              'MGEX Hard Red Spring Wheat']
softs_returns = df_dict['Std 2 Day Returns'][softs_labels].copy()

# Removing nan dates
softs_returns = softs_returns.dropna()


softs_pca = PCA(n_components=softs_returns.shape[1])
softs_pca.fit(softs_returns)

print(softs_pca.explained_variance_ratio_)


#%% Bonds Cluster

bonds_labels = ['LIFFE EURIBOR',
                                              'LIFFE Short Sterling',
                                              'EUREX EuroOAT',
                                              'LIFFE Long Gilt',
                                              'EUREX EuroSchatz',
                                              'EUREX EuroBobl',
                                              'EUREX EuroBund',
                                              'CME Eurodollar',
                                              'CBOT 2-year US Treasury Note',
                                              'CBOT 30-year US Treasury Bond',
                                              'CBOT 10-year US Treasury Note',
                                              'CBOT 5-year US Treasury Note']
bonds_returns = df_dict['Std 2 Day Returns'][bonds_labels].copy()

# Removing nan dates
bonds_returns = bonds_returns.dropna()


bonds_pca = PCA(n_components=bonds_returns.shape[1])
bonds_pca.fit(bonds_returns)

print(bonds_pca.explained_variance_ratio_)

#%% Visualisation of PCs

def PCA_visualiser(pca_fit, y_lab, col_vec, title):


    x_axis_labels = ['PC{} ({:.2f}%)'.format(i+1, pca_fit.explained_variance_ratio_[i]*100) for i in range(len(col_vec))]

    plot_comps = [a*b for a,b in zip(pca_fit.components_[:len(col_vec)], col_vec)]

    # create seabvorn heatmap with required labels
    sns.heatmap(np.transpose(plot_comps), xticklabels=x_axis_labels, yticklabels=y_lab)
    plt.title('Top {} Principal Components of {} Cluster'.format(len(col_vec), title))
    # plt.show()



PCA_visualiser(equities_pca, equities_labels, [-1,-1,-1], 'Equities')
#%%
PCA_visualiser(energies_pca, energies_labels, [-1,-1,-1], 'Energies')
#%%
PCA_visualiser(currencies_pca, currencies_labels, [1,1,-1], 'Currencies')
#%%
PCA_visualiser(metals_pca, metals_labels, [-1,1,1], 'Metals')
#%%
PCA_visualiser(gold_pca, gold_labels, [1,1,-1], 'Precious Metals')
#%%
PCA_visualiser(softs_pca, softs_labels, [1,1,1,1], 'Softs')
#%%
PCA_visualiser(bonds_pca, bonds_labels, [-1,-1,-1,1,1], 'Bonds')
#%% Stability of PC through time


cluster_keys = [equities_labels, energies_labels, currencies_labels, metals_labels, gold_labels, softs_labels, bonds_labels]
cluster_names = ['Equities', 'Energies', 'Currencies', 'Metals', 'Precious Metals','Softs', 'Bonds']

# Calculating from which date to begin (by finding smallest contract dataset)
min_date = df_dict['Std 2 Day Returns']['SHFE Zinc'].dropna().index[0]
smallest_contract = 'SHFE Zinc'

for cluster in cluster_keys:
    for contract in cluster:
        # print(len(df_dict['Std 2 Day Returns'][contract].dropna()), contract)
        if df_dict['Std 2 Day Returns'][contract].dropna().index[0] > min_date:
            min_date = df_dict['Std 2 Day Returns'][contract].dropna().index[0]
            smallest_contract = contract
print(smallest_contract, min_date)                                           
#%% 
    
for i, cluster in enumerate(cluster_keys): 
    # print(cluster)
    prop_of_var(trimmed_dates[cluster], cluster_names[i], 80)                                                                                                                                   
#%% 
# Trimming to desired date with mask
mask = df_dict['Std 2 Day Returns'].index >= '2012-05-17'
trimmed_dates = df_dict['Std 2 Day Returns'][mask].copy()


proportion_dict = dict(zip([],[]))
def prop_of_var(returns_data, title, lookback):
    
    # Dropping nan rows
    returns_data = returns_data.dropna()
    # print(returns_data.shape)
    
    # Creating list of pcas for look back window
    proportion_list = []
    
    for i in range(returns_data.shape[0]-lookback):
        
        # Splice data frame
        returns_window = returns_data.iloc[i:i+lookback,:]
        
        # Perform PCA
        cur_pca = PCA(n_components = returns_data.shape[1])
        cur_pca.fit(returns_window)
        
        # Assign top PC explained variance to column
        proportion_list.append(cur_pca.explained_variance_ratio_)
        proportion_array = np.array(proportion_list).transpose()
    
    # Plotting array
    # print(returns_data.index)
    # print(pd.date_range(start = returns_data.index[0], end = returns_data.index[-1], periods = len(proportion_array)+1))
    pal = sns.color_palette("Set1")
    plt.stackplot(np.array(returns_data.index)[lookback:], proportion_array, labels=['PC{}'.format(i+1) for i in range(returns_data.shape[1])],colors=pal,alpha=0.4)
    # plt.stackplot(pd.date_range(start = returns_data.index[0], end = returns_data.index[-1], periods = np.int(len(returns_data)/30)), proportion_array, labels=['PC{}'.format(i+1) for i in range(returns_data.shape[1])],colors=pal,alpha=0.4)
    plt.margins(0,0)
    plt.title('PC Explained Variance in {} Cluster'. format(title))
    plt.legend(loc='lower left')
    plt.show()
#%% Rolling PCA Projection


proportion_dict = dict(zip([],[]))
def PC_proj_ts(returns_data, lookback, title):

    # Initialising empty projection array
    proj_col = [np.nan]*returns_data.shape[0]

    # Rolling through each day
    for t in range(returns_data.shape[0]-lookback):
        
        # Splice data frame to last ookback-1 days
        returns_window = returns_data.iloc[t:t+lookback-1,:]
        
        # Perform PCA on all non-na rows
        cur_pca = PCA(n_components = returns_data.shape[1], svd_solver='full')
        cur_pca.fit(returns_window.dropna())
        
        # Check for NaN in day t+1
        if returns_data.iloc[t+lookback,:].dropna().shape[0] < returns_data.shape[1]:
            # If PCA cant be computed, set projection to np.nan
            proj_col[t+lookback] = np.nan
        else:
            # Project day t's returns onto PC space and take first component
            day_t_proj = cur_pca.transform(np.array(returns_data.iloc[t+lookback,:]).reshape(1,-1))
            proj_col[t+lookback] = day_t_proj[0][0]

            
    # Save results into a dataframe
    proj_df = pd.DataFrame([], index = returns_data.index)
    proj_df['1st PC Projection'] = proj_col
    
    # Plot resulting time series
    plt.figure(figsize=(20,10))    
    plt.plot(proj_df['1st PC Projection'].iloc[lookback:], lw=1)
    plt.xlabel('Year')
    plt.ylabel('Projection onto 1st PC')
    plt.title('Projection of {} Returns onto 1st PC'.format(title))
    # plt.legend()
    plt.show()
    
    return proj_df
#%% Create index of commodities

    
# commodities_labels = energies_labels+metals_labels+softs_labels+gold_labels
commods_index_labels = ['CBOT Corn',
                        'CBOT Oats',
                        'CBOT Rough Rice',
                        'CBOT Soybean Meal',
                        'CBOT Soybean Oil',
                        'CBOT Soybeans',
                        'CBOT Wheat',
                        'CME Class III Milk',
                        'CME Kansas City Wheat',
                        'CME Lean Hogs',
                        'CME Live Cattle',
                        'CME Random Length Lumber',
                        'COMEX Copper',
                        'ICE Brent Crude Oil',
                        'ICE Cocoa',
                        'ICE Coffee C',
                        'ICE Cotton',
                        'ICE Gasoil',
                        'ICE Heating Oil',
                        'ICE Orange Juice',
                        'ICE Sugar No11',
                        'ICE WTI Crude Oil',
                        'LIFFE London Cocoa',
                        'MGEX Hard Red Spring Wheat',
                        'NYMEX Gasoline',
                        'NYMEX Gold',
                        'NYMEX Heating Oil',
                        'NYMEX Natural Gas',
                        'NYMEX Palladium',
                        'NYMEX Platinum',
                        'NYMEX Silver',
                        'NYMEX WTI Crude Oil']



commodities_2013 = trimmed_dates[commods_index_labels].drop(energies_labels, axis = 1)
commodities_2013 = commodities_2013.drop('LIFFE London Cocoa', axis = 1)
#%%
PC_proj_df = PC_proj_ts(commodities_2013, 180, 'Commodities')
#%% Predictor Columns creation


# Create average column of commods currencies
# PC_proj_df['Commods Currencies Avg'] = trimmed_dates[['CME Australian Dollar AUD',
#                                                       'CME Mexican Peso',
#                                                       'CME Canadian Dollar CAD']].mean(axis = 1)

PC_proj_df['Commods Currencies Avg'] = (trimmed_dates['CME Mexican Peso'] + trimmed_dates['CME Australian Dollar AUD'] + trimmed_dates['CME Canadian Dollar CAD']) / 3
 

# US Dollar Index Predictor
PC_proj_df['US Dollar Index'] = trimmed_dates['ICE US Dollar Index']
# Energies Index Predictor
PC_proj_df['Energies Index'] = trimmed_dates[energies_labels].drop('ICE Heating Oil', axis = 1).mean(axis = 1)


# Yield Curve Predictors
# 5/10 Year Yield Curve
yc_two_ten = pd.DataFrame([], index = trimmed_dates.index)
yc_two_ten['Difference'] = contract_dict['CBOT 2-year US Treasury Note']['Close'] - contract_dict['CBOT 10-year US Treasury Note']['Close']
yc_two_ten['2 Day Returns'] = yc_two_ten.diff(periods=2)
yc_two_ten['Rolling EWM Std'] = yc_two_ten['2 Day Returns'].ewm(halflife = 15).std()
yc_two_ten['2 Day Std Returns'] = yc_two_ten['2 Day Returns'] / yc_two_ten['Rolling EWM Std']

# 5/10 Year Yield Curve
yc_two_five = pd.DataFrame([], index = trimmed_dates.index)
yc_two_five['Difference'] = contract_dict['CBOT 2-year US Treasury Note']['Close'] - contract_dict['CBOT 5-year US Treasury Note']['Close']
yc_two_five['2 Day Returns'] = yc_two_five.diff(periods=2)
yc_two_five['Rolling EWM Std'] = yc_two_five['2 Day Returns'].ewm(halflife = 15).std()
yc_two_five['2 Day Std Returns'] = yc_two_five['2 Day Returns'] / yc_two_five['Rolling EWM Std']

# 5/10 Year Yield Curve
yc_five_ten = pd.DataFrame([], index = trimmed_dates.index)
yc_five_ten['Difference'] = contract_dict['CBOT 5-year US Treasury Note']['Close'] - contract_dict['CBOT 10-year US Treasury Note']['Close']
yc_five_ten['2 Day Returns'] = yc_five_ten.diff(periods=2)
yc_five_ten['Rolling EWM Std'] = yc_five_ten['2 Day Returns'].ewm(halflife = 15).std()
yc_five_ten['2 Day Std Returns'] = yc_five_ten['2 Day Returns'] / yc_five_ten['Rolling EWM Std']




# Adding to df
PC_proj_df['2/10 Year Yield Curve'] = yc_two_ten['2 Day Std Returns']
PC_proj_df['2/5 Year Yield Curve'] = yc_two_five['2 Day Std Returns']
PC_proj_df['5/10 Year Yield Curve'] = yc_five_ten['2 Day Std Returns']


# Taking first principal component of yield curve and energies returns
PC_proj_df['Energies PC'] = PC_proj_ts(trimmed_dates[energies_labels].drop('ICE Heating Oil', axis = 1), 180, 'Energies')
PC_proj_df['Yield Curve PC'] = PC_proj_ts(PC_proj_df[['2/10 Year Yield Curve', '2/5 Year Yield Curve', '5/10 Year Yield Curve']], 180, 'Yield Curve')

# PC_proj_df['Yield Curve Average'] = (yc_five_ten['2 Day Std Returns'] + yc_two_five['2 Day Std Returns'] + yc_two_ten['2 Day Std Returns']) / 3
# PC_proj_df = PC_proj_df.drop(['2/10 Year Yield Curve', '2/5 Year Yield Curve', '5/10 Year Yield Curve'], axis = 1)
#%% Seaborn Clustermap Plot of predictor columns



# Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

pred_corr_map = sns.clustermap(PC_proj_df.corr(), xticklabels = PC_proj_df.columns, yticklabels = PC_proj_df.columns, cmap="RdYlGn", center=0, square=True)

pred_corr_map.fig.suptitle('Correlation of Predictors in Rolling Linear Regression')

hm = pred_corr_map.ax_heatmap.get_position()
plt.setp(pred_corr_map.ax_heatmap.yaxis.get_majorticklabels(), fontsize=6)
plt.setp(pred_corr_map.ax_heatmap.xaxis.get_majorticklabels(), fontsize=6)
pred_corr_map.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height])
col = pred_corr_map.ax_col_dendrogram.get_position()
pred_corr_map.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height])
#%% Perform rolling linear regression

from sklearn.linear_model import LinearRegression


def rolling_lr(outcome, predictors, lookback, intercept):

    # Merging input data
    full_df = predictors.join(outcome, on = outcome.index)
    
    # Initialising empty array for beta coefficients
    beta_df = pd.DataFrame([[np.nan]*predictors.shape[1]]*predictors.shape[0], columns = predictors.columns, index = predictors.index)
    
    # Initialising empty array for R^2
    r_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
    # Initialising empty array for MSE
    mse_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
    # Initialising empty array for prediction
    pred_ts = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
    # Rolling through each day
    for t in range(predictors.shape[0]-lookback):
        
        # Splice data frame to the last lookback-1 days
        regression_window = full_df.iloc[t:t+lookback-1,:].dropna()
        
        # Perform linear regression
        cur_lr = LinearRegression(fit_intercept=intercept)
        cur_lr.fit(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
        # Save beta values for current day
        beta_df.iloc[t+lookback-1,:] = cur_lr.coef_
        
        # Save R^2 for current day
        r_df.iloc[t+lookback] = cur_lr.score(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
        # Save MSE for current day
        mse_df.iloc[t+lookback] = np.square(cur_lr.predict(regression_window.iloc[:,1:]) - regression_window.iloc[:,0]).mean()
        
        # Save prediction for current day
        pred_ts.iloc[t+lookback] = cur_lr.predict(regression_window.iloc[:,1:])[-1]
    
    # Plot beta time series
    plt.figure(figsize=(20,10))
    for col in beta_df.columns:    
        plt.plot(beta_df[col].iloc[lookback:], lw=1, label = col)
    plt.xlabel('Year')
    plt.ylabel('Value of Coefficicent in Linear Regression')
    plt.title('Beta Coefficients in Rolling Linear Regression')
    plt.legend(loc=3)
    plt.show()
    
    # Plot coefficient of determination time series
    plt.figure(figsize=(20,10))
    plt.plot(r_df[lookback:], lw=1, label = 'R Squared')
    plt.plot(mse_df[lookback:], lw=1, label = 'MSE')
    plt.xlabel('Year')
    plt.ylabel('Coefficient of Determination')
    plt.title('Plot of R^2 Over Time in Rolling Linear Regression')
    plt.legend(loc=3)
    plt.show()
    
    
    # Plot fitted time series against observed time series
    print(pred_ts[lookback:] - outcome[lookback:])
    # plt.figure(figsize=(20,10))
    # plt.plot(pred_ts[lookback:] - outcome[lookback:], lw=1, label = 'Prediction')
    # # plt.plot(outcome[lookback:], lw=1, label = 'True Outcome')
    # plt.xlabel('Year')
    # plt.ylabel('Coefficient of Determination')
    # plt.title('Plot of Prediction Compared to True Outcome')
    # plt.legend(loc=3)
    # plt.show()
    
    return beta_df

test = rolling_lr(PC_proj_df['1st PC Projection'], PC_proj_df[['Commods Currencies Avg',
                                                               'US Dollar Index',
                                                               'Energies PC',
                                                               'Yield Curve PC']],500, False)

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