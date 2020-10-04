import pandas as pd
import numpy as np
import seaborn as sns
#%% Data Loading

# Reading in contract list
contract_list = pd.read_csv("/Users/andrewmelville/Documents/SummerFinanceProject/Future Contract List.csv")
contract_names = contract_list['NAME']

# Importing from file names and creating dctionary of dataframes for each variable and each contract
contract_dict = dict(zip([],[]))
for i, contract in enumerate(contract_names):
    contract_dict[contract]  = pd.DataFrame([], index = pd.bdate_range(start = '1/1/1980', end = '7/31/2020'))

    current = pd.read_csv("/Users/andrewmelville/Documents/SummerFinanceProject/ContinuousSeries/{}.csv".format(contract), index_col = 0, skiprows = 0, skipfooter = 1, header = 1, engine = 'python')
    current.index = pd.to_datetime(current.index)
    
    contract_dict[contract] = contract_dict[contract].join(current)# -*- coding: utf-8 -*-
    
 #%% Label setting


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
        
#%% Taking trimmed dates (2013 onwards)
        
# Trimming to desired date with mask
mask = df_dict['Std 2 Day Returns'].index >= '2012-05-17'
trimmed_dates = df_dict['Std 2 Day Returns'][mask].copy()
