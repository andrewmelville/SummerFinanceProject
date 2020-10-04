from lin_reg_analysis import rolling_lr, commodities_2013
from preprocessing import trimmed_dates, df_dict
from PlottingFunctions import series_plot
import pandas as pd
import numpy as np
#%%

# Create average currency column
cur_commod_avg = pd.DataFrame(trimmed_dates[['CME Australian Dollar AUD',
                         'CME Mexican Peso',
                         'CME Canadian Dollar CAD']].mean(axis = 1), columns = ['Commodity Currencies Simp Avg'])

# Create empty beta df
beta_df = pd.DataFrame().reindex_like(commodities_2013)

# Perfrom rolling regression and fill residual df
for i, contract in enumerate(commodities_2013):
    
    beta = rolling_lr(pd.DataFrame(commodities_2013[contract]), cur_commod_avg, lookback = 150, intercept = False)[0]
 
    beta_df[contract] = beta['Commodity Currencies Simp Avg']
    print('{} betas computed {}/{}'.format(contract, i+1, commodities_2013.shape[1]))
# %%

# Initialise balance and price series
live_prices = df_dict['Close'].fillna(method='backfill')
# .copy().fillna(method = 'ffill')

# Split into monthly chunks
monthly_beta = np.array_split(beta_df.dropna(), 150)
month_index = [month.index[0] for month in monthly_beta]

# Create df of monthly signals for each contract
monthly_beta_signals = pd.DataFrame([], columns = beta_df.columns, index = month_index)

# Fill in monthly signals df with mean of signal over the month
for month in monthly_beta:
    print(month.index[0])
    monthly_beta_signals.loc[month.index[0]] = month.mean()
monthly_beta_signals = monthly_beta_signals.dropna()

    

#%%

# Initialise portfolio dataframe
portfolio = pd.DataFrame([]).reindex_like(monthly_beta_signals)
portfolio['Cash'] = np.nan
portfolio.iloc[0,:] = 0
portfolio['Cash'].iloc[0] = 10000

pos_list = []

# Perform buy/sell on monthly basis
for i, month in enumerate(monthly_beta_signals.index[1:]):

    # Carry balance over
    portfolio['Cash'].loc[month] = portfolio['Cash'].iloc[i]
    
    # Update positions prices
    for contract in pos_list:
        portfolio.loc[month][contract] = portfolio[contract].iloc[i] * (live_prices[contract].loc[month] / live_prices[contract].iloc[i])
        # print('Sold: {} of {}'.format(portfolio.loc[month][contract], contract))
        
    # Close previous months positions
    balance = portfolio.loc[month].sum()
    portfolio.loc[month] = 0
    portfolio['Cash'].loc[month] = balance
    
    # Open new positions
    current_month_list = monthly_beta_signals.loc[month].sort_values(axis=0, ascending=False)
    
    pos_mask = current_month_list > 0
    neg_mask = current_month_list < 0
    
    buy_list = current_month_list[neg_mask]
    sell_list = current_month_list[pos_mask]

    pos_list = list(sell_list.index[-3:]) + list(buy_list.index[:3])

    # Add amount bought/sold to each contract, remove same amount from cash balance
    for contract in pos_list:

        portfolio.loc[month][contract] -= 10 * live_prices.loc[month][contract] * monthly_beta_signals.loc[month][contract] 
        portfolio.loc[month]['Cash'] += 10 * live_prices.loc[month][contract] * monthly_beta_signals.loc[month][contract]
    
# Print performance statistics
print('Full Returns of {:.2f}%'.format(portfolio['Cash'][-1]/portfolio['Cash'][0] * 100))
print('Lowest Balance: {:.2f}'.format(min(portfolio['Cash'])))
print('Largest Balance: {:.2f}'.format(max(portfolio['Cash'])))
print('Average Returns: {:.2f}'.format(portfolio['Cash'].diff().mean()))
print('Std Dev of returns: {:.2f}'.format(portfolio['Cash'].diff().std()))