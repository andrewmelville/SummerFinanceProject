from lin_reg_analysis import rolling_lr, commodities_2013
from preprocessing import trimmed_dates, df_dict
from PlottingFunctions import series_plot
import pandas as pd
import numpy as np

# Create column for average of currency basket
cur_commod_avg = pd.DataFrame(trimmed_dates[['CME Australian Dollar AUD',
                         'CME Mexican Peso',
                         'CME Canadian Dollar CAD']].mean(axis = 1), columns = ['Commodity Currencies Simp Avg'])

# Create empty residual dataframe
residual_df = pd.DataFrame().reindex_like(commodities_2013)

# Perfrom rolling regression and fill residual dataframe for each contract in commodities basket
for i, contract in enumerate(commodities_2013):
    
    # Take prediction from rolling linear regression
    pred = rolling_lr(pd.DataFrame(commodities_2013[contract]), cur_commod_avg, lookback = 200, intercept = False)[1]
    
    # Set residual column for current contract
    residual_df[contract] = commodities_2013[contract] - pred['Prediction']
    
    # Output progress
    print('{} residuals completed {}/{}'.format(contract, i+1, commodities_2013.shape[1]))
residual_df = residual_df.fillna(method='ffill')
#%%         Trading strategy: 
# Long bottom three negative residuals, Short top three postive residuals.




# Create empty signals df
signal_df = pd.DataFrame([0]).reindex_like(residual_df)
signal_df = signal_df.fillna(0)

# Loop through each month and make a df of that months residuals
for i in range(2012,2021):
    for j in range(1,13):
        
        mask = (residual_df.index > '{}-{}-01'.format(i,j)) & (residual_df.index <= '{}-{}-{}'.format(i,j,calendar.monthrange(i,j)[1]))
        month = residual_df.loc[mask]
        current_month_list = month.mean().sort_values(axis=0,ascending = False)
        
        # Get top three positive residual contracts
        pos_mask = current_month_list > 0
        sell_list = current_month_list[pos_mask]

        # Mask to select month ahead
        signal_mask = (signal_df.index > pd.to_datetime('{}-{}-01'.format(i,j)) + relativedelta(months=1)) & (signal_df.index <= pd.to_datetime('{}-{}-{}'.format(i,j,calendar.monthrange(i,j)[1])) + relativedelta(months=1))
        
        # Assign negative value to contracts for month ahead
        signal_df.loc[signal_mask, sell_list.index[:3]] = -1
        
        
        # Get bottom three negative residual contracts
        neg_mask = current_month_list < 0
        buy_list = current_month_list[neg_mask]

        # Assign postive value to contracts for month ahead
        signal_df.loc[signal_mask, buy_list.index[-3:]] = 1
    

# Get daily (simple) commodities returns with same contracts as signal_df
daily_returns = df_dict['Close'][signal_df.columns].loc[signal_df.index].fillna(method='ffill').diff()

# Multiply signals df by simple daily returns df to get daily P/L for each contract
contract_PL = signal_df * daily_returns

# Sum across columns for daily P/L, cumsum daily P/L for P/L curve
PL_curve = contract_PL.sum(axis=1).cumsum()

# Plot P/L Curve
series_plot(pd.DataFrame(PL_curve),'P ')
#%%     Trading Strategy:
# Long top three and Short bottom three residuals regardless of direction.



# Create empty signals df
signal_df = pd.DataFrame([0]).reindex_like(residual_df)
signal_df = signal_df.fillna(0)

# Loop through each month and make a df of that months residuals
for i in range(2012,2021):
    for j in range(1,13):
        
        mask = (residual_df.index > '{}-{}-01'.format(i,j)) & (residual_df.index <= '{}-{}-{}'.format(i,j,calendar.monthrange(i,j)[1]))
        month = residual_df.loc[mask]
        current_month_list = month.mean().sort_values(axis=0,ascending = False)
        
        # Go long bottom three contracts (signal shows over-performance)
        buy_list = current_month_list[-3:]
        
        # Mask to select month ahead
        signal_mask = (signal_df.index > pd.to_datetime('{}-{}-01'.format(i,j)) + relativedelta(months=1)) & (signal_df.index <= pd.to_datetime('{}-{}-{}'.format(i,j,calendar.monthrange(i,j)[1])) + relativedelta(months=1))
        
        # Assign positive value to contracts for month ahead
        signal_df.loc[signal_mask, buy_list.index] = 1
        
        
        # Go short top three contracts (signal shows under-performance)
        sell_list = current_month_list[:3]
        
        # Assign negative value to contracts for month ahead
        signal_df.loc[signal_mask, sell_list.index] = -1
    

# Get daily (simple) commodities returns with same contracts as signal_df
daily_returns = df_dict['Close'][signal_df.columns].loc[signal_df.index].fillna(method='ffill').diff()

# Multiply signals df by simple daily returns df to get daily P/L for each contract
contract_PL = signal_df * daily_returns

# Sum across columns for daily P/L, cumsum daily P/L for P/L curve
PL_curve = contract_PL.sum(axis=1).cumsum()

# Plot P/L Curve
series_plot(pd.DataFrame(PL_curve),'P/L Curve')