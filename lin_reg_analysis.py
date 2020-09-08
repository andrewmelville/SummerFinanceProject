from preprocessing import contract_dict, df_dict, equities_labels, energies_labels, currencies_labels, metals_labels, gold_labels, softs_labels, bonds_labels
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings("ignore")
#%% Rolling PCA Projection

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


from preprocessing import trimmed_dates
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
# # Energies Index Predictor
# PC_proj_df['Energies Avg'] = trimmed_dates[energies_labels].drop('ICE Heating Oil', axis = 1).mean(axis = 1)

# Energies reduced to just NYMEX WTI Crude Oil
PC_proj_df['WTI Oil'] = trimmed_dates['NYMEX WTI Crude Oil']

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
# PC_proj_df['Energies Index'] = PC_proj_ts(trimmed_dates[energies_labels].drop('ICE Heating Oil', axis = 1), 180, 'Energies')
PC_proj_df['Yield Curve Index'] = PC_proj_ts(PC_proj_df[['2/10 Year Yield Curve', '2/5 Year Yield Curve', '5/10 Year Yield Curve']], 180, 'Yield Curve')

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

rolling_reg_coeffs = rolling_lr(PC_proj_df['1st PC Projection'], PC_proj_df.iloc[:,1:], 500, False)


# #%% 

# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, mean_squared_error

# hyperparams = {'feature_selection__score_func': [f_regression, mutual_info_regression], 'feature_selection__k': [3, 4, 5]}

# regression_pipeline = Pipeline([('feature_selection', SelectKBest(score_func = mutual_info_regression)), ('regression', LinearRegression())])

# grid_search = GridSearchCV(regression_pipeline, hyperparams, scoring = make_scorer(mean_squared_error))
# PC_proj_df.dropna(inplace = True)
# grid_search.fit(X = PC_proj_df.iloc[:, 1:], y = PC_proj_df.iloc[:, 0])

# features = grid_search.best_estimator_.named_steps['feature_selection'].get_support()
# select_features = PC_proj_df.columns[1:][features]

# print('{} MSE'.format(grid_search.best_score_))

# #%%

# def rolling_lr(outcome, predictors, lookback, intercept):

#     # Merging input data
#     full_df = predictors.join(outcome, on = outcome.index)
    
#     # Initialising empty array for beta coefficients
#     beta_df = pd.DataFrame([[np.nan]*PC_proj_df.columns[1:][features].shape[0]]*predictors.shape[0], columns = PC_proj_df.columns[1:][features], index = predictors.index)
    
#     # Initialising empty array for R^2
#     r_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Initialising empty array for MSE
#     mse_df = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Initialising empty array for prediction
#     pred_ts = pd.DataFrame([[np.nan]*1]*predictors.shape[0], index = predictors.index)
    
#     # Rolling through each day
#     for t in range(predictors.shape[0]-lookback):
        
#         # Splice data frame to the last lookback-1 days
#         regression_window = full_df.iloc[t:t+lookback-1,:].dropna()
        
#         # Perform linear regression
#         grid_search.fit(X = regression_window.iloc[:,1:], y = regression_window.iloc[:,0])
        
#         # Save beta values for current day
#         print(grid_search.best_estimator_['regression'].coef_)
#         break
#         beta_df.iloc[t+lookback-1,:] = grid_search.best_estimator_['regression'].coef_
        
#         # Save R^2 for current day
#         r_df.iloc[t+lookback] = grid_search.score(regression_window.iloc[:,1:], regression_window.iloc[:,0])
        
#         # Save MSE for current day
#         # mse_df.iloc[t+lookback] = np.square(grid_search.predict(regression_window.iloc[:,1:]) - regression_window.iloc[:,0]).mean()
        
#         # Save prediction for current day
#         pred_ts.iloc[t+lookback] = grid_search.predict(regression_window.iloc[:,1:])[-1]
    
#     # Plot beta time series
#     plt.figure(figsize=(20,10))
#     for col in beta_df.columns:    
#         plt.plot(beta_df[col].iloc[lookback:], lw=1, label = col)
#     plt.xlabel('Year')
#     plt.ylabel('Value of Coefficicent in Linear Regression')
#     plt.title('Beta Coefficients in Rolling Linear Regression')
#     plt.legend(loc=3)
#     plt.show()
    
#     # # Plot coefficient of determination time series
#     # plt.figure(figsize=(20,10))
#     # plt.plot(r_df[lookback:], lw=1, label = 'R Squared')
#     # plt.plot(mse_df[lookback:], lw=1, label = 'MSE')
#     # plt.xlabel('Year')
#     # plt.ylabel('Coefficient of Determination')
#     # plt.title('Plot of R^2 Over Time in Rolling Linear Regression')
#     # plt.legend(loc=3)
#     # plt.show()
    
    
#     # Plot fitted time series against observed time series
#     # print(pred_ts[lookback:] - outcome[lookback:])
#     # plt.figure(figsize=(20,10))
#     # plt.plot(pred_ts[lookback:] - outcome[lookback:], lw=1, label = 'Prediction')
#     # # plt.plot(outcome[lookback:], lw=1, label = 'True Outcome')
#     # plt.xlabel('Year')
#     # plt.ylabel('Coefficient of Determination')
#     # plt.title('Plot of Prediction Compared to True Outcome')
#     # plt.legend(loc=3)
#     # plt.show()
    
#     return beta_df

# test = rolling_lr(PC_proj_df['1st PC Projection'], PC_proj_df.iloc[:,1:], 500, intercept=False)
       