from sklearn.decomposition import PCA
from preprocessing import df_dict, equities_labels, energies_labels, currencies_labels, metals_labels, gold_labels, softs_labels, bonds_labels
#%% Equities Cluster
# Creating sub dataframes for obvious clusters


# Indicies Cluster
equities_returns = df_dict['Std 2 Day Returns'][equities_labels].copy()

# Removing nan dates
equities_returns = equities_returns.dropna()


#'CME Russell 1000 Index Mini Futures' is removed as it only has 1024 available dates


equities_pca = PCA(n_components=equities_returns.shape[1])
equities_pca.fit(equities_returns)



energies_returns = df_dict['Std 2 Day Returns'][energies_labels].copy()

# Removing nan dates
energies_returns = energies_returns.dropna()




energies_pca = PCA(n_components=energies_returns.shape[1])
energies_pca.fit(energies_returns)


## 'ICE British Pound GBP dropped due to correlation with CME and less data

# Indicies Cluster
currencies_returns = df_dict['Std 2 Day Returns'][currencies_labels].copy()

# Removing nan dates
currencies_returns = currencies_returns.dropna()



currencies_pca = PCA(n_components=currencies_returns.shape[1])
currencies_pca.fit(currencies_returns)


# Indicies Cluster
metals_returns = df_dict['Std 2 Day Returns'][metals_labels].copy()

# Removing nan dates
metals_returns = metals_returns.dropna()


metals_pca = PCA(n_components=metals_returns.shape[1])
metals_pca.fit(metals_returns)


gold_returns = df_dict['Std 2 Day Returns'][gold_labels].copy()

# Removing nan dates
gold_returns = gold_returns.dropna()


gold_pca = PCA(n_components=gold_returns.shape[1])
gold_pca.fit(gold_returns)


softs_returns = df_dict['Std 2 Day Returns'][softs_labels].copy()

# Removing nan dates
softs_returns = softs_returns.dropna()


softs_pca = PCA(n_components=softs_returns.shape[1])
softs_pca.fit(softs_returns)


bonds_returns = df_dict['Std 2 Day Returns'][bonds_labels].copy()

# Removing nan dates
bonds_returns = bonds_returns.dropna()


bonds_pca = PCA(n_components=bonds_returns.shape[1])
bonds_pca.fit(bonds_returns)



#%% Principal Component Analysis 


# equities_transformed_returns = equities_pca.transform(equities_returns)

# # Getting transformed unit basis vectors in PC space
# equities_basis_transformed = [equities_pca.transform(a.reshape(1,-1)) for a in np.diag([1]*len(equities_labels))]
# equities_basis_transformed = [a / np.linalg.norm(a) for a in equities_basis_transformed]
# # equities_basis_transformed = np.array(equities_basis_transformed)

# for vector in equities_basis_transformed:
#     plt.quiver(0,0,vector[0][0],vector[0][1],scale=3)
#     # plt.quiver([0]*len(equities_labels),equities_basis_transformed[0],equities_basis_transformed[1])
# plt.scatter(equities_transformed[:,0], equities_transformed[:,1], s = 0.5)