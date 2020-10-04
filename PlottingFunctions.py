import matplotlib.pyplot as pltdef series_plot(data, title, xlab = 'Date', ylab = 'Price'):    plt.figure(figsize=(20,10))    plt.title(title)    plt.xlabel(xlab)    plt.ylabel(ylab)        for series in data:        plt.plot(data[series], label = series)    plt.legend()def PCA_loadings_visualiser(pca_fit, y_lab, col_vec, title):    x_axis_labels = ['PC{} ({:.2f}%)'.format(i+1, pca_fit.explained_variance_ratio_[i]*100) for i in range(len(col_vec))]    plot_comps = [a*b for a,b in zip(pca_fit.components_[:len(col_vec)], col_vec)]    # create seabvorn heatmap with required labels    sns.heatmap(np.transpose(plot_comps), xticklabels=x_axis_labels, yticklabels=y_lab)    plt.title('Top {} Principal Components of {} Cluster'.format(len(col_vec), title))    # plt.show()    def prop_of_var(returns_data, title, lookback):        # Dropping nan rows    returns_data = returns_data.dropna()    # print(returns_data.shape)        # Creating list of pcas for look back window    proportion_list = []        for i in range(returns_data.shape[0]-lookback):                # Splice data frame        returns_window = returns_data.iloc[i:i+lookback,:]                # Perform PCA        cur_pca = PCA(n_components = returns_data.shape[1])        cur_pca.fit(returns_window)                # Assign top PC explained variance to column        proportion_list.append(cur_pca.explained_variance_ratio_)        proportion_array = np.array(proportion_list).transpose()        # Plotting array    # print(returns_data.index)    # print(pd.date_range(start = returns_data.index[0], end = returns_data.index[-1], periods = len(proportion_array)+1))    pal = sns.color_palette("Set1")    plt.stackplot(np.array(returns_data.index)[lookback:], proportion_array, labels=['PC{}'.format(i+1) for i in range(returns_data.shape[1])],colors=pal,alpha=0.4)    # plt.stackplot(pd.date_range(start = returns_data.index[0], end = returns_data.index[-1], periods = np.int(len(returns_data)/30)), proportion_array, labels=['PC{}'.format(i+1) for i in range(returns_data.shape[1])],colors=pal,alpha=0.4)    plt.margins(0,0)    plt.title('PC Explained Variance in {} Cluster'. format(title))    plt.legend(loc='lower left')    plt.show()