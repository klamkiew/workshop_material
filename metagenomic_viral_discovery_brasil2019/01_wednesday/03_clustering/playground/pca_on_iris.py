#!/usr/bin/env python3

"""
This Python3 script shows how to perform a PCA with the 
implementation of scikit-learn. We will use the iris dataset
and make use of pandas and numpy in order to transform our data 
in a DataFrame that can be used for the PCA.
This tutorial is inspired / adapted from Michael Galarnyk, see:
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


"""

## Import Section

# we need this once again for the iris dataset
from sklearn import datasets
# this is needed for the actual PCA
from sklearn.decomposition import PCA
# this will be needed to normalize the data
from sklearn.preprocessing import StandardScaler
# we will use numpy and pandas for our data structures
import numpy as np 
import pandas as pd 
# matplotlib is needed for some nice plots in the end
import matplotlib.pyplot as plt

# only do this if the script gets called 
# from the command line
if __name__ == '__main__':

    iris = datasets.load_iris()

    # that line is a tough one:
    # first, we want to create a dataframe object with pandas.
    # for this, we need two parameters, namely the data and the column names.
    # the data is taken from the iris['data'] array and the actual target names.
    # we use numpy.c_ function to concatenate the array iris['data'] with the names
    # of the target (note: iris['target'] is a number, which is the index of the target name)
    # Further note the special syntax here: it is np.c_[] and not np.c_()
    # honestly, I have no idea why it is that way though...
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target_names'][iris['target']]], 
                  columns= iris['feature_names'] + ['target'])

    # now we want to normalize our data
    # since scaling influences the PCA

    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # Here we seperate features from the target column
    # since we only want to normalize the actual data - not the targets
    x = df.loc[:, features].values
    y = df.loc[:,['target']].values
    
    # For normalization we use the StandardScaler from
    # sklearn.preprocessing
    x = StandardScaler().fit_transform(x)

    # last but not least, we are applying the PCA on the data
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    # this attribute of the PCA object tells
    # us, how much variance is explained by the
    # left dimension. In this case we reduced the 
    # 4-dimensional dataset to two dimension, thus,
    # we are expecting information loss!!!
    # Usually you want to report these values on the axes
    # of your plot as well.
    print("The reduced dimensions explain these ratios of variance, respectively:")
    print(pca.explained_variance_ratio_)
    ratios = pca.explained_variance_ratio_


    # creating a new DataFrame here, which will be used for plotting
    principalDf = pd.DataFrame(data = principalComponents
         , columns = ['principal component 1', 'principal component 2'])
    # the DF is merged with the targets once again - we will need this for
    # coloring the PCA
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

    # now comes the plotting.
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(f'PC1 - {ratios[0]:,.3f}', fontsize = 15)
    ax.set_ylabel(f'PC2 - {ratios[1]:,.3f}', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.grid()
    targets = ['setosa', 'versicolor', 'virginica']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    
    print()
    print("Figure saved at iris_pca_2components.pdf")
    fig.savefig("iris_pca_2components.pdf")
