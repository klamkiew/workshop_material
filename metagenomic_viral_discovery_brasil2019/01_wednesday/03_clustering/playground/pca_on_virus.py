#!/usr/bin/env python3

"""
This Python3 script shows how to perform a PCA with the 
implementation of scikit-learn. We will use the iris dataset
and make use of pandas and numpy in order to transform our data 
in a DataFrame that can be used for the PCA.

"""

## Import Section

# this is needed for the actual PCA
from sklearn.decomposition import PCA
# this will be needed to normalize the data
from sklearn.preprocessing import StandardScaler
# we will use numpy and pandas for our data structures
import numpy as np 
import pandas as pd 
# matplotlib is needed for some nice plots in the end
import matplotlib.pyplot as plt
# csv is needed to parse our dataset
import csv

# only do this if the script gets called 
# from the command line
if __name__ == '__main__':

    class2id = {
      'EBV' : 0,
      'HCoV' : 1,
      'SARS' : 2
    }

    id2class = {
        0 : 'EBV',
        1 : 'HCoV',
        2 : 'SARS'
    }

    data = [] 
    target = [] 
    target_names = list(class2id.keys())
    with open('virus.csv', 'r') as inputStream: 
        reader = csv.reader(inputStream, delimiter=',') 
        for idx, row in enumerate(reader): 
            if idx == 0: 
                continue 
            data.append(list(map(float,row[:4])))
            target.append(class2id[row[4]])



    # that line is a tough one:
    # first, we want to create a dataframe object with pandas.
    # for this, we need two parameters, namely the data and the column names.
    # the data is taken from the iris['data'] array and the actual target names.
    # we use numpy.c_ function to concatenate the array iris['data'] with the names
    # of the target (note: iris['target'] is a number, which is the index of the target name)
    # Further note the special syntax here: it is np.c_[] and not np.c_()
    # honestly, I have no idea why it is that way though...
    df = pd.DataFrame(data= np.c_[data, [id2class[x] for x in target]], 
                  columns= ['feature1','feature2','feature3','feature4'] + ['target'])

    # now we want to normalize our data
    # since scaling influences the PCA

    features = ['feature1','feature2','feature3','feature4']
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
    targets = ['EBV', 'HCoV', 'SARS']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    
    print()
    print("Figure saved at virus_pca_2components.pdf")
    fig.savefig("virus_pca_2components.pdf")
