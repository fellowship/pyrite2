
"""

    Feature Importance is added to the implementation, and it
    is not part of the original paper.

    Note: Incomplete features:
        - Exception handling for input parameters

"""
import numpy as np
import math
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

####################################

def sibyl(dataset, ssamples_num, ssample_size, anomalies=None):
    """
    Input:
    dataset: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    ssamples_num: number of subsamples
    ssample_size: subsample size
    anomalies: list of indices of dataset instances whose feature importance is desired
    Output:
    scores: Score for each instance in the dataset (np array)
    or
    (scores, anomalies) if anomalies is passed where anomalies has attached column of frequently absent feature pairs
    """


    (n,d) = dataset.shape
    columns = dataset.columns

    zeros = np.zeros(d)
    scores = np.zeros(n)
    global theta

    # Loop over all subsamples
    for i in range(0,ssamples_num):

        #create a subsample of indices
        d_i = random.sample(range(0,n),ssample_size)
        dataset = np.array(dataset)
        ssample = np.array(dataset[d_i])
        theta = np.zeros(dataset.shape)


        #create random d random subspaces by permuting columnss 0:d-1 and pairing adjacent values
        # randomly reorder set of column names
        s = random.sample(range(0,d),d)
        sspace_0 = s[0:len(s)]
        sspace_1 = s[1:len(s)]+[s[0]]

        # create frequency table
        # in this version we compare each row of a subsample to the whole data frame
        # and then sum them up along the subsample in a global theta variable
        np.apply_along_axis(computeFrequency,1,ssample,dataset, sspace_0,sspace_1)
        scores = scores + np.sum(theta == zeros,axis = 1)

    return 1.0*scores/ssamples_num/d
#####################################

def computeFrequency(single_instance, dataset,sspace_0,sspace_1):

    """
    Helper method - Compute Frequency of "single_instance" in "ssample"

    Input:
    single_instance: A single instance out of subsample
    ssample: A subsample
    s: A list of subspaces

    Output:
    freq_in_subspaces: For the current subsample, it's a "pandas series" of frequencies
    of single_instance in each subspace
    """

    global theta
    n, d = dataset.shape

    # elementwise comparison operation between the "single_instance" in subsample and each raw in the original data frame
    # summed up along subsample in a global theta array simultaneously for all the instances
    occurances = single_instance==dataset
    occurances0 = occurances[:,sspace_0]
    occurances1 = occurances[:,sspace_1]

    theta = theta + occurances0*occurances1

    return 0

#####################################

def sibyl_score(single_instance,dataset,ssamples_num,ssample_size):
    """
    Same as sibyl but scores a single instance.

    Input:
    single_instance: instances whose score is desired
    dataset: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    ssamples_num: number of subsamples
    ssample_size: subsample size

    Output:
    score: Score of df_row_idx indexed instance float number
    """
    n,d = dataset.shape

    single_instance = np.array(single_instance)
    dataset = np.array(dataset)

    zeros = np.zeros(d)
    score = 0

    # Loop over all subsamples
    for i in range(0,ssamples_num):

        #create a subsample of indices
        d_i = random.sample(range(0,n),ssample_size)
        ssample = dataset[d_i]

        #create random d random subspaces by permuting columnss 0:d-1 and pairing adjacent values
        # randomly reorder set of column names
        s = random.sample(range(0,d),d)
        sspace_0 = s[0:len(s)]
        sspace_1 = s[1:len(s)]+[s[0]]

        # compute score
        occurances = single_instance==ssample
        occurances0 = occurances[:,sspace_0]
        occurances1 = occurances[:,sspace_1]
        score = score + np.sum((occurances0*occurances1).sum(axis = 0) == zeros)

    return 1.0*score/d/ssamples_num

####################################

def anomaly_inspect(single_instance,dataset, plot = False):
    """
    Compute the inverse relative frequency of a category (pair of categories) in each column (pairs of columns)
    for categories in an anomalous instance index by idx.
    inverse relative frequency = total # of instances/(# of instances with fixed category
                                                        X # of categories for that feature )
    Plot relative frequencies (optional).

    Input:
    dataset: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    idx: index of anomaly in the data frame dataset
    plot: True - plot, False - do not plot

    Output:
    (freq_1_d,freq_2d)
        freq_1d: inverse relative frequencies for categories 1xd np array
        freq_2d: inverse relative frequencies for pairs of categories dxd np array
    """
    n, d = dataset.shape

    colnames = dataset.columns
    sizes = []
    for c in colnames:
        sizes = sizes + [len(dataset[c].unique())]
    sizes = np.array(sizes)

    single_instance = np.array(single_instance)
    dataset = np.array(dataset)


    # elementwise comparison operation between outlier single_instance and each raw in the original data frame
    # summed up along columns and divided by total number of elements per category
    occurances = (single_instance==dataset)
    freq_1d = 1.0/((occurances.sum(axis = 0)/(1.0*n/sizes)))

    #form all feature pairs
    freq_2d = np.zeros((d,d))
    for i in range(0,d-1):
        for j in range(i+1,d):
            freq_2d[i,j] = 1.0/(1.0*(occurances[:,i]*occurances[:,j]).sum(axis = 0)/(1.0*n/sizes[i]/sizes[j]))


    # plot tables
    if plot:
        fig = plt.figure(figsize=(10,15))
        ax = fig.add_subplot(2,1,1)
        plt.subplots_adjust(hspace = 0.5)
        w=0.35
        ax.bar(range(0,d),freq_1d)
        ax.set_xlim(0,d)
        ax.set_ylim(0,max(freq_1d)*1.1)
        ax.set_ylabel('1/relative frequency of a category', fontsize = 14)
        ax.set_title('Category inverted relative frequencies\n'+
                     '(frequency of specified category times number of categories)\n'+
                     'for each column',
                        fontsize =16,)
        xTickMarks = [str(colnames[i]) + ': ' + single_instance[i] for i in range(0,d)]
        ax.set_xticks(np.array(range(0,d))+0.5)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=90, fontsize=10)

        ax1 = fig.add_subplot(2,1,2)
        ax1.set_ylabel('1/relative frequency of a pair of categories', fontsize = 14)
        ax1.set_title('Pairs of categories inverted relative frequencies\n'+
                      '(frequency of specified category times number ofcategories)\n'+
                      'for each column',
                         fontsize = 16)
        im = ax1.imshow(freq_2d.T,interpolation = "none")
        tickMarks = [str(colnames[i]) + ': ' + single_instance[i] for i in range(0,d)]
        ax1.set_xticks(np.array(range(0,d))+0)
        ax1.set_yticks(np.array(range(0,d))+0)
        xtickNames = ax1.set_xticklabels(tickMarks)
        ytickNames = ax1.set_yticklabels(tickMarks)
        plt.setp(xtickNames, rotation=90, fontsize=10)
        plt.setp(ytickNames, rotation=0, fontsize=10)
        plt.colorbar(im)
        plt.show()

    return (freq_1d,freq_2d)

################################

def get_feature_importance(single_instance,dataset):
    """
    Calls anomaly inspect(), selects maximum elements of the tables and organizes output into dictionary

    Input:
    dataset: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    idx: index of anomaly in the data frame dataset

    Output:
    dictionary with locations and scores of single most rare feature and single most rare column
    """
    t1,t2 = anomaly_inspect(single_instance,dataset, plot = False)
    columns = list(dataset.columns)
    d1_score = t1.max()
    d1_loc = [columns[i] for i in np.where(t1 == t1.max())[0]]

    d2_score = t2.max()
    d2_loc = np.where(t2 == t2.max())
    d2_loc = [(columns[d2_loc[0][i]],columns[d2_loc[1][i]]) for i in range(len(d2_loc[0]))]

    return {'single feature':(d1_loc,d1_score),'feature pairs':(d2_loc,d2_score)}
