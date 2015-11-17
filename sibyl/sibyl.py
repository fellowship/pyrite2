
"""

    Feature Importance is added to the implementation, and it
    is not part of the original paper.

    Note: Incomplete features:
        - Exception handling for input parameters

"""
import numpy
import math
import pandas
import random
import time
import matplotlib.pyplot as plt

####################################

def sibyl(df,t,phi,anomalies=None):
    """
    Input:
    df: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    t: number of subsamples
    phi: subsample size
    anomalies: list of indices of df instances whose feature importance is desired
    Output:
    scores: Score for each instance in the dataset (numpy array)
    or
    (scores, anomalies) if anomalies is passed where anomalies has attached column of frequently absent feature pairs
    """


    (n,d) = df.shape
    columns = df.columns

    zeros = numpy.zeros(d)
    scores = numpy.zeros(n)
    global theta

    # Loop over all subsamples
    for i in range(0,t):

        #create a subsample of indices
        d_i = random.sample(range(0,n),phi)
        df = numpy.array(df)
        diDF = numpy.array(df[d_i])
        theta = numpy.zeros(df.shape)


        #create random d random subspaces by permuting columnss 0:d-1 and pairing adjacent values
        # randomly reorder set of column names
        s = random.sample(range(0,d),d)
        s0 = s[0:len(s)]
        s1 = s[1:len(s)]+[s[0]]

        # create frequency table
        # in this version we compare each row of a subsample to the whole data frame
        # and then sum them up along the subsample in a global theta variable
        numpy.apply_along_axis(computeFrequency,1,diDF,df, s0,s1)
        scores = scores + numpy.sum(theta == zeros,axis = 1)

    return 1.0*scores/t/d
#####################################

def computeFrequency(y, df,s0,s1):

    """
    Helper method - Compute Frequency of "y" in "diDF"

    Input:
    y: A single instance out of subsample
    diDF: A subsample
    s: A list of subspaces

    Output:
    freq_in_subspaces: For the current subsample, it's a "pandas series" of frequencies
    of y in each subspace
    """

    global theta
    n, d = df.shape

    # elementwise comparison operation between the "y" in subsample and each raw in the original data frame
    # summed up along subsample in a global theta array simultaneously for all the instances
    occurances = y==df
    occurances0 = occurances[:,s0]
    occurances1 = occurances[:,s1]

    theta = theta + occurances0*occurances1

    return 0

#####################################

def sibyl_score(y,df,t,phi):
    """
    Same as sibyl but scores a single instance.

    Input:
    y: instances whose score is desired
    df: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    t: number of subsamples
    phi: subsample size

    Output:
    score: Score of df_row_idx indexed instance float number
    """
    n,d = df.shape

    y = numpy.array(y)
    df = numpy.array(df)

    zeros = numpy.zeros(d)
    score = 0

    # Loop over all subsamples
    for i in range(0,t):

        #create a subsample of indices
        d_i = random.sample(range(0,n),phi)
        diDF = df[d_i]

        #create random d random subspaces by permuting columnss 0:d-1 and pairing adjacent values
        # randomly reorder set of column names
        s = random.sample(range(0,d),d)
        s0 = s[0:len(s)]
        s1 = s[1:len(s)]+[s[0]]

        # compute score
        occurances = y==diDF
        occurances0 = occurances[:,s0]
        occurances1 = occurances[:,s1]
        score = score + numpy.sum((occurances0*occurances1).sum(axis = 0) == zeros)

    return 1.0*score/d/t

####################################

def anomaly_inspect(y,df, plot = False):
    """
    Compute the inverse relative frequency of a category (pair of categories) in each column (pairs of columns)
    for categories in an anomalous instance index by idx.
    inverse relative frequency = total # of instances/(# of instances with fixed category
                                                        X # of categories for that feature )
    Plot relative frequencies (optional).

    Input:
    df: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    idx: index of anomaly in the data frame df
    plot: True - plot, False - do not plot

    Output:
    (freq_1_d,freq_2d)
        freq_1d: inverse relative frequencies for categories 1xd numpy array
        freq_2d: inverse relative frequencies for pairs of categories dxd numpy array
    """
    n, d = df.shape

    colnames = df.columns
    sizes = []
    for c in colnames:
        sizes = sizes + [len(df[c].unique())]
    sizes = numpy.array(sizes)

    y = numpy.array(y)
    df = numpy.array(df)


    # elementwise comparison operation between outlier y and each raw in the original data frame
    # summed up along columns and divided by total number of elements per category
    occurances = (y==df)
    freq_1d = 1.0/((occurances.sum(axis = 0)/(1.0*n/sizes)))

    #form all feature pairs
    freq_2d = numpy.zeros((d,d))
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
        xTickMarks = [str(colnames[i]) + ': ' + y[i] for i in range(0,d)]
        ax.set_xticks(numpy.array(range(0,d))+0.5)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=90, fontsize=10)

        ax1 = fig.add_subplot(2,1,2)
        ax1.set_ylabel('1/relative frequency of a pair of categories', fontsize = 14)
        ax1.set_title('Pairs of categories inverted relative frequencies\n'+
                      '(frequency of specified category times number ofcategories)\n'+
                      'for each column',
                         fontsize = 16)
        im = ax1.imshow(freq_2d.T,interpolation = "none")
        tickMarks = [str(colnames[i]) + ': ' + y[i] for i in range(0,d)]
        ax1.set_xticks(numpy.array(range(0,d))+0)
        ax1.set_yticks(numpy.array(range(0,d))+0)
        xtickNames = ax1.set_xticklabels(tickMarks)
        ytickNames = ax1.set_yticklabels(tickMarks)
        plt.setp(xtickNames, rotation=90, fontsize=10)
        plt.setp(ytickNames, rotation=0, fontsize=10)
        plt.colorbar(im)
        plt.show()

    return (freq_1d,freq_2d)

################################

def get_feature_importance(y,df):
    """
    Calls anomaly inspect(), selects maximum elements of the tables and organizes output into dictionary

    Input:
    df: input dataset, of size (n, d) where:
        n: number of samples
        d: number of attributes
    idx: index of anomaly in the data frame df

    Output:
    dictionary with locations and scores of single most rare feature and single most rare column
    """
    t1,t2 = anomaly_inspect(y,df, plot = False)
    columns = list(df.columns)
    d1_score = t1.max()
    d1_loc = [columns[i] for i in numpy.where(t1 == t1.max())[0]]

    d2_score = t2.max()
    d2_loc = numpy.where(t2 == t2.max())
    d2_loc = [(columns[d2_loc[0][i]],columns[d2_loc[1][i]]) for i in range(len(d2_loc[0]))]

    return {'single feature':(d1_loc,d1_score),'pair features':(d2_loc,d2_score)}
