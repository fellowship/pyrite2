import numpy
import math
import pandas
import random
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
from astroML.plotting import hist

class Pyrite:


    def __init__(self, dataset):
        # Convert input dataset to datafraem and handle nulls
        # Store dataset in self.df
        self.df = pandas.DataFrame(dataset.copy())


    def auto_discretize(self,num_data,method,range_min_max):

        """
        Perform automatic discretization of a selected feature; a method
        (bayesian blocks, scott method or fixed bin number) along the desired data range is passed to a special version of hist which gives cutpoints for discretization and returns the "categorized" version of the original data
        """
        hist_data = hist(num_data, bins=method,range=range_min_max)
        plt.close('all')
        leng = len(hist_data[1])
        # fix cutoff to make sure outliers are properly categorized as well if necessary
        hist_data[1][leng-1] = num_data.max()
        #hist_data[1][0] = num_data.min()
        # automatically assign category labels of '1','2',etc
        cat_data = pandas.cut(num_data,hist_data[1],labels=range(1,leng),include_lowest='TRUE')
        return pandas.Series(cat_data).astype(str)

    def discretize(self, columns, method = 'blocks'):
        """
    Discretizes user provided list of numerical columns using auto_discretization function above. If a different method for discretization is desired, for a specific column range, this function should be called on a per column basis with parameters set for each particular column

        Input:
        columns: list of columns to dicretize.
        method: default method is 'blocks' (Bayesian blocks), other options is 'scott' or fixed number of bins
        plot: if True - plot histogram, default = False

        Output:
        None, changes columns in the self.df data frame.

        """
        print "\ndiscretizing numerical attributes ..."

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


        for col in columns:
            if not(self.df[col].dtype in numerics):
                err = 'Column ' + str(col) + ' is not numeric!'
                raise Exception(err)

        num_cols = len(columns)
        for col in columns:
            col_range = (self.df[col].min(),self.df[col].max())
            self.df[col] = self.auto_discretize(self.df[col], method, range_min_max = col_range)



    def compute_frequency(self,y, s0, s1):

        """
        Helper method - Compute Frequency of "y" in "diDF"

        Input:
        y: A single instance out of subsample
        df: dataset
        s0: list of features that represent the first dimension in 2D subspaces
        s1: list of features that represent the second dimension in 2D subspaces

        Output:
        freq_in_subspaces: For the current subsample, it's a "pandas series" of frequencies
        of y in each subspace
        """

        global theta
        n, d = self.df.shape

        # elementwise comparison operation between the "y" in subsample and each raw in the original data frame
        # summed up along subsample in a global theta array simultaneously for all the instances
        occurances = y==numpy.array(self.df)
        occurances0 = occurances[:,s0]
        occurances1 = occurances[:,s1]

        theta = theta + occurances0*occurances1

        return 0




    def score_dataset(self, samples_num, sample_size,seed = None):
        """
        Returns a numpy array of scores for every instance in the dataset

        Input:
        samples_num: number of subsamples chosen randomly without replacement
        sample_size: size of each subsample
        seed: seed for random number generator

        Output:
        scores: Score for each instance in the dataset (numpy array)
        """
        print "\ncomputing anomaly score for the dataset instances ..."

        (n,d) = (self.df).shape
        columns = (self.df).columns
        indices = (self.df).index

        zeros = numpy.zeros(d)
        scores = numpy.zeros(n)
        global theta

        random.seed(seed)
        # Loop over all subsamples
        for i in range(0,samples_num):
            #create a subsample of indices
            d_i = random.sample(range(0,n),sample_size)

            df_ndarray = numpy.array(self.df)
            diDF = df_ndarray[d_i]
            theta = numpy.zeros(self.df.shape)


            #create random d random subspaces by permuting columnss 0:d-1 and pairing adjacent values
            # randomly reorder set of column names
            s = random.sample(range(0,d),d)
            s0 = s[0:len(s)]
            s1 = s[1:len(s)]+[s[0]]

            # create frequency table
            # in this version we compare each row of a subsample to the whole data frame
            # and then sum them up along the subsample in a global theta variable
            numpy.apply_along_axis(self.compute_frequency, 1,diDF, s0,s1)
            scores = scores + numpy.sum(theta == zeros,axis = 1)


        return pandas.Series(1.0*scores/samples_num/d, index = indices)




    def score_instance(self, idx, samples_num, sample_size, seed = None):

        """
        Same as pyrite but scores a single instance.

        Input:
        idx: Index of the instance whose score is desired
        samples_num: number of subsamples chosen randomly without replacement
        sample_size: size of each subsample

        Output:
        score: float - Anomaly Score of single_instance
        """

        print "\ncomputing anomaly score for a single instance ..."

        n,d = self.df.shape

        #y = single_instance.fillna('')

        y = numpy.array(self.df.ix[idx])
        df_ndarray = numpy.array(self.df)

        zeros = numpy.zeros(d)
        score = 0

        random.seed(seed)
        # Loop over all subsamples
        for i in range(0,samples_num):

            #create a subsample of indices
            d_i = random.sample(range(0,n),sample_size)
            diDF = df_ndarray[d_i]

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

        return 1.0*score/samples_num/d


    def instance_inspect(self, idx, plot = False):
        """
        Compute the inverse relative frequency of a category (pair of categories) in each column (pairs of columns)
        for categories in an anomalous instance index by idx.
        inverse relative frequency = total # of instances/(# of instances with fixed category X # of categories for that feature )
        Plot relative frequencies (optional).

        Input:
        idx: index of a single instance to inspect.
        plot: Boolean - plot inverse relative frequency

        Output:
        (freq_1_d,freq_2d)
            freq_1d: inverse relative frequencies for categories 1xd numpy array
            freq_2d: inverse relative frequencies for pairs of categories dxd numpy array
        """
        n, d = self.df.shape


        colnames = self.df.columns
        sizes = []
        for c in colnames:
            sizes = sizes + [len(self.df[c].unique())]
        sizes = numpy.array(sizes)

        y = numpy.array(self.df.ix[idx])
        df_ndarray = numpy.array(self.df)


        # elementwise comparison operation between outlier y and each raw in the original data frame
        # summed up along columns and divided by total number of elements per category
        occurances = (y==df_ndarray)
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
            xTickMarks = [str(colnames[i]) + ': ' + str(y[i]) for i in range(0,d)]
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
            tickMarks = [str(colnames[i]) + ': ' + str(y[i]) for i in range(0,d)]
            ax1.set_xticks(numpy.array(range(0,d))+0)
            ax1.set_yticks(numpy.array(range(0,d))+0)
            xtickNames = ax1.set_xticklabels(tickMarks)
            ytickNames = ax1.set_yticklabels(tickMarks)
            plt.setp(xtickNames, rotation=90, fontsize=10)
            plt.setp(ytickNames, rotation=0, fontsize=10)
            plt.colorbar(im)
            plt.show()

        return (freq_1d,freq_2d)





    def get_feature_importance(self, idx):
        """
        Calls instance_inspect, selects maximum elements of the tables and organizes output into dictionary

        Input:
        idx: index of anomaly in the data frame df

        Output:
        dictionary with locations and scores of single most rare feature and single most rare column
        """
        print "\ngetting important features ..."
        t1,t2 = self.instance_inspect(idx, plot = False)
        columns = list(self.df.columns)
        d1_score = t1.max()
        d1_loc = [columns[i] for i in numpy.where(t1 == t1.max())[0]]

        d2_score = t2.max()
        d2_loc = numpy.where(t2 == t2.max())
        d2_loc = [(columns[d2_loc[0][i]],columns[d2_loc[1][i]]) for i in range(len(d2_loc[0]))]
        return {'single feature':(d1_loc,d1_score),'pair features':(d2_loc,d2_score)}


