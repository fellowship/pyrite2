from pyrite import *
import wget
import os.path
from sklearn.metrics import roc_auc_score

"""

This example uses the Abalone dataset that has mixed attributes - one
categorical attribute out of 8 attributes in total - and discretization
will be performed on the numerical attributes before applying pyrite
anomaly detection.

The dataset is available at UC Irvine Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets.html
"""


# Read dataset
if(not os.path.isfile("abalone.data")):
    print "Downloading dataset ..."
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    filename = wget.download(url)

abalone_data = pandas.read_csv('abalone.data',header = None)

# Create a pyrite object for this dataset
abalone_pyrite = Pyrite(abalone_data)

# Perform discretization for numerical attributes (from 1 - 8)
cont_cols = range(1,9)
print "Discretizing numeric columns"
abalone_pyrite.discretize(cont_cols)

# Return anomaly score for every sample in the dataset, sample 50
# times and include 100 instances in each sample

print "Training model"
score_vec = abalone_pyrite.score_dataset(50, 100)
print "Model training completed!\n"

# Calculating the AUC Score
'''
In this dataset, the number of rings (last feature in the dataframe) is to be
predicted, either as a continuous value or as a classification problem. The motivation
behind predicting the number of rings in this dataset is to calculate the age of the abalone
since it is simply equal to the number of rings after adding 1.5 to it.

The anomaly class is chosen to be the abalones that have a number of rings less than 3, and
larger than 21. These numbers are chosed based on the "Class Distribution" in this link:
https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names  
'''
y_true = abalone_data[abalone_data.columns[8]]
y_true = y_true.apply(lambda x: (x>21 or x<3))

# Computing the AUC Score
print "AUC Score:\n"
print('{0:.5f}'.format(roc_auc_score(y_true, score_vec.values)))


# outputs:
'''
AUC Score:

0.92700
'''

# convert to numpy array to plot the histogram of the scores
score_array = score_vec.values
score_hist = numpy.histogram(score_array)

print "Histogram of anomaly scores:\n"
print "Anomaly Score Bin           	      # of instances"
for i in range(len(score_hist[1])-1):
	print '{0:.5f}'.format(score_hist[1][i])," - ", '{0:.5f}'.format(score_hist[1][i+1]), "		     ", score_hist[0][i]

# outputs
'''
Anomaly Score Bin           	      # of instances
0.07778  -  0.16178 		      1098
0.16178  -  0.24578 		      1853
0.24578  -  0.32978 		      646
0.32978  -  0.41378 		      280
0.41378  -  0.49778 		      168
0.49778  -  0.58178 		      68
0.58178  -  0.66578 		      27
0.66578  -  0.74978 		      22
0.74978  -  0.83378 		      11
0.83378  -  0.91778 		      4
'''

# Index of the instance that has the highest anomaly score
anomaly_score_highest = score_vec.argmax()
max_anomaly_score = abalone_pyrite.score_instance(anomaly_score_highest, 50, 100)

print "\nanomaly score for the most anomalous instance: ", '{0:.5f}'.format(max_anomaly_score)

# To check the most important features and pair of features for that instance
# most_important_feature = abalone_pyrite.get_feature_importance(anomaly_score_highest)
# outputs: most important single feature, and most important feature-pairs, in terms of contribution to
# the total anomaly score for the instance with the index "anomaly_score_highest"
'''
Most important feature and features_pair:

Most important single feature: 
[5]
Most important feature-pair: 
[(0, 5)]
'''

'''
For further inspection - you can uncomment the line below - of other features
and feature-pairs in terms of contribution to the total anomaly
score: (instance_inspect) returns the contribution of each single
feature, and feature-pair, in the total anomaly score for a specific
instance.
'''

# abalone_pyrite.instance_inspect(anomaly_score_highest, plot=True)
