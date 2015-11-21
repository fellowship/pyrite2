from sibyl import *
import wget
import os.path

"""

This example uses the Abalone dataset that has mixed attributes - one
categorical attribute out of 8 attributes in total - and discretization
will be performed on the numerical attributes before applying sibyl
anomaly detection.

The dataset is available at UC Irvine Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets.html
"""


# Read dataset
if(not os.path.isfile("abalone.data")):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    filename = wget.download(url)

abalone_data = pandas.read_csv('abalone.data',header = None)

# Create a sibyl object for this dataset
abalone_sibyl = Sibyl(abalone_data)

# Perform discretization for numerical attributes (from 1 - 8)
cont_cols = range(1,9)
abalone_sibyl.discretize(cont_cols)

# Return anomaly score for every sample in the dataset, sample 50
# times and include 100 instances in each sample

score_vec = abalone_sibyl.score_dataset(50, 100)

# convert to numpy array to plot the histogram of the scores
score_array = score_vec.values
score_hist = numpy.histogram(score_array)

print "\nHistogram of anomaly scores: "
print "\n Anomaly Score Bin             	# of instances"
for i in range(len(score_hist[1])-1):
	print '{0:.5f}'.format(score_hist[1][i])," - ", '{0:.5f}'.format(score_hist[1][i+1]), "		     ", score_hist[0][i]


# Index of the instance that has the highest anomaly score
anomaly_score_highest = score_vec.argmax()
max_anomaly_score = abalone_sibyl.score_instance(anomaly_score_highest, 50, 100)

# To check the most important features and pair of features for that instance
print(abalone_sibyl.get_feature_importance(anomaly_score_highest))
# outputs: most important single feature, and most important feature-pairs, in terms of contribution to the total anomaly score for the instance with the index "anomaly_score_highest"
'''
{'single feature': ([5], 59.671428571428564), 'pair features': ([(3, 5)], 83.539999999999992)}
'''

'''
For further inspection - you can uncomment the line below - of other features
and feature-pairs in terms of contribution to the total anomaly
score: (instance_inspect) returns the contribution of each single
feature, and feature-pair, in the total anomaly score for a specific
instance.
'''

# abalone_sibyl.instance_inspect(anomaly_score_highest, plot=True)
