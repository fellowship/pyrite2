from sibyl import *
import wget
import os.path

"""

This example uses the mushroom dataset that has mixed attributes - one
categorical attribute out of 8 attributes in total - and discretization
will be performed on the numerical attributes before applying sibyl
anomaly detection.

The dataset is available at UC Irvine Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets.html
"""

# Read the mushroom dataset from UC Irvine Dataset Repository
if (not os.path.isfile("agaricus-lepiota.data")):
    print "Downloading dataset ..."
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    filename = wget.download(url)
mushroom_data = pandas.read_csv("agaricus-lepiota.data",header = None)

# Return anomaly score for every sample in the dataset, sample 50 times and include 100 instances in each sample
mushroom_sibyl = Sibyl(mushroom_data)
print "Training model"
score_vec = mushroom_sibyl.score_dataset(50, 100)
print "Model training completed!"

# convert to numpy array to plot the histogram of the scores
score_array = score_vec.values
score_hist = numpy.histogram(score_array)

print "\nHistogram of anomaly scores: "
print "\n Anomaly Score Bin             	# of instances"
for i in range(len(score_hist[1])-1):
	print '{0:.5f}'.format(score_hist[1][i])," - ", '{0:.5f}'.format(score_hist[1][i+1]), "		     ", score_hist[0][i]

# Index of the instance that has the highest anomaly score
anomaly_score_highest = score_vec.argmax()
max_anomaly_score = mushroom_sibyl.score_instance(anomaly_score_highest, 50, 100)

# To check the most important features and pair of features for that instance
# print(mushroom_sibyl.get_feature_importance(anomaly_score_highest))
# outputs: most important single feature, and most important feature-pairs, in terms of contribution to the total anomaly score for the instance with the index "anomaly_score_highest"
'''
{'single feature': ([22], 3.9745596868884547), 'pair features': ([(16, 22)], 3.9745596868884547)}
'''

'''
For further inspection - you can uncomment the line below - of other features
and feature-pairs in terms of contribution to the total anomaly
score: (instance_inspect) returns the contribution of each single
feature, and feature-pair, in the total anomaly score for a specific
instance.
'''

# mushroom_sibyl.instance_inspect(anomaly_score_highest, plot=True)
