from pyrite import *
import wget
import os.path
from sklearn.metrics import roc_auc_score

"""

This example uses the mushroom dataset that has mixed attributes - one
categorical attribute out of 8 attributes in total - and discretization
will be performed on the numerical attributes before applying pyrite
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

'''
Prepare the dataset for anomaly detection:
The poisonous mushrooms will be considered the anomaly class. 5% of the
poisonous mushrooms will be kept.
'''
data_e = mushroom_data[mushroom_data[0]=='e']
data_p = mushroom_data[mushroom_data[0]=='p']

len_poisonous = len(data_p)
data_anomaly = pandas.concat([data_e,data_p.ix[data_p.index[random.sample(range(0,len_poisonous),225)]]],ignore_index = True)

y_true = data_anomaly[data_anomaly.columns[0]]
y_true = y_true.apply(lambda x: x=='p')

data_anomaly.drop(data_anomaly.columns[0], 1, inplace=True)

# Return anomaly score for every sample in the dataset, sample 50 times and include 8 instances in each sample
mushroom_pyrite = Pyrite(data_anomaly)
print "Training model"
score_vec = mushroom_pyrite.score_dataset(50, 8)
print "Model training completed!\n"

# Computing the AUC Score
print "AUC Score:\n"
print('{0:.5f}'.format(roc_auc_score(y_true, score_vec.values)))

# outputs
'''
AUC Score:

0.93912
'''

# convert to numpy array to plot the histogram of the scores
score_array = score_vec.values
score_hist = numpy.histogram(score_array)

print "\nHistogram of anomaly scores: "
print "\n Anomaly Score Bin             	# of instances"
for i in range(len(score_hist[1])-1):
	print '{0:.5f}'.format(score_hist[1][i])," - ", '{0:.5f}'.format(score_hist[1][i+1]), "		     ", score_hist[0][i]

# outputs
'''
Histogram of anomaly scores: 

 Anomaly Score Bin             	# of instances
0.08727  -  0.15536 		      1473
0.15536  -  0.22345 		      440
0.22345  -  0.29155 		      1217
0.29155  -  0.35964 		      588
0.35964  -  0.42773 		      136
0.42773  -  0.49582 		      315
0.49582  -  0.56391 		      241
0.56391  -  0.63200 		      19
0.63200  -  0.70009 		      0
0.70009  -  0.76818 		      4
'''
# Index of the instance that has the highest anomaly score
anomaly_score_highest = score_vec.argmax()
max_anomaly_score = mushroom_pyrite.score_instance(anomaly_score_highest, 50, 100)

print "\nanomaly score for the most anomalous instance: ", '{0:.5f}'.format(max_anomaly_score)


# To check the most important features and pair of features for that instance
# print(mushroom_pyrite.get_feature_importance(anomaly_score_highest))
# outputs: most important single feature, and most important feature-pairs, in terms of contribution to 
# the total anomaly score for the instance with the index "anomaly_score_highest"
'''
Most important feature and features_pair:

Most important single feature: 
[17]
Most important feature-pair: 
[(16, 17)]
'''

'''
For further inspection - you can uncomment the line below - of other features
and feature-pairs in terms of contribution to the total anomaly
score: (instance_inspect) returns the contribution of each single
feature, and feature-pair, in the total anomaly score for a specific
instance.
'''

# mushroom_pyrite.instance_inspect(anomaly_score_highest, plot=True)
