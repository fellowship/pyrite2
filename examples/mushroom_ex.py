from sibyl import *
import wget
import os.path
from sklearn.metrics import roc_auc_score

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

# Return anomaly score for every sample in the dataset, sample 50 times and include 100 instances in each sample
mushroom_sibyl = Sibyl(data_anomaly)
print "Training model"
score_vec = mushroom_sibyl.score_dataset(50, 100)
print "Model training completed!\n"

# Computing the AUC Score
print "AUC Score:\n"
print('{0:.5f}'.format(roc_auc_score(y_true, score_vec.values)))

# outputs
'''
AUC Score:

0.98474
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
0.00000  -  0.05782 		      3828
0.05782  -  0.11564 		      434
0.11564  -  0.17345 		      106
0.17345  -  0.23127 		      59
0.23127  -  0.28909 		      3
0.28909  -  0.34691 		      0
0.34691  -  0.40473 		      0
0.40473  -  0.46255 		      0
0.46255  -  0.52036 		      2
0.52036  -  0.57818 		      1
'''
# Index of the instance that has the highest anomaly score
anomaly_score_highest = score_vec.argmax()
max_anomaly_score = mushroom_sibyl.score_instance(anomaly_score_highest, 50, 100)

# To check the most important features and pair of features for that instance
most_important_feature = mushroom_sibyl.get_feature_importance(anomaly_score_highest)
print "\nMost important feature and features_pair:\n"
print "Most important single feature: "
print most_important_feature['single feature'][0]
print "Most important feature-pair: " 
print most_important_feature['pair features'][0]

# To check the most important features and pair of features for that instance
# print(mushroom_sibyl.get_feature_importance(anomaly_score_highest))
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

# mushroom_sibyl.instance_inspect(anomaly_score_highest, plot=True)
