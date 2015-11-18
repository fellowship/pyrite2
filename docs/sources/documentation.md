# Sibyl: Anomaly Detection in High Dimensional Heterogeneous Datasets

## Introduction

Sibyl provides a way to discover anomalies in categorical datasets. If your data contains numberical attributes mixed with categorical ones, the method can be extended to take care of that case as well. But first, any numerical attributes need to be discretized before being supplied to the Sibyl function. Sibyl also calculates the ***features*** and ***feature pairs*** that were the main contributers in the anomaly score of some instance y.


## Methods Descriptions
#### sibyl(dataset, ssamples_num, ssample_size, anomalies)
```python
anomaly_score = sibyl(dataset, ssamples_num, ssample_size, anomalies)
```

The main method of the sibyl module,  and it takes the following parameters as an input:
dataset: The dataset consisting of categorical attributes.
ssample_num: The number of subsamples that are randomly chosen for checking random subspaces.
ssample_size: The subsample size.
anomalies: Optional, the indexes of instances that you would like to see the most important features contributing to its anomaly score.

The output is an anomaly_score list, where the i_th element in the list is the anomaly score for the i_th sample in the dataset.

#### sibyl_score(single_instance, dataset, ssamples_num, ssample_size)
```python
anomaly_score_one_instance = sibyl_score(single_instance, dataset, ssamples_num, ssample_size)
```

This method is essentially the same as the one above, except that it takes an extra input argument (single_instance), and it returns a single number, which is the anomaly_score for that instance by itself.

#### get_feature_importance(single_instance,dataset)
```python
important_features = get_feature_importance(single_instance,dataset)
```
It takes a single instance, and the dataset, as parameters, and returns the most important features and feature_pairs in a dictionary, that contribute the most to the anomaly_score.

#### anomaly_inspect(single_instance,dataset, plot = False)
'''python
anomaly_inspect(single_instance,dataset, plot = False)
'''

The same as the *get_feature_importance* method, but with the extra option of plotting the inverse relative frequency for anmalous features in 1D and 2D subspaces.
