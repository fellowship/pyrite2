### Instantiation

Sibyl class takes a single argument (a Pandas DataFrame) when initialized.  The returned object provides the necessary methods for anomaly scoring either a single instance or the entire dataset.

- __init__(self, dataframe)
    - __Arguments__:
        - __dataframe__: Pandas Dataframe
    - __Return__: Sibyl object over which we detect anomalies

### Anomaly Scoring
Once the dataframe is loaded, Sibyl randomly samples rows from the dataframe. The number of times sampled is controlled by: `sample_num` and number of instances picked in each sample is controlled by: `sample_size`. Sibyl can compute the anomaly score for each instance (via score_dataset), or for a single instance (via score_instance).

- __score_dataset__(sample_num, sample_size)
    - __Arguments__:
        - __sample_num__: int (number of random samples to perform).
        - __sample_size__: int (number of instances to include in each sample)
    - __Return__: numpy.ndarray (a list where the i_th element is the anomaly score for the i_th instance in the dataset).

- __score_instance__(single_instance, sample_num, sample_size)
    - __Arguments__:
        - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
        - __sample_num__: int (number of random samples chosen without replacement from the dataset)
        - __sample_size__: int (size of the random sample)
    - __Return__: float (Anomaly score for the single instance)

### Feature Importance

Sibyl provides a way to further explore the anomalous instances in a dataset, through supplying a list of features, and feature-pairs, that contribute the most in its anomaly score and thus gain insights on why an instance was classified as an anomaly.

- __get_feature_importance__(single_instance)
    - __Arguments__:
        - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
    - __Return__: A dictionary consisting of the most important single feature in the anomaly score, as well as the single most important pair of features.

- __instance_inspect__(single_instance, plot)
    - __Arguments__:
        - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
        - __plot__: Boolean, to show plots of the freq_1d and freq_2d values
    - __Return__:
        - __freq_1d__: ndarray, containing the inverse relative frequency for each single feautre.
        - __freq_2d__: dxd ndarray, where ith column and jth row corresponds to the anomaly score due to features i and j.

### Discretization of Numerical Features

Discretization converts numerical data features into discrete categories before anomaly detection. To call the discretize function,  the list of column numbers must be specified. It is assumed that the data does not contain NANs.  

The function `discretize` performs discretization by calling the `auto_discretize` function individually on each column. To perform a different discretization, `auto_discretize` should be called directly on a per column basis.

- __discretize__(columns, method = 'blocks')        
    - __Arguments__: 
        - __columns__: list (columns from self.df dataframe to discretize.)
        - __method__: int or str (method to use of either 'blocks'(default),'scott' or specific bin number)
    - __Return__: None (changes columns in self.df dataframe.)  

- __auto_discretize__(num_data,method,range_min_max):
    - __Arguments__:
        - __num_data__: numpy.array (numerical feature data to be discretized)
        - __method__: int or str (method to use of either 'blocks','scott' or specific bin number)
        - __range_min_max__: Tuple (data range from num_array to apply chosen method to)
    - __Return__: pandas.Series cast as str (categorical labels after discretization)


                

