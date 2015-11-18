## Feature Importance

Sibyl provides a way to further explore the anomalous instances in a dataset, through supplying a list of features, and feature-pairs, that contribte most in its anomaly score and thus gain insight on why an instance was classified as an anomaly.

- __Methods__:

  - __score_dataset__(dataset, ssample_num, ssample_size, anomalies=None)
        - __Return__: numpy.ndarray (a list where the i_th element is the anomaly score for the i_th instance in the dataset)
        - __Arguments__:
            - __dataset__: pandas.DataFrame or numpy.ndarray (a dataset of categorical attributes)
            - __ssample_num__: int (number of random samples chosen without replacement from the dataset)
            - __ssample_size__: int (size of the random sample)
            - __anomalies__: list of indices (Optional: list of indeices for instances to get the anomaly score for. If not specified, anomaly scores for every instance in the dataset will be returned.)

    - __score_instance__(single_instance, dataset, ssample_num, ssample_size)
        - __Return__: float (Anomaly score for the single instance)
        - __Arguments__:
            - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
            - __dataset__: pandas.DataFrame or numpy.ndarray (a dataset of categorical attributes)
            - __ssample_num__: int (number of random samples chosen without replacement from the dataset)
            - __ssample_size__: int (size of the random sample)
