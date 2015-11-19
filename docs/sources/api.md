## Anomaly Scoring

Sibyl can be used to compute an anomaly score for each instance in the dataset, or for a single instance.

- __Methods__:
  - __score_dataset__(sample_num, sample_size)
    - __Return__: numpy.ndarray (a list where the i_th element is the anomaly score for the i_th instance in the dataset).
    - __Arguments__:
      - __sample_num__: int (number of random samples chosen without replacement from the dataset).
      - __sample_size__: int (size of the random sample)
  - __score_instance__(single_instance, sample_num, sample_size)
    - __Return__: float (Anomaly score for the single instance)
    - __Arguments__:
      - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
      - __sample_num__: int (number of random samples chosen without replacement from the dataset)
      - __sample_size__: int (size of the random sample)

---

## Feature Importance

Sibyl provides a way to further explore the anomalous instances in a dataset, through supplying a list of features, and feature-pairs, that contribte most in its anomaly score and thus gain insight on why an instance was classified as an anomaly.

- __Methods__:
  - __get_feature_importance__(single_instance)
      - __Return__: A dictionary consisting of the most important single feature in the anomaly score, as well as the single most important pair of features.
      - __Arguments__:
      - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
  - __instance_inspect__(single_instance, plot)
    - __Return__:
      - __freq_1d__: ndarray, containing the inverse relative frequency for each single feautre.
      - __freq_2d__: dxd ndarray, where ith column and jth row corresponds to the anomaly score due to features i and j.
    - __Arguments__:
      - __single_instance__: pandas.Series (a single instance for which the anomaly score is to be computed)
      - __plot__: Boolean, to show plots of the freq_1d and freq_2d values
