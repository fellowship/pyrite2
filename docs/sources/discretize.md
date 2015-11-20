### Discretization of Numerical Data to Categories

Discretization converts numerical data features into discrete categories before Sibyl anomaly detection. The user calls the discretize function and inputs the list of column numbers to discretize and particular automatic method to use. It is assumed that the data features does not contain NANs.  The function 'discretize' performs discretization on the user provided list of numerical feature columns by calling the 'auto_discretization' function individually on each column. If the user desires a different discretization method for each column, this function should be called individually on a per column basis with paritcular parameters set per column.

- __auto_discretize__(self,num_data,method,range_min_max):
  - __Return__: pandas.Series cast as str (categorical labels after discretization)
  - __Arguments__:
    - __num_data__: numpy.array (numerical feature data to be discretized)
    - __method__: int or str (method to use of either 'blocks','scott' or specific bin number)
    - __range_min_max__: Tuple (data range from num_array to apply chosen method to)

- __discretize__(self,columns, method = 'blocks')        
  - __Return__: None (changes columns in self.df dataframe.)  
  - __Arguments__: 
    - __columns__: list (columns from self.df dataframe to discretize.)
    - __method__: int or str (method to use of either 'blocks'(default),'scott' or specific bin number)

                
