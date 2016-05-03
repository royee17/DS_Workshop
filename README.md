## Classes

### DataManager

##### init:
  Receives the root directory

##### loadData: 
  Loads the data and transforms all the string columns into integers
  
###### Params:
      fields (not required): Column names that we want to read  

### AlgorithmManager

##### init:
  Receives the dataManager object
  
##### runGRU4Rec: 
Runs recurrent neural network based on the paper: http://arxiv.org/pdf/1511.06939v4.pdf

##### runKMeans: 
Runs K means on the Dataset
