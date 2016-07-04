## Classes

### DataManager

##### init:
  Receives the root directory

##### loadData: 
  Loads the data and transforms all the string columns into integers
  
###### Params:
      fields (not required): Column names that we want to read  

##### splitData: 
  Allows to split the data into train and test sets
  
##### getEncodedByLabel
  Searches for the label in the decoded dataset, then encodes back and returns the encoded value for the label
  
### AlgorithmManager

##### init:
  Receives the dataManager object
  
##### displayGraph:
 Displays a graph where nodes are the different methods. Edges are sequences of method->method by timestamp.
 
##### displayCountByQueryName
  Displays the count of IsFirst for each QueryName divided by the number of IsFirst in the dataset
  
##### displayCountSessionByQueryName
    Displays the count of below and above session for each QueryName divided by the number of queries in session in the dataset.
    For each unique query it displays the percentage of the users that have 3 or more sessions
  
##### displayAvgTimeSessions
  Displays session duration times for users that visited the site more than 3 times and users that visited 3 times or less.
  
##### printDecisionTreeForBelow3
  Creates a decision tree in order to detect which features are best to predict the low retention
  
##### genericBelowAverageDisplay
  Displays the percentage of the users that have 3 sessions or less
  
##### displaySidAndQueryName
  Displays Sessions and queries
  
#####  displayAidBySid
  Finds Aid's with many sessions
  
#####  compareGRUtoBaselines
   Runs baseline methods: Random and ItemKNN
   
##### runGRU4Rec: 
  Runs recurrent neural network based on the paper: http://arxiv.org/pdf/1511.06939v4.pdf

##### learnFromExperiencedGRU
  Runs the GRU, learns from the users that have more than average amount of sessions, and predicts on users that have less than 3 sessions
  
##### runGRU4RecForSpecificAid
  Runs the gru on a specific Aid, and then on the aid's sessions
  
#####  runPCA
  Runs PCA on the data
  
##### runKMeans: 
  Runs K means on the Dataset

##### runHierarchicalClustering
  Runs Hirarchical Clustering on the Dataset, creating a Dendrogram visualizing the clustering process for further analysis.

##### loadOperationsByOneOfK
  Utility function for clustering.
  
##### runTSNE
  Runs T-SNE on the Dataset

