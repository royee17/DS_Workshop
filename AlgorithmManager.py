﻿import warnings
warnings.filterwarnings('ignore', 'numpy not_equal will not check object identity in the future')
from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering as hc
from sklearn.metrics import silhouette_samples, silhouette_score
from time import clock
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import baselines
import gru4rec

class AlgorithmManager(object):

    dataManager = {}

    '''
        Init
    '''
    def __init__(self,dataManager):
        self.dataManager = dataManager

    '''
        Displays the count of IsFirst for each QueryName divided by the number of IsFirst in the dataset
    ''' 
    def displayCountByQueryName(self):
        data = self.dataManager.loadData(["QueryName","IsFirst"],transformFields=False)
        firstCount = (data.IsFirst).sum()
        notFirstCount = (data.IsFirst).count() - firstCount
        
        result = data.groupby('QueryName').apply(
             lambda group: (group.IsFirst.sum() / # Sum = count of true
                            float(firstCount))
         ).to_frame('First')

        result['NotFirst'] = data.groupby('QueryName').apply(
             lambda group: ((group.IsFirst.count() - group.IsFirst.sum()) /  # Count of total minus count of true
                            float(notFirstCount))
         )
        result.plot(kind='bar')
        plt.show()     

    '''
        Find Aid with many sessions
    ''' 
    def displayAidBySid(self):
        data = self.dataManager.loadData(["Aid","Sid"],transformFields=False)        
        result = data.groupby('Aid').apply(
             lambda group: (group.Sid.nunique())
         )

        # Remove top 5 extreme
        for i in np.arange(5):
            print(result.idxmax())
            result = result[result < result.max()] 
        # Aids:
            # UnknownIbizaUser
            #14bcea76-45fd-4b66-87a9-f63844a086c2
            #53a13610-68d0-49a6-87ad-d3c121b0c2c2
            #b7d6a7e8-6390-4e54-b653-0d1c055c7da3
            #185610be-92f9-4d83-a567-4675a5615fac

        ax = result.plot(kind='barh',color='red',figsize =(20,20))
        ax.set_xlabel("# of unique Sid for each Aid")
        ax.set_ylabel("Aid")
        ax.axvline(x=result.mean(),c="blue",linewidth=2,ls='dashed')  
        plt.xticks(np.arange(result.min(), result.max()+1, 5.0))
        plt.savefig('GroupByAidCountSidForAll.png')
        plt.show()
          

    '''
        Runs recurrent neural network based on the paper: http://arxiv.org/pdf/1511.06939v4.pdf
    ''' 
    
    def runGRU4Rec(self):   
    
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "QueryName"

        data = self.dataManager.loadData([time_key,item_key,"Aid","Sid"]) 
        
        train, test = self.dataManager.splitData(data,isRandom=False)
        print('Training GRU4Rec')    
            
        for batch_size in [5,10,15]:
            for momentum in [1,2,3,4,5]:
                for dropOut in [1,2,3,4,5]:
                    print('Batch Size: ' + str(batch_size) + ' Dropout: ' + str(float(dropOut)/10.0) + ' Momentum: ' + str(float(momentum)/10.0))
                    gru = gru4rec.GRU4Rec(layers=[1000], loss='top1', batch_size=batch_size, dropout_p_hidden=float(dropOut)/10.0, learning_rate=0.05, momentum=float(momentum)/10.0
                                            ,n_epochs=10,hidden_act = 'tanh', final_act='tanh'
                                            ,session_key=session_key, item_key=item_key, time_key=time_key)
                    gru.fit(train)
        
                    res = gru.evaluate_sessions_batch(test, cut_off=2, batch_size=batch_size, 
                                                session_key=session_key, item_key=item_key, time_key=time_key)
                    print('Recall@2: {}'.format(res[0]))
                    #print('MRR@2: {}'.format(res[1]))

    '''
        Runs recurrent neural network based on the paper: http://arxiv.org/pdf/1511.06939v4.pdf
        Runs the gru on a specific Aid, and then on the sessions for the aid
        
        Aid:            21674def-1c93-46e5-95ab-015e904fb10f	
        Encoded Aid:    3807	
        Train Size: 1184 Test Size: 296	
			
        Best Results:

        Batch Size: 5 Dropout: 0.5 Momentum: 0.4
			Correct: 108
			Accuracy: 0.372413793103	
            					
		Batch Size: 10 Dropout: 0.5 Momentum: 0.2
				Correct: 109
				Accuracy: 0.382456140351				
            		
		Batch Size: 15 Dropout: 0.5 Momentum: 0.3
			Correct: 110
			Accuracy: 0.392857142857
		


    ''' 
    def runGRU4RecForSpecificAid(self):   
    
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "QueryName"

        data = self.dataManager.loadData([time_key,item_key,"Aid","Sid"]) 

        # Select a specific Aid
        selectedAid = "21674def-1c93-46e5-95ab-015e904fb10f";
        encodedVal = self.dataManager.getEncodedByLabel(data,"Aid",selectedAid);
        print(str(encodedVal))    
        data = data.loc[data['Aid'] == encodedVal]

        train, test = self.dataManager.splitData(data,isRandom=False)
        print('Training GRU4Rec')    
            
        batch_size = 5
        momentum = 0.4
        dropOut = 0.5

        print('Batch Size: ' + str(batch_size) + ' Dropout: ' + str(float(dropOut)) + ' Momentum: ' + str(float(momentum)))
        gru = gru4rec.GRU4Rec(layers=[1000], loss='top1', batch_size=batch_size, dropout_p_hidden=float(dropOut), learning_rate=0.05, momentum=float(momentum)
                                ,n_epochs=10,hidden_act = 'tanh', final_act='tanh'
                                ,session_key=session_key, item_key=item_key, time_key=time_key)
        gru.fit(train)
                            
        test.is_copy = False
        test.sort_values([time_key,session_key], inplace=True) # Sort by time_key first and then by session_key
 
        correct = 0;
        for i in range(len(test)-batch_size-1):
            # Goes from 1 to len(test) and gets the batches - for example: [1-5],[2-6],[3-6]
            curSessions = test[session_key].values[range(i,i+batch_size)] 
            curItems = test[item_key].values[range(i,i+batch_size)]

            # Predicts the next batch (if we give [1-5] it predicts [6-10])
            preds = gru.predict_next_batch(curSessions, curItems, None, batch_size)

            # Take only the first element from the next batch and compare it (for example: predict([1-5]) returns [6-10] and we check if predict[6]==test[6])
            if(preds[0].idxmax() == test[item_key].values[i+batch_size+1]):
                print(str(i+batch_size+1) + " " + str(preds[0].idxmax()))
                correct = correct+1;
       
        print('Correct: {}'.format(correct))
        print('Accuracy: {}'.format(float(correct)/float(len(test)-batch_size-1)))

    '''
        Runs K means on the Dataset
        Goes over K=2:5 and checks the performance using slihouette
    '''
    def runKMeans(self, pivot, file = False, normalize = False, factor = 100):
        
        # Preparing the data : loading, normalizing (if selected) and selecting 100k record randomly.
        data = self.loadOperationsByOneOfK(False, pivot, normalize, factor)
        np.random.shuffle(data)
        #data = data[1:50000,:] # Higher amount of vectors cause memory error on silhouette_score
        
        # Preparing the data to compare, creating an average vector of the pivots.
        avgVec = np.mean(data,axis = 0)        
        fid = open(file, 'w')
        np.set_printoptions(precision=3,suppress = True) # Prettier printing.
        fid.write(str(avgVec))

        for n_clusters in range(2,25):
            print "Running KMeans on {0} Clusters".format(n_clusters)          
            # Initialize the clusterer with n_clusters value, 15 runs of the algorithm and a random generator
            clusterer = KMeans(n_clusters=n_clusters, n_init=300, init='k-means++', max_iter = 1000)
            cluster_labels = clusterer.fit_predict(data)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed clusters
            #silhouette_avg = silhouette_score(data, cluster_labels) 
            
            fid.write("The cluster centers for K={0}\n".format(n_clusters))
            fid.write(str(clusterer.cluster_centers_))
            #fid.write("\nThe cluster centers compared to the mean for K={0}\n".format(n_clusters))
            #fid.write(str(clusterer.cluster_centers_/avgVec * 100))
              
            for i in range(n_clusters):
                cluster_i_size = 0
                for val in cluster_labels:
                    if val == i:
                        cluster_i_size = cluster_i_size + 1               
                fid.write("\nThe size of cluster {0} is {1}".format(i,cluster_i_size))


            fid.write("\nThe inertia is {0}\n\n".format(clusterer.inertia_))
            fid.write("\n")
           
            
            fid.write("\n\n")
            
            #fid.write("For {0} clusters the average silhouette score is : {1}".format(n_clusters, silhouette_avg))            

    def runHierarchicalClustering(self,file,pivot='Aid',normalize=False,factor=100):        
        print 'Running hierarchical clustering'
       
        # Preparing the data : loading, normalizing (if selected) and selecting 100k record randomly.
        data = self.loadOperationsByOneOfK(False, pivot, normalize, factor)
        np.random.shuffle(data)
        data = data[1:10000,:] # The clustering is slow, so attempting on 10k random samples.
        
        # Preparing the data to compare, creating an average vector of the pivots.        
        if file != False:
            avgVec = np.mean(data,axis = 0)       
            fid = open(file, 'w')
            np.set_printoptions(precision=3,suppress = True) # Prettier printing.
            fid.write(str(avgVec))

        for n_clusters in range(2,25):  
            # Compute clustering           
            print("Running hierarchical clustering for {0} clusters".format(n_clusters))
            st = clock()
            ward = linkage(data, 'average', metric = 'cosine')

            # calculate the dendrogram
            plt.figure(figsize=(25, 10))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('sample index')
            plt.ylabel('distance')
            dendrogram(
                ward,
                leaf_rotation=90.,  # rotates the x axis labels
                leaf_font_size=8.,  # font size for the x axis labels
                truncate_mode='lastp',  # show only the last p merged clusters
                p=100,  # show only the last p merged clusters
                show_leaf_counts=True,  # False = numbers in brackets are counts               
                show_contracted=True,  # to get a distribution impression in truncated branches
            )
            plt.savefig("plot{0}{1}.png".format(pivot,n_clusters))
            '''
            ward = hc(n_clusters, linkage='ward').fit(data)
            elapsed_time = clock() - st
            label = ward.labels_
          
            for i in range(n_clusters):
                cluster_i_size = 0
                for val in label:
                    if val == i:
                        cluster_i_size = cluster_i_size + 1
                if file != False:               
                    fid.write("\nThe size of cluster {0} is {1}".format(i,cluster_i_size))
                else
                    print("\nThe size of cluster {0} is {1}".format(i,cluster_i_size))
            
            if file != False:
                #fid.write("\nThe inertia is {0}\n\n".format(clusterer.inertia_))
                fid.write("\n")           
                fid.write("\n\n")
            '''

    '''
        Utility function for clustering.

        Pre: gets a file path or False, and a pivot label, default is Aid.

        Post: Process the dataframe, and creates a ndarray of operations by the pivot label.
                If the file path isn't False, saves the ndarray to file.
    '''
    def loadOperationsByOneOfK(self, file, pivot='Aid', normalize=True, factor=100):
        
        data = self.dataManager.loadData([pivot,"QueryName"],transformFields=True)
        print("\n")
        data = data.sort_values(by=pivot)
        #print(data.head(50))

        # Compute the One of K matrix's size.
        temp = data.max()     
        numOfTypes = temp['QueryName'] + 1
        numOfPivots = temp[pivot]  + 1      
        
        print("num of operation types {0} , num of pivot items {1}".format(numOfTypes, numOfPivots))
        
        vectByTypes = np.zeros((numOfPivots,numOfTypes)) # Constructs a ndarray of size number of unique Sids x number of possible requests
        
        # Initializing status parameters.
        t1 = clock()       
        i = 0

        for row in data.itertuples():             
            curPivot = row[1] # the current Session\User ID
            oper = row[2] # the current Operation Type
            value = vectByTypes.item((curPivot,oper)) # reads the former count of SID\AID x Operation
            
            vectByTypes.itemset(((curPivot,oper)), value + 1) # Adding 1 to the current operation type for the current SID using Numpy quick access functions
            value = vectByTypes.item((curPivot,oper))            
            
            # Status print every 100,000 iterations
            t2 = clock()
            if (i%100000 == 0 and i!=0):
                print('Processed {0} rows in {1} seconds'.format(i, t2-t1))

            i = i + 1  
        
        if normalize == True:
            maxVect = vectByTypes.max(axis = 0)
            print "The maximum of each row before normalization is : {0}".format(maxVect)
            vectByTypes = vectByTypes / maxVect * factor
            maxVect = vectByTypes.max(axis = 0)
            print "and after normalization is : {0}".format(maxVect)


        # print(vectByTypes[1:10])
        if file != False:
            vectByTypes.tofile(file, sep = "," , format = "%s")
        
        print('Finshed processing all the data.')
        return vectByTypes