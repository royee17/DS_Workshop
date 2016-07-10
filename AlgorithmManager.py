import warnings
warnings.filterwarnings('ignore', 'numpy not_equal will not check object identity in the future')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#from sklearn.cluster import AgglomerativeClustering as hc
from sklearn.metrics import silhouette_samples, silhouette_score
from time import clock
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import baselines
import networkx as nx
from matplotlib import colors
from random import randint


'''
        Feature Selection
        ------------------------------
        criterion function
'''
def criterion_func(data,features):    
    from sklearn.linear_model import SGDClassifier
    from sklearn.cross_validation import train_test_split
   
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

    y, _ = pd.factorize(data['IsBelow'])
    X_train, X_test, y_train, y_test = train_test_split(data[features], y, test_size=0.33, random_state=42)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


'''
        Implementation of a Sequential Forward Selection algorithm.
        features (list): The feature space as a list of features.
        max_k: Termination criterion; the size of the returned feature subset.
        criterion_func (function): Function that is used to evaluate the performance of the feature subset.
        print_steps (bool): Prints the algorithm procedure if True.
        Returns the selected feature subset, as a list of features of length max_k.
'''

def seq_forw_select(data,features, max_k, criterion_func, print_steps=False):
       
    # Initialization
    feat_sub = []
    k = 0
    d = len(features)
    if max_k > d:
        max_k = d
    
    while True:
        
        # Inclusion step
        if print_steps:
            print('\nInclusion from feature space', features)
            print(feat_sub + [features[0]])
        crit_func_max = criterion_func(data,feat_sub + [features[0]])
        best_feat = features[0]
        for x in features[1:]:
            crit_func_eval = criterion_func(data,feat_sub + [x])
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        feat_sub.append(best_feat)
        if print_steps:
            print('include: {} -> feature subset: {}'.format(best_feat, feat_sub))
        features.remove(best_feat)
        
        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break
                
    return feat_sub

class AlgorithmManager(object):

    dataManager = {}

    '''
        Init
    '''
    def __init__(self,dataManager):
        self.dataManager = dataManager

    '''
        Graph -
            Each node is a different method
            Each edge is a sequence of method->method by timestamp
            Add 1 for weight when the same tuple of methods are called
    '''
    def displayGraph(self):

        data = self.dataManager.loadData(["QueryName","TimeStamp","Sid","Aid"],transformFields=False)
        
        #data = data[data.Aid == "012abc55-5801-494f-a77f-a799f1d855de"]
        data.is_copy = False
        data.sort_values(["Sid","TimeStamp"], inplace=True) # Sort by "TimeStamp"
                
        G=nx.DiGraph()
        lastSid = 0;
        colors_list = "bgrcmyk";#["red","black","yellow","blue"];
        curColor = "red";
        for i in range(len(data)-1):
            fromQuery = data["QueryName"].values[i];
            toQuery = data["QueryName"].values[i+1];
            if(data["Sid"].values[i] != lastSid):
                lastSid = data["Sid"].values[i];
                curColor = colors_list[randint(0,len(colors_list)-1)]
                continue;
            try:
                G[fromQuery][toQuery]['weight']=G[fromQuery][toQuery]['weight']+1;
            except:
                G.add_edge(fromQuery,toQuery,weight=1,color=curColor)

        elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >5000]
        esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <5000]

        pos=nx.circular_layout(G)
        plt.figure(figsize=(20,20))

        # nodes
        nx.draw_networkx_nodes(G,pos,
                               node_size=[v * 10 for v in nx.degree(G).values()]
                               ,alpha=0.5)

        # edges

        nx.draw_networkx_edges(G,pos,edgelist=elarge,
                                width=1)

        nx.draw_networkx_edges(G,pos,edgelist=esmall,
                        width=1,alpha=0.1,edge_color='b',style='dashed')

        # weights
        #weights = nx.get_edge_attributes(G,'weight')
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=weights)

        labels = {}    
        for edge in esmall:
            labels[edge[0]] = edge[0]
            labels[edge[1]] = edge[1]

        for edge in elarge:
            labels[edge[0]] = edge[0]
            labels[edge[1]] = edge[1]
        # Labels for the nodes in eLarge
        nx.draw_networkx_labels(G,pos,labels,font_size=12,font_color='b')
        
        #plt.savefig("D:\weighted_graph_noconnection.png")

        plt.show()

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
        Displays the count of below and above session for each QueryName divided by the number of queries in session in the dataset
    ''' 
    
    '''
        For each unique query it displays the percentage of the users that have 3 or more sessions
    '''
    def displayCountSessionByQueryName(self):
        data = self.dataManager.loadData(["QueryName","Aid","IsFirst","Sid"],transformFields=False)

        result = data.groupby('Aid').apply(
             lambda group: (group.Sid.nunique() <= 3) # 3 is the average (faster this way)
         )
        resultBelow = result[result == True]
        resultAbove = result[result == False]

        belowCount = (data[data.Aid.isin(resultBelow.index)]).sum()
        aboveCount = data[data.Aid.isin(resultAbove.index)].sum()
        
        result = data.groupby('QueryName').apply(
             lambda group: (group.Aid.isin(resultBelow.index).sum() / # Sum = count of true
                            float(belowCount))
         ).to_frame('Below')

        result['Above'] = data.groupby('QueryName').apply(
             lambda group: ((group.Aid.isin(resultAbove.index)).sum() /  # Count of total minus count of true
                            float(aboveCount))
         )
        result.plot(kind='bar')
        plt.show()        

    '''
        Session duration for users that visited the site more then 3 times is significantly higher than those who visited 3 times or less
    '''
    def displayAvgTimeSessions(self):
        data = dataManager.loadData(["QueryName","TimeStamp","Sid","Aid"],transformFields=False)

        # Get the timestamp difference
        def get_stats(group):
            return (pd.to_datetime(group['TimeStamp'])-pd.to_datetime(group['TimeStamp']).shift()).fillna(0).sum() / np.timedelta64(1, 's');

        result = data.groupby('Aid').apply(
             lambda group: (group.Sid.nunique() <= 3) # 3 is the average
         )
        resultBelow = result[result == True]
        resultAbove = result[result == False]

        sumOfTimestampBelow3 = data[data.Aid.isin(resultBelow.index)].groupby('Sid').apply(get_stats)
        sumOfTimestampAbove3 = data[data.Aid.isin(resultAbove.index)].groupby('Sid').apply(get_stats)

        timeMeans = (sumOfTimestampBelow3.mean()/60,sumOfTimestampAbove3.mean()/60)
        timeStd = (sumOfTimestampBelow3.std()/60,sumOfTimestampAbove3.std()/60)
        ind = np.arange(2)  # the x locations for the groups
        width = 0.35       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, timeMeans, width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Average time of sessions (Minutes)')
        ax.set_xticks(ind + 0.2)
        ax.set_xticklabels(('Below average stay on site', 'Above average stay on site'))

        plt.show()

        # T-Test with unequal variance
        stats.ttest_ind(sumOfTimestampBelow3, sumOfTimestampAbove3, equal_var=False)

    '''
        Trying to perform a decision tree in order to detect which features best forcast low retention
    '''
    def printDecisionTreeForBelow3(self):
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import numpy as np
        from sklearn.cross_validation import train_test_split
        data = self.dataManager.loadData(["QueryName","TimeStamp","Sid","Aid","Country","IsFirst","Browser","Os","Continent"],transformFields=True)


        features = data.columns[1:4]
        result = data.groupby('Aid').apply(
            lambda group: (group.Sid.nunique()<=3)
        )
        below = result[result == True]
        data['IsBelow'] = data['Aid'].isin(below.index)
        y, _ = pd.factorize(data['IsBelow'])
        X_train, X_test, y_train, y_test = train_test_split(data[features], y, test_size=0.33, random_state=42)
        from sklearn.metrics import accuracy_score
        from sklearn import svm
        #clf = svm.SVC();
        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy_score(y_test, y_pred)
        from sklearn import tree
        i_tree = 0
        for tree_in_forest in clf.estimators_:
            with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
                my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
            i_tree = i_tree + 1



    ''' 
        runs the forward feature selection
    '''
    def run_seq_forw_select(self):
        data = self.dataManager.loadData(["QueryName","TimeStamp","Sid","Aid","Country","IsFirst","Browser","Os","Continent"],transformFields=True)
        result = data.groupby('Aid').apply(
            lambda group: (group.Sid.nunique()<=3)
        )
        below = result[result == True]
        data['IsBelow'] = data['Aid'].isin(below.index)
        
        features = ["QueryName","TimeStamp","Sid","Aid","Country","IsFirst","Browser","Os","Continent"]
        res_forw = seq_forw_select(data,features=features, max_k=4,criterion_func=criterion_func, print_steps=True) 
        
        return res_forw


 

    '''
        For each unique value in the column, we display the percentage of the users that have 3 sessions or more
    '''
    def genericBelowAverageDisplay(self,column):
        data = self.dataManager.loadData(["Sid","Aid",column],transformFields=False,removeFirstK=5)

        result = data.groupby('Aid').apply(
            lambda group: (group.Sid.nunique()<=3)
        )
        below = result[result == True]
        belowData = data.loc[data['Aid'].isin(below.index)]

        col = data.groupby([column]).apply(
            lambda group: (group.Aid.nunique())
        ).to_frame('Total')
        col['Below'] = belowData.groupby([column]).apply(
            lambda group: (group.Aid.nunique())
        )
        (col['Below']/col['Total']).plot(kind='barh', width=0.5)
        plt.tight_layout()
        plt.show()

    '''
        Display Sessions and queries
    ''' 
    def displaySidAndQueryName(self):
        data = self.dataManager.loadData(["Aid","Sid","QueryName"],transformFields=False)        
        
        # Specific Aid
        dataAid = data[data.Aid == "012abc55-5801-494f-a77f-a799f1d855de"]
        colors = cm.gist_ncar(np.linspace(0,1,dataAid.QueryName.nunique()))
        pd.crosstab(dataAid.Sid, dataAid.QueryName).plot.barh(stacked=True, color=colors,figsize=(20, 20))
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
        ax.axvline(x=result.median(),c="red",linewidth=2,ls='dashed')  

        plt.xticks(np.arange(result.min(), result.max()+1, 5.0))
        plt.savefig('GroupByAidCountSidForAll.png')
        plt.show()
      
        
    '''
        Running baseline methods: Random and ItemKNN
    '''
    def compareGRUtoBaselines():
        import gru4rec
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "QueryName" #"QueryName"

        data = self.dataManager.loadData([time_key,item_key,session_key,"Aid","Sid"],removeFirstK=5) 

        train, test = self.dataManager.splitData(data,isRandom=False)
                
        rand = baselines.RandomPred()
        rand.fit(train)
        res = baselines.evaluate_sessions(rand,test,train,cut_off=2, session_key = session_key, item_key = item_key, time_key = time_key)
        print('Sid: Recall@2: {}'.format(res[0]))
        res = baselines.evaluate_sessions(rand,test,train,cut_off=2, session_key = "Aid", item_key = item_key, time_key = time_key)
        print('AidRecall@2: {}'.format(res[0]))
        #itemknn = baselines.ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = session_key, item_key = item_key, time_key = time_key)

        itemknn.fit(train)
        res = baselines.evaluate_sessions(itemknn,test,train,cut_off=2, session_key = session_key, item_key = item_key, time_key = time_key)
        print('Recall@2: {}'.format(res[0]))
        itemknn = baselines.ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = "Aid", item_key = item_key, time_key = time_key)
        itemknn.fit(train)
        res = baselines.evaluate_sessions(itemknn,test,train,cut_off=2, session_key = session_key, item_key = item_key, time_key = time_key)
        print('Recall@2: {}'.format(res[0]))
        
        
            
    '''
        Runs recurrent neural network based on the paper: http://arxiv.org/pdf/1511.06939v4.pdf
    ''' 
    def runGRU4Rec(self):   
        import gru4rec
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "QueryName" #"QueryName"

        data = self.dataManager.loadData([time_key,item_key,session_key,"Aid","Sid"],removeFirstK=5) 

        train, test = self.dataManager.splitData(data,isRandom=False)

        print('Training GRU4Rec') 
        
        batch_size = 200
        for layers in [100,1000]:
            for loss_type in ['top1']:
                for momentum in [3,4,5]:
                    for dropOut in [3,4,5]:
                        try:
                            print('Batch Size: ' + str(batch_size) + ' Dropout: ' + str(float(dropOut)/10.0) + ' Momentum: ' + str(float(momentum)/10.0))
                            gru = gru4rec.GRU4Rec(layers=[layers], loss=loss_type, batch_size=batch_size, dropout_p_hidden=float(dropOut)/10.0, learning_rate=0.05, momentum=float(momentum)/10.0
                                                    ,n_epochs=3,hidden_act = 'tanh', final_act='tanh'
                                                    ,session_key=session_key, item_key=item_key, time_key=time_key)
                            gru.fit(train)
        
                            res = gru.evaluate_sessions_batch(test, cut_off=2, batch_size=batch_size, 
                                                        session_key=session_key, item_key=item_key, time_key=time_key)

                            print('Recall : {}'.format(res[0]))
                            
                        except:
                            print("Unexpected error")
            
    '''
        Runs the GRU, learns from the users that have more than average sessions and predicts on users that have less than 3 sessions and above
    ''' 
    def learnFromExperiencedGRU(self):   
        import gru4rec
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "QueryName" #"QueryName"

        data = self.dataManager.loadData([time_key,item_key,session_key,"Aid","Sid"],removeFirstK=5) 

                
        result = data.groupby('Aid').apply(
             lambda group: (group.Sid.nunique() <= 3) 
         )
        resultBelow = result[result == True]
        resultAbove = result[result == False]

        dataBelow = data[data.Aid.isin(resultBelow.index)]        
        dataAbove = data[data.Aid.isin(resultAbove.index)]

        train, test = self.dataManager.splitData(dataAbove,isRandom=False)
        testBelow = dataBelow[:len(test)]

        print('Training GRU4Rec') 
        
        batch_size = 200
        for layers in [1000]:
            for loss_type in ['top1']:
                for momentum in [3,4,5]:
                    for dropOut in [3]:
                        try:
                            print('Batch Size: ' + str(batch_size) + ' Dropout: ' + str(float(dropOut)/10.0) + ' Momentum: ' + str(float(momentum)/10.0))
                            gru = gru4rec.GRU4Rec(layers=[1000], loss=loss_type, batch_size=batch_size, dropout_p_hidden=float(dropOut)/10.0, learning_rate=0.05, momentum=float(momentum)/10.0
                                                    ,n_epochs=3,hidden_act = 'tanh', final_act='tanh'
                                                    ,session_key=session_key, item_key=item_key, time_key=time_key)
                            gru.fit(train)
        
                            res = gru.evaluate_sessions_batch(test, cut_off=2, batch_size=batch_size, 
                                                        session_key=session_key, item_key=item_key, time_key=time_key)

                            print('Above Test Recall : {}'.format(res[0]))
                            
                            res = gru.evaluate_sessions_batch(testBelow, cut_off=2, batch_size=batch_size, 
                                                        session_key=session_key, item_key=item_key, time_key=time_key)
                            print('Below Recall : {}'.format(res[0]))
                        except:
                            print("Unexpected error")

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
    def runGRU4RecForSpecificAid(self,numOfAids=10):   
    
        import gru4rec
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "QueryName"

        data = self.dataManager.loadData([time_key,item_key,"Aid","Sid"],removeFirstK=5,onlyFirstFile=True) 

        result = data.groupby('Aid').apply(
             lambda group: (group.Sid.nunique())
         )

        result.sort_values(inplace=True,ascending=False)
        
        for selectedAid in result.keys()[2:(numOfAids+2)]:
            idx = self.dataManager.fields.index("Aid");
            ret = self.dataManager.mcle.all_encoders_[idx].inverse_transform(selectedAid);
            print(str(ret))    
            aidData = data.loc[data['Aid'] == selectedAid]

            train, test = self.dataManager.splitData(aidData,isRandom=False)

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
 
            specificSession = test[test[session_key] == test[session_key].values[0]]
            preds = gru.predict_next_batch(specificSession[session_key], specificSession[item_key], None, len(specificSession))
                    
       
            print('Correct: {}'.format(correct))
            if(len(test)-batch_size-1 > 0):
                print('Accuracy: {}'.format(float(correct)/float(len(test)-batch_size-1)))

    def getRecallForPrevious(self):
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "Pn" #"QueryName"
        data = self.dataManager.loadData(["TimeStamp","Pn","QueryName","Sid","Aid"],transformFields=False,removeFirstK=10,isFirstMonth=True) 
        
        data.is_copy = False
        data.sort_values([session_key,"TimeStamp"], inplace=True) # Sort by "TimeStamp"
                
        lastSid = 0;
        totalCount = 0;
        recallSuccess = 0;
        for i in range(len(data)-1):
            fromQuery = data[item_key].values[i];
            toQuery = data[item_key].values[i+1];
            if(data[session_key].values[i] != lastSid): # Switched Session
                lastSid = data[session_key].values[i]; # not adding to total count (not counting first of every session)
            else:
                if(fromQuery == toQuery):
                    recallSuccess = recallSuccess + 1;
                totalCount = totalCount + 1
        print(float(float(recallSuccess) /float(totalCount))) 

        
    def runGRU4RecAndDisplaySessions(self):
        session_key = "Sid" #"Aid" # Or Sid
        time_key = "TimeStamp"
        item_key = "Pn" #"QueryName"
        data = self.dataManager.loadData(["TimeStamp","Pn","QueryName","Sid","Aid"],transformFields=False,removeFirstK=10,isFirstMonth=True)    

        train, test = self.dataManager.splitData(data,isRandom=False)

        print('Training GRU4Rec') 

        batch_size = 200
        loss_type = 'top1'
        momentum = 3
        dropOut = 3
        for layers in [100,1000]:   
            print('Layers: ' + str(layers) + ' Batch Size: ' + str(batch_size) + ' Dropout: ' + str(float(dropOut)/10.0) + ' Momentum: ' + str(float(momentum)/10.0))
            gru = GRU4Rec(layers=[layers], loss=loss_type, batch_size=batch_size, dropout_p_hidden=float(dropOut)/10.0, learning_rate=0.05, momentum=float(momentum)/10.0
                                    ,n_epochs=3,hidden_act = 'tanh', final_act='tanh'
                                    ,session_key=session_key, item_key=item_key, time_key=time_key)
            gru.fit(train)

            res = gru.evaluate_sessions_batch(test, cut_off=2, batch_size=1, # Batch size = 1 inorder to better view each session
                                        session_key=session_key, item_key=item_key, time_key=time_key,display=True)

            print('Recall : {}'.format(res[0]))

    '''
        Runs PCA on the data
    '''
    def runPCA(self,data, n_components='mle', plot=False, pivot = 'Aid'):
        pca = PCA(n_components, copy=True)
        # Plot the PCA spectrum
        print "Running PCA"
        startTime = clock()
        new_data = pca.fit_transform(data)
        print "finished running after {0} secs".format(clock()-startTime)
        if plot==True:
            plt.figure(1, figsize=(4, 3))
            plt.clf()
            plt.axes([.2, .2, .7, .7])
            plt.plot(pca.explained_variance_, linewidth=2)
            plt.axis('tight')
            plt.xlabel('n_components')
            plt.ylabel('explained_variance_')
            plt.savefig("PCA{0}.png".format(pivot))
        return new_data

    '''
        Runs K means on the Dataset        
    '''
    def runKMeans(self, pivot, file = False, normalize = False, factor = 100, n_components=False, n_clusters = 5):
        
        # Preparing the data : loading, normalizing (if selected) and selecting 100k record randomly.
        data = self.loadOperationsByOneOfK(False, pivot, normalize, factor)
        np.random.shuffle(data)
        #data = data[1:50000,:] # Higher amount of vectors cause memory error on silhouette_score

        # Preparing the data to compare, creating an average vector of the pivots.
        if file!= False:
            avgVec = np.mean(data,axis = 0)        
            fid = open(file, 'w')
            np.set_printoptions(precision=3,suppress = True) # Prettier printing.
            fid.write(str(avgVec))

        if n_components != False:
            data = self.runPCA(data,n_components)       
        

        
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

    '''
        Runs Hirarchical Clustering on the Dataset,
        Creating a Dendrogram visualizing the clustering process for further analysis.
    '''
    def runHierarchicalClustering(self,file,n_clusters=2, pivot='Aid',normalize=False,factor=100, n_components=False):        
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

        if n_components != False:
            data = self.runPCA(data,n_components)
         
        # Compute clustering           
        print("Running hierarchical clustering for {0} clusters".format(n_clusters))
        startTime = clock()
        ward = linkage(data, 'average', metric = 'cosine')
        print("finished running after {0} secs".format(clock()-startTime))
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
            count_sort = 'descendent',
            distance_sort = False
        )
        plt.savefig("plot{0}{1}attempt.png".format(pivot,n_clusters))        

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
            if (i%500000 == 0 and i!=0):
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

    '''
        Runs T-SNE on the Dataset        
    '''
    def runTSNE(self, pivot, normalize = False, factor = 100):
        
        # Preparing the data : loading, normalizing (if selected) and selecting 100k record randomly.
        data = self.loadOperationsByOneOfK(False, pivot, normalize, factor)
        np.random.shuffle(data)

        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        X_tsne = TSNE(learning_rate=100).fit_transform(data)
        X_pca = PCA().fit_transform(data)
        figure(figsize=(10, 5))
        subplot(121)
        scatter(X_tsne[:, 0], X_tsne[:, 1])
        subplot(122)
        scatter(X_pca[:, 0], X_pca[:, 1])
        plt.show();


            