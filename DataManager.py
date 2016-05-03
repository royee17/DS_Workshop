import pandas as pd
import numpy as np
import os
import os.path


class MultiColumnLabelEncoder(object):
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        from sklearn.preprocessing import LabelEncoder

        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class DataManager(object):
    
    '''
        Init

        Params:
            dir: Full path to the directory that holds each date folder (root directory)
    '''
    def __init__(self,dir):
        self.rootdir = dir
      
    
    '''
    Return: training, test
            (as pandas dataframes)
    Params:
        df: pandas dataframe
        trainPerc: float | percentage of data for training set (default=0.8)
        testPerc: float | percentage of data for test set (default=0.2)
        isRandom: bool | shuffle the data before the split (default = True)
    '''
    def splitData(self,df, trainPerc=0.8, testPerc=0.2, isRandom=True):

        # create random list of indices
        from random import shuffle
        N = len(df)
        l = range(N)
        if(isRandom):
            shuffle(l)

        # get splitting indicies
        trainLen = int(N*trainPerc)
        testLen  = int(N*testPerc)

        # get training, and test sets
        train = df.ix[l[:trainLen]]
        test     = df.ix[l[trainLen:]]
        
        print("Train Size: " + str(len(train)) + " Test Size: " + str(len(test)))

        return train, test

    '''
        Load Data

        Return: Data object (all categorical fields are converted to integers)
            (as pandas dataframes)

        Params:
            fields (not required): Column names that we want to read
    '''
    def loadData(self,fields=False):

        if(not fields):
            fields = ["Browser","Device","Os","Resolution","Continent", "Country","Sid","Aid","Pn", "QueryName","Response",	"Result","Status","StatusText","Type"]
       
        ajax_events_list = []
        for subdir, dirs, files in os.walk(self.rootdir):
            for dir in dirs:
                # Load the csv and append to a list
                ajax_events_list.append(pd.read_csv(os.path.join(self.rootdir,dir,'ajax_events.csv'),usecols=fields))

        # Concat the list of the dataframes
        df = pd.concat(ajax_events_list)
                                                                        
        # Transform all the string columns into integers
        mcle = MultiColumnLabelEncoder(columns=fields)
        mcle.fit(df)

        # Returns a matrix of integers 
        res = mcle.transform(df)

        print("The data has been loaded")
        
        return res
