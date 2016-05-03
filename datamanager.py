import pandas as pd
import numpy as np
import os
import os.path
import warnings
warnings.filterwarnings('ignore', 'numpy equal will not check object identity in the future')

class MultiColumnLabelEncoder:
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

class DataManager:
    
    '''
        Init

        Params:
            dir: Full path to the directory that holds each date folder (root directory)
    '''
    def __init__(self,dir):
        self.rootdir = dir
       
    '''
        Load Data

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
                break;

        # Concat the list of the dataframes
        df = pd.concat(ajax_events_list)
                                                                        
        # Transform all the string columns into integers
        mcle = MultiColumnLabelEncoder(columns=fields)
        mcle.fit(df)

        # Returns a matrix of integers 
        return mcle.transform(df)