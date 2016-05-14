import pandas as pd
import numpy as np
import os
import os.path
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = np.asarray(columns)

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            self.all_labels_ = np.ndarray(shape=self.columns.shape,
                                          dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelEncoder()
                # fit and transform labels in the column
                dframe.loc[:, column] =\
                    le.fit_transform(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
                self.all_labels_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                dframe.loc[:, column] = le.fit_transform(
                        dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return dframe

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[
                    idx].transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .transform(dframe.loc[:, column].values)
        return dframe.loc[:, self.columns].values

    def inverse_transform(self, dframe,specificColumn=None):
        """
        Transform labels back to original encoding.
        """    

        if self.columns is not None:
            
            for idx, column in enumerate(self.columns):
                if specificColumn is not None and column == specificColumn:
                    dframe.loc[:, specificColumn] = self.all_encoders_[idx]\
                            .inverse_transform(dframe.loc[:, specificColumn].values)
                    return dframe
                else:
                    dframe.loc[:, column] = self.all_encoders_[idx]\
                        .inverse_transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        return dframe

class DataManager(object):
    
    '''
        Init

        Params:
            dir: Full path to the directory that holds each date folder (root directory)
    '''
    def __init__(self,dir):
        self.rootdir = dir
        self.fields = [];
        self.mcle = {}
       
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

        # get training, and test sets
        train = df.iloc[l[:trainLen]]
        test     = df.iloc[l[trainLen:]]
        
        print("Train Size: " + str(len(train)) + " Test Size: " + str(len(test)))

        return train, test

    '''
        Load Data

        Return: Data object (all categorical fields are converted to integers)
            (as pandas dataframes)

        Params:
            fields (not required): Column names that we want to read
            transformFields (not required): Transforms the String fields into int
    '''
    def loadData(self,fields=False,transformFields=True):

        self.fields = fields;
        if(not self.fields):
            self.fields = ["Browser","Device","Os","Resolution","Continent", "Country","Sid","Aid","Pn", "QueryName","Response","Result","Status","StatusText","Type"]
       
        ajax_events_list = []
        for subdir, dirs, files in os.walk(self.rootdir):
            for dir in dirs:
                # Load the csv and append to a list
                ajax_events_list.append(pd.read_csv(os.path.join(self.rootdir,dir,'ajax_events.csv'),usecols=self.fields))

        # Concat the list of the dataframes
        df = pd.concat(ajax_events_list)
        
        if(transformFields):                                             
            # Transform all the string columns into integers
            self.mcle = MultiColumnLabelEncoder(columns=self.fields)

            # Returns a matrix of integers 
            res = self.mcle.fit_transform(df)
        else:
            res = df

        print("The data has been loaded")
        
        return res

    '''
        Searches for the label in the decoded dataset and then encodes back and returns the encoded value for the label 

        Ex: dataManager.getEncodedByLabel(data,"Aid","21674def-1c93-46e5-95ab-015e904fb10f")

        Params:
            data: The encoded data (transformFields=true in load data)
            field: The field that the label comes from
            value: The value we are looking for
    '''
    def getEncodedByLabel(self,data,field,value):

        idx = self.fields.index(field);
        # Decode the dataset to the regular dataset
        inversed = self.mcle.all_encoders_[idx].inverse_transform(data.loc[:, field].values);

        # Search for the label and get the encoded value
        ret = self.mcle.all_encoders_[idx].transform(inversed[inversed == value])[0];

        return ret;
