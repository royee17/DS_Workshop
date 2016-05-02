import pandas as pd
import numpy as np
import os
import os.path

class DataManager:
    
    ajax_events_Df = {}

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
                ajax_events_list.append(pd.read_csv(os.path.join(self.rootdir,dir,'ajax_events.csv'),usecols=fields))

        self.ajax_events_Df = pd.concat(ajax_events_list)

        return self.ajax_events_Df