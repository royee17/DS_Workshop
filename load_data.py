
# coding: utf-8

# In[22]:

import pandas as pd
import numpy as np
import os, os.path


# In[24]:

os.getcwd()


# In[53]:

rootdir = "C:\Users\sony\school\year3\semesterB\data_science_workshop\data\data"
ajax_events_Df = []
for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        ajax_events_Df.append(pd.read_csv(os.path.join(rootdir,dir,'ajax_events.csv')))


# In[ ]:

#frame = pd.DataFrame()


# In[ ]:

#frame = pd.concat(ajax_events_Df)


# In[38]:

for i in range(31):
    print ajax_events_Df[i]


# In[ ]:




# In[ ]:



