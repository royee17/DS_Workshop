from datamanager import *

data = DataManager("C:/Users/sony/school/year3/semesterB/data_science_workshop/data/data")
ajax_events_Df = data.loadData()

print(ajax_events_Df)