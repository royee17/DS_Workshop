from AlgorithmManager import *
from DataManager import *

# Lior's Directory: C:\Workspace\University\DataScienceWorkshop
# Royee's Directory: 
# Nadav's Directory:
dataManager = DataManager("C:\Workspace\University\DataScienceWorkshop")

data = dataManager.loadData(["QueryName","Aid"])

algorithmManager = AlgorithmManager()
algorithmManager.runKMeans(data)