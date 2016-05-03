from AlgorithmManager import *
from DataManager import *

# Lior's Directory: C:\Workspace\University\DataScienceWorkshop
liorDir = "C:\Workspace\University\DataScienceWorkshop"
# Royee's Directory: 
royeeDir = ""
# Nadav's Directory:
nadavDir = ""

dataManager = DataManager(liorDir)

algorithmManager = AlgorithmManager(dataManager)

#algorithmManager.runKMeans()

algorithmManager.runGRU4Rec()