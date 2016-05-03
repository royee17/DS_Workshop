from AlgorithmManager import *
from DataManager import *

# Lior's Directory: C:\Workspace\University\DataScienceWorkshop
# Royee's Directory: 
# Nadav's Directory:
dataManager = DataManager("C:\Workspace\University\DataScienceWorkshop")

algorithmManager = AlgorithmManager(dataManager)

#algorithmManager.runKMeans()

algorithmManager.runGRU4Rec()