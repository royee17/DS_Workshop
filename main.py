from AlgorithmManager import *
from DataManager import *

# Lior's Directory
liorDir = "D:\DataScienceWorkshop"

# Royee's Directory
royeeDir = ""

# Nadav's Directory
nadavDir = ""

dataManager = DataManager(liorDir)

algorithmManager = AlgorithmManager(dataManager)

#algorithmManager.runKMeans()

algorithmManager.runGRU4RecForSpecificAid()

#algorithmManager.displayAidBySid();
#algorithmManager.displayCountByQueryName()

