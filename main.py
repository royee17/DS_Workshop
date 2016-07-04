from AlgorithmManager import *
from DataManager import *

# Lior's Directory
liorDir = "D:\DataScienceWorkshop"

# Royee's Directory
royeeDir = ""

# Nadav's Directory
nadavfile = "attempt2.txt"

dataManager = DataManager(liorDir)
algorithmManager = AlgorithmManager(dataManager)
#algorithmManager.runTSNE(pivot = 'Aid', normalize = False)

#algorithmManager.runHierarchicalClustering(file="HierarchicalClusteringAttempt1.txt", normalize = False,n_components=5)
#algorithmManager.runHierarchicalClustering(pivot='Sid', file=False, normalize = False,n_components = 5)

#algorithmManager.runKMeans(pivot = 'Aid', file="Aidcenters5PCA.txt", normalize = False,n_clusters=5, n_components = 5)
#algorithmManager.runKMeans(pivot = 'Aid', file="AidcentersNoSil2.txt", normalize = True)
#algorithmManager.runKMeans(pivot = 'Sid', file="Sidcenters5PCA.txt", normalize = False,n_clusters=5, n_components = 5)
#algorithmManager.runKMeans(pivot = 'Sid', file="SidcentersAfterPCA1.txt", normalize = False,n_components=5)

#algorithmManager.displayAidBySid()
#algorithmManager.displayCountByQueryName()
#algorithmManager.displaySidAndQueryName();
#algorithmManager.displayGraph()
#algorithmManager.runKMeans()

#algorithmManager.displayCountSessionByQueryName()
#algorithmManager.displayGraph()
#algorithmManager.genericBelowAverageDisplay("Browser");

#algorithmManager.genericBelowAverageDisplay("Country")
#algorithmManager.genericBelowAverageDisplay("QueryName")

#algorithmManager.runGRU4Rec()

algorithmManager.getRecallForPrevious()

algorithmManager.runGRU4RecAndDisplaySessions()