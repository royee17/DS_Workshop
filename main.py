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



'''
    Trying to recover the ndarray


str = ""
for i in range(64):
    str = str + "i4,"
str = str + "i4"
AidArray = np.dtype(str)
vects = np.fromfile(nadavfile, dtype = AidArray)
print "\n\n\n The data recovered:\n\n"
print(vects[1:10])
'''
algorithmManager.runKMeans(pivot = 'Aid', file="AidcentersNoSil1.txt", normalize = False)
algorithmManager.runKMeans(pivot = 'Aid', file="AidcentersNoSil2.txt", normalize = True)
algorithmManager.runKMeans(pivot = 'Sid', file="SidcentersNoSil3.txt", normalize = False)
algorithmManager.runKMeans(pivot = 'Sid', file="SidcentersNoSil4.txt", normalize = True)
#algorithmManager.displayAidBySid()
#algorithmManager.displayCountByQueryName()


#algorithmManager.runKMeans()

#algorithmManager.runGRU4RecForSpecificAid()




