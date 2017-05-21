import numpy as np

a = np.load("target_names.npy")
np.savetxt("names.csv", a, delimiter=",", fmt="%s")

a = np.load("testAcc.npy")
np.savetxt("testAcc.csv", a, delimiter=",")

a = np.load("trainAcc.npy")
np.savetxt("trainAcc.csv", a, delimiter=",")

a = np.load("trainCost.npy")
np.savetxt("trainCost.csv", a, delimiter=",")

a = np.load("testCost.npy")
np.savetxt("testCost.csv", a, delimiter=",")

a = np.load("personRig.npy")
np.savetxt("personRig.csv", a, delimiter=",")

a = np.load("personNum.npy")
np.savetxt("personNum.csv", a, delimiter=",")
