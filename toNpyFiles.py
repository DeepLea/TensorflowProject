import tensorflow.contrib.learn as skflow
from sklearn.datasets import fetch_lfw_people
import numpy as np

lfw_people = fetch_lfw_people(min_faces_per_person = 50, resize = 0.88, funneled = False,color = False, slice_ = (slice(0,250),slice(0,250)))  # Det har tar tid, upp till 2 min

np.save("data.npy",lfw_people.data)
np.save("target.npy",lfw_people.target)
np.save("target_names.npy",lfw_people.target_names)

