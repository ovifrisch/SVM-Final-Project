import numpy as np
from svc2 import SVC
import time
from sklearn.utils import shuffle

def gen_ndim_norm(N, M, C):
	# map class label to its normal distb ctr.
	def loc_func(i):
		return i * 5

	partition = int(M / C) # each class will have this many samples
	mat = np.zeros((partition*C, N))
	for i in range(C):
		for j in range(partition):
			mat[(i*partition) + j, :] = np.random.normal(loc=loc_func(i), scale = 1, size=N)
	return mat

def gen_labels(M, C):
	# create a 1 dimensional ndarray like: [0,0,0,1,1,1] if M was 6 and c was 2
	partition = int(M / C)
	y = np.zeros(partition * C)
	for i in range(C):
		y[i*partition:i*partition+partition] = np.repeat(i, partition)
	return y



def accuracy(y, yhat):
	return sum(y==yhat)

labels = 2
dimensions = 1
samples = 100

x = gen_ndim_norm(N = dimensions, M = samples, C = labels)
y = gen_labels(samples, labels)
print(x)
print(y)
x, y = shuffle(x, y)
clf = SVC()
clf.fit(x, y)
yhat = clf.predict(x)
acc = accuracy(yhat, y)
print("Samples: " + str(100) + "  Dims: " + str(1) + "  Accuracy: " + str(acc))
