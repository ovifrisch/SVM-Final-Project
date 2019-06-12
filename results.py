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
	return round(sum(y==yhat) / y.shape[0], 3)

dimensions = [1, 5, 10, 20, 50, 100, 300, 1000]
samples = [100, 200, 500, 1000, 2000, 5000]
min_dims = [1, 5, 10, 20, 50]
min_samples = [100, 200, 500]
labels =10

x = gen_ndim_norm(N = 1, M = 1000, C = labels)
y = gen_labels(1000, labels)
x, y = shuffle(x, y)
clf = SVC(max_iter=1000)
start = time.time()
clf.fit(x, y)
train_time = round(time.time() - start, 3)
start = time.time()
yhat = clf.predict(x)
test_time = round(time.time() - start, 3)
acc = accuracy(y, yhat)
print("  Accuracy: " + str(acc) + "  Train Time: " + str(train_time))

# for d in min_dims:
#     for s in min_samples:
#         x = gen_ndim_norm(N = d, M = s, C = labels)
#         y = gen_labels(s, labels)
#         x, y = shuffle(x, y)
#         clf = SVC()
#         start = time.time()
#         clf.fit(x, y)
#         train_time = round(time.time() - start, 3)
#         start = time.time()
#         yhat = clf.predict(x)
#         test_time = round(time.time() - start, 3)
#         acc = accuracy(y, yhat)
#         print("Samples: " + str(s) + "  Dims: " + str(d) + "  Accuracy: " + str(acc) + "  Train Time: " + str(train_time))
