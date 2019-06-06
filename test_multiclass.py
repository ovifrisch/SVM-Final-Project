import numpy as np
import svm

num_samples = 90

clf = svm.svm()

# generate 3 classes of data
zeros = np.random.normal(loc=0, scale = 1, size=int(num_samples/3))
ones = np.random.normal(loc=5, scale = 1, size=int(num_samples/3))
twos = np.random.normal(loc=10, scale = 1, size=int(num_samples/3))

xs = np.transpose(np.matrix(np.append(np.append(zeros, ones), twos)))
#ys = np.append(np.append(np.zeros(int(num_samples/3)), np.ones(int(num_samples/3))), np.ones(int(num_samples/3)) + 1)

ys = np.append(np.ones(30), np.ones(60) * -1)

clf.fit(xs, ys)
# print("yo")
# clf.predict(xs)
