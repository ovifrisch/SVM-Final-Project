import svm
import numpy as np
from sklearn.svm import SVC
from kernels import Kernel
from sklearn.utils import shuffle

num_samples = 10

if __name__ == "__main__":
	clf = svm.svm("rbf", 1, 1)
	posX = np.random.normal(loc=0, scale = 1, size=int(num_samples/2))
	negX = np.random.normal(loc=10, scale=1, size=int(num_samples/2))
	X = np.transpose(np.matrix(np.append(posX, negX)))
	ys = np.append(np.ones(int(num_samples / 2)), (-1 * np.ones(int(num_samples / 2))))
	clf.fit(X, ys)

	posX_test = np.random.normal(loc=0, scale = 1, size=int(num_samples / 2))
	negX_test = np.random.normal(loc=10, scale = 1, size=int(num_samples / 2))
	X_test = np.append(posX_test, negX_test)
	X_test, ys = shuffle(X_test, ys)
	print(X_test.shape)
	print(ys.shape)
	pred_ys = clf.predict(np.transpose(np.matrix(X_test)))

	acc = np.sum(ys==pred_ys) / num_samples
	print("Accuracy: " + str(acc))

	# alphas = clf.alphas
	# pos = alphas[0:int(len(alphas) / 2)]
	# neg = alphas[int(len(alphas) / 2):]
	# pos_supp = len([x for x in pos if x != 0])
	# neg_supp = len([x for x in neg if x != 0])
	# print(pos_supp, neg_supp)
	#
	#
	# clf2 = SVC(kernel="rbf")
	# clf2.fit(X, ys)
	# print(clf2.n_support_)
