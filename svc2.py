import numpy as np
import multiprocessing as mp
from multiprocess_svm import MP_SVM
from sklearn.utils import shuffle

class SVC:
	"""
	Multiclass SVM
	"""
	def __init__(
	self,
	kernel_type='rbf',
	C=1,
	gamma=1,
	degree=3,
	tolerance=0.01,
	epsilon=0.01,
	solver = "smo"):
		self.__kernel_type = "rbf"
		self.__gamma = gamma
		self.__degree = degree
		self.__C = C
		self.__tol = tolerance
		self.__error_cache = {}
		self.__eps = epsilon
		self.__solver = solver

	def get_y_matrix(self):
		"""
		Returns
		-------
		ymat - ndarray
			Label vectors for each classifier
		"""
		y_mat = np.zeros((self.__size, self.__num_classes))
		for i in range(self.__num_classes):
			this_y = np.copy(self.__ys)
			for j in range(self.__size):
				if (self.__ys[j] == i):
					this_y[j] = 1
				else:
					this_y[j] = -1
			y_mat[:, i] = this_y
		return y_mat

	def __train_one(self, i):
		"""
		Parameters:
		-----------
		i : int
			- Index for the y_mat matrix

		This function trains the ith classifier and
		appends the instance of the classifier to
		the multiprocessing queue
		"""
		y = self.__y_mat[:, i]
		clf = MP_SVM(self.__kernel_type, self.__C, self.__gamma, self.__degree, self.__tol, self.__eps, self.__solver)
		clf.fit(self.__xs, y)
		self.__classifiers.put((i, clf))

	def fit(self, x, y):
		"""
		Parameters:
		----------
		x : ndarray
			-training examples
		y : ndarray
			-training classes

		Returns:
		self : obj

		assume classes are 0...N
		"""
		# build a classifier for each class
		self.__size = x.shape[0]
		self.__num_classes = np.unique(y).shape[0]
		self.__xs = x
		self.__ys = y

		# y_mat is the matrix where each column are the class labels for each of the N classifiers
		if (self.__num_classes <= 2):
			self.__y_mat = y

		else:
			self.__y_mat = self.get_y_matrix()

		

		# train each of the classifiers serially
		self.__classifiers = []
		for i in range(self.__num_classes):
			clf = MP_SVM(self.__kernel_type, self.__C, self.__gamma, self.__degree, self.__tol, self.__eps, self.__solver)
			if (self.__num_classes == 2):
				clf.fit(self.__xs, self.__y_mat)
				self.__classifiers.append(clf)
				break
			clf.fit(self.__xs, self.__y_mat[:, i])
			self.__classifiers.append(clf)






	def predict(self, x):
		"""
		Parameters:
		-----------
		x : ndarray
			- input matrix

		Returns:
		--------
		preds : ndarray
			- predictions for each input vector
		"""
		class_predictions = np.zeros((xs.shape[0], self.__num_classes))
		for i in range(len(self.__classifiers)):
			preds = self.__classifiers[i].predict(x)
			if (self.__num_classes == 2):
				return preds
			for j in range(preds.shape[0]):
				if (preds[j] == 1):
					class_predictions[j, i] += 1
				else:
					class_predictions[j, :] += 1
					class_predictions[j, i] -= 1
		return np.argmax(class_predictions, 1)



if __name__ == "__main__":

	# BINARY
	# num_samples = 500
	# pos = np.random.normal(loc=0, scale = 1, size=int(num_samples/2))
	# neg = np.random.normal(loc=5, scale = 1, size=int(num_samples/2))
	# xs = np.transpose(np.matrix(np.append(pos, neg)))
	# ys = np.append(np.ones(int(num_samples/2)), -1*np.ones(int(num_samples/2)))
	# xs, ys = shuffle(xs, ys)
	# s = SVC()
	# s.fit(xs, ys)
	# print("done training")
	# preds = s.predict(xs)
	# acc = np.sum(preds == ys) / num_samples
	# print(acc)


	# MULTICLASS
	num_samples = 200
	one = np.random.normal(loc=0, scale = 1, size=int(num_samples/3))
	two = np.random.normal(loc=5, scale = 1, size=int(num_samples/3))
	three = np.random.normal(loc=10, scale = 1, size=int(num_samples/3))
	xs = np.transpose(np.matrix(np.append(np.append(one, two), three)))
	ys = np.append(np.append(np.zeros((int(num_samples/3))), np.ones((int(num_samples/3)))), np.ones(int(num_samples / 3)) + 1)
	xs, ys = shuffle(xs, ys)
	s = SVC()
	s.fit(xs, ys)
	print("done training")
	preds = s.predict(xs)
	acc = np.sum(preds == ys) / num_samples
	print(acc)
