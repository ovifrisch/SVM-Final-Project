import numpy as np
from kernels import Kernel
import random

class svm:
	"""
	A class used to perform binary (and multiclass) classification using
	a Support Vector Machine (SVM)
	"""

	def __init__(self, kernel_type='rbf', C=1, gamma=1, degree=3):
		"""
		Parameters
		----------
		kernel_type : str
			The type of kernel to be used (support for "rbf", "linear", "poly")
		C : float
			Parameter for controlling compromise between soft and hard margin violations
		gamma : float
			Kernel coefficient for RBF kernel
		degree: float
			Exponent for ploynomial kernel
		"""
		self.__kernel = Kernel(kernel_type, gamma, degree)
		self.__C = C
		self.__tol = 0.01
		self.__max_passes = 500

	# Accepts X and Y training data and creates the model using SMO
	def fit(self, X, y):
		"""
		Parameters
		----------
		X : ndarray
			Training x
		y : ndarray
			Training y
		"""

		# first make sure params are good
		if (not self.__validate_fit_params(X, y)):
			print("Bad params")
			return

		self.xs = X
		self.ys = y
		self.__size = X.shape[0]

		# Computationally Intensive SMO
		self.__smo()

	def __train_predict(self, xs):
		preds = np.zeros((xs.shape[0])) # predictions
		for i in range(xs.shape[0]):
			x = xs[i, :]
			preds[i] = self.__sum_w_tranpose_x(x) + self.__b
		return preds

	def __objective(self):
		a = sum(self.alphas)
		b = 0
		for i in range(self.__size):
			for j in range(self.__size):
				b += self.ys[i]*self.ys[j]*self.alphas[i]*self.alphas[j]*self.__kernel.apply(self.xs[i, :], self.xs[j, :])
		return a - (0.5 * b)


	# Accepts a list of testing samples and returns the predicted classes
	# Modeled after discriminant function in (6)
	def predict(self, xs):
		"""
		Parameters
		----------
		xs : ndarray
			testing samples

		Return : ndarray
			predicted classes of testing samples
		"""
		preds = np.zeros((xs.shape[0])) # predictions
		for i in range(xs.shape[0]):
			x = xs[i, :]
			preds[i] = np.sign(self.__sum_w_tranpose_x(x) + self.__b)
		return preds


	def __sum_w_tranpose_x(self, x):
		"""
		Implements the summation part of (6) in BTL using only the support vectors
		"""
		kernel_vector = np.apply_along_axis(self.__kernel.apply, 1, self.xs, x2=x)
		zipped = zip(kernel_vector, self.ys, self.alphas)
		products = [np.prod(tup) for tup in zipped]
		return sum(products)


	def __validate_fit_params(self, X, y):
		"""
		Parameters
		----------
		X : ndarray
			Training X
		y : ndarray
			Training y

		Return
		------
		True if parameters are valid
		False otherwise
		"""
		# need mxn matrix
		if (len(X.shape) != 2):
			return False

		# y must have 1 column
		if (len(y.shape) > 1):
			return False

		# number of rows in x must equal number of rows in y
		if (X.shape[0] != y.shape[0]):
			return False

		# Everything good
		return True

	def examine_example(self, i1):
		y1 = self.ys[i1]
		alph1 = self.alphas[i1]

		if (alph1 > 0 and alph1 < C):
			E1 = self.error_cache[i1]
		else:
			E1 = self.__train_predict(self.xs[i1, :]) - y1

		r1 = y1*E1
		if ((r1 < -self.__tol and alph1 < self.__C) or (r1 > self.__tol and alph1 > 0)):



	def __smo(self):
		num_changed = 0
		examine_all = 1
		self.alphas = [0] * self.__size
		while (num_changed > 0 or examine_all):
			num_changed = 0
			if (examine_all):
				for k in range(self.__size):
					num_changed += self.examine_example(k)
			else:
				for k in range(self.__size):
					if (self.alphas[k] != 0 and self.alphas[k] != self.__C):
						num_changed += self.examine_example(k)
			if (examine_all == 1):
				examine_all = 0
			elif (num_changed == 0):
				examine_all = 1
