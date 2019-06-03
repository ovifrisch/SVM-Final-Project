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



	def __smo(self):
		self.alphas = [0] * self.__size
		self.__b = 0
		passes = 0
		while (passes < self.__max_passes):
			print("Pass " + str(passes) + "/" + str(self.__max_passes))
			num_changed_alphas = 0
			for i in range(self.__size):
				Ei = self.__train_predict(self.xs[i, :])[0] - self.ys[i]
				if ((self.ys[i] * Ei < -self.__tol and self.alphas[i] < self.__C) or (self.ys[i] * Ei > self.__tol and self.alphas[i] > 0)):
					while (True):
						j = random.randint(0, self.__size - 1)
						if (j != i):
							break

					Ej = self.__train_predict(self.xs[j, :])[0] - self.ys[j]
					alpha_i_old = self.alphas[i]
					alpha_j_old = self.alphas[j]

					s = self.ys[i] * self.ys[j]
					if (s > 0):
						L = max(0, self.alphas[i] + self.alphas[j] - self.__C)
						H = min(self.__C, self.alphas[i] + self.alphas[j])
					else:
						L = max(0, self.alphas[i] - self.alphas[j])
						H = min(self.__C, self.__C + self.alphas[j] - self.alphas[i])
					if (L == H):
						continue

					n = (2*self.__kernel.apply(self.xs[i, :], self.xs[j, :])) - self.__kernel.apply(self.xs[i, :], self.xs[i, :]) - self.__kernel.apply(self.xs[j, :], self.xs[j, :])
					if (n >= 0):
						continue

					self.alphas[j] -= (self.ys[j] * (Ei - Ej)) / n
					if (self.alphas[j] > H):
						self.alphas[j] = H
					elif (self.alphas[j] < L):
						self.alphas[j] = L


					if (abs(self.alphas[j] - alpha_j_old) < 0.00001):
						continue

					self.alphas[i] += s*(alpha_j_old - self.alphas[j])
					b1 = self.__b - Ei - (self.ys[i] * (self.alphas[i] - alpha_i_old) * self.__kernel.apply(self.xs[i, :], self.xs[i, :])) - (self.ys[j] * (self.alphas[j] - alpha_j_old) * self.__kernel.apply(self.xs[i, :], self.xs[j, :]))
					b2 = self.__b - Ej - (self.ys[i] * (self.alphas[i] - alpha_i_old) * self.__kernel.apply(self.xs[i, :], self.xs[j, :])) - (self.ys[j] * (self.alphas[j] - alpha_j_old) * self.__kernel.apply(self.xs[j, :], self.xs[j, :]))

					if (self.alphas[i] > 0 and self.alphas[i] < self.__C):
						self.__b = b1
					elif (self.alphas[j] > 0 and self.alphas[j] < self.__C):
						self.__b = b2
					else:
						self.__b = (b1 + b2) / 2

					num_changed_alphas  += 1
				# end if
			# end for

			if (num_changed_alphas == 0):
				passes += 1
			else:
				passes += 1
