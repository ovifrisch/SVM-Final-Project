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
		self.__tol = 1

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

		# Initialize some constants for SMO
		self.__init_As()
		self.__init_Bs()

		# Computationally Intensive SMO
		self.alphas = [0] * self.__size
		self.__sequential_minimal_optimization()

		# Extract support vectors and b
		self.support_vectors, self.support_vector_classes = self.__get_support_vectors()
		self.__b = self.__get_b()

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
		kernel_vector = np.apply_along_axis(self.__kernel.apply, 1, self.support_vectors, x2=x)
		zipped = zip(kernel_vector, self.ys, self.alphas)
		products = [np.prod(tup) for tup in zipped]
		return sum(products)

	def __get_b(self):
		"""
		Calculates B from the constraint equation using an arbitarary support vector
		"""

		## https://github.com/LasseRegin/SVM-w-SMO/blob/master/SVM.py
		## ^^ here it says to take mean of below expression for all support vectors,

		## but in Platt 1998 it says any support vector will do

		# currently sticking with Platt

		# for training
		if (self.support_vectors.shape[0] == 0):
			return 0
		return self.support_vector_classes[0] - self.__sum_w_tranpose_x(self.support_vectors[0, :])

	# need to test this
	def __get_support_vectors(self):
		"""
		Return : (ndarray, ndarray)
			The support vectors for this classifier and their corresponding class membership (+1 or -1)
		"""
		num_supp_vecs = np.nonzero(self.alphas)
		supp_vecs = np.matrix((num_supp_vecs, self.xs.shape[1]))
		supp_vecs_idxs = np.matrix((num_supp_vecs, 1))
		idx2 = 0
		for i in range(self.__size):
			if (self.alphas[i] != 0):
				supp_vecs[idx2, :] = self.xs[i, :]
				supp_vecs_idxs[idx2] = i
				idx2 += 1
		return supp_vecs, supp_vecs_idxs

	def test_conditions(self):
		s = sum(list(map(lambda a, y: a*y, self.alphas, self.ys)))
		print(s)

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


	def __init_As(self):
		"""
		Sets self.__As
		self.__As is a list where __As[i] is the lower bound for alphas[i] * ys[i]
		self.__As[i] is -C if ys[i] == -1
		self.__As[i] is  0 if ys[i] == +1
		"""
		self.__As = []
		for i in range(self.__size):
			if (self.ys[i] == -1):
				self.__As.append(-self.__C)
			else:
				self.__As.append(0)

	def __init_Bs(self):
		"""
		Sets self.__Bs
		self.__Bs is a list where __As[i] is the upper bound for alphas[i] * ys[i]
		self.__Bs[i] is 0 if ys[i] == -1
		self.__Bs[i] is C if ys[i] == +1
		"""
		self.__Bs = []
		for i in range(self.__size):
			if (self.ys[i] == 1):
				self.__Bs.append(self.__C)
			else:
				self.__Bs.append(0)

	def list_nonzero_non_c(self):
		res = []
		for i in range(self.__size):
			if (self.alphas[i] != 0 and self.alphas[i] != self.__C):
				res.append(alphas[i])

		return res

	def takeStep(self, i1, i2, E2):
		if (i1 == i2):
			return 0
		alph1 = self.alphas[i1]
		y1 = self.ys[i1]
		y2 = self.ys[i2]
		E1 = self.predict(self.xs[i1, :]) - y1
		s = y1*y2
		if (s > 0):
			L = max(0, self.alphas[i2] - self.alphas[i1])
			H = min(self.__C, self.__C + self.alphas[i2] - self.alphas[i1])
		else:
			L = max(0, self.alphas[i2] + self.alphas[i1] - self.__C)
			H = min(self.__C, self.alphas[i2] + self.alphas[i1])
		if (L == H):
			return 0
		k11 = self.__kernel.apply(self.xs[i1, :], self.xs[i1, :])
		k12 = self.__kernel.apply(self.xs[i1, :], self.xs[i2, :])
		k22 = self.__kernel.apply(self.xs[i2, :], self.xs[i2, :])
		eta = k11 + k22 - (2*k12)
		if (eta > 0):
			a2 = alph2 + ((y2*(E1 - E2))/eta)
			if (a2 < L):
				a2 = L
			elif (a2 > H):
				a2 = H
		else:
			Lobj = 1# objective function at a2=L
			Hobj = 1# objective function at a2=H
			if (Lobj < Hobj - self.__eps):
				a2 = L
			elif (Lobj < Hobj+self.__eps):
				a2 = H
			else:
				a2 = alph2

		if (abs(a2 - alph2) < self.__eps * (a2 + alph2 + self.__eps)):
			return 0
		a1 = alph1 + s*(alph2 - a2)
		# Update threshold to reflect change in Lagrange multipliers
		# Update weight vector to reflect change in a1 & a2, if SVM is linear
		# Update weight vector to reflect change in a1 & a2, if SVM is linear
		self.alphas[i1] = a1
		self.alphas[i2] = a2
		return 1



	def examineExample(self, i2):
		y2 = self.ys[i2]
		alph2 = self.alphas[i2]
		E2 = self.predict(self.xs[i2, :]) - y2
		r2 = E2*y2
		if ((r < -self.__tol and alph2 < self.__C) or (r2 > self.__tol and alph2 > 0)):
			if (len(self.list_nonzero_non_c()) > 1):
				i1 = 1# reult of second choice heuristic
				if (self.takeStep(i1, i2)):
					return 1
			startIdx1 = random.randint(0, self.__size)
			for i1 in range(startIdx, self.__size):
				if (self.alphas[i1] == 0 or self.alphas[i1] == self.__C):
					continue
				if (takeStep(i1, i2, E2)):
					return 1
			for i1 in range(0, startIdx):
				if (self.alphas[i1] == 0 or self.alphas[i1] == self.__C):
					continue
				if (takeStep(i1, i2, E2)):
					return 1

			startIdx2 = random.randint(0, self.__size)
			for i1 in range(startIdx2, self.__size):
				if (takeStep(i1, i2)):
					return 1
			for i1 in range(0, startIdx2):
				if (takeStep(i1, i2)):
					return 1
		return 0






	def __sequential_minimal_optimization(self):
		numChanged = 0
		examineAll = 1
		while (numChanged > 0 or examineAll):
			if (examineAll):
				for i in range(self.__size):
					numChanged += self.examineExample(i)
			else:
				for i in range(self.__size):
					if (self.alphas[i] != 0 and self.alphas[i] != self.__C):
						numChanged += examineExample(i)
			if (examineAll == 1):
				examineAll = 0
			elif (numChanged == 0):
				examineAll = 1
