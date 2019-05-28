import numpy as np
from kernels import Kernel



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
		self.alphas = self.__sequential_minimal_optimization()

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


	def __update_gradient(self, i, j, k, gradients, lamb):
		"""
		Implements Line 8 of SMO
		"""
		return gradients[k] - (lamb * self.ys[k] * self.__kernel.apply(self.xs[i, :], self.xs[k, :])) + (lamb * self.ys[k] * self.__kernel.apply(self.xs[j, :], self.xs[k, :]))

	def __get_lambda(self, i, j, gradients, alphas):
		"""
		Implements Line 7 of SMO
		"""
		arg1 = self.__Bs[i] - (self.ys[i] * alphas[i])
		arg2 = (self.ys[j] * alphas[j]) - self.__As[j]
		yigi = self.ys[i] * gradients[i]
		yjgj = self.ys[j] * gradients[j]
		Kii = self.__kernel.apply(self.xs[i, :], self.xs[i, :])
		Kjj = self.__kernel.apply(self.xs[j, :], self.xs[j, :])
		Kij = self.__kernel.apply(self.xs[i, :], self.xs[j, :])
		arg3 = (yigi - yjgj) / (Kii + Kjj - (2*Kij))
		return min(arg1, min(arg2, arg3))

	def __get_i(self, gradients, alphas):
		"""
		Implements Line 4 of SMO
		"""
		temp = list(map(lambda y, g: y*g, self.ys, gradients))
		max_idx = 0
		max_el = temp[0]
		for i in range(len(temp)):
			if (temp[i] > max_el and self.ys[i]*alphas[i] < self.__Bs[i]):
				max_el = temp[i]
				max_idx = i
		return max_idx


	def __get_j(self, gradients, alphas):
		"""
		Implements Line 5 of SMO
		"""
		temp = list(map(lambda y, g: y*g, self.ys, gradients))
		min_idx = 0
		min_el = temp[0]
		for j in range(len(temp)):
			if (temp[j] < min_el and self.ys[j]*alphas[j] > self.__As[j]):
				min_el = temp[j]
				min_idx = j
		return min_idx

	def __optimality_criterion(self, i, j, gradients):
		"""
		Implements Line 6 of SMO
		"""

		# This is the optimality criterion (11)
		print(self.ys[i] * gradients[i])
		print("must be less than")
		print(self.ys[j] * gradients[j])
		return (self.ys[i] * gradients[i] <= self.ys[j] * gradients[j])

	def __sequential_minimal_optimization(self):
		"""
		This function implements the SMO algorithm from Bottou Lin 2006 (BTL) section 6.3
		It maximizes the objective function and finds the Lagrange Multipliers (denoted alphas here)
		"""
		alphas = [0] * self.__size # Line 1
		gradients = [1] * self.__size # Line 2
		while (True):
			i = self.__get_i(gradients, alphas)
			j = self.__get_j(gradients, alphas)
			if (self.__optimality_criterion(i, j, gradients)):
				return alphas
			lambda_ = self.__get_lambda(i, j, gradients, alphas)
			for k in range(self.__size):
				gradients[k] = self.__update_gradient(i, j, k, gradients, lambda_)
				alphas[i] = alphas[i] + (self.ys[i] * lambda_) # Line 9
				alphas[j] = alphas[j] - (self.ys[j] * lambda_) # Line 9
