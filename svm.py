import numpy as np
import random
from kernels import Kernel
import time


class SVM:
	"""
	Binary Support Vector Classification

	Parameters
	----------
	kernel_type : string, optional (default = "rbf")
		- Kernel to be used to transform data
	C : float, optional (default = 1)
		- Coefficient of error term in Soft Margin Obj Function
	gamma : float, optional (default = 1)
		- Paramter in RBF and Sigmoid Kernel Functions
	degree : float, optional (default = 3)
		- Degree of Polynomial in Polynomial Kernel Function
	tolerance : float, optional (default = 1e-4)
		- tolerance for stopping criteria
	epsilon : float, optional (defualt = 1e-4)
		- UPDATE AFTER UNDERSTANDING
	solver : string, optional (default = "smo")
		- Which optimization algorithm to use for the dual form of the Obj
	"""

	def __init__(
	self,
	kernel_type='rbf',
	C=1,
	gamma=1,
	degree=3,
	tolerance=0.1,
	epsilon=0.1,
	solver = "smo"
	):
		self.__kernel = Kernel(kernel_type, gamma, degree)
		self.__C = C
		self.__tol = tolerance
		self.__error_cache = {}
		self.__eps = epsilon
		self.__solver = solver

	def __del__(self):
		pass

	def fit(self, x, y):
		"""
		Parameters
		----------
		x : ndarray - Data
		y : ndarray - Labels

		Returns
		-------
		self : object
		"""
		# Make sure data is clean
		# if (not self.valid_train_params(X, Y)):
		# 	print("Invalid")

		self.__xs = x
		self.__ys = y
		self.__size = x.shape[0]
		self.__kernel_mat = self.__get_kernel_matrix()
		# Get the alphas and b from smo

		self.__alphas, self.__b = self.__solve(self.__solver)
		supp_idxs = np.nonzero(self.__alphas)[0]
		self.__supp_x = np.take(self.__xs, supp_idxs, axis=0)
		self.__supp_y = np.take(self.__ys, supp_idxs)
		self.__supp_a = np.take(self.__alphas, supp_idxs)

	def __get_kernel_matrix(self):
		"""
		Precompute the Kernel Matrix
		"""
		self.__KM = np.zeros((self.__size, self.__size))
		for i in range(self.__size):
			for j in range(self.__size):
				self.__KM[i][j] = self.__kernel.eval(self.__xs[i], self.__xs[j])

	def __initialize_error_cache(self):
		for i in range(self.__size):
			self.__error_cache[i] = -self.__ys[i]


	def __solve(self, solver):
		if (solver == "smo"):
			return self.__smo()


	def __takeStep(self, i1, i2, E2):
		if (i1 == i2 or i1 == -1):
			return 0
		alph1 = self.__alphas[i1]
		alph2 = self.__alphas[i2]
		y1 = self.__ys[i1]
		y2 = self.__ys[i2]
		E1 = self.__error_cache[i1]
		s = y1*y2

		# y1 == y2
		if (s > 0):
			L = max(0, self.__alphas[i2] - self.__alphas[i1] - self.__C)
			H = min(self.__C, self.__alphas[i2] + self.__alphas[i1])

		# y1 != y2
		else:
			L = max(0, self.__alphas[i2] - self.__alphas[i1])
			H = min(self.__C, self.__C + self.__alphas[i2] - self.__alphas[i1])
		if (L == H):
			return 0
		k11 = self.__kernel.eval(self.__xs[i1, :], self.__xs[i1, :])
		k12 = self.__kernel.eval(self.__xs[i1, :], self.__xs[i2, :])
		k22 = self.__kernel.eval(self.__xs[i2, :], self.__xs[i2, :])
		eta = k11 + k22 - (2*k12)
		if (eta > 0):
			# optimum value for a2
			a2 = alph2 + ((y2*(E1 - E2))/eta)
			if (a2 < L):
				a2 = L
			elif (a2 > H):
				a2 = H

		# eta is equal to 0,so we can't get the optimum for a2
		# settle for setting it to the max of the endpoints of the constraint line
		else:
			f1 = (y1*(E1 + self.__b)) - (self.__alphas[i1]*k11) - (s*self.__alphas[i2]*k12)
			f2 = (y2*(E2 + self.__b)) - (s*self.__alphas[i1]*k12) - (self.__alphas[i2]*k22)
			L1 = self.__alphas[i1] + (s*(self.__alphas[i2] - L))
			H1 = self.__alphas[i1] + (s*(self.__alphas[i2] - H))
			Lobj = L1*f1 + L*f2 + 0.5*pow(L1, 2)*k11 + 0.5*pow(L, 2)*k22 + s*L*L1*k12
			Hobj = H1*f1 + H*f2 + 0.5*pow(H1, 2)*k11 + 0.5*pow(H, 2)*k22 + s*H*H1*k12

			if (Lobj < Hobj - self.__eps):
				a2 = L
			elif (Lobj < Hobj+self.__eps):
				a2 = H
			else:
				a2 = alph2

		if (abs(a2 - alph2) < self.__eps * (a2 + alph2 + self.__eps)):
			return 0
		a1 = alph1 + s*(alph2 - a2)
		if (a1 < 0):
			a2 += s * a1
			a1 = 0
		elif (a1 > self.__C):
			a2 += s * (a1 - self.__C)
			a1 = self.__C

		# Update threshold to reflect change in Lagrange multipliers
		b1 = self.__b + E1 + y1*(a1 - alph1)*k11 + y2*(a2 - alph2)*k12
		b2 = self.__b + E2 + y1*(a1 - alph1)*k12 + y2*(a2 - alph2)*k22
		b_new = (b1 + b2) / 2
		delta_b = b_new = self.__b
		self.__b = b_new

		# Update error cache using new Lagrange multipliers
		t1 = y1 * (a1 - alph1)
		t2 = y2 * (a2 - alph2)

		for i in range(self.__size):
			if (self.__alphas[i] > 0 and self.__alphas[i] < self.__C):
				self.__error_cache[i] += t1 * self.__kernel.eval(self.__xs[i1, :], self.__xs[i, :]) + t2 * self.__kernel.eval(self.__xs[i2, :], self.__xs[i, :]) - delta_b

		self.__error_cache[i1] = 0
		self.__error_cache[i2] = 0

		# store new alphas
		self.__alphas[i1] = a1
		self.__alphas[i2] = a2
		return 1

	def __second_choice_heuristic(self, E2):
		tmax = 0
		i1 = -1
		for k in range(self.__size):
			if (self.__alphas[k] > 0 and self.__alphas[k] < self.__C):
				E1 = self.__error_cache[k]
				temp = abs(E2 - E1)
				if (temp > tmax):
					tmax = temp
					i1 = k
		return i1

	def __examine_example(self, i2):
		"""
		Paramters
		---------
		i2 : int
			- Index of current training example


		Returns
		------
		int (0 or 1)
		 - Whether or not alphas[i2] was changed
		"""

		y2 = self.__ys[i2]
		alph2 = self.__alphas[i2]
		E2 = np.sign(self.__u(idx=i2)) - y2
		r2 = E2*y2
		if ((r2 < -self.__tol and alph2 < self.__C) or (r2 > self.__tol and alph2 > 0)):
			i1 = self.__second_choice_heuristic(E2)
			if (self.__takeStep(i1, i2, E2)):
				return 1
			idx = random.randint(0, self.__size)
			for i1 in list(range(idx, self.__size)) + list(range(0, idx)):
				if (self.__alphas[i1] > 0 and self.__alphas[i1] < self.__C):
					if (self.__takeStep(i1, i2, E2)):
						return 1

			for i1 in list(range(idx, self.__size)) + list(range(0, idx)):
				if (self.__takeStep(i1, i2, E2)):
					return 1
		

		return 0

	def __smo(self):
		"""
		Sequential Minimal Optimization of Dual Form of Obj Function

		Returns
		-------
		self.__alphas : ndarray
			- Optimal Lagrange Multipliers in Obj Function
		self.__b : float
			- straight line distance to optimal hyperplane
		"""
		self.__alphas = np.zeros(self.__size)
		self.__b = 0

		# num alphas changed in 1 pass of training set
		num_changed = 0

		# variable used to alternate between 1 pass over all
		# training examples and 1 pass over examples whose
		# Lagrange multipliers are not at their bounds
		examine_all = 1

		# Store errors to speed up algorithm
		self.__initialize_error_cache()
		while (num_changed > 0 or examine_all):
			num_changed = 0
			if (examine_all):
				# loop over entire training set
				for i in range(self.__size):
					num_changed += self.__examine_example(i)
			else:
				for i in range(self.__size):

					# loop over non boundary training examples
					if (self.__alphas[i] != 0 and self.__alphas[i] != self.__C):
						num_changed += self.__examine_example(i)

			if (examine_all == 1):
				examine_all = 0

			elif (num_changed == 0):
				examine_all = 1


		return self.__alphas, self.__b


	def __u(self, x=None, idx=None):
		"""
		Parameters
		----------
		x : ndarray, optional (default = None)
			One data vector
		idx : int, optional (default = None)
			If not None, x = self.xs[idx]

		Returns
		-------
		u : float
			-The evaluation of the decision function at point x.

		Note: When this function is called during training,
		we use the precomputed kernel vector. When called
		during testing, we compute the kernel vector.
		"""
		if (idx != None):
			kernel_vector = self.__KM[idx, :]
			return np.sum(kernel_vector * self.__ys *self.__alphas) - self.__b
		else:
			kernel_vector = np.apply_along_axis(self.__kernel.eval, 1, self.__supp_x, x2=x)
			mult = kernel_vector * self.__supp_y, * self.__supp_a
			return np.sum(mult[0]) - self.__b


	def predict(self, xs):

		def sign(x):
			return np.sign(self.__u(x = x))

		return np.apply_along_axis(sign, 1, xs)


	def predict_accuracy(self, xs, ys):
		preds = self.predict(xs)
		accuracy = np.sum(ys==preds) / xs.shape[0]
		return accuracy
