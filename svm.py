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
	max_iter : int
		- The maximum number of iterations of SMO.
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
	max_iter = 100,
	solver = "smo"
	):
		self.__kernel = Kernel(kernel_type, gamma, degree)
		self.__C = C
		self.__tol = tolerance
		self.__error_cache = {}
		self.__eps = epsilon
		self.__max_iter = max_iter
		self.__solver = solver

	def __del__(self):
		pass


	def fit(self, x, y):
		"""
		Parameters
		----------
		x : ndarray
			- Data
		y : ndarray
			- Labels

		Actions
		-------
		Creates the SVM model over x and y
			- Calls SMO to solve for alphas and b
			- Sets the support vectors

		"""

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

		Actions:
		-------
		Compute and store the kernel matrix KM over the input data

		"""
		self.__KM = np.zeros((self.__size, self.__size))
		for i in range(self.__size):
			for j in range(self.__size):
				self.__KM[i, j] = self.__kernel.eval(self.__xs[i], self.__xs[j])

	def __initialize_error_cache(self):
		"""

		Actions
		-------
		Initialize the error over each training example
		Since the model output is initially 0 for all xi,
		the Error is 0 - yi = -yi
		"""
		for i in range(self.__size):
			self.__error_cache[i] = -self.__ys[i]


	def __solve(self, solver):
		"""
		Parameters
		----------
		solver : string
			- The quadratic optimization algorithm to use

		Actions:
		--------
		Calls the appropriate optimizer function
		"""
		if (solver == "smo"):
			return self.__smo()

	def __get_bounds(self, s, i1, i2):
		"""
		Parameters
		----------
		s : int
			- 1 if x[i1] and x[i2] have same label, -1 otherwise
		i1 : int
			- Index of second chosen Lagrange multiplier
		i2 : int
			- Index of first chosen Lagrange multiplier

		Returns:
		--------
		The minimum and maximum values that the new alpha2 can take
		"""
		

		return L, H

	def __optimize_and_clip(self, a2_old, y2, E1, E2, eta, L, H):
		"""
		Parameters
		----------
		a2_old : float
			- current values of alphas[i2]
		y2 : int
			- label of x[i2]
		E1 : float
			- Error on x[i1]
		E2 : float
			- Error on x[i2]
		eta : floats
			- Second derivative of dual objective
		L : float
			- Lower bound for new alpha2
		H : float
			- Upper bound for new alpha2

		Returns:
		--------
		a2 : float
			- The new optimal, clipped lagrangian multiplier
		"""
		a2 = a2_old + ((y2*(E1 - E2))/eta)
		if (a2 < L):
			a2 = L
		elif (a2 > H):
			a2 = H
		return a2


	def __take_step(self, i1, i2, E2):
		"""
		Parameters
		----------
		i1 : int
			- Index of second chosen Lagrange multiplier
		i2 : int
			- Index of second first Lagrange multiplier
		E2 : float
			- Error on the first chosen Lagrange multiplier

		Actions
		-------
		Update alphas by optimizing 2
		Find optimal value for 1 alpha, clip so constraints are not violated, then solve for 

		Returns
		-------
		"""

		# Same alphas or invalid alpha
		if (i1 == i2 or i1 == -1):
			return 0

		# old alphas
		alph1 = self.__alphas[i1]
		alph2 = self.__alphas[i2]

		# ys
		y1 = self.__ys[i1]
		y2 = self.__ys[i2]

		# Error on alph1
		E1 = self.__error_cache[i1]
		s = y1*y2

		# Compute Lower and Upper bounds
		if (s > 0):
			L = max(0, self.__alphas[i2] - self.__alphas[i1] - self.__C)
			H = min(self.__C, self.__alphas[i2] + self.__alphas[i1])

		else:
			L = max(0, self.__alphas[i2] - self.__alphas[i1])
			H = min(self.__C, self.__C + self.__alphas[i2] - self.__alphas[i1])

		# No feasible alphas here
		if (L == H):
			return 0

		# eta = k11 + k22 - k12
		k11 = self.__KM[i1, i1]
		k22 = self.__KM[i2, i2]
		k12 = self.__KM[i1, i2]
		eta = k11 + k22 - (2*k12)

		# Optimum is valid. Set new a2 to optimal value and then clip
		if (eta > 0):
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


		# Not a significant change in a2 so dont change it and return 0
		if (abs(a2 - alph2) < self.__eps * (a2 + alph2 + self.__eps)):
			return 0

		# Compute a1 from box linear constraint
		a1 = alph1 + s*(alph2 - a2)

		# Make sure a1 is in feasible region
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
				self.__error_cache[i] += t1 * self.__KM[i1, i] + t2 * self.__KM[i2, i] - delta_b

		self.__error_cache[i1] = 0
		self.__error_cache[i2] = 0

		# store new alphas
		self.__alphas[i1] = a1
		self.__alphas[i2] = a2
		return 1

	def __second_choice_heuristic(self, E2):
		"""
		Parameters
		----------
		E2 : int
			- Error on first selected alpha value

		Returns
		-------
		i1 : int
			- The index of the training example that maximizes the absolute value of errors E1 and E2
		"""
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

		Actions:
		-------
		Checks if alphas[i2] violates the KKT conditions
		If it does, use the heuristics to select the second alpha value
		Then optimize them jointly by calling takeStep


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
			if (self.__take_step(i1, i2, E2)):
				return 1
			idx = random.randint(0, self.__size)
			for i1 in list(range(idx, self.__size)) + list(range(0, idx)):
				if (self.__alphas[i1] > 0 and self.__alphas[i1] < self.__C):
					if (self.__take_step(i1, i2, E2)):
						return 1

			for i1 in list(range(idx, self.__size)) + list(range(0, idx)):
				if (self.__take_step(i1, i2, E2)):
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
		steps = 0
		while (num_changed > 0 or examine_all):
			steps += 1
			if (steps > self.__max_iter):
				break
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
			if (self.__supp_x.shape[0] == 0):
				return -self.__b
			kernel_vector = np.apply_along_axis(self.__kernel.eval, 1, self.__supp_x, x2=x)
			mult = kernel_vector * self.__supp_y, * self.__supp_a
			return np.sum(mult[0]) - self.__b


	def predict(self, xs):
		"""
		Parameters
		----------
		xs : ndarray
			- Input samples to predict the labels of

		Returns
		-------
		ndarray
			- Predicted labels of xs
		"""
		def sign(x):
			return np.sign(self.__u(x = x))

		return np.apply_along_axis(sign, 1, xs)
