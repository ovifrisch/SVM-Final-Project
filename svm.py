import numpy as np
from kernels import Kernel
import random



class svm:

	def __init__(self, kernel_type='rbf', C=1, gamma=1, degree=3, tolerance=0.1, epsilon=0.1):
		self.__kernel = Kernel(kernel_type, gamma, degree)
		self.__C = C
		self.__tol = 0.001
		self.__error_cache = {}
		self.__eps = 0.001

	# Accepts X and Y training data and creates the model using SMO
	def fit(self, X, y):
		self.num_classes = np.unique(y).shape[0]
		self.xs = X
		self.__size = X.shape[0]
		self.alphas = [0] * self.__size
		self.__b = 0

		# multiclass
		if (self.num_classes > 2):
			self.multiclass_fit(X, y)
			return
		self.ys = y

		# Computationally Intensive SMO
		self.__sequential_minimal_optimization()


	# train one v all
	# assuming the classes start from 0, 1, 2, ...
	def multiclass_fit(self, X, y):
		self.new_ys = []
		# generate ys for each classifier
		for i in range(self.num_classes):
			new_y = np.copy(y)
			for j in range(y.shape[0]):
				if (new_y[j] == i):
					new_y[j] = 1
				else:
					new_y[j] = -1
			self.new_ys.append(new_y)

		# train each classifier
		self.multiclass_models = []
		for i in range(self.num_classes):
			self.ys = self.new_ys[i]
			self.__sequential_minimal_optimization()
			self.multiclass_models.append((self.alphas, self.__b))

			# reset model paramters for next classifier
			self.alphas = [0] * self.__size
			self.__b = 0

	def predict_multiclass(self, xs):
		preds = np.zeros((xs.shape[0]))

		for i in range(xs.shape[0]):
			x = xs[i]
			votes = [0] * self.num_classes
			for idx, (alpha, b) in enumerate(self.multiclass_models):
				self.alphas = alpha
				self.__b = b
				self.ys = self.new_ys[idx]
				decision = np.sign(self.__sum_w_tranpose_x(x) - self.__b)
				if (decision == 1):
					# add 1 vote to this class
					votes[idx] += 1
				else:
					# add 1 vote to all other classes
					votes = [x+1 for x in votes]
					votes[idx] -= 1
			preds[i] = np.argmax(votes)
		return preds


	# Accepts a list of testing samples and returns the predicted classes
	# Modeled after discriminant function in (6)
	def predict(self, xs):

		if (self.num_classes > 2):
			return self.predict_multiclass(xs)
		preds = np.zeros((xs.shape[0])) # predictions
		for i in range(xs.shape[0]):
			x = xs[i]
			preds[i] = np.sign(self.__sum_w_tranpose_x(x) - self.__b)
		return preds

	def predict_training(self, x):
		return np.sign(self.__sum_w_tranpose_x(x) - self.__b)

	def __sum_w_tranpose_x(self, x):
		kernel_vector = np.apply_along_axis(self.__kernel.apply, 1, self.xs, x2=x)
		zipped = zip(kernel_vector, self.ys, self.alphas)
		products = [np.prod(tup) for tup in zipped]
		return sum(products)

	def __initialize_error_cache(self):
		for i in range(self.__size):
			self.__error_cache[i] = -self.ys[i]

	def takeStep(self, i1, i2, E2):
		if (i1 == i2 or i1 == -1):
			return 0
		alph1 = self.alphas[i1]
		alph2 = self.alphas[i2]
		y1 = self.ys[i1]
		y2 = self.ys[i2]
		E1 = self.__error_cache[i1]
		s = y1*y2

		# y1 == y2
		if (s > 0):
			L = max(0, self.alphas[i2] - self.alphas[i1] - self.__C)
			H = min(self.__C, self.alphas[i2] + self.alphas[i1])

		# y1 != y2
		else:
			L = max(0, self.alphas[i2] - self.alphas[i1])
			H = min(self.__C, self.__C + self.alphas[i2] - self.alphas[i1])
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
			f1 = (y1*(E1 + self.__b)) - (self.alphas[i1]*k11) - (s*self.alphas[i2]*k12)
			f2 = (y2*(E2 + self.__b)) - (s*self.alphas[i1]*k12) - (self.alphas[i2]*k22)
			L1 = self.alphas[i1] + (s*(self.alphas[i2] - L))
			H1 = self.alphas[i1] + (s*(self.alphas[i2] - H))
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


		# Update weight vector to reflect change in a1 & a2, if SVM is linear

		# Update error cache using new Lagrange multipliers
		t1 = y1 * (a1 - alph1)
		t2 = y2 * (a2 - alph2)

		for i in range(self.__size):
			if (self.alphas[i] > 0 and self.alphas[i] < self.__C):
				self.__error_cache[i] += t1 * self.__kernel.apply(self.xs[i1, :], self.xs[i, :]) + t2 * self.__kernel.apply(self.xs[i2, :], self.xs[i, :]) - delta_b

		self.__error_cache[i1] = 0
		self.__error_cache[i2] = 0

		# store new alphas
		self.alphas[i1] = a1
		self.alphas[i2] = a2
		return 1

	def __second_choice_heuristic(self, E2):
		tmax = 0
		i1 = -1
		for k in range(self.__size):
			if (self.alphas[k] > 0 and self.alphas[k] < self.__C):
				E1 = self.__error_cache[k]
				temp = abs(E2 - E1)
				if (temp > tmax):
					tmax = temp
					i1 = k
		return i1


	def examineExample(self, i2):
		y2 = self.ys[i2]
		alph2 = self.alphas[i2]
		E2 = self.predict_training(self.xs[i2, :]) - y2
		r2 = E2*y2
		if ((r2 < -self.__tol and alph2 < self.__C) or (r2 > self.__tol and alph2 > 0)):
			i1 = self.__second_choice_heuristic(E2)
			if (self.takeStep(i1, i2, E2)):
				return 1
			startIdx1 = random.randint(0, self.__size)
			for i1 in range(startIdx1, self.__size):
				if (self.alphas[i1] > 0 and self.alphas[i1] < self.__C):
					if (self.takeStep(i1, i2, E2)):
						return 1
			for i1 in range(0, startIdx1):
				if (self.alphas[i1] > 0 and self.alphas[i1] < self.__C):
					if (self.takeStep(i1, i2, E2)):
						return 1

			startIdx2 = random.randint(0, self.__size)
			for i1 in range(startIdx2, self.__size):
				if (self.takeStep(i1, i2, E2)):
					return 1
			for i1 in range(0, startIdx2):
				if (self.takeStep(i1, i2, E2)):
					return 1
		return 0


	def __sequential_minimal_optimization(self):
		numChanged = 0
		examineAll = 1
		self.__initialize_error_cache()
		while (numChanged > 0 or examineAll):
			numChanged = 0
			if (examineAll):
				for i in range(self.__size):
					numChanged += self.examineExample(i)
			else:
				for i in range(self.__size):
					if (self.alphas[i] != 0 and self.alphas[i] != self.__C):
						numChanged += self.examineExample(i)
			if (examineAll == 1):
				examineAll = 0
			elif (numChanged == 0):
				examineAll = 1
