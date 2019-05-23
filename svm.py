import numpy as np
from kernels import Kernel

class svm:
	def __init__(self, kernel, C):
		self.w = None
		self.alphas = None
		self.xs = None
		self.ys = None
		self.kernel = Kernel(kernel)
		self.b = None
		self.C = C
		self.size = None # the training set size
		self.As = None
		self.Bs = None

	# Accepts X and Y training data and creates the model using SMO
	def fit(self, X, y):
		self.xs = X
		self.ys = y
		self.size = X.shape[0]
		self.init_As()
		self.init_Bs()
		self.alphas = sequential_minimal_optimization()

	# Accepts a list of testing samples and returns the predicted classes
	def predict(self, xs):
		preds = np.zeros((xs.shape[0])) # predictions
		for i in range(xs.shape[0]):
			x = xs[i, :]
			kernel_vector = np.apply_along_axis(self.kernel.apply, 1, self.xs, x2=x)
			zipped = zip(kernel_vector, self.ys, self.alphas)
			products = [np.prod(tup) for tup in zipped]
			summed = sum(products)
			expr = summed + self.b
			preds[i] = np.sign(expr)
		return preds

	def init_As(self):
		self.As = []
		for i in range(self.size):
			if (self.ys[i] == -1):
				self.As[i] = -self.C
			else:
				self.As[i] = 0

	def init_Bs(self):
		self.Bs = []
		for i in range(self.size):
			if (self.ys[i] == 1):
				self.Bs[i] = self.C
			else:
				self.Bs[i] = 0


	def update_gradient(self, i, j, k, gradients, lamb):
		retval = gradients[k] - (lamb * self.ys[k] * self.kernel.apply(self.xs[i, :], self.xs[k, :])) + (lamb * self.ys[k] * self.kernel.apply(self.xs[j, :], self.xs[k, :]))
		return retval

	def get_lambda(self, i, j, gradients, alphas):
		a1 = self.Bs[i] - (self.ys[i] * alphas[i])
		a2 = (self.ys[j] * alphas[j]) - self.As[j]
		a3 = ((self.ys[i] * gradients[i]) - (self.ys[j] * gradients[j])) / (self.kernl.apply(self.xs[i, :], self.xs[i, :]) + self.kernl.apply(self.xs[j, :], self.xs[j, :]) - (2 * self.kernel.apply(self.xs[i, :]), self.kernel.apply(self.xs[j, :])))
		return min(a1, min(a2, a3))

	def get_i(self, gradients, alphas):
		temp = list(map(lambda y, g: y*g, self.ys, gradients))
		max_idx = 0
		max_el = temp[0]
		for i in range(len(temp)):
			if (temp[i] > max_el and self.ys[i]*alphas[i] < self.Bs[i]):
				max_el = temp[i]
				max_idx = i
		return max_idx


	def get_j(self, gradients, alphas):
		temp = list(map(lambda y, g: y*g, self.ys, gradients))
		min_idx = 0
		min_el = temp[0]
		for i in range(len(temp)):
			if (temp[i] < min_el and self.ys[i]*alphas[i] > self.As[i]):
				min_el = temp[i]
				min_idx = i
		return min_idx

	def sequential_minimal_optimization(self):
		alphas = [0] * self.size
		gradients = [1] * self.size
		while (True):
			i = self.get_i(gradients, alphas)
			j = self.get_j(gradients, alphas)

			# optimality criterion (11)
			if (self.ys[i] * gradients[i] <= self.ys[j] * gradients[j]):
				return
			lamb = get_lambda(i, j, gradients, alphas)
			for k in range(self.size):
				gradients[k] = self.update_gradient(i, j, k, gradients, lamb)
				alphas[i] = alphas[i] + (self.ys[i] * lamb)
				alphas[j] = alphas[j] - (self.ys[j] * lamb)
