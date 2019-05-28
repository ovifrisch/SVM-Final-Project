import numpy as np
import math


class Kernel:
	"""
	A class used to peform Kernel computations on pairs of vectors
	"""
	def __init__(self, kernel_type, gamma):
		"""
		Parameters
		----------
		kernel_type : str
			The type of kernel function (support for "rbf", "linear", "poly")
		gamma : float
			Kernel coefficient for RBF kernel
		"""
		self.kernel_type = kernel_type
		self.gamma = gamma

	def apply(self, x1, x2):
		"""
		Computes K(x1, x2) depending on kernel_type
		"""
		if (self.kernel_type == "rbf"):
			return self.__rbf(x1, x2)
		elif (self.kernel_type == "linear"):
			return self.__linear(x1, x2)

	def __linear(self, x1, x2):
		return np.dot(x1, x2)

	def __rbf(self, x1, x2):
		return math.exp((self.gamma * -1) * pow(np.linalg.norm(x1 - x2), 2))
