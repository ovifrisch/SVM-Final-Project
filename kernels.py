import numpy as np
import math


class Kernel:
	"""
	A class used to peform Kernel computations on pairs of vectors
	"""
	def __init__(self, kernel_type, gamma, degree=3):
		"""
		Parameters
		----------
		kernel_type : str
			The type of kernel function (support for "rbf", "linear", "poly", "sgm")
		gamma : float
			Kernel coefficient for RBF kernel
		degree: float
			Exponent for ploynomial kernel
		"""
		self.__kernel_type = kernel_type
		self.__gamma = gamma
		self.__degree = degree

	def eval(self, x1, x2):
		"""
		Computes K(x1, x2) depending on kernel_type
		"""
		if (self.__kernel_type == "rbf"):
			return self.__rbf(x1, x2)
		elif (self.__kernel_type == "linear"):
			return self.__linear(x1, x2)
		elif (self.__kernel_type == "sigmoid"):
			return self.__sigmoid(x1, x2)
		elif (self.__kernel_type == "poly"):
			return self.__poly(x1, x2)

	def __linear(self, x1, x2):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		return np.dot(x1, x2)

	def __rbf(self, x1, x2):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		return math.exp((self.__gamma * -1) * pow(np.linalg.norm(x1 - x2), 2))

	# this might be wrong
	def __sigmoid(self, x1, x2):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		return np.tanh(self.__gamma * np.dot(x1, x2)[0,0])

	def __poly(self, x1, x2):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		return pow(np.dot(x1, x2)[0,0] + 1, self.__degree)
