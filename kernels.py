import numpy as np
import math



class Kernel:
	# kernel type can be rbf...
	# gamma is the coefficient for rbf kernel
	def __init__(self, kernel_type, gamma=None):
		self.type = kernel_type
		self.gamma = gamma

	def apply(x1, x2):
		if (self.type == "rbf"):
			return self.apply_rbf(x1, x2)

	def apply_rbf(self, x1, x2):
		return math.exp(-self.gamma * pow(np.linalg.norm(x1 - x2), 2))


		
