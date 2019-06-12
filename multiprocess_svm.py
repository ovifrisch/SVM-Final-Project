from svm import SVM
from kernels import Kernel
import numpy as np
import multiprocessing as mp
import time

class MP_SVM:
	"""
	SVM Classifier that splits the training/testing
	work among several processes
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
	solver = "smo",
	num_processes = 5
	):
		self.__kernel_type = kernel_type
		self.__gamma = gamma
		self.__degree = degree
		self.__C = C
		self.__tol = tolerance
		self.__error_cache = {}
		self.__eps = epsilon
		self.__max_iter = max_iter
		self.__solver = solver
		self.__num_processes = num_processes

	def __fit_one(self, start, end, shared_classifiers):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		clf = SVM(self.__kernel_type, self.__C, self.__gamma, self.__degree, self.__tol, self.__eps, self.__max_iter, self.__solver)
		xs = self.__xs[start:end, :]
		ys = self.__ys[start:end]
		clf.fit(xs, ys)
		
		shared_classifiers.put(clf)

	def fit(self, x, y):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		self.__xs = x
		self.__ys = y


		sub_samples = int(self.__xs.shape[0] / self.__num_processes)
		manager = mp.Manager()
		shared_classifiers = manager.Queue()
		processes = []

		for i in range(self.__num_processes):
			start_idx = i*sub_samples
			end_idx = start_idx + sub_samples
			processes.append(mp.Process(target = self.__fit_one, args=(start_idx, end_idx, shared_classifiers)))

		for p in processes:
			p.start()

		for p in processes:
			p.join()


		# save the classifiers
		self.__classifiers = []
		while (shared_classifiers.empty() == False):
			self.__classifiers.append(shared_classifiers.get())



	def predict_one(self, start, end, predictions):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		pred_ys = np.zeros((end - start))
		for clf in self.__classifiers:
			pred_ys += clf.predict(self.test_xs[start:end, :])

		for i in range(end - start):
			if (pred_ys[i] >= 0):
				pred_ys[i] = 1
			else:
				pred_ys[i] = -1

		predictions.put((pred_ys, start, end))


	def predict(self, x):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		self.test_xs = x
		sub_samples = int(self.test_xs.shape[0] / self.__num_processes)
		mananger = mp.Manager()
		predictions = mananger.Queue()
		processes = []
		for i in range(self.__num_processes):
			start_idx = i*sub_samples
			end_idx = start_idx + sub_samples
			processes.append(mp.Process(target = self.predict_one, args=(start_idx, end_idx, predictions)))

		for p in processes:
			p.start()

		for p in processes:
			p.join()

		preds = []
		while (predictions.empty() == False):
			preds.append(predictions.get())

		# sort by starting index
		preds = sorted(preds, key=lambda pred: pred[1])
		return np.concatenate([pred[0] for pred in preds])

	def predict_accuracy(self, x, y):
		"""
		Parameters
		----------


		Returns
		-------
		"""
		preds = self.predict(x)
		return np.sum(preds == y) / x.shape[0]
