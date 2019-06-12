from svm import SVM
from kernels import Kernel
import numpy as np
import multiprocessing as mp
import time

class MP_SVM:
	"""
	MULTIPROCESS SVM
	This class handles the data parallelization of our SVM.
	In both training and testing, we split the training set and create an SVM object to fit/predict on each partition.


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
	num_processes
		- The number of proecesses to use to train the SVM
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
		start : int
			- The starting index of the training set that we are partitioning
		end : int
			- The stopping index of the training set that we are partitioning
		shared_classifiers : multiprocessing.managers.AutoProxy[Queue]
			- Interprocess communication queue that we push our classifier onto when it finishes training

		Actions:
		-------
		Creates an SVM object and trains it on the given partition
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
		x : ndarray
			training examples
		y : ndarray
			training labels

		Actions:
		-------
		Splits the data into num_processes partitions
		Creates a classifier for each partition
		Spawns a new process for each classifier to fit on its partition
		Collects the classifiers using an interprocess communication Queue.
		"""
		self.__xs = x
		self.__ys = y

		# The number of training examples per process
		sub_samples = int(self.__xs.shape[0] / self.__num_processes)

		# Create the interprocess communication method
		manager = mp.Manager()
		shared_classifiers = manager.Queue()

		# Create each process where the target is the fit_one function
		processes = []
		for i in range(self.__num_processes):
			start_idx = i*sub_samples
			end_idx = start_idx + sub_samples
			processes.append(mp.Process(target = self.__fit_one, args=(start_idx, end_idx, shared_classifiers)))

		# Start each process
		for p in processes:
			p.start()

		# Wait for all of them to finish
		for p in processes:
			p.join()


		# collect the classifiers
		self.__classifiers = []
		while (shared_classifiers.empty() == False):
			self.__classifiers.append(shared_classifiers.get())



	def predict_one(self, start, end, predictions):
		"""
		Parameters
		----------
		start : int
			The starting index of the testing set that we are partitioning
		end : int
			The stopping index of the testing set that we are partitioning
		predictions: multiprocessing.managers.AutoProxy[Queue]
			- Interprocess communication queue that we push our predictions onto when it finishes training

		Actions
		-------
		Iterates over each classifer and collects their predictions on the given partition of the testing set
		The class with the maximum number of votes for each sample is the selected class
		"""

		# Initialize votes for each class to 0
		pred_ys = np.zeros((end - start))

		# Since predictions are either 1 or -1, we can sum all the predictions and take the sign of the sum to be the class
		for clf in self.__classifiers:
			pred_ys += clf.predict(self.__test_xs[start:end, :])

		# Take the sign of the votes as the selected label (arbitrarily assign votes summing 0 to postive class)
		for i in range(end - start):
			if (pred_ys[i] >= 0):
				pred_ys[i] = 1
			else:
				pred_ys[i] = -1

		# Add the predictions to the queue, also including start index so we can reorder
		predictions.put((pred_ys, start))


	def predict(self, x):
		"""
		Parameters
		----------
		x : ndarray
			- Traning set

		Actions
		-------
		Split the testing set into num_processes partitions and spawn a process to predict each partition of the data

		Returns:
		--------
		preds : ndarray
			- Predicted labels of xs
		"""
		self.__test_xs = x

		# The number of training examples per process
		sub_samples = int(self.__test_xs.shape[0] / self.__num_processes)

		# Create the interprocess communication method
		mananger = mp.Manager()
		predictions = mananger.Queue()

		# Create each process where the target is the predict_one function
		processes = []
		for i in range(self.__num_processes):
			start_idx = i*sub_samples
			end_idx = start_idx + sub_samples
			processes.append(mp.Process(target = self.predict_one, args=(start_idx, end_idx, predictions)))

		# Start each process
		for p in processes:
			p.start()

		# Wait for them to finish
		for p in processes:
			p.join()

		# Collected the predictions
		preds = []
		while (predictions.empty() == False):
			preds.append(predictions.get())

		# sort the predictions by starting index
		preds = sorted(preds, key=lambda pred: pred[1])

		# concetenate the predictions
		preds = np.concatenate([pred[0] for pred in preds])
		return preds

