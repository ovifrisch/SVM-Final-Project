import numpy as np
import multiprocessing as mp
from multiprocess_svm import MP_SVM
from sklearn.utils import shuffle
import time

class SVC:
	"""
	Multiclass SVM
	"""
	def __init__(
	self,
	kernel_type='rbf',
	C=1,
	gamma=3,
	degree=3,
	tolerance=0.01,
	epsilon=0.01,
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

	def get_y_matrix(self):
		"""
		Returns
		-------
		ymat - ndarray
			Label vectors for each classifier
		"""
		y_mat = np.zeros((self.__size, self.__num_classes))
		for i in range(self.__num_classes):
			this_y = np.copy(self.__ys)
			for j in range(self.__size):
				if (self.__ys[j] == i):
					this_y[j] = 1
				else:
					this_y[j] = -1
			y_mat[:, i] = this_y
		return y_mat

	

	def __unmap_labels(self, preds):
		"""
		Undo __map_labels
		"""

		"""
		Parameters
		----------


		Returns
		-------
		"""

		for i in range(preds.shape[0]):
			preds[i] = self.__class_map[preds[i]]
		return preds

	def __map_labels(self):
		"""
		If there are 2 classes, map one class to +1 and the other to -1
		If there are >2 classes, map them to 0, 1, 2, 3, ...
		Save the mappings so that you can convert back after testing
		"""

		"""
		Parameters
		----------


		Returns
		-------
		"""
		self.__class_map = {}
		labels = np.unique(self.__ys)
		if (self.__num_classes == 2):
			for i in range(self.__size):
				if (self.__ys[i] == labels[0]):
					self.__class_map[1] = labels[0]
					self.__ys[i] = 1
				else:
					self.__class_map[-1] = labels[1]
					self.__ys[i] = -1

		else:
			for i in range(self.__size):
				new_label = np.where(labels == self.__ys[i])[0][0]
				self.__class_map[new_label] = self.__ys[i]
				self.__ys[i] = new_label

	def __train_one(self, i, shared_classifiers):
		"""
		Parameters:
		-----------
		i : int
			- Index for the y_mat matrix

		This function trains the ith classifier and
		appends the instance of the classifier to
		the multiprocessing queue
		"""
		y = self.__y_mat[:, i]
		clf = MP_SVM(self.__kernel_type, self.__C, self.__gamma, self.__degree, self.__tol, self.__eps, self.__max_iter, self.__solver, self.__num_processes)
		clf.fit(self.__xs, y)
		shared_classifiers.put((clf, i))

	def fit(self, x, y):
		"""
		Parameters:
		----------
		x : ndarray
			-training examples
		y : ndarray
			-training classes

		Returns:
		self : obj

		Assumptions:
		If there are only two classes, we assume they are labeled -1 and 1
		If there are more than two classes, we assume they are labeled 0, 1, 2, etc...
		Also assuming max 5 classes (because one process per sub_classifier, and mp can only handle 5?)
		"""

		self.__size = x.shape[0]
		self.__num_classes = np.unique(y).shape[0]
		self.__xs = np.copy(x)
		self.__ys = np.copy(y)
		self.__map_labels()
		self.__ovr_classifiers = [] # "One v. Rest" Classifiers

		# Only 2 classes, so classify the usual way
		if (self.__num_classes <= 2):
			clf = MP_SVM(self.__kernel_type, self.__C, self.__gamma, self.__degree, self.__tol, self.__eps, self.__max_iter, self.__solver, self.__num_processes)
			clf.fit(self.__xs, self.__ys)
			self.__ovr_classifiers = [clf]
			return

		# More than 2 classes. First need to get the y labels for each classifier
		self.__y_mat = self.get_y_matrix()


		# set up interprocess communication
		manager = mp.Manager()
		shared_classifiers = manager.Queue()
		currently_executing_processes = []

		# Create 1 process for each ovr classification task
		# We are using the loop index as class labels (WARNING: Class labels may not be properly formatted (ie 0, 1, 2, ..))
		for i in range(min(4, self.__num_classes)):
			currently_executing_processes.append(mp.Process(target=self.__train_one, args=(i,shared_classifiers)))


		# you need to start the max number of processes, move these processes from

		# Start each process
		for p in currently_executing_processes:
			p.start()


		while (len(currently_executing_processes) > 0):
			# wait for the oldest process, remove it from the currently_executing_processes
			currently_executing_processes.pop(0).join()
			if (i + 1 < self.__num_classes):
				i += 1 # increment i for next label
				p = mp.Process(target=self.__train_one, args=(i, shared_classifiers))
				currently_executing_processes.append(p) # add to executing processes because of next line
				p.start()

		# Collect the classifiers from the queue
		while (shared_classifiers.empty() == False):
			self.__ovr_classifiers.append(shared_classifiers.get())

	def predict_one(self, clf, label, x, shared_predictions):
		shared_predictions.put((clf.predict(x), label))


	# REMEMBER TO CREATE QUEUE OF PROCESSES!! DONT SPAWNW MORE THAN COMP CAN HANDLE
	def predict(self, x):
		"""
		Parameters:
		-----------
		x : ndarray
			- input matrix

		Returns:
		--------
		preds : ndarray
			- predictions for each input vector
		"""

		# Create one process for each classifier prediction
		# Each process will get the prediction vector and store it in the queue along with the label it is predicting for
		# Main process will collect each of these and determine which class got the most votes for each of the samples

		if (self.__num_classes == 2):
			return self.__unmap_labels(self.__ovr_classifiers[0].predict(x))

		manager = mp.Manager()
		shared_predictions = manager.Queue()
		processes = []

		# Create one process for each predition task
		for i in range(self.__num_classes):
			processes.append(mp.Process(target=self.predict_one, args=(self.__ovr_classifiers[i][0], self.__ovr_classifiers[i][1], x, shared_predictions)))

		# Start each process
		for p in processes:
			p.start()

		# Wait for them to finish
		for p in processes:
			p.join()

		class_predictions = np.zeros((x.shape[0], self.__num_classes))
		while (shared_predictions.empty() == False):
			tup = shared_predictions.get()
			preds = tup[0]
			label = tup[1]
			for i in range(preds.shape[0]):
				if (preds[i] == 1):
					class_predictions[i, label] += 1
				else:
					class_predictions[i, :] += 1
					class_predictions[i, label] -= 1
		return self.__unmap_labels(np.argmax(class_predictions, 1))

