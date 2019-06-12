import numpy as np
import multiprocessing as mp
from multiprocess_svm import MP_SVM
from sklearn.utils import shuffle
import time

class SVC:
	"""
	MULTICLASS SVM

	This class handles the task of performing multiclass classification
	in which the number of classes is greater than 2. It does so by using
	the One vs. Rest method. This method creates one classifier to classify
	each class in the class set against all the other classes. The results
	of each classifier are aggregated by a voting scheme in which a classifier
	either casts a vote for its class or 1 vote for all the other classes,
	depending on the classification made. The class with the most votes is the
	label for the sample.

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
	num_processes : int
		- The maximum number of processes to use
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
		Actions:
		--------
		Need to transform the multiclass label vecotr into N binary label vectors, where N is the number of classes
		For class i, the each label will be 1 if the class is 1, 0 otherwise

		Returns
		-------
		ymat - ndarray
			Label vectors for each classifier
		"""
		N = self.__num_classes
		y_mat = np.zeros((self.__size, N))
		for i in range(N):
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
		preds : ndarray
			- The (mapped) label predictions for each sample in testing set

		Returns
		-------
		preds: ndarray
			- The predictions for each sample with their original label predictions 
		"""

		for i in range(preds.shape[0]):
			preds[i] = self.__class_map[preds[i]]
		return preds

	def __map_labels(self):
		"""
		Actions
		-------
		We want to get the labels in the right format.
		The right format is:
			If binary: +1 and -1
			If >2: 0, 1, 2, 3, ....
		So we need to map each label to a label in the correct format
		And we need to save these mapping so we can unmap them when we return the labels back to the user.
		"""

		# Initialize map
		self.__class_map = {}

		# These are the labels
		labels = np.unique(self.__ys)

		# Binary classification
		if (self.__num_classes == 2):
			for i in range(self.__size):
				# map each labels[0] in self.__ys to 1 and each labels[1] in self.__ys to -1
				if (self.__ys[i] == labels[0]):
					self.__class_map[1] = labels[0]
					self.__ys[i] = 1
				else:
					self.__class_map[-1] = labels[1]
					self.__ys[i] = -1

		# Multiclass classification
		else:
			for i in range(self.__size):
				# map each label in self.__ys to its index in the labels array (guaranteed to be 0...N-1)
				new_label = np.where(labels == self.__ys[i])[0][0]
				self.__class_map[new_label] = self.__ys[i]
				self.__ys[i] = new_label

	def __train_one(self, i, shared_classifiers):
		"""
		Parameters:
		-----------
		i : int
			- Index for the y_mat matrix
		shared_classifiers : multiprocessing.managers.AutoProxy[Queue]
			- Interprocess communication queue that we push our classifier onto when it finishes training

		This function trains the ith classifier and
		appends the instance of the classifier to
		the multiprocessing queue
		"""
		y = self.__y_mat[:, i]
		clf = MP_SVM(self.__kernel_type, self.__C, self.__gamma, self.__degree, self.__tol, self.__eps, self.__max_iter, self.__solver, self.__num_processes)
		clf.fit(self.__xs, y)
		shared_classifiers.put((clf, i))

	def __valid_train_params(self, x, y):
		"""
		Parameters
		----------
		x : ndarray
			- Data
		y : ndarray
			- Labels

		Returns
		-------
		bool
			- True if training params are valid, False otherwise
		"""

		# Check at least 1 row
		if (x.shape == () or y.shape == ()):
			print("Empty array(s)!")
			return False

		# Check that they have same number of rows
		if (x.shape[0] != y.shape[0]):
			print("Input sizes do not match")
			return False

		return True

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

		if (not self.__valid_train_params(x, y)):
			exit(1)

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
		"""
		Parameters:
		-----------
		clf : MP_SVM
			- Classifier to classify label against rest of labels
		label : int
			- The label that clf is classifying for
		x : ndarray
			- the testing data
		shared_classifiers : multiprocessing.managers.AutoProxy[Queue]
			- IPC Queue to push prediction when done training
		"""
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

		# Binary prediction, so just need one classifier
		if (self.__num_classes == 2):
			return self.__unmap_labels(self.__ovr_classifiers[0].predict(x))

		# Multiclass prediction
		# Set up IPC
		manager = mp.Manager()
		shared_predictions = manager.Queue()

		# Create one process for each predition task
		currently_executing_processes = []
		for i in range(min(4, self.__num_classes)):
			currently_executing_processes.append(mp.Process(target=self.predict_one, args=(self.__ovr_classifiers[i][0], self.__ovr_classifiers[i][1], x, shared_predictions)))

		# Start each process                                                                 
		for p in currently_executing_processes:
			p.start()


		while (len(currently_executing_processes) > 0):
			# wait for the oldest process, remove it from the currently_executing_processes
			currently_executing_processes.pop(0).join()
			if (i + 1 < self.__num_classes):
				i += 1 # increment i for next label
				p = mp.Process(target=self.predict_one, args=(self.__ovr_classifiers[i][0], self.__ovr_classifiers[i][1], x, shared_predictions))
				currently_executing_processes.append(p) # add to executing processes because of next line
				p.start()

		# Aggregate predictions from each classifier
		# Initialize vote counts to all zeros for each class
		class_predictions = np.zeros((x.shape[0], self.__num_classes))

		# For each classifier
		while (shared_predictions.empty() == False):
			tup = shared_predictions.get()
			preds = tup[0]
			label = tup[1]

			# for each prediction on x[i]
			for i in range(preds.shape[0]):
				# vote for current label
				if (preds[i] == 1):
					class_predictions[i, label] += 1
				else:
					# vote for all other labels
					class_predictions[i, :] += 1
					class_predictions[i, label] -= 1

		# return the unmapped indices with maximum vote counts
		return self.__unmap_labels(np.argmax(class_predictions, 1))

	def predict_accuracy(self, xs, ys):
		"""
		Parameters
		----------
		xs : ndarray
			- Input samples to predict the labels of
		ys : ndarray
			- Labels of xs

		Returns
		-------
		accuracy : float
			- The accuracy of the prediction
		"""
		preds = self.predict(xs)
		accuracy = np.sum(ys==preds) / xs.shape[0]
		return accuracy

