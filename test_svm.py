import svm
import numpy as np
from sklearn.svm import SVC
import time
import multiprocessing as mp
from sklearn.utils import shuffle

class SVM_Tester:
	def __init__(self, num_dimensions = 1, num_samples = 100, loc1 = 1, loc2 = 4,
	tolerance=0.1, epsilon=0.1, C=0.05, kernel = "rbf", gamma = 1, degree = 3):

		self.loc1 = loc1
		self.loc2 = loc2
		self.num_samples = num_samples
		self.num_dimensions = num_dimensions
		self.kernel = kernel
		self.C = C
		self.gamma = gamma
		self.degree = degree
		self.tolerance = tolerance
		self.epsilon = epsilon
		self.generate_data()


	def __del__(self):
		pass

	def generate_data(self):
		self.train_xs = self.generate_xs()
		self.test_xs = self.generate_xs()
		self.train_ys = np.append(np.ones(int(self.num_samples / 2)), (-1 * np.ones(int(self.num_samples / 2))))
		self.test_ys = np.append(np.ones(int(self.num_samples / 2)), (-1 * np.ones(int(self.num_samples / 2))))

		# shuffle the dataset randomly (so not all ones at start and neg ones at end)
		self.train_xs, self.train_ys = shuffle(self.train_xs, self.train_ys)
		self.test_xs, self.test_ys = shuffle(self.test_xs, self.test_ys)

	def generate_xs(self):
		xs = np.zeros((self.num_samples, self.num_dimensions))
		for i in range(self.num_dimensions):
			posX = np.random.normal(loc=self.loc1, scale = 1, size=int(self.num_samples/2))
			negX = np.random.normal(loc=self.loc2, scale = 1, size=int(self.num_samples/2))
			X_i = np.transpose(np.matrix(np.append(posX, negX)))
			xs[:, i:i+1] = X_i
		return xs

	def train_one(self, start, end):
		clf = svm.svm(self.kernel, self.C, self.gamma, self.degree, self.tolerance, self.epsilon)
		xs = self.train_xs[start:end, :]
		ys = self.train_ys[start:end]
		clf.fit(xs, ys)
		self.classifiers.put(clf)

	def train_all(self):
		start_time = time.time()
		num_processes = 5
		sub_samples = int(self.train_xs.shape[0] / num_processes)
		self.classifiers = mp.Queue()
		processes = []
		for i in range(num_processes):
			start_idx = i*sub_samples
			end_idx = start_idx + sub_samples
			processes.append(mp.Process(target = self.train_one, args=(start_idx, end_idx)))

		for p in processes:
			p.start()

		for p in processes:
			p.join()

		self.train_time = time.time() - start_time

	def test_one(self, start, end):
		pred_ys = np.zeros((end - start))
		while (self.classifiers.empty() == False):
			pred_ys += self.classifiers.get().predict(self.test_xs[start:end, :])

		for i in range(end - start):
			if (pred_ys[i] >= 0):
				pred_ys[i] = 1
			else:
				pred_ys[i] = -1

		self.predictions.put((pred_ys, start, end))


	def test_all(self):
		start_time = time.time()
		num_processes = 5
		sub_samples = int(self.test_xs.shape[0] / num_processes)
		self.predictions = mp.Queue()
		processes = []
		for i in range(num_processes):
			start_idx = i*sub_samples
			end_idx = start_idx + sub_samples
			processes.append(mp.Process(target = self.test_one, args=(start_idx, end_idx)))

		for p in processes:
			p.start()

		for p in processes:
			p.join()

		preds = []
		while (self.predictions.empty() == False):
			preds.append(self.predictions.get())

		# sort by starting index
		preds = sorted(preds, key=lambda pred: pred[1])
		self.pred_ys = np.concatenate([pred[0] for pred in preds])
		# get the predictions of each of the classifiers and assign
		# to the class with most votes
		# self.pred_ys = self.clf.predict(self.test_xs)
		self.test_time = time.time() - start_time

	def get_training_data(self):
		return self.train_xs, self.train_ys

	def get_testing_data(self):
		return self.test_xs, self.test_ys

	def get_results(self):
		acc = np.sum(self.test_ys==self.pred_ys) / self.num_samples
		print_results("Our Accuracy: ", acc, self.train_time, self.test_time)


def print_results(whos, acc, train_time, test_time):
	print(whos + str(acc) + "  Train time: " + str(round(train_time, 3)) + "s  Test time: " + str(round(test_time, 3)) + "s")

if __name__ == "__main__":
	# Our SVM
	tester = SVM_Tester(num_dimensions=3, num_samples=500)
	tester.train_all()
	tester.test_all()
	tester.get_results()

	# Sci Kit Learn SVM
	clf2 = SVC(kernel="rbf", C=tester.C, gamma=tester.gamma, tol=tester.tolerance)
	start_time = time.time()
	clf2.fit(tester.get_training_data()[0], tester.get_training_data()[1])
	train_time = time.time() - start_time
	start_time = time.time()
	pred_ys = clf2.predict(tester.get_testing_data()[0])
	test_time = time.time() - start_time
	acc = np.sum(tester.get_testing_data()[1]==pred_ys) / tester.num_samples
	print_results("Scikit Accuracy: ", acc, train_time, test_time)
