import svm
import numpy as np
from sklearn.svm import SVC
import time

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
		self.train_xs = self.generate_xs()
		self.test_xs = self.generate_xs()
		self.ys = np.append(np.ones(int(num_samples / 2)), (-1 * np.ones(int(num_samples / 2))))
		self.clf = svm.svm(self.kernel, self.C, self.gamma, self.degree, self.tolerance, self.epsilon)

	def __del__(self):
		pass

	def generate_xs(self):
		xs = np.zeros((self.num_samples, self.num_dimensions))
		for i in range(self.num_dimensions):
			posX = np.random.normal(loc=self.loc1, scale = 1, size=int(self.num_samples/2))
			negX = np.random.normal(loc=self.loc2, scale = 1, size=int(self.num_samples/2))
			X_i = np.transpose(np.matrix(np.append(posX, negX)))
			xs[:, i:i+1] = X_i
		return xs

	def train(self):
		start_time = time.time()
		self.clf.fit(self.train_xs, self.ys)
		self.train_time = time.time() - start_time

	def test(self):
		start_time = time.time()
		self.pred_ys = self.clf.predict(self.test_xs)
		self.test_time = time.time() - start_time

	def get_training_data(self):
		return self.train_xs, self.ys

	def get_testing_data(self):
		return self.test_xs, self.ys

	def get_results(self):
		acc = np.sum(self.ys==self.pred_ys) / self.num_samples
		print_results("Our Accuracy: ", acc, self.train_time, self.test_time)


def print_results(whos, acc, train_time, test_time):
	print(whos + str(acc) + "  Train time: " + str(round(train_time, 3)) + "s  Test time: " + str(round(test_time, 3)) + "s")

if __name__ == "__main__":
	# Our SVM
	tester = SVM_Tester(num_dimensions=1, num_samples=100)
	tester.train()
	tester.test()
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
