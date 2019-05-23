import svm
import numpy as np

num_samples = 1000

if __name__ == "__main__":
	clf = svm.svm()
	posX = np.random.normal(loc=0, scale = 1, size=int(num_samples/2))
	negX = np.random.normal(loc=4, scale=1, size=int(num_samples/2))
	X = np.append(posX, negX)
	ys = np.append(np.ones(int(num_samples / 2)), (-1 * np.ones(int(num_samples / 2))))

	clf.fit(X,ys)
	
	posX_test = np.random.normal(loc=0, scale = 1, size=num_samples)
	negX_test = np.random.normal(loc=0, scale = 1, size=num_samples)
	X_test = np.append(posX_test, negX_test)	
	pred_ys = clf.predict(X_test)
	
	acc = np.sum(ys==pred_ys) / num_samples
	print("Accuracy: " + str(acc))
