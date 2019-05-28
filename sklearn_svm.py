

import numpy as np
from sklearn.svm import SVC

num_samples = 1000

posX = np.random.normal(loc=0, scale = 1, size=int(num_samples/2))
negX = np.random.normal(loc=10, scale=1, size=int(num_samples/2))
X = np.append(posX, negX)

ys = np.append(np.ones(int(num_samples / 2)), (-1 * np.ones(int(num_samples / 2))))
clf = SVC(kernel="rbf")
clf.fit(np.transpose(np.matrix(X)), ys)

posX_test = np.random.normal(loc=0, scale = 1, size=int(num_samples / 2))
negX_test = np.random.normal(loc=4, scale = 1, size=int(num_samples / 2))
X_test = np.append(posX_test, negX_test)
pred_ys = clf.predict(np.transpose(np.matrix(X_test)))
acc = np.sum(ys==pred_ys) / num_samples
print("Accuracy: " + str(acc))
