import numpy as np
import svm2

num_samples = 90

clf = svm2.SVM()

# generate 3 classes of data
one = np.random.normal(loc=0, scale = 1, size=int(num_samples/3))
two = np.random.normal(loc=5, scale = 1, size=int(num_samples/3))
three = np.random.normal(loc=10, scale = 1, size=int(num_samples/3))

xs = np.transpose(np.matrix(np.append(np.append(one, two), three)))
ys = np.append(np.append(np.zeros((int(num_samples/3))), np.ones((int(num_samples/3)))), np.ones(num_samples / 3) + 1)

# ys = np.append(np.ones(30), np.ones(60) * -1)

clf.fit(xs, ys)
pred_ys = clf.predict(xs)
acc = np.sum(ys==pred_ys) / num_samples
print(acc)
