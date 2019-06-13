from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_moons
import matplotlib.pyplot as plt
from svc2 import SVC
import time
import sklearn.svm as skl
import numpy as np
from sklearn.utils import shuffle

def generate_xs(num_rows, num_cols, num_classes, class_separation):
    
    # Function that sets the center of a class as a function of its label
    def loc_func(i):
        return i * class_separation
    
    partition = int(num_rows / num_classes) # each class will have this many samples
    mat = np.zeros((partition*num_classes, num_cols))
    for i in range(num_classes):
        for j in range(partition):
            mat[(i*partition) + j, :] = np.random.normal(loc=loc_func(i), scale = 1, size=num_cols)
    return mat


def generate_ys(num_rows, num_classes):
    # create a 1 dimensional ndarray like: [0,0,0,1,1,1] if M was 6 and c was 2
    partition = int(num_rows / num_classes)
    y = np.zeros(partition * num_classes)
    for i in range(num_classes):
        y[i*partition:i*partition+partition] = np.repeat(i, partition)
    return y

# RESULTS 1
# num = 21
# row_sizes = list(map(lambda x: 100*x, list(range(num)[1:])))
# num_cols = 5

# svc1 = SVC(kernel_type="linear", num_processes=5)
# svc2 = SVC(kernel_type="linear", num_processes=1)
# svc3 = skl.SVC(kernel="linear")

# models = [(svc1, "SVC2", np.zeros(len(row_sizes)), np.zeros(len(row_sizes))),
# (svc2, "SVC2'", np.zeros(len(row_sizes)), np.zeros(len(row_sizes))),
# (svc3, "SVC", np.zeros(len(row_sizes)), np.zeros(len(row_sizes)))]

# for i, num_rows in enumerate(row_sizes):
# 	x_train, x_test = gen_ndim_norm(num_cols, num_rows, 2, 2), gen_ndim_norm(num_cols, num_rows, 2, 2)
# 	y_train, y_test = gen_labels(num_rows, 2), gen_labels(num_rows, 2)
# 	x_train, y_train = shuffle(x_train, y_train)
# 	for j, (model, name, _, _) in enumerate(models):
# 		t = time.time()
# 		model.fit(x_train, y_train)
# 		t = round(time.time() - t, 3)
# 		yhat = model.predict(x_test)
# 		acc = sum(yhat == y_test) / num_rows
# 		models[j][2][i] = acc
# 		models[j][3][i] = t
# 	print("Done " + str(num_rows) + " samples")

# Plot the times
# def plot_2d(data_idx, title, ylabel):
# 	colors = ["bo-", "ro-", "go-"]
# 	labels = [model[1] for model in models]
# 	for i in range(len(models)):
# 		x1 = np.asarray(row_sizes)
# 		x2 = models[i][data_idx]
# 		plt.plot(x1, x2, colors[i], label=labels[i])
# 	plt.title(title)
# 	plt.ylabel(ylabel)
# 	plt.xlabel('Training Set Size')
# 	plt.legend()
# 	plt.show()


# plot_2d(3, "Training Time vs Num Samples", "Time (in seconds)")
# plot_2d(2, "Accuracy vs Num Samples", "Accuracy")


# RESULTS 2
# from sklearn.datasets.samples_generator import make_circles
# num_samples = 1000
# tr_x, tr_y = make_circles(num_samples, factor=0.2, noise=0.1)
# te_x, te_y = make_circles(num_samples, factor=0.2, noise=0.1)


# kernels = ["linear", "rbf", "sigmoid", "poly"]
# kern_acc = np.zeros((len(kernels), 2))
# for idx, kernel in enumerate(kernels):
# 	svc1 = SVC(kernel_type=kernel)
# 	svc2 = skl.SVC(kernel=kernel)
# 	svc1.fit(tr_x, tr_y)
# 	svc2.fit(tr_x, tr_y)
# 	p1 = svc1.predict(te_x)
# 	p2 = svc2.predict(te_x)
# 	acc1 = sum(p1 == te_y) / num_samples
# 	acc2 = sum(p2 == te_y) / num_samples
# 	kern_acc[idx, 0] = acc1
# 	kern_acc[idx, 1] = acc2

# colors = ['bo', 'ro', 'go', 'yo']
# arr_ = np.array([1, 2])
# for i in range(len(kernels)):
# 	plt.plot(arr_, kern_acc[i, :], colors[i], label=kernels[i])
# plt.title("Kernel Type vs Accuracy for make_circles")
# plt.ylabel("Accuracy")
# plt.xticks([])
# plt.xlabel("SVC2                                                                                             SVC")
# plt.legend()
# plt.show()

# RESULTS 3

# dimensions = list(range(0, 400, 5))[1:]
# num_samples = 100
# results = np.zeros((len(dimensions), 2))

# for idx, dim in enumerate(dimensions):
# 	x_train, x_test = gen_ndim_norm(dim, num_samples, 2, 2), gen_ndim_norm(dim, num_samples, 2, 2)
# 	y_train, y_test = gen_labels(num_samples, 2), gen_labels(num_samples, 2)
# 	x_train, y_train = shuffle(x_train, y_train)
# 	svc1 = SVC(max_iter=1000)
# 	svc2 = skl.SVC(max_iter=1000, gamma='auto')
# 	svc1.fit(x_train, y_train)
# 	svc2.fit(x_train, y_train)
# 	p1 = svc1.predict(x_test)
# 	p2 = svc2.predict(x_test)
# 	acc1 = sum(p1 == y_test) / num_samples
# 	acc2 = sum(p2 == y_test) / num_samples
# 	results[idx, 0] = acc1
# 	results[idx, 1] = acc2
# 	print("finished dim: " + str(dim))

# plt.plot(np.asarray(dimensions), results[:, 0], 'ro', label = "SVC2")
# plt.plot(np.asarray(dimensions), results[:, 1], 'bo', label = "SVC")
# plt.title("Dimensions vs Accuracy")
# plt.xlabel("# dimensions")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()


# RESULTS 4

# seps = list(range(0, 400, 5))
# seps = [x / 100 for x in seps]
# results = np.zeros(len(seps))
# num_samples = 400
# dim = 3
# for i, sep in enumerate(seps):
# 	x_train, x_test = gen_ndim_norm(dim, num_samples, 2, sep), gen_ndim_norm(dim, num_samples, 2, sep)
# 	y_train, y_test = gen_labels(num_samples, 2), gen_labels(num_samples, 2)
# 	x_train, y_train = shuffle(x_train, y_train)
# 	svc1 = SVC(max_iter=1000)
# 	svc1.fit(x_train, y_train)
# 	p1 = svc1.predict(x_test)
# 	acc1 = sum(p1 == y_test) / num_samples
# 	results[i] = acc1
# 	print("done " + str(sep))

# plt.plot(np.asarray(seps), results, 'ro', label = "SVC2")
# plt.title("Data Separation vs Accuracy")
# plt.xlabel("Units of separation")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()


# RESULTS 5
num_classes_list = list(range(2, 11))

# Keep the # of samples, # of dimensions fixed, and separation fixed
num_samples = 500
num_dims = 3
sep = 4

# Put the models we want to test in a list
models = [(SVC(num_processes1=4), "SVC2"), (skl.SVC(gamma="auto"), "SVC"), (SVC(num_processes1=1), "SVC2'"), (SVC(num_processes1=4, num_processes2 = 1), "SVC2''")]

accuracies = np.zeros((len(models), len(num_classes_list)))
times = np.zeros((len(models), len(num_classes_list)))

for i, num_classes in enumerate(num_classes_list):
    # generate data
    tr_x, te_x = generate_xs(num_samples, num_dims, num_classes, sep), generate_xs(num_samples, num_dims, num_classes, sep)
    tr_y, te_y = generate_ys(num_samples, num_classes), generate_ys(num_samples, num_classes)
    tr_x, tr_y = shuffle(tr_x, tr_y)
    for j, model in enumerate(models):
        t = time.time()
        model[0].fit(tr_x, tr_y)
        t = time.time() - t
        preds = model[0].predict(te_x)
        acc = sum(preds == te_y) / num_samples
        accuracies[j, i] = acc
        times[j, i] = round(t, 3)
    print("done " + str(num_classes))

def plot_(data_arr, title, ylabel):
    colors = ["bo-", "ro-", "go-", "yo-"]
    for i, model in enumerate(models):
        if (ylabel == "Accuracy" and i > 1):
            break
        plt.plot(np.asarray(num_classes_list), data_arr[i, :], colors[i], label=model[1])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('# Classes')
    plt.legend()
    if (ylabel == "Accuracy"):
        plt.ylim((0.9,1.001))
    plt.show()


plot_(times, "# Classes vs. Time", "Time (in seconds)")
plot_(accuracies, "# Classes vs Accuracy", "Accuracy")


# num_labels = 2
# num_samples = 200
# num_dims = 5
# sep = 2

# x_train, x_test = gen_ndim_norm(num_dims, num_samples, num_labels, sep), gen_ndim_norm(num_dims, num_samples, num_labels, sep)
# y_train, y_test = gen_labels(num_samples, num_labels), gen_labels(num_samples, num_labels)
# x_train, y_train = shuffle(x_train, y_train)

# svc2 = SVC()
# svc2.fit(x_train, y_train)
# acc = svc2.predict_accuracy(x_test, y_test)
# print(acc)















