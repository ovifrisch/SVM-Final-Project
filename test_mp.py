from multiprocessing import Pool



def f(x):
	return x[1]

x = [(1, 2), (5, 1)]
print(sorted(x, key=lambda el: el[0]))
