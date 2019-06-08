import multiprocessing as mp


class XX:
	def __init__(self):
		pass

	def fit_one(self, i):
		self.__shared_classifiers.put(i)

	def fit(self):
		manager = mp.Manager()
		self.__shared_classifiers = manager.Queue()
		processes = []

		for i in range(2):
			processes.append(mp.Process(target = self.fit_one, args=(i,)))

		for p in processes:
			p.start()

		for p in processes:
			p.join()


		# save the classifiers
		while (self.__shared_classifiers.empty() == False):
			x = self.__shared_classifiers.get()



def train_one(i):
	x = XX()
	x.fit()
	shared_classifiers.put((i, x))




manager = mp.Manager()
shared_classifiers = manager.Queue()
processes = []
for i in range(3):
	processes.append(mp.Process(target=train_one, args=(i,)))

for p in processes:
	p.start()

for p in processes:
	p.join()

while (shared_classifiers.empty() == False):
	x = shared_classifiers.get()



# def fit_one(i,q):
# 	q.put(i)

# def fit(x):

# 	manager = mp.Manager()
# 	q = manager.Queue()

# 	ps = []
# 	for i in range(2):
# 		p = mp.Process(target=fit_one, args=(i,q))
# 		ps.append(p)

# 	for p in ps:
# 		p.start()

# 	for p in ps:
# 		p.join()

# 	while (q.empty() == False):
# 		print(q.get())

# 	q2.put(x)





# manager2 = mp.Manager()
# q2 = manager2.Queue()

# ps = []
# for i in range(2):
# 	ps.append(mp.Process(target=fit, args=(i,)))

# for p in ps:
# 	p.start()

# for p in ps:
# 	p.join()

# while (q2.empty() == False):
# 	print(q2.get())
