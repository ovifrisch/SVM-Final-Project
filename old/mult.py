import multiprocessing as mp
def p1(x):
	q2.put(x)
	def f(i):
		q.put(i)

	manager = mp.Manager()
	q = manager.Queue()

	ps = []
	for i in range(2):
		p = mp.Process(target=f, args=(i,))
		ps.append(p)

	for p in ps:
		p.start()

	for p in ps:
		p.join()





manager2 = mp.Manager()
q2 = manager2.Queue()

ps = []
for i in range(2):
	p = mp.Process(target=p1, args=(i,))
	ps.append(p)

for p in ps:
	p.start()

for p in ps:
	p.join()

while (q2.empty() == False):
	print(q2.get())
