import multiprocessing as mp


def square(nums, res):
	for i, el in enumerate(nums):
		res[i] = el*el


if __name__ == "__main__":
	nums = [2, 3, 4]
	result = mp.Array('i', 3)
	p = mp.Process(target=square, args=(nums, result))
	
	p.start()
	p.join()
	print(result[:])
