import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


tests = ['rosenbrock', 'ackley', 'bird']
# tests = ['ackley']
# dates=['2023-12-22_175710','2023-12-22_175710', '2023-12-22_175710']
dates=['2023-12-31_123129', '2023-12-31_123129'] #date i corresponds to alg i's date
# algs = ['ES','BUCB','TS_UCB']
algs = ['BUCB', 'TS_UCB_SEQ']
n_iters = 150

for i in range(len(tests)):
	test = tests[i]
	for j in range(len(algs)):
		date = dates[j]
		alg = algs[j]
		all_info = pd.read_csv("../result/%s/%s%s/data/data.csv"%(test, alg,date))
		regret = all_info['regret'].to_numpy()
		regret_matrix = np.reshape(regret, (-1,n_iters))
		# print(0, regret[0])
		# print(150, regret[150])
		# print(regret_matrix)
		# print("average", np.mean(regret_matrix,axis = 0)[0])
		plt.plot(np.arange(n_iters),np.mean(regret_matrix,axis = 0), label = alg)
		out_path = "../comparison_plots/%s/" %(test)
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		plt.grid()
		plt.xlabel("iteration")
		plt.ylabel("instant regret")
		plt.yscale('log')
		plt.title(test)
		plt.legend()
	plt.savefig(out_path+"%s_%s.pdf" % (test,date), bbox_inches='tight')
	plt.close()