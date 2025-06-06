import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

y_scale = [ 10, 240]
tests = ['ackley_3d']








# # # #new era with gp noise =1e-4, n_agent =20. ackley_3d. algs (in order) = TS_UCB_SEQ, DPPTS, BUCB,UCBPE,SP, TS, EI. 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
dates = [['2024-03-25_182547'] + ['2024-03-27_100605'] + ['2024-03-25_182547'] * 5]





algs = ['TS_UCB_SEQ','DPPTS','BUCB',  'UCBPE','SP','TS',  "EI"] 

# n_ysamples_list = [3,3,3,3,3,10,1,5]
# n_ysamples_list = [3,3,3,3,3,10]
# n_ysamples_list = [3,3,3,3,3,1]
# n_ysamples_list = [1] * 5 
n_ysamples_list = [1] * 7
# n_ysamples_list = [10] * 6
# n_ysamples_list = [10] * 6 + [1, 30]
# algs = ['TS_UCB_SEQ']



n_runs = 10
# n_agents = 1
# n_agents = 3
# n_agents = 5
# n_agents = 5
n_agents = 20
# n_iters = 150
# n_iters = 11
# n_iters = 51
n_iters = 16
# n_iters = 100
# n_iters = 75
# n_iters = 80



# n_ysamples = 1
# n_ysamples = 2 #for the other algs when n_agents = 5 (for these algs, y_samples not important)
# n_ysamples = 10
# n_ysamples = 3
# ts_ucb_seq_n_ysamples = 10
# ts_ucb_seq_n_ysamples = 3
n_restarts = 0 #restarts for TS_UCB and TS_UCB_vals
# diff_fstar = "false"
diff_fstar = "true"


heatmap = np.empty((len(algs), len(tests)))

colors = ['pink', 'blue', 'green', 'red', 'purple', 'orange', 'brown']

last_plot_iter = n_iters
# last_plot_iter = 30
# last_plot_iter = 100
for i in range(len(tests)):
	test = tests[i]
	for j in range(len(algs)):
		n_ysamples = n_ysamples_list[j]
		date = dates[i][j]
		alg = algs[j]
		# if alg in ['TS_RSR', 'TS_UCB_SEQ', 'TS_UCB', 'TS_UCB_MOD', 'TS_RSR_MOD']:
		# 	all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
		# 		%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
		# else:
		# 	all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d.csv"
		# 		%(test, alg,date,n_agents, n_runs,n_agents))
		all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
			%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
		# if alg == "TS_UCB_SEQ":
		# 	all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
		# 	%(test, alg,date,n_agents, n_runs,n_agents,ts_ucb_seq_n_ysamples))
		# else:
		# 				all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
		# 	%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
		regret = all_info['regret'].to_numpy()
		# regret = all_info['regret'].to_numpy()[:750] #try first 5 x 150 to compare against previous benchmark
		regret_matrix = np.reshape(regret, (-1,n_iters))
		regret_matrix = regret_matrix[:,:last_plot_iter]
		# regret_matrix = regret_matrix[:5,:] #take away last regret
		# print(0, regret[0])
		# print(150, regret[150])
		# print(regret_matrix)
		# print("average", np.mean(regret_matrix,axis = 0)[0])
		heatmap[j,i] = np.mean(regret_matrix,axis = 0)[last_plot_iter-1]
		if alg == "TS_UCB_VALS":
			alg = "TS-RSR-non-decay"
		elif alg == "TS_UCB_SEQ":
			# alg = "TS-RSR-%d" %n_ysamples
			alg = "TS-RSR"
		# plt.plot(np.arange(last_plot_iter),np.log10(np.mean(regret_matrix,axis = 0)), label = alg)
		plt.plot(np.arange(last_plot_iter),np.mean(regret_matrix,axis = 0), label = alg, color = colors[j])
		for k in range(last_plot_iter):
			if (k % 5 == 0 or k == -1) and k != 0:
				plt.errorbar(x = k, y = np.mean(regret_matrix,axis = 0)[k], 
					yerr = np.std(regret_matrix, axis=0)[k]/np.sqrt(n_runs),
					capsize = 2.0,
					color = colors[j])
		# if alg == "TS-RSR-SEQ": #90th percent inside the interval
		# 	plt.fill_between(np.arange(n_iters),np.mean(regret_matrix,axis = 0) + 1.28* np.std(regret_matrix,axis = 0), 
		# 		np.mean(regret_matrix,axis = 0), alpha = 0.2)
		print(test, "last instant regret: %.6f, last instant variance: %.4f" % 
			(np.mean(regret_matrix,axis = 0)[-1],np.std(regret_matrix,axis = 0)[-1]), alg)
		out_path = "../comparison_plots/%s/" %(test)
		if not os.path.exists(out_path):
			os.makedirs(out_path)
	plt.grid()
	plt.xlabel("iteration")
	plt.ylabel("simple regret")
	plt.yscale('log')
	plt.title(test + " (m = %d)" %n_agents)
	plt.legend()
	plt.savefig(out_path+"%s_%s_n_agents=%d_n_restarts=%d_diff_ystar=%s_after_icml_last_iter=%d.pdf" % (test,date,n_agents,
			 n_restarts,diff_fstar, last_plot_iter), bbox_inches='tight')
	plt.close()

for i in range(len(tests)):
	best = np.min(heatmap[:,i])
	heatmap[:,i] /= best

#compute a last column of the average ratio

heatmap_plus_avg = np.empty((len(algs), len(tests)+1))
heatmap_plus_avg[:,:-1] = heatmap
heatmap_plus_avg[:,-1] = np.mean(heatmap_plus_avg[:,:-1], axis=1)


print(heatmap_plus_avg)