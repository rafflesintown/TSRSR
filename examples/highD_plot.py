import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})

y_scale = [ 10, 240]
tests = ['Michalewicz10d']
# tests = ['Hartmann6d','Griewank8d', 'Michalewicz10d']
# tests = ['Hartmann6d']
# tests = ['Griewank8d']




# # # # #new era with gp noise =1e-4, n_agent =5. hartmann6d. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # # #n_ysamples = 1
# # # n_runs = 10. everybody starting from same initial conditions (for each run)
# # # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-27_033831'] + ['2024-03-24_135553'] * 5 + ['2024-03-24_135552']]
# dates = [[[dates[0][i]] for i in range(len(dates[0]))]]
# print("dates", dates)

# dates[0][-2].append('2024-04-02_192347') 
# dates[0][-1].append('2024-04-02_194137') 



# # # # #new era with gp noise =1e-4, n_agent =10. Griewank8d. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # # #n_ysamples = 1
# # # n_runs = 10. everybody starting from same initial conditions (for each run)
# # # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-25_091527'] + ['2024-03-27_033831'] + ['2024-03-25_091452'] * 5 ]






# # # # #new era with gp noise =1e-4, n_agent =5. hartmann6d. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # # #n_ysamples = 1
# # # n_runs = 10. everybody starting from same initial conditions (for each run)
# # # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-24_135553'] * 5 + ['2024-03-24_135552']]


# # # # #new era with gp noise =1e-4, n_agent =5. mike10d. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # # #n_ysamples = 1
# # # n_runs = 10. everybody starting from same initial conditions (for each run)
# # # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-24_202056'] + ['2024-03-27_045740']+ ['2024-03-24_162425'] * 5]


# # # #new era with gp noise =1e-4, n_agent =5. mike10d. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
dates = [ ['2024-03-27_045740']+ ['2024-03-24_162425'] * 5 + ['2024-03-24_202056']]
dates = [[[dates[0][i]] for i in range(len(dates[0]))]]
print("dates", dates)

dates[0][3].append('2024-04-02_192212')
dates[0][-2].append('2024-04-02_192212')  #EI
dates[0][-1].append('2024-04-02_192212')  #ts-rsr







# algs = ['BUCB',  'UCBPE','SP','TS',  'TS_UCB_SEQ'] 

# algs = ['BUCB',  'UCBPE','SP','TS',  'TS_UCB_SEQ',"EI"] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI", 'TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI", 'TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ'] 
algs = ['DPPTS','BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ'] 
# algs = ['TS_UCB_SEQ', 'DPPTS','BUCB',  'UCBPE','SP','TS',  "EI"] 
# algs = ['BUCB',  'UCBPE','TS',"EI",'TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ', 'TS_UCB_SEQ','TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','TS',  "EI",'TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','TS', 'TS_UCB_SEQ'] 
# n_ysamples_list = [3,3,3,3,3,10,1,5]
# n_ysamples_list = [3,3,3,3,3,10]
# n_ysamples_list = [3,3,3,3,3,1]
# n_ysamples_list = [1] * 5 
n_ysamples_list = [1] * 7
seeds = [-1,10]
# n_ysamples_list = [10] * 6
# n_ysamples_list = [10] * 6 + [1, 30]
# algs = ['TS_UCB_SEQ']



n_runs = 10
# n_agents = 1
# n_agents = 3
# n_agents = 5
n_agents = 5
# n_agents = 10
# n_iters = 150
# n_iters = 11
# n_iters = 51
# n_iters = 31
# n_iters_all = [31] + [51]* 6
# n_iters_all = [31] * 7
# n_iters_all = [(31,)]  + [(51,)] * 4 + [(51,31)] * 2 #hartmann6d

n_iters_all = [(31,)]  * 3 + [(31,31)]  + [(31,)] + [(31,31)]*2 #mike10d

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


n_algs = 7
# n_repeats = [1,1,1,1,1,2,2] #number of times each alg is repeated for hartmann 6d
n_repeats = [1,1,1,2,1,2,2] #number of times each alg is repeated for mike10d
heatmap = np.empty((n_algs, len(tests)))


# last_plot_iter = n_iters
last_plot_iters = [31] * 7
# last_plot_iters = n_iters_all
# last_plot_iter = 30
# last_plot_iter = 100
min_val = 1e5
# colors = ['pink', 'blue', 'green', 'red', 'purple', 'orange', 'brown']
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
percentile_high = 0.9
percentile_low = 0.1
for i in range(len(tests)):
	test = tests[i]
	for j in range(n_algs):
		last_plot_iter = last_plot_iters[j]
		alg = algs[j]
		n_repeat = n_repeats[j]
		regret_matrix_alg = np.empty((n_repeat * n_runs, last_plot_iter))
		for l in range(n_repeat):
			n_iters = n_iters_all[j][l]
			n_ysamples = n_ysamples_list[j]
			date = dates[i][j][l]
			print("date here", date)
			seed = seeds[l]
			# if alg in ['TS_RSR', 'TS_UCB_SEQ', 'TS_UCB', 'TS_UCB_MOD', 'TS_RSR_MOD']:
			# 	all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
			# 		%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
			# else:
			# 	all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d.csv"
			# 		%(test, alg,date,n_agents, n_runs,n_agents))
			if seed == -1:
				all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
				%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
			else:
				all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d_seed=%d.csv"
				%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples, seed))
			# if alg == "TS_UCB_SEQ":
			# 	all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
			# 	%(test, alg,date,n_agents, n_runs,n_agents,ts_ucb_seq_n_ysamples))
			# else:
			# 				all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
			# 	%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
			regret = all_info['regret'].to_numpy()
			# regret = all_info['regret'].to_numpy()[:750] #try first 5 x 150 to compare against previous benchmark
			regret_matrix = np.reshape(regret, (-1,n_iters))
			regret_matrix_alg[n_runs * l : n_runs * (l+1), :] = regret_matrix[:,:last_plot_iter]
		heatmap[j,i] = np.mean(regret_matrix_alg,axis = 0)[last_plot_iter-1]
		if alg == "TS_UCB_VALS":
			alg = "TS-RSR-non-decay"
		elif alg == "TS_UCB_SEQ":
			# alg = "TS-RSR-%d" %n_ysamples
			alg = "TS-RSR"
		plt.plot(np.arange(last_plot_iter),np.mean(regret_matrix_alg,axis = 0), label = alg,color = colors[j])

		for k in range(last_plot_iter):
			if (k % 5 == 0 or k == -1) and k != 0:
				plt.errorbar(x = k, y = np.mean(regret_matrix_alg,axis = 0)[k], 
					yerr = np.std(regret_matrix_alg, axis=0)[k]/np.sqrt(n_runs * n_repeat),
					color = colors[j])
		print(test, "last instant regret: %.6f, last instant variance: %.4f" % 
			(np.mean(regret_matrix_alg,axis = 0)[-1],np.std(regret_matrix_alg,axis = 0)[-1]), alg)
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


if len(tests) == 1 and tests[0] == 'Boston':
	print("this is min val for Boston", min_val)
	heatmap -= min_val
for i in range(len(tests)):
	best = np.min(heatmap[:,i])
	heatmap[:,i] /= best

#compute a last column of the average ratio

heatmap_plus_avg = np.empty((len(algs), len(tests)+1))
heatmap_plus_avg[:,:-1] = heatmap
heatmap_plus_avg[:,-1] = np.mean(heatmap_plus_avg[:,:-1], axis=1)


print(heatmap_plus_avg)