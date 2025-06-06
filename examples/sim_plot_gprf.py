import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


import sys
sys.path.append("../src")

from rf_functions import *

y_scale = [ 10, 240]
# tests = ['simGP3d_rbf_seed=0','simGP3d_rbf_seed=1', 'simGP3d_rbf_seed=2'
# 	'simGP3d_rbf_seed=3', 'simGP3d_rbf_seed=4','simGP3d_rbf_seed=5']
seeds = 10
tests = ['simGP_rf_dim=2_seed=%d'%i for i in range(10)]

min_val = find_min(seeds = seeds, N = 200) #find_min comes from rf_functions








# # # #new era with gp noise =1e-4, n_agent =10. simGPrf2d algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
dates = [['2024-03-27_225754'] + ['2024-03-27_225759'] * 6,
['2024-03-27_231223']  + ['2024-03-27_231240'] * 6, 
['2024-03-27_232631'] + ['2024-03-27_232723'] + ['2024-03-27_232724'] + ['2024-03-27_232723']* 4,
['2024-03-27_234050'] + ['2024-03-27_234148'] * 6,
['2024-03-27_235516'] + ['2024-03-27_235608'] * 6,
['2024-03-28_000846'] + ['2024-03-28_001029'] * 6,
['2024-03-28_002330'] + ['2024-03-28_002432']* 6,
['2024-03-28_003754'] + ['2024-03-28_003847'] * 2 + ['2024-03-28_003846'] * 2 + ['2024-03-28_003847']  + ['2024-03-28_003846'] ,
['2024-03-28_005236'] + ['2024-03-28_005302'] *6,
['2024-03-28_010619'] + ['2024-03-28_010600'] * 6]




# # # # #new era with gp noise =1e-4, n_agent =20. simGPrf2d algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar). 
# # # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # # #n_ysamples = 1
# # # n_runs = 10. everybody starting from same initial conditions (for each run)
# # # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-28_012040'] + ['2024-03-28_011956'] * 6,
# ['2024-03-28_013648']  + ['2024-03-28_014709'] * 6, 
# ['2024-03-28_014656'] + ['2024-03-28_021445'] * 6,
# ['2024-03-28_020450'] + ['2024-03-28_024610'] * 6,
# ['2024-03-28_021805'] + ['2024-03-28_031422'] * 6,
# ['2024-03-28_023322'] + ['2024-03-28_034236'] * 2 + ['2024-03-28_034237'] + ['2024-03-28_034236'] + ['2024-03-28_034237'] + ['2024-03-28_034236'],
# ['2024-03-28_024616'] + ['2024-03-28_041248']* 6,
# ['2024-03-28_030352'] + ['2024-03-28_044309'] * 6,
# ['2024-03-28_031619'] + ['2024-03-28_051037'] * 6,
# ['2024-03-28_033210'] + ['2024-03-28_053836'] * 6]





algs = ['DPPTS','BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ'] 

# algs = ['DPPTS','BUCB',  'UCBPE','SP','TS',  'TS_UCB_SEQ'] 


# n_ysamples_list = [3,3,3,3,3,10,1,5]
# n_ysamples_list = [3,3,3,3,3,10]
# n_ysamples_list = [3,3,3,3,3,1]
# n_ysamples_list = [1] * 5 
n_ysamples_list = [1] * 7
# n_ysamples_list = [10] * 6
# n_ysamples_list = [10] * 6 + [1, 30]
# algs = ['TS_UCB_SEQ']



n_runs = 10
# n_runs = 5
# n_agents = 1
# n_agents = 3
# n_agents = 5
# n_agents = 20
n_agents = 10
# n_iters = 150
n_iters = 21
# n_iters = 51
# n_iters = 16
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



emp_min_val = min_val

#first do a run to identify empirical min val

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
		if np.min(regret_matrix[:,-1]) <= emp_min_val[i]:
			emp_min_val[i] = np.min(regret_matrix[:-1])

heatmap = np.empty((len(algs), len(tests)))


last_plot_iter = n_iters
regret_all = np.zeros((len(algs),len(tests) * n_runs, last_plot_iter))
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
		regret_matrix = regret_matrix[:,:last_plot_iter] - emp_min_val[i]
		# regret_all[j,:] += np.sum(regret_matrix,axis = 0)
		for k in range(n_runs):	
			regret_all[j,i * n_runs + k, : ]  = regret_matrix[k,:]
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
		plt.plot(np.arange(last_plot_iter),np.mean(regret_matrix,axis = 0), label = alg)
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
	plt.title(test)
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




#Compute rank of each alg for each test
heatmap_rank = np.empty((len(algs), len(tests)))
for i in range(len(tests)):
	heatmap_rank[:,i] = heatmap[:,i].argsort().argsort()
	for j in range(len(algs)):
		print("(test %d) this is order for alg %s: " % (i,algs[j]), heatmap_rank[j,i]+1)

for j in range(len(algs)):
	print("(n workers = %d) avg rank for alg %s: " % (n_agents,algs[j]), np.mean(heatmap_rank[j,:]+1))



for j in range(len(algs)):
	print("(n workers = %d) avg ratio (to best) for alg %s: " % (n_agents,algs[j]), heatmap_plus_avg[j,-1])

print("regret_all shape", regret_all.shape)
print("this is average simple regret at last step for the algorithms")
print(np.mean(regret_all[:,:,-1], axis = 1))
print("this is std of simple regret at last step for the algorithms")
print(np.std(regret_all[:,:,-1], axis = 1))
print("this is ratio of simple regret at last step for the algorithms")
print(np.std(regret_all[:,:,-1], axis = 1))

test = 'simGP2d_rf'
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
plt.close()
for i in range(len(algs)):
	alg = algs[i]
	if alg == 'TS_UCB_SEQ':
		alg = 'TS-RSR'
	plt.plot(np.arange(last_plot_iter),np.mean(regret_all[i,:,:],axis = 0), label = alg, color = colors[i])

	for k in range(last_plot_iter):
		if (k % 5 == 0 or k == -1) and k != 0:
			plt.errorbar(x = k, y = np.mean(regret_all[i,:,:],axis = 0)[k], 
				yerr = np.std(regret_all[i,:,:], axis=0)[k]/np.sqrt(n_runs * 10),
				color = colors[i], capsize = 2.0)

out_path = "../comparison_plots/%s/" %(test)
if not os.path.exists(out_path):
	os.makedirs(out_path)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("simple regret")
plt.yscale('log')
plt.title(test)
plt.legend()
plt.savefig(out_path+"%s_%s_n_agents=%d_n_restarts=%d_diff_ystar=%s_after_icml_last_iter=%d.pdf" % (test,date,n_agents,
		 n_restarts,diff_fstar, last_plot_iter), bbox_inches='tight')
plt.close()

# regret_all_avg_last = np.mean(regret_all[:,:,-1], axis = 1)
# regret_all_ratio = np.empty(len(algs))
# for i in range(len(algs)):
# 	alg = algs[i]
# 	if alg == 'TS_UCB_SEQ':
# 		alg = 'TS-RSR'
# 	best = np.min(regret_all_avg_last)
# 	regret_all_ratio[i] = regret_all_avg_last[i]/best
# 	print("ratio of simple regret to best regret at last step", regret_all_ratio[i], alg)


