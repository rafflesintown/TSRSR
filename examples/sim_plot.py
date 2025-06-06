import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

y_scale = [ 10, 240]
# tests = ['simGP3d_rbf_seed=0','simGP3d_rbf_seed=1', 'simGP3d_rbf_seed=2'
# 	'simGP3d_rbf_seed=3', 'simGP3d_rbf_seed=4','simGP3d_rbf_seed=5']
tests = ['simGP3d_rbf_seed=%d'%i for i in range(10)]
# tests = ['simGP3d_rbf_seed=0']








# # # #new era with gp noise =1e-4, n_agent =20. simGP3d algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
dates = [['2024-03-26_211454'] + ['2024-03-26_211450'] * 5 + ['2024-03-26_211445'],
['2024-03-26_213538']  + ['2024-03-26_213542'] * 5 + ['2024-03-26_213547'], 
['2024-03-26_215329'] + ['2024-03-26_214450'] * 5 + ['2024-03-26_215018'],
['2024-03-26_221111'] + ['2024-03-26_215418'] * 5 + ['2024-03-26_220104'],
['2024-03-26_222845'] + ['2024-03-26_220332'] * 5 + ['2024-03-26_221431'],
['2024-03-26_225340'] + ['2024-03-26_221236'] * 5 + ['2024-03-26_222700'],
['2024-03-26_233055'] + ['2024-03-26_222207']* 5+ ['2024-03-26_223734'],
['2024-03-26_235611'] + ['2024-03-26_223055'] * 5 +['2024-03-26_225844'],
['2024-03-27_000951'] + ['2024-03-26_223951'] * 5 + ['2024-03-26_232319'],
['2024-03-27_002117'] + ['2024-03-26_225918'] * 5 + ['2024-03-26_234019']]

dates = [[[dates[j][i]] for i in range(len(dates[j]))] for j in range(len(tests))]
# EI_tsrsr_new_dates = ['2024-04-02_214342','2024-04-02_215532', '2024-04-02_220851',
# 						'2024-04-02_221951', '2024-04-02_223321', '2024-04-02_224521',
# 						'2024-04-02_225736','2024-04-02_231019','2024-04-02_232346','2024-04-02_233807']

ts_tsrsr_new_dates = ['2024-04-03_101421','2024-04-03_102317', '2024-04-03_103251',
						'2024-04-03_104109', '2024-04-03_105152', '2024-04-03_110114',
						'2024-04-03_111035','2024-04-03_112011','2024-04-03_112930','2024-04-03_113840']
for j in range(len(tests)):
	dates[j][4] += [ts_tsrsr_new_dates[j], ts_tsrsr_new_dates[j]]
	dates[j][-1] += [ts_tsrsr_new_dates[j], ts_tsrsr_new_dates[j]]
	print("dates j", dates[j])




algs = ['DPPTS','BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ'] 

# algs = ['DPPTS','BUCB',  'UCBPE','SP','TS',  'TS_UCB_SEQ'] 


# n_ysamples_list = [3,3,3,3,3,10,1,5]
# n_ysamples_list = [3,3,3,3,3,10]
# n_ysamples_list = [3,3,3,3,3,1]
# n_ysamples_list = [1] * 5 
n_ysamples_list = [1] * 7
seeds = [-1,10,20] #only EI and tsrsr uses the last two seeds.
# n_ysamples_list = [10] * 6
# n_ysamples_list = [10] * 6 + [1, 30]
# algs = ['TS_UCB_SEQ']



n_runs = 5
# n_runs = 5
# n_agents = 1
# n_agents = 3
# n_agents = 5
n_agents = 5
# n_agents = 10
# n_iters = 150
n_iters = 51
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

n_algs = 7
heatmap = np.empty((len(algs), len(tests)))
n_repeats = [1,1,1,1,3,1,3] #number of times each alg is repeated
# n_runs_all = [5,10,10]


last_plot_iter = n_iters
regret_all = [np.zeros((len(tests) * n_runs * n_repeats[i], last_plot_iter)) for i in range(n_algs)]
# regret_all = np.zeros((len(algs),len(tests) * n_runs, last_plot_iter))
# last_plot_iter = 30
# last_plot_iter = 100
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']

for i in range(len(tests)):
	test = tests[i]
	for j in range(len(algs)):
		n_ysamples = n_ysamples_list[j]
		n_repeat = n_repeats[j]
		alg = algs[j]
		for l in range(n_repeat):
			seed = seeds[l]
			date = dates[i][j][l]
			alg = algs[j]
			if seed == -1:
				all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d.csv"
				%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples))
			else:
				# n_runs_new = 10 #temporary since I messed up the bash script
				all_info = pd.read_csv("../result/%s/%s%s_nworkers=%d/data/data_nruns=%d_n_agents=%d_n_ysamples=%d_seed=%d.csv"
				%(test, alg,date,n_agents, n_runs,n_agents,n_ysamples, seed))

			regret = all_info['regret'].to_numpy()
			# regret = all_info['regret'].to_numpy()[:750] #try first 5 x 150 to compare against previous benchmark
			regret_matrix = np.reshape(regret, (-1,n_iters))
			# regret_matrix = regret_matrix[:,:last_plot_iter]
			# regret_matrix = regret_matrix[:n_runs,:last_plot_iter] #temporary fix for messsing up bash script
			# regret_all[j,:] += np.sum(regret_matrix,axis = 0)
			for k in range(n_runs):	
				# regret_all[j,i * n_runs  + k, : ]  = regret_matrix[k,:]
				regret_all[j][i * n_runs * n_repeat  + l * n_runs + k, : ]  = regret_matrix[k,:]				
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
					color = colors[j])
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

# for i in range(len(tests)):
# 	best = np.min(heatmap[:,i])
# 	heatmap[:,i] /= best



# #compute a last column of the average ratio

# heatmap_plus_avg = np.empty((len(algs), len(tests)+1))
# heatmap_plus_avg[:,:-1] = heatmap
# heatmap_plus_avg[:,-1] = np.mean(heatmap_plus_avg[:,:-1], axis=1)


# print(heatmap_plus_avg)
# print("regret_all shape", regret_all.shape)
# print("this is average simple regret at last step for the algorithms")
# print(np.mean(regret_all[:,:,-1], axis = 1))
# print("this is std of simple regret at last step for the algorithms")
# print(np.std(regret_all[:,:,-1], axis = 1))
# print("this is ratio of simple regret at last step for the algorithms")
# print(np.std(regret_all[:,:,-1], axis = 1))

test = 'simGP3d_rbf'
plt.close()
for i in range(len(algs)):
	alg = algs[i]
	if alg == 'TS_UCB_SEQ':
		alg = 'TS-RSR'
	plt.plot(np.arange(last_plot_iter),np.mean(regret_all[i],axis = 0), label = alg, color = colors[i])

	for k in range(last_plot_iter):
		if (k % 5 == 0 or k == -1) and k != 0:
			plt.errorbar(x = k, y = np.mean(regret_all[i],axis = 0)[k], 
				yerr = np.std(regret_all[i], axis=0)[k]/np.sqrt(n_runs * 10 * n_repeats[i]),
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


