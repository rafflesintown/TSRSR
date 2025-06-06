import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})

y_scale = [ 10, 240]
# tests = ['ackley','bird','rosenbrock']
tests = ['ackley']
# tests = ['ackley','rosenbrock']
# tests = ['bird']
# tests = ['braninToy']
# tests = ['Boston']
# tests = ['rosenbrock']
# tests = ["goldstein","griewank"]
# tests = ['ackley', 'rosenbrock']
# tests = ['rastrigin']






# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, same fstar
# dates = [['2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-18_220548'],
# 		['2024-02-17_211809', '2024-02-17_211809'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-18_220548'],
# 		['2024-02-17_211909','2024-02-17_211908','2024-02-17_211909','2024-02-17_211909','2024-02-18_220548']] #for n_agents 5


# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, same fstar
# dates = [['2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-18_220548'],
# 		['2024-02-17_211809', '2024-02-17_211809'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-18_220548'],
# 		['2024-02-17_211909','2024-02-17_211908','2024-02-17_211909','2024-02-17_211909','2024-02-18_220548']] #for n_agents 5

# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar
# dates = [['2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-18_231819'],
# 		['2024-02-17_211809', '2024-02-17_211809'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-18_231819'],
# 		['2024-02-17_211909','2024-02-17_211908','2024-02-17_211909','2024-02-17_211909','2024-02-18_231819']] #for n_agents 5


# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar. TS_UCB_SEQ now gradient steps 10.
# dates = [['2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-20_221652'],
# 		['2024-02-17_211809', '2024-02-17_211809'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-21_222656'],
# 		['2024-02-17_211909','2024-02-17_211908','2024-02-17_211909','2024-02-17_211909','2024-02-20_223005']] #for n_agents 5


# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# dates = [['2024-02-22_191302','2024-02-17_205349','2024-02-17_205349','2024-02-17_205349','2024-02-20_221652','2024-02-22_195137'],
# 		['2024-02-22_191302', '2024-02-17_211809'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-21_222656','2024-02-22_195137'],
# 		['2024-02-22_191302','2024-02-17_211908','2024-02-17_211909','2024-02-17_211909','2024-02-20_223005','2024-02-22_195137']] #for n_agents 5


# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels
# dates = [['2024-02-22_202424','2024-02-22_202424','2024-02-17_205349','2024-02-17_205349','2024-02-20_221652','2024-02-22_195137'],
# 		['2024-02-22_202424', '2024-02-22_202424'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-21_222656','2024-02-22_195137'],
# 		['2024-02-22_202424','2024-02-22_202424','2024-02-17_211909','2024-02-17_211909','2024-02-20_223005','2024-02-22_195137']] #for n_agents 5


# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# dates = [['2024-02-22_202424','2024-02-22_202424','2024-02-17_205349','2024-02-17_205349','2024-02-23_163749','2024-02-22_195137'],
# 		['2024-02-22_202424', '2024-02-22_202424'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-23_163749','2024-02-22_195137'],
# 		['2024-02-22_202424','2024-02-22_202424','2024-02-17_211909','2024-02-17_211909','2024-02-23_163749','2024-02-22_195137']] #for n_agents 5




# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# dates = [['2024-02-22_202424','2024-02-22_202424','2024-02-17_205349','2024-02-17_205349','2024-02-23_163749','2024-02-22_195137'],
# 		['2024-02-22_202424', '2024-02-22_202424'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-23_163749','2024-02-22_195137'],
# 		['2024-02-22_202424','2024-02-22_202424','2024-02-17_211909','2024-02-17_211909','2024-02-23_163749','2024-02-22_195137']] #for n_agents 5



# #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss
# dates = [['2024-02-22_202424','2024-02-22_202424','2024-02-17_205349','2024-02-17_205349','2024-02-23_231620','2024-02-22_195137'],
# 		['2024-02-22_202424', '2024-02-22_202424'  ,'2024-02-17_211809','2024-02-17_211809','2024-02-23_231620','2024-02-22_195137'],
# 		['2024-02-22_202424','2024-02-22_202424','2024-02-17_211909','2024-02-17_211909','2024-02-23_231620','2024-02-22_195137']] #for n_agents 5


# #new era with gp noise =1e-4, n_agent =5. goldstein and griewank. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# dates = [['2024-02-23_202202' for i in range(6)],
# 		['2024-02-23_213734' for i in range(6)]]


# #new era with gp noise =1e-4, n_agent =5. goldstein and griewank. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# #also TS_UCB_SEQ checks if loss after GD is better than argmin loss
# dates = [['2024-02-23_202202' for i in range(6)],
# 		['2024-02-23_213734' for i in range(4)] + ['2024-02-23_224603'] + ['2024-02-23_213734']]



#new era with gp noise =1e-4, n_agent =5. rastrigin. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
#also TS_UCB_SEQ checks if loss after GD is better than argmin loss
# dates = [['2024-02-24_003306' for i in range(6)]]



# # #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# #n_ysamples = 3
# # actual noise has sigma 1e-3, likelihood assumes variance is 1e-3 
# dates = [['2024-02-24_194834' for i in range(6)],
# 		['2024-02-24_212134' for i in range(6)]	,
# 		['2024-02-24_221454' for i in range(6)]]

# # # #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 10
# # actual noise has sigma 1e-3, likelihood assumes variance is 1e-3 
# dates = [['2024-02-25_140127' for i in range(5) ] + ['2024-02-25_090540'] ,
# 		['2024-02-25_151340' for i in range(5)] + ['2024-02-25_090540'],
# 		['2024-02-25_193733' for i in range(5)] + ['2024-02-25_093950']]


# # # #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # actual noise has sigma 1e-3, likelihood assumes variance is 1e-3 
# dates = [['2024-02-25_140127' for i in range(5) ] + ['2024-02-26_095918'] ,
# 		['2024-02-25_151340' for i in range(5)] + ['2024-02-26_095918'],
# 		['2024-02-25_193733' for i in range(5)] + ['2024-02-26_095918']]


# # # #new era with gp noise =1e-4, n_agent =10. braninToy. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-27_045740'] + ['2024-03-09_201118'] * 5 + ['2024-03-09_205859']]

# # # #new era with gp noise =1e-4, n_agent =10. braninToy. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-09_201118'] * 5 + ['2024-03-09_205859']]



# # # #new era with gp noise =1e-4, n_agent =20. braninToy. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-10_212031'] * 5 + ['2024-03-10_210942']]



# # # #new era with gp noise =1e-4, n_agent =5. rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-26_012827'] * 6]


# # # #new era with gp noise =1e-4, n_agent =5. ackley, bird rosenbrock. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-27_022821'] + ['2024-03-26_224133'] * 6,
# ['2024-03-27_090106'] * 7,
# ['2024-03-27_022821'] + ['2024-03-26_012827'] * 6]



# # #new era with gp noise =1e-4, n_agent =5. ackley, bird rosenbrock. algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# #n_ysamples = 1
# n_runs = 10. everybody starting from same initial conditions (for each run)
# actual noise and likelihood both sigma = 1e-3
dates = [['2024-03-27_022821'] + ['2024-03-26_224133'] * 6]
dates = [[[dates[0][i]] for i in range(len(dates[0]))]]
print("dates", dates)

dates[0][0] += ['2024-04-02_234954']*2  #add for dppts
dates[0][1]+= ['2024-04-02_234954']*2  #add for BUCB
dates[0][-1]+= ['2024-04-02_234954'] * 2  #add for ts_ucb_seq

# # # #new era with gp noise =1e-4, n_agent =5. bird, algs (in order) = DPPTS, BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-03-27_090106'] * 7]

# # # #new era with gp noise =1e-4, n_agent =5. ackley, algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# #  likelihood sigma^2 = 1e-3, noise sigma = 1e-3
# dates = [['2024-03-29_155552'] * 6]



# # # #new era with gp noise =1e-4, n_agent =5. ackley, algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 1
# # n_runs = 10. everybody starting from same initial conditions (for each run)
# # actual noise and likelihood both sigma = 1e-3
# dates = [['2024-04-01_113518'] * 6]






# # # #new era with gp noise =1e-4, n_agent =5. ackley, bird, rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 10
# # actual noise has sigma 1e-3, likelihood assumes variance is 1e-3 
# #TS_seq_Ucb for n_ysamples = 10,1,5
# dates = [['2024-02-25_140127' for i in range(5) ] + ['2024-02-25_090540'] + ['2024-02-26_095918'] + ['2024-02-26_095918'],
# 		['2024-02-25_151340' for i in range(5)] + ['2024-02-25_090540']	+ ['2024-02-26_095918'] + ['2024-02-26_095918'],
# 		['2024-02-25_193733' for i in range(5)] + ['2024-02-25_093950'] + ['2024-02-26_095918'] + ['2024-02-26_095918']]





# # #new era with gp noise =1e-4, n_agent =5. rosenbrock only. algs =TS_UCB_SEQ (diff fstar) only. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# #n_ysamples = 10 now for ts_ucb_seq, for rosenbrock
# dates = [['2024-02-25_090246']]

# # #new era with gp noise =1e-4, n_agent =5. rosenbrock only. algs =TS_UCB_SEQ (diff fstar) only. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# #n_ysamples = 3 now for ts_ucb_seq, for rosenbrock
# dates = [['2024-02-25_105451']]



# # # #new era with gp noise =1e-4, n_agent =10. ackley, bird, rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 10
# dates = [['2024-02-25_212623' for i in range(6) ],
# 		['2024-02-25_230035' for i in range(6)],
# 		['2024-02-25_230035' for i in range(6)]]






# #new era with gp noise =1e-4, n_agent =10. rastrigin. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# #also TS_UCB_SEQ checks if loss after GD is better than argmin loss
# dates = [['2024-02-24_084616' for i in range(6)]]


# #new era with gp noise =1e-4, n_agent =10. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar
# dates = [['2024-02-19_133941','2024-02-19_113906','2024-02-19_113906','2024-02-19_133941','2024-02-19_100523'], 
# 			['2024-02-19_133941','2024-02-19_113906','2024-02-19_113906','2024-02-19_133941','2024-02-19_100523'], 
# 			['2024-02-19_133941','2024-02-19_113906','2024-02-19_113906','2024-02-19_133941','2024-02-19_100523']] #for n_agents=10





# #new era with gp noise =1e-4, n_agent =10. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar. TS_UCB_SEQ now gradient steps 10.
# dates = [['2024-02-19_133941','2024-02-19_113906','2024-02-19_113906','2024-02-19_133941','2024-02-21_224955'], 
# 			['2024-02-19_133941','2024-02-19_113906','2024-02-19_113906','2024-02-19_133941','2024-02-21_224955'], 
# 			['2024-02-19_133941','2024-02-19_113906','2024-02-19_113906','2024-02-19_133941','2024-02-21_224955']] #for n_agents=10




# #new era with gp noise =1e-4, n_agent =10. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ, BUCB, EI now gradient steps 10.
# # BUCB/UCBPE, sigma now using Desautels
# dates = [['2024-02-22_204015','2024-02-22_204015','2024-02-19_113906','2024-02-19_133941','2024-02-21_224955','2024-02-22_204015'] for i in range(3)] #for n_agents=10


#new era with gp noise =1e-4, n_agent =10. ackley, bird, rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ (diff fstar), EI. TS_UCB_SEQ, BUCB, EI now gradient steps 10.
# BUCB/UCBPE, sigma now using Desautels
# #also TS_UCB_SEQ checks if loss after GD is better than argmin loss
# dates = [['2024-02-22_204015','2024-02-22_204015','2024-02-19_113906','2024-02-19_133941','2024-02-24_113137','2024-02-22_204015'] for i in range(3)] #for n_agents=10



# #new era with gp noise =1e-4, n_agent =20. ackley, and rosenbrock only. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar
# dates = [['2024-02-19_223341','2024-02-19_223341','2024-02-19_223341','2024-02-19_212642','2024-02-19_172912'] for i in range(2)] #for n_agents=20

# #new era with gp noise =1e-4, n_agent =20. ackley, and rosenbrock only. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar. 
# dates = [['2024-02-19_223341','2024-02-19_223341','2024-02-19_223341','2024-02-19_212642','2024-02-19_172912'] for i in range(2)] #for n_agents=20


# #new era with gp noise =1e-4, n_agent =20. ackley, bird(now with sigma^2 = 1e-3 for BUCB) and rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar
# dates = [['2024-02-19_223341','2024-02-19_223341','2024-02-19_223341','2024-02-19_212642','2024-02-21_224955'],
# 			['2024-02-20_094754','2024-02-19_223341','2024-02-19_223341','2024-02-19_212642','2024-02-21_224955'],
# 			['2024-02-19_223341','2024-02-19_223341','2024-02-19_223341','2024-02-19_212642','2024-02-21_224955']] #for n_agents=20

# #new era with gp noise =1e-4, n_agent =1. ackley, bird and rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar
# dates = [['2024-02-20_154241','2024-02-20_154241','2024-02-20_154241','2024-02-20_134312','2024-02-20_134312'] for i in range(3)] #for n_agents = 1


# #new era with gp noise =1e-4, n_agent =3. ackley, bird and rosenbrock. algs = BUCB,UCBPE,SP, TS, TS_UCB_SEQ, diff fstar
# dates = [['2024-02-20_161955','2024-02-20_161955','2024-02-20_161955','2024-02-20_162513','2024-02-20_162513'] for i in range(3)] #for n_agents = 1


# # # #new era with gp noise =1e-4, n_agent =1. ackley, bird, rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 10
# dates = [['2024-02-26_223940' for i in range(6) ] + ['2024-02-27_121210','2024-02-27_121210'],
# 		['2024-02-27_083939' for i in range(6)] + ['2024-02-27_121210','2024-02-27_121210'],
# 		['2024-02-27_105445' for i in range(6)] + ['2024-02-27_121210','2024-02-27_121210']]


# # # #new era with gp noise =1e-4, n_agent =1. ackley, rosenbrock. algs (in order) = BUCB,UCBPE,SP, TS, EI, TS_UCB_SEQ (diff fstar). 
# # TS_UCB_SEQ and BUCB, EI now gradient steps 10.
# # # # BUCB/UCBPE, sigma now using Desautels. TS_UCB_SEQ now updated to set fstar_hat to be np.max(mu) + 1e-4 if if it's smaller than that
# # # # #also TS_UCB_SEQ checks if loss after GD is better than argmin loss. Moreover, adding new (high mu) points to the sample x for everybody.
# # #n_ysamples = 10
# dates = [['2024-02-26_223940' for i in range(6) ] + ['2024-02-27_121210','2024-02-27_121210'],
# 		['2024-02-27_105445' for i in range(6)] + ['2024-02-27_121210','2024-02-27_121210']]




# algs = ['BUCB',  'UCBPE','SP','TS',  'TS_UCB_SEQ'] 

# algs = ['BUCB',  'UCBPE','SP','TS',  'TS_UCB_SEQ',"EI"] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI", 'TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI", 'TS_UCB_SEQ'] 
# algs = ['BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ'] 
algs = ['DPPTS','BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ'] 

# algs = ['BUCB',  'UCBPE','SP','TS',  "EI",'TS_UCB_SEQ', 'TS_UCB_SEQ','TS_UCB_SEQ'] 
# n_ysamples_list = [3,3,3,3,3,10,1,5]
# n_ysamples_list = [3,3,3,3,3,10]
# n_ysamples_list = [3,3,3,3,3,1]
n_ysamples_list = [1] * 7
seeds = [-1,10,20]
# n_ysamples_list = [1] * 6
# n_ysamples_list = [10] * 6
# n_ysamples_list = [10] * 6 + [1, 30]
# algs = ['TS_UCB_SEQ']



n_runs = 10
# n_agents = 1
# n_agents = 3
# n_agents = 10
n_agents = 5
# n_agents = 20
# n_iters = 150
# n_iters = 11
# n_iters = 6
# n_iters_all = [21] + [11] * 6
# n_iters_all = [151] * 7
n_iters_all = [(151,51,51)] + [(151,51,51)] + [(151,)] * 4 + [(151,51,51)]

# n_iters_all = [151] * 6

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


# heatmap = np.empty((len(algs), len(tests)))

n_algs = 7
# n_repeats = [1,2,1,1,1,1,2] #number of times each alg is repeated
n_repeats = [3,3,1,1,1,1,3] #number of times each alg is repeated

heatmap = np.empty((n_algs, len(tests)))


# last_plot_iter = n_iters
# last_plot_iters = n_iters_all
last_plot_iters = [51] * 7
# last_plot_iters = [31] * 6
# last_plot_iters = [51] * 6
# colors = plt.cm.jet(np.linspace(0,1,len(algs)))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
print("colors", colors)
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
		# plt.plot(np.arange(last_plot_iter),np.log10(np.mean(regret_matrix,axis = 0)), label = alg)
		# plt.plot(np.arange(last_plot_iter),np.mean(regret_matrix,axis = 0), label = alg)
		plt.plot(np.arange(last_plot_iter),np.mean(regret_matrix_alg,axis = 0), label = alg,color = colors[j])
		# for k in range(n_runs):
		# 	plt.plot(np.arange(last_plot_iter),regret_matrix[k], color = colors[j], linewidth=0.1)

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

for i in range(len(tests)):
	best = np.min(heatmap[:,i])
	heatmap[:,i] /= best

#compute a last column of the average ratio

heatmap_plus_avg = np.empty((len(algs), len(tests)+1))
heatmap_plus_avg[:,:-1] = heatmap
heatmap_plus_avg[:,-1] = np.mean(heatmap_plus_avg[:,:-1], axis=1)


print(heatmap_plus_avg)