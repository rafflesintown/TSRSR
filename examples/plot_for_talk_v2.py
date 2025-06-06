import os
import csv
import copy
import json
import imageio.v2 as imageio
import datetime
import warnings
import itertools
import __main__
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm
from matplotlib import cm
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
import torch
import gpytorch
import pandas as pd

import sys


sys.path.append("../src")
from bayesian_optimization import *


np.random.seed(0)

def f_true(x):
	return((np.sin(3* x + 0.8) + 0.5 * np.cos(x - 1.1) + np.cos(x-0.5))/3.)


def ES_grad(x, model,beta = 2,  radius=1.0, n_workers = 3):
    """
            Entropy search acquisition function (for 1d functions)
            Args:
                a: # agents
                x: array-like, shape = [n_samples, n_hyperparams]
                n: agent nums
                projection: if project to a close circle
                radius: circle of the projected circle
            """

    x = x.reshape(-1, 1)
    domain = np.asarray([[-5,5]])

    mu, sigma = model.predict(x, return_std=True, return_tensor=True)
    sigma = torch.sqrt(sigma)
    ucb = mu + beta * sigma
    # ucb = mu + self.beta * torch.sqrt(sigma)
    amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
    # x = np.vstack([amaxucb for _ in range(self.n_workers)])
    init_x = np.random.normal(amaxucb, radius, (n_workers, domain.shape[0]))

    x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
    # print("INITIAL x diffable", x.grad)
    optimizer = torch.optim.Adam([x], lr=0.1)
    # training_iter = 200
    training_iter = 50
    best_loss = 1e6
    best_x = copy.deepcopy(x)
    for i in range(training_iter):
        optimizer.zero_grad()
        joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
        cov_x_xucb = model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
        cov_x_x = model.predict(x, return_cov=True, return_tensor=True)[1]
        loss = -torch.matmul(
                torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 1e-6 * torch.eye(len(cov_x_x)))), cov_x_xucb)
        # print("loss at iter %d" %i, loss)
        loss.backward()
        # print("x grad at %d" %i, x.grad)
        optimizer.step()
        # print("x grad at %d after step" %i, x.grad)
        # # project back to domain
        # x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
        # x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
        # x.detach_()

        # project back to domain
        if torch.all(x == torch.clamp(x, torch.tensor(domain[:,0]),torch.tensor(domain[:, 1]))) == False:
            # print("x_opt for alg %d" %k, x_opt)
            x = torch.clamp(x, torch.tensor(domain[:, 0]),torch.tensor(domain[:,1]))
            x = torch.tensor(x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x], lr=0.01)
            optimizer.zero_grad()
        else:
            pass
    return x.clone().detach().numpy()


def plot_prior():

	#enable latex
	plt.rcParams['text.usetex'] = True
	plt.rcParams.update({'font.size': 20})

	model = TorchGPModel(X = None,Y = None)
	domain = np.linspace(-5,5,30)
	x = np.reshape(domain,(-1,1))
	x_mean, x_var = model.predict(x, return_std=True)
	x_std = np.sqrt(x_var)
	plt.plot(domain, x_mean, label = r'mean($\mu (\cdot))$')
	plt.fill_between(domain, x_mean + x_std, x_mean - x_std, 
		label = r"$\mu(\cdot) \pm \sigma(\cdot)$", alpha = 0.2)
	plt.xlabel(r"x")
	plt.ylabel(r"f(x)")
	plt.legend()
	plt.savefig("prior.pdf", bbox_inches='tight')


def plot_acq():
	#create a torchGPModel
	pt1 = -2
	pt2 = 2
	X = np.asarray([[pt1], [pt2]])
	Y = np.asarray([sin_1d(pt1), sin_1d(pt2)])

	#enable latex
	plt.rcParams['text.usetex'] = True
	plt.rcParams.update({'font.size': 20})

	model = TorchGPModel(torch.tensor(X) ,torch.tensor(Y))
	model.model.covar_module.base_kernel.lengthscale = 0.3
	domain = np.linspace(0,1,30)
	x = np.reshape(domain,(-1,1))
	x_mean, x_var = model.predict(x, return_std=True)
	x_std = np.sqrt(x_var)
	plt.plot(domain, np.sin(3. * domain), '--', label = r"true $f$", color = "grey")
	plt.plot(domain, x_mean, label = r'mean($\mu (\cdot))$', color = 'blue')
	plt.scatter([pt1, pt2],[sin_1d(pt1), sin_1d(pt2)], 
		s = 50, color = "black", marker = "X")
	plt.fill_between(domain, x_mean + x_std, x_mean - x_std, 
		label = r"$\mu(\cdot) \pm \sigma(\cdot)$", alpha = 0.2, color = 'blue')
	plt.xlabel(r"$x$")
	plt.ylabel(r"$f(x)$")
	plt.legend(loc = 'lower right')
	plt.savefig("posterior_before_acq_v2.pdf", bbox_inches='tight')

	x_next = x[np.argmax(x_mean + x_std)]
	plt.scatter(x_next, np.max(x_mean + x_std), 
		s = 50,marker = "^", color = "magenta")
	plt.savefig("posterior_with_acq_v2.pdf", bbox_inches='tight')
	plt.close()

	x_next = x_next[0]
	X = np.vstack([X, [x_next]])
	print("x next", x_next)
	Y = np.append(Y, sin_1d(x_next))
	model.fit(X,Y)
	x_mean, x_var = model.predict(x, return_std=True)
	x_std = np.sqrt(x_var)
	plt.plot(domain, np.sin(3. * domain), '--', label = r"true $f$", color = "grey")
	plt.plot(domain, x_mean, label = r'mean($\mu (\cdot))$', color = 'blue')
	plt.scatter([pt1, pt2,x_next],[sin_1d(pt1), sin_1d(pt2), sin_1d(x_next)], 
		color = "black", s = 50, marker = "X")
	plt.fill_between(domain, x_mean + x_std, x_mean - x_std, 
		label = r"$\mu(\cdot) \pm \sigma(\cdot)$", alpha = 0.2, color = 'blue')
	plt.xlabel(r"$x$")
	plt.ylabel(r"$f(x)$")
	plt.legend(loc = 'lower right')
	plt.savefig("posterior_after_acq_v2.pdf", bbox_inches='tight')


def plot_batch():
	#create a torchGPModel
	X_pts = np.random.uniform(low=-5,high=5, size = 3)
	X_pts = np.array([-1.5,-0.5,0.5,1.5])
	X = np.reshape(X_pts,(-1,1))
	Y = f_true(X_pts)

	#enable latex
	plt.rcParams['text.usetex'] = True
	plt.rcParams.update({'font.size': 20})

	model = TorchGPModel(torch.tensor(X) ,torch.tensor(Y), objective_name = 'testing')
	model.model.covar_module.base_kernel.lengthscale = 0.3
	domain = np.linspace(-5,5,60)
	x = np.reshape(domain,(-1,1))
	x_mean, x_var = model.predict(x, return_std=True)
	x_std = np.sqrt(x_var)


	#ES acq
	x_samples = np.random.uniform(low=-5,high=5, size = 100)
	x_new = ES_grad(x, model,beta = 0.15,  radius=0.2, n_workers = 3)
	x_new = x_new[:-1]
	print("x_new", x_new)

	plot_domain = np.linspace(-2.5,2.5,80)
	x_plot = np.reshape(plot_domain,(-1,1))
	x_mean, x_var = model.predict(x_plot, return_std=True)
	x_std = np.sqrt(x_var)


	plt.plot(plot_domain, f_true(plot_domain), '--', label = r"true $f$", color = "grey")
	plt.plot(plot_domain, x_mean, label = r'mean($\mu (\cdot))$', color = 'blue')
	plt.scatter(X_pts,f_true(X), 
		color = "black", s = 50, marker = "X")
	plt.scatter(x_new, f_true(x_new), 
		s = 50,marker = "X", color = "magenta")
	plt.fill_between(plot_domain, x_mean + 1.5 * x_std, x_mean - 1.5 * x_std, 
		label = r"$\mu(\cdot) \pm \sigma(\cdot)$", alpha = 0.2, color = 'blue')
	plt.xlabel("Stimulation parameter")
	# ax = plt.gca();                   
	# ax.get_yaxis().set_visible(False)
	# ax.get_xaxis().set_ticks([])
	# ax.axis("off")
	# plt.ylabel(r"$f(x)$")
	# plt.legend(loc = 'lower center')
	plt.savefig("prior_with_ES_batch_v2.pdf", bbox_inches='tight')
	plt.close()


	#update model.
	X = np.vstack([X, x_new.reshape(-1,1)])
	Y = np.append(Y, (f_true(x_new[0]), f_true(x_new[1])))
	model.fit(X,Y)

	plot_domain = np.linspace(-1.5,2.5,80)
	x_plot = np.reshape(plot_domain,(-1,1))
	x_mean, x_var = model.predict(x_plot, return_std=True)
	x_std = np.sqrt(x_var)



	plt.plot(plot_domain, f_true(plot_domain), '--', label = r"true $f$", color = "grey")
	plt.plot(plot_domain, x_mean, label = r'mean($\mu (\cdot))$', color = 'blue')
	plt.scatter(X_pts,f_true(X)[:-2], 
		color = "black", s = 50, marker = "X")
	plt.scatter(x_new, f_true(x_new), 
		s = 50,marker = "X", color = "magenta")
	plt.fill_between(plot_domain, x_mean + 1.5 * x_std, x_mean - 1.5 * x_std, 
		label = r"$\mu(\cdot) \pm \sigma(\cdot)$", alpha = 0.2, color = 'blue')
	plt.xlabel("Stimulation parameter")
	# ax = plt.gca();                   
	# ax.get_yaxis().set_visible(False)
	# ax.get_xaxis().set_ticks([])
	# ax.axis("off")
	# plt.ylabel(r"$f(x)$")
	# plt.legend(loc = 'lower center')
	plt.savefig("posterior_with_ES_batch_v2.pdf", bbox_inches='tight')
	plt.close()




if __name__ == '__main__':
	# main()
	# plot_acq()
	plot_batch()