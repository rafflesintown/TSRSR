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

warnings.filterwarnings("ignore")

class bayesian_optimization:
    def __init__(self, objective, domain, arg_max = None, n_workers = 1,
                 network = None, kernel = kernels.RBF(), alpha=10**(-10),
                 acquisition_function = 'ei', policy = 'greedy', fantasies = 0,
                 epsilon = 0.01, regularization = None, regularization_strength = None,
                 pending_regularization = None, pending_regularization_strength = None,
                 grid_density = 100, n_ysamples =1, args=dict()):

        # Optimization setup
        self.objective_name = args.objective
        self.objective = lambda x: - objective.function(x)
        self.n_workers = n_workers
        if network is None:
            self.network = np.eye(n_workers)
        else:
            self.network = network
        self._policy = policy
        if policy not in ['greedy', 'boltzmann']:
            print("Supported policies: 'greedy', 'boltzmann' ")
            return

        # Acquisition function
        if acquisition_function == 'ei':
            self._acquisition_function = self._expected_improvement
        elif acquisition_function == 'ts':
            self._acquisition_function = self._thompson_sampling
        elif acquisition_function == 'es':
            self._acquisition_function = self._entropy_search_single
        elif acquisition_function == 'ucb' or acquisition_function == 'bucb' or acquisition_function == 'ucbpe':
            self._acquisition_function = self._upper_confidential_bound
        elif acquisition_function == 'sp':
            pass
        elif acquisition_function == 'ts_ucb':
            pass
        elif acquisition_function == 'ts_ucb_seq':
            pass
        elif acquisition_function == 'ts_ucb_det':
            pass
        elif acquisition_function == 'ts_ucb_vals':
            pass
        elif acquisition_function == 'ts_rsr':
            pass
        elif acquisition_function == "ts_rsr_mod":
            self._acquisition_function = self._TS_RSR_mod
        elif acquisition_function == "ts_ucb_mod":
            self._acquisition_function = self._TS_UCB_mod
        elif acquisition_function == "ts_rsr_combine":
            self._acquisition_function = self._TS_RSR_combine
        else:
            print('Supported acquisition functions: ei, ts, es, ucb, ts_ucb(mod), ts_ucb_seq, ts_ucb_det,ts_ucb_vals, ts_rsr(mod), ts_rsr_combine')
            return
        self._epsilon = epsilon
        self._num_fantasies = fantasies

        # Regularization function
        self._regularization = None
        if regularization is not None:
            if regularization == 'ridge':
                self._regularization = self._ridge
            else:
                print('Supported regularization functions: ridge')
                return
        self._pending_regularization = None
        if pending_regularization is not None:
            if pending_regularization == 'ridge':
                self._pending_regularization = self._ridge
            else:
                print('Supported pending_regularization functions: ridge')
                return

        # Domain
        self.domain = domain    #shape = [n_params, 2]
        self._dim = domain.shape[0]
        self._grid_density = grid_density
        grid_elemets = []
        for [i,j] in self.domain:
            grid_elemets.append(np.linspace(i, j, self._grid_density))
        self._grid = np.array(list(itertools.product(*grid_elemets)))

        # Global Maximum
        self.arg_max = arg_max
        if self.arg_max is None:
            obj_grid = [self.objective(i) for i in self._grid]
            self.arg_max = np.array(self._grid[np.array(obj_grid).argmax(), :]).reshape(-1, self._dim)

        # Model Setup
        self.alpha = alpha
        self.kernel = kernel
        self._regularization_strength = regularization_strength
        self._pending_regularization_strength = pending_regularization_strength
        self.model = [GaussianProcessRegressor(  kernel=self.kernel,
                                                    alpha=self.alpha,
                                                    n_restarts_optimizer=10)
                                                    for i in range(self.n_workers) ]
        self.scaler = [StandardScaler() for i in range(n_workers)]

        # Data holders
        self.bc_data = None
        self.X_train = self.Y_train = None
        self.X = self.Y = None
        self._acquisition_evaluations = [[] for i in range(n_workers)]

        # file storage
        self.args = args
        # print("hi",self.args)
        self._DT_ = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._ROOT_DIR_ = os.path.dirname(os.path.dirname(__main__.__file__))

        alg_name = acquisition_function.upper()

        if args.fantasies:
            alg_name = alg_name + '-MC'
        if args.regularization is not None:
            alg_name = alg_name + '-DR'
        if args.pending_regularization is not None:
            alg_name = alg_name + '-PR'
        if args.policy != 'greedy':
            alg_name = alg_name + '-SP'
        # else:
        #     if args.unconstrained:
        #         alg_name = 'EI'
        #     else:
        #         alg_name = 'CWEI'

        # if args.n_workers > 1:
        #     alg_name = 'MA-' + alg_name
        # else:
        #     alg_name = 'SA-' + alg_name

        if 'sim' in self.args:
            if self.args.sim == True:
                alg_name = alg_name + '-SIM'
        self.alg_name = alg_name


        self.n_ysamples = n_ysamples

        self._TEMP_DIR_ = os.path.join(os.path.join(self._ROOT_DIR_, "result"), self.args.objective)
        self._ID_DIR_ = os.path.join(self._TEMP_DIR_, alg_name + self._DT_+"_nworkers=%s"%self.n_workers)
        self._DATA_DIR_ = os.path.join(self._ID_DIR_, "data")
        self._FIG_DIR_ = os.path.join(self._ID_DIR_, "fig")
        self._PNG_DIR_ = os.path.join(self._FIG_DIR_, "png")
        self._PDF_DIR_ = os.path.join(self._FIG_DIR_, "pdf")
        self._GIF_DIR_ = os.path.join(self._FIG_DIR_, "gif")
        for path in [self._TEMP_DIR_, self._DATA_DIR_, self._FIG_DIR_, self._PNG_DIR_, self._PDF_DIR_, self._GIF_DIR_]:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        # # Directory setup
        # self._DT_ = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # self._ROOT_DIR_ = os.path.dirname(os.path.dirname( __main__.__file__ ))
        # self._TEMP_DIR_ = os.path.join(self._ROOT_DIR_, "temp")
        # self._ID_DIR_ = os.path.join(self._TEMP_DIR_, self._DT_)
        # self._DATA_DIR_ = os.path.join(self._ID_DIR_, "data")
        # self._FIG_DIR_ = os.path.join(self._ID_DIR_, "fig")
        # self._PNG_DIR_ = os.path.join(self._FIG_DIR_, "png")
        # self._PDF_DIR_ = os.path.join(self._FIG_DIR_, "pdf")
        # self._GIF_DIR_ = os.path.join(self._FIG_DIR_, "gif")
        # for path in [self._TEMP_DIR_, self._DATA_DIR_, self._FIG_DIR_, self._PNG_DIR_, self._PDF_DIR_, self._GIF_DIR_]:
        #     try:
        #         os.makedirs(path)
        #     except FileExistsError:
        #         pass

        self.beta = None

    def _regret(self, y):
        return self.objective(self.arg_max[0]) - y

    def _mean_regret(self):
        r_mean = [np.mean(self._simple_regret[:,iter]) for iter in range(self._simple_regret.shape[1])]
        r_std = [np.std(self._simple_regret[:,iter]) for iter in range(self._simple_regret.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*rst/self._simple_regret.shape[0] for rst in r_std]
        return range(self._simple_regret.shape[1]), r_mean, conf95

    def _cumulative_regret(self):
        r_cum = [np.sum(self._simple_regret[:, : iter + 1], axis=1) for iter in range(self._simple_regret.shape[1])]
        r_cum = np.array(r_cum).T
        r_cum_mean = [np.mean(r_cum[:,iter]) for iter in range(r_cum.shape[1])]
        r_std = [np.std(r_cum[:,iter]) for iter in range(r_cum.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*rst/self._simple_regret.shape[0] for rst in r_std]
        return range(self._simple_regret.shape[1]), r_cum_mean, conf95

    def _mean_distance_traveled(self):
        d_mean = [np.mean(self._distance_traveled[:,iter]) for iter in range(self._distance_traveled.shape[1])]
        d_std = [np.std(self._distance_traveled[:,iter]) for iter in range(self._distance_traveled.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*dst/self._distance_traveled.shape[0] for dst in d_std]
        return range(self._distance_traveled.shape[1]), d_mean, conf95

    def _save_data(self, data, name):
        with open(self._DATA_DIR_ + '/config.json', 'w', encoding='utf-8') as file:
            json.dump(vars(self.args), file, ensure_ascii=False, indent=4)
        with open(self._DATA_DIR_ + '/' + name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i in zip(*data):
                writer.writerow(i)
        return

    def _ridge(self, x, center = 0):
        return np.linalg.norm(x - center)

    def _regularize(self, x, a, mu, Y_max):

        if self._regularization is None and self._pending_regularization is None:
            mu = mu - self._epsilon
        else:
            # Distance regularization
            if self._regularization is not None:
                if self._regularization == self._ridge:
                    if self._regularization_strength is not None:
                        reg = np.array([self._regularization_strength*self._ridge(i, self.X[a][-1]) for i in x])
                    else:
                        reg = []
                        kernel = copy.deepcopy(self.model[a].kernel_)
                        param = {"length_scale": 0.7*max([d[1]-d[0] for d in self.domain])}
                        kernel.set_params(**param)
                        for i in x:
                            k = float(kernel(np.array([i]), np.array([self.X[a][-1]])))
                            reg.append(0.1*(1 - k))
                        reg = np.array(reg)
                mu = mu - Y_max*reg

            # Pending query regularization
            if self._pending_regularization is not None:
                # Pending queries
                x_p = []
                for neighbour_agent, neighbour in enumerate(self.network[a]):
                    if neighbour and neighbour_agent < a:
                        x_p.append(self._next_query[neighbour_agent])
                x_p = np.array(x_p).reshape(-1, self._dim)
                if self._pending_regularization == self._ridge:
                    if self._pending_regularization_strength is not None:
                        pending_reg = np.array([self._pending_regularization_strength*sum([1/self._ridge(i, xp) for xp in x_p]) for i in x])
                    else:
                        pending_reg = np.array([sum([0.1*float(self.model[a].kernel_(np.array([i]), np.array([xp]))) for xp in x_p]) for i in x])
                mu = mu - Y_max*pending_reg

        return mu

    def _entropy_search_single(self, a, x, n, model = None):
        """
        Entropy search acquisition function.
        Args:
            a: # agents
            x: array-like, shape = [n_samples, n_hyperparams]
            model:
        """

        x = x.reshape(-1, self._dim)

        if model is None:
            model = self.model[a]

        # if self.beta is None:
        #     self.beta = 2.
        self.beta = 3 - 0.019 * n
        # print(self.beta)

        mu, sigma = model.predict(x, return_std=True)
        mu = np.squeeze(mu)
        ucb = mu + self.beta * sigma
        amaxucb = x[np.argmax(ucb)][np.newaxis, :]
        self.amaxucb = amaxucb

        # _, var_amaxucb_x = model.predict(amaxucb[np.newaxis, :], return_cov=True)
        cov_amaxucb_x = np.asarray([model.predict(np.vstack((xi, amaxucb)), return_cov=True)[1][-1, 0] for xi in x])
        var_amaxucb_x = model.predict(amaxucb, return_cov=True)[1].squeeze()

        acq = 1 / (var_amaxucb_x + 1) * cov_amaxucb_x ** 2
        return -1 * acq



    def _upper_confidential_bound(self, a, x, n, model = None):
        """
        Entropy search acquisition function.
        Args:
            a: # agents
            x: array-like, shape = [n_samples, n_hyperparams]
            model:
        """

        x = x.reshape(-1, self._dim)

        if model is None:
            model = self.model[a]

        # if self.beta is None:
        #     self.beta = 2.
        self.beta = 0.15 + 0.019 * n
        # print(self.beta)

        mu, sigma = model.predict(x, return_std=True)
        mu = np.squeeze(mu)
        acq = mu + self.beta * sigma
        amaxucb = x[np.argmax(acq)][np.newaxis, :]
        self.amaxucb = amaxucb
        #
        # # _, var_amaxucb_x = model.predict(amaxucb[np.newaxis, :], return_cov=True)
        # cov_amaxucb_x = np.asarray([model.predict(np.vstack((xi, amaxucb)), return_cov=True)[1][-1, 0] for xi in x])
        # var_amaxucb_x = model.predict(amaxucb, return_cov=True)[1].squeeze()
        #
        # acq = 1 / (var_amaxucb_x + 1e-8) * cov_amaxucb_x ** 2
        return -1 * acq


    def _expected_improvement(self, a, x, n, model = None):
        """
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
        """

        x = x.reshape(-1, self._dim)

        if model is None:
            model = self.model[a]

        Y_max = np.max(model.y_train_)

        mu, sigma = model.predict(x, return_std=True)
        mu = np.squeeze(mu)
        mu = self._regularize(x, a, mu, Y_max)

        with np.errstate(divide='ignore'):
            Z = (mu - Y_max) / sigma
            expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0
            expected_improvement[expected_improvement < 10**(-100)] = 0

        return -1 * expected_improvement

    def _thompson_sampling(self, a, x, n=None, model = None, num_samples = 1):
        """
        Thompson sampling acquisition function.
        Arguments:
        ----------
            agent: integer
                Agent id to find next query for.
            model: sklearn model
                Surrogate model used for acquisition function
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the thompson samples needs to be computed.
        """
        x = x.reshape(-1, self._dim)

        if model is None:
            model = self.model[a]

        Y_max = np.max(model.y_train_)

        yts = model.sample_y(x, n_samples=num_samples, random_state = None)

        if num_samples > 1:
            yts = np.squeeze(yts)
        ts = np.squeeze(np.mean(yts, axis=1))
        ts = self._regularize(x, a, ts, Y_max)

        return -1 * ts

    def _expected_acquisition(self, a, x, n=None):

        x = x.reshape(-1, self._dim)

        # Pending queries
        x_p = []
        for neighbour_agent, neighbour in enumerate(self.network[a]):
            if neighbour and neighbour_agent < a:
                x_p.append(self._next_query[neighbour_agent])
        x_p = np.array(x_p).reshape(-1, self._dim)

        if not x_p.shape[0]:
            return self._acquisition_function(a, x, n)
        else:
            # Sample fantasies
            rng = check_random_state(0)
            mu, cov = self.model[a].predict(x_p, return_cov=True)
            mu = mu[:,np.newaxis]
            mu = self.scaler[a].inverse_transform(mu)
            cov = self.scaler[a].scale_**2 * cov
            if mu.ndim == 1:
                y_fantasies = rng.multivariate_normal(mu, cov, self._num_fantasies).T
            else:
                y_fantasies = [rng.multivariate_normal(mu[:, i], cov, self._num_fantasies).T[:, np.newaxis] for i in range(mu.shape[1])]
                y_fantasies = np.hstack(y_fantasies)

            # models for fantasies
            fantasy_models = [GaussianProcessRegressor( kernel=self.kernel,
                                                        alpha=self.alpha,
                                                        optimizer=None)
                                                        for i in range(self._num_fantasies) ]

            # acquisition over fantasies
            fantasy_acquisition = np.zeros((x.shape[0], self._num_fantasies))
            for i in range(self._num_fantasies):

                f_X_train = self.X_train[a][:]
                f_y_train = self.Y_train[a][:]

                fantasy_scaler = StandardScaler()
                fantasy_scaler.fit(np.array(f_y_train).reshape(-1, 1))

                # add fantasy data
                for xf,yf in zip(x_p, y_fantasies[:,0,i]):
                    f_X_train = np.append(f_X_train, xf).reshape(-1, self._dim)
                    f_y_train = np.append(f_y_train, yf).reshape(-1, 1)

                # fit fantasy surrogate
                f_y_train = fantasy_scaler.transform(f_y_train)
                fantasy_models[i].fit(f_X_train, f_y_train)

                # calculate acqusition
                acquisition = self._acquisition_function(a,x,fantasy_models[i])
                for j in range(x.shape[0]):
                    fantasy_acquisition[:,i] = acquisition

            # compute expected acquisition
            expected_acquisition = np.zeros(x.shape[0])
            for j in range(x.shape[0]):
                expected_acquisition[j] = np.mean(fantasy_acquisition[j,:])

        return expected_acquisition

    def _boltzmann(self, n, x, acq):
        """
        Softmax distribution on acqusition function points for stochastic query selection
        Arguments:
        ----------
            n: integer
                Iteration number.
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the blotzmann needs to be computed and selected from.
            acq: array-like, shape = [n_samples, 1]
                The acqusition function value for x.
        """
        C = max(abs(max(acq)-acq))
        if C > 10**(-2):
            beta = 3*np.log(n+self._initial_data_size+1)/C
            _blotzmann_prob = lambda e: np.exp(beta*e)
            bm = [_blotzmann_prob(e) for e in acq]
            norm_bm = [float(i)/sum(bm) for i in bm]
            idx = np.random.choice(range(x.shape[0]), p=np.squeeze(norm_bm))
        else:
            idx = np.random.choice(range(x.shape[0]))
        return x[idx]

    def _find_next_query(self, n, a, random_search):
        """
        Proposes the next query.
        Arguments:
        ----------
            n: integer
                Iteration number.
            agent: integer
                Agent id to find next query for.
            model: sklearn model
                Surrogate model used for acqusition function
            random_search: integer.
                Number of random samples used to optimize the acquisition function. Default 1000
        """
        # Candidate set
        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], size=(random_search, self._dim))


        X = x[:]
        if self._record_step:
            X = np.append(self._grid, x).reshape(-1, self._dim)

        # Calculate acquisition function
        if self._num_fantasies:
            acq = - self._expected_acquisition(a, X, n)
        else:
            # Vanilla acquisition functions
            acq = - self._acquisition_function(a, X, n)

        if self._record_step:
            self._acquisition_evaluations[a].append(-1*acq[0:self._grid.shape[0]])
            acq = acq[self._grid.shape[0]:]

        # Apply policy
        if self._policy == 'boltzmann':
            # Boltzmann Policy
            x = self._boltzmann(n, x, acq)
        else:
            #Greedy Policy
            x = x[np.argmax(acq), :]

        return x

    def optimize(self, n_iters, n_runs = 1, x0=None, n_pre_samples=5, random_search=100, plot = True):
        """
        Arguments:
        ----------
            n_iters: integer.
                Number of iterations to run the search algorithm.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B to optimize the acquisition function.
            plot: bool or integer
                If integer, plot iterations with every plot number iteration. If True, plot every interation.
        """

        self._simple_regret = np.zeros((n_runs, n_iters+1))
        self._simple_cumulative_regret = np.zeros((n_runs, n_iters + 1))
        self._distance_traveled = np.zeros((n_runs, n_iters+1))

        for run in tqdm(range(n_runs), position=0, leave = None, disable = not n_runs > 1):



            # Reset model and data before each run
            self._next_query = [[] for i in range(self.n_workers)]
            self.bc_data = [[[] for j in range(self.n_workers)] for i in range(self.n_workers)]
            self.X_train = [[] for i in range(self.n_workers)]
            self.Y_train =[[] for i in range(self.n_workers)]
            self.X = [[] for i in range(self.n_workers)]
            self.Y = [[] for i in range(self.n_workers)]

            # Initial data
            if x0 is None:
                for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
                    for a in range(self.n_workers):
                        self.X[a].append(params)
                        self.Y[a].append(self.objective(params))
            else:
                # Change definition of x0 to be specfic for each agent
                for params in x0:
                    for a in range(self.n_workers):
                        self.X[a].append(params)
                        self.Y[a].append(self.objective(params))
            self._initial_data_size = len(self.Y[0])

            if self.args.model == 'sklearn':
                self.model = [GaussianProcessRegressor(kernel=self.kernel,
                                                       alpha=self.alpha,
                                                       n_restarts_optimizer=10)
                              for i in range(self.n_workers)]
            else:
                for a in range(self.n_workers):
                    Y = self.scaler[a].fit_transform(np.array(self.Y[a]).reshape(-1, 1)).squeeze()
                    self.model[a] = TorchGPModel(torch.tensor(self.X[a]).float(), torch.tensor(Y).float())


            for n in tqdm(range(n_iters+1), position = n_runs > 1, leave = None):

                # record step indicator
                self._record_step = False
                if plot and n_runs == 1:
                    if n == n_iters or not n % plot:
                        self._record_step = True



                self._prev_bc_data = copy.deepcopy(self.bc_data)

                for a in range(self.n_workers):

                    # Updata data knowledge
                    if n == 0:
                        X = self.X[a]
                        Y = self.Y[a]
                        self.X_train[a] = self.X[a][:]
                        self.Y_train[a] = self.Y[a][:]
                    else:
                        self.X[a].append(self._next_query[a])
                        self.Y[a].append(self.objective(self._next_query[a]))
                        self.X_train[a].append(self._next_query[a])
                        self.Y_train[a].append(self.objective(self._next_query[a]))

                        X = self.X[a]
                        Y = self.Y[a]
                        for transmitter in range(self.n_workers):
                            for (x,y) in self._prev_bc_data[transmitter][a]:
                                X = np.append(X,x).reshape(-1, self._dim)
                                Y = np.append(Y,y).reshape(-1, 1)
                                self.X_train[a].append(x)
                                self.Y_train[a].append(y)

                    # Standardize
                    Y = self.scaler[a].fit_transform(np.array(Y).reshape(-1, 1))
                    # Fit surrogate
                    self.model[a].fit(X, Y)

                    # Find next query
                    random_search *= max(int(self.n_workers/10),1) #higher for more workers
                    # print("random search number", random_search)
                    x = self._find_next_query(n, a, random_search)
                    self._next_query[a] = x

                    # In case of a "duplicate", randomly sample next query point.
                    # if np.any(np.abs(x - self.model[a].X_train_) <= 10**(-7)):
                    #     x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], self.domain.shape[0])

                    # Broadcast data to neighbours
                    self._broadcast(a,x,self.objective(x))

                # Calculate regret
                self._simple_regret[run,n] = self._regret(np.max([y_max for y_a in self.Y_train for y_max in y_a]))
                self._simple_cumulative_regret[run, n] = self._regret(np.max([y_max for y_a in self.Y_train for y_max in y_a]))
                # Calculate distance traveled
                if not n:
                    self._distance_traveled[run,n] = 0
                else:
                    self._distance_traveled[run,n] =  self._distance_traveled[run,n-1] + sum([np.linalg.norm(self.X[a][-2] - self.X[a][-1]) for a in range(self.n_workers)])

                # Plot optimization step
                if self._record_step:
                    self._plot_iteration(n, plot)

        self.pre_arg_max = []
        self.pre_max = []
        for a in range(self.n_workers):
            self.pre_arg_max.append(np.array(self.model[a].y_train_).argmax())
            self.pre_max.append(self.model[a].X_train_[np.array(self.model[a].y_train_).argmax()])

        # Compute and plot regret
        iter, r_mean, r_conf95 = self._mean_regret()
        self._plot_regret(iter, r_mean, r_conf95)
        iter, r_cum_mean, r_cum_conf95 = self._cumulative_regret()
        self._plot_regret(iter, r_cum_mean, r_cum_conf95, reward_type='cumulative')

        iter, d_mean, d_conf95 = self._mean_distance_traveled()

        # Save data
        self._save_data(data = [iter, r_mean, r_conf95, d_mean, d_conf95, r_cum_mean, r_cum_conf95], name = 'data')

        # Generate gif
        if plot and n_runs == 1:
            self._generate_gif(n_iters, plot)

    def _broadcast(self, agent, x, y):
        for neighbour_agent, neighbour in enumerate(self.network[agent]):
            if neighbour and neighbour_agent != agent:
                self.bc_data[agent][neighbour_agent].append((x,y))
        return

    def _plot_iteration(self, iter, plot_iter):
        """
        Plots the surrogate and acquisition function.
        """
        mu = []
        std = []
        for a in range(self.n_workers):
            try:
                mu_a, std_a = self.model[a].predict(self._grid, return_std=True)
            except:
                print("Cannot predict while plotting iteration")
                return
            mu.append(mu_a)
            std.append(std_a)
            acq = [-1 * self._acquisition_evaluations[a][iter//plot_iter] for a in range(self.n_workers)]

        for a in range(self.n_workers):
            mu[a] = self.scaler[a].inverse_transform(mu[a].reshape(-1, 1))
            std[a] = self.scaler[a].scale_ * std[a]

        if self._dim == 1:
            self._plot_1d(iter, mu, std, acq)
        elif self._dim == 2:
            self._plot_2d(iter, mu, acq)
        else:
            print("Can't plot for higher dimensional problems.")

    def _plot_1d(self, iter, mu, std, acq):
        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]
        if self.n_workers == 1:
            rgba = ['k']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10), sharex=True)

        class ScalarFormatterForceFormat(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.2f"
        fmt = ScalarFormatterForceFormat()
        fmt.set_powerlimits((0,0))
        fmt.useMathText = True

        #Objective function
        y_obj = [self.objective(i) for i in self._grid]
        ax1.plot(self._grid, y_obj, 'k--', lw=1)
        for a in range(self.n_workers):
            # Surrogate plot
            ax1.plot(self._grid, mu[a], color = rgba[a], lw=1)
            ax1.fill_between(np.squeeze(self._grid), np.squeeze(mu[a]) - 2*std[a], np.squeeze(mu[a]) + 2*std[a], color = rgba[a], alpha=0.1)
            ax1.scatter(self.X[a], self.Y[a], color = rgba[a], s=20, zorder=3)
            ax1.yaxis.set_major_formatter(fmt)
            ax1.set_ylim(bottom = -10, top=14)
            ax1.set_xticks(np.linspace(self._grid[0],self._grid[-1], 5))
            # Acquisition function plot
            ax2.plot(self._grid, acq[a], color = rgba[a], lw=1)
            ax2.axvline(self._next_query[a], color = rgba[a], lw=1)
            ax2.set_xlabel("x", fontsize = 16)
            ax2.yaxis.set_major_formatter(fmt)
            ax2.set_xticks(np.linspace(self._grid[0],self._grid[-1], 5))

        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.tick_params(axis='both', which='minor', labelsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='both', which='minor', labelsize=16)
        ax1.yaxis.offsetText.set_fontsize(16)
        ax2.yaxis.offsetText.set_fontsize(16)

        # Legends
        if self.n_workers > 1:
            c = 'k'
        else:
            c = rgba[a]
        legend_elements1 = [Line2D([0], [0], linestyle = '--', color='k', lw=0.8, label='Objective'),
                           Line2D([0], [0], color=c, lw=0.8, label='Surrogate'),
                           Line2D([], [], marker='o', color=c, label='Observations', markerfacecolor=c, markersize=4)]
        leg1 = ax1.legend(handles=legend_elements1, fontsize = 16, loc='upper right', fancybox=True, framealpha=0.2)
        ax1.add_artist(leg1)
        ax1.legend(["Iteration %d" % (iter)], fontsize = 16, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)


        legend_elements2 = [ Line2D([0], [0], color=c, lw=0.8, label='Acquisition'),
                            Line2D([], [], color=c, marker='|', linestyle='None',
                          markersize=10, markeredgewidth=1, label='Next Query')]
        ax2.legend(handles=legend_elements2, fontsize = 16, loc='upper right', fancybox=True, framealpha=0.5)

        plt.tight_layout()
        plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d.pdf' % (iter), bbox_inches='tight')
        plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d.png' % (iter), bbox_inches='tight')

    def _plot_2d(self, iter, mu, acq):

        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]
        if self.n_workers == 1:
            rgba = ['k']

        class ScalarFormatterForceFormat(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.2f"
        fmt = ScalarFormatterForceFormat()
        fmt.set_powerlimits((0,0))
        fmt.useMathText = True

        x = np.array(self.X)
        y = np.array(self.Y)

        first_param_grid = np.linspace(self.domain[0,0], self.domain[0,1], self._grid_density)
        second_param_grid = np.linspace(self.domain[1,0], self.domain[1,1], self._grid_density)
        X, Y = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')

        for a in range(self.n_workers):

            fig, ax = plt.subplots(1, 3, figsize=(10,4), sharey=True) # , sharex=True
            (ax1, ax2, ax3) = ax
            plt.setp(ax.flat, aspect=1.0, adjustable='box')

            N = 100
            # Objective plot
            Y_obj = [self.objective(i) for i in self._grid]
            clev1 = np.linspace(min(Y_obj), max(Y_obj),N)
            cp1 = ax1.contourf(X, Y, np.array(Y_obj).reshape(X.shape), clev1,  cmap = cm.coolwarm)
            for c in cp1.collections:
                c.set_edgecolor("face")
            cbar1 = plt.colorbar(cp1, ax=ax1, shrink = 0.9, format=fmt, pad = 0.05, location='bottom')
            cbar1.ax.tick_params(labelsize=10)
            cbar1.ax.locator_params(nbins=5)
            ax1.autoscale(False)
            ax1.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            ax1.axvline(self._next_query[a][0], color='k', linewidth=1)
            ax1.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax1.set_ylabel("y", fontsize = 10, rotation=0)
            leg1 = ax1.legend(['Objective'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax1.add_artist(leg1)
            ax1.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax1.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax1.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax1.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax1.get_yticklabels()[0], visible=False)
            ax1.tick_params(axis='both', which='both', labelsize=10)
            ax1.scatter(self.arg_max[:,0], self.arg_max[:,1], marker='x', c='gold', s=30)

            if self.n_workers > 1:
                ax1.legend(["Iteration %d" % (iter), "Agent %d" % (a)], fontsize = 10, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            else:
                ax1.legend(["Iteration %d" % (iter)], fontsize = 10, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)

            # Surrogate plot
            d = 0
            if mu[a].reshape(X.shape).max() - mu[a].reshape(X.shape).min() == 0:
                d = acq[a].reshape(X.shape).max()*0.1
            clev2 = np.linspace(mu[a].reshape(X.shape).min() - d, mu[a].reshape(X.shape).max() + d,N)
            cp2 = ax2.contourf(X, Y, mu[a].reshape(X.shape), clev2,  cmap = cm.coolwarm)
            for c in cp2.collections:
                c.set_edgecolor("face")
            cbar2 = plt.colorbar(cp2, ax=ax2, shrink = 0.9, format=fmt, pad = 0.05, location='bottom')
            cbar2.ax.tick_params(labelsize=10)
            cbar2.ax.locator_params(nbins=5)
            ax2.autoscale(False)
            ax2.scatter(x[a][:, 0], x[a][:, 1], zorder=1, color = rgba[a], s = 10)
            if self._acquisition_function in ['es', 'ucb', 'ts_ucb']:
                ax2.scatter(self.amaxucb[0, 0], self.amaxucb[0, 1], marker='o', c='red', s=30)
            ax2.axvline(self._next_query[a][0], color='k', linewidth=1)
            ax2.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax2.set_ylabel("y", fontsize = 10, rotation=0)
            ax2.legend(['Surrogate'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax2.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax2.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax2.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax2.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            # plt.setp(ax2.get_yticklabels()[0], visible=False)
            # plt.setp(ax2.get_yticklabels()[-1], visible=False)
            ax2.tick_params(axis='both', which='both', labelsize=10)

            # Broadcasted data
            for transmitter in range(self.n_workers):
                x_bc = []
                for (xbc,ybc) in self._prev_bc_data[transmitter][a]:
                    x_bc = np.append(x_bc,xbc).reshape(-1, self._dim)
                x_bc = np.array(x_bc)
                if x_bc.shape[0]>0:
                    ax1.scatter(x_bc[:, 0], x_bc[:, 1], zorder=1, color = rgba[transmitter], s = 10)
                    ax2.scatter(x_bc[:, 0], x_bc[:, 1], zorder=1, color = rgba[transmitter], s = 10)

            # Acquisition function contour plot
            d = 0
            if acq[a].reshape(X.shape).max() - acq[a].reshape(X.shape).min() == 0.0:
                d = acq[a].reshape(X.shape).max()*0.1
                d = 10**(-100)
            clev3 = np.linspace(acq[a].reshape(X.shape).min() - d, acq[a].reshape(X.shape).max() + d,N)
            cp3 = ax3.contourf(X, Y, acq[a].reshape(X.shape), clev3, cmap = cm.coolwarm)
            cbar3 = plt.colorbar(cp3, ax=ax3, shrink = 0.9, format=fmt, pad = 0.05, location='bottom')
            for c in cp3.collections:
                c.set_edgecolor("face")
            cbar3.ax.locator_params(nbins=5)
            cbar3.ax.tick_params(labelsize=10)
            ax3.autoscale(False)
            ax3.axvline(self._next_query[a][0], color='k', linewidth=1)
            ax3.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax3.set_xlabel("x", fontsize = 10)
            ax3.set_ylabel("y", fontsize = 10, rotation=0)
            ax3.legend(['Acquisition'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax3.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax3.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax3.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax3.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            # plt.setp(ax3.get_yticklabels()[-1], visible=False)
            ax3.tick_params(axis='both', which='both', labelsize=10)

            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax1.tick_params(axis='both', which='minor', labelsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.tick_params(axis='both', which='minor', labelsize=10)
            ax3.tick_params(axis='both', which='major', labelsize=10)
            ax3.tick_params(axis='both', which='minor', labelsize=10)
            ax1.yaxis.offsetText.set_fontsize(10)
            ax2.yaxis.offsetText.set_fontsize(10)
            ax3.yaxis.offsetText.set_fontsize(10)

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d_agent_%d.pdf' % (iter, a), bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (iter, a), bbox_inches='tight')

    def _plot_regret(self, iter, r_mean, conf95, reward_type='instant', log = False):

        use_log_scale = max(r_mean)/min(r_mean) > 10 if reward_type == 'instant' else False

        if not use_log_scale:
            # absolut error for linear scale
            lower = [r + err for r, err in zip(r_mean, conf95)]
            upper = [r - err for r, err in zip(r_mean, conf95)]
        else:
            # relative error for log scale
            lower = [10**(np.log10(r) + (0.434*err/r)) for r, err in zip(r_mean, conf95)]
            upper = [10**(np.log10(r) - (0.434*err/r)) for r, err in zip(r_mean, conf95)]

        fig = plt.figure()

        if use_log_scale:
            plt.yscale('log')

        plt.plot(iter, r_mean, '-', linewidth=1)
        plt.fill_between(iter, upper, lower, alpha=0.3)
        plt.xlabel('iterations')
        plt.ylabel(reward_type + ' regret')
        plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        if use_log_scale:
            plt.savefig(self._PDF_DIR_ + '/' + reward_type + '_regret_log.pdf', bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/' + reward_type + '_regret_log.png', bbox_inches='tight')
        else:
            plt.savefig(self._PDF_DIR_ + '/' + reward_type + '_regret.pdf', bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/' + reward_type + '_regret.png', bbox_inches='tight')

    def _generate_gif(self, n_iters, plot):
        if self._dim == 1:
            plots = []
            for i in range(n_iters+1):
                if plot is True or i == n_iters:
                    try:
                        plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d.png' % (i)))
                    except: pass
                elif not i % plot:
                    try:
                        plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d.png' % (i)))
                    except: pass
            imageio.mimsave(self._GIF_DIR_ + '/bo.gif', plots, duration=1.0)
        else:
            for a in range(self.n_workers):
                plots = []
                for i in range(n_iters+1):
                    if plot is True or i == n_iters:
                        try:
                            plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (i, a)))
                        except: pass
                    elif not i % plot:
                        try:
                            plots.append(imageio.imread(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d.png' % (i, a)))
                        except: pass
                imageio.mimsave(self._GIF_DIR_ + '/bo_agent_%d.gif' % (a), plots, duration=1.0)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TorchGPModel():
    def __init__(self, X, Y, alg_name=None, n_workers = None, objective_name = None):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        # self.likelihood.noise = 1e-4
        # self.likelihood.noise = 1e-3
        self.likelihood.noise = 1e-6
        # if (alg_name == 'BUCB' and n_workers >= 20):
        #     print("alg name", alg_name)
        #     self.likelihood.noise = 1e-3 #this is to deal with BUCB having issues with sigma being too small when n_workers is 20.
        # if objective_name == 'bird':
        #     print("alg name", alg_name)
        #     print("objective_name", objective_name)
        #     self.likelihood.noise = 1e-3 #this is to deal with bird having issues sometimes.
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        print("GP INITIAL NOISE", self.likelihood.noise)
        self.model = ExactGPModel(X, Y, self.likelihood)
        print("self module length scale", self.model.covar_module.base_kernel.lengthscale.item())
        print("objective name", objective_name)
        if objective_name == 'rastrigin':
            self.model.covar_module.base_kernel.lengthscale = 0.1
            print("self module length scale if rastrigin", self.model.covar_module.base_kernel.lengthscale.item())
            # self.train()
        # print("X", X)
        # print("Y", Y)
        # self.train()


    # def train(self, train_iters = 10):
    #     # Use the adam optimizer
    #     optimizer = torch.optim.Adam(
    #         self.model.parameters(), lr=0.01
    #     )  # Does not includes GaussianLikelihood parameters, since we set those to be untrainable

    #     # print("self model params", self.model.parameters)
    #     l1loss = torch.nn.L1Loss()
    #     # "Loss" for GPs - the marginal log likelihood
    #     # self.model.likelihood.noise = 0.2 #set noise 
    #     # lengthscales = []
    #     # noises = []
    #     for i in range(train_iters):
    #         # lengthscales.append(self.model.covar_module.base_kernel.lengthscale.item())
    #         # noises.append(self.model.likelihood.noise.item())
    #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
    #         # Zero gradients from previous iteration
    #         optimizer.zero_grad()
    #         self.model.eval()
    #         # self.likelihood.eval()
    #         self.model.train()
    #         # self.likelihood.train()
    #         # Output from model
    #         output = self.model(self.model.train_inputs[0])
    #         # Calc loss and backprop gradients
    #         loss = -mll(output, self.model.train_targets)
    #         loss.backward()
    #         optimizer.step()
    #     print("self module length scale (after training)if rastrigin", self.model.covar_module.base_kernel.lengthscale.item())

    # def train(self):
    #     self.model.train()
    #     self.likelihood.train()


    def fit(self, X, Y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.tensor(Y).float()
        if len(X.shape) == 2:
            X = X
        if len(Y.shape) == 2:
            Y = torch.reshape(Y, [-1, ])
        # try:
        self.model.set_train_data(X, Y, strict=False)
        # except:
        #     self.__init__(X, Y, likelihood)

    def predict(self, X, return_std= False, return_cov = False, return_tensor=False):
        self.model.eval()
        self.likelihood.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).float()
        if len(X.shape) == 1:
            X = torch.reshape(X, [1, -1])
        with gpytorch.settings.fast_pred_var():
            f_pred = self.model(X)
            if return_tensor:
                if return_std:
                    return f_pred.mean, f_pred.variance
                elif return_cov:
                    return f_pred.mean, f_pred.covariance_matrix
                else:
                    return f_pred.mean
            else:
                if return_std:
                    return f_pred.mean.detach().numpy(), f_pred.variance.detach().numpy()
                elif return_cov:
                    return f_pred.mean.detach().numpy(), f_pred.covariance_matrix.detach().numpy()
                else:
                    return f_pred.mean.detach().numpy()

    def sample_y(self, X, n_samples, random_state = None):
        rng = check_random_state(random_state)
        # print(" I am here")
        # print(" X shaoe", X.shape)
        y_mean, y_cov = self.predict(X, return_cov=True)
        # print("y_mean")
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples

    @property
    def y_train_(self):
        return self.model.train_targets.detach().numpy()

    @property
    def X_train_(self):
      return self.model.train_inputs[0].detach().numpy()


class BayesianOptimizationCentralized(bayesian_optimization):
    def __init__(self, objective, domain, arg_max = None, n_workers = 1,
                 network = None, kernel = kernels.RBF(), alpha=10**(-10),
                 acquisition_function = 'ei', policy = 'greedy', fantasies = 0,
                 epsilon = 0.01, regularization = None, regularization_strength = None,
                 pending_regularization = None, pending_regularization_strength = None,
                 grid_density = 100, x0=None, n_pre_samples=5, n_ysamples = 1, args=dict()):

        super(BayesianOptimizationCentralized, self).__init__(objective, domain=domain, arg_max=arg_max, n_workers=n_workers,
                 network=network, kernel=kernel, alpha=alpha,
                 acquisition_function=acquisition_function, policy = policy, fantasies = fantasies,
                 epsilon = epsilon, regularization = regularization, regularization_strength = regularization_strength,
                 pending_regularization = pending_regularization, pending_regularization_strength = pending_regularization_strength,
                 grid_density = grid_density, n_ysamples = n_ysamples, args=args)
        assert self.args.decision_type == 'parallel' or self.n_workers == 1
        self.diversity_penalty = args.diversity_penalty
        self.radius = args.div_radius
        self.acq_name = None
        if acquisition_function == 'es':
            self._acquisition_function = self._entropy_search_grad
        elif acquisition_function == 'bucb' or acquisition_function == 'ucbpe':
            self._acquisition_function = self._batch_upper_confidential_bound
            self.acq_name = acquisition_function
        # elif acquisition_function == 'ei' and fantasies == self.n_workers:
        #     self._acquisition_function = self._expected_improvement_fantasized
        elif acquisition_function == 'ei': #not sure why need fantasies == self.n_workers
            self._acquisition_function = self._expected_improvement_fantasized
        elif acquisition_function == 'ts':
            self._acquisition_function = self._thompson_sampling_centralized
        elif acquisition_function == 'sp':
            self._acquisition_function = self._stochastic_policy_centralized
        elif acquisition_function == "ts_ucb":
            self._acquisition_function = self._TS_UCB
        elif acquisition_function == "ts_ucb_seq":
            self._acquisition_function = self._TS_UCB_seq
        elif acquisition_function == "ts_ucb_det":
            self._acquisition_function = self._TS_UCB_det
        elif acquisition_function == "ts_ucb_vals":
            self._acquisition_function = self._TS_UCB_vals
        elif acquisition_function == "ts_rsr":
            self._acquisition_function = self._TS_RSR
        elif acquisition_function == "ts_rsr_mod":
            self._acquisition_function = self._TS_RSR_mod
        elif acquisition_function == "ts_ucb_mod":
            self._acquisition_function = self._TS_UCB_mod
        elif acquisition_function == "ts_rsr_combine":
            self._acquisition_function = self._TS_RSR_combine
        else:
            print('Supported acquisition functions: ei, ts, es, bucb, ucbpe, ts_ucb(mod), ts_ucb_seq, ts_ucb_det,ts_ucb_vals, ts_rsr(mod),ts_rsr_combine')
            return




    def _entropy_search_grad(self, a, x, n, radius=0.1):
        """
                Entropy search acquisition function.
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)
        # x = x.float()
        # self.beta = 3 - 0.019 * n
        delta = 0.01
        self.beta = 0.1 * 2. * np.log(x.shape[0] * n**2 * np.pi**2/(6. * delta)) #compare to the Desaultels paper 

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain

        mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        sigma = torch.sqrt(sigma)
        ucb = mu + self.beta * sigma
        # ucb = mu + self.beta * torch.sqrt(sigma)
        amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        init_x = np.random.normal(amaxucb, 1.0, (self.n_workers, self.domain.shape[0]))

        x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
        # print("INITIAL x diffable", x.grad)
        optimizer = torch.optim.Adam([x], lr=0.01)
        # training_iter = 200
        training_iter = 200
        best_loss = 1e6
        best_x = copy.deepcopy(x)
        for i in range(training_iter):
            # if i % 50 == 0:
            #     print("I am", i)
            #     print("x at iter %d: " %i, x)
            optimizer.zero_grad()
            joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
            # joint_x = joint_x.float()
            # x = x.float()
            cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
            cov_x_x = self.model.predict(x, return_cov=True, return_tensor=True)[1]
            if self.diversity_penalty:
                penalty = []
                for i in itertools.combinations(range(self.n_workers - 1), 2):
                    penalty.append(torch.clip(- 1./100. * torch.log(torch.norm(x[i[0]] - x[i[1]]) - self.radius), 0., torch.inf))
                loss = -torch.matmul(torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb) + sum(penalty)
            else:
                loss = -torch.matmul(
                    torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb)
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
            if torch.all(x == torch.clamp(x, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
                # print("x_opt for alg %d" %k, x_opt)
                x = torch.clamp(x, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
                x = torch.tensor(x, requires_grad=True,dtype=torch.float32)
                optimizer = torch.optim.Adam([x], lr=0.01)
                optimizer.zero_grad()
                # print("im here, projection happend: ", loss)
                # print("im here", n,i)
            else:
                pass
                # # print("x requires grad iter %d" %i, x.requires_grad)
                # # x.detach_() #remove detach
                # # print("x requires grad iter %d after detach" %i, x.requires_grad)
                # if loss < best_loss:
                #     best_x = x
                #     best_loss = loss.detach_()
                # #     print("im here, loss is: ", loss)
                # #     print("best loss is (iter %d" %i, best_loss)
                # #     print("x requires grad iter %d after best_x" %i, x.requires_grad)
                # # print("no projection, loss here: ", loss)
        return x.clone().detach().numpy()

    def _entropy_search_grad_with_issues(self, a, x, n, radius=0.1):
        """
                Entropy search acquisition function.
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)
        # self.beta = 1. + 0.2 * n
        self.beta = 3 - 0.019 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain

        mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        ucb = mu + self.beta * sigma
        # ucb = mu + self.beta * torch.sqrt(sigma) #try this?
        amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        # init_x = np.random.normal(amaxucb, 1.0, (self.n_workers, self.domain.shape[0]))
        init_x = np.random.normal(amaxucb, 1.0, (self.n_workers, self.domain.shape[0]))
        # init_x = init_x.float()

        x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
        # x = torch.tensor(init_x, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=0.01)
        # optimizer = torch.optim.SGD([x], lr=0.01)
        training_iter = 200
        # training_iter = 2
        for i in range(training_iter):
            if i % 50 == 0:
                print("I am", i)
                print("x at iter %d: " %i, x)
            # x = torch.tensor(x, requires_grad=True,dtype=torch.float32)
            # optimizer = torch.optim.Adam([x], lr=0.01)
            # optimizer = torch.optim.SGD([x], lr=0.01)
            # print("x at inner iter %d" %i, x)
            optimizer.zero_grad()
            joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
            # joint_x = joint_x.float()
            # x = x.float()
            cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
            cov_x_x = self.model.predict(x, return_cov=True, return_tensor=True)[1]
            if self.diversity_penalty:
                penalty = []
                for i in itertools.combinations(range(self.n_workers - 1), 2):
                    penalty.append(torch.clip(-1./100. * torch.log(torch.norm(x[i[0]] - x[i[1]]) - self.radius), 0., torch.inf))
                loss = -torch.matmul(torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb) + sum(penalty)
            else:
                loss = -torch.matmul(
                    torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb)
            # print("loss at inner iter %d" %i, loss)
            loss.backward()
            # print("loss grad times lr at %d" %i, x.grad * 0.01)
            # print("before step", x)
            optimizer.step()
            # print("is leaf", x.is_leaf)
            # print("after step", x)
            # project back to domain
            x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
            x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
            # print("after project", x)
            # print("loss grad at %d, after project" %i, x.grad)
            x.detach_()
        return x.clone().detach().numpy()


    def _TS_UCB_mod(self, a, x, n, radius=0.1,n_random = 200, n_restarts = 1500):
        """
                TS-UCB
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
            fstar_hat[j] /= self.n_ysamples
        fstar_hat = torch.tensor(fstar_hat)
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)

        self.beta = 1. + 0.2 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain
        # print(self.model.predict(x, return_std=True, return_tensor=True))
        # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        # print("sigma", sigma)
        # ucb = mu + self.beta * sigma
        # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        # self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        training_iter = 1
        best_loss = 1e6
        for k in range(n_restarts):
            init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x], lr=0.1)
            optimizer.zero_grad()
            for i in range(training_iter):
                # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
                # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
                x = x.float()
                # print(x.dtype, "x dtype")
                # print("hi")
                mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=True)
                if self.diversity_penalty:
                    penalty = []
                    for i in itertools.combinations(range(self.n_workers - 1), 2):
                        penalty.append(torch.clip(- 1./100. * torch.log(torch.norm(x[i[0]] - x[i[1]]) - self.radius), 0., torch.inf))
                    loss = -torch.matmul(torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb) + sum(penalty)
                else:
                    if self.n_workers == 1:
                        loss = torch.log((fstar_hat - mu_x)) - torch.log(torch.sqrt(cov_x_x))
                    else:
                        loss_num = torch.sum(fstar_hat - mu_x)
                        loss_den = 0
                        for j in range(self.n_workers):
                            # cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
                            # cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
                            # cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
                            # cov_j_new = cov_x_x[j,j] - torch.matmul(
                            # torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 0.01 * torch.eye(len(cov_x_x_sub)))), 
                            # cov_xj_x_no_j.T)
                            # loss_den += torch.sqrt(cov_j_new)
                            cov_x_x_sub = cov_x_x[:j,:j]
                            cov_xj_x_upto_j = cov_x_x[j,:j]
                            cov_j_new = cov_x_x[j,j] - torch.matmul(
                            torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                            cov_xj_x_upto_j.T)
                            # loss_den += torch.sqrt(cov_j_new)
                            loss_den += cov_j_new
                        # print("loss", loss)
                        loss = torch.log(loss_num) - torch.log(torch.sqrt(loss_den))
                    # print("loss at inner iter %d" %i, loss)
                loss.backward()
                optimizer.step()
                # project back to domain
                x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
                x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
                x.detach_()
            if loss < best_loss:
                # print(loss, "iter %d" %k)
                best_x = x
                best_loss = loss.detach_()
                # print(best_loss, "best loss after iter %d" %k)
        return best_x.clone().detach().numpy()

    def _TS_UCB(self, a, x, n, radius=0.1,n_random = 200, n_restarts = 3000):
        """
                TS-UCB
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
            fstar_hat[j] /= self.n_ysamples
        fstar_hat = torch.tensor(fstar_hat)
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)

        self.beta = 1. + 0.2 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain
        # print(self.model.predict(x, return_std=True, return_tensor=True))
        # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        # print("sigma", sigma)
        # ucb = mu + self.beta * sigma
        # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        # self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        training_iter = 1
        best_loss = 1e6
        for k in range(n_restarts):
            init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x], lr=0.1)
            optimizer.zero_grad()
            for i in range(training_iter):
                # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
                # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
                x = x.float()
                # print(x.dtype, "x dtype")
                # print("hi")
                mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=True)
                if self.diversity_penalty:
                    penalty = []
                    for i in itertools.combinations(range(self.n_workers - 1), 2):
                        penalty.append(torch.clip(- 1./100. * torch.log(torch.norm(x[i[0]] - x[i[1]]) - self.radius), 0., torch.inf))
                    loss = -torch.matmul(torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb) + sum(penalty)
                else:
                    if self.n_workers == 1:
                        loss = torch.log((fstar_hat - mu_x)) - torch.log(torch.sqrt(cov_x_x))
                    else:
                        loss_num = torch.sum(fstar_hat - mu_x)
                        loss_den = 0
                        for j in range(self.n_workers):
                            # cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
                            # cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
                            # cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
                            # cov_j_new = cov_x_x[j,j] - torch.matmul(
                            # torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 0.01 * torch.eye(len(cov_x_x_sub)))), 
                            # cov_xj_x_no_j.T)
                            # loss_den += torch.sqrt(cov_j_new)
                            cov_x_x_sub = cov_x_x[:j,:j]
                            cov_xj_x_upto_j = cov_x_x[j,:j]
                            cov_j_new = cov_x_x[j,j] - torch.matmul(
                            torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                            cov_xj_x_upto_j.T)
                            loss_den += torch.sqrt(cov_j_new)
                            # loss_den += cov_j_new
                        # print("loss", loss)
                        loss = torch.log(loss_num) - torch.log(loss_den)
                    # print("loss at inner iter %d" %i, loss)
                loss.backward()
                optimizer.step()
                # project back to domain
                x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
                x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
                x.detach_()
            if loss < best_loss:
                # print(loss, "iter %d" %k)
                best_x = x
                best_loss = loss.detach_()
                # print(best_loss, "best loss after iter %d" %k)
        return best_x.clone().detach().numpy()



    def _TS_RSR_mod(self, a, x, n, radius=0.1,n_random = 200, n_restarts = 1500):
        """
                TS-RSR
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
            fstar_hat[j] /= self.n_ysamples
        fstar_hat = torch.tensor(fstar_hat)
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)

        self.beta = 1. + 0.2 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain
        # print(self.model.predict(x, return_std=True, return_tensor=True))
        # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        # print("sigma", sigma)
        # ucb = mu + self.beta * sigma
        # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        # self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        training_iter = 1
        best_loss = 1e6
        for k in range(n_restarts):
            init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x], lr=0.1)
            optimizer.zero_grad()
            for i in range(training_iter):
                # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
                # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
                x = x.float()
                # print(x.dtype, "x dtype")
                # print("hi")
                mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=True)
                if self.diversity_penalty:
                    penalty = []
                    for i in itertools.combinations(range(self.n_workers - 1), 2):
                        penalty.append(torch.clip(- 1./100. * torch.log(torch.norm(x[i[0]] - x[i[1]]) - self.radius), 0., torch.inf))
                    loss = -torch.matmul(torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb) + sum(penalty)
                else:
                    if self.n_workers == 1:
                        loss = torch.log((fstar_hat[0] - mu_x)) - torch.log(torch.sqrt(cov_x_x))
                    else:
                        loss_num = torch.sum(fstar_hat - mu_x)
                        loss_den = 0
                        for j in range(self.n_workers):
                            cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
                            cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
                            cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
                            cov_j_new = cov_x_x[j,j] - torch.matmul(
                            torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                            cov_xj_x_no_j.T)
                            # loss_den += torch.sqrt(cov_j_new)
                            loss_den += cov_j_new
                            # cov_x_x_sub = cov_x_x[:j,:j]
                            # cov_xj_x_upto_j = cov_x_x[j,:j]
                            # cov_j_new = cov_x_x[j,j] - torch.matmul(
                            # torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                            # cov_xj_x_upto_j.T)
                            # loss_den += torch.sqrt(cov_j_new)
                        # print("loss", loss)
                        loss = torch.log(loss_num) - torch.log(torch.sqrt(loss_den))
                    # print("loss at inner iter %d" %i, loss)
                loss.backward()
                optimizer.step()
                # project back to domain
                x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
                x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
                x.detach_()
            if loss < best_loss:
                # print(loss, "iter %d" %k)
                best_x = x
                best_loss = loss.detach_()
        print("best_loss:%.1f" % (best_loss))
        return best_x.clone().detach().numpy()




    def _TS_RSR(self, a, x, n, radius=0.1,n_random = 200, n_restarts = 3000):
        """
                TS-RSR
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
            fstar_hat[j] /= self.n_ysamples
        fstar_hat = torch.tensor(fstar_hat)
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)

        self.beta = 1. + 0.2 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain
        # print(self.model.predict(x, return_std=True, return_tensor=True))
        # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        # print("sigma", sigma)
        # ucb = mu + self.beta * sigma
        # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        # self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        training_iter = 1
        best_loss = 1e6
        for k in range(n_restarts):
            init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x], lr=0.1)
            optimizer.zero_grad()
            for i in range(training_iter):
                # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
                # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
                x = x.float()
                # print(x.dtype, "x dtype")
                # print("hi")
                mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=True)
                if self.diversity_penalty:
                    penalty = []
                    for i in itertools.combinations(range(self.n_workers - 1), 2):
                        penalty.append(torch.clip(- 1./100. * torch.log(torch.norm(x[i[0]] - x[i[1]]) - self.radius), 0., torch.inf))
                    loss = -torch.matmul(torch.matmul(cov_x_xucb.T, torch.linalg.inv(cov_x_x + 0.01 * torch.eye(len(cov_x_x)))), cov_x_xucb) + sum(penalty)
                else:
                    if self.n_workers == 1:
                        loss = torch.log((fstar_hat[0] - mu_x)) - torch.log(torch.sqrt(cov_x_x))
                    else:
                        loss_num = torch.sum(fstar_hat - mu_x)
                        loss_den = 0
                        for j in range(self.n_workers):
                            cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
                            cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
                            cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
                            cov_j_new = cov_x_x[j,j] - torch.matmul(
                            torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                            cov_xj_x_no_j.T)
                            loss_den += torch.sqrt(cov_j_new)
                            # loss_den = cov_j_new
                            # cov_x_x_sub = cov_x_x[:j,:j]
                            # cov_xj_x_upto_j = cov_x_x[j,:j]
                            # cov_j_new = cov_x_x[j,j] - torch.matmul(
                            # torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                            # cov_xj_x_upto_j.T)
                            # loss_den += torch.sqrt(cov_j_new)
                        # print("loss", loss)
                        loss = torch.log(loss_num) - torch.log(loss_den)
                    # print("loss at inner iter %d" %i, loss)
                loss.backward()
                optimizer.step()
                # project back to domain
                x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
                x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
                x.detach_()
            if loss < best_loss:
                # print(loss, "iter %d" %k)
                best_x = x
                best_loss = loss.detach_()
                # print(best_loss, "best loss after iter %d" %k)
        return best_x.clone().detach().numpy()



    def _ts_rsr_sampling(self, a, x, n,n_random = 200,n_ysamples = 10):
        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples)
        # print(samples.shape, "shape samples")
        fstar_hat = 0
        for i in range(self.n_ysamples):
            fstar_hat += np.max(samples[:, i])
        fstar_hat = np.mean(fstar_hat)

        # print("x shape here", x.shape)
        model = copy.deepcopy(self.model)
        mu_x, cov_x_x = model.predict(x, return_cov =True, return_tensor=False)

        ts_ucb = (fstar_hat - mu_x)/np.sqrt(np.diag(cov_x_x)) #a proxy for ts_ucb
        # for i in range(self.n_workers):
        #     queries.append(self._boltzmann(n, x, acq))
        # print("shape", ts_ucb.shape)
        C = max(abs(max(ts_ucb) - ts_ucb))
        if C > 10 ** (-2):
            beta = 3 * np.log(n + self._initial_data_size + 1) / C
            _blotzmann_prob = lambda e: np.exp(beta * e)
            bm = [_blotzmann_prob(e) for e in ts_ucb]
            norm_bm = [float(i) / sum(bm) for i in bm]
            idxes = np.random.choice(range(x.shape[0]), p=np.squeeze(norm_bm), size=(self.n_workers,))
        else:
            idxes = np.random.choice(range(x.shape[0]), size=(self.n_workers,))
        queries = [x[idx] for idx in idxes]

        return np.array(queries)


    # def _TS_RSR_combine(self, a, x, n, radius=0.1,n_random = 200, n_restarts = 100):
    #     """
    #             TS-RSR
    #             Args:
    #                 a: # agents
    #                 x: array-like, shape = [n_samples, n_hyperparams]
    #                 n: agent nums
    #                 projection: if project to a close circle
    #                 radius: circle of the projected circle
    #             """

    #     x = x.reshape(-1, self._dim)

    #     Y_max = np.max(self.model.y_train_)
    #     x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
    #     x_random = x_random.reshape(-1,self._dim)
    #     samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
    #     # print(samples.shape, "shape samples")
    #     fstar_hat = np.zeros(self.n_workers)
    #     for j in np.arange(self.n_workers):
    #         for i in range(self.n_ysamples):
    #             fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
    #         fstar_hat[j] /= self.n_ysamples
    #     fstar_hat = torch.tensor(fstar_hat)
    #     # print("f star hat", fstar_hat)
    #     # print("Y_max", Y_max)
    #     # print("x shape here", x.shape,a,n)

    #     # self.model.eval()
    #     # self.likelihood.eval()
    #     domain = np.empty((self.n_workers, self.domain.shape[0], self.domain.shape[1]))
    #     for i in np.arange(self.n_workers):
    #         domain[i,:,:] = self.domain
    #     # print("domain shape", domain.shape)
    #     # print(self.model.predict(x, return_std=True, return_tensor=True))
    #     # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
    #     # print("sigma", sigma)
    #     # ucb = mu + self.beta * sigma
    #     # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
    #     # self.amaxucb = amaxucb
    #     # x = np.vstack([amaxucb for _ in range(self.n_workers)])
    #     # training_iter = 1
    #     best_loss = 1e6
    #     best_x = self._ts_rsr_sampling(a,x,n) #set first best_x to always be something
    #     final_alg_idx = -1
    #     best_inner_iter = -1
    #     acq_fns = [self._ts_rsr_sampling]
    #     n_acq = len(acq_fns)
    #     for k in range(n_acq):
    #         # init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
    #         acq_fn = acq_fns[k]
    #             # print("this is k", k)
    #         for i in range(n_restarts):
    #             # print("nrestarts", n_restarts)
    #             # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
    #             # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
    #             x_guess = acq_fn(a,x,n)
    #             # x_guess = x_guess.float()
    #             mu_x, cov_x_x = self.model.predict(x_guess, return_cov =True, return_tensor=True)
    #             if self.n_workers == 1:
    #                 loss = torch.log((fstar_hat[0] - mu_x)) - torch.log(torch.sqrt(cov_x_x))
    #             else:
    #                 loss_num = torch.sum(fstar_hat - mu_x)
    #                 loss_den = 0
    #                 for j in range(self.n_workers):
    #                     cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
    #                     cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
    #                     cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
    #                     cov_j_new = cov_x_x[j,j] - torch.matmul(
    #                     torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
    #                     cov_xj_x_no_j.T)
    #                     loss_den += cov_j_new
    #                 loss = torch.log(loss_num) - 0.5*torch.log(loss_den) #0.5 corresponds to sqrt
    #             if loss < best_loss:
    #                 best_x = x_guess
    #                 best_loss = loss
    #                 final_alg_idx = k
    #                 best_inner_iter = i
    #                 # print(best_loss, "best loss after iter %d" %i)
    #     print("final alg idx:%d, inner iter %d, best_loss:%.1f" % (final_alg_idx, best_inner_iter,best_loss))
    #     return best_x



    # def _TS_RSR_combine(self, a, x, n, radius=0.1,n_random = 200, n_restarts = 10):
    #     """
    #             TS-RSR
    #             Args:
    #                 a: # agents
    #                 x: array-like, shape = [n_samples, n_hyperparams]
    #                 n: agent nums
    #                 projection: if project to a close circle
    #                 radius: circle of the projected circle
    #             """

    #     x = x.reshape(-1, self._dim)

    #     Y_max = np.max(self.model.y_train_)
    #     x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
    #     x_random = x_random.reshape(-1,self._dim)
    #     samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
    #     # print(samples.shape, "shape samples")
    #     fstar_hat = np.zeros(self.n_workers)
    #     for j in np.arange(self.n_workers):
    #         for i in range(self.n_ysamples):
    #             fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
    #         fstar_hat[j] /= self.n_ysamples
    #     fstar_hat = torch.tensor(fstar_hat)
    #     # print("f star hat", fstar_hat)
    #     # print("Y_max", Y_max)


    #     self.beta = 1. + 0.2 * n

    #     # self.model.eval()
    #     # self.likelihood.eval()
    #     domain = np.empty((self.n_workers, self.domain.shape[0], self.domain.shape[1]))
    #     for i in np.arange(self.n_workers):
    #         domain[i,:,:] = self.domain
    #     # print("domain shape", domain.shape)
    #     # print(self.model.predict(x, return_std=True, return_tensor=True))
    #     # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
    #     # print("sigma", sigma)
    #     # ucb = mu + self.beta * sigma
    #     # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
    #     # self.amaxucb = amaxucb
    #     # x = np.vstack([amaxucb for _ in range(self.n_workers)])
    #     training_iter = 1
    #     best_loss = 1e6
    #     acq_fns = [self._thompson_sampling_centralized,self._stochastic_policy_centralized,self._batch_ucb_no_ucbpe, self._batch_ucbpe]
    #     n_acq = len(acq_fns)
    #     best_x = acq_fns[0](a,x,n) #set first best_x to always be something
    #     final_alg_idx = -1
    #     best_inner_iter = -1
    #     for k in range(n_acq):
    #         # init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
    #         acq_fn = acq_fns[k]
    #         if k == 2 : #bucb (position 2)/ucbpe does not need restart
    #             n_restarts = 1 
    #             # print("this is k", k)
    #         for i in range(n_restarts):
    #             # print("nrestarts", n_restarts)
    #             # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
    #             # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
    #             x_guess = acq_fn(a,x,n)
    #             # x_guess = x_guess.float()
    #             mu_x, cov_x_x = self.model.predict(x_guess, return_cov =True, return_tensor=True)
    #             if self.n_workers == 1:
    #                 loss = torch.log((fstar_hat[0] - mu_x)) - torch.log(torch.sqrt(cov_x_x))
    #             else:
    #                 loss_num = torch.sum(fstar_hat - mu_x)
    #                 loss_den = 0
    #                 for j in range(self.n_workers):
    #                     cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
    #                     cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
    #                     cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
    #                     cov_j_new = cov_x_x[j,j] - torch.matmul(
    #                     torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
    #                     cov_xj_x_no_j.T)
    #                     loss_den += cov_j_new
    #                 loss = torch.log(loss_num) - 0.5*torch.log(loss_den) #0.5 corresponds to sqrt
    #             if loss < best_loss:
    #                 best_x = x_guess
    #                 best_loss = loss
    #                 final_alg_idx = k
    #                 best_inner_iter = i
    #                 # print(best_loss, "best loss after iter %d" %i)
    #     print("final alg idx:%d, inner iter %d, best_loss:%.1f" % (final_alg_idx, best_inner_iter,best_loss))
    #     return best_x




    def _TS_RSR_combine(self, a, x, n, radius=0.1,n_random = 1500, n_restarts = 1):
        """
                TS-RSR
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)

        # Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        # samples = self.model.sample_y(x, n_samples=self.n_ysamples * self.n_workers) #try this
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
                # print("max value candidate %d for worker %d" %(i,j), np.max(samples[:, j*self.n_ysamples + i]))
                # print("samples", samples.shape)
            fstar_hat[j] /= self.n_ysamples
            # print("f star hat %d" %j, fstar_hat[j])
        fstar_hat = torch.tensor(fstar_hat)
        fstar_hat = torch.max(fstar_hat)
        # print("final fstar_hat", fstar_hat)
        mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=False)
        # print("mu_x shape", mu_x.shape)
        # print("max mu_x", np.max(mu_x))
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)
        std_x = np.sqrt(np.diag(cov_x_x))
        # print("min sigma at iter %d" %n, std_x[np.argmin(std_x)])
        # print("GP NOISE at iter %d" %n, self.model.likelihood.noise)

        self.beta = 1. + 0.2 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = np.empty((self.n_workers, self.domain.shape[0], self.domain.shape[1]))
        for i in np.arange(self.n_workers):
            domain[i,:,:] = self.domain
        # print("domain shape", domain.shape)
        # print(self.model.predict(x, return_std=True, return_tensor=True))
        # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        # print("sigma", sigma)
        # ucb = mu + self.beta * sigma
        # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        # self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        training_iter = 1
        best_loss = 1e6
        # acq_fns = [self._batch_ucb_no_ucbpe,self._thompson_sampling_centralized,self._stochastic_policy_centralized, self._batch_ucbpe]
        # acq_fns = [self._thompson_sampling_centralized,self._stochastic_policy_centralized, self._batch_ucbpe]
        # acq_fns = [self._stochastic_policy_centralized, self._batch_ucbpe]
        # acq_fns = [self._ts_rsr_sampling, self._stochastic_policy_centralized, self._batch_ucbpe]
        acq_fns = [self._batch_ucb_no_ucbpe]
        n_acq = len(acq_fns)
        best_x = torch.tensor(acq_fns[0](a,x,n)) #set first best_x to always be something
        final_alg_idx = -1
        best_inner_iter = -1
        for k in range(n_acq):
            # init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            acq_fn = acq_fns[k]
            init_x = acq_fn(a,x,n)
            print("x output at try %d (iter %d)" % (k,n), init_x)
            x_opt = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x_opt], lr=0.01)
            optimizer.zero_grad()
            for i in range(training_iter):
                # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
                # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
                # print(x.dtype, "x dtype")
                # print("hi")
                mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=False)
                std_x = np.sqrt(np.diag(cov_x_x))
                print("min sigma at iter %d, inner iter %d" %(n,k), std_x[np.argmin(std_x)])
                # for l in np.arange(x_opt.shape[0]):
                    # for j in np.arange(x.shape[0]):
                    #     if np.linalg.norm(x[j,:] - x_opt[l,:].clone().detach().numpy()) <= 1e-7:
                    #         print("x row: ", x[j,:])
                    #         print("x_opt", x_opt[l,:].clone().detach().numpy())
                    # print("is x_opt l row in x?: ", 
                    #     any((x[:]==x_opt[l,:].clone().detach().numpy()).all(1)))
                x_opt = x_opt.float()
                mu_x, cov_x_x = self.model.predict(x_opt, return_cov =True, return_tensor=True)
                # print("max Y observed", np.max(self.model.y_train_))
                print("max mu_x now", torch.max(mu_x))
                if self.n_workers == 1:
                    loss = (fstar_hat - mu_x)/(torch.sqrt(cov_x_x))
                else:
                    # loss_num = torch.sum(fstar_hat - mu_x)
                    loss_num = fstar_hat - mu_x
                    # loss_den = 0
                    loss_den = torch.empty(self.n_workers)
                    for j in range(self.n_workers):
                        cov_x_x_sub = torch.cat((cov_x_x[:j,:], cov_x_x[j+1:,:]), dim = 0)
                        cov_x_x_sub = torch.cat((cov_x_x_sub[:,:j], cov_x_x_sub[:,j+1:]), dim = 1)
                        cov_xj_x_no_j = torch.cat((cov_x_x[j,:j], cov_x_x[j,j+1:]))
                        cov_j_new = cov_x_x[j,j] - torch.matmul(
                        torch.matmul(cov_xj_x_no_j, torch.linalg.inv(cov_x_x_sub + 1e-6 * torch.eye(len(cov_x_x_sub)))), 
                        cov_xj_x_no_j.T)
                        # loss_den += cov_j_new
                        loss_den[j] = torch.sqrt(cov_j_new)
                        print("new cov of x_j", cov_j_new)
                        print("old cov of x_j", cov_x_x[j,j])
                        # loss_den += torch.sqrt(cov_j_new)
                        # loss_den = cov_j_new
                        # cov_x_x_sub = cov_x_x[:j,:j]
                        # cov_xj_x_upto_j = cov_x_x[j,:j]
                        # cov_j_new = cov_x_x[j,j] - torch.matmul(
                        # torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 1e-4 * torch.eye(len(cov_x_x_sub)))), 
                        # cov_xj_x_upto_j.T)
                        # loss_den += torch.sqrt(cov_j_new)
                    # print("loss", loss)
                    # loss = torch.log(loss_num) - 0.5*torch.log(loss_den) #0.5 corresponds to sqrt
                    loss = torch.sum(loss_num/loss_den)
                    # loss = loss_num/loss_den
                    print("loss here_num for alg %d" %k, loss_num)
                    print("loss here_den for %d" %k, loss_den)
                    print("mu_x here", mu_x)
                    print("f star hat", fstar_hat)
                # print(loss_num, "loss num")
                # print(loss_den, 'loss den')
                # print("loss at inner iter %d" %i, loss)
                loss.backward()
                optimizer.step()
                # # project back to domain
                # x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
                # x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
                # with torch.no_grad():
                #     print("x before",x)
                #     print("limit", torch.tensor(domain[:,:, 0]), torch.tensor(domain[:,:, 1]))
                #     torch.clamp(x, torch.tensor(domain[:,:, 0]),torch.tensor(domain[:,:, 1]))
                #     print("x after",x)
                print("alg %d" %k, loss)
                if torch.all(x_opt == torch.clamp(x_opt, torch.tensor(domain[:,:, 0]),torch.tensor(domain[:,:, 1]))) == False:
                    # print("x_opt for alg %d" %k, x_opt)
                    x_opt = torch.clamp(x_opt, torch.tensor(domain[:,:, 0]),torch.tensor(domain[:,:, 1]))
                    x_opt = torch.tensor(x_opt, requires_grad=True,dtype=torch.float32)
                    optimizer = torch.optim.Adam([x_opt], lr=0.01)
                    optimizer.zero_grad()
                    # print("im here", n,i)
                else:
                    x_opt.detach_()
                    if loss < best_loss:
                        best_x = x_opt
                        best_loss = loss.detach_()
                        final_alg_idx = k
                        best_inner_iter = i
                        # print(best_loss, "best loss after iter %d" %i)
            # if loss < best_loss:
            #     # print("bool", torch.all(torch.clamp(x_opt, torch.tensor(domain[:,:, 0]),torch.tensor(domain[:,:, 1])) <= 5.0))
            #     if torch.all(x_opt == torch.clamp(x_opt, torch.tensor(domain[:,:, 0]),torch.tensor(domain[:,:, 1]))): #test if x is in domain
            #         best_x = x_opt
            #         best_loss = loss.detach_()
            #         final_alg_idx = k
            #         best_inner_iter = i
                # print(best_loss, "best loss after iter %d" %k)
        print("final alg idx:%d, inner iter %d, best_loss:%.1f" % (final_alg_idx, best_inner_iter,best_loss))
        return best_x.clone().detach().numpy()



    # def _TS_UCB_det(self, a, x, n, radius=0.1,n_random = 200, n_samples = 200, n_restarts = 500):
    #     """
    #             TS-UCB-det 
    #             Args:
    #                 a: # agents
    #                 x: array-like, shape = [n_samples, n_hyperparams]
    #                 n: agent nums
    #                 projection: if project to a close circle
    #                 radius: circle of the projected circle
    #             """

    #     x = x.reshape(-1, self._dim)

    #     Y_max = np.max(self.model.y_train_)
    #     x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
    #     x_random = x_random.reshape(-1,self._dim)
    #     samples = self.model.sample_y(x_random, n_samples=n_samples)
    #     # print(samples.shape, "shape samples")
    #     fstar_hat = 0.0
    #     for i in range(n_samples):
    #         fstar_hat += np.max(samples[:,i])
    #     fstar_hat /= n_samples
    #     # print("f star hat", fstar_hat)
    #     # print("Y_max", Y_max)

    #     self.beta = 1. + 0.2 * n

    #     # self.model.eval()
    #     # self.likelihood.eval()
    #     domain = self.domain
    #     # print(self.model.predict(x, return_std=True, return_tensor=True))
    #     # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
    #     # print("sigma", sigma)
    #     # ucb = mu + self.beta * sigma
    #     # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
    #     # self.amaxucb = amaxucb
    #     # x = np.vstack([amaxucb for _ in range(self.n_workers)])
    #     training_iter = 1
    #     best_loss = 1e6
    #     for k in range(n_restarts):
    #         init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
    #         x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
    #         optimizer = torch.optim.Adam([x], lr=0.1)
    #         optimizer.zero_grad()
    #         for i in range(training_iter):
    #             # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
    #             # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
    #             x = x.float()
    #             # print(x.dtype, "x dtype")
    #             # print("hi")
    #             mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=True)
    #             loss = torch.log(torch.sum(fstar_hat - mu_x)) - torch.log(torch.sqrt(torch.linalg.det(cov_x_x)))
    #             loss.backward()
    #             optimizer.step()
    #             # project back to domain
    #             x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
    #             x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
    #             x.detach_()
    #         if loss < best_loss:
    #             # print(loss, "iter %d" %k)
    #             best_x = x
    #             best_loss = loss.detach_()
    #             # print(best_loss, "best loss after iter %d" %k)
    #     return best_x.clone().detach().numpy()


    def _TS_UCB_vals(self, a, x, n, radius=0.1,n_random = 200,n_restarts = 3000):
        """
                TS-UCB-vals
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)

        Y_max = np.max(self.model.y_train_)
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
            fstar_hat[j] /= self.n_ysamples
        fstar_hat = torch.tensor(fstar_hat)
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)

        self.beta = 1. + 0.2 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain
        # print(self.model.predict(x, return_std=True, return_tensor=True))
        # mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        # print("sigma", sigma)
        # ucb = mu + self.beta * sigma
        # amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        # self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        training_iter = 1
        best_loss = 1e6
        for k in range(n_restarts):
            init_x = np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x], lr=0.1)
            optimizer.zero_grad()
            for i in range(training_iter):
                # joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
                # cov_x_xucb = self.model.predict(joint_x, return_cov=True, return_tensor=True)[1][-1, :-1].reshape([-1,1])
                x = x.float()
                # print(x.dtype, "x dtype")
                # print("hi")
                mu_x, cov_x_x = self.model.predict(x, return_cov =True, return_tensor=True)
                # cov_x_x_updated = cov_x_x - torch.matmul(
                #             torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 0.01 * torch.eye(len(cov_x_x_sub)))), 
                #             cov_xj_x_upto_j.T)
                loss = torch.log(torch.sum(fstar_hat - mu_x)) - torch.log(
                    torch.sum(torch.sqrt(torch.diag(cov_x_x)))) #TRY THIS NEW VERSION
                loss.backward()
                optimizer.step()
                # project back to domain
                x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
                x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
                x.detach_()
            if loss < best_loss:
                # print(loss, "iter %d" %k)
                best_x = x
                best_loss = loss.detach_()
                # print(best_loss, "best loss after iter %d" %k)
        return best_x.clone().detach().numpy()






    #try new_variant where TS changes at each iter
    def _TS_UCB_seq(self, a, x, n, radius=0.1,n_random = 1000,n_restarts = -1):
        #trying out n_samples = 1 for now
        """
                TS-UCB-seq
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)
        queries = []
        Y_max = np.max(self.model.y_train_)
        # n_random *= max(int(self.n_workers/10),1) 
        # # n_random *= max(int(self.n_workers/10),1) * 5 #testing for now
        # x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        # x_random = x_random.reshape(-1,self._dim)
        # samples = self.model.sample_y(x_random, n_samples=self.n_ysamples)
        # samples = self.model.sample_y(x, n_samples = self.n_ysamples) 
        l = 0
        eps = 1e-4
        while True: #this is to catch the bug in bird (but doesn't work yet)
            try:
                # print("before samplimg")
                # print("x.shape")
                samples = self.model.sample_y(x, n_samples = self.n_ysamples * self.n_workers)
                # print("after sampling")
            except:
                # l += 1
                # self.model.likelihood.noise += eps
                # print("samples failed for the %d-th time, new likelihood noise: " %l, self.model.likelihood.noise)
                print("alg threw up an error at sampling at iter %d" %n)
                return np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
            else: #if no exception
                break
        # samples = self.model.sample_y(x, n_samples = self.n_ysamples * self.n_workers)
        # print(samples.shape, "shape samples")
        # print("GP length_scale", self.model.model.covar_module.base_kernel.lengthscale.item())


        # fstar_hat = 0.0
        # for i in range(self.n_ysamples):
        #     fstar_hat += np.max(samples[:,i])
        # fstar_hat /= self.n_ysamples
        mu, sigma = self.model.predict(x,return_std=True)
        max_mu_in_x = np.max(mu)



        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(self.n_ysamples):
                fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
            fstar_hat[j] /= self.n_ysamples
            # print("fstar_hat (%d) - Y_max, iter %d " %(i, n), fstar_hat[j] - Y_max)
            # fstar_hat[j] = np.maximum(fstar_hat[j], Y_max + 1e-4)
        # fstar_hat = torch.tensor(fstar_hat)


        # fstar_hat = 0
        # for i in range(self.n_ysamples):
        #     fstar_hat = np.maximum(np.max(samples[:,i]), fstar_hat) #take max of fstar hat


        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)
        # print("self noise now (iter %d)" %n, self.model.likelihood.noise)



        mu,sigma = self.model.predict(x,return_std=True)
        max_mu = np.max(mu)
        min_sigma = np.sqrt(np.min(sigma))
        sigma_of_max_mu = np.sqrt(sigma[np.argmax(mu)])
        # print("sigma of max mu", sigma_of_max_mu)
        # print("Ymax now", Y_max)

        mu_max_x = np.max(mu)
        sigma_max_x = np.sqrt(sigma[np.argmax(mu)])
        y_target = np.empty(self.n_workers)
        for i in range(self.n_workers):
            if fstar_hat[i] < mu_max_x:
                while True:
                    # y = np.random.normal(mu_max_x, sigma_max_x)
                    samples = self.model.sample_y(x, n_samples = 1)
                    y = np.max(samples)
                    if y > mu_max_x:
                        y_target[i] = y
                        break
                fstar_hat[i] = y_target[i]
            # print("y target agent %d, iter %d" % (i,n), fstar_hat[i])
        fstar_hat = np.maximum(fstar_hat, y_target) #try this to see what happens
        # print("fstar_hat - max_mu, iter %d " %( n), fstar_hat - max_mu)
        # print("fstar_hat - Y_max, iter %d " %( n), fstar_hat - Y_max)
        # print("min sigma %d" %n, min_sigma)
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()
        # ts_ucb = (fstar_hat - mu)/np.sqrt(sigma)
        ts_ucb = (fstar_hat[0] - mu)/np.sqrt(sigma)
        query = x[np.argmin(ts_ucb)]
        # print("x shape here", x.shape)
        # fantasized_y = self.model.predict(query)[0]
        fantasized_y = 0.
        queries.append(query)
        # print("mu at ts usb argmin", mu[np.argmin(ts_ucb)])
        # print("sigma at ts_ucb argmin (iter %d)" %n, np.sqrt(sigma[np.argmin(ts_ucb)]))
        # print("this is ts_ucb argmin", np.argmin(ts_ucb))
        # print("this is X[0]", self.X[0])
        for i in range(self.n_workers- 1):
            fantasized_X.append(query)
            fantasized_Y.append(fantasized_y)
            fantastic_model = copy.deepcopy(self.model) #take care to separate fantasy model from self.model
            fantastic_model.fit(np.array(fantasized_X),np.array(fantasized_Y))
            try:
                _,sigma = fantastic_model.predict(x,return_std=True)
                # print("x dtype", x.dtype)
                # print("sigma shape here", sigma.shape)
                _,sigma_last = fantastic_model.predict(x[np.argmin(ts_ucb)],return_std=True)
                _,sigma_0 = fantastic_model.predict(np.asarray([self.X[0]]),return_std=True)
                # print("sigma last: ", np.sqrt(sigma_last))
                # print("sigma 0: ", np.sqrt(sigma_0))
            except:
                print("somehow failed")
                pass #use old sigma in this case.
            # print("sigma at ts_ucb argmin after training (iter %d), agent %d" % (n,i), np.sqrt(sigma[np.argmin(ts_ucb)]))
            # print("this is ts_ucb argmin (after training)", np.argmin(ts_ucb))

            # #resample f_star_hat using fantasized model
            # samples = fantastic_model.sample_y(x_random, n_samples=self.n_ysamples)
            # # print(samples.shape, "shape samples")
            # fstar_hat = 0.0
            # for i in range(self.n_ysamples):
            #     fstar_hat += np.max(samples[:,i])
            # fstar_hat /= self.n_ysamples

            # ts_ucb = (fstar_hat - mu)/np.sqrt(sigma)
            ts_ucb = (fstar_hat[i+1] - mu)/np.sqrt(sigma)
            # print(" (before optimizing) min ts_ucb, iter %d, outer iter %d" % (i,n), np.min(ts_ucb))
            # print("sigma of min ts_ucb, iter %d, outer iter %d " %(i,n), np.sqrt(sigma[np.argmin(ts_ucb)]))
            # print("num of min ts_ucb, iter %d, outer iter %d" % (i,n), (fstar_hat - mu)[np.argmin(ts_ucb)])
            # print("mu of min ts_ucb", mu[np.argmin(ts_ucb)])
            # print("mu of a neighbor of min ts_ucb", self.model.predict(x[np.argmin(ts_ucb)] + np.asarray([1e-4,1e-4])))
            # print("x of min ts_ucb, iter %d, outer iter %d" % (i,n), x[np.argmin(ts_ucb)])
            # print("min sigma across all x %d, outer iter %d" %(i,n), np.sqrt(sigma[np.argmin(sigma)]))
            # print(fstar_hat,
            #         np.max(mu),
            #         sigma[np.argmin(ts_ucb)],
            #     "min ts_ucb numerator fstar_hat, mu and denominator std")


            init_x = copy.deepcopy(x[np.argmin(ts_ucb)])
            # init_x = np.random.normal(x[np.argmin(ts_ucb)], 0.2)
            x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            # x0 = x0.float()
            optimizer = torch.optim.Adam([x0], lr=0.01)
            optimizer.zero_grad()
            best_loss = torch.tensor(ts_ucb[np.argmin(ts_ucb)])
            best_x = torch.tensor(x[np.argmin(ts_ucb)])
            # iters = 10
            # iters = 10
            iters = 10
            # iters =0 
            # if self.objective_name == 'bird': #bird covariance sometimes gets too small, see if this fixes things
            #     iters = 1
            for k in np.arange(iters):
                optimizer.zero_grad()
                mu_x= self.model.predict(x0,return_tensor=True)
                joint_x = torch.vstack((torch.tensor(queries).float(),x0))
                _, cov_x_x = self.model.predict(joint_x, return_cov = True, return_tensor = True)
                print("I am here iter %d" %n)
                print("cov_x0 at opt iter %d, agent %d, iter %d" %(k,i,n), cov_x_x[-1,-1])
                cov_x0_updated = cov_x_x[-1,-1] - torch.matmul(
                            torch.matmul(cov_x_x[-1,:-1], torch.linalg.inv(cov_x_x[:-1,:-1] + 1e-6 * torch.eye(len(cov_x_x[:-1,:-1])))), 
                            cov_x_x[-1,:-1].T)
                print("cov_x0 updated at opt iter %d, agent %d, iter %d" %(k,i,n), cov_x0_updated)
                print("(after) I am here iter %d" %n) #NO PROBLEM WITH THIS CODE
                # loss = (fstar_hat - mu_x)/torch.sqrt(cov_x0_updated) #TRY THIS NEW VERSION
                print("cov_x0_updated value (iter %d), inner iter: %d" %(n,k), cov_x0_updated)
                try:
                    loss = (fstar_hat[i+1] - mu_x)/torch.sqrt(cov_x0_updated) #TRY THIS NEW VERSION
                    print("loss updated at opt iter %d, agent %d, iter %d" %(k,i,n), loss)
                    print("loss num updated at opt iter %d, agent %d, iter %d" %(k,i,n), (fstar_hat[i+1] - mu_x))
                except:
                    # print("cov_x0_updated value (actual value:%.6f) is probably nan (iter %d), inner iter: %d, just break" %(cov_x0_updated, n,k))
                    break
                # print("this is loss, x0, mu_x, sigma_x", loss, x0, mu_x,torch.sqrt(cov_x0_updated))
                # print("avg mu_x", np.mean(mu))
                loss.backward()
                # print("x0 grad at iter %d" %k, x0.grad)
                optimizer.step()
                # print("loss at opt iter %d, agent %d, iter %d" %(k,i,n), loss)
                # print("x0 at opt iter %d, agent %d, iter %d" %(k,i,n), x0)
                # project back to domain
                if  torch.all(x0 == torch.clamp(x0, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
                    # print("x_opt for alg %d" %k, x_opt)
                    x0 = torch.clamp(x0, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
                    x0 = torch.tensor(x0, requires_grad=True,dtype=torch.float32)
                    optimizer = torch.optim.Adam([x0], lr=0.01)
                    optimizer.zero_grad()
                    # print("im here", n,i)
                else:
                    if loss < best_loss and loss > 0:
                        best_loss = loss.clone().detach()
                        best_x = x0.detach().clone()
                    # # print("x0 requires grad iter %d" %k, x0.requires_grad)
                    # # x0.detach_()
                    # # print("x0 requires grad iter %d" %k, x0.requires_grad)
                    # if loss < best_loss and loss > 0:
                    #     best_x = x0.detach().clone() #please be careful with assignment, assignment or shallow copy will be bad!!
                    #     best_loss = loss.detach()
                    # elif loss < best_loss and loss < 0:
                    #     print("IM HERE (iter %d)" %n)
                    # elif loss < 0: #retry (but within the budget of 10 total gradient steps)
                    #     init_x = np.random.normal(x[np.argmin(ts_ucb)], 0.2)
 #     x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
                    #     # x0 = x0.float()
                    #     optimizer = torch.optim.Adam([x0], lr=0.01)
                    #     optimizer.zero_grad()
                    #     print("best_x and x0 same (by way of difference)? (shouldn't be)", best_x- x0)

            mu_x0, sigma_x0  = self.model.predict(x0.clone().detach(), return_std = True)
            sigma_x0 = np.sqrt(sigma_x0)
            loss_x0 = (fstar_hat[i+1] - mu_x0)/sigma_x0 #note the (i+1)!
            print("best loss is (agent %i), iter %d:" %(i,n), best_loss)
            print("loss_x0 is (agent %i), iter %d:" %(i,n), loss_x0)
            print("fstar_hat[%d+1] is" %i, fstar_hat[i+1] )
            print("mu_argmin_ts_ucb is ", mu[np.argmin(ts_ucb)])
            print("min_ts_ucb is (before GD) ", np.min(ts_ucb))
            print("sigma_argmin is", np.sqrt(sigma[np.argmin(ts_ucb)]))
            mu_x_opt, sigma_x_opt  = self.model.predict(np.asarray([-1.,-1.]), return_std = True)
            print("mu_opt is ", mu_x_opt)
            print("sigma_x_opt is", np.sqrt(sigma_x_opt))
            print("loss_opt is", (fstar_hat[i+1] - mu_x_opt)/np.sqrt(sigma_x_opt))
            print("mu_x0 is  ", mu_x0)
            print("sigma_x0 is ", sigma_x0)
            print("initial x0 is", x[np.argmin(ts_ucb)])
            print("initial mu_x is(agent %i), iter %d:" %(i,n), mu[np.argmin(ts_ucb)])
            print("initial sigma_x is(agent %i), iter %d:" %(i,n), np.sqrt(sigma[np.argmin(ts_ucb)]))

            query = best_x.clone().detach().numpy()
            # if loss_x0 < best_loss:
            #     # print("initial loss (agent %d, iter %d)" %(i,n), best_loss)
            #     query = x0.clone().detach().numpy()
            #     # print("loss now (agent %d, iter %d)" %(i,n), loss)
            # else:
            #     query = x[np.argmin(ts_ucb)]
                # print("best_loss still best after GD: agent %d, iter %d" %(i,n))
                # print("would have been x0", x0.clone().detach().numpy())
                # print("actual x0", query)
            # print("this is query (agent %d), iter %d" %(i,n), query)
            # query = x0.clone().detach().numpy()
            # print("x0 after update is(agent %i), iter %d:" %(i,n), query)
            # print("mu_x after update is(agent %i), iter %d:" %(i,n), mu_x0)
            # print("sigma_x after update is (agent %i), iter %d:" %(i,n), sigma_x0)

            # if loss_x0 < best_loss and loss_x0 > 0:
            #     query = x0.clone().detach().numpy()
            # else: 
            #     print("setting query to be argmin ts_ucb")
            #     query =  x[np.argmin(ts_ucb)]

            # best_x.detach() #does this do anything?
            # query = best_x.clone().detach().numpy()
            # print("loss of min_ts_ucb", ts_ucb[np.argmin(ts_ucb)])
            # print("best loss for agent %d at iter %d" %(i,n), best_loss)
            mu_best_x, sigma_x = self.model.predict(query, return_std = True)
            sigma_x = np.sqrt(sigma_x)
            # print("fsar_hat - best_x for agent %d at iter %d" %(i,n), fstar_hat[i] - mu_best_x)
            # print("sigma_x for agent %d at iter %d" %(i,n), sigma_x)
            # print(" loss for agent %d at iter %d after training" %(i,n), (fstar_hat[i] - mu_best_x)/sigma_x)
            # print("query at iter %d, worker %d: " %(n,i), query)
            if np.isnan(query[0]): #this happens quite rarely.
                print("had to do random query")
                query = np.random.normal(x[np.argmin(ts_ucb)], 1e-2) #just an arbitrary small neighborhood around the actual argmin
            queries.append(query)
            # mu_x, cov_x_x = fantastic_model.predict(x0, return_cov =True, return_tensor=False)
            # print(" (after optimizing) min ts_ucb, outer iter %d" % (n), best_loss)
            # print("sigma of min ts_ucb,outer iter %d " %(n), np.sqrt(cov_x_x))
            # print("num of min ts_ucb, outer iter %d" % (n), (fstar_hat - mu_x))
            # print("x of min ts_ucb, outer iter %d" % (n), query)
        return np.array(queries)


    # #try new_variant where TS changes at each iter
    # def _TS_UCB_seq(self, a, x, n, radius=0.1,n_random = 1000,n_restarts = -1):
    #     #trying out n_samples = 1 for now
    #     """
    #             TS-UCB-seq
    #             Args:
    #                 a: # agents
    #                 x: array-like, shape = [n_samples, n_hyperparams]
    #                 n: agent nums
    #                 projection: if project to a close circle
    #                 radius: circle of the projected circle
    #             """

    #     x = x.reshape(-1, self._dim)
    #     queries = []
    #     # Y_max = np.max(self.model.y_train_)
    #     # n_random *= max(int(self.n_workers/10),1) 
    #     # # n_random *= max(int(self.n_workers/10),1) * 5 #testing for now
    #     # x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
    #     # x_random = x_random.reshape(-1,self._dim)
    #     # samples = self.model.sample_y(x_random, n_samples=self.n_ysamples)
    #     # samples = self.model.sample_y(x, n_samples = self.n_ysamples) 
    #     l = 0
    #     eps = 1e-4
    #     while True: #this is to catch the bug in bird (but doesn't work yet)
    #         try:
    #             samples = self.model.sample_y(x, n_samples = self.n_ysamples * self.n_workers)
    #         except:
    #             # l += 1
    #             # self.model.likelihood.noise += eps
    #             # print("samples failed for the %d-th time, new likelihood noise: " %l, self.model.likelihood.noise)
    #             print("alg threw up an error at sampling at iter %d" %n)
    #             return np.random.uniform(self.domain[:,0], self.domain[:,1], (self.n_workers, self.domain.shape[0]))
    #         else: #if no exception
    #             break
    #     # samples = self.model.sample_y(x, n_samples = self.n_ysamples * self.n_workers)
    #     # print(samples.shape, "shape samples")
    #     # print("GP length_scale", self.model.model.covar_module.base_kernel.lengthscale.item())


    #     # fstar_hat = 0.0
    #     # for i in range(self.n_ysamples):
    #     #     fstar_hat += np.max(samples[:,i])
    #     # fstar_hat /= self.n_ysamples

    #     Y_max = np.max(self.model.y_train_)
    #     x_max = self.model.X_train_[np.argmax(self.model.y_train_)]
    #     print("x_max", x_max)
    #     mu_max_x,sigma_max_x = self.model.predict(x_max,return_std=True)
    #     print("actual predicted mu_max_x and sigma_max_x %d" %n, mu_max_x, np.sqrt(sigma_max_x))
    #     # print("sigma of max mu", sigma_of_max_mu)
    #     print("Ymax now", Y_max)

    #     fstar_hat = np.zeros(self.n_workers)
    #     for j in np.arange(self.n_workers):
    #         for i in range(self.n_ysamples):
    #             fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
    #         fstar_hat[j] /= self.n_ysamples
    #         # print("fstar_hat (%d) - Y_max, iter %d " %(i, n), fstar_hat[j] - Y_max)
    #         # fstar_hat[j] = np.maximum(fstar_hat[j], Y_max + 1e-4)
    #     # fstar_hat = torch.tensor(fstar_hat)


    #     # fstar_hat = 0
    #     # for i in range(self.n_ysamples):
    #     #     fstar_hat = np.maximum(np.max(samples[:,i]), fstar_hat) #take max of fstar hat


    #     # print("f star hat", fstar_hat)
    #     # print("Y_max", Y_max)
    #     # print("self noise now (iter %d)" %n, self.model.likelihood.noise)



    #     mu,sigma = self.model.predict(x,return_std=True)
    #     max_mu = np.max(mu)
    #     min_sigma = np.sqrt(np.min(sigma))
    #     sigma_of_max_mu = np.sqrt(sigma[np.argmax(mu)])
    #     print("sigma of max mu", sigma_of_max_mu)
    #     # print("Ymax now", Y_max)
    #     y_target = np.empty(self.n_workers)
    #     for i in range(self.n_workers):
    #         if fstar_hat[i] < max_mu:
    #             while True:
    #                 y = np.random.normal(max_mu, sigma_of_max_mu)
    #                 if y > max_mu:
    #                     y_target[i] = y
    #                     break
    #             fstar_hat[i] = y_target[i]
    #         print("y target agent %d, iter %d" % (i,n), fstar_hat[i])
    #     # fstar_hat = np.maximum(fstar_hat, y_target) #try this to see what happens
    #     # print("fstar_hat - max_mu, iter %d " %( n), fstar_hat - max_mu)
    #     # print("fstar_hat - Y_max, iter %d " %( n), fstar_hat - Y_max)
    #     # print("min sigma %d" %n, min_sigma)
    #     fantasized_X = self.X.copy()
    #     fantasized_Y = self.Y.copy()
    #     # ts_ucb = (fstar_hat - mu)/np.sqrt(sigma)
    #     ts_ucb = (fstar_hat[0] - mu)/np.sqrt(sigma)
    #     query = x[np.argmin(ts_ucb)]
    #     # fantasized_y = self.model.predict(query)[0]
    #     fantasized_y = 0.
    #     queries.append(query)
    #     # print("mu at ts usb argmin", mu[np.argmin(ts_ucb)])
    #     # print("sigma at ts_ucb argmin", np.sqrt(sigma[np.argmin(ts_ucb)]))
    #     for i in range(self.n_workers- 1):
    #         fantasized_X.append(query)
    #         fantasized_Y.append(fantasized_y)
    #         fantastic_model = copy.deepcopy(self.model) #take care to separate fantasy model from self.model
    #         fantastic_model.fit(np.array(fantasized_X),np.array(fantasized_Y))
    #         try:
    #             _,sigma = fantastic_model.predict(x,return_std=True)
    #         except:
    #             pass #use old sigma in this case.

    #         # #resample f_star_hat using fantasized model
    #         # samples = fantastic_model.sample_y(x_random, n_samples=self.n_ysamples)
    #         # # print(samples.shape, "shape samples")
    #         # fstar_hat = 0.0
    #         # for i in range(self.n_ysamples):
    #         #     fstar_hat += np.max(samples[:,i])
    #         # fstar_hat /= self.n_ysamples

    #         # ts_ucb = (fstar_hat - mu)/np.sqrt(sigma)
    #         ts_ucb = (fstar_hat[i+1] - mu)/np.sqrt(sigma)
    #         # print(" (before optimizing) min ts_ucb, iter %d, outer iter %d" % (i,n), np.min(ts_ucb))
    #         # print("sigma of min ts_ucb, iter %d, outer iter %d " %(i,n), np.sqrt(sigma[np.argmin(ts_ucb)]))
    #         # print("num of min ts_ucb, iter %d, outer iter %d" % (i,n), (fstar_hat - mu)[np.argmin(ts_ucb)])
    #         # print("mu of min ts_ucb", mu[np.argmin(ts_ucb)])
    #         # print("mu of a neighbor of min ts_ucb", self.model.predict(x[np.argmin(ts_ucb)] + np.asarray([1e-4,1e-4])))
    #         # print("x of min ts_ucb, iter %d, outer iter %d" % (i,n), x[np.argmin(ts_ucb)])
    #         # print("min sigma across all x %d, outer iter %d" %(i,n), np.sqrt(sigma[np.argmin(sigma)]))
    #         # print(fstar_hat,
    #         #         np.max(mu),
    #         #         sigma[np.argmin(ts_ucb)],
    #         #     "min ts_ucb numerator fstar_hat, mu and denominator std")


    #         init_x = copy.deepcopy(x[np.argmin(ts_ucb)])
    #         # init_x = np.random.normal(x[np.argmin(ts_ucb)], 0.2)
    #         x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
    #         # x0 = x0.float()
    #         optimizer = torch.optim.Adam([x0], lr=0.01)
    #         optimizer.zero_grad()
    #         best_loss = ts_ucb[np.argmin(ts_ucb)]
    #         best_x = torch.tensor(x[np.argmin(ts_ucb)])
    #         # iters = 10
            # iters = 10
            # iters = 20
    #         # iters =0 
    #         # if self.objective_name == 'bird': #bird covariance sometimes gets too small, see if this fixes things
    #         #     iters = 1
    #         for k in np.arange(iters):
    #             optimizer.zero_grad()
    #             mu_x= self.model.predict(x0,return_tensor=True)
    #             joint_x = torch.vstack((torch.tensor(queries).float(),x0))
    #             _, cov_x_x = self.model.predict(joint_x, return_cov = True, return_tensor = True)
    #             # print("I am here iter %d" %n)
    #             cov_x0_updated = cov_x_x[-1,-1] - torch.matmul(
    #                         torch.matmul(cov_x_x[-1,:-1], torch.linalg.inv(cov_x_x[:-1,:-1] + 1e-6 * torch.eye(len(cov_x_x[:-1,:-1])))), 
    #                         cov_x_x[-1,:-1].T)
    #             # print("(after) I am here iter %d" %n) #NO PROBLEM WITH THIS CODE
    #             # loss = (fstar_hat - mu_x)/torch.sqrt(cov_x0_updated) #TRY THIS NEW VERSION
    #             # print("cov_x0_updated value (iter %d), inner iter: %d" %(n,k), cov_x0_updated)
    #             try:
    #                 loss = (fstar_hat[i+1] - mu_x)/torch.sqrt(cov_x0_updated) #TRY THIS NEW VERSION
    #             except:
    #                 print("cov_x0_updated value (actual value:%.6f) is probably nan (iter %d), inner iter: %d, just break" %(cov_x0_updated, n,k))
    #                 break
    #             # print("this is loss, x0, mu_x, sigma_x", loss, x0, mu_x,torch.sqrt(cov_x0_updated))
    #             # print("avg mu_x", np.mean(mu))
    #             loss.backward()
    #             # print("x0 grad at iter %d" %k, x0.grad)
    #             optimizer.step()
    #             # print("loss at opt iter %d" %k, loss)
    #             # project back to domain
    #             if  torch.all(x0 == torch.clamp(x0, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
    #                 # print("x_opt for alg %d" %k, x_opt)
    #                 x0 = torch.clamp(x0, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
    #                 x0 = torch.tensor(x0, requires_grad=True,dtype=torch.float32)
    #                 optimizer = torch.optim.Adam([x0], lr=0.01)
    #                 optimizer.zero_grad()
    #                 # print("im here", n,i)
    #             else:
    #                 pass
    #                 # # print("x0 requires grad iter %d" %k, x0.requires_grad)
    #                 # # x0.detach_()
    #                 # # print("x0 requires grad iter %d" %k, x0.requires_grad)
    #                 # if loss < best_loss and loss > 0:
    #                 #     best_x = x0.detach().clone() #please be careful with assignment, assignment or shallow copy will be bad!!
    #                 #     best_loss = loss.detach()
    #                 # elif loss < best_loss and loss < 0:
    #                 #     print("IM HERE (iter %d)" %n)
    #                 # elif loss < 0: #retry (but within the budget of 10 total gradient steps)
    #                 #     init_x = np.random.normal(x[np.argmin(ts_ucb)], 0.2)
    #                 #     x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
    #                 #     # x0 = x0.float()
    #                 #     optimizer = torch.optim.Adam([x0], lr=0.01)
    #                 #     optimizer.zero_grad()
    #                 #     print("best_x and x0 same (by way of difference)? (shouldn't be)", best_x- x0)

    #         mu_x0, sigma_x0  = self.model.predict(x0.clone().detach(), return_std = True)
    #         sigma_x0 = np.sqrt(sigma_x0)
    #         loss_x0 = (fstar_hat[i+1] - mu_x0)/sigma_x0 #note the (i+1)!
    #         # print("best loss is (agent %i), iter %d:" %(i,n), best_loss)
    #         # print("loss_x0 is (agent %i), iter %d:" %(i,n), loss_x0)
    #         # print("initial x0 is", x[np.argmin(ts_ucb)])
    #         # print("initial mu_x is(agent %i), iter %d:" %(i,n), mu[np.argmin(ts_ucb)])
    #         # print("initial sigma_x is(agent %i), iter %d:" %(i,n), np.sqrt(sigma[np.argmin(ts_ucb)]))
    #         if loss_x0 < best_loss and loss > 0:
    #             print("initial loss (agent %d, iter %d)" %(i,n), best_loss)
    #             query = x0.clone().detach().numpy()
    #             print("loss now (agent %d, iter %d)" %(i,n), loss)
    #         else:
    #             query = x[np.argmin(ts_ucb)]
    #             print("best_loss still best after GD: agent %d, iter %d" %(i,n))
    #             print("would have been x0", x0.clone().detach().numpy())
    #             print("actual x0", query)
    #         # query = x0.clone().detach().numpy()
    #         # print("x0 after update is(agent %i), iter %d:" %(i,n), query)
    #         # print("mu_x after update is(agent %i), iter %d:" %(i,n), mu_x0)
    #         # print("sigma_x after update is (agent %i), iter %d:" %(i,n), sigma_x0)

    #         # if loss_x0 < best_loss and loss_x0 > 0:
    #         #     query = x0.clone().detach().numpy()
    #         # else: 
    #         #     print("setting query to be argmin ts_ucb")
    #         #     query =  x[np.argmin(ts_ucb)]

    #         # best_x.detach() #does this do anything?
    #         # query = best_x.clone().detach().numpy()
    #         # print("loss of min_ts_ucb", ts_ucb[np.argmin(ts_ucb)])
    #         # print("best loss for agent %d at iter %d" %(i,n), best_loss)
    #         mu_best_x, sigma_x = self.model.predict(query, return_std = True)
    #         sigma_x = np.sqrt(sigma_x)
    #         # print("fsar_hat - best_x for agent %d at iter %d" %(i,n), fstar_hat[i] - mu_best_x)
    #         # print("sigma_x for agent %d at iter %d" %(i,n), sigma_x)
    #         # print(" loss for agent %d at iter %d after training" %(i,n), (fstar_hat[i] - mu_best_x)/sigma_x)
    #         # print("query at iter %d, worker %d: " %(n,i), query)
    #         if np.isnan(query[0]): #this happens quite rarely.
    #             print("had to do random query")
    #             query = np.random.normal(x[np.argmin(ts_ucb)], 1e-2) #just an arbitrary small neighborhood around the actual argmin
    #         queries.append(query)
    #         # mu_x, cov_x_x = fantastic_model.predict(x0, return_cov =True, return_tensor=False)
    #         # print(" (after optimizing) min ts_ucb, outer iter %d" % (n), best_loss)
    #         # print("sigma of min ts_ucb,outer iter %d " %(n), np.sqrt(cov_x_x))
    #         # print("num of min ts_ucb, outer iter %d" % (n), (fstar_hat - mu_x))
    #         # print("x of min ts_ucb, outer iter %d" % (n), query)
    #     return np.array(queries)


##  f_star_hat does not get resampled per new agent
    # def _TS_UCB_seq_old(self, a, x, n, radius=0.1,n_random = 200,n_restarts = 300):
    #     #trying out n_samples = 1 for now
    #     """
    #             TS-UCB-seq
    #             Args:
    #                 a: # agents
    #                 x: array-like, shape = [n_samples, n_hyperparams]
    #                 n: agent nums
    #                 projection: if project to a close circle
    #                 radius: circle of the projected circle
    #             """

    #     x = x.reshape(-1, self._dim)
    #     queries = []
    #     # Y_max = np.max(self.model.y_train_)
    #     # n_random *= max(int(self.n_workers/10),1) 
    #     n_random *= max(int(self.n_workers/10),1) * 5 #testing for now
    #     x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
    #     x_random = x_random.reshape(-1,self._dim)
    #     samples = self.model.sample_y(x_random, n_samples=self.n_ysamples * self.n_workers)
    #     # print(samples.shape, "shape samples")
    #     fstar_hat = np.zeros(self.n_workers)
    #     for j in np.arange(self.n_workers):
    #         for i in range(self.n_ysamples):
    #             fstar_hat[j] += np.max(samples[:, j*self.n_ysamples + i])
    #         fstar_hat[j] /= self.n_ysamples
    #     fstar_hat = torch.tensor(fstar_hat)
    #     # print("f star hat", fstar_hat)
    #     # print("Y_max", Y_max)



    #     mu,sigma = self.model.predict(x,return_std=True)
    #     fantasized_X = self.X.copy()
    #     fantasized_Y = self.Y.copy()
    #     ts_ucb = (fstar_hat[0] - mu)/np.sqrt(sigma)
    #     fantasized_y = 0.
    #     query = x[np.argmin(ts_ucb)]
    #     queries.append(query)
    #     # print("mu at ts usb argmin", mu[np.argmin(ts_ucb)])
    #     # print("sigma at ts_ucb argmin", np.sqrt(sigma[np.argmin(ts_ucb)]))
    #     for i in range(self.n_workers- 1):
    #         fantasized_X.append(query)
    #         fantasized_Y.append(fantasized_y)
    #         self.model.fit(np.array(fantasized_X),np.array(fantasized_Y))
    #         _,sigma = self.model.predict(x,return_std=True)
    #         ts_ucb = (fstar_hat[i+1] - mu)/np.sqrt(sigma)
    #         # print(fstar_hat,
    #         #         np.max(mu),
    #         #         sigma[np.argmin(ts_ucb)],
    #         #     "min ts_ucb numerator fstar_hat, mu and denominator std")
    #         query = x[np.argmin(ts_ucb)]
    #         queries.append(query)
    #     return np.array(queries)



    def _TS_UCB_seq_1_f(self, a, x, n, radius=0.1,n_random = 200, n_samples = 1, n_restarts = 300):
        """
                TS-UCB-seq
                Args:
                    a: # agents
                    x: array-like, shape = [n_samples, n_hyperparams]
                    n: agent nums
                    projection: if project to a close circle
                    radius: circle of the projected circle
                """

        x = x.reshape(-1, self._dim)
        queries = []
        # Y_max = np.max(self.model.y_train_)
        # n_random *= max(int(self.n_workers/10),1) 
        n_random *= max(int(self.n_workers/10),1) * 5 #testing for now
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=n_samples)
        # print(samples.shape, "shape samples")
        fstar_hat = 0.0
        for i in range(n_samples):
            fstar_hat += np.max(samples[:,i])
        fstar_hat /= n_samples
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)



        mu,sigma = self.model.predict(x,return_std=True)
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()
        ts_ucb = (fstar_hat - mu)/np.sqrt(sigma)
        fantasized_y = 0.
        query = x[np.argmin(ts_ucb)]
        queries.append(query)
        # print("mu at ts usb argmin", mu[np.argmin(ts_ucb)])
        # print("sigma at ts_ucb argmin", np.sqrt(sigma[np.argmin(ts_ucb)]))
        for i in range(self.n_workers- 1):
            fantasized_X.append(query)
            fantasized_Y.append(fantasized_y)
            self.model.fit(np.array(fantasized_X),np.array(fantasized_Y))
            _,sigma = self.model.predict(x,return_std=True)
            ts_ucb = (fstar_hat - mu)/np.sqrt(sigma)
            # print(fstar_hat,
            #         np.max(mu),
            #         sigma[np.argmin(ts_ucb)],
            #     "min ts_ucb numerator fstar_hat, mu and denominator std")
            query = x[np.argmin(ts_ucb)]
            queries.append(query)
        return np.array(queries)





    def _batch_upper_confidential_bound(self, a, x, n):
        """
        Entropy search acquisition function.
        Args:
            a: # agents
            x: array-like, shape = [n_samples, n_hyperparams]
            model:
        """

        x = x.reshape(-1, self._dim)
        queries = []

        model = self.model

        # if self.beta is None:
        #     self.beta = 2.
        # self.beta = 1. + 0.01 * n
        # self.beta = 3 - 0.019 * n
        delta = 0.01
        self.beta = 0.1 * 2. * np.log(x.shape[0] * n**2 * np.pi**2/(6. * delta)) #compare to the Desaultels paper 
        # self.beta = 0.15 + 0.019 * n

        mu, sigma = model.predict(x, return_std=True)
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()
        # ucb = mu + self.beta * sigma
        ucb = mu + self.beta * np.sqrt(sigma)
        amaxucb = x[np.argmax(ucb)]
        fantasized_y = 0.  # fantasized y will not affect sigma
        query = amaxucb
        queries.append(query)
        self.amaxucb = amaxucb[np.newaxis, :]

        # print("GP noise iter %d" %n, self.model.likelihood.noise)
        # print("sigma of bucb, iter %d" %n, np.sqrt(sigma[np.argmax(ucb)]))
        for i in range(self.n_workers - 1):
            fantasized_X.append(query)
            fantasized_Y.append(fantasized_y)
            model.fit(np.array(fantasized_X), np.array(fantasized_Y))
            _,sigma_bucb = self.model.predict(amaxucb, return_std = True)
            # print("self.model predict bucb, iter %d (agent %d)" %(n,i), np.sqrt(sigma_bucb))
            _, sigma = model.predict(x, return_std=True)
            if self.acq_name == 'bucb':
                # ucb = mu + self.beta * sigma
                ucb = mu + self.beta * np.sqrt(sigma)
                # init_x = np.random.normal(x[np.argmax(ucb)], 0.2)
                init_x = copy.deepcopy(ucb)
                x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
                optimizer = torch.optim.Adam([x0], lr=0.01)
                optimizer.zero_grad()
                best_loss = -np.max(ucb)
                best_x = torch.tensor(x[np.argmax(ucb)])
                # iters = 10
                iters = 0
                for k in np.arange(iters):
                    optimizer.zero_grad()
                    mu_x, sigma_x = self.model.predict(x0, return_std = True, return_tensor = True)
                    # loss = -(mu_x + self.beta * sigma_x)
                    loss = -(mu_x + self.beta * torch.sqrt(sigma_x))
                    loss.backward()
                    optimizer.step()
                    if torch.all(x0 == torch.clamp(x0, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
                        # print("x_opt for alg %d" %k, x_opt)
                        x0 = torch.clamp(x0, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
                        x0 = torch.tensor(x0, requires_grad=True,dtype=torch.float32)
                        optimizer = torch.optim.Adam([x0], lr=0.01)
                        optimizer.zero_grad()
                        # print("im here", n,i)
                    else:
                        # print("x0 requires grad iter %d" %k, x0.requires_grad)
                        # x0.detach_()
                        # print("x0 requires grad iter %d" %k, x0.requires_grad)
                        if loss < best_loss:
                            best_x = x0
                            best_loss = loss.detach_()
                query = best_x.clone().detach().numpy()
                queries.append(query)
            elif self.acq_name == 'ucbpe':
                query = x[np.argmax(sigma)]
                queries.append(query)
        # print("queries shape", len(queries))
        return np.array(queries)




    def _batch_ucb_no_ucbpe(self, a, x, n):
        """
        Entropy search acquisition function.
        Args:
            a: # agents
            x: array-like, shape = [n_samples, n_hyperparams]
            model:
        """

        x = x.reshape(-1, self._dim)
        queries = []

        model = copy.deepcopy(self.model)

        # if self.beta is None:
        #     self.beta = 2.
        self.beta = 1. + 0.01 * n
        # self.beta = 0.15 + 0.019 * n

        mu, sigma = model.predict(x, return_std=True)
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()
        ucb = mu + self.beta * sigma
        amaxucb = x[np.argmax(ucb)]
        fantasized_y = 0.  # fantasized y will not affect sigma
        query = amaxucb
        queries.append(query)
        self.amaxucb = amaxucb[np.newaxis, :]


        print("max mu in bucb at %d" %n, np.max(mu))
        for i in range(self.n_workers - 1):
            fantasized_X.append(query)
            fantasized_Y.append(fantasized_y)
            model.fit(np.array(fantasized_X), np.array(fantasized_Y))
            _, sigma = model.predict(x, return_std=True)
            ucb = mu + self.beta * sigma
            query = x[np.argmax(ucb)]
            print("query sigma (%d)" %i, sigma[np.argmax(ucb)])
            print("bucb query %d" % i, query)
            print("query mu (%d)" %i, mu[np.argmax(ucb)])
            print("max sigma (%d)" %i, sigma[np.argmax(sigma)])
            print("mu at max sigma (%d)" %i, mu[np.argmax(sigma)])
            queries.append(query)
        return np.array(queries)



    def _batch_ucbpe(self, a, x, n):
        """
        Entropy search acquisition function.
        Args:
            a: # agents
            x: array-like, shape = [n_samples, n_hyperparams]
            model:
        """

        x = x.reshape(-1, self._dim)
        queries = []

        model = copy.deepcopy(self.model)

        # if self.beta is None:
        #     self.beta = 2.
        self.beta = 1. + 0.01 * n
        # self.beta = 0.15 + 0.019 * n

        mu, sigma = model.predict(x, return_std=True)
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()
        ucb = mu + self.beta * sigma
        amaxucb = x[np.argmax(ucb)]
        fantasized_y = 0.  # fantasized y will not affect sigma
        query = amaxucb
        queries.append(query)
        self.amaxucb = amaxucb[np.newaxis, :]

        for i in range(self.n_workers - 1):
            fantasized_X.append(query)
            fantasized_Y.append(fantasized_y)
            model.fit(np.array(fantasized_X), np.array(fantasized_Y))
            _, sigma = model.predict(x, return_std=True)
            query = x[np.argmax(sigma)]
            queries.append(query)
        return np.array(queries)



    def _thompson_sampling_centralized(self, a, x, n):
        x = x.reshape(-1, self._dim)
        queries = []

        model = copy.deepcopy(self.model)

        samples = model.sample_y(x, n_samples=self.n_workers)

        for i in range(self.n_workers):
            query = x[np.argmax(samples[:, i])]
            queries.append(query)

        return np.array(queries)

    def _stochastic_policy_centralized(self, a, x, n):
        x = x.reshape(-1, self._dim)
        # print("x shape", x.shape,a,n)
        model = copy.deepcopy(self.model)
        acq = model.predict(x)
        # for i in range(self.n_workers):
        #     queries.append(self._boltzmann(n, x, acq))
        C = max(abs(max(acq) - acq))
        if C > 10 ** (-2):
            beta = 3 * np.log(n + self._initial_data_size + 1) / C
            _blotzmann_prob = lambda e: np.exp(beta * e)
            bm = [_blotzmann_prob(e) for e in acq]
            norm_bm = [float(i) / sum(bm) for i in bm]
            idxes = np.random.choice(range(x.shape[0]), p=np.squeeze(norm_bm), size=(self.n_workers,))
        else:
            idxes = np.random.choice(range(x.shape[0]), size=(self.n_workers,))
        queries = [x[idx] for idx in idxes]

        return np.array(queries)

    def _expected_improvement_fantasized(self, a, x, n):


        x = x.reshape(-1, self._dim)
        queries = []

        model = self.model

        # if self.beta is None:
        #     self.beta = 2.
        # self.beta = 3. + 0.19 * n
        mu, sigma = model.predict(x, return_std=True)
        sigma = np.sqrt(sigma) #correct for syntax err in code
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()

        Y_max = np.max(model.y_train_)
        with np.errstate(divide='ignore'):
            Z = (mu - Y_max) / sigma
            expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0
            expected_improvement[expected_improvement < 10 ** (-100)] = 0
        query = x[np.argmax(expected_improvement)]
        queries.append(query)
        fantasized_y = float(model.predict(query))

        for i in range(self.n_workers - 1):
            fantasized_X.append(query)
            fantasized_Y.append(fantasized_y)
            model.fit(np.array(fantasized_X), np.array(fantasized_Y))
            mu, sigma = model.predict(x, return_std=True)
            sigma = np.sqrt(sigma)
            with np.errstate(divide='ignore'):
                Z = (mu - Y_max) / sigma
                expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
                expected_improvement[sigma == 0.0] = 0
                expected_improvement[expected_improvement < 10 ** (-100)] = 0

            # init_x = np.random.normal(x[np.argmax(expected_improvement)], 0.2)
            init_x = copy.deepcopy(x[np.argmax(expected_improvement)])
            x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
            optimizer = torch.optim.Adam([x0], lr=0.01)
            optimizer.zero_grad()
            best_loss = -np.max(expected_improvement)
            best_x = torch.tensor(x[np.argmax(expected_improvement)])
            # iters = 10
            iters = 0
            for k in np.arange(iters):
                optimizer.zero_grad()
                mu_x, sigma_x = self.model.predict(x0, return_std = True, return_tensor = True)
                sigma_x = torch.sqrt(sigma_x) #take sqrt!!
                # loss = -(mu_x + self.beta * sigma_x)
                Z_x = (mu_x - Y_max) / sigma_x
                normal_dist = torch.distributions.normal.Normal(loc = 0, scale = 1)
                loss = -((mu_x - Y_max) * normal_dist.cdf(Z_x) + sigma_x * torch.exp(normal_dist.log_prob(Z_x)))
                loss.backward()
                optimizer.step()
                if torch.all(x0 == torch.clamp(x0, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
                    # print("x_opt for alg %d" %k, x_opt)
                    x0 = torch.clamp(x0, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
                    x0 = torch.tensor(x0, requires_grad=True,dtype=torch.float32)
                    optimizer = torch.optim.Adam([x0], lr=0.01)
                    optimizer.zero_grad()
                    # print("im here", n,i)
                else:
                    # print("x0 requires grad iter %d" %k, x0.requires_grad)
                    # x0.detach_()
                    # print("x0 requires grad iter %d" %k, x0.requires_grad)
                    if loss < best_loss:
                        best_x = x0
                        best_loss = loss.detach_()
            query = best_x.clone().detach().numpy()
            queries.append(query)                
            fantasized_y = float(model.predict(query))
        return np.array(queries)

    def _find_next_query(self, n, a, random_search, decision_type='distributed'):
        """
        Proposes the next query.
        Arguments:
        ----------
            n: integer
                Iteration number.
            agent: integer
                Agent id to find next query for.
            model: sklearn model
                Surrogate model used for acqusition function
            random_search: integer.
                Number of random samples used to optimize the acquisition function. Default 1000
        """
        # Candidate set
        # random_search *= max(int(self.n_workers/10),1) #higher for more workers
        # print("random search num", random_search)
        # print("random search number", random_search)
        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], size=(random_search, self._dim))
        # print("x shape", x.shape)
        mu = self.model.predict(x)

        #add in new points to x (potential high mu points)


        # first do GD from argmax mu

        x_max_mu = x[np.argmax(mu)]
        init_x = copy.deepcopy(x_max_mu)
        x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
        optimizer = torch.optim.Adam([x0], lr=1e-3)
        optimizer.zero_grad()
        highest_x = torch.tensor(np.max(mu))
        iters = 200
        print("max_mu: n = %d" %n, np.max(mu))
        for k in np.arange(iters):
            optimizer.zero_grad()
            mu_x= self.model.predict(x0,return_tensor=True)
            try:
                loss = -mu_x
            except:
                print("cov_x0_updated value (actual value:%.6f) is probably nan (iter %d), inner iter: %d, just break" %(cov_x0_updated, n,k))
                break
            loss.backward()
            optimizer.step()
            # project back to domain
            if  torch.all(x0 == torch.clamp(x0, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
                x0 = torch.clamp(x0, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
                x0 = torch.tensor(x0, requires_grad=True,dtype=torch.float32)
                optimizer = torch.optim.Adam([x0], lr=1e-4)
                optimizer.zero_grad()
            else:
                pass
        print("final mu after running GD from x_max_mu: n = %d" %n, -loss)
        x = np.vstack((x,x0.detach().clone().numpy()))



        # next try GD from the points already seen


        # Y_max = np.max(self.model.y_train_)
        x_max = self.model.X_train_[np.argmax(self.model.y_train_)]
        # print("x_max", x_max)
        mu_max_x,sigma_max_x = self.model.predict(x_max,return_std=True)
        # print("actual predicted mu_max_x and sigma_max_x %d" %n, mu_max_x, np.sqrt(sigma_max_x))
        # print("sigma of max mu", sigma_of_max_mu)
        # print("Ymax now", Y_max)

        #GD to maximize mu
        thres = 0.1 #thres for deciding whether we need to add a new point (with high posterior mean) to x 
        for i in range(self.model.y_train_.size):
            # print(" I am here (iter %d)" %n, i)
            x_i = self.model.X_train_[i,:]
            x_i = np.reshape(x_i,(1,-1))
            mu_i,sigma_i = self.model.predict(x_i,return_std=True)
            if mu_i <= np.max(mu):
                pass 
            else:
                # print("x_i - x shape", (x-x_i).shape)
                # print(x[0,:] - x_i, "hi")
                dist_i_to_x = np.min(np.linalg.norm( x - x_i, axis = 1))
                if dist_i_to_x > thres : #only consider adding x_i to x if its distance to x is larger than a thres
                    # print("starting at: ", mu_i, "x_i: ", x_i, "dist_i", dist_i_to_x)
                    init_x = copy.deepcopy(x_i)
                    # init_x = np.random.normal(x[np.argmin(ts_ucb)], 0.2)
                    x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
                    # x0 = x0.float()
                    # optimizer = torch.optim.Adam([x0], lr=1e-4)
                    optimizer = torch.optim.Adam([x0], lr=1e-2)
                    optimizer.zero_grad()
                    highest_x = torch.tensor(x_max)
                    # iters = 10
                    # iters = 10
                    iters = 200
                    # iters =0 
                    # if self.objective_name == 'bird': #bird covariance sometimes gets too small, see if this fixes things
                    #     iters = 1
                    for k in np.arange(iters):
                        optimizer.zero_grad()
                        mu_x= self.model.predict(x0,return_tensor=True)
                        try:
                            loss = -mu_x
                        except:
                            print("cov_x0_updated value (actual value:%.6f) is probably nan (iter %d), inner iter: %d, just break" %(cov_x0_updated, n,k))
                            break
                        # print("this is loss, x0, mu_x, sigma_x", loss, x0, mu_x,torch.sqrt(cov_x0_updated))
                        # print("avg mu_x", np.mean(mu))
                        loss.backward()
                        # print("x0 grad at iter %d" %k, x0.grad)
                        optimizer.step()
                        # print("mu at opt iter %d" % k, loss)
                        # project back to domain
                        if  torch.all(x0 == torch.clamp(x0, torch.tensor(self.domain[:,0]),torch.tensor(self.domain[:, 1]))) == False:
                            # print("x_opt for alg %d" %k, x_opt)
                            x0 = torch.clamp(x0, torch.tensor(self.domain[:, 0]),torch.tensor(self.domain[:,1]))
                            x0 = torch.tensor(x0, requires_grad=True,dtype=torch.float32)
                            optimizer = torch.optim.Adam([x0], lr=1e-4)
                            optimizer.zero_grad()
                            # print("im here", n,i)
                        else:
                            pass
                            # # print("x0 requires grad iter %d" %k, x0.requires_grad)
                            # # x0.detach_()
                            # # print("x0 requires grad iter %d" %k, x0.requires_grad)
                            # if loss < best_loss and loss > 0:
                            #     best_x = x0.detach().clone() #please be careful with assignment, assignment or shallow copy will be bad!!
                            #     best_loss = loss.detach()
                            # elif loss < best_loss and loss < 0:
                            #     print("IM HERE (iter %d)" %n)
                            # elif loss < 0: #retry (but within the budget of 10 total gradient steps)
                            #     init_x = np.random.normal(x[np.argmin(ts_ucb)], 0.2)
                            #     x0 = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
                            #     # x0 = x0.float()
                            #     optimizer = torch.optim.Adam([x0], lr=0.01)
                            #     optimizer.zero_grad()
                            #     print("best_x and x0 same (by way of difference)? (shouldn't be)", best_x- x0)
                        if k % 100 == 0 and k > 0:
                            # print("maximum mu_x after training (iter %d), after %d steps" % (n,k), -loss)
                            # print("argmax_mu_x", x0)
                            _, sigma_mu_x = self.model.predict(x0,return_std=True)
                            # print("sigma of argmax_mu_x", np.sqrt(sigma_mu_x))

                    #insert the argmax_mu_x into the x array/also update the mu/sigma arrays. gotta include the best mu_x iterate into the Thompson Sampling!!! 
                    # mu_x, sigma_mu_x = self.model.predict(x0,return_std=True)
                    x = np.vstack((x,x0.detach().clone().numpy()))
                    # print("x.shape at y idx %d" % i, x.shape)
                    # mu = np.append(mu, mu_x)
                    # sigma = np.append(sigma, sigma_mu_x)
                    # if mu_x > mu_max_x:
                    #     mu_max_x = mu_x
                    #     sigma_max_x = sigma_mu_x



        X = x[:]
        if self._record_step:
            X = np.append(self._grid, x).reshape(-1, self._dim)

        # Calculate acquisition function
        x = self._acquisition_function(a, X, n)
        if self._record_step:
            for acq_evaluation in self._acquisition_evaluations:
                acq_evaluation.append(np.zeros_like(x))


        return x

    def optimize(self, n_iters, n_runs = 1, x0=None, n_pre_samples=15, random_search=100, plot = False, noise_scale = 1e-3):
        """
        Arguments:
        ----------
            n_iters: integer.
                Number of iterations to run the search algorithm.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B to optimize the acquisition function.
            plot: bool or integer
                If integer, plot iterations with every plot number iteration. If True, plot every interation.
        """

        self._simple_regret = np.zeros((n_runs, n_iters+1))
        self._simple_cumulative_regret = np.zeros((n_runs, n_iters + 1))
        self._distance_traveled = np.zeros((n_runs, n_iters+1))


        self.init_X = [[]for i in range(n_runs)]
        self.init_Y = [[] for i in range(n_runs)]

        for i in range(n_runs):
        # Initial data (for all n_runs, to ensure consistency of random seed)
            if x0 is None:
                for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
                    self.init_X[i].append(params)
                    self.init_Y[i].append(self.objective(params))
            else:
                # Change definition of x0 to be specfic for each agent
                for params in x0:
                    self.init_X[i].append(params)
                    self.init_Y[i].append(self.objective(params))

        for run in tqdm(range(n_runs), position=0, leave = None, disable = not n_runs > 1):



            # Reset model and data before each run
            self._next_query = []
            # self.bc_data = [[[] for j in range(self.n_workers)] for i in range(self.n_workers)]
            # self.X_train = []
            # self.Y_train =[]
            # self.X = []
            # self.Y = []
            self.X = self.init_X[run]
            self.Y = self.init_Y[run]


            # # Initial data
            # if x0 is None:
            #     for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
            #         self.X.append(params)
            #         self.Y.append(self.objective(params))
            # else:
            #     # Change definition of x0 to be specfic for each agent
            #     for params in x0:
            #         self.X.append(params)
            #         self.Y.append(self.objective(params))

            self._initial_data_size = len(self.Y)

            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            eps_noise = np.random.normal(loc = 0, scale = noise_scale, size = (len(self.Y)))
            # Standardize
            Y = self.scaler[0].fit_transform(np.array(self.Y).reshape(-1, 1)).squeeze() + eps_noise
            # Y = self.scaler[0].fit_transform(np.array(self.Y).reshape(-1, 1)).squeeze() #try no noise
            # self.model = TorchGPModel(torch.tensor(self.X).float(), torch.tensor(Y).float())
            print("self objective_name", self.objective_name)
            self.model = TorchGPModel(torch.tensor(self.X).float(), torch.tensor(Y).float(), 
                alg_name = self.alg_name, n_workers =self.n_workers, objective_name = self.objective_name)
            # self.model.train()
            # self.likelihood.train()


            for n in tqdm(range(n_iters+1), position = n_runs > 1, leave = None):

                # record step indicator
                self._record_step = False
                if plot and run == 0 and self.domain.shape[0] <= 2: #for 3d and above, don't record step
                    if isinstance(plot, int):
                        if n == n_iters or not n % plot:
                            self._record_step = True
                    elif isinstance(plot, list):
                        if n == n_iters or n in plot:
                            self._record_step = True

                # parallel/centralized decision
                # if n > 0:
                if n > 0:
                    obs = [self.objective(q) for q in self._next_query]
                    self.X = self.X + [q for q in self._next_query]
                    self.Y = self.Y + obs
                # self.X_train = self.X_train + [q for q in self._next_query]
                # self.Y_model(train = self.Y_train + obs
                # Updata data knowledge
                # if n == 0:
                #     self.X_train = self.X
                #     self.Y_train = self.Y

                X = np.array(self.X)
                eps_noise = np.random.normal(loc = 0, scale = noise_scale, size = (len(self.Y)))
                # Standardize
                Y = self.scaler[0].fit_transform(np.array(self.Y).reshape(-1, 1)).squeeze() + eps_noise
                # Y = self.scaler[0].fit_transform(np.array(self.Y).reshape(-1, 1)).squeeze() #try no noise
                # Fit surrogate
                self.model.fit(X, Y)
                # if n % 40 == 0:
                # try not re-training
                # if n % 30 == 0 and self.objective_name == 'rastrigin': #try more frequent updates
                #     print("retraining!", self.acq_name)
                #     self.model.train()

                # Find next query
                self._next_query = self._find_next_query(n, 0, random_search, decision_type='parallel')

                # Calculate regret
                _simple_regret = self._regret(np.max(self.Y))
                _simple_cumulative_regret = self._regret(np.max(self.Y))
                # Calculate distance traveled
                if not n:
                    self._distance_traveled[run,n] = 0
                else:
                    if self.domain.shape[0] <= 2:
                        XinAgent = np.array(self.X[self._initial_data_size - self.n_workers:]).reshape([-1, self.n_workers, len(self.domain.shape)])
                        XinAgent = np.swapaxes(XinAgent, 0, 1)
                        self._distance_traveled[run,n] =  self._distance_traveled[run,n-1] + sum([np.linalg.norm(XinAgent[a][-2] - XinAgent[a][-1]) for a in range(self.n_workers)])
                    else: #for higher d, ignore  the distyance travelled
                        self._distance_traveled[run,n] = -1
                # Plot optimization step
                if self._record_step:
                    self._plot_iteration(n, plot)

                try:
                    mu = self.model.predict(X)
                    argmax_mean = X[np.argmax(mu)]
                except:
                    print("model no longer predicting (iter %d), giving an arbitrary argmax_mean" %n)
                    argmax_mean = X[np.argmax(mu)]

        # self.pre_arg_max = []
        # self.pre_max = []
        # for a in range(self.n_workers):
        #     self.pre_arg_max.append(np.array(self.model[a].y_train_).argmax())
        #     self.pre_max.append(self.model[a].X_train_[np.array(self.model[a].y_train_).argmax()]) todo: used for what?

        # Compute and plot regret
        # iter, r_mean, r_conf95 = self._mean_regret()
        # self._plot_regret(iter, r_mean, r_conf95)
        # iter, r_cum_mean, r_cum_conf95 = self._cumulative_regret()
        # self._plot_regret(iter, r_cum_mean, r_cum_conf95, reward_type='cumulative')
        #
        # iter, d_mean, d_conf95 = self._mean_distance_traveled()

        # Save data
        # self._save_data(data = [iter, r_mean, r_conf95, d_mean, d_conf95, r_cum_mean, r_cum_conf95], name = 'data')
                # if n > 0:
                if n >= 0: #change to also record regret at first step, so everybody is the same.
                    query_df_col_name = []
                    obs_df_col_name = []
                    for i in range(self.n_workers):
                        if self.domain.shape[0] == 2:
                            query_df_col_name = query_df_col_name + ['agent{}_x1'.format(i + 1), 'agent{}_x2'.format(i + 1)]
                        elif self.domain.shape[0] == 3:
                            query_df_col_name = query_df_col_name + ['agent{}_x1'.format(i + 1), 'agent{}_x2'.format(i + 1),'agent{}_x3'.format(i + 1)]
                        else:
                            print("ERROR in DOMAIN FOR CSV!!!")
                        obs_df_col_name = obs_df_col_name + ['agent{}_obs'.format(i + 1)]
                        # print("query_df_col_name", query_df_col_name)
                        # print("self._next_query", self._next_query)
                    query_df = pd.DataFrame(np.asarray(self._next_query).reshape([1, -1]), columns=query_df_col_name)
                    if n > 0:
                        obs_df = pd.DataFrame(np.asarray(obs).reshape([1, -1]), columns=obs_df_col_name)
                    else:
                        obs_df = pd.DataFrame(np.zeros(self.n_workers).reshape([1, -1]), columns=obs_df_col_name)
                    if self.domain.shape[0] <= 2:
                        data = dict(iteration=[n], runs=[run], alg=[self.alg_name], regret=[_simple_regret], distance_traveled=[self._distance_traveled[run, n]],
                                argmax_mean_x1=[argmax_mean[0]], argmax_mean_x2=[argmax_mean[1]])
                    else:
                        data = dict(iteration=[n], runs=[run], alg=[self.alg_name], regret=[_simple_regret],
                            argmax_mean_x1=[argmax_mean[0]], argmax_mean_x2=[argmax_mean[1]],argmax_mean_x3=[argmax_mean[2]])
                    df = pd.DataFrame().from_dict(data)
                    total_df = pd.concat([df,obs_df, query_df],axis=1)
                    filepath = os.path.join(self._DATA_DIR_, 'data_nruns=%s_n_agents=%s_n_ysamples=%d.csv' %(n_runs,self.n_workers,self.n_ysamples))
                    if run == 0 and n == 0:
                        total_df.to_csv(filepath)
                    else:
                        total_df.to_csv(filepath, mode='a', header=False)

        with open(self._DATA_DIR_ + '/config.json', 'w', encoding='utf-8') as file:
            json.dump(vars(self.args), file, ensure_ascii=False, indent=4)

            # Generate gif
        # if plot and n_runs == 1:
        #     self._generate_gif(n_iters, plot)


    def _plot_iteration(self, iter, plot_iter):
        """
        Plots the surrogate and acquisition function.
        """
        try:
            mu, std = self.model.predict(self._grid, return_std=True)
        except:
            print("Cannot predict, so not plotting")
            return


        if self._dim == 1:
            pass
        elif self._dim == 2:
            self._plot_2d(iter, mu) #.detach().numpy()
        else:
            print("Can't plot for higher dimensional problems.")

    def _plot_2d(self, iter, mu, acq=None):

        cmap = cm.get_cmap('jet')
        rgba = [cmap(i) for i in np.linspace(0,1,self.n_workers)]
        if self.n_workers == 1:
            rgba = ['k']

        class ScalarFormatterForceFormat(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.2f"
        fmt = ScalarFormatterForceFormat()
        fmt.set_powerlimits((0,0))
        fmt.useMathText = True

        x = np.array(self.X)
        y = np.array(self.Y)
        # _next_query = np.array(self._next_query).reshape([3, -1])

        first_param_grid = np.linspace(self.domain[0,0], self.domain[0,1], self._grid_density)
        second_param_grid = np.linspace(self.domain[1,0], self.domain[1,1], self._grid_density)
        # first_param_grid = np.linspace(-0.5, 0.5, self._grid_density)
        # second_param_grid = np.linspace(-0.5, 0.5, self._grid_density)
        X, Y = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')

        for a in range(1):

            fig1, ax1 = plt.subplots(figsize=(4, 4), sharey=True) # , sharex=True
            # plt.setp(ax.flat, aspect=1.0, adjustable='box')

            N = 100
            # Objective plot
            Y_obj = [self.objective(i) for i in self._grid]
            clev1 = np.linspace(min(Y_obj), max(Y_obj),N)
            cp1 = ax1.contourf(X, Y, np.array(Y_obj).reshape(X.shape), clev1,  cmap = cm.coolwarm)
            for c in cp1.collections:
                c.set_edgecolor("face")
            cbar1 = plt.colorbar(cp1, ax=ax1, shrink = 0.9, format=fmt, pad = 0.05, location='right')
            cbar1.ax.tick_params(labelsize=10)
            cbar1.ax.locator_params(nbins=5)
            ax1.autoscale(False)
            ax1.scatter(x[:, 0], x[:, 1], zorder=1, color = rgba[a], s = 10)
            # ax1.axvline(self._next_query[a][0], color='k', linewidth=1)
            # ax1.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax1.set_ylabel("y", fontsize = 10, rotation=0)
            leg1 = ax1.legend(['Objective'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax1.add_artist(leg1)
            ax1.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax1.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax1.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax1.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            plt.setp(ax1.get_yticklabels()[0], visible=False)
            ax1.tick_params(axis='both', which='both', labelsize=10)
            ax1.scatter(self.arg_max[:,0], self.arg_max[:,1], marker='x', c='gold', s=30)

            # if self.n_workers > 1:
            #     ax1.legend(["Iteration %d" % (iter), "Entropy Search"], fontsize = 10, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            # else:
            #     ax1.legend(["Iteration %d" % (iter)], fontsize = 10, loc='upper left', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax1.legend([self.alg_name], fontsize=10, loc='upper left', handletextpad=0, handlelength=0,
                       fancybox=True, framealpha=0.2)

            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax1.tick_params(axis='both', which='minor', labelsize=10)
            fig1.subplots_adjust(wspace=0, hspace=0)
            ax1.yaxis.offsetText.set_fontsize(10)
            plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d_agent_%d_obj.pdf' % (iter, a), bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d_obj.png' % (iter, a), bbox_inches='tight')

            # Surrogate plot
            fig2, ax2 = plt.subplots(figsize=(4, 4), sharey=True)  # , sharex=True
            d = 0
            if mu.reshape(X.shape).max() - mu.reshape(X.shape).min() == 0:
                d = mu.reshape(X.shape).max()*0.1
            clev2 = np.linspace(mu.reshape(X.shape).min() - d, mu.reshape(X.shape).max() + d,N)
            cp2 = ax2.contourf(X, Y, mu.reshape(X.shape), clev2,  cmap = cm.coolwarm)
            for c in cp2.collections:
                c.set_edgecolor("face")
            cbar2 = plt.colorbar(cp2, ax=ax2, shrink = 0.9, format=fmt, pad = 0.05, location='right')
            cbar2.ax.tick_params(labelsize=10)
            cbar2.ax.locator_params(nbins=5)
            ax2.autoscale(False)
            ax2.scatter(x[:, 0], x[:, 1], zorder=1, color = rgba[a], s = 10)
            if self._acquisition_function in ['es', 'ucb']:
                ax2.scatter(self.amaxucb[0, 0], self.amaxucb[0, 1], marker='o', c='red', s=30)
            # ax2.axvline(self._next_query[a][0], color='k', linewidth=1)
            # ax2.axhline(self._next_query[a][1], color='k', linewidth=1)
            ax2.set_ylabel("y", fontsize = 10, rotation=0)
            ax2.legend(['Surrogate'], fontsize = 10, loc='upper right', handletextpad=0, handlelength=0, fancybox=True, framealpha = 0.2)
            ax2.set_xlim([first_param_grid[0], first_param_grid[-1]])
            ax2.set_ylim([second_param_grid[0], second_param_grid[-1]])
            ax2.set_xticks(np.linspace(first_param_grid[0],first_param_grid[-1], 5))
            ax2.set_yticks(np.linspace(second_param_grid[0],second_param_grid[-1], 5))
            ax2.tick_params(axis='both', which='both', labelsize=10)
            ax2.legend([self.alg_name], fontsize=10, loc='upper left', handletextpad=0, handlelength=0,
                       fancybox=True, framealpha=0.2)
            ax2.scatter(self.arg_max[:, 0], self.arg_max[:, 1], marker='x', c='gold', s=30)


            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.tick_params(axis='both', which='minor', labelsize=10)
            ax2.yaxis.offsetText.set_fontsize(10)
            # ax3.yaxis.offsetText.set_fontsize(10)

            fig2.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(self._PDF_DIR_ + '/bo_iteration_%d_agent_%d_sur.pdf' % (iter, a), bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/bo_iteration_%d_agent_%d_sur.png' % (iter, a), bbox_inches='tight')




