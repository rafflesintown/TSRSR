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
                 grid_density = 100, args=dict()):

        # Optimization setup
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
        else:
            print('Supported acquisition functions: ei, ts, es, ucb, ts_ucb, ts_ucb_seq, ts_ucb_det,ts_ucb_vals')
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
                    print("random search number", random_search)
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
            mu_a, std_a = self.model[a].predict(self._grid, return_std=True)
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
    def __init__(self, X, Y):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(X, Y, self.likelihood)
        self.train()
    
    def train(self):
        self.model.train()
        self.likelihood.train()

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

        y_mean, y_cov = self.predict(X, return_cov=True)
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
                 grid_density = 100, x0=None, n_pre_samples=5, args=dict()):

        super(BayesianOptimizationCentralized, self).__init__(objective, domain=domain, arg_max=arg_max, n_workers=n_workers,
                 network=network, kernel=kernel, alpha=alpha,
                 acquisition_function=acquisition_function, policy = policy, fantasies = fantasies,
                 epsilon = epsilon, regularization = regularization, regularization_strength = regularization_strength,
                 pending_regularization = pending_regularization, pending_regularization_strength = pending_regularization_strength,
                 grid_density = grid_density, args=args)
        assert self.args.decision_type == 'parallel' or self.n_workers == 1
        self.diversity_penalty = args.diversity_penalty
        self.radius = args.div_radius
        self.acq_name = None
        if acquisition_function == 'es':
            self._acquisition_function = self._entropy_search_grad
        elif acquisition_function == 'bucb' or acquisition_function == 'ucbpe':
            self._acquisition_function = self._batch_upper_confidential_bound
            self.acq_name = acquisition_function
        elif acquisition_function == 'ei' and fantasies == self.n_workers:
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
        else:
            print('Supported acquisition functions: ei, ts, es, bucb, ucbpe, ts_ucb, ts_ucb_seq, ts_ucb_det,ts_ucb_vals')
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
        self.beta = 3 - 0.019 * n

        # self.model.eval()
        # self.likelihood.eval()
        domain = self.domain

        mu, sigma = self.model.predict(x, return_std=True, return_tensor=True)
        ucb = mu + self.beta * sigma
        amaxucb = x[np.argmax(ucb.clone().detach().numpy())][np.newaxis, :]
        self.amaxucb = amaxucb
        # x = np.vstack([amaxucb for _ in range(self.n_workers)])
        init_x = np.random.normal(amaxucb, 1.0, (self.n_workers, self.domain.shape[0]))

        x = torch.tensor(init_x, requires_grad=True,dtype=torch.float32)
        optimizer = torch.optim.Adam([x], lr=0.01)
        # training_iter = 200
        training_iter = 200
        for i in range(training_iter):
            if i % 50 == 0:
                print("I am", i)
                print("x at iter %d: " %i, x)
            optimizer.zero_grad()
            joint_x = torch.vstack((x,torch.tensor(amaxucb).float()))
            joint_x = joint_x.float()
            x = x.float()
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
            loss.backward()
            optimizer.step()
            # project back to domain
            x = torch.where(x > torch.tensor(domain[:, 1]), torch.tensor(domain[:, 1]), x)
            x = torch.where(x < torch.tensor(domain[:, 0]), torch.tensor(domain[:, 0]), x)
            x.detach_()
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

    def _TS_UCB(self, a, x, n, radius=0.1,n_random = 200, n_samples = 1, n_restarts = 1500):
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
        samples = self.model.sample_y(x_random, n_samples=n_samples)
        # print(samples.shape, "shape samples")
        fstar_hat = 0.0
        for i in range(n_samples):
            fstar_hat += np.max(samples[:,i])
        fstar_hat /= n_samples
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
                            torch.matmul(cov_xj_x_upto_j, torch.linalg.inv(cov_x_x_sub + 0.01 * torch.eye(len(cov_x_x_sub)))), 
                            cov_xj_x_upto_j.T)
                            loss_den += torch.sqrt(cov_j_new)
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


    def _TS_UCB_vals(self, a, x, n, radius=0.1,n_random = 200, n_samples = 1, n_restarts = 1500):
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
        samples = self.model.sample_y(x_random, n_samples=n_samples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            for i in range(n_samples):
                fstar_hat[j] += np.max(samples[:, j*self.n_workers + i])
            fstar_hat[j] /= n_samples
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
                loss = torch.log(torch.sum(fstar_hat - mu_x)) - torch.log(
                    torch.sqrt(torch.sum(torch.torch.linalg.svdvals(cov_x_x))))
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




    def _TS_UCB_seq(self, a, x, n, radius=0.1,n_random = 200, n_samples = 1, n_restarts = 300):
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
        # Y_max = np.max(self.model.y_train_)
        # n_random *= max(int(self.n_workers/10),1) 
        n_random *= max(int(self.n_workers/10),1) * 5 #testing for now
        x_random = np.random.uniform(self.domain[:,0], self.domain[:,1], (n_random, self.domain.shape[0]))
        x_random = x_random.reshape(-1,self._dim)
        samples = self.model.sample_y(x_random, n_samples=n_samples * self.n_workers)
        # print(samples.shape, "shape samples")
        fstar_hat = np.zeros(self.n_workers)
        for j in np.arange(self.n_workers):
            print("here", j)
            for i in range(n_samples):
                print("j", j )
                print("i", i)
                print("max", np.max(samples[:, j*self.n_workers + i]))
                fstar_hat[j] += np.max(samples[:, j*self.n_workers + i])
            fstar_hat[j] /= n_samples
        # print("f star hat", fstar_hat)
        # print("Y_max", Y_max)



        mu,sigma = self.model.predict(x,return_std=True)
        fantasized_X = self.X.copy()
        fantasized_Y = self.Y.copy()
        ts_ucb = (fstar_hat[0] - mu)/np.sqrt(sigma)
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
            ts_ucb = (fstar_hat[i+1] - mu)/np.sqrt(sigma)
            # print(fstar_hat,
            #         np.max(mu),
            #         sigma[np.argmin(ts_ucb)],
            #     "min ts_ucb numerator fstar_hat, mu and denominator std")
            query = x[np.argmin(ts_ucb)]
            queries.append(query)
        return np.array(queries)



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
        # self.beta = 0.15 + 0.019 * n
        delta = 0.01
        #this uses the hyperparameter setting prescribed in BUCB Desaultels paper, with pre-constant 0.1
        self.beta = 0.1 *(2. * log(x.shape[0] * n**2 * pi**2/(6 * delta))) 

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
            if self.acq_name == 'bucb':
                ucb = mu + self.beta * sigma
                query = x[np.argmax(ucb)]
            elif self.acq_name == 'ucbpe':
                query = x[np.argmax(sigma)]
            queries.append(query)
        return np.array(queries)

    def _thompson_sampling_centralized(self, a, x, n):
        x = x.reshape(-1, self._dim)
        queries = []

        model = self.model

        samples = model.sample_y(x, n_samples=self.n_workers)

        for i in range(self.n_workers):
            query = x[np.argmax(samples[:, i])]
            queries.append(query)

        return np.array(queries)

    def _stochastic_policy_centralized(self, a, x, n):
        x = x.reshape(-1, self._dim)
        acq = self.model.predict(x)
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
        self.beta = 3. + 0.19 * n
        mu, sigma = model.predict(x, return_std=True)
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
            with np.errstate(divide='ignore'):
                Z = (mu - Y_max) / sigma
                expected_improvement = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
                expected_improvement[sigma == 0.0] = 0
                expected_improvement[expected_improvement < 10 ** (-100)] = 0
            query = x[np.argmax(expected_improvement)]
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
        random_search *= max(int(self.n_workers/10),1) #higher for more workers
        # print("random search number", random_search)
        x = np.random.uniform(self.domain[:, 0], self.domain[:, 1], size=(random_search, self._dim))

        X = x[:]
        if self._record_step:
            X = np.append(self._grid, x).reshape(-1, self._dim)

        # Calculate acquisition function
        x = self._acquisition_function(a, X, n)
        if self._record_step:
            for acq_evaluation in self._acquisition_evaluations:
                acq_evaluation.append(np.zeros_like(x))


        return x

    def optimize(self, n_iters, n_runs = 1, x0=None, n_pre_samples=15, random_search=100, plot = False):
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
            self._next_query = []
            # self.bc_data = [[[] for j in range(self.n_workers)] for i in range(self.n_workers)]
            self.X_train = []
            self.Y_train =[]
            self.X = []
            self.Y = []


            # Initial data
            if x0 is None:
                for params in np.random.uniform(self.domain[:, 0], self.domain[:, 1], (n_pre_samples, self.domain.shape[0])):
                    self.X.append(params)
                    self.Y.append(self.objective(params))
            else:
                # Change definition of x0 to be specfic for each agent
                for params in x0:
                    self.X.append(params)
                    self.Y.append(self.objective(params))
            self._initial_data_size = len(self.Y)

            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # Standardize
            Y = self.scaler[0].fit_transform(np.array(self.Y).reshape(-1, 1)).squeeze()
            self.model = TorchGPModel(torch.tensor(self.X).float(), torch.tensor(Y).float())
            self.model.train()
            # self.likelihood.train()


            for n in tqdm(range(n_iters+1), position = n_runs > 1, leave = None):

                # record step indicator
                self._record_step = False
                if plot and run == 0:
                    if isinstance(plot, int):
                        if n == n_iters or not n % plot:
                            self._record_step = True
                    elif isinstance(plot, list):
                        if n == n_iters or n in plot:
                            self._record_step = True

                # parallel/centralized decision
                if n > 0:
                    obs = [self.objective(q) for q in self._next_query]
                    self.X = self.X + [q for q in self._next_query]
                    self.Y = self.Y + obs
                # self.X_train = self.X_train + [q for q in self._next_query]
                # self.Y_train = self.Y_train + obs
                # Updata data knowledge
                # if n == 0:
                #     self.X_train = self.X
                #     self.Y_train = self.Y

                X = np.array(self.X)
                # Standardize
                Y = self.scaler[0].fit_transform(np.array(self.Y).reshape(-1, 1)).squeeze()
                # Fit surrogate
                self.model.fit(X, Y)
                if n % 40 == 0:
                    self.model.train()

                # Find next query
                self._next_query = self._find_next_query(n, 0, random_search, decision_type='parallel')

                # Calculate regret
                _simple_regret = self._regret(np.max(self.Y))
                _simple_cumulative_regret = self._regret(np.max(self.Y))
                # Calculate distance traveled
                if not n:
                    self._distance_traveled[run,n] = 0
                else:
                    XinAgent = np.array(self.X[self._initial_data_size - self.n_workers:]).reshape([-1, self.n_workers, len(self.domain.shape)])
                    XinAgent = np.swapaxes(XinAgent, 0, 1)
                    self._distance_traveled[run,n] =  self._distance_traveled[run,n-1] + sum([np.linalg.norm(XinAgent[a][-2] - XinAgent[a][-1]) for a in range(self.n_workers)])

                # Plot optimization step
                if self._record_step:
                    self._plot_iteration(n, plot)

                mu = self.model.predict(X)
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
                if n > 0:
                    query_df_col_name = []
                    obs_df_col_name = []
                    for i in range(len(obs)):
                        query_df_col_name = query_df_col_name + ['agent{}_x1'.format(i + 1), 'agent{}_x2'.format(i + 1)]
                        obs_df_col_name = obs_df_col_name + ['agent{}_obs'.format(i + 1)]
                    query_df = pd.DataFrame(np.asarray(self._next_query).reshape([1, -1]), columns=query_df_col_name)
                    obs_df = pd.DataFrame(np.asarray(obs).reshape([1, -1]), columns=obs_df_col_name)
                    data = dict(iteration=[n], runs=[run], alg=[self.alg_name], regret=[_simple_regret], distance_traveled=[self._distance_traveled[run, n]],
                                argmax_mean_x1=[argmax_mean[0]], argmax_mean_x2=[argmax_mean[1]])
                    df = pd.DataFrame().from_dict(data)
                    total_df = pd.concat([df,obs_df, query_df],axis=1)
                    filepath = os.path.join(self._DATA_DIR_, 'data_nruns=%s_n_agents=%s.csv' %(n_runs,self.n_workers))
                    if run == 0 and n == 1:
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
        mu, std = self.model.predict(self._grid, return_std=True)


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




