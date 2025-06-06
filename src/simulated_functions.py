import numpy as np


#Note np.rint is used in intermediate idx calc. since otherwise numerical errors occur
#e.g. 0.04 - 0.0 = 0.03999 or 0.04-0.0 =0.0400005 (i.e. floating point errors in either directions)


class simGP1d_Matern:
    def __init__(self):
        root_path = '/Users/zhr568/desktop/research/batch_bayesian/dbo/src/'
        f_data = np.load(root_path + "GP_1d_Matern.npz")
        self.dim = f_data['dim']
        self.domain = f_data['domain']
        self.function = lambda x: self.get_f(x)
        self.min = f_data['min_val']
        self.arg_min = f_data['argmin']
        self.start = f_data['start'] #start of domain for each coord (assume same across coords)
        self.end  = f_data['end']
        self.dt = f_data['dt']
        self.samples = f_data['samples']


    def get_f(self,x):
        #assumes x is an array
        total_steps = np.rint((self.end-self.start)/self.dt) + 1
        x_indices = np.asarray([np.rint((x[i] - self.start)/self.dt) for i in range(self.dim)])
        x_idx = int(round(np.sum(x_indices * (np.ones(self.dim)*total_steps)**(range(self.dim)[::-1]))))
        res = self.samples[x_idx]
        return res


class simGP1d_rbf:
    def __init__(self):
        root_path = '/Users/zhr568/desktop/research/batch_bayesian/dbo/src/'
        f_data = np.load(root_path + "GP_1d_rbf.npz")
        self.dim = f_data['dim']
        self.domain = f_data['domain']
        self.function = lambda x: self.get_f(x)
        self.min = f_data['min_val']
        self.arg_min = f_data['argmin']
        self.start = f_data['start'] #start of domain for each coord (assume same across coords)
        self.end  = f_data['end']
        self.dt = f_data['dt']
        self.samples = f_data['samples']


    def get_f(self,x):
        #assumes x is an array
        total_steps = np.rint((self.end-self.start)/self.dt) + 1
        x_indices = np.asarray([np.rint((x[i] - self.start)/self.dt) for i in range(self.dim)])
        x_idx = int(round(np.sum(x_indices * (np.ones(self.dim)*total_steps)**(range(self.dim)[::-1]))))
        res = self.samples[x_idx]
        return res


class simGP2d_rbf:
    def __init__(self,seed = 0):
        root_path = '/Users/zhr568/desktop/research/batch_bayesian/dbo/src/sim_GP_related/f_data/'
        f_data = np.load(root_path + "GP_2d_rbf_seed=%d.npz" % seed)
        self.dim = f_data['dim']
        self.domain = f_data['domain']
        self.function = lambda x: self.get_f(x)
        self.min = f_data['min_val']
        self.arg_min = f_data['argmin']
        self.start = f_data['start'] #start of domain for each coord (assume same across coords)
        self.end  = f_data['end']
        self.dt = f_data['dt']
        self.samples = f_data['samples']


    def get_f(self,x):
        #assumes x is an array
        total_steps = np.rint(((self.end-self.start)/self.dt)) + 1
        x_indices = np.asarray([np.rint((x[i] - self.start)/self.dt) for i in range(self.dim)])
        x_idx = int(round(np.sum(x_indices * (np.ones(self.dim)*total_steps)**(range(self.dim)[::-1]))))
        res = self.samples[x_idx]
        return res

class simGP3d_rbf:
    def __init__(self,seed = 0):
        root_path = '/Users/zhr568/desktop/research/batch_bayesian/dbo/src/sim_GP_related/f_data/'
        f_data = np.load(root_path + "GP_3d_rbf_seed=%d.npz" % seed)
        self.dim = f_data['dim']
        self.domain = f_data['domain']
        self.function = lambda x: self.get_f(x)
        self.min = f_data['min_val']
        self.arg_min = f_data['argmin']
        self.start = f_data['start'] #start of domain for each coord (assume same across coords)
        self.end  = f_data['end']
        self.dt = f_data['dt']
        self.samples = f_data['samples']


    def get_f(self,x):
        #assumes x is an array
        total_steps = np.rint(((self.end-self.start)/self.dt)) + 1
        x_indices = np.asarray([np.rint((x[i] - self.start)/self.dt) for i in range(self.dim)])
        x_idx = int(round(np.sum(x_indices * (np.ones(self.dim)*total_steps)**(range(self.dim)[::-1]))))
        res = self.samples[x_idx]
        return res


class simGP2d_Matern:
    def __init__(self):
        root_path = '/Users/zhr568/desktop/research/batch_bayesian/dbo/src/'
        f_data = np.load(root_path + "GP_2d_Matern.npz")
        self.dim = f_data['dim']
        self.domain = f_data['domain']
        self.function = lambda x: self.get_f(x)
        self.min = f_data['min_val']
        self.arg_min = f_data['argmin']
        self.start = f_data['start'] #start of domain for each coord (assume same across coords)
        self.end  = f_data['end']
        self.dt = f_data['dt']
        self.samples = f_data['samples']


    def get_f(self,x):
        #assumes x is an array
        total_steps = np.rint((self.end-self.start)/self.dt) + 1
        x_indices = np.asarray([np.rint((x[i] - self.start)/self.dt) for i in range(self.dim)])
        x_idx = int(round(np.sum(x_indices * (np.ones(self.dim)*total_steps)**(range(self.dim)[::-1]))))
        res = self.samples[x_idx]
        return res


if __name__ == '__main__':
    gp1d = simGP1d_rbf()
    fun = lambda x: gp1d.function(x)
    print("gp1d argmin", gp1d.arg_min)
    print(fun(gp1d.arg_min[0]))
    print("gp1d min",gp1d.min)
    print("val for 0", fun([0]))
    # gp2d = simGP2d_Matern()
    # fun = lambda x: gp2d.function(x)
    # print("gp2d argmin", gp2d.arg_min)
    # print("gp2d dim", gp2d.dim)
    # print("this is [0] of argmin",gp2d.arg_min[0])
    # print(fun(gp2d.arg_min[0]))