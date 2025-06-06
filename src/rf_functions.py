import numpy as np






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

class simGP_rf:
    def __init__(self,seed = 0, dim = 2, D = 256):
        self.dim = dim
        self.D = D
        self.domain = np.array([[-5, 5] for i in range(dim)])
        self.w, self.b,self.theta = self.get_rf_theta(seed = seed)
        self.function = lambda x: self.get_f(x)
        self.min = 1e3
        self.arg_min = np.array([[0.]*dim])


    def get_rf_theta(self,sigma = 0.25, seed = 0):
        np.random.seed(seed)
        b = np.random.uniform(low = 0, high = 2. * np.pi, size = self.D)
        w = np.random.normal(scale = 1./sigma, size = (self.D,self.dim))
        theta = np.random.uniform(size = self.D)
        return (w,b,theta)


    def get_f(self,x):
        zx = np.sqrt(2./self.D) * np.cos(np.matmul(self.w,x) + self.b)
        fx = np.dot(zx, self.theta)
        return fx
    def get_f_vec(self,x):
        #we assume x is a matrix of dimension d x N, where N is number of points
        zx = np.sqrt(2./self.D) * np.cos(np.matmul(self.w,x).T + self.b)
        fx = np.matmul(zx,self.theta)
        return fx

# class Ackley:
#     def __init__(self):
#         self.domain = np.array([[-5, 5], [-5, 5]])
#         self.function = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20
#         # self.function = lambda x: (-20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20)/100.
#         self.min = 0
#         self.arg_min = np.array([[0,0]])




def find_min(seeds =10,dim = 2, start = -5, end = 5,N = 500):
    x= np.linspace(start,end,N)
    y = np.linspace(start,end,N)
    X,Y = np.meshgrid(x,y)
    print("len X is", len(X))
    XY = np.array([[X[i,j],Y[i,j]] for i in range(N) for j in range(N)]).T
    min_val = [None] * seeds
    for seed in range(seeds):
        gprf = simGP_rf(dim = dim,seed = seed)
        fx = gprf.get_f_vec(XY)
        min_val[seed] = np.min(fx)
    return min_val



if __name__ == '__main__':
    # gprf = simGP_rf(dim = 2)
    # fun = lambda x: gprf.function(x)
    # print(fun(np.array([0,0.])))
    find_min()
