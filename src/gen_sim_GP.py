import numpy as np
import torch
import gpytorch
from gpytorch.priors import MultivariateNormalPrior

# Define the GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood, dim = 1,kernel = 'Matern',ls =None,outputscale= None):
        super(GPModel, self).__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == "Matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims = dim))
        elif kernel == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = dim))
        if ls != None:
            self.covar_module.base_kernel.lengthscale = ls
        if outputscale != None:
            self.covar_module.outputscale = outputscale


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TorchGPModel():
    def __init__(self, X= None, Y=None, dim = 1, kernel = 'Matern',ls=None,outputscale=None):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(self.likelihood,dim=dim,kernel =kernel, ls = ls,outputscale = outputscale)

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


def get_f(x, samples =None,X_values = None, start=-1, end=1,step = 2.0,dim = 1):
    #takes in x and outputs the corresponding val in the sample from GP prior
    #assumes x has the same dimension as the GP prior
    #we use np.rint in intermediate calculations since otherwise weird off-by-1 occur
    #e.g. 0.04 - 0.0 = 0.03999 or 0.04-0.0 =0.0400005 (i.e. floating point errors in either directions)

    total_steps = np.rint((end-start)/step) + 1
    x_indices = np.asarray([np.rint((x[i] - start)/step) for i in range(dim)])
    print("this is inter", x_indices * (np.ones(dim)*total_steps)**(range(dim)[::-1]))
    x_idx = int(round((np.sum(x_indices * (np.ones(dim)*total_steps)**(range(dim)[::-1])))))
    print("this is x_idx", x_idx)
    # print("x_idx", x_idx)
    res = samples[x_idx]
    # print("res shape", res.shape)
    return res



def gen_f_from_GP_prior(dim = 1,kernel = "Matern",start = -5,end = 5,dt = 0.01,ls = None,outputscale = None, seed = 0):

    torch.manual_seed(seed)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    torchModel = TorchGPModel(dim = dim,kernel = kernel, ls = ls,outputscale = outputscale)


    # Sample from the prior distribution
    steps = int((end - start)/dt) + 1
    x = [None]*dim
    for i in range(dim):
        x[i] = torch.linspace(start, end, steps)
    X = np.meshgrid(*x, indexing='ij') #this returns a tuple
    X_values = np.array([X[i].flatten() for i in range(dim)]).T
    # x_values = torch.tensor([[1,1.]])
    # print("x_values", x_values)
    X_tensor = torch.tensor(X_values)
    prior_dist = torchModel.model.forward(X_tensor)


    samples = prior_dist.rsample(sample_shape=torch.Size([1])).detach().numpy()[0] #this returns 1d arr
    # print("x values", X_values)
    # print("samples", samples)
    file_path = "sim_GP_related/f_data/GP_%dd_%s_seed=%d" %(dim,kernel,seed)
    min_val = np.min(samples)
    max_val = np.max(samples)
    argmin = np.asarray([X_values[np.argmin(samples)]])
    print("this is argmin", argmin)
    argmin_idx = np.argmin(samples)
    print("this is argmin idx", np.argmin(samples))
    print("X values near argmin idx =%d" %argmin_idx,X_values[argmin_idx-1:argmin_idx+2])
    print("this is supposed to be min", get_f(argmin[0], samples = samples, start = start,end=end,step=dt,dim=dim))
    print("this is true min", min_val)
    print("this is true max", max_val)
    print("first few elements of samples", samples[:6])
    np.savez(file_path, X_values = X_values, samples = samples, 
        domain = np.array([[start,end] for i in range(dim)]),
        dim = dim, argmin = argmin, min_val = min_val,start = start,end = end,dt = dt)

    # Plot the samples
    import matplotlib.pyplot as plt
    plt.close()
    plt.plot(X_values, samples, linestyle='-', marker='o', label=f'Kernel=%s'%kernel)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Samples from GP Prior')
    plt.legend()
    plt.savefig('sim_GP_related/plots/GP-%dd_%s_seed=%d.pdf' %(dim,kernel,seed))
    plt.close()
    # return(samples,X_values)



class Michalewicz10d:
    def __init__(self):
        # self.dim = 2
        self.dim = 10
        self.domain = np.array([[0,np.pi] for i in range(self.dim)])
        self.m = 10
        self.function = lambda x: -np.sum(np.sin(x) * (np.sin(x**2 * np.arange(1,self.dim+1)/np.pi))**(2*self.m))
        self.min = -9.66015
        # self.arg_min = np.array([[0.20169, 0.150011,0.476874,0.275332,0.311652,0.6573]])
        self.arg_min = np.array([[0]*self.dim])


if __name__ == '__main__':
    # torch.manual_seed(0)
    # gen_f_from_GP_prior(dim = 1,kernel = "Matern")
    # gen_f_from_GP_prior(dim = 1,kernel = "rbf")
    # gen_f_from_GP_prior(dim = 2,kernel = "Matern",start = -2,end = 2, dt =0.01)
    seeds = range(10)
    # seeds=[2]
    for seed in seeds:
        # gen_f_from_GP_prior(dim = 1,kernel = "rbf",start = 0,end = 1, dt =0.01, ls = 0.05,outputscale = 10, seed = seed)
        # gen_f_from_GP_prior(dim = 2,kernel = "rbf",start = 0,end = 1, dt =0.01, ls = 0.05, outputscale = 10,seed = seed)
        gen_f_from_GP_prior(dim = 3,kernel = "rbf",start = 0,end = 1.0, dt =0.04, ls = 0.15, outputscale = 10,seed = seed)
    # gen_f_from_GP_prior(dim = 2,kernel = "rbf",start = 0,end = 1, dt =0.01, ls = 0.05, outputscale = 10)
    # gen_f_from_GP_prior(dim = 1,kernel = "rbf")
    # print("samples shape",samples.shape)
    # dim =1
    # for i in np.arange(8):
    #     x = X_values[i,:]
    #     print("this is for x", x, get_f(x,samples=samples,dim=dim))
    # simGP = TorchGPModel(X = torch.tensor([[0,1]]),Y = torch.tensor([1]))
    # y = simGP.predict(np.zeros(2))
    # print("this is y", y)



    # hart = Hartmann6d()
    # mike = Michalewicz10d()
    # fun = lambda x: mike.function(x)
    # print(fun(mike.arg_min[0]))
    # fun = lambda x: hart.function(x)

    # print(fun(hart.arg_min[0]))