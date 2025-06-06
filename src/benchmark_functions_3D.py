import numpy as np

class Bohachevsky:
    def __init__(self):
        self.domain = np.array([[-100, 100], [-100, 100]])
        self.function = lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1])+0.7
        self.min = 0
        self.arg_min = np.array([[0, 0]])

class Rosenbrock:
    def __init__(self):
        self.domain = np.array([[-2, 2], [-1, 3]])
        self.function = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        self.min = 0
        self.arg_min = np.array([[1, 1]])

class Ackley_3d:
    def __init__(self):
        self.domain = np.array([[-5, 5], [-5, 5],[-5,5]])
        d = 3
        # self.function = lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20
        self.function = lambda x: -20*np.exp(-0.2*np.sqrt((1./d) * np.sum(x**2))) - np.exp((1./d)*( np.sum(np.cos(2.*np.pi* x)))) + np.exp(1) + 20
        # self.function = lambda x: (-20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.exp(1) + 20)/100.
        self.min = 0
        self.arg_min = np.array([np.zeros(d)])


class Rastrigin:
    def __init__(self):
        self.domain = np.array([[-5.12, 5.12], [-5.12, 5.12]])
        self.function = lambda x: 20 + x[0]**2 - 10*np.cos(2*np.pi*x[0]) + x[1]**2 - 10*np.cos(2*np.pi*x[1])
        self.min = 0
        self.arg_min = np.array([[0,0]])


class Himmelblau:
    def __init__(self):
        self.domain = np.array([[-5, 5], [-5, 5]])
        self.function = lambda x: (x[0]**(2)+x[1]-11)**(2)+(x[0]+x[1]**(2)-7)**(2)
        self.min = 0
        self.arg_min = np.array([[3,2],[-2.805118,3.131312],[-3.779310,-3.283186],[3.584428,-1.848126]])


class Eggholder:
    def __init__(self):
        self.domain = np.array([[-512, 512], [-512, 512]])
        self.function = lambda x: -(x[1]+47)*np.sin(np.sqrt(np.abs((x[0]/2) + (x[1]+47)))) - x[0]*np.sin(np.sqrt(np.abs(x[0] - (x[1]+47))))
        self.min = -959.6407
        self.arg_min = np.array([[512,404.2319]])


class GoldsteinPrice:
    def __init__(self):
        self.domain = np.array([[-2, 2], [-2, 2]])
        self.function = lambda x: (1+(x[0]+x[1]+1)**2 * (19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30 + (2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
        self.min = 3
        self.arg_min = np.array([[0,-1]])


class Branin:
    def __init__(self):
        self.domain = np.array([[-5, 10], [0, 15]])
        self.b = 5.1 / (4 * np.pi ** 2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)
        self.function = lambda x: np.square(x[1] - self.b * x[0] ** 2 + self.c * x[0] - self.r) + self.s * (1 - self.t)* np.cos(x[0]) + self.s
        # self.function = lambda x: (np.square(x[1] - self.b * x[0] ** 2 + self.c * x[0] - self.r) + self.s * (1 - self.t)* np.cos(x[0]) + self.s)/10.
        self.min = 0.397887
        self.arg_min = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])


class BraninToy: #check out https://github.com/ma921/SOBER/blob/2c587b7b116a00dd72c1d1f58c0b0f2c20ee0bef/tutorials/08%20Benchmarking%20against%20batch%20BO%20methods.ipynb
    def __init__(self):
        self.domain = np.array([[-2, 3], [-2, 3]])
        self.function = lambda x: -np.prod((np.sin(x) + np.cos(3*x)/2)**2/((x/2.)**2+0.3))
        self.min = -10.604333877563477
        self.arg_min = np.array([[-1.02543108, -1.02543108]])
        print("Testing argmin value: discrepncy to self.min", self.function(self.arg_min) - self.min)


class Griewank:
    def __init__(self):
        self.domain = np.array([[-10,10],[-10,10]])
        self.function = lambda x: np.linalg.norm(x)**2/4000. - np.prod(np.cos(x/np.asarray([1,2]))) + 1
        self.min = 0
        self.arg_min = np.array([[0.,0.]])


class Disk:
    def __init__(self):
        self.domain = np.array([[-5, 10], [0, 15]])
        self.compute_disk()

    def compute_disk(self):
        center = np.mean(self.domain, axis=1)
        radius = min(self.domain[0, 1] - self.domain[0, 0], self.domain[1, 1] - self.domain[1, 0]) / 2
        self.function = lambda x: -(np.square(x[0] - center[0]) + np.square((x[1] - center[1]))) + radius ** 2
        self.center = center
        self.radius = radius

    def set_domain(self, objective):
        self.domain = objective.domain
        self.compute_disk()

class Plate:
    def __init__(self):
        self.domain = np.array([[-5, 10], [0, 15]])
        self.compute_plate()

    def compute_plate(self):
        center = np.mean(self.domain, axis=1)
        radius = min(self.domain[0, 1] - self.domain[0, 0], self.domain[1, 1] - self.domain[1, 0]) / 2
        self.function = lambda x: np.where(-(np.square(x[0] - center[0]) + np.square(x[1] - center[1])) + radius ** 2, [1., -1.])
        self.center = center
        self.radius = radius

    def set_domain(self, objective):
        self.domain = objective.domain
        self.compute_plate()



if __name__ == '__main__':
    branin = Branin()
    fun = lambda x: branin.function(x)
    print(fun(branin.arg_min[1]))