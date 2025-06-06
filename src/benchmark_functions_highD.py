import numpy as np

# class Bohachevsky:
#     def __init__(self):
#         self.domain = np.array([[-100, 100], [-100, 100]])
#         self.function = lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1])+0.7
#         self.min = 0
#         self.arg_min = np.array([[0, 0]])


class Hartmann6d:
    def __init__(self):
        self.domain = np.array([[0, 1], [0, 1],[0, 1],[0, 1],[0, 1],[0, 1]])
        self.A = np.asarray([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
        self.P = 1e-4 * np.asarray([[1312,1696,5569,124,8283,5886],
            [2329,4135,8307,3736,1004,9991],
            [2348,1451,3522,2883,3047,6650],
            [4047,8828,8732,5743,1091,381]])
        self.alpha = np.asarray([1.0,1.2,3.0,3.2])

        self.function = lambda x: -np.dot(self.alpha, np.exp(-np.sum(self.A*(self.P - x)**2,axis = 1)))
        self.min = -3.3227
        self.arg_min = np.array([[0.20169, 0.150011,0.476874,0.275332,0.311652,0.6573]])


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


class Michalewicz5d:
    def __init__(self):
        # self.dim = 2
        self.dim = 5
        self.domain = np.array([[0,np.pi] for i in range(self.dim)])
        self.m = 10
        self.function = lambda x: -np.sum(np.sin(x) * (np.sin(x**2 * np.arange(1,self.dim+1)/np.pi))**(2*self.m))
        self.min = -4.687658
        # self.arg_min = np.array([[0.20169, 0.150011,0.476874,0.275332,0.311652,0.6573]])
        self.arg_min = np.array([[0]*self.dim])


class Griewank8d:
    def __init__(self):
        self.dim = 8
        self.domain = np.array([[-1,4] for i in np.arange(8)])
        self.function = lambda x: np.linalg.norm(x)**2/4000. - np.prod(np.cos(x/np.arange(1,self.dim+1))) + 1
        self.min = 0
        self.arg_min = np.array([np.zeros(self.dim)])


if __name__ == '__main__':
    hart = Hartmann6d()
    mike = Michalewicz10d()
    grie = Griewank8d()
    # fun = lambda x: mike.function(x)
    fun = lambda x: grie.function(x)
    # print(fun(mike.arg_min[0]))
    print(fun(grie.arg_min[0]))
    # fun = lambda x: hart.function(x)

    # print(fun(hart.arg_min[0]))