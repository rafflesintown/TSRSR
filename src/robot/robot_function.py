#Adapted from https://github.com/zi-w/Max-value-Entropy-Search/blob/master/test_functions/python_related/generate_simudata3.py
#Robotic 3d code credit to Zi Wang

import numpy as np
import scipy.io 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/Users/zhr568/desktop/research/batch_bayesian/dbo/src/robot/')
from push_world import *




class Robot_4d:
    def __init__(self):
        self.domain = np.array([[-5, 5], [-5, 5],[1,30],[0,np.pi]])
        self.function = lambda x, goal: self.get_diff(x,goal)
        self.min = 0 #fake argmin
        self.arg_min = np.array([[-3, -3]])   

    # difference between goal and the pushing "action" corresponding to x
    def get_diff(self,x, goal = np.array([1.0,1.0])):
        # print("x", x)
        # print("goal", goal)
        rx = x[0]
        ry = x[1]
        simu_steps = int(x[2] * 10)
        angle = x[3]
        gx = goal[0]
        gy = goal[1]
        world = b2WorldInterface(False)
        # world = b2WorldInterface(True)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
        thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))
        init_angle = angle
        # print("init angle", init anglw)
        robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
        ret = simu_push(world, thing, robot, base, simu_steps)
        # print("ret", ret)
        ret = np.linalg.norm(np.array([gx, gy]) - ret)
        # ret = -1./(np.linalg.norm(np.array([gx, gy]) - ret))
        # ret = np.linalg.norm(np.array([gx, gy]) - ret, ord = 1)
        return ret  


class Robot_3d:
    def __init__(self):
        self.domain = np.array([[-5, 5], [-5, 5],[1,30]])
        self.function = lambda x, goal: self.get_diff(x,goal)
        self.min = 0 #fake argmin
        self.arg_min = np.array([[-3, -3]])   

    # difference between goal and the pushing "action" corresponding to x
    def get_diff(self,x, goal = np.array([1.0,1.0])):
        # print("x", x)
        # print("goal", goal)
        rx = x[0]
        ry = x[1]
        simu_steps = int(x[-1] * 10)
        gx = goal[0]
        gy = goal[1]
        world = b2WorldInterface(False)
        # world = b2WorldInterface(True)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
        thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))
        init_angle = np.arctan(ry/rx)
        # print("init angle", init anglw)
        robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
        ret = simu_push(world, thing, robot, base, simu_steps)
        # print("ret", ret)
        ret = np.linalg.norm(np.array([gx, gy]) - ret)
        # ret = -1./(np.linalg.norm(np.array([gx, gy]) - ret))
        # ret = np.linalg.norm(np.array([gx, gy]) - ret, ord = 1)
        return ret       



if __name__ == '__main__':
    x = [-0.6,-0.1,6.1,0.4]
    robot = Robot_4d()
    loss = robot.function(x, goal = np.zeros(2))
    print("loss", loss)


