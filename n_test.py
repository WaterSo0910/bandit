
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import zipfile
import os
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

numActions = 10

class Net(nn.Module):
    def __init__(self, d, m, l, theta):
        super(Net, self).__init__()

        self.d = d
        self.m = m
        self.l = l
        # theta to W
        idx = 0
        theta = torch.tensor(theta)
        self.W = []

        w = theta.narrow_copy(0, idx, idx+d*m)
        torch.reshape(w, (m, d))
        self.W.append(w)
        idx = idx + d*m

        for i in range(l-2):
            w = theta.narrow_copy(0, idx, idx+m*m)
            torch.reshape(w, (m, m))
            self.W.append(w)
            idx = idx + m*m

        w = theta.narrow_copy(0, idx, idx+m)
        torch.reshape(w, (m, 1))
        self.W.append(w)
        idx = idx + 1*m
        
        # init L
        self.L = []
        self.L.append(nn.Linear(d, m))
        with torch.no_grad():
            self.L[i].weight.copy_(W[i])

        for i in range(l-2):
            self.L.append(nn.Linear(m, m))
            with torch.no_grad():
                self.L[i].weight.copy_(W[i])
        
        self.L.append(nn.Linear(m, 1))
        with torch.no_grad():
            self.L[l-1].weight.copy_(W[l-1])

    def forward(self, x):
        for i in range(self.l-1):
            x = self.L[i](x)
            x = F.relu(x)
        x = self.L[self.l-1](x)
        x = math.sqrt(self.m) * x
        return x

def nurl_rbmle(contexts, theta, bias, d, m, l):
    u_t = np.zeros(numActions)
    device = torch.device("cuda" if use_cuda else "cpu")
    f = []
    g = []
    x = []
    for k in range(numActions):
        temp = torch.tensor(contexts[k])
        x.append(torch.autograd.Variable(temp, requires_grad=True))

    model = Net(d, m, l, theta).to(device)

    for k in range(numActions):
        f.append(model.forward(x[k]))
        f[k].backward()
        g.append(x[k].grad())

    for k in range(numActions):
        u_t[k] = f[k] + 0.5 * bias * np.matmul(np.matmul(g[k].T, np.linalg.inv(A)), g[k])/m
    arm = np.argmax(u_t)
    # End of TODO
    return arm, g[arm]







