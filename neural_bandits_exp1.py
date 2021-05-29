# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import zipfile
import os
import math
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('GTK3Cairo')
# os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch
from  torch import linalg as LA
import torch.nn as nn
import torch.nn.functional as F
import argparse

class Net(nn.Module):
    def __init__(self, d, m, l):
        super(Net, self).__init__()

        self.l = l
        self.m = m
        self.d = d
        self.fc1 = nn.Linear(d, m, False).cuda()
        self.fc2 = nn.Linear(m, 1, False).cuda()

    def forward(self, x):
        # print(x.shape)
        # print(self.W[0].shape)

        for i in range(self.l-1):
            out = self.fc1(x.float())
            out = F.relu(out.float())
            x = out
        x = self.fc2(x.float())
        x = math.sqrt(self.m) * x
        # print(f"x: {x}")

        # self.x_grad[idx] = self.get_grad(outputs=out, inputs=x, size=self.d)

        return x



#------------------------------------------------------------------------------------------------------
theta = np.array([-0.3,0.5,0.8])
excelID = 2
numActions = 10
isTimeVary = True
numExps = 3
T = int(3e4)
seed = 46
path = ""


#---(Optional) a constraint on the norm of theta-------------------------------------------------------
if np.linalg.norm(theta) >= 1:
    raise ValueError("The L2-norm of theta is bigger than 1!")
#------------------------------------------------------------------------------------------------------
methods = ["lin_ucb", "lin_ts", "lin_rbmle"]
numMethods = len(methods)
numMethods = len(methods)
dictResults = {}
allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
np.random.seed(seed)
rewardSigma = 1
torch.manual_seed(seed)

def generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary):
    if isTimeVary:
        contexts = np.random.multivariate_normal(contextMus, contextSigma,(numExps, T, numActions))
    else:
        contexts = np.random.multivariate_normal(contextMus, contextSigma, (numExps, numActions))
    temp = np.linalg.norm(contexts, ord=2, axis=-1)
    contextsNorm = temp[..., np.newaxis]
    contexts = contexts / contextsNorm
    return contexts

def generate_rewards(theta, contexts, isTimeVary, T, rewardSigma):
    if isTimeVary:
        numExps, _, numActions, _ = contexts.shape
    else:
        numExps, numActions, _ = contexts.shape
    
    allRewards = np.zeros((numExps, T, numActions), dtype=float)
    meanRewards = np.zeros((numExps, T, numActions), dtype=float)
    tempSigma = np.eye(numActions) * rewardSigma
    for i in range(numExps):
        for j in range(T):
            tempContexts = contexts[i, j] if isTimeVary else contexts[i]
            tempMus = np.array([np.dot(theta, context) for context in tempContexts])
            meanRewards[i, j] = tempMus
            allRewards[i, j] = np.random.multivariate_normal(tempMus, tempSigma)
    return meanRewards, allRewards

contextMus = np.zeros(len(theta))
contextSigma = np.eye(len(theta)) * 10
allContexts = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
allMeanRewards, allRewards = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
#------------------------------------------------------------------------------------------------------

def lin_ucb(contexts, A, b, alpha):
    # TODO: Implement LinUCB
    # theta_temp=np.matmul(np.linalg.inv(A), b)
    # p_temp= np.zeros(numActions)
    # for a in range(numActions):
    #     p_temp[a]=np.matmul(np.transpose(theta_temp), contexts[a])+ alpha * math.sqrt(np.matmul(np.matmul(np.transpose(contexts[a]),np.linalg.inv(A)), contexts[a]))
    # arm = np.argmax(p_temp)
    # End of TODO
    return 0 # arm

def nurl_rbmle(contexts, A, bias, theta_n, good):
    u_t = np.zeros(numActions)
    f = []
    g = []
    x = []

    device = torch.device("cuda") # if use_cuda else "cpu")

    for k in range(numActions):
        temp = torch.tensor(contexts[k]).to(device)
        x.append(torch.autograd.Variable(temp, requires_grad=True))


    for k in range(numActions):
        model = Net(3, 40, 2)

        # print(model.state_dict())
        model.state_dict()['fc1.weight'][:] = torch.narrow(theta_n, 1, 0, 120).reshape(40, 3)
        model.state_dict()['fc2.weight'][:] = torch.narrow(theta_n, 1, 120, 40).reshape(1, 40)

        out = model.forward(x[k])
        model.zero_grad()
        f.append(out)
        out.backward()
        g_temp = model.fc1.weight.grad.flatten()
        g_temp = torch.cat((g_temp, model.fc2.weight.grad.flatten())).resize(160, 1)
        # g_temp = model.L[0].weight.grad.flatten()
        # for i in range(1,model.l):
        #     g_temp = torch.cat((g_temp, model.L[i].weight.grad.flatten()))

        # g_temp = g_temp.unsqueeze(0).t()
        g.append(g_temp)


    for k in range(numActions):
        u_t[k] = f[k].float() + 0.4 *math.sqrt((torch.mm(torch.mm(g[k].t().float(), torch.inverse(A.float()).cuda()), g[k].float())/40.))
    arm = np.argmax(u_t)
    # print(f'f[arm]: {f[arm]}, var[arm]: {u_t[arm]-f[arm]}')
    # End of TODO
    print('arm: ', str(arm))
    return arm, g[arm], f[good]

def lin_ts(contexts, A, b, v_lints):
    # TODO: Implement LinRBMLE
    arm = 0

    # End of TODO
    return arm

def init_theta(d, m, l):
    w1a = torch.normal(mean=0, std=(4/m), size=(int(m/2), 1))
    w1b = torch.normal(mean=0, std=(4/m), size=(int(m/2), 2))
    zero1a = torch.zeros(size=(int(m/2), 1))
    zero1b = torch.zeros(size=(int(m/2), 2))

    w_left1 = torch.cat((w1a, zero1a), dim=0)
    w_right1 = torch.cat((zero1b, w1b), dim=0)
    W1 = torch.cat((w_left1, w_right1), dim=1)
    W1 = torch.reshape(W1, (1, d*m))
    # print(W1.shape)

    w2 = torch.normal(mean=0, std=(4/m), size=(int(m/2), int(m/2)))
    zero2 = torch.zeros(size=(int(m/2), int(m/2)))

    w_left2 = torch.cat((w2, zero2), dim=0)
    w_right2 = torch.cat((zero2, w2), dim=0)
    W2 = torch.cat((w_left2, w_right2), dim=1)
    W2 = torch.reshape(W2, (1, m*m))
    # print(W2.shape)

    wn = torch.normal(mean=0, std=(2/m), size=(int(m/2), 1))
    Wn = torch.cat((wn.T, -wn.T), dim=1)
    # print(Wn.shape)

    for i in range(l-2):
        W1 = torch.cat((W1, W2), dim=1)
    W1 = torch.cat((W1, Wn), dim=1)

    # print(W1.shape)
    return W1

def decayed_learning_rate(initial_learning_rate, step, decay_steps, alpha):
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + math.cos(3.1416 * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  return initial_learning_rate * decayed



L2norm = []
def L(x, r, theta_now, m, lda, theta_0, d, l, g, Fx):
    # device = torch.device("cuda") # if use_cuda else "cpu")
    # theta_now.detach().requires_grad = True
    # y = m*lda*torch.matmul((theta_now - theta_0), (theta_now - theta_0).t())/2.
    # theta_grad = m * lda * (theta_now - theta_0)
    # print("theta_now ", theta_now)
    # print("theta_0 ", theta_0)
    # y.backward()
    # theta_grad = theta_now.grad
    # theta_now.detach().requires_grad = False

    torch.autograd.Variable(x, requires_grad=True)

    model = Net(d, m, l)
    model.state_dict()['fc1.weight'][:] = torch.narrow(theta_now, 1, 0, 120).reshape(40, 3)
    model.state_dict()['fc2.weight'][:] = torch.narrow(theta_now, 1, 120, 40).reshape(1, 40)

    fx = model.forward(x[-100:])
    model.zero_grad()

    weight = model.fc1.weight.flatten()
    weight = torch.cat((weight, model.fc2.weight.flatten())).reshape(1, 160)
    # print("fx ", fx)
    loss = nn.MSELoss(reduction='mean')
    ll = loss(fx.double(), r.double()[-100:].squeeze(-1))
    norm = torch.norm(theta_now-theta_0) * m * lda
    ll += norm
    # + m * lda * torch.mm((theta_now - theta_0), (theta_now - theta_0).t())

    # y = m*lda*torch.matmul((theta_now - theta_0), (theta_now - theta_0).t())/2.
    # ll = ll + y
    # for i in range(len(fx)):
    #     ll += (fx[i] - r[i]) ** 2 / 2
    # ll = torch.matmul((fx-r).t().float(), (fx-r).float())/2
    ll.backward()

    theta_grad = model.fc1.weight.grad.flatten()
    theta_grad = torch.cat((theta_grad, model.fc2.weight.grad.flatten())).reshape(1, 160)

    return theta_grad, norm


def TrainNN(lda, eta, U, m, x, r, theta_0, d, l, g, Fx):

    theta_now = theta_0.clone()
    # theta_now.requires_grad = True

    for i in range(U):
        # theta_now.detach().requires_grad = True
        theta_grad, l2norm = L(x, r, theta_now, m, lda, theta_0, d, l, g, Fx)
        # if i == U-1:
        #     L2norm.append(l2norm)
        # if i==20 and len(Fx)==300:
        #     print('theta_grad: ', str(theta_grad))
        #     print('theta_now: ', str(theta_now))
        #     input()
        # theta_grad = torch.autograd.grad(outputs=lost, inputs=theta_now, grad_outputs=torch.ones_like(lost))
        # print("lost ",lost)
        theta_now = theta_now - eta*theta_grad
    
    # print("theta_new ", theta_new)
    return theta_now

#------------------------------------------------------------------------------------------------------
for expInd in tqdm(range(numExps)):
    # lin_ucb
    A_linucb = np.eye(len(theta))
    b_linucb = np.zeros(len(theta))
    alpha = 1
    theta_n = init_theta(len(theta), 40, 2).cuda()
    # print(f'theta_n: {theta_n.shape}')
    # lin_rbmle
    A_rbmle = 0.01 * np.eye(theta_n.shape[1]) #! Vt
    A_rbmle = torch.tensor(A_rbmle).cuda()
    b_rbmle = np.zeros(len(theta)) #! Rt
    
    lost = 0
    theta_0 = theta_n.clone()

    Fx = []
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]

        contexts.shape = (numActions,len(theta))

        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]

        #*    HELLO WAKKA
        #!    HELLO WAKKA
        #TODO HELLO WAKKA

        good = np.argmax(meanRewards)

        maxMean = np.max(meanRewards)
    #------------------------------------------------------------------------------------------------
        mPos  = methods.index("lin_ucb")
        startTime = time.time()
        arm = lin_ucb(contexts, A_linucb, b_linucb, alpha)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        # print('regrets: ', str(np.sum(allRegrets[mPos][expInd])))
	# TODO: Update A_linucb, b_linucb
        # A_linucb=A_linucb+np.matmul(contexts[arm], np.transpose(contexts[arm]))
        # b_linucb=b_linucb+contexts[arm]*rewards[arm]
	# End of TODO
    #------------------------------------------------------------------------------------------------
        bias = np.sqrt(t * np.log(t))
        mPos = methods.index("lin_rbmle")
        startTime = time.time()
        arm, grad_now, fx = nurl_rbmle(contexts, A_rbmle, bias, theta_n, good)
        Fx.append(fx)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
	# TODO: Update A_rbmle, b_rbmle
        if(t==1):
            x = torch.tensor(contexts[arm]).cuda()
            x = torch.reshape(x, (1, 3))
            r = torch.tensor(rewards[arm]).cuda()
            r = torch.reshape(r, (1, 1))
            g = grad_now
        else:
            x_now = torch.tensor(contexts[arm]).cuda()
            x_now = torch.reshape(x_now, (1, 3))
            r_now = torch.tensor(rewards[arm]).cuda()
            r_now = torch.reshape(r_now, (1, 1))
            x = torch.cat((x, x_now), dim=0)
            r = torch.cat((r, r_now), dim=0)
            g = torch.cat((g, grad_now), dim=1)
        # print("meanRewards[arm]= ",  meanRewards[arm])
        print('t: ' + str(t))
        # lr = decayed_learning_rate(1e-5, t, 300, 0.99)
        theta_n = TrainNN(0.01, 1e-4, t if t<200 else 200, 40, x, r, theta_0, len(theta), 2, g, Fx)
        A_rbmle = A_rbmle + torch.matmul(grad_now, grad_now.t()) / 40

    # if expInd == 0:
    #     plt.plot(range(1, T+1), Fx)
    #     plt.show()
    #     plt.plot(range(1, T+1), L2norm)
    #     plt.show()

	# End of TODO

#------------------------------------------------------------------------------------------------------
np.random.seed(seed)
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
for expInd in tqdm(range(numExps)):
    # ts
    A_lints = np.eye(len(theta))
    # print(len(theta))
    b_lints = np.zeros(len(theta))
    R_lints = rewardSigma
    delta_lints = 0.5
    epsilon_lints = 0.9
    v_lints = R_lints * np.sqrt(24 * len(theta) * np.log(1 / delta_lints))
    mPos = methods.index("lin_ts")
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
        
        startTime = time.time()
        arm = lin_ts(contexts, A_lints, b_lints, v_lints)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
	# TODO: Update A_lints, b_lints

	# End of TODO
#------------------------------------------------------------------------------------------------------
cumRegrets = np.cumsum(allRegrets,axis=2)
meanRegrets = np.mean(cumRegrets,axis=1)
stdRegrets = np.std(cumRegrets,axis=1)
meanFinalRegret = meanRegrets[:,-1]
stdFinalRegret = stdRegrets[:,-1]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
finalRegretQuantiles = np.quantile(cumRegrets[:,:,-1], q=quantiles, axis=1)

##
plt.plot(range(1, T+1), meanRegrets[methods.index("lin_rbmle")])
# plt.ylim(0, 2500)
plt.show()
##

cumRunningTimes = np.cumsum(allRunningTimes,axis=2)
meanRunningTimes = np.mean(cumRunningTimes,axis=1)
stdRunningTimes = np.std(cumRunningTimes,axis=1)
meanTime = np.sum(allRunningTimes, axis=(1,2))/(T*numExps)
stdTime = np.std(allRunningTimes, axis=(1,2))
runningTimeQuantiles = np.quantile(cumRunningTimes[:,:,-1], q=quantiles, axis=1)

for i in range(len(methods)):
    method = methods[i]
    dictResults[method] = {}
    dictResults[method]["allRegrets"] = np.copy(allRegrets[i])
    dictResults[method]["cumRegrets"] = np.copy(cumRegrets[i])
    dictResults[method]["meanCumRegrets"] = np.copy(meanRegrets[i])
    dictResults[method]["stdCumRegrets"] = np.copy(stdRegrets[i])
    dictResults[method]["meanFinalRegret"] = np.copy(meanFinalRegret[i])
    dictResults[method]["stdFinalRegret"] = np.copy(stdFinalRegret[i])
    dictResults[method]["finalRegretQuantiles"] = np.copy(finalRegretQuantiles[:,i])
    
    
    dictResults[method]["allRunningTimes"] = np.copy(allRunningTimes[i])
    dictResults[method]["cumRunningTimes"] = np.copy(cumRunningTimes[i])
    dictResults[method]["meanCumRunningTimes"] = np.copy(meanRunningTimes[i])
    dictResults[method]["stdCumRunningTimes"] = np.copy(stdRunningTimes[i])
    dictResults[method]["meanTime"] = np.copy(meanTime[i])
    dictResults[method]["stdTime"] = np.copy(stdTime[i])
    dictResults[method]["runningTimeQuantiles"] = np.copy(runningTimeQuantiles[:,i])
    
with open(path + 'ID=' + str(excelID) + '_linbandits_seed_' + str(seed) + '.pickle', 'wb') as handle:
    pickle.dump(dictResults, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print out the average cumulative regret all methods
with open(path + 'ID=' + str(excelID) + '_linbandits_seed_' + str(seed) + '.pickle', 'rb') as handle:
    dictResults = pickle.load(handle)
for method in dictResults:
    print (method, '--', dictResults[method]["meanFinalRegret"])
    
zipfile.ZipFile(path + 'ID=' + str(excelID) + '_linbandits_seed_' + str(seed) + '.zip', mode='w').write('ID=' + str(excelID) + '_linbandits_seed_' + str(seed) + '.pickle')

os.remove(path + 'ID=' + str(excelID) + '_linbandits_seed_' + str(seed) + '.pickle')
#------------------------------------------------------------------------------------------------------


# %%