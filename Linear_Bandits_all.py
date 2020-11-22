
# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import zipfile
import os
#------------------------------------------------------------------------------------------------------
theta = np.array([-0.3,0.5,0.8])
excelID = 2
numActions = 10
isTimeVary = False

numExps = 50
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
    """
    :Implementation of Algorithm 1 in the paper: Contextual Bandits with Linear Payoff Functions
    :A: A in Algorithm 1 o fthe paper
    :b: b in Algorithm 1 of the paper
    :alpha: Line 7 in Algorithm 1
    Returns: the index of the arm with the largest index
    """
    # TODO: Implement LinUCB
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + alpha * np.sqrt(np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])) 
    arm = np.argmax(indVals)
    # End of TODO
    return arm

def lin_rbmle(contexts, A, b, bias):
    """
    :Implementation of Algorithm 1 in the submitted paper
    :A: Vt in Algorithm 1
    :b: Rt in Line 112 in the paper
    :bias: alpha(t) in Algorithm 1
    Returns: the index of the arm with the largest index
    """
    # TODO: Implement LinRBMLE
    A_inv = np.linalg.inv(A)
    thetaHat = np.dot(A_inv, b.T)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i]) + 0.5 * bias * np.linalg.multi_dot([contexts[i], A_inv, contexts[i].T])
    arm = np.argmax(indVals)
        # End of TODO
    return arm

def lin_ts(contexts, A, b, v_lints):
    """
    Implementation of Algorithm 1 in the paper: Thompson Sampling for Contextual Bandits with Linear Payoffs
    :A: as above
    :b: as above
    :v_lints: v in Section 2.2, first paragraph i nthe paper
    Returns: the index of the arm with the largest index
    """
    # TODO: Implement LinRBMLE
    A_inv = np.linalg.inv(A)
    Mean = np.dot(A_inv, b)
    Cov = v_lints**2 * A_inv
    thetaHat = np.random.multivariate_normal(Mean, Cov)
    indVals = np.zeros(contexts.shape[0])
    for i in range(contexts.shape[0]):
        indVals[i] = np.dot(thetaHat, contexts[i])
    arm = np.argmax(indVals)
    # End of TODO
    return arm

def update_posterior(A, b, x, r):
    """
    Preparation for posteiror update in Bayes family of contextual bandit algorithms
    :A: matrix A in all algorihms' input
    :b: vector b in all algorithms' input
    Returns: updated A and updated b
    """
    A_new = np.add(A, x[:, np.newaxis] * x)
    b_new = np.add(b, r * x)
    return A_new, b_new

#------------------------------------------------------------------------------------------------------
for expInd in tqdm(range(numExps)):
    # lin_ucb
    A_linucb = np.eye(len(theta))
    b_linucb = np.zeros(len(theta))
    alpha = 1
    
    # lin_rbmle
    A_rbmle = np.eye(len(theta))
    b_rbmle = np.zeros(len(theta))
    
    
    for t in range(1, T+1):
        contexts = allContexts[expInd, t-1, :] if isTimeVary else allContexts[expInd, :]
        rewards = allRewards[expInd, t-1, :]
        meanRewards = allMeanRewards[expInd, t-1, :]
        maxMean = np.max(meanRewards)
    #------------------------------------------------------------------------------------------------
        mPos = methods.index("lin_ucb")
        startTime = time.time()
        arm = lin_ucb(contexts, A_linucb, b_linucb, alpha)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
        # TODO: Update A_linucb, b_linucb
        A_linucb, b_linucb = update_posterior(A_linucb, b_linucb, contexts[arm], rewards[arm])
	    # End of TODO
    #------------------------------------------------------------------------------------------------
        bias = np.sqrt(t * np.log(t))
        mPos = methods.index("lin_rbmle")
        startTime = time.time()
        arm = lin_rbmle(contexts, A_rbmle, b_rbmle, bias)
        duration = time.time()-startTime
        allRunningTimes[mPos][expInd][t-1] = duration
        allRegrets[mPos][expInd][t-1] =  maxMean - meanRewards[arm]
	    # TODO: Update A_rbmle, b_rbmle
        A_rbmle, b_rbmle = update_posterior(A_rbmle, b_rbmle, contexts[arm], rewards[arm])
	    # End of TODO

#------------------------------------------------------------------------------------------------------
np.random.seed(seed)
allContexts_forSeedReset = generate_norm_contexts(contextMus, contextSigma, numExps, T, numActions, isTimeVary)
allMeanRewards_forSeedReset, allRewards_forSeedReset = generate_rewards(theta, allContexts, isTimeVary, T, rewardSigma)
for expInd in tqdm(range(numExps)):
    # ts
    A_lints = np.eye(len(theta))
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
        A_lints, b_lints = update_posterior(A_lints, b_lints, contexts[arm], rewards[arm])
	    # End of TODO
#------------------------------------------------------------------------------------------------------
cumRegrets = np.cumsum(allRegrets,axis=2)
meanRegrets = np.mean(cumRegrets,axis=1)
stdRegrets = np.std(cumRegrets,axis=1)
meanFinalRegret = meanRegrets[:,-1]
stdFinalRegret = stdRegrets[:,-1]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
finalRegretQuantiles = np.quantile(cumRegrets[:,:,-1], q=quantiles, axis=1)

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

