# +
import os
import sys
import math
import torch
import argparse



import numpy as np

from tqdm import tqdm

from copy import deepcopy
from scipy.stats import entropy
from scipy.stats import multivariate_normal

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split


from itertools import chain
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
# -

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
                    type=str,
                    default="glove",)
args = parser.parse_args("")


def get_mse(samples_erased, samples_orig):
    clf = MLPRegressor(random_state=0, max_iter=500)
    clf.fit(samples_erased, samples_orig)
    
    samples_predict = clf.predict(samples_erased)
    return mean_squared_error(samples_predict, samples_orig)


def get_mi(samples, A, HA):
    clf = MLPClassifier(random_state=0, max_iter=500)
    clf.fit(samples, A)
    
    HAX = np.mean([entropy(x, base=2) for x in clf.predict_proba(samples)])
    IAX = HA - HAX
    return IAX


def get_stats(P, distributions, A, samples):
    HX = entropy(P, base=2)
    cond_entropy = [entropy(P, base=2) for P in distributions]
    HXA = 1/len(distributions) * sum(cond_entropy)
    
    vals = np.array(list(Counter(A).values()))
    P_A = vals/sum(vals)
    HA = entropy(P_A, base=2)
    IAX = get_mi(samples, A, HA)
    return HX, HXA, HA, IAX


def sample_uni_3D(x_min=0, x_max=10, y_min=0, y_max=10, z_min= 0, z_max=10, num_points = 100):
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(z_min, z_max, num_points)

    points = np.column_stack((x, y, z))
    return points


# +
def load_dump(file_path):
    """
    Load data from a .pkl file.

    Args:
        file_path (str): The path to the .pkl file.

    Returns:
        The data from the .pkl file.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def form_dataset(male_words, fem_words, neut_words):
    X, Y = [], []

    for w, v in male_words.items():
        X.append(v)
        Y.append(0)

    for w, v in fem_words.items():
        X.append(v)
        Y.append(1)
    
    for w, v in neut_words.items():
        X.append(v)
        Y.append(2)

    return np.array(X), np.array(Y)


# -

def load_glove(data_path='../data/glove/'):
    '''Loads the GloVe embeddings of gender-biased words.'''

    male_words = load_dump(os.path.join(data_path, 'male_words.pkl'))
    fem_words = load_dump(os.path.join(data_path, 'fem_words.pkl'))
    neut_words = load_dump(os.path.join(data_path, 'neut_words.pkl'))

    X, Y = form_dataset(male_words, fem_words, neut_words)
    
    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.3, random_state=0)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0)
    return X_train, Y_train, Y_train, X_test, Y_test, Y_test


def get_glove():
    X_train, Y_train, Y_train, X_test, Y_test, Y_test = load_glove()
    samples = np.concatenate([X_train, X_test], 0)
    A = np.concatenate([Y_train, Y_test], 0)
    
    samples = samples[A!=2]
    A = A[A!=2]
    
    support_1 = np.unique(samples[A==0], axis=0)
    support_2 = np.unique(samples[A==1], axis=0)
    
    supports = [support_1, support_1, support_2]
    
    idx_1 = list(range(len(support_1)))
    idx_2 = list(range(len(support_2)))

    indices = [idx_1, idx_2]
    
    kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(support_1)
    kde_2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(support_2)
    
    P_1 = np.exp(kde_1.score_samples(support_1))
    P_1 = P_1/sum(P_1)
    P_2 = np.exp(kde_2.score_samples(support_2))
    P_2 = P_2/sum(P_2)
    
    distributions = [P_1, P_2]
    P = np.concatenate(distributions, axis=0)/len(distributions)
    
    return samples, A, distributions, P, indices, supports


def synthetic_data(distribution_type='equal'):
    N = 50
    
    # sample distribution support
    support_1 = sample_uni_3D(0, 10, 0, 10, 0, 10, N)
    support_2 = sample_uni_3D(-10, 0, -10, 0, -10, 0, N)
    
    
    # sample the support for Z
    support_Q = sample_uni_3D(10, 0, 10, 0, 10, 0)
    supports = [support_Q, support_1, support_2]
    
    # Define the distributions
    mean_1 = [5, 5, 5]  
    mean_2 = [-5, -5, -5]  
    cov_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 

    pdf_1 = multivariate_normal(mean=mean_1, cov=cov_matrix)
    pdf_2 = multivariate_normal(mean=mean_2, cov=cov_matrix)
    
    
    # Get the probablity distribution over the supports
    if distribution_type == 'unequal':
        P_1 = np.array([pdf_1.pdf(x) for i, x in enumerate(support_1)])
        P_1 = P_1/sum(P_1)
        P_2 = np.array([pdf_2.pdf(x) for i, x in enumerate(support_2)])
        P_2 = P_2/sum(P_2)
    elif distribution_type == 'equal_gauss':
        # Equal distribution (Gaussian)
        P_1 = np.array([pdf_1.pdf(x) for i, x in enumerate(support_1)])
        P_1 = P_1/sum(P_1)
        P_2 = deepcopy(P_1)
    elif distribution_type == 'equal':
        # Equal distribution (Uniform)
        P_1 = [1/N]*N
        P_2 = [1/N]*N
    
    # final representation distribution
    P = np.concatenate([P_1, P_2], axis=0)/2
    
    # sample 10K representation from each concept group
    num_samples = 10000
    idx_1 = np.random.choice(list(range(len(support_1))), size=num_samples, p=P_1)
    idx_2 = np.random.choice(list(range(len(support_2))), size=num_samples, p=P_2)
    idx = np.concatenate([idx_1, idx_2+N], 0)
    indices = [idx_1, idx_2]

    samples_1 = np.array([support_1[x] for x in idx_1])
    samples_2 = np.array([support_2[x] for x in idx_2])
    samples = np.concatenate([samples_1, samples_2], 0)
    
    distributions = [P_1, P_2]
    A = np.array([0]*len(samples_1) + [1]*len(samples_2))
    return samples, A, distributions, P, indices, supports, idx

if args.dataset == 'synthetic':
    samples, A, distributions, P, indices, supports, idx = synthetic_data()
elif args.dataset == 'glove':
    samples, A, distributions, P, indices, supports = get_glove()



# get MI stats for the synthetic data
HX, HXA, HA, IAX = get_stats(P, distributions, A, samples)
print(f"HX: {HX}, HXA: {HXA}, HA: {HA}, IAX: {IAX}")



# ## Perfect Erasure Functions (PEF)

def get_args(P):
    kwargs = {}
    for i, p in enumerate(P):
        kwargs[f'p_{i}'] = p
    return kwargs


def mec(**kwargs):
    Q = [] 
    for _, p in kwargs.items():
        Q.append(np.exp(p))
    Q = Q/sum(Q)
    Hmin = 0
    for P in (distributions):
        coupling = min_entropy(P, Q)
        flatten_P = list(chain.from_iterable(coupling))
        e = entropy(flatten_P, base=2)
        Hmin += 1./len(distributions)*(e)
    return entropy(Q, base=2) - Hmin


# +
def _min_entropy(p, q):
    coupling = []
    p = sorted(p, reverse=True)
    q = sorted(q, reverse=True)
    coupling.append([0]*len(p))
    
    idx_1, idx_2 = 0, 0
    
    while idx_1 < len(p) and idx_2 < len(q):
        r = min(p[idx_1], q[idx_2])
        p[idx_1] -= r
        q[idx_2] -= r
        
        coupling[idx_2][idx_1] += r
        
        if p[idx_1] == 0:
            idx_1 += 1
            
        if q[idx_2] == 0:
            idx_2 += 1
            if idx_2 < len(q):
                coupling.append([0]*len(p))

    
    return np.array(coupling)

def min_entropy(p, q):
    return _min_entropy(q, p)

def optimal_erase(indices, distributions, Q, support_Q):
    erased_samples = []
    for index, P in zip(indices, distributions):
        coupling = min_entropy(P, Q)
        
        ix1, ix2 = np.argsort(P)[::-1], np.argsort(Q)[::-1]
        
        ix1_inv = {}
        for k, v in enumerate(ix1):
            ix1_inv[v] = k
        
        
        candidates = list(range(len(Q)))
        
        for i in index:
            p = coupling[ix1_inv[i]] / sum(coupling[ix1_inv[i]])
            c = np.random.choice(candidates, p=p)

            erased_samples.append(support_Q[ix2[c]])
    return np.array(erased_samples)


# -

def pef(distributions, samples, indices, supports, idx, A, HA, HX):
    # iterate over local minimas
    choice_idx = np.argmax([mec(**get_args(p)) for p in distributions])
    
    samples_pef = optimal_erase(
        indices, 
        distributions, 
        distributions[choice_idx], 
        supports[choice_idx+1]
    )
    
    IAX_pef = get_mi(samples_pef, A, HA)
    IXZ_pef = get_mi(samples_pef, idx, HX)
    print(f"IXZ_pef: {IXZ_pef}, IAX_pef: {IAX_pef}")


# perform perfect erasure
pef(distributions, samples, indices, supports, idx, A, HA, HX)


# ## PEF (w/ Bayesian Optimization)

def get_Q(distributions, num_iters=200):
    max_len = max([len(P) for P in distributions])
    
    pbounds = {}
    for i in range(max_len):
        pbounds[f'p_{i+1}'] = (-1, 1)
    
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    
    for _ in tqdm(range(num_iters)):
        next_point = optimizer.suggest(utility)
        target = mec(**next_point)
        optimizer.register(params=next_point, target=target)
    
    P = optimizer.max['params']
    Q = [np.exp(v) for k, v in P.items()]
    Q = Q/sum(Q)
    return Q


def pef_BO(distributions, samples, indices, supports, idx, A, HA, HX):
    Q = get_Q(distributions)
    choices.extend(distributions)
    choice_idx = np.argmax([mec(**get_args(p)) for p in choices])
    
    samples_pef_bo = optimal_erase(
        indices, 
        distributions, 
        choices[choice_idx], 
        supports[choice_idx]
    )
    
    IAX_pef_bo = get_mi(samples_pef_bo, A, HA)
    IXZ_pef_bo = get_mi(samples_pef_bo, idx, HX)
    
    
    print(f"IXZ_pef_bo: {IXZ_pef_bo}, IAX_pef_bo: {IAX_pef_bo}")

pef_BO(distributions, samples, indices, supports, idx, A, HA, HX)


