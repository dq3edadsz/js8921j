from MSPM.mspm_config import *
import numpy as np
import random
import math
from data.data_processing import find_clusters
from glob import glob
import json
from tqdm import tqdm
from scipy.optimize import curve_fit
from multiprocessing import Pool
from numba import cuda
from collections import defaultdict
from data.data_processing import cluster_modification
import cupy as cp
import collections

def unreuse_p(i):
    """

    :param i:
    :return: the unreused probability for i+1 th password considering the former i passwords
    """
    return 1 / (3.15685852 + 0.22641382 * i) # _bc
    #
    #1 / (0.02455*i**3 - 0.2945*i**2 + 3.409*i + 0.0852) # _pb

def unreusedprob_vault(v, mxsize, progressbar):
    print(progressbar)
    # get alpha for v
    groups = find_clusters(v, 0)[0] # similar pws
    alph, deno = 0, 0
    for g in groups:
        same_pairs = [g.count(pw) * (g.count(pw) - 1) for pw in list(set(g))]
        if sum(same_pairs) == 0:
            alph += 0
        else:
            alph += math.exp(math.log(sum(same_pairs)) - math.log(len(g) * (len(g) - 1)))
        deno += 1
    alph = (alph / deno) # if alph != 0 else 0
    #alph = math.exp(math.log(np.array([math.factorial(v.count(pw) - 1) for pw in list(set(v))]).sum()) - math.log(math.factorial(len(v) - 1)))

    unreuse_indicator = np.zeros((mxsize))
    directreuse_indicator = np.zeros((mxsize))
    distinct_pws = list(set(v))
    id_v = np.array([distinct_pws.index(pw) for pw in v])[:, None]
    groups_pwid = np.ones((len(groups), max([len(g) for g in groups]))) * -1
    for g in range(len(groups)):
        for i, pw in enumerate(groups[g]):
            groups_pwid[g, i] = distinct_pws.index(pw)
    groupid_v = np.ones_like(id_v)
    for i in range(len(v)):
        for g in range(groups_pwid.shape[0]):
            if id_v[i, 0] in list(groups_pwid[g]):
                groupid_v[i, 0] = g
                break
    # vectorize to speed up
    rn = 2000
    id_v = np.repeat(id_v, rn, axis=1) # vaultsize x rn
    groupid_v = np.repeat(groupid_v, rn, axis=1) # vaultsize x rn
    for clmid in range(rn):
        id_v[:, clmid] = np.random.RandomState(seed=clmid).permutation(id_v[:, clmid])
        groupid_v[:, clmid] = np.random.RandomState(seed=clmid).permutation(groupid_v[:, clmid])
    with cp.cuda.Device(0):#random.choice([0, 1])
        id_v = cp.array(id_v, dtype=np.uint8)
        groupid_v = cp.array(groupid_v, dtype=np.uint8)
        #groupid_eachpw = cp.array(groups_pwid)
        #groupidx = ((id_v[None][None] - groups_pwid[:, :, None]).prod(1) == 0).nonzero()[0][:rn] # rn
        for i in range(len(v) - 1):
            #i = len(v) - 2
            unreuse_indicator[i] += cp.asnumpy(((groupid_v[:i+1] - groupid_v[i+1]).prod(0) != 0).sum()) / rn
            # get group idx for each in id_v[i+1]
            #group_each = groups_pwid[groupidx] # rn x max_groupsize
            #print(group_each.shape, id_v.get()[i+1][:, None].shape, id_v.get()[i+1].shape, groups_pwid.shape)
            #directreuse_rate = (((group_each - id_v.get()[i+1][:, None]) == 0).sum(1) / (group_each > -1).sum(1))[((id_v[:i+1] - id_v[i+1]).prod(0) == 0).get()]
            directreuse_indicator[i] += cp.asnumpy(((id_v[:i+1] - id_v[i+1]).prod(0) == 0).sum()) / rn

    return unreuse_indicator, np.concatenate(([1]*(len(v)-2), [1], [0]*(mxsize-len(v)+1))), alph, directreuse_indicator


def fit_unreuse_p():
    """
    :return: the fitting function of unreuse_p, unreuse_p(i) is the probabilty that i+1 th password not reused from the former i passwords
    """
    def func(x, a, b, c, d):#
        return 1 / (a + b*x + c*x**2 + d*x**3) #
    vaults = {}
    vaultpaths = ['/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_50_tr.json']#, '/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_51_200_tr.json' glob('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/pastebin/fold5/*') #
    for path in vaultpaths:
        with open(path) as f:
            vaults.update(json.load(f))

    vaults = list(vaults.values())
    # randomly select 10,000 vaults
    vs2times = defaultdict(int)
    vault_retained = []
    for v in vaults:
        if vs2times[len(v)] < 100: # 100 for bc obtained
            vault_retained.append(v)
            vs2times[len(v)] += 1
    vaults = vault_retained

    mxsize = max([(len(v) - 1) for v in vaults])
    unreuse_arr = np.zeros((len(vaults), mxsize))
    indicator_arr = np.zeros((len(vaults), mxsize))
    dirreuse_arr = np.zeros((len(vaults), mxsize))
    pool = Pool(30)
    workers = []
    alp = 0
    deno = 0
    for i, vault in enumerate(vaults):
        workers.append(pool.apply_async(unreusedprob_vault, (vault, mxsize, (i+1)/len(vaults) * 100)))
    pool.close()
    pool.join()
    for rowid, w in enumerate(workers):
        re = w.get()
        unreuse_arr[rowid] += re[0]
        indicator_arr[rowid] += re[1]
        dirreuse_arr[rowid] += re[3]
        alp += re[2]
        deno += 1
    print("alpha:", alp / deno)
    x = np.arange(1, mxsize+1)
    y = unreuse_arr.sum(0) / (indicator_arr != 0).sum(0)
    y_dirreuse = dirreuse_arr.sum(0) / (indicator_arr != 0).sum(0)
    # save x, y to csv
    np.savetxt('unreusefit.csv', np.stack([x, y, y_dirreuse], axis=1), delimiter=',')
    popt, pcov = curve_fit(func, x, y)#, bounds=([0, 0, -1, 0], [10, 10, 0, 1])
    print(popt)

def interval_reuse(N, N_itv):
    """
    1. password reuse in the interval is defined as n_1,n_2,...,n_m, where n_1+n_2+...+n_m=N_itv
    2. n_i is the times of password i reused within the interval, and it is at least 1
    3. unreuse_p is used as password reuse behavior to create the password vault: if
    :param N: number of passwords in the password vault, password vault in the function is represented as two lists
                e.g., password list : [0,0,1,0,2,0,3,3,4,1,0,...]. "0“ represents the password is not reused,
                     any number "r" greater than 0 represents the password is reused, and is reused from the r-th password,
                     reuse has two ways: direct reuse and modification, each has certain probability, and the former will contribute
                     identical password
                      identical group list : [1,2,3,4,3,5,4,6,3,2,7,...]. each number shows a group of password
                                             passwords with the same number are identical
    :param N_itv: number of passwords in the interval
    :return: the average number of password reuse in each interval
    """
    identical_group_list = []
    cluster_list = []
    for i in range(N):
        if random.random() > unreuse_p(i) and i != 0:
            # if reused (from the first i passwords)
            ridx = random.choice(list(range(len(identical_group_list))))
            potential_group = identical_group_list[ridx]
            cluster_list.append(cluster_list[ridx])
            if random.random() < i * ALPHA / (i * ALPHA + 1 - ALPHA):
                # reused but direct reuse
                identical_group_list.append(potential_group)
            else:
                # reused but modification
                identical_group_list.append(max(identical_group_list)+1)
        else:
            # if not reused
            identical_group_list.append(max(identical_group_list)+1 if len(identical_group_list) > 0 else 1)
            cluster_list.append(max(cluster_list)+1 if len(cluster_list) > 0 else 1)
    return identical_group_list, cluster_list

def exp_interval_reuse(num_tests, N, N_itv):
    rr, grnum = [], []
    for _ in range(num_tests):
        identical_group_list, cluster_lst = interval_reuse(N, N_itv)
        rr.append(N / len(set(cluster_lst)))
    return np.mean(np.array(rr))

def password_group(N, x=6):
    """
    :param N: number of passwords in the password vault
    :return: expected password group for vault size N
    """
    interval_count = exp_interval_reuse(3*10 ** 4, N, N)
    for k in interval_count:
        interval_count[k] = math.ceil(interval_count[k] * N)
    group_count = []
    total = 0
    for k in interval_count.keys():
        if total + interval_count[k] <= N:
            group_count.append(interval_count[k])
            total += interval_count[k]
        elif N-total > 0:
            group_count.append(N - total)
            break
    log_factorial_N = math.log(math.factorial(N))
    log_10_x = float(np.array([math.log(math.factorial(v)) for v in group_count]).sum()) + math.log(10) * x
    log_result = log_factorial_N - log_10_x
    return math.exp(log_result) # math.exp(min(0, log_result))

def get_unique_pw(N):
    unique_pw = []
    for rn in range(1000):
        ltmp = interval_reuse(N, 10)
        unique_pw.append(max(ltmp))
    return np.array(unique_pw).mean()


def main():
    fit_unreuse_p()

    # get avg rr for given fi and alpha
    '''rr = []
    for vs in tqdm(range(50, 200)):
        rr.append(exp_interval_reuse(10**3, vs, vs))
    print('mean:', np.mean(np.array(rr)))
    print('median:', np.median(np.array(rr)))'''

if __name__ == '__main__':
    #main()

    # ************* fit function *************************
    def func1(x, a, b, c, d):#
        return 1 / (a + b*x + c*x**2 + d*x**3) #
    def func1_2(x, a, b):
        return 1 / (a + b*x)
    xy = np.loadtxt('unreusefit.csv', delimiter=',')
    # save x, y to csv
    '''popt, pcov = curve_fit(func1_2, xy[:, 0], xy[:, 1])  # , bounds=([0, 0, -1, 0], [10, 10, 0, 1])
    print(popt)
    def func2(x, a):
        return x * a / (x * a + 1 - a)
    popt, pcov = curve_fit(func2, xy[:, 0], xy[:, 2])  # , bounds=([0, 0, -1, 0], [10, 10, 0, 1])
    print(popt)'''


    # ************* draw ****************
    # plot rational 1 / (a + b*x + c*x**2 + d*x**3) with [a, b, c, d] = [1.74854998e+00  5.50041571e-01 -1.80087098e-02  3.69128122e-04]
    x = np.arange(1, 50)
    y = 1 / (1.74854998e+00 + 5.50041571e-01 * x - 1.80087098e-02 * x**2 + 3.69128122e-04 * x**3)
    import matplotlib.pyplot as plt
    plt.plot(x, y, label='unreuse square')
    x = np.arange(1, 200)
    # plot 1/ (a + b * x) with [a, b] = [1.17451165 0.06109705]
    y1 = 1 / (1.17451165 + 0.06109705 * x)
    plt.plot(x, y1, label='unreuse uni')
    # plot 1 / (a + b * x) with [a, b] = [1.88559658 0.15510953]
    y2 = 1 / (1.88559658 + 0.15510953 * x)
    plt.plot(x, y2, label='unreuse uni2')
    #plt.plot(x, func2(x, 0.192), label='dir reuse')
    # read unreusefit.csv file and scattar
    #print(popt)
    #plt.scatter(xy[:, 0], xy[:, 1], label='unreuse')
    #plt.scatter(xy[:, 0], xy[:, 2], label='dirreuse')
    #plt.legend()
    plt.show()
