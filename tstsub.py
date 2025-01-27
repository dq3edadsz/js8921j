import subprocess
import cupy
import numpy as np
import pickle
import json # import json
import orjson
import os
import torch
import cudf
import cupy as cp
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
from time import time

def fast_unique_v2(arr, mask, u_concat=None):
    # arr can be either of size 2xN or N
    # return unique elements and their indices where first appeared
    if arr.ndim == 2:
        # determine wether uint8
        if arr.dtype == np.uint8 and u_concat is None: # transform into int16 for pytorch unique
            arr_t = torch.from_dlpack(arr[:, mask].astype(np.int16))
        else:
            arr_t = torch.from_dlpack(arr[:, mask]) if u_concat is None else torch.concat(u_concat, dim=1)
        unique, idx = torch.unique(arr_t, dim=1, sorted=True, return_inverse=True)
        idx = torch.scatter_reduce(torch.full(size=[unique.size(1)], fill_value=0, device=idx.device, dtype=torch.int64), dim=0, index=idx, src=torch.arange(idx.size(0), device=idx.device), reduce="amin", include_self=False)
        unique = arr_t[:, torch.sort(idx)[0]] # arr_t[:, idx] #
        idx = cp.from_dlpack(idx)
    else:
        idx = cudf.Series(arr[mask]).drop_duplicates().index.to_cupy() if u_concat is None else cudf.Series(cp.concatenate(u_concat, axis=1)[0]).drop_duplicates().index.to_cupy()
    if isinstance(mask, list):
        idx = cp.concatenate(mask, axis=0)[idx]
    else:
        idx = mask.nonzero()[0][idx]
    if arr.ndim == 2:
        return unique, idx
    return arr[idx][None], idx

def iter_fast_unique(arr, mask, divide=10):
    if arr.ndim == 2:
        axis1size = arr.shape[1]
    elif arr.ndim == 1:
        axis1size = arr.size
    else:
        raise ValueError("Input array must be 1D or 2D.")
    batch = axis1size // divide
    idx_concat = []#cp.empty(axis1size, dtype=cp.bool_)
    u_concat = []
    for x in range(divide):
        if arr.ndim == 2:
            u_x, indices_x = fast_unique_v2(arr[:, x*batch:(x+1)*batch], mask[x*batch:(x+1)*batch])
        else:
            u_x, indices_x = fast_unique_v2(arr[x*batch:(x+1)*batch], mask[x*batch:(x+1)*batch])
        idx_concat.append(cp.sort(x*batch + indices_x)) # [x*batch + indices_x] = True
        u_concat.append(u_x)
    u, indices = fast_unique_v2(arr, idx_concat, u_concat)
    return u, indices


'''read_path = '/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_200_ts.json'
with open(read_path, "rb") as f:
    vaultdict = orjson.loads(f.read())
items_ = list(vaultdict.items())
toselect_size = dict()
for vs in range(2, 201):
    toselect_size[vs] = 0
toselect_num = 300
testset = {}
for k, v in tqdm(items_):
    if toselect_size[len(v)] < toselect_num:
        testset[k] = v
        toselect_size[len(v)] += 1
    if sum([nums == toselect_num for nums in toselect_size.values()]) == len(toselect_size):
        break
write_path = '/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_200_minits.json'
with open(write_path, "w") as f:
    json.dump(testset, f)'''

ts = time()
randomseeds = np.random.randint(low=0, high=0xFFFFFFFFFF, size=80*10**6)#cp.random.randint(low=0, high=0xFFFFFFFFFF, size=80*10**6).get()#
print('used ', time()-ts, 's')


# write back to file
with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc200/100vaults/fold_0_1.json', 'r') as f: #'/hdd1/bubble_experiments/results/MSPM/bc100/vaults_cmr_pl8/fold_0_1.json'
   vaults = json.load(f)

'''from utils import uni_vault
import matplotlib.pyplot as plt
beta_avg = []
for itvsize in range(2, 14):
    beta_epoch = []
    for v_ in vaults.values():
        beta_epoch.append(uni_vault(v_, 2, True, nitv=itvsize))
    beta_avg.append(np.mean(np.array(beta_epoch)))
plt.plot(beta_avg)
plt.show()'''


ranks = np.genfromtxt('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/rank.csv', delimiter=',')
print('avg rank', ranks.mean(0))
print('avg rank (min)', min(list(ranks.mean(0))))

blocked = np.genfromtxt('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/blocked.csv', delimiter=',')
print('blocked prob', blocked.mean(0))

fps = np.genfromtxt('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/Fp.csv', delimiter=',')


failed = np.genfromtxt('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/fail.csv', delimiter=',')
print('fp prob', failed.mean(0) - blocked.mean(0))
print('resist prob', failed.mean(0))
