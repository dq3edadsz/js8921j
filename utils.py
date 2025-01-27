import numpy as np
from MSPM.mspm_config import SOURCE_PATH, MAX_PW_NUM, MAX_PW_LENGTH, SEED_LEN, SCALAR, MAX_INT, SEED_MAX_RANGE, ALPHABET, TRAIN_LENGTH, Nitv, ALPHA
import random
from pylab import axes
import cupy as cp
from tqdm import tqdm
from cuda_voila import *
import pandas as pd
import torch
import pylcs
from multiprocessing import Pool
from time import time
import matplotlib.pyplot as plt
from data.data_processing import find_clusters, cluster_modification
import os
import scipy
from attack.para import check_pin_batch
from opts import opts
import pickle
from zxcvbn import zxcvbn
import json
import csv
args = opts().parse()
import cudf

class ExtendVault:
    def __init__(self):
        with open('/home/beeno/Dropbox/research_project/pycharm/crack_tools/wordlist/10_xato.txt', 'r') as f:
            self.pws = f.readlines()

    def expand(self, vault, Nprime):
        """
        :param vault: list of pw
        :param Nprime: target size of vault (after expansion)
        :return: target vault
        """
        for i in range(len(vault), Nprime):
            if random.random() > 1 / (0.02455*i**3 - 0.2945*i**2 + 3.409*i + 0.0852): #unreuse_p(i)
                # if reused (from the first i passwords)
                if random.random() < i * ALPHA / (i * ALPHA + 1 - ALPHA):
                    # reused but direct reuse
                    vault.append(random.choices(list(set(vault)), weights=[len(pw) for pw in list(set(vault))])[0])
                else:
                    # reused but modification
                    basepws = self.find_base_pw(list(set(vault)))
                    # random choice according to the length of the base pw
                    vault.append(self.reuse_pw(random.choice(basepws)))
            else:
                # if not reused
                vault.append(self.new_pw())
        return vault

    def reuse_pw(self, pw):
        # modify string 'pw' with minor changes
        # return the modified pw
        word = pw
        pw_list = []
        el = ['', '0', '1', '01', '001', '2', '02', '002', '3', '03', '003', '123', '12', '321',
              '666', '321', '1!', '1!!', '!', '!!', '!!!', '!@#', '?']
        # l33t chars - add more if you want
        leet = {'a': '@', 'e': '3', 'i': '1', 's': '5', 'o': '0'}

        def single_leet(text, dic): # at the front or end of password
            for item_ in dic.items():
                i, j = item_
                if text.find(i) < 2 or text.find(i) > len(text) - 3:
                    pw_list.append(text.replace(i, j))

        def dumb_suffix(text, suff):
            for lastidx in range(1, 3):
                for i in suff:
                    pw_list.append(text[:-lastidx] + str(i))
            for i in suff:
                pw_list.append(text + str(i))

        def dumb_preffix(text, suff):
            for lastidx in range(1,3):
                for i in suff:
                    pw_list.append(str(i) + text[lastidx:])
            for i in suff:
                pw_list.append(str(i) + text)

        # dumb suffix
        dumb_suffix(word, el)
        dumb_preffix(word, el)
        # single l33t
        single_leet(word, leet)
        # UPPERCASE
        uppercase = word.capitalize()
        dumb_suffix(uppercase, el)
        dumb_preffix(uppercase, el)

        # print out & good luck!
        remocc = set(pw_list)
        max_len = max([len(pw) for pw in remocc])
        resultpw = random.choice(list(remocc))
        while len(resultpw) < (7 if 7 <= max_len else max_len):
            resultpw = random.choice(list(remocc))
        return resultpw

    def new_pw(self):
        # generate a new pw
        sample_n = 10000
        start = random.randint(0, len(self.pws)-sample_n)
        pws = self.pws[start : start+sample_n].copy()
        random.shuffle(pws)
        for pw in pws:
            pw = pw.strip()
            if zxcvbn(pw)['score'] >= 2: #
                return pw

    def find_base_pw(self, distinctpws):
        distincts_copy = distinctpws.copy()
        basepws = []
        if len(distincts_copy) == 1:
            return distincts_copy
        while len(distincts_copy) > 0:
            pw = distincts_copy.pop()
            distinct_flag = 1
            for pw_ in distincts_copy:
                cm = longest_common_substring(pw.lower(), pw_.lower())
                if len(cm) >= int(max(len(pw), len(pw_))/2):
                    distinct_flag = 0
                    basepws.append(cm)
                    distincts_copy.remove(pw_)
            if distinct_flag == 1:
                basepws.append(pw.lower())
        basepws = list(set(basepws))
        return basepws

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]

    longest, x_longest = 0, 0

    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0

    return s1[x_longest - longest: x_longest]

def merge_sublist(list):
    total = []
    for sublist in list:
        total += sublist
    return total

def grouprandom_fixeditv(dv, seedconst=random.randint(0, MAX_INT)):
    """
    Group randomization (gr) is an operation used in fixed interval shuffling, which essentially inter-changes the keys
    within each group, a group is a set of keys with the same values in the dictionary.

    gr is only performed in the unfull interval, i.e., the last interval. Any previous interval does not need gr.
    :param dv: a dictionary
    :param seedconst:
    :return:
    """
    dv_ = {} # new dictionary with n fold of the interval
    vaultsize = len(dv)
    for ith in range(math.ceil(vaultsize / Nitv)):
        if ith == vaultsize // Nitv and args.fixeditv and args.fixeditvmode == 0: # last interval & needs padding
            for i in range(Nitv - vaultsize % Nitv):  # padding dummies
                dv.update({i + vaultsize: [-1] * 2})

        ### randomly change the order of the keys within each group
        # first group the keys by values
        dv_group = {}
        for k, v in list(dv.items())[ith * Nitv:(ith + 1) * Nitv]:
            if tuple(v) not in dv_group: # k is the metaid, v is the [pwid, distance from leaked pw]
                dv_group[tuple(v)] = [k]
            else:
                dv_group[tuple(v)].append(k)
        # second shuffle the keys within each group
        for k, v in dv_group.items():
            if ith == vaultsize // Nitv: # simulating gr on current interval
                random.shuffle(v)
            else:
                random.Random(seedconst+ith).shuffle(v)
        # third put the shuffled keys back to the dictionary according to original order of values
        for v in list(dv.values())[ith * Nitv:(ith + 1) * Nitv]:
            dv_[dv_group[tuple(v)].pop(0)] = v

    return dv_

def get_Mefp_candidatelist_para(logis, r, intraoffset, vaultsize, reshuedidx, gpuid):
    cp.cuda.Device(gpuid).use()
    # r starting from 0
    Nol_exps = [vaultsize-1] # [1, 3, 5, 7, vaultsize-1]   list of Nols to be tried in the experiment
    # results for attacker using sorted order
    metrics_d = {'r': r, 'Nol_exps': Nol_exps, 'Me': [0]*len(Nol_exps), 'verify_ends': [False]*len(Nol_exps), 'end_index': [None]*len(Nol_exps), 'fp': [1]*len(Nol_exps), 'fp_butrealmpw': [0]*len(Nol_exps), 'cracked_metaid': [cp.array([]) for _ in range(len(Nol_exps))], 'tried_metapwid': [cp.array([[]]) for _ in range(len(Nol_exps))], 'Me_eachbsz': []}

    metrics_d_intersec = {'r': r, 'Nol_exps': Nol_exps, 'Me': [0]*len(Nol_exps), 'verify_ends': [False]*len(Nol_exps), 'end_index': [None]*len(Nol_exps), 'fp': [1]*len(Nol_exps), 'fp_butrealmpw': [0]*len(Nol_exps), 'cracked_metaid': [cp.array([]) for _ in range(len(Nol_exps))], 'tried_metapwid': [cp.array([[]]) for _ in range(len(Nol_exps))], 'Me_eachbsz': []}

    mpwbatch = 2 if int(args.pinlength) > 6 else int(20) #20     if '6' in args.pin else (r+1) #debug setting 3   previously used "30 * 21 / vaultsize"
    # calculate batchsize for each ith_batch, key idea is to split r+1 into r and 1, meaning the last batch is always 1, for the real exclusively, the first r into split into [b1, b2, ..., bm] such that b1=b2=b_m-1=mpwbatch, bm = r - (m-1)*mpwbatch
    if r % mpwbatch == 0:
        bsz_list = [mpwbatch] * (r // mpwbatch) + [1]
    else:
        bsz_list = [mpwbatch] * (r // mpwbatch) + [r - (r // mpwbatch) * mpwbatch] + [1]
    for ith_batch in range(len(bsz_list)): # range(int(np.ceil((r + 1) / mpwbatch)))
        batch_start = sum(bsz_list[:ith_batch]) # ith_batch * mpwbatch
        batch_end = sum(bsz_list[:(ith_batch + 1)]) # (ith_batch + 1) * mpwbatch
        #ts = time()
        result_metapwid_batch = extract_logis(logis[:r+1][batch_start: batch_end], r - batch_start, intraoffset, reshuedidx, gpuid)
        #print('extract logis time: ', time()-ts)
        thelastbatch = ith_batch == (len(bsz_list) - 1)
        nol_max = (result_metapwid_batch[0].shape[0] - 2) if args.intersection else (result_metapwid_batch[0].shape[0] - 1)# vaultsize - 1
        for ith, nol in enumerate(Nol_exps):
            nol_ = nol if nol_max >= nol else nol_max
            parse_batchmpw(r - batch_start, [metrics_d, metrics_d_intersec], intraoffset, nol_ if not thelastbatch else nol_, nol_max if not thelastbatch else nol_max, ith, result_metapwid_batch, gpuid=gpuid)
    return metrics_d, metrics_d_intersec

def parse_batchmpw(r, metrics, intraoffset, nol, nol_max, ith, result_metapwid_batch, gpuid=0):
    # r starting from 0
    # result: size0 x ((r+1) * PIN_SAMPLE)
    # meta_pw_id: 3 x ((r+1) * Nol_max * PIN_SAMPLE)

    #nol_max = len(logis[0][0]) - 1
    #nol = nol if nol_max >= nol else nol_max
    if int(args.pinlength) <= 6:
        if not metrics[0]['verify_ends'][ith]:
            pass
            #perform_logis(metrics[0], 1, result_metapwid_batch[0], result_metapwid_batch[1], nol, nol_max, ith, intraoffset, r, gpuid)
        if not metrics[1]['verify_ends'][ith]:
            #pass
            perform_logis(metrics[1], 2, result_metapwid_batch[0], result_metapwid_batch[1], nol, nol_max, ith, intraoffset, r, gpuid)
    else:
        loopnum = 10
        rb = result_metapwid_batch[0]
        mpb = result_metapwid_batch[1]
        pins_tmp = PIN_SAMPLE // loopnum
        for ln in range(loopnum):
            rb_tmp = rb[:, rb.shape[1]//loopnum*ln : rb.shape[1]//loopnum*(ln+1)]
            mpb_tmp = mpb[:, mpb.shape[1]//loopnum*ln : mpb.shape[1]//loopnum*(ln+1)]
            if not metrics[0]['verify_ends'][ith]:
                perform_logis(metrics[0], 1, rb_tmp, mpb_tmp, nol, nol_max, ith, intraoffset % pins_tmp, r, gpuid)
            if not metrics[1]['verify_ends'][ith]:
                perform_logis(metrics[1], 2, rb_tmp, mpb_tmp, nol, nol_max, ith, intraoffset % pins_tmp, r, gpuid)

def extract_logis(logis, real_idx, intraoffset, reshuedidx, gpuid):
    cp.cuda.Device(gpuid).use()
    #ts = time()
    vaults, DVs = zip(*[logi for logi in logis])
    vaults, DVs = list(vaults), list(DVs)
    result_batch, meta_pw_id_batch = check_pin_batch(reshuedidx[0], vaults, DVs, real=real_idx == 0, gpuid=gpuid)

    if real_idx == 0:
        assert result_batch[:, 0].prod() == 1
        Nolmax = len(vaults[0]) - 1
        mapping = np.arange(PIN_SAMPLE)
        mapping[intraoffset] = 0
        mapping[0] = intraoffset
        result_batch = result_batch[:, mapping]
        meta_pw_id_batch = meta_pw_id_batch.reshape(2, PIN_SAMPLE, Nolmax)[:, mapping, :].reshape(2, -1)
    #print('extract (batch)', time() - ts)
    return result_batch, meta_pw_id_batch

def perform_logis(metrics, basic_requirement, result_batch, meta_pw_id_batch, nol, nol_max, ith, intraoffset, r, gpuid):
    cp.cuda.Device(gpuid).use()
    # list all indexs of meta-pw id possibly to be tried
    #ts = time()
    try_indicator = parse_metapwid(basic_requirement, result_batch, meta_pw_id_batch, nol, nol_max, metrics['cracked_metaid'][ith], metrics['tried_metapwid'][ith], gpuid)
    #print('perform_logis:parse meta pw id time: ', time()-ts)
    #ts = time()
    #ts1 = time()
    row_idx = list(np.arange(basic_requirement))
    row_idx.extend(list(np.arange(2, 2+nol)))
    all_verifies = result_batch[row_idx].prod(axis=0).astype(np.uint8)  # PIN_SAMPLE*bsz
    metrics['verify_ends'][ith] = all_verifies.nonzero()[0].shape[0] != 0 #(all_verifies - 1).prod() == 0
    metrics['end_index'][ith] = cp.where(all_verifies - 1 == 0)[0][0] if metrics['verify_ends'][ith] else all_verifies.shape[0] - 1
    intercept_idx = (metrics['end_index'][ith] + 1) * nol_max
    #tried_mask = try_indicator[:intercept_idx] == 1 # (end_index x nol_max)
    try_indicator[intercept_idx:] = 0
    metapwid_result = cp.transpose(result_batch[2:]).reshape(1, -1)
    cracked_mask = (try_indicator == 1) * (metapwid_result[0] == 1)  # (end_index x nol_max)
    #print('perform_logis:perform others: calculation: ', time()-ts1)
    #ts1 = time()
    for _, metaid in enumerate(meta_pw_id_batch[0, cracked_mask]):
        metrics['cracked_metaid'][ith] = cp.concatenate([metrics['cracked_metaid'][ith], cp.array([metaid])])
    triedpairs = cp.concatenate([meta_pw_id_batch[:, try_indicator == 1], metapwid_result[:, try_indicator == 1]], axis=0)
    if triedpairs.shape[1] > 0:
        if metrics['tried_metapwid'][ith].shape[1] > 0:
            metrics['tried_metapwid'][ith] = cp.concatenate([metrics['tried_metapwid'][ith], triedpairs], axis=1)
        else:
            metrics['tried_metapwid'][ith] = triedpairs
    #print('perform_logis:perform others: record tried entries times: ', time()-ts1)

    #ts1 = time()
    me_thisbsz = ((try_indicator.reshape(-1, nol_max).sum(1))[:metrics['end_index'][ith] + 1]).sum()
    #print('perform_logis:perform others: calculate Me times: ', time()-ts1)
    metrics['Me'][ith] += me_thisbsz
    metrics['Me_eachbsz'].append([meta_pw_id_batch.shape[1] / (nol_max * PIN_SAMPLE), me_thisbsz])
    # only the first end for each guess shot of a real password vault is allowed
    if metrics['verify_ends'][ith]:
        if metrics['end_index'][ith] // PIN_SAMPLE == r and metrics['end_index'][ith] % PIN_SAMPLE == intraoffset:  #
            # print('correctly verified!')
            metrics['fp'][ith] = 0
        if metrics['end_index'][ith] // PIN_SAMPLE == r and metrics['end_index'][ith] % PIN_SAMPLE != intraoffset:
            metrics['fp_butrealmpw'][ith] = 1
    #print('perform_logis:perform others time: ', time()-ts)

def parse_eachmpw(metrics, mpw_guessed, intraoffset, result, meta_pw_id, intersec=False):
    # verification method 1: in decreasing order of distance between \hat{pw} and pws in vault
    basic_requirement = 2 if intersec else 1
    basic = result[:basic_requirement].prod(axis=0) # PIN_SAMPLE
    nol_max = result.shape[0] - 2
    meta_pw_id = np.concatenate([meta_pw_id, np.transpose(result[2:2 + nol_max]).reshape(1, -1)]) # 3 x (PIN_SAMPLE*nol_max)
    for ith, nol in enumerate(metrics['Nol_exps']):
        nol = nol if nol_max >= nol else nol_max
        try_indicator = parse_metapwid(basic, result, meta_pw_id, nol, nol_max, metrics['cracked_metaid'][ith], list(metrics['tried_metapwid'][ith].keys())) # list all indexs of meta-pw id possibly to be tried
        all_verifies = basic * result[2:2 + nol].prod(axis=0) # PIN_SAMPLE
        metrics['verify_ends'][ith] = (all_verifies - 1).prod() == 0
        metrics['end_index'][ith] = np.where(all_verifies - 1 == 0)[0][0] if metrics['verify_ends'][ith] else PIN_SAMPLE - 1

        intercept_idx = (metrics['end_index'][ith]+1) * nol_max
        tried_mask = try_indicator[:intercept_idx] == 1 # (end_index x nol_max)
        cracked_mask = tried_mask * (meta_pw_id[2][:intercept_idx] == 1) # (end_index x nol_max)
        for metaid in (meta_pw_id[0, :intercept_idx][cracked_mask]).tolist():
            assert metaid not in metrics['cracked_metaid'][ith]
            metrics['cracked_metaid'][ith].append(metaid)
        triedpairs = meta_pw_id[:, :intercept_idx][:, tried_mask]
        for pairid in range(triedpairs.shape[1]):
            k_ = tuple(triedpairs[:2, pairid].tolist())
            assert k_ not in list(metrics['tried_metapwid'][ith].keys())
            metrics['tried_metapwid'][ith][k_] = triedpairs[2, pairid]

        metrics['Me'][ith] += ((try_indicator.reshape(PIN_SAMPLE, nol_max).sum(1))[:metrics['end_index'][ith]+1]).sum()
        # only the first end for each guess shot of a real password vault is allowed
        if metrics['verify_ends'][ith] and metrics['end_count'][ith] == 0:
            metrics['end_count'][ith] += 1
            if mpw_guessed and metrics['end_index'][ith] == intraoffset: #
                # print('correctly verified!')
                metrics['fp'][ith] = 0
            if not (mpw_guessed and metrics['end_index'][ith] == intraoffset) and mpw_guessed:
                metrics['fp_butrealmpw'][ith] = 1

def parse_metapwid(basic, result, meta_pw_id, nol, nol_max, cracked_metaid, checked_pairs, gpuid):
    # meta_pw_id: 3 x (PIN_SAMPLE * nol_max * batch) => try_indicator: PIN_SAMPLE * (r+1), each element for total trials of such PIN
    cp.cuda.Device(gpuid).use()

    batch = meta_pw_id.shape[1] / (nol_max * PIN_SAMPLE)

    threadsperblock = (256, 4)
    blockspergrid_x = math.ceil(PIN_SAMPLE * nol * batch/ threadsperblock[0])
    blockspergrid_y = math.ceil(nol / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    try_indicator = cp.array([1 if i < nol else 0 for i in range(nol_max)], dtype=bool) # size nol_max * PIN_SAMPLE * batch
    # repeat  "try_indicator" PIN_SAMPLE * batch times
    try_indicator = cp.tile(try_indicator, int(PIN_SAMPLE*batch))
    #ts = time()
    mark_totry[blockspergrid, threadsperblock](result[:basic].prod(0) if basic>1 else result[0], result[2:2+nol], nol, nol_max, try_indicator)
    #print('perform_logis:parse meta pw id:mark to try time: ', time()-ts)

    ## filter out the meta-pw id that has been tried
    #ts = time()
    if int(args.pinlength) <= 5:
        u, indices = fast_unique_v2(meta_pw_id, try_indicator)
    else: # x-pin (x>6)
        u, indices = iter_fast_unique(meta_pw_id, try_indicator, 10)
    #indices = indices[try_indicator[indices] == 1] # idx for size nol_max*PIN_SAMPLE
    uni_indicator = cp.zeros_like(try_indicator, dtype=bool)
    uni_indicator[indices] = 1
    try_indicator *= uni_indicator
    #print('perform_logis:parse meta pw id:filter out tried time: ', time()-ts)

    ## filter out the meta id would be tried that has been verified "real": meta_pw_id[2] == 1, find unique meta id with meta_pw_id[:2]==1
    #ts = time()
    col_mask = cp.transpose(result[2:]).reshape(-1) == 1 #meta_pw_id[2, :] == 1 np.transpose(result_batch[2:2 + result_batch.shape[0] - 2]).reshape(1, -1)
    if int(args.pinlength) <= 5:
        u, indices = fast_unique_v2(meta_pw_id[0], try_indicator * col_mask) # get unique meta id that has been verified "real"
    else:
        u, indices = iter_fast_unique(meta_pw_id[0], try_indicator * col_mask, divide=10) # get unique meta id that has been verified "real"
    #indices = indices[((meta_pw_id[0]+1) * try_indicator * col_mask).squeeze()[indices] != 0]
    nowcracked_metaid_idx = indices
    nowcracked_metaid = list(meta_pw_id[0, nowcracked_metaid_idx])
    #print('perform_logis:parse meta pw id:filter time: ', time()-ts)

    #ts = time()
    try_idxs = try_indicator.nonzero()[0]
    mask_try = cp.zeros_like(try_indicator, dtype=np.uint8)
    if try_idxs.shape[0] > 0:
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil(try_idxs.shape[0] / threadsperblock[0])
        if cracked_metaid.shape[0] > 0:
            tryrectify_previousfindreal[(blockspergrid_x, math.ceil(cracked_metaid.shape[0] / threadsperblock[1])), threadsperblock](try_idxs, meta_pw_id, cracked_metaid, mask_try)
        if nowcracked_metaid_idx.shape[0] > 0:
            tryrectify_nowfindreal[(blockspergrid_x, math.ceil(nowcracked_metaid_idx.shape[0] / threadsperblock[1])), threadsperblock](try_idxs, meta_pw_id, meta_pw_id[0, nowcracked_metaid_idx], nowcracked_metaid_idx, mask_try)
        if checked_pairs.shape[1] > 0:
            tryrectify_previouschecked[(blockspergrid_x, math.ceil(checked_pairs.shape[1] / threadsperblock[1])), threadsperblock](try_idxs, meta_pw_id, checked_pairs, mask_try)
    #print('perform_logis:parse meta pw id:filter out previous checked time (parallel): ', time() - ts)


    return try_indicator * (1 - mask_try)#cp.asnumpy(try_indicator) #.reshape(PIN_SAMPLE, nol_max).sum(1)

def fast_unique(arr, mask):
    # arr can be either of size 2xN or N
    # return unique elements and their indices where first appeared
    if arr.ndim == 2:
        # cp.cuda.Device(1).use()
        #idx = cudf.Series((arr[0][mask].astype(int) * 1024 + arr[1][mask].astype(int) * 99)).drop_duplicates().index.to_cupy() #cudf.Series((arr[0][mask].astype(int) * 1024 + arr[1][mask].astype(int) * 99)).drop_duplicates().index.to_cupy()
        idx = cudf.Series((arr[0][mask].astype(int) * 1024 + arr[1][mask].astype(int) * 99)).drop_duplicates().index.to_cupy()
    else:
        idx = cudf.Series(arr[mask]).drop_duplicates().index.to_cupy()
    idx = mask.nonzero()[0][idx]
    if arr.ndim == 2:
        return arr[:, idx], idx
    return arr[idx][None], idx

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

class Digit_Vault:
    def __init__(self, real_vault, vid=0, originsize=None):
        self.pw2id = {}
        if not args.expandtestset:
            real_vault_tmp = real_vault.copy()
        else:
            assert originsize is not None
            real_vault_tmp = real_vault[:originsize]
        self.leakpw = random.Random(vid).choice(real_vault_tmp[:-1] if args.intersection else real_vault_tmp) # fixed leakpw for each vault (randomness control)

        if not args.intersection: # self.possibleidx 1.used in ecnoding and decoding; 2.the first of which is leaked metaidx
            self.possibleidx = [i for i, x in enumerate(real_vault_tmp) if x == self.leakpw]
        else:
            self.possibleidx = [i for i, x in enumerate(real_vault_tmp) if x == self.leakpw and i != len(real_vault_tmp)-1]
        random.Random(vid).shuffle(self.possibleidx)
        self.leakmetaid = self.possibleidx[0] # fixed leakmetaid for each vault (randomness control)
        self.digit_realvault = self.create_dv(real_vault)[0]
        if real_vault_tmp[-1] == self.leakpw:
            self.possibleidx.append(len(real_vault_tmp)-1)

    def create_dv(self, vault):
        """
        digit vault (of variable "vault"): dictionary, key: meta id, value: [password id, edit distance from leakpw].
        password id is exclusive for each unique string password
        """
        for pw in vault:
            if pw not in list((self.pw2id).keys()):
                self.pw2id[pw] = len(self.pw2id)
        dv = {i: [self.pw2id[pw], pylcs.edit_distance(pw, self.leakpw)] for i, pw in enumerate(vault)}
        return dv, self.pw2id[self.leakpw]

    def dv2pwid(self, dv):
        return [x[0] for x in list(dv.values())]

    def dv2dis(self, dv):
        return [x[1] for x in list(dv.values())]

    def getpwid_bymetaid(self, dv, metaid):
        return dv[metaid][0]
def check_guessed_pw(path2guess, path2truth):
    with open(path2guess, 'r') as f:
        lines = f.readlines()
        guessed = [line.strip().split(':')[-1] for line in lines]
    with open(path2truth, 'r') as f:
        lines = f.readlines()
        truth = [line.strip() for line in lines]
    return truth[0] in guessed

def jaccard_similarity(list1, list2):
    # list1 is guess and list2 is truth
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1) # len(set1.union(set2))
    return intersection / union # reflect fraction of true guess

def write_list2file(path, l):
    with open(path, 'w') as f:
        if len(l) > 0:
            f.write(l.pop(0) + '\n')
        else:
            f.write('')
    with open(path, 'a') as f:
        for l_ in l:
            f.write(l_ + '\n')


def kl_divergence(p, q):
    return np.where(p != 0, p * np.log(p / q), 0)

def getmean(results, t, ith, thpin):
    result = results[len(t)*thpin:len(t)*(thpin+1)]
    result = np.concatenate([r[ith] for r in result], axis=0)
    result[:, 1] = np.where(result[:, 1] == 0, 1, result[:, 1])
    return np.mean(result[:, 0] / result[:, 1])

def split_pin(pin_path):
    # pin is a txt file, with each line as a pin code, split it into train-test file with 80-20 ratio
    with open(pin_path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    train = lines[:int(len(lines) * 0.8)]
    test = lines[int(len(lines) * 0.8):]
    pin_path = pin_path.split('.')[0]
    with open(pin_path + '_train.txt', 'w') as f:
        f.writelines(train)
    with open(pin_path + '_test.txt', 'w') as f:
        f.writelines(test)

def double_testvaults(results):
    expander = ExtendVault()
    decoy_vaults = {}
    for idx, r in enumerate(results[:75]):
        # if idx == unique_vids[14]:
        vault = r[2][0][0]
        # print('vault size: ', len(vault))
        decoy_vaults[str(idx)] = expander.expand(vault, len(vault) * 2)
    # write to json, with each element related to idx as key
    '''with open(write_dir + 'decoy_vaults' + '.json', 'w') as f:
        json.dump(decoy_vaults, f)'''
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/pastebintest_expanded/expaneded_vaults.json', 'w') as f:
        json.dump(decoy_vaults, f)

def multidouble_selectedvault(results, unique_vids):
    expander = ExtendVault()
    decoy_vaults = {}
    for idx in unique_vids:
        vault = results[idx][2][0][0]
        if len(vault) == 14:
            print(vault)
            vault = results[idx][2][0][0]
            for i in range(8):
                decoy_vaults[str(idx+i)] = expander.expand(vault.copy(), len(vault) * (i+3))
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/pastebintest_expanded/expaneded_avault.json', 'w') as f:
        json.dump(decoy_vaults, f)

def multisamesize_selectedvault(results, unique_vids):
    expander = ExtendVault()
    for idx in unique_vids:
        vault = results[idx][2][0][0]
        if len(vault) == 10:
            print(vault)
            vs = 60
            idx = 2345
            decoy_vaults = {str(idx): expander.expand(vault[:5], 120)}
            pool = Pool(10)
            workers = []
            for i in range(10):
                workers.append(pool.apply_async(expander.expand, (vault[:5], 120)))
            pool.close()
            pool.join()
            for i, w in enumerate(workers):
                decoy_vaults[str(idx+i+1)] = w.get()
            with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/synthesis_vaults/vaultsizeany/vaults.json', 'w') as f:
                json.dump(decoy_vaults, f)

def get_rrandmr(jsonpath, rr_cmr, rr_cmr_sep, write_lhm=False):
    # jsonpath to a number of password vaults, each is V, a list of passwords
    # rr: reuse rate, |V|/|set(V)|
    # mr: modification rate, |V|/|Cluster(V)|, Cluster shows the number of clusters of password based on same "base password"
    with open(jsonpath, 'r') as f:
        vaultdict = json.load(f)
    vaults = list(vaultdict.values())
    vss = [len(v) for v in vaults]
    unique_vss = sorted(list(set(vss)))  # unique vault size
    unique_vs =  vaults #[vaults[vss.index(vs)] for vs in unique_vss]#
    results = [] # in the format of "vault size, rr, mr"
    for k, v in tqdm(list(vaultdict.items())):
        cmr, _, clst_len, _ = cluster_modification(v, 0, 0)#1,1,1#
        results.append([uni_vault(v, 2, False, nitv=10)]) #  #clst_len, cmr
    # write to csv file
    with open('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/rrandmr.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    '''with open(jsonpath, 'w') as f:
        json.dump(vaultdict, f)'''

    if write_lhm:
        results = rr_cmr
        fail = pd.read_csv('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/fail.csv', sep=',', header=None).values # table_data/rrandmr.csv len(unique_vs) x (num of experiments with different paramenters)
        blocked = pd.read_csv('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/blocked.csv', sep=',', header=None).values
        success_ratio_lm, success_ratio_hm = np.zeros((len(rr_cmr_sep)//2, fail.shape[1])), np.zeros((len(rr_cmr_sep)//2, fail.shape[1]))
        blocked_lhm, fail_lhm = [[], []], [[], []]
        for i, rcs in enumerate(rr_cmr_sep):
            rr_l, rr_h, cmr_l, cmr_h = rcs
            idx = (results[:, 1] >= cmr_l) * (results[:, 1] < cmr_h) * (results[:, 0] >= rr_l) * (results[:, 0] < rr_h)
            if i % 2 == 0:
                success_ratio_lm[i//2] = np.sum(fail[idx, :] >= 80, axis=0) / idx.sum()
            else:
                success_ratio_hm[i//2] = np.sum(fail[idx, :] >= 80, axis=0) / idx.sum()
            blocked_lhm[i % 2].append(blocked[idx, :])
            fail_lhm[i % 2].append(fail[idx, :])
        results_blocked, reusults_fp, results_ratio = [], [], []
        for lhid in range(len(blocked_lhm)):
            blocked_lorh = np.concatenate(blocked_lhm[lhid], axis=0) # num_lorhm x num_exps
            fail_lorh = np.concatenate(fail_lhm[lhid], axis=0)
            fp_lorh = fail_lorh - blocked_lorh
            results_blocked.append(list(np.mean(blocked_lorh, axis=0)))
            reusults_fp.append(list(np.mean(fp_lorh, axis=0)))
            results_ratio.append(list((fail_lorh >= 80).sum(0) / fail_lorh.shape[0]))


def uni_utility(vault, cons, nitv=None, single=True):
    # calculating the number of unique mappings of web and pws in a vault (a list of pws)
    # cons: 1: vanilla shuffling, 2: fixed iterval shuffling

    itvsize = len(vault) if cons == 1 else nitv
    if cons == 2:
        assert nitv is not None
    uni = 0
    for n in range(int(np.ceil(len(vault) / itvsize))):
        vault_tmp = vault[n * itvsize:(n + 1) * itvsize]
        # using log for factorial division for stability
        numerator = np.log(scipy.special.factorial(len(vault_tmp)))
        denominator = np.log(scipy.special.factorial(np.array([vault_tmp.count(pw) for pw in set(vault_tmp)]))).sum() if not (not single and cons==1) else 0 # 0 suggests one to one mapping, do not consider reuse
        uni = uni + numerator - denominator
    return round(np.log10(np.exp(uni)), 3)

def uni_vault(vault, cons, single=True, nitv=None):
    # single (single=True): uni is calculated over vault alone; multi version (single=False): uni is calculated over vault (vt+1) and vault[:-1] (vt)
    if single:
        return uni_utility(vault, cons, nitv)
    elif cons == 1:
        return uni_utility(vault, cons, nitv, single=single) + uni_utility(vault[:-1], cons, nitv, single=single)
    else:
        assert nitv is not None
        if (len(vault)-1) % nitv == 0:
            return uni_utility(vault, cons, nitv, single=single)
        else:
            if len(vault) % nitv > 0:
                return uni_utility(vault, cons, nitv, single=single) + uni_utility(vault[len(vault)//nitv*nitv:-1], cons, nitv, single=single)
            else:
                return uni_utility(vault, cons, nitv, single=single) + uni_utility(vault[(len(vault) // nitv-1) * nitv:-1], cons, nitv, single=single)


def assign_gidx(mr_rcm, mrsep, rcmsep): # spliting the samples into 4 groups
    if mr_rcm[0] <= mrsep[0] and mr_rcm[1] <= rcmsep[0]:
        groupidx = 1 # lclm
        '''elif mr_rcm[0] <= mrsep[0] and mr_rcm[1] > rcmsep[1]:
            groupidx = 2 # lchm
        elif mr_rcm[0] > mrsep[1] and mr_rcm[1] <= rcmsep[0]:
            groupidx = 3 # hclm'''
    else:
        groupidx = 2 # hchm
    return groupidx


def main():
    already_capture = np.ones(231, dtype=bool) # 231
    # open csv file '/home/beeno/Dropbox/encryption/writing/bubble/results_handling/Data 65.csv in numeric form
    re = pd.read_csv('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/valuewise_plcmr_newtestset.csv', sep=',', header=None) #table_data/rrandmr.csv
    re = re.values
    print('Averages:', re.mean(0))
    cmr_sep = 0.25 #0.25
    #gen_dual_statistic(re, rr_cmr_sep=[6, 9, 0, 0.1], rank_thr=0.005, rank_idx=7, already_capture=already_capture)

    '''jsonpaths = ['/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/pastebin/fold2_pb/fold_0_1.json',
                 '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2_bc50/fold_0_1.json',
                 '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2_bc100/fold_0_1.json',
                 '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2_bc200/fold_0_1.json']
    avgbeta_jsons = []
    for jsonpath in jsonpaths:
        print('reading json file: ', jsonpath)
        with open(jsonpath, 'r') as f:
            vaultdict = json.load(f)
        avgbeta_json = []
        for nitv_ in range(2, 15):
            avgbeta = 0
            for k, v in list(vaultdict.items()):
                avgbeta += uni_vault(v, 2, False, nitv=nitv_)
            avgbeta /= len(vaultdict)
            avgbeta_json.append(avgbeta)
        avgbeta_jsons.append(avgbeta_json)
    # plot the average beta value as y-axis and the nitv as x-axis, with each json as a line
    fig, ax = plt.subplots()
    for i, avgbeta_json in enumerate(avgbeta_jsons):
        ax.plot(np.arange(2, 15), avgbeta_json, label=jsonpaths[i].split('/')[-2])
    ax.set_xlabel('nitv')
    ax.set_ylabel('Average Beta')
    ax.legend()
    plt.show()'''


if __name__ == '__main__':
    main()