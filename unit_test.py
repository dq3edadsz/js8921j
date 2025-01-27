import tqdm
import multiprocessing
from MSPM.mspm_config import *
from opts import opts
args = opts().parse()
import copy
import numpy as np
import math
import itertools
import pandas as pd
import pickle
from data.data_processing import cluster_modification
import random
from attack.weapons import list_shuffle
from attack.para import check_logical
from collections import defaultdict
import os
from utils import grouprandom_fixeditv
import json
import matplotlib.pyplot as plt
from time import time
from utils import Digit_Vault, uni_vault, ExtendVault, assign_gidx
from cuda_voila import convert2seed, recovershuffle
from numba.cuda.random import create_xoroshiro128p_states
expander = ExtendVault()

def test_grouprandom():
    """
    test the grouprandom function (gr)
    :return:
    """
    pws = ["zach1996", "chanman96"]
    changed_times, total_trails = 0, 500
    for _ in range(total_trails):
        # add a new random string or current one
        pws.append(''.join([chr(random.randint(97, 122)) for _ in range(8)]) if random.random() > 0.1 else pws[-1])
        dv = Digit_Vault(pws)

        ### 1. test whether grouprandom is performed inside each group
        dvt = dv.create_dv(pws)[0]
        dvt_gred = grouprandom_fixeditv(dvt)
        # gr should keep the items same abielt in different order
        assert len(dvt_gred) == len(dvt)
        for k in list(dvt.keys()):
            assert dvt_gred[k] == dvt[k]
        # gr may change the order of items
        if list(dvt.keys()) != list(dvt_gred.keys()):
            changed_times += 1

        ### 2. test wether seedconst is working among dvt and dvt_1, we expect the intervals with size Nitv before the last (unfull) interval are the same
        dvt_1 = dv.create_dv(pws[:-1])[0]
        seed_ = random.randint(0, 2**32-1)
        dvt_gred_1 = grouprandom_fixeditv(dvt_1, seedconst=seed_)
        dvt_gred = grouprandom_fixeditv(dvt, seedconst=seed_)
        # all intervals before the last
        if len(dvt) % Nitv != 0:
            assert list(dvt_gred.keys())[:len(dvt)//Nitv*Nitv] == list(dvt_gred_1.keys())[:len(dvt_1)//Nitv*Nitv]
    print("grouprandom_fixeditv changed the order of items {} times out of {} trials".format(changed_times, total_trails))



def test_list_shuffle():
    ### test vanilla shuffling
    '''print('test vanilla shuffling')
    for vs in range(1, 500):
        itvsize = vs
        a = list_shuffle(mpw='anduiq', pin='123445', vault_size=vs, recover=False, existing_list=None)
        assert len(a) % vs == 0
        for i in range(int(math.ceil(vs/itvsize))):
            #print('=========' + str(i) + '=========')
            #print('max', max(a[i*Nitv:(i+1)*Nitv]), 'min', min(a[i*Nitv:(i+1)*Nitv]))
            assert max(a[i*itvsize:(i+1)*itvsize]) < (i+1)*itvsize and min(a[i*itvsize:(i+1)*itvsize]) >= i*itvsize

        a_re = list_shuffle(mpw='anduiq', pin='123445', vault_size=vs, recover=True, existing_list=a)
        assert list(np.arange(vs)) == a_re
    print('vanilla shuffling test passed')'''

    ### test fixed interval shuffling
    print('test fixed interval shuffling')
    for vs in range(1, 500):
        itvsize = vs if not args.fixeditv else Nitv
        paddedsize = int(math.ceil(vs/Nitv))*Nitv if args.fixeditv and args.fixeditvmode==0 else vs
        existing_list = []
        # extend sublist el to existing_list, el is shuffle within each interval
        for i in range(int(math.ceil(vs / itvsize))):
            el = list(range(i * itvsize, (i + 1) * itvsize))
            if args.fixeditv and args.fixeditvmode == 1 and i == vs // itvsize:
                el = el[:vs % itvsize]
            random.shuffle(el)
            existing_list.extend(el)
        a = list_shuffle(mpw='anduiq', pin='123445', vault_size=paddedsize, recover=False, existing_list=existing_list)
        assert len(a) % paddedsize == 0
        # print(a)
        for i in range(int(math.ceil(paddedsize / itvsize))):
            # print('=========' + str(i) + '=========')
            # print('max', max(a[i*Nitv:(i+1)*Nitv]), 'min', min(a[i*Nitv:(i+1)*Nitv]))
            assert min(a[i * itvsize:(i + 1) * itvsize]) >= i * itvsize
            assert max(a[i * itvsize:(i + 1) * itvsize]) < (i + 1) * itvsize
        a_re = list_shuffle(mpw='anduiq', pin='123445', vault_size=paddedsize, recover=True, existing_list=a)
        assert existing_list == a_re
    print('fixeditv shuffling test passed')

def test_pastebin():
    pastebin = {}
    flst = os.listdir(PASTB_PATH)
    for fname in flst:
        with open(os.path.join(PASTB_PATH, fname)) as f:
            pastebin.update(json.load(f))
    # pastebin is a dictionary of k-v (vault_id - vault), vault is a list of passwords
    # record size of each vault in pastebin
    vault_size = []
    for k, v in pastebin.items():
        vault_size.append(len(v))

    # plot distribution of vault size
    plt.hist(vault_size, bins=100)
    plt.xlabel('vault size')
    plt.ylabel('number of vaults')
    plt.show()

def test_check_logical():
    vault_size = 40
    loginum = 1
    idx_gt_shuffled = list_shuffle(mpw='anduiq', pin='123445', vault_size=40, recover=False)
    reshuedidx = check_logical(idx_gt_shuffled, loginum, vault_size, gpuid=0)

    print('Checking one of shuffle mode (1. shuffling; 2. fixed interval shuffling):',
          1 if not args.fixeditv else 2)

    itvsize = Nitv if args.fixeditv else vault_size
    padded_size = int(math.ceil(vault_size / itvsize) * itvsize)
    print('Checking shape of reshuffled index: ',
          reshuedidx.shape[0] == loginum and reshuedidx.shape[1] == PIN_SAMPLE * padded_size)

    print('Checking context of reshuffled index...')
    # expect under mode 1 (shuffling), each row of reshuffled index is different than other rows
    if not args.fixeditv:
        check = True
        for i in range(loginum):
            for j in range(loginum):
                if i != j and np.array_equal(reshuedidx[i], reshuedidx[j]):
                    check = False
                    break
        print('Results (expecting row of reshuffled index is different than other rows) ', check)
    # expect under mode 2 (fixed interval shuffling), each row of reshuffled index is the same
    else:
        check = True
        for i in range(loginum):
            for j in range(loginum):
                if i != j and not np.array_equal(reshuedidx[i], reshuedidx[j]):
                    check = False
                    break
        print('Results (expecting row of reshuffled index is the same) ', check)

    reshuedidx1 = check_logical(idx_gt_shuffled, loginum, vault_size, gpuid=0, seed_=10)
    reshuedidx2 = check_logical(idx_gt_shuffled, loginum, vault_size, gpuid=0, seed_=10)
    print('Checking seed control:', np.array_equal(reshuedidx1, reshuedidx2))



def test_gpu_shuffle_reshuffle(loginum=10, vault_size=40):
    idx_gt_shuffled = list_shuffle(mpw='anduiq', pin='123445', vault_size=vault_size, recover=False)

    # use random sampling for six-digit PIN (PIN_SAMPLE)
    itvsize = Nitv
    padded_size = int(math.ceil(vault_size / itvsize) * itvsize)

    print('Checking re/shuffle index and reshuffled result...')
    with cuda.gpus[0]:
        # get reshuffle index
        threadsperblock = 32
        blockspergrid = (loginum * PIN_SAMPLE * padded_size + threadsperblock - 1) // threadsperblock
        reshuidx = np.zeros(loginum * PIN_SAMPLE * padded_size, dtype=np.uint8)
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.randint(0, MAX_INT))
        # get "reshuidx" as reshuflle index
        convert2seed[blockspergrid, threadsperblock](rng_states, 0, 1, loginum*PIN_SAMPLE*padded_size, reshuidx,
                                                     itvsize, padded_size//itvsize)
        # recover shuffle index
        blockspergrid = (loginum * PIN_SAMPLE + threadsperblock - 1) // threadsperblock
        reshuedidx = np.array(idx_gt_shuffled*loginum * PIN_SAMPLE, dtype=np.uint8)
        recovershuffle[blockspergrid, threadsperblock](reshuidx, reshuedidx, padded_size)

    # check reshuffle index under fixed interval shuffling
    check = True
    for pinidx in range(PIN_SAMPLE):
        reshuidx_tmp = reshuidx[pinidx*padded_size:(pinidx+1)*padded_size]
        for i in range(int(math.ceil(vault_size / itvsize))):
            if max(reshuidx_tmp[i*itvsize:(i+1)*itvsize]) > (i+1)*itvsize-1 or \
                    min(reshuidx_tmp[i*itvsize:(i+1)*itvsize]) < i*itvsize:
                check = False
                break
    print('Assert reshuffle index: ', check)

    # check reshuffle index under fixed interval shuffling
    check = True
    for pinidx in range(PIN_SAMPLE):
        reshuidx_tmp = reshuedidx[pinidx * padded_size:(pinidx + 1) * padded_size]
        for i in range(int(math.ceil(vault_size / itvsize))):
            if max(reshuidx_tmp[i * itvsize:(i + 1) * itvsize]) > (i + 1) * itvsize - 1 or \
                    min(reshuidx_tmp[i * itvsize:(i + 1) * itvsize]) < i * itvsize:
                check = False
                break
    print('Assert reshuffled result: ', check)
    print('Checking re/shuffle index and reshuffled result... Done')

    print("Checking whether can reshuffled to original order by using correct reshuffle index")
    with cuda.gpus[0]:
        # get reshuffle index
        threadsperblock = 32
        reshuidx = list_shuffle(mpw='anduiq', pin='123445', vault_size=vault_size, recover=True)
        reshuidx = np.array(loginum * PIN_SAMPLE * reshuidx, dtype=np.uint8)
        # recover shuffle index
        blockspergrid = (loginum * PIN_SAMPLE + threadsperblock - 1) // threadsperblock
        reshuedidx = np.array(idx_gt_shuffled*loginum * PIN_SAMPLE, dtype=np.uint8)
        recovershuffle[blockspergrid, threadsperblock](reshuidx, reshuedidx, padded_size)

    # check reshuffle index under fixed interval shuffling
    check = True
    for pinidx in range(PIN_SAMPLE):
        reshuidx_tmp = reshuedidx[pinidx * padded_size:(pinidx + 1) * padded_size]
        for i in range(int(math.ceil(vault_size / itvsize))):
            if (reshuidx_tmp-np.arange(padded_size)).sum() != 0:
                check = False
                break
    print('Can recover to origin?: ', check)
    print('Checking Done')

    return reshuedidx.reshape(loginum, padded_size*PIN_SAMPLE)


def test_convert2seed():
    ### test the setting arg.fixeditv = True and arg.fixeditvmode = 1wes
    print('test fixed interval shuffling')
    for vs in range(1, 100):
        with cuda.gpus[1]:
            itvsize = vs if not args.fixeditv else Nitv
            paddedsize = int(math.ceil(vs / Nitv)) * Nitv if args.fixeditv and args.fixeditvmode == 0 else vs
            # get reshuffle index
            threadsperblock = 32
            blockspergrid = (PIN_SAMPLE * paddedsize + threadsperblock - 1) // threadsperblock
            reshuidx = np.zeros(PIN_SAMPLE * paddedsize, dtype=np.uint8)
            rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.randint(0, MAX_INT))
            # get "reshuidx" as reshuflle index
            convert2seed[blockspergrid, threadsperblock](rng_states, 0, 1, PIN_SAMPLE * paddedsize, reshuidx, itvsize, paddedsize // itvsize, paddedsize if args.fixeditv and args.fixeditvmode==1 else -1)
        assert reshuidx.shape[0] % paddedsize == 0
        # print(a)
        for j in range(PIN_SAMPLE):
            a = reshuidx[j * paddedsize:(j + 1) * paddedsize]
            for i in range(int(math.ceil(paddedsize / itvsize))):
                # print('=========' + str(i) + '=========')
                # print('max', max(a[i*Nitv:(i+1)*Nitv]), 'min', min(a[i*Nitv:(i+1)*Nitv]))
                assert min(a[i * itvsize:(i + 1) * itvsize]) >= i * itvsize
                if (i + 1) * itvsize > paddedsize:
                    # catch the error and print the error message
                    try:
                        assert max(a[i * itvsize:(i + 1) * itvsize]) < paddedsize
                    except AssertionError:
                        print("vault size:", vs, "itvsize:", itvsize, "paddedsize:", paddedsize)
                        print('Error: ', a[i * itvsize:(i + 1) * itvsize], '>=', paddedsize)
                else:
                    try:
                        assert max(a[i * itvsize:(i + 1) * itvsize]) < (i + 1) * itvsize
                    except AssertionError:
                        print("vault size:", vs, "itvsize:", itvsize, "paddedsize:", paddedsize)
                        print('Error: ', a[i * itvsize:(i + 1) * itvsize], '>=', (i + 1) * itvsize)
    print('convert2seed test passed')

def select_testset_bycr():
    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/fold_0_1.json', 'r') as f:
        vaultdict = json.load(f)
    vault_ofsamesize = defaultdict(list)
    for k, v in vaultdict.items():
        vault_ofsamesize[len(v)].append(v)
    testset = {} # rr 2 ~ 85
    itms = list(vaultdict.items())
    random.Random(0).shuffle(itms)
    selectsize = 55
    select_range = 2
    repeat_num = 5
    for rn in range(repeat_num):
        select_vaultlist = []
        for vs_ in range(selectsize-select_range, selectsize+select_range+1):
            select_vaultlist += vault_ofsamesize[vs_]
        toselect_rr = [5,8,11,14,18,26,28] # list(set([round(len(v) / len(set(v))) for v in select_vaultlist])) [5,8,11,14,18,26,28]
        for k, v in itms:
            if np.abs(len(v)-selectsize) > select_range or round(len(v) / len(set(v))) not in toselect_rr: # round(len(v) / len(set(v)))
                continue
            elif k not in list(testset.keys()):
                testset[k] = v
                toselect_rr.remove(round(len(v) / len(set(v)))) #
                if len(toselect_rr) == 0:
                    break
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_size55_differentleaksforvault/fold_0_1.json', 'w') as f:
        json.dump(testset, f)

def select_testset_beta():
    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/fold_0_1.json', 'r') as f:
        vaultdict = json.load(f)
    testset = {} # rr 2 ~ 85
    itms = list(vaultdict.items())
    random.Random(0).shuffle(itms)
    select_beta = list(range(0, 13)) * 10
    for k, v in tqdm.tqdm(itms):
        if len(v) > 99:
            continue
        v_beta = uni_vault(v, 1, single=True, nitv=None)
        if int(v_beta) in select_beta:
            testset[k] = v
            select_beta.remove(int(v_beta)) # round(len(v) / len(set(v)))
            if len(select_beta) == 0:
                break
    print(select_beta)
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/fp_exp/fold_0_1.json', 'w') as f:
        json.dump(testset, f)


def select_testset_byvs():
    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_200_minits.json', 'r') as f:
        vaultdict = json.load(f)
    testset = {} # rr 2 ~ 85
    itms = list(vaultdict.items())
    random.shuffle(itms)

    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/pastebin_ts.json', 'r') as f:
        pbvaults = json.load(f)
    vaultsize2select = [len(v) for v in pbvaults.values()] * 1

    vs2num = defaultdict(int)
    sample_set = {}
    for k, v in itms:
        if len(v) <= 10 or len(v) > 200 or vs2num[len(v)] >= 40: # len(v) not in vaultsize2select: #
            continue
        vs2num[len(v)] += 1
        sample_set[k] = v
        #vaultsize2select.remove(len(v))
    #testset = sample_set
    samples = list(sample_set.items())
    random.shuffle(samples)
    for k, v in samples:
        testset[k] = v
        if len(testset) == 100:
            break
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc200/mversion/fold_0_1.json', 'w') as f:
        json.dump(testset, f)


def select_testset_byrrcm():
    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_50_ts.json', 'r') as f:
        vaultdict = json.load(f)
    testset = {}  # rr 2 ~ 85
    itms = list(vaultdict.items())
    random.shuffle(itms)
    toselect_pl = [round(num, 1) for num in list(range(6, 14))] * 20 #[8]*20 #
    toselect_cmr = [0.2] #[0, 0.1, 0.2, 0.3, 0.4, 0.5] #
    toselect_tuple = []
    for v1 in toselect_pl:
        for v2 in toselect_cmr:
            toselect_tuple.append((v1, v2))
    pool = multiprocessing.Pool(processes=30)

    workers = []
    for i in range(len(itms)):
        k, v = itms[i]
        if len(v) < 30 or len(v) >= 40:
            continue
        elif random.random() <= 1:
            workers.append(pool.apply_async(cluster_modification, (v, i, k)))
    pool.close()
    pool.join()
    k2cm = defaultdict(list)
    for w in workers:
        v2, k, v1, _ = w.get()
        k2cm[k] = [v1, v2]

    for k, v in tqdm.tqdm(itms):
        if len(k2cm[k]) == 0:
            continue
        totuple = (k2cm[k][0], k2cm[k][1])
        for toselect_t in toselect_tuple:
            if np.abs(totuple[0] - toselect_t[0]) < 0.3 and np.abs(totuple[1] - toselect_t[1]) < 0.05: #np.abs(len(v) - 65) < 3 and
                if k not in list(testset.keys()):
                    testset[k] = v
                    toselect_tuple.remove(toselect_t)
                    break
    print('Number of vaults selected:', len(testset))
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc50/vaults_pl_cmr015_025/fold_0_1.json', 'w') as f:
        json.dump(testset, f)


def select_testset_4groups():
    # select testset from 4 sub groups {(low mr (6~18), low rcm (0~0.2)), (low mr (6~18), high rcm (0.2~0.5)), (high mr (18~36), low rcm (0~0.2)), (high mr (18~36), high rcm (0.2~0.5))}
    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/fold_0_1.json', 'r') as f:
        vaultdict = json.load(f) # do not pick any vault that has been chosen
    testset = {}  # rr 2 ~ 85
    itms = list(vaultdict.items())
    random.Random(0).shuffle(itms)
    toselect_rr = [list(range(5, 16)), list(range(29, 44))]  # cr
    toselect_cm = [[0, 0.1, 0.1], [0.4, 0.4, 0.4]]  # [round(num, 1) for num in list(np.arange(0, 1, 0.05))] # relative cluster modification rate
    toselect_tuple = []
    for rr in toselect_rr:
        for cm in toselect_cm:
            toselect_tuple.extend([[r1,m1] for r1 in rr for m1 in cm])
    pool = multiprocessing.Pool(processes=32)
    workers = []
    for i in range(len(itms)):
        k, v = itms[i]
        workers.append(pool.apply_async(cluster_modification, (v, i, k)))
    pool.close()
    pool.join()
    k2cm = {}
    for w in workers:
        cm, k, clst_len = w.get()
        k2cm[k] = [cm, clst_len]
    for k, v in itms:
        mr_rcm = (len(v) / k2cm[k][1], k2cm[k][0])
        for totuple in toselect_tuple:
            if round(mr_rcm[0]) == totuple[0] and round(mr_rcm[1], 1) == totuple[1] and mr_rcm[1] != 0: # get group idx
                testset[k] = v
                toselect_tuple.remove(totuple)
                break
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_4groups/fold_0_1.json', 'w') as f:
        json.dump(testset, f)

def synvaults2testset():
    with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/fold_0_1.json', 'r') as f:
        vaultdict = json.load(f)
    vault_ofsamesize = defaultdict(list)
    for k, v in vaultdict.items():
        vault_ofsamesize[len(v)].append(v)
    syn_vs = [85, 105, 125, 145, 165]
    syn_pren = [10, 10, 10, 10, 10]
    max_n_each = 15
    select_rr = 15
    select_range = 3
    testset = {}
    for nth, vs in enumerate(syn_vs):
        n = 1
        for k, v in vaultdict.items():
            if np.abs(len(v) - vs) < 3:
                v_expanded = expander.expand(v[:syn_pren[nth]], vs)
                if np.abs(round(len(v_expanded) / len(set(v_expanded))) - select_rr) < select_range and n <= max_n_each:
                    testset[k] = v_expanded
                    n += 1
    # update the testset into path below
    target_path = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_rr15_vswise/fold_0_1.json'
    with open(target_path, 'r') as f:
        testset_pre = json.load(f)
    testset_pre.update(testset)
    with open(target_path, 'w') as f:
        json.dump(testset_pre, f)


def duplcate_testset(ntimes):
    # duplicate to make testset of size (x ntimes)
    # given a testpath, read a testset, a dictionary of k-v (vault_id - vault), and duplicate each vault in testset ntimes, each replica should have different vault_id
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_size55_differentleaksforvault/fold_0_1_copy.json', 'r') as f:
        testset_ = json.load(f)
    testsetpath = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_size55_differentleaksforvault/fold_0_1.json'
    with open(testsetpath, 'r') as f:
        testset = json.load(f)
    for i, k in enumerate(list(testset.keys())):
        if i < 14:
            del testset[k]
    #testset_ = copy.deepcopy(testset)
    testset_.update(testset)
    for k, v in testset.items():
        for i in range(ntimes - 1):
            vid = random.randint(0, 2**14-1)
            while str(vid) in testset_:
                vid = random.randint(0, 2**14-1)
            testset_[str(vid)] = v
    with open(testsetpath, 'w') as f:
        json.dump(testset_, f)

def testset_join():
    testset_path1 = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_plcmr/fold_0_1.json'
    with open(testset_path1, 'r') as f:
        testset1 = json.load(f)
    testset1_coy = copy.deepcopy(testset1)
    testset_path2 = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_plcmr/newselect.json'
    with open(testset_path2, 'r') as f:
        testset2 = json.load(f)
    todelete_list = []
    for k, v in testset2.items():
        if round(np.array([len(pw_) for pw_ in v]).mean(), 4) <= 14: # if true, remove from testset2
            todelete_list.append(k)
    # get the first three of testset2 into testset1
    testset1.update({k: v for k, v in list(testset2.items())[:3]})
    with open(testset_path1, 'w') as f:
        json.dump(testset1, f)


def testset_prune():
    testset_path = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/fp_exp/fold_0_1.json'
    with open(testset_path, 'r') as f:
        testset = json.load(f)
    toprune_idx2k = {}
    for i, (k, v) in enumerate(list(testset.items())):
        if round(np.array([len(pw_) for pw_ in v]).mean(), 4) > 13.5: # if true, remove from testset2
            toprune_idx2k[i] = k
    # remove all the experiments in toprune_idx2k, in path testset_path there are several folders, each folder contains experiment under certain parameters
    # each v's results in the folder is in the name of e.g., "results_v261_shot4.data", (in this name, means the 261th vault, results of 4-th repeat)
    # remove all the results of the vaults in toprune_idx2k (including multiple repeats) and rename the rest of the results following the order of the vaults in testset
    # process each folder
    for folder in os.listdir('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_plcmr/'):
        if 'attack' not in folder:
            continue
        folderpath = os.path.join('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc/vaults_plcmr', folder)
        removed_files = 0
        for file in os.listdir(folderpath):
            # if filename includes i-th vault, remove the file
            for i in toprune_idx2k.keys():
                if ('v' + str(i)) in file:
                    os.remove(os.path.join(folderpath, file))
                    removed_files += 1
                    break
        print('removed', removed_files, 'files in', folder)
        # rename the rest of the files
        # get the idx behind v from the filename, and compare the idx within keys of toprune_idx2k, count how many idxs in the toprune_idx2k are smaller than the idx from the filename
        # process each file
        origin_nfiles_eachshot = 262
        repeattimes = 5
        for idx in range(origin_nfiles_eachshot):
            if idx in toprune_idx2k.keys():
                continue
            for rn in range(repeattimes):
                offset = sum([1 if i < idx else 0 for i in toprune_idx2k.keys()])
                if offset == 0:
                    break
                os.rename(os.path.join(folderpath, 'results_v' + str(idx) + '_shot' + str(rn) + '.data'), os.path.join(folderpath, 'results_v' + str(idx-offset) + '_shot' + str(rn) + '.data'))

    # write back testset file after pruning
    for k in toprune_idx2k.values():
        del testset[k]
    with open(testset_path, 'w') as f:
        json.dump(testset, f)



from numba import cuda
@cuda.jit
def add_kernel(a, b):
  """
  A simple CUDA kernel that adds elements of two arrays and stores the result in a third array.

  Args:
      a: The first input array (type: int[]).
      b: The second input array (type: int[]).
      c: The output array to store the sum (type: int[]).
  """
  x, y = cuda.grid(2)
  if x < a.shape[0] and y < a.shape[1]:
      sx, sy = cuda.gridsize(2)
      a[x, y] = sx
      b[x, y] = sy

if __name__ == "__main__":
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc200/mversion/fold_0_1.json', 'r') as f:
        pbvaults = json.load(f)
    '''vaultdict_ = {k: v for k, v in list(vaultdict.items())[:100]}
    for i, v in enumerate(list(vaultdict.values())[:100]):
        assert v == list(vaultdict_.values())[i]
    for k, v in vaultdict_.items():
        assert v == vaultdict[k]
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/bc200/moreversions/fold_0_1.json', 'w') as f:
        json.dump(vaultdict_, f)'''

    #dir_path = '/hdd1/bubble_experiments/results/MSPM/bc200/100vaults/attack_result_4000_testdataid0_cons1_pin6'
    dir_path = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPMnobubble/bc200/100vaults/attack_result_1000_testdataid0_cons1_pin3'
    re = []
    for shotid in range(10):  # [2]:#
        for vid in range(100):
            filename = 'results_v' + str(vid) + '_shot' + str(shotid) + '.data'
            with open(os.path.join(dir_path, filename), 'rb') as f:
                r = pickle.load(f)
                re.append(r)
    print(np.array([r[1][0][0][0]['r_three'] for r in re]).mean(axis=0))
    print((np.array([r[1][0][0][0]['r_three'] for r in re]) == 0).sum(0) / 1000)

    def print_nxm(vec, rows, cols):
        # if vec is a scalar
        if len(vec.shape) == 0:
            return
        for i in range(rows):
            for j in range(cols):
                if j == cols - 1:
                    print(vec[i*cols + j])
                else:
                    print(vec[i*cols + j], end=',')
    n_rows, n_cols = 1,1
    blocked = np.genfromtxt('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/blocked.csv', delimiter=',')
    failed = np.genfromtxt('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/fail.csv', delimiter=',')
    fp = failed - blocked
    print('avg blocked:', blocked.mean(axis=0))
    print_nxm(blocked.mean(axis=0), n_rows, n_cols)
    np.savetxt("/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/blocked_avg.csv", blocked.mean(axis=0).reshape(n_rows, n_cols), delimiter=",")
    print('avg fp:', fp.mean(axis=0))
    print_nxm(fp.mean(axis=0), n_rows, n_cols)
    np.savetxt("/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/fp_avg.csv", fp.mean(axis=0).reshape(n_rows, n_cols), delimiter=",")
    print('avg failed:', failed.mean(axis=0))
    print_nxm(failed.mean(axis=0), n_rows, n_cols)
    np.savetxt("/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/failed_avg.csv", failed.mean(axis=0).reshape(n_rows, n_cols), delimiter=",")

    #testset_prune()

    #select_testset_beta()
    #select_testset_4groups()
    #select_testset_byrrcm()
    #select_testset_bycr()
    #duplcate_testset(5)
    #select_testset_byvs()
    #synvaults2testset()

    re = pd.read_csv('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/valuewise_plcmr_bc50_cmr02.csv', sep=',', header=None)
    re = re.values
    print('test set total vaults:', re.shape[0])
    # check group size separated  by length
    '''for pl in range(6, 14):
        print('pl:', pl, 'group size:', ((re[:, 0] > (pl-0.3)) * (re[:, 0] < (pl+0.3))).sum())'''

    # check group size separated by cluster modification rate (cmr)
    for cmr in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        print('cmr:', cmr, 'group size:', ((re[:, 1] > (cmr-0.03)) * (re[:, 1] < (cmr+0.03))).sum())