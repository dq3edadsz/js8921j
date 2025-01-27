from linecache import getline
from random import seed, randint, choice, shuffle
import random
from copy import deepcopy
from hashlib import sha256
import logging
from collections import defaultdict
import copy
from time import time
import gc
import os
import struct
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from mpire import WorkerPool
from concurrent.futures import ProcessPoolExecutor, as_completed
import secrets
from MSPM.incre_pw_coding import Incremental_Encoder
import math
import json
from tqdm import tqdm
import numpy as np
import pickle
from Vault.utils import set_crypto, Counter
from utils import grouprandom_fixeditv
from opts import opts
args = opts().parse()
from MSPM.mspm_config import *
PATH = SOURCE_PATH
'''import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})'''

cipherlen_pw = SEED_LEN * 8
dte_global = Incremental_Encoder(args.nottrain)

def list_shuffle(mpw, pin, vault_size, recover=False, existing_list=None):
    """
    passed unit test function 'tes_list_shuffle()'
    shuffle in direct or fixed interval way
    :param mpw:
    :param pin:
    :param vault_size:
    :param recover:
    :return:
    """
    itvsize = Nitv if args.fixeditv else vault_size
    shuffled_list = []
    if existing_list is not None:
        assert len(existing_list) == vault_size
    for ith in range(math.ceil(vault_size / itvsize)):
        shuffled_list_tmp = list(np.arange(0, itvsize)) if existing_list is None else copy.deepcopy(existing_list)[ith*itvsize:(ith+1)*itvsize]
        if args.fixeditv and args.fixeditvmode == 1 and ith==vault_size//itvsize and vault_size%itvsize != 0: # requires no padding
            shuffled_list_tmp = shuffled_list_tmp[:vault_size % itvsize]
        rng = random.Random(int(sha256(str.encode(pin + str(ith))).hexdigest(), 16) % int(1e8)) if args.fixeditv else random.Random(int(sha256(str.encode(pin)).hexdigest(), 16) % int(1e8)) #
        rolls = [rng.randint(0, len(shuffled_list_tmp)-1) for _ in range(len(shuffled_list_tmp))] # vault_size
        if not recover:
            for i in range(len(shuffled_list_tmp)-1, -1, -1):
                shuffled_list_tmp[i], shuffled_list_tmp[rolls[i]] = shuffled_list_tmp[rolls[i]], shuffled_list_tmp[i]
        else:
            for i in range(len(shuffled_list_tmp)):
                shuffled_list_tmp[i], shuffled_list_tmp[rolls[i]] = shuffled_list_tmp[rolls[i]], shuffled_list_tmp[i]
        sub_list = list(np.array(shuffled_list_tmp) + ith * itvsize) if existing_list is None else shuffled_list_tmp
        shuffled_list.extend(sub_list)
    return shuffled_list

def createvault_frompws(pws):
    # create pw vault, a dictionary with domain as key and pw as value
    # load domains from os.path.join(SOURCE_PATH, 'data/top500domains_cloudflare.csv')
    domain_file = os.path.join(SOURCE_PATH, 'data/top500domains_cloudflare.csv')
    with open(domain_file, 'r') as f:
        domains = f.readlines()

    # create vault
    vault = {}
    for i in range(len(pws)):
        vault[domains[i].strip()] = pws[i]
    return vault

def get_hash(s):
    assert isinstance(s, str)
    h = sha256()
    h.update(s.encode())
    return int(h.hexdigest()[:32], 16)

def seed_or_mpw_substitution(pw_seed, pw, pw_block_prior, independent_indicator, independent_prob_list, mpw_tmp, mpw_history,alp=1):
    if pw not in pw_block_prior: # if not same password before
        return pw_seed, 1, mpw_tmp
    else: # if has same password
        samepw_index = [i for i, pw_ in enumerate(pw_block_prior) if pw_==pw]
        for idx_ in samepw_index:
            if independent_indicator[idx_] == 1:
                return pw_seed, 1, mpw_history[idx_]
        if len(independent_prob_list) > 0:
            threshold = 1
            for idx_ in samepw_index:
                if independent_indicator[idx_] == -1:
                    threshold *= (1 - independent_prob_list[idx_]) / alp
            # threshold is prob of able to independent encode the seed (do not need to substitute)
            if random.random() > threshold:
                return [randint(0, MAX_INT) for _ in range(len(pw_seed))], 0, mpw_tmp
        return pw_seed, 1, mpw_tmp


def encrypt_ithblock(que, ts__, new_pws, block_id_start, block_id_end, pw_block_new, decoympws, independent_prob_list, allpws_spmprob, real_vault_id, mpw, T, salts):
    print('threading time', time() - ts__)
    len_oldblock = len(pw_block_new) - len(new_pws)
    independent_indicator = [-1] * len_oldblock
    mpw_history = [None] * len_oldblock
    incre_mc = [bytearray() for _ in range(T)]

    for block_id in tqdm(range(block_id_start, block_id_end)):
        for ith_newpw, pw in enumerate(new_pws):
            if lencheck(pw):
                raise Exception("Password length not qualified!", pw)
            mpw_tmp = copy.deepcopy(decoympws[block_id][ith_newpw]) if block_id != real_vault_id else mpw
            pw_seed, independent_flag, _, _ = dte_global.encode_pw_frompriorpws(pw_block_new[:(len_oldblock + ith_newpw)], pw, allpws_spmprob[:(len_oldblock + ith_newpw)])
            if block_id != real_vault_id and independent_flag == 1:
                pw_seed, independent_flag, mpw_tmp = seed_or_mpw_substitution(pw_seed, pw, pw_block_new[:(len_oldblock + ith_newpw)], independent_indicator, independent_prob_list, mpw_tmp, mpw_history)
            independent_indicator.append(independent_flag)
            mpw_history.append(mpw_tmp)
            ctr = Counter.new(128, initial_value=(len_oldblock + ith_newpw)*cipherlen_pw // 16) # "cipherlen_pw // 16" is the number of blocks per pw cipher
            aes = set_crypto(mpw_tmp, salts[ith_newpw + len_oldblock], ctr=ctr)
            incre_mc[block_id] += aes.encrypt(struct.pack('{}L'.format(len(pw_seed)), *pw_seed))
    #return incre_mc, [fg for fg in independent_indicator if fg != -1], time()
    que.put((incre_mc, [fg for fg in independent_indicator if fg != -1], time()))


class Vault:
    def __init__(self, mpw, pin, T=1000):
        self.mpw = mpw
        self.pin = pin
        self.T = T
        self.init_decoympws()
        self.real_vault_id = get_hash(mpw) % T
        self.pw_block = []
        self.salts = [] # same length as dm_block
        self.dm_block = []
        self.real_dm = None # debug only
        self.real_pw = None
        self.MC = [bytearray() for _ in range(T)]
        self.independent_indicator = [[] for _ in range(T)]
        self.sizeondisk = [0, 0]

    def init_decoympws(self):
        decoympws = [line.strip() for line in
         open('/home/beeno/Dropbox/research_project/pycharm/crack_tools/wordlist/prepared/10_xato_prepared.txt', 'r') if
         len(line.strip()) > 6]
        self.decopmpws_init = defaultdict(list) # mpwe prepared for initiliazation
        for mpw in decoympws[:3000000]:
            self.decopmpws_init[get_hash(mpw) % self.T].append(mpw)

        self.decopmpws_update = defaultdict(list)
        for bid_ in self.decopmpws_init.keys():
            self.decopmpws_update[bid_].extend(self.decopmpws_init[bid_][-2:])

    def init_vault(self, pwvault):
        # pwvault: a dictionary with domain as key and pw as value
        # init_vault returns (shuffled) plaintext of domain block & ciphertext of pw blocks (duplicated T blocks)
        tst = time()
        self.real_dm, self.real_pw = list(pwvault.keys()), list(pwvault.values())
        self.dm_block, self.pw_block = list(pwvault.keys()), list(pwvault.values())
        tsp = self.save_vault()
        return tsp - tst

    def save_vault(self):
        if len(self.MC[0]) // cipherlen_pw < len(self.pw_block):
            if len(self.pw_block) - len(self.MC[0]) // cipherlen_pw > 0:
                self.encrypt_block(self.pw_block[:len(self.MC[0]) // cipherlen_pw], self.pw_block, use_multithread=True)
            else:
                self.encrypt_block(self.pw_block[:len(self.MC[0]) // cipherlen_pw], self.pw_block, use_multithread=False)
        storage_file = {'metadata': list_shuffle(self.mpw, self.pin, len(self.dm_block), existing_list=self.dm_block),
                        'MC': self.MC, 'salts': self.salts, 'realdm': self.real_dm, 'realpw': self.real_pw}
        tsp = time()

        with open("storagefile.pkl", 'wb') as f:
            pickle.dump(storage_file, f)
        # get size in KB and MB
        self.sizeondisk = [os.path.getsize("storagefile.pkl") / 1024, os.path.getsize("storagefile.pkl") / 1024 / 1024]
        return tsp

    def load_vault(self):
        with open("storagefile.pkl", 'rb') as f:
            storagefile = pickle.load(f)
        self.real_dm = storagefile['realdm']
        self.real_pw = storagefile['realpw']
        self.MC = storagefile['MC']
        self.dm_block = list_shuffle(self.mpw, self.pin, len(storagefile['metadata']), recover=True, existing_list=storagefile['metadata'])
        self.salts = storagefile['salts']
        self.pw_block = self.decrypt_block(self.MC[self.real_vault_id])
        #assert self.real_pw == self.pw_block
        if self.real_pw != self.pw_block:
            print('unequal_load results!')
            print(self.real_pw)

    def add_salts(self, target_length):
        while len(self.salts) < target_length:
            self.salts.append(secrets.token_hex(8))

    def encrypt_block(self, pw_block_old, pw_block_new, use_multithread=False):
        """

        :param pw_block_old: a pw list
        :param pw_block_new: a pw list with n more pws than pw_block_old (n>=1)
        :param MC: a list of self.T ciphertext lists, each ciphertext list is the same length as pw_block_old (encrypted version)
        :return: new MC, a list of self.T ciphertext lists, each ciphertext list is the same length as pw_block_new (encrypted version)
        """
        ts__ = time()
        assert sum([len(pw_block_old) == len(cipher)//cipherlen_pw for cipher in self.MC]) == self.T
        ts_ = time()
        new_pws = pw_block_new[len(pw_block_old):]
        allpws_spmprob = [np.log(np.array(dte_global.spm.encode_pw(pw_)[1])).sum() for pw_ in pw_block_new]
        independent_prob_list = [dte_global.encode_pw_frompriorpws(pw_block_old[:i], pw_block_old[i], allpws_spmprob[:i])[2] for i in range(len(pw_block_old))]
        self.add_salts(len(pw_block_new))
        if len(pw_block_new) - len(pw_block_old) == 1:
            print('prepare time:', time()-ts__)
        if use_multithread:
            ts__ = time()
            nproc = 25 if len(pw_block_new) - len(pw_block_old) > 1 else 8 # initiliazation takes more threads
             #WorkerPool(n_jobs=nproc)#
            print('before call multi time:', time() - ts__)
            works = []
            batchsz = int(np.ceil(self.T / nproc))
            batch_blockid = [list(range(i * batchsz, (i + 1) * batchsz)) if (i + 1) * batchsz <= self.T else list(range(i * batchsz, self.T)) for i in range(nproc)]
            #pool = multiprocessing.Pool(processes=nproc)
            gc.disable()
            que = multiprocessing.Queue()
            for batchbid in batch_blockid:
                decoympw_batch = [[] for _ in range(self.T)]
                if len(pw_block_new) - len(pw_block_old) > 1:
                    for bid_ in batchbid:
                        decoympw_batch[bid_] = self.decopmpws_init[bid_]
                else:
                    for bid_ in batchbid:
                        decoympw_batch[bid_] = self.decopmpws_update[bid_]
                if 1 in batchbid:
                    ts__ = time()
                #works.append(pool.apply_async(encrypt_ithblock, (ts__, new_pws, batchbid[0], batchbid[-1]+1, pw_block_new, decoympw_batch, independent_prob_list, allpws_spmprob, self.real_vault_id, self.mpw, self.T, self.salts)))
                procs = multiprocessing.Process(target=encrypt_ithblock, args=(que, ts__, new_pws, batchbid[0], batchbid[-1]+1, pw_block_new, decoympw_batch, independent_prob_list, allpws_spmprob, self.real_vault_id, self.mpw, self.T, self.salts))
                works.append(procs)
                procs.start()
            #pool.close()
            #pool.join()
            ts__ = time()
            for w in works:
                mc_, independent_ind, ts___ = que.get()
                print('return mc time:', time() - ts___)
                for ith_ in range(len(self.MC)):
                    self.MC[ith_] += mc_[ith_]
                    self.independent_indicator.extend(independent_ind)
            if len(pw_block_new) - len(pw_block_old) == 1:
                print('receive & concatenate time:', time() - ts__)
            for w in works:
                w.terminate()
            gc.enable()
            if len(pw_block_new) - len(pw_block_old) > 1:
                gc.collect()
            #pool.stop_and_join()

        else:
            mc_, independent_ind = encrypt_ithblock(new_pws, 0, self.T, pw_block_new, self.decopmpws, independent_prob_list, allpws_spmprob, self.real_vault_id, self.mpw, self.T, self.salts)
            for ith_ in range(len(self.MC)):
                self.MC[ith_] += mc_[ith_]
                self.independent_indicator.extend(independent_ind)
        if len(pw_block_new) - len(pw_block_old) == 1:
            print('encrypt time:', time()-ts_)
        assert sum([len(pw_block_new) == len(cipher) // cipherlen_pw for cipher in self.MC]) == self.T

    def decrypt_block(self, cipher_block):
        seed_block = []
        for i in range(len(cipher_block) // cipherlen_pw):
            ctr = Counter.new(128, initial_value=i * cipherlen_pw // 16)
            aes = set_crypto(self.mpw, self.salts[i], ctr=ctr)
            pw_seed = aes.decrypt(cipher_block[i*cipherlen_pw:(i+1)*cipherlen_pw])
            seed_block.extend(list(struct.unpack('{}L'.format(len(pw_seed)//8), pw_seed)))

        return dte_global.decode_pw(seed_block)

    def decrypt_1pw(self, cipher_block, metaid):
        seed_block = []
        for i in range(metaid+1):
            ctr = Counter.new(128, initial_value=i * cipherlen_pw // 16)
            aes = set_crypto(self.mpw, self.salts[i], ctr=ctr)
            pw_seed = aes.decrypt(cipher_block[i*cipherlen_pw:(i+1)*cipherlen_pw])
            seed_block.extend(list(struct.unpack('{}L'.format(len(pw_seed) // 8), pw_seed)))

        return dte_global.decode_pw(seed_block, seed_block[-SEED_LEN:], i=metaid)

    def get_pw(self, dm):
        ts = time()
        if dm not in self.dm_block:
            print("Domain not found in the vault!")
            return None
        pw = self.decrypt_1pw(self.MC[self.real_vault_id], self.dm_block.index(dm))
        #assert pw == self.real_pw[self.real_dm.index(dm)]
        if pw != self.real_pw[self.real_dm.index(dm)]:
            print('unequal search result!')
        #print("Website password for domain {} is {}".format(dm, pw))
        return time() - ts


    def add_pw(self, pw, dm):
        tst = time()
        if dm in self.dm_block:
            print("Domain already exists in the vault!")
            tsp = time()
        else:
            self.load_vault()
            #print("Adding domain {} with password {}".format(dm, pw))
            self.dm_block.append(dm)
            self.pw_block.append(pw)
            vault_tmp = {self.dm_block[i]: self.pw_block[i] for i in range(len(self.dm_block))}
            #vault_tmp = grouprandom_fixeditv(vault_tmp)
            self.dm_block = list(vault_tmp.keys())
            self.real_dm.append(dm)
            self.real_pw.append(pw)
            tsp = self.save_vault()
        return tsp - tst

def test_system_cost(vs_list, T_list, rn):
    """

    :param vs: vault size of vault to be used to test
    :param T: expansion of pw_blocks i.e., len(self.MC)
    :param rn: repeat number for each test (given vs and T)
    :return:
    """
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/Vault_system/samplevaults.json','r') as f:
        pwvaults = list(json.load(f).values())
    results = {}
    progress = 1
    for seedid, vs in enumerate(vs_list):
        selected_vaults = []
        random.Random(seedid).shuffle(pwvaults)
        for pwv in pwvaults:
            if len(pwv) == vs:
                selected_vaults.append(pwv)
                if len(selected_vaults) == rn:
                    break

        if len(selected_vaults) < rn:
            selected_vaults = selected_vaults*rn
            selected_vaults = selected_vaults[:rn]

        for T in T_list:
            results[(vs, T)] = defaultdict(list)
            for pwvault in selected_vaults:
                print("Progress: {}/{}".format(progress, len(vs_list) * len(T_list)), '; with # test vaults', len(selected_vaults))

                # create vault under pws and domains to be used in bubble system
                vault_t = Vault('qnmeu12n90s1', '124871', T=T)
                init_timecost_singleshot = vault_t.init_vault(createvault_frompws(pwvault))
                results[(vs, T)]['init_timecost'].append(init_timecost_singleshot)
                results[(vs, T)]['sizeondisk'].append(vault_t.sizeondisk)

                # test serach with domain
                vault_t.load_vault()
                search_timecost_median = []
                for i in range(len(vault_t.dm_block)):
                    search_timecost_median.append(vault_t.get_pw(vault_t.dm_block[i]))
                results[(vs, T)]['search_timecost'].append(np.mean(np.array(search_timecost_median)))

                # test add_pw and then search
                add_timecost_singleshot = vault_t.add_pw('da2we2j9dj@23qdq', 'bbbhjajkdh32.com') # pwvault[-1]
                #print(add_timecost_singleshot)
                for i in range(len(vault_t.dm_block)):
                    _ = vault_t.get_pw(vault_t.dm_block[i])
                results[(vs, T)]['add_timecost'].append(add_timecost_singleshot)
            progress += 1
    # write results to file
    with open('cost_experiments.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    vs_list = [20, 50, 100, 200] #[200] # vs_list=
    T_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000] #[4000] # T_list=
    test_system_cost(vs_list=vs_list, T_list=T_list, rn=20)

    with open('cost_experiments.pkl', 'rb') as f:
        results = pickle.load(f)
    results = results
    # print median value of each
    key_list = ['init_timecost', 'search_timecost', 'add_timecost', 'sizeondisk']
    n_T, n_vs = len(T_list), len(vs_list)
    # print results of each key once in matrix
    for exp_k in key_list:
        print(exp_k)
        result_exp_k = []
        for k, dict_ in results.items():
            values = np.mean(np.array(dict_[exp_k]), axis=0)
            if values.size != 1:
                values = values[1]
            result_exp_k.append(round(float(values), 4))
        print_arr = np.array(result_exp_k).reshape(n_vs, n_T).transpose()
        # print each row without bracket
        for row in print_arr:
            print(','.join([str(r) for r in row]))