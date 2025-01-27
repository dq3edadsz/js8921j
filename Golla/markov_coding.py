import copy
import random
from Golla.SSPM.pcfg import RulePCFGCreator
from Golla.utils import gen_gilst
import struct
import logging
import pylcs
import numpy as np
from MSPM.mspm_config import *
from multiprocessing import Pool
from time import time
import os
import json
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
#from Golla.configs.configure import *
#from Golla.ngram_creator import NGramCreator
from numba import cuda
import cupy as cp
from Golla.SPM.configs.configure import Configure
from Golla.SPM.ngram_creator import NGramCreator
from opts import opts
args = opts().parse()
#from numba import jit

class Markov_Encoder:
    """
    1. each encoding sequence has to be chosen randomly (according to sequence probability)
    2. seed sequence has to be padded to certain length (a hyperparameter #len_padded)
    3. modulo of spm and sspm can be different (seed itself is random, unless modulo is discovered)
    4.
    """

    def __init__(self, train=True):
        # initialization of spm
        CONFIG = Configure({'name': '4-gram'}, train=False)
        length, progress_bar = 6, True
        # initialization of gi (based on groups of edit distance 0, 1, 2, 3, 4 (related passwords) and 5 (unrelated passwords))
        self.gi_lst = gen_gilst()

        self.spm = NGramCreator({
            "name": (
                "NGramCreator, Session: {}, Length: {}, Progress bar: {}".format(CONFIG.NAME, length, progress_bar)),
            "ngram_size": CONFIG.NGRAM_SIZE,
            "training_file": CONFIG.TRAINING_FILE,
            "length": length, # minimum length of password trained for spm
            "laplace_smooth": CONFIG.LAPLACE_SMOOTH,
            "ls_compensation": CONFIG.LS_COMPENATE,
            "progress_bar": progress_bar,
            "train": CONFIG.TRAIN,
        })
        self.spm.load("ip_list")
        self.spm.load("cp_list")
        self.spm.load("ep_list")
        print("single password model loading done ...")

        # initialization of sspm
        self.sspm = RulePCFGCreator(args)
        self.sspm.load()
        #print("single similar password model loading done ...")

    def to_cpu(self):
        self.spm.ip_list = self.spm.ip_list.copy_to_host()
        self.spm.cp_list = self.spm.cp_list.copy_to_host()

    def to_gpu(self):
        with cuda.gpus[args.gpu]:
            self.spm.ip_list = cuda.to_device(self.spm.ip_list)
            self.spm.cp_list = cuda.to_device(self.spm.cp_list)

    def encode_pw(self, pws):
        """
            note the password length better falls in the range of [MIN_PW_LENGTH, MAX_PW_LENGTH]
        :param concatenation: of pws in seeds
        :param pws: all the passwords in the vualt
        :return: concatenation of seeds over all pws
        """
        concatenation,seed2editdist = [], []
        # select most frequent password as basepw
        basepw = max(pws, key=pws.count)
        seed, prob = self.spm.encode_pw(basepw)
        baseprob = np.log(np.array(prob)).sum()
        seed.insert(0, self.encode_pathnum(random.randint(0, 5))[0]) # encoding whole password by markov model
        prob_lst = []
        concatenation.extend(seed)
        for pw in pws:
            seed_spm, prob_spm = self.spm.encode_pw(pw)
            if self.sspm.check_pw_pair(basepw, pw): # check whether encode as related passwords (related passwords, i.e., edit distance <= 4)
                seed, prob, modi_path = self.sspm.encode_pw(basepw, pw)
                assert len(modi_path) == 1 or len(modi_path) == 2

                edit_dist = pylcs.edit_distance(basepw, pw)
                seed2editdist.append(edit_dist)
                if edit_dist > 0 and modi_path[1] != 'Delete': # e.g., basepw='123456' => pw='123456ab' edit_dist=2 'Add' mode; basepw='123456' => pw='12345ab' edit_dist=2 'Delete-then-add' mode
                    seed.extend(seed_spm[len(pw) - 1 -1*edit_dist : -3])
                    prob.extend(prob_spm[-1*edit_dist-1 : -1])
                prob = np.log(np.array(prob)).sum() + baseprob
                extra = SEED_LEN - len(seed)
                seed.extend(self.sspm.convert2seed(0, 1, extra))
            else: # unrelated password
                seed2editdist.append(5)
                seed_path, prob_path, modi_path = self.sspm.encode_pw(basepw, pw)
                seed_path.extend(seed_spm)
                seed = seed_path
                assert len(seed) == SEED_LEN
                prob_spm.extend(prob_path)
                prob = np.log(np.array(prob_spm)).sum()

            prob_lst.append(prob)
            concatenation.extend(seed)
        return concatenation, seed2editdist, prob_lst

    def decode_pw(self, concatenation):
        """

        :param concatenation: of seeds
        :return:
        """
        pw_num = len(concatenation) / SEED_LEN - 1 # len(pw_lst), include 1 additional base password "basepw" in the front
        pw_lst, prob_spm_mspm = [], []

        seed = concatenation[1: SEED_LEN]
        basepw, basepw_prob = self.spm.decode_pw(seed)

        for i in range(int(pw_num)):
            seed = concatenation[(i+1)*SEED_LEN : (i+2)*SEED_LEN]
            modi_path, prob_lst = self.sspm.decode_modipath(seed)
            if modi_path[0] == 0:
                pw_lst.append(basepw)
                prob_spm_mspm.append([basepw_prob, basepw_prob + np.log(np.array(prob_lst)).sum()])
            elif modi_path[0] == 5:
                unrelated_pw, unrelated_prob = self.spm.decode_pw(seed)
                pw_lst.append(unrelated_pw)
                prob_spm_mspm.append([unrelated_prob, unrelated_prob + np.log(np.array(prob_lst)).sum()])
            else:
                if 'Delete' == modi_path[1]:
                    pw_ = basepw[:-1]
                    prob_ = 0
                elif 'Add' == modi_path[1]:
                    pw_, prob_ = self.spm.decode_pw_continue(seed, copy.deepcopy(basepw), target_len=(len(basepw)+modi_path[0]))
                else:
                    pw_, prob_ = self.spm.decode_pw_continue(seed, copy.deepcopy(basepw)[:-1], target_len=(len(basepw)+modi_path[0]-1))
                pw_lst.append(pw_)
                spm_prob = np.log(np.array(self.spm.encode_pw(pw_)[1])).sum() if len(pw_)>3 and len(pw_)<30 else 0
                prob_spm_mspm.append([spm_prob, prob_ + basepw_prob + np.log(np.array(prob_lst)).sum()])
        assert len(pw_lst) == len(prob_spm_mspm) == pw_num
        return pw_lst, prob_spm_mspm

    def encode_pathnum(self, i): # 1, ith+1  or  i+2, ith+1
        """
            note: tot needs to be gi_lst[i] for the decode unambiguity (when incorrect mpw comes)
        :param i: i-th path for encode (0<i<num_pws)
        :param tot_ind: every index of encodings has corresponding tot for decoding purpose
        :return: path choice seed
        """
        tot = self.gi_lst[-1]
        l_pt = self.gi_lst[i]
        r_pt = self.gi_lst[i+1] - 1
        return self.sspm.convert2seed(random.randint(l_pt, r_pt), tot), (r_pt-l_pt+1)/tot

    def decode_pathnum(self, seed):
        """

        :param seed:
        :return: i-th path to decode (0<i<num_pws)
        """
        tot = self.gi_lst[-1]
        decode = seed % tot
        cum = np.array(self.gi_lst) - decode
        cum = (cum * np.roll(cum, -1))[:-1]  # roll: <--
        return (cum.argmin() + 1)

    def encode_vault(self, vault, mpw):
        """

        :param vault: list of plaintext
        :param mpw: master password
        :return: ciphertext list
        """
        concat = []
        for pw in vault:
            concat = self.encode_pw(concat, pw)
        return concat, len(concat)

    def decode_vault(self, vault, mpw, len_conca):
        """

        :param vault: ciphertext sequence (not a list maybe!)
        :param mpw: master password
        :return: plaintext list
        """
        pws = self.decode_pw(vault)
        return pws

dte_global = Markov_Encoder()

def generate_decoyvault(realvault, realseeds, randomseeds, basepw, seed2editdist, leakpw=None):
    """
    params:
        realvault: list of N pws
        realseeds: list of real seeds, each seed has SEED_LEN encoded numbers
        randomseeds: list of (N+1) random seeds
        seed2editdist: list of N numbers, indicating the edit distance between basepw and each pw in realvault
        leakpw: leaked password
    """
    ## get decoy seeds
    decoyseeds = copy.deepcopy(realseeds)
    keep_seed_position = [] # the position of the seed that is kept in the realseeds, position is 0-based and other seeds are randomized
    if leakpw is not None:
        if basepw == leakpw:
            keep_seed_position.append(0)
            ed1_4pws = [realvault[idx_] for idx_, ed_ in enumerate(seed2editdist) if ed_ != 5 and ed_ != 0]
            if len(ed1_4pws) > 0:
                randompw = random.choice(ed1_4pws)
                keep_seed_position.extend([(idx_+1) for idx_, pw_ in enumerate(realvault) if pw_==randompw])
        elif seed2editdist[realvault.index(leakpw)] == 5: # find all positions of pws in realvault same as leakpw
            keep_seed_position.extend([(i+1) for i, pw_ in enumerate(realvault) if pw_ == leakpw])
        else: # basepw should be kept as well as keeping seed with edit distance 1-4 with prob 1/(the number of passwords with edit distance 1-4)
            keep_seed_position.append(0)
            num_ed1_4 = sum([1 for ed_ in seed2editdist if ed_ != 5 and ed_ != 0])
            if random.random() < sum([1 for pw_ in realvault if pw_==leakpw]) / num_ed1_4:
                keep_seed_position.extend([(idx_ + 1) for idx_, pw_ in enumerate(realvault) if pw_ == leakpw])
    if 0 not in keep_seed_position:
        decoyseeds[:SEED_LEN] = randomseeds[:SEED_LEN]
    for n in range(1, len(realvault)+1):
        if n in keep_seed_position:
            continue
        decoyseeds[n*SEED_LEN: (n+1)*SEED_LEN] = randomseeds[n*SEED_LEN: (n+1)*SEED_LEN]

    ## get probs_spm_mspm and decoyvaults from decoyseeds
    # probs_spm_mspm: each pw in vault has prob [from spm model, from mspm model], len(probs_spm_mspm)==len(realvault)
    # decoyvaults: each element is a decoy vault with N pws
    decoyvaults, probs_spm_mspm = dte_global.decode_pw(decoyseeds)
    return decoyvaults, probs_spm_mspm


def generate_batchdecoys(realvault, leakpw, decoynum):
    # generate random seeds: length "(|realvault|+1) * SEED_LEN * decoynum"
    seed_len_eachvault = (len(realvault)+1) * SEED_LEN
    randomseeds = cp.random.randint(low=0, high=SEED_MAX_RANGE, size=(seed_len_eachvault * decoynum)).get()

    # encode vault to get realseeds, and seed2editdist (indicate edit distance between basepw and pw being encoded)
    basepw = max(realvault, key=realvault.count)
    realseeds, seed2editdist, _ = dte_global.encode_pw(realvault)

    decoyvaults, probs_spm_mspm = [], []
    # parallel to get decoy vaults
    pool = Pool(8)
    workers = []
    for nth_decoy in range(decoynum):
        workers.append(pool.apply_async(generate_decoyvault, (realvault, realseeds, randomseeds[nth_decoy*seed_len_eachvault: (nth_decoy+1)*seed_len_eachvault], basepw, seed2editdist, leakpw)))
    pool.close()
    pool.join()
    for worker in workers:
        decoyvaults_, probs_spm_mspm_ = worker.get()
        decoyvaults.append(decoyvaults_)
        probs_spm_mspm.extend(probs_spm_mspm_)

    # iterate to get decoy vaults
    '''for nth_decoy in range(decoynum):
        decoyvaults_, probs_spm_mspm_ = generate_decoyvault(realvault, realseeds, randomseeds[nth_decoy * seed_len_eachvault: (nth_decoy + 1) * seed_len_eachvault], basepw, seed2editdist, leakpw)
        decoyvaults.append(decoyvaults_)
        probs_spm_mspm.extend(probs_spm_mspm_)'''
    return decoyvaults, probs_spm_mspm

def main():
    vault = {}
    flst = os.listdir('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2_bc50')
    for fname in flst:
        f = open(os.path.join('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2_bc50', fname))
        vault.update(json.load(f))
    #dte = Markov_Encoder()
    for ith, vault_id in enumerate(vault):
        for pw in vault[vault_id]:
            if lencheck(pw):
                print('dropping => pw of length', len(pw))
                vault[vault_id].remove(pw)
        if len(vault[vault_id]) < 100:
            continue
        print('ith vault', ith)
        conca = dte_global.encode_pw(vault[vault_id])[0]
        assert len(conca) == (len(vault[vault_id]) + 1) * SEED_LEN
        #for i in range(len(vault[vault_id])):
         #   print('encoding pw:', vault[vault_id][i], '-> seed:', conca[(i+1) * SEED_LEN: (i+2) * SEED_LEN])
        vault_decrypted = dte_global.decode_pw(conca)[0]
        #print('decoding pw:', pw)
        #print('decrypted:', vault_decrypted)
        print('real', vault[vault_id])
        assert vault_decrypted == vault[vault_id]

        leakpw = random.choice(vault[vault_id])
        decoynum = 4000
        ts = time()
        decoyvaults = generate_batchdecoys(vault[vault_id], leakpw, decoynum=decoynum)[0]
        print('generate',  decoynum, 'decoy vaults of size', len(vault[vault_id]), 'using:', time()-ts, 's')
        '''print('basepw', max(vault[vault_id], key=vault[vault_id].count), '; leakpw:', leakpw)
        for decoyith in range(len(decoyvaults)):
            print('decoy', decoyith, decoyvaults[decoyith])'''

if __name__ == '__main__':
    main()