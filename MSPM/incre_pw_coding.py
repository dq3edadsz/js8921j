from MSPM.unreused_prob import unreuse_p
from MSPM.SSPM.pcfg import RulePCFGCreator
from MSPM.utils import random, gen_gilst
from Vault.utils import set_crypto
import struct
import logging
import numpy as np
from MSPM.mspm_config import *
import os
import json
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
#from Golla.configs.configure import *
#from Golla.ngram_creator import NGramCreator
from numba import cuda
from MSPM.SPM.configs.configure import Configure
from MSPM.SPM.ngram_creator import NGramCreator
#from numba import jit
from opts import opts
args = opts().parse()

class Incremental_Encoder:
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
        # initialization of gi
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

    def encode_pw(self, concatenation, pw):
        """
            note the password length better falls in the range of [MIN_PW_LENGTH, MAX_PW_LENGTH]
        :param concatenation: of pws in seeds
        :param pw: i+1 th password to be encoded
        :return: concatenation of seeds over all pws
        """
        pw_lst = self.decode_pw(concatenation)
        ith = len(pw_lst) # starting from 1 for f(ith)
        seed, prob = self.spm.encode_pw(pw)
        seed.insert(0, self.encode_pathnum(1, ith))
        if ith > 0:
            prob_lst = [np.log(np.array(prob)).sum() + np.log(unreuse_p(ith))]
            seeds_lst = [seed]
            for i in range(ith):
                seed, prob = self.sspm.encode_pw(pw_lst[i], pw, ith)
                seed.insert(0, self.encode_pathnum(i+2, ith))
                prob = np.log(np.array(prob)).sum() + np.log((1-unreuse_p(ith)) / ith) \
                    if len(prob) != 0 else -np.inf
                prob_lst.append(prob)
                seeds_lst.append(seed)
            prob_lst = np.exp(np.array(prob_lst)) / np.exp(np.array(prob_lst)).sum()
            ind = [i for i in range(len(prob_lst))]
            id_chosen = np.random.choice(ind, p=prob_lst)
            print('using', str(id_chosen+1) + '/' + str(len(prob_lst)))
            seed = seeds_lst[id_chosen]
        else:
            print('using 1/1')
        concatenation.extend(seed)
        return concatenation

    def encode_pw_frompriorpws(self, pw_lst, pw, pw_lst_spmprob=None):
        """
            note the password length better falls in the range of [MIN_PW_LENGTH, MAX_PW_LENGTH]
        :param concatenation: of pws in seeds
        :param pw: i+1 th password to be encoded
        :return: concatenation of seeds over all pws
        """
        #pw_lst = self.decode_pw(concatenation)
        ith = len(pw_lst) # starting from 1 for f(ith)
        seed, prob = self.spm.encode_pw(pw)
        seed.insert(0, self.encode_pathnum(1, ith))
        independent_flag = 1 # 1 for independently, 0 for dependently encoded seed

        prob_seed_lst = [np.log(np.array(prob)).sum() + (np.log(unreuse_p(ith)) if ith > 0 else 0)]
        seeds_lst = [seed]
        id_chosen = 0
        if ith > 0:
            for i in range(ith):
                seed, prob = self.sspm.encode_pw(pw_lst[i], pw, ith)
                seed.insert(0, self.encode_pathnum(i+2, ith))
                prob = np.log(np.array(prob)).sum() + np.log((1-unreuse_p(ith)) / ith) if len(prob) != 0 else -np.inf
                if prob != -np.inf:
                    ind_pr = np.log(np.array(self.spm.encode_pw(pw_lst[i])[1])).sum() if pw_lst_spmprob is None else pw_lst_spmprob[i]
                    prob += ind_pr
                prob_seed_lst.append(prob)
                seeds_lst.append(seed)
            prob_lst = np.exp(np.array(prob_seed_lst)) / np.exp(np.array(prob_seed_lst)).sum()
            ind = [i for i in range(len(prob_lst))]
            id_chosen = np.random.choice(ind, p=prob_lst)
            #print('using', str(id_chosen+1) + '/' + str(len(prob_lst)))
            if id_chosen != 0:
                independent_flag = 0
            seed = seeds_lst[id_chosen]
            independent_prob = prob_lst[0]
        else:
            independent_prob = 1
            #print('using 1/1')
        return seed, independent_flag, independent_prob, prob_seed_lst[id_chosen]

    def encode_pw_testunit(self, concatenation, pw, tset):
        """
            note the password length better falls in the range of [MIN_PW_LENGTH, MAX_PW_LENGTH]
        :param concatenation: of pws in seeds
        :param pw: i+1 th password to be encoded
        :return: concatenation of seeds over all pws
        """
        pw_lst = tset#self.decode_pw(concatenation)
        assert pw_lst == tset[:len(pw_lst)]
        ith = len(pw_lst) # starting from 1 for f(ith)
        seed, prob = self.spm.encode_pw(pw)
        seed.append(self.encode_pathnum(1, ith))
        if ith > 0:
            prob_lst = [np.log(np.array(prob)).sum() + np.log(unreuse_p(ith))]
            seeds_lst = [seed]
            for i in range(ith):
                seed, prob = self.sspm.encode_pwtest(pw_lst[i], pw, ith)
                if seed.size != 0:
                    seed[-1] = self.encode_pathnum(i+2, ith)
                prob = np.log(np.array(prob)).sum() + np.log((1-unreuse_p(ith)) / ith) \
                    if len(prob) != 0 else -np.inf
                prob_lst.append(prob)
                seeds_lst.append(seed)
            prob_lst = np.exp(np.array(prob_lst)) / np.exp(np.array(prob_lst)).sum()
            ind = [i for i in range(len(prob_lst))]
            id_chosen = np.random.choice(ind, p=prob_lst)
            print('using', str(id_chosen+1) + '/' + str(len(prob_lst)))
            seed = seeds_lst[id_chosen]
        else:
            print('using 1/1')
        concatenation.extend(seed)
        return concatenation

    def decode_pw(self, concatenation, seed=None, i=None):
        """

        :param concatenation: of seeds
        :param seed: the specific seed needed to be decoded
        :return:
        """
        if seed is None: # decode following 'bottom up'
            pw_num = len(concatenation) / SEED_LEN
            pw_lst = []
            for i in range(int(pw_num)):
                seed = concatenation[i*SEED_LEN : (i+1)*SEED_LEN]
                pathnum = self.decode_pathnum(seed.pop(0), len(pw_lst))
                if pathnum == 1: # decode with spm
                    pw_lst.append(self.spm.decode_pw(seed))
                else: # decode with sspm
                    pw_lst.append(self.sspm.decode_pw(seed, pw_lst[pathnum-2], i))
            return pw_lst
        else: # decode with complexity of log(n) 'up bottom'
            assert i is not None
            i_ = []
            seed_lst = [seed]
            while True:
                seed_tmp = seed_lst[-1].pop(0) # remove path seed: MAX_PW_LENGTH => MAX_PW_LENGTH-1
                pathnum = self.decode_pathnum(seed_tmp, i)
                i_.append(i)
                if pathnum == 1:  # decode with spm
                    pw = self.spm.decode_pw(seed_lst.pop(-1))
                    i_.pop(-1)
                    break
                else:  # decode with sspm
                    i = pathnum - 2
                    seed_lst.append(concatenation[i*SEED_LEN : (i+1)*SEED_LEN])

            while len(seed_lst) != 0:
                assert len(i_) == len(seed_lst)
                pw = self.sspm.decode_pw(seed_lst.pop(-1), pw, i_.pop(-1))
            return pw

    def encode_pathnum(self, i, tot_ind): # 1, ith+1  or  i+2, ith+1
        """
            note: tot needs to be gi_lst[i] for the decode unambiguity (when incorrect mpw comes)
        :param i: i-th path for encode (0<i<num_pws)
        :param tot_ind: every index of encodings has corresponding tot for decoding purpose
        :return: path choice seed
        """
        tot = self.gi_lst[tot_ind][-1]
        l_pt = self.gi_lst[tot_ind][i-1]
        r_pt = self.gi_lst[tot_ind][i] - 1
        return self.sspm.convert2seed(random.randint(l_pt, r_pt), tot)

    def decode_pathnum(self, seed, tot_ind):
        """

        :param seed:
        :return: i-th path to decode (0<i<num_pws)
        """
        tot = self.gi_lst[tot_ind][-1]
        decode = seed % tot
        cum = np.array(self.gi_lst[tot_ind]) - decode
        cum = (cum * np.roll(cum, -1))[:-1]  # roll: <--
        return (cum.argmin() + 1)

    def encode_encrypt(self, vault, mpw):
        """

        :param vault: list of plaintext
        :param mpw: master password
        :return: ciphertext list
        """
        concat = []
        for pw in vault:
            concat = self.encode_pw(concat, pw)
        #concat = list(np.array(concat) / self.spm.ls_compensa_scale)
        aes = set_crypto(mpw)
        return aes.encrypt(struct.pack('{}L'.format(len(concat)), *concat)), len(concat)

    def decrypt_decode(self, vault, mpw, len_conca):
        """

        :param vault: ciphertext sequence (not a list maybe!)
        :param mpw: master password
        :return: plaintext list
        """
        aes = set_crypto(mpw)
        seed = aes.decrypt(vault)
        seed = struct.unpack('{}L'.format(len_conca), seed)
        #seed = list(np.array(seed) * self.spm.ls_compensa_scale)
        #print(len(seed), seed)
        assert len(seed) % SEED_LEN == 0
        pws = self.decode_pw(list(seed))
        return pws


def main():
    vault = {}
    flst = os.listdir(SOURCE_PATH + '/data/pastebin/fold5')
    for fname in flst:
        f = open(os.path.join(SOURCE_PATH + '/data/pastebin/fold5', fname))
        vault.update(json.load(f))
    incre_ecoder = Incremental_Encoder()
    for vault_id in vault:
        print(vault[vault_id])
        conca = []
        for pw in vault[vault_id]:
            if lencheck(pw):
                print('dropping => pw of length', len(pw))
                vault[vault_id].remove(pw)
        for i in range(len(vault[vault_id])):
            conca = incre_ecoder.encode_pw(conca, vault[vault_id][i])
            print('encoding pw:', vault[vault_id][i], '-> seed:', conca[-SEED_LEN:])
            pw = incre_ecoder.decode_pw(conca, conca[-SEED_LEN:], i)
            print('decoding pw:', pw)
            assert pw == vault[vault_id][i]

if __name__ == '__main__':
    main()