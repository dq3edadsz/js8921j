from MSPM.unreused_prob import unreuse_p
from MSPM.SPM.configs.configure import *
from MSPM.SPM.ngram_creator import NGramCreator
from MSPM.SSPM.pcfg import RulePCFGCreator
from MSPM.mspm_config import *
import json
import numpy as np
from collections import OrderedDict
from MSPM.utils import random
from Vault.utils import set_crypto
import struct

class Incremental_Encoder:
    """
    1. each encoding sequence has to be chosen randomly (according to sequence probability)
    2. seed sequence has to be padded to certain length (a hyperparameter #len_padded)
    3. modulo of spm and sspm can be different (seed itself is random, unless modulo is discovered)
    4.
    """

    def __init__(self):
        # initialization of spm
        CONFIG = Configure({'name': '4-gram'})
        length, progress_bar = 6, True
        self.spm = NGramCreator({
            "name": (
                "NGramCreator, Session: {}, Length: {}, Progress bar: {}".format(CONFIG.NAME, length,
                                                                                 progress_bar)),
            "alphabet": CONFIG.ALPHABET,
            "ngram_size": CONFIG.NGRAM_SIZE,
            "training_file": CONFIG.TRAINING_FILE,
            "length": length, # minimum length of password trained for spm
            "laplace_smooth": CONFIG.LAPLACE_SMOOTH,
            "progress_bar": progress_bar
        })
        self.spm.load("ip_list")
        self.spm.load("cp_list")
        self.spm.load("ep_list")
        print("single password model loading done ...")

        # initialization of sspm
        self.sspm = RulePCFGCreator()
        self.sspm.load()
        print("single similar password model loading done ...")

        # initialization of gi
        gi_lst = [unreuse_p(i) for i in range(MAX_PW_NUM)]
        gi_lst[0] = 1.
        self.gi_lst = [round((np.array(gi_lst[:i+1]).sum()-1) * SCALAR) \
                       for i in range(len(gi_lst))] # cumulative prob used for seed sampling

    def encode_pw(self, concatenation, pw):
        """

        :param concatenation: of pws in seeds
        :param pw: i+1 th password to be encoded
        :return: concatenation of seeds over all pws
        """
        pw_lst = self.decode_pw(concatenation)
        ith = len(pw_lst) # starting from 1 for f(ith)
        seed, prob = self.spm.encode_pw(pw)
        seed.insert(0, self.encode_pathnum(1, ith+1))
        if ith > 0:
            prob_lst = [np.log(np.array(prob)).sum() + np.log(unreuse_p(ith))]
            seeds_lst = [seed]
            for i in range(ith):
                seed, prob = self.sspm.encode_pw(pw_lst[i], pw)
                seed.insert(0, self.encode_pathnum(i+2, ith+1))
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
            pw_num = len(concatenation) / MAX_PW_LENGTH
            pw_lst = []
            for i in range(int(pw_num)):
                seed = concatenation[i*MAX_PW_LENGTH : (i+1)*MAX_PW_LENGTH]
                pathnum = self.decode_pathnum(seed.pop(0), len(pw_lst)+1)
                if pathnum == 1: # decode with spm
                    pw_lst.append(self.spm.decode_pw(seed))
                else: # decode with sspm
                    pw_lst.append(self.sspm.decode_pw(seed, pw_lst[pathnum-2]))
            return pw_lst
        else: # decode with complexity of log(n) 'up bottom'
            assert i is not None
            seed_lst = [seed]
            while True:
                seed_tmp = seed_lst[-1].pop(0)
                pathnum = self.decode_pathnum(seed_tmp, i+1)
                if pathnum == 1:  # decode with spm
                    pw = self.spm.decode_pw(seed_lst.pop(-1))
                    break
                else:  # decode with sspm
                    i = pathnum - 2
                    seed_lst.append(concatenation[i*MAX_PW_LENGTH : (i+1)*MAX_PW_LENGTH])

            while len(seed_lst) != 0:
                pw = self.sspm.decode_pw(seed_lst.pop(-1), pw)
            return pw

    def encode_pathnum(self, i, tot_ind):
        """
            note: tot needs to be gi_lst[i] for the decode unambiguity (when incorrect mpw comes)
        :param i: i-th path for encode (0<i<num_pws)
        :param tot_ind: every index of encodings has corresponding tot for decoding purpose
        :return: path choice seed
        """
        tot = self.gi_lst[tot_ind]
        l_pt = self.gi_lst[i-1]
        r_pt = self.gi_lst[i] - 1
        return self.sspm.convert2seed(random.randint(l_pt, r_pt), tot)

    def decode_pathnum(self, seed, tot_ind):
        """

        :param seed:
        :return: i-th path to decode (0<i<num_pws)
        """
        tot = self.gi_lst[tot_ind]
        decode = seed % tot
        cum = np.array(self.gi_lst) - decode
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
        aes = set_crypto(mpw)
        return aes.encrypt(struct.pack('{}I'.format(len(concat)), *concat)), len(concat)

    def decrypt_decode(self, vault, mpw, len_conca):
        """

        :param vault: ciphertext sequence (not a list maybe!)
        :param mpw: master password
        :return: plaintext list
        """
        aes = set_crypto(mpw)
        seed = aes.decrypt(vault)
        seed = struct.unpack('{}I'.format(len_conca), seed)
        #print(len(seed), seed)
        assert len(seed) % MAX_PW_LENGTH == 0
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
        for i in range(len(vault[vault_id])):
            conca = incre_ecoder.encode_pw(conca, vault[vault_id][i])
            print('encoding pw:', vault[vault_id][i], '-> seed:', conca[-MAX_PW_LENGTH:])
            pw = incre_ecoder.decode_pw(conca, conca[-MAX_PW_LENGTH:], i)
            print('decoding pw:', pw)
            assert pw == vault[vault_id][i]

if __name__ == '__main__':
    main()