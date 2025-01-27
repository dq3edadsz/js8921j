import numpy as np
import pylcs
import os
from MSPM.mspm_config import *
import json
from linecache import getline
import random
from MSPM.incre_pw_coding import Incremental_Encoder

class Weight:
    """
    most are the derivations from eq  p_real(V_i)/p_decoy(V_i)
    """
    def __init__(self):
        self.rockyou = {} # maybe k-v => word-count
        self.pastebin = {}
        self.n = None # total number of passwords in rockyou
        self.f = Feature()
        self.features = [[self.f.lcsstr, self.f.lcs],
                         [self.f.lcs, self.f.lcsstr]] # self.features are lists of M and I (i.e., each list with two functions)
        self._init_dataset()

    def _init_dataset(self):
        """
            step1:
                load dataset for statistics
            step2:
                pre-compute proportion of every feature within pastebin
        :return:
        """
        # step1
        flst = os.listdir(PASTB_PATH)
        for fname in flst:
            if str(TEST_FOLD) in fname:
                print('loading test file:', fname)
                with open(os.path.join(PASTB_PATH, fname)) as f:
                    self.pastebin = json.load(f)

        with open(ROCKY_PATH, encoding='latin1') as f:
            for i in range(DICT_SIZE):
                contents = f.readline().strip().split() # in the format of 'count password'
                if len(contents) != 2:
                    continue
                self.rockyou[contents[1]] = int(contents[0])
                #print(contents.strip().split())
                self.n = np.array(list(self.rockyou.values())).sum()

        # step2
        self.p_real = {} # k - v (feature - true proportion)
        for fpair in self.features:
            self.p_real[self.dicname(fpair)] = self.pfeature4vault(self.pastebin, fpair)

    def dicname(self, fpair):
        # set name for the proportion dictionary key
        assert len(fpair) == 2
        return fpair[0].__name__ + '/' + fpair[1].__name__

    def singlepass(self, vault, spm, a_s=1, f_d=5):
        """
            calculate
                p_decoy(pw) through single password model
                p_real(pw) with statistic upon rockyou
        :param vault: list of passwords
        :param spm: dte single password model
        :param a_s:
        :param f_d:
        :return: priority weight(vault)  a scalar
        """
        sp_probs = []
        for pw in vault:
            _, p_decoy = spm.encode_pw(pw)
            p_decoy = np.clip(np.array(p_decoy).prod(), a_min=1e-10, a_max=np.inf)
            if pw in self.rockyou:
                fq = self.rockyou[pw]
            else:
                fq = 0
            if fq <= f_d and self.freq(fq, a_s)/p_decoy > 1:
                sp_probs.append(1.)
            else:
                sp_probs.append(self.freq(fq, a_s) / p_decoy)
        return np.array(sp_probs).prod()

    def freq(self, fq, smoo):
        return (fq + smoo) / (self.n + smoo)

    def passsimi(self, vi, dte): # not that certain!
        """
            calculate
                p_decoy(F=x)
                    step1: encoding vi and encrypt it with one mpw
                    step2: decrypt-then-decode vi with different mpws,
                        creating e.g., 100 decoy vaults, which gonna used for probability calculation
                also
                brewed with p_real(F=x) with statistic upon pastebin. NOTE: which has been pre-computed!
        :param vi: a password vault (a list)
        :param dte: vola!
        :return:
        """
        # step1
        mpw_en = getline(SOURCE_PATH + "/data/password_dict.txt", random.randint(1, DICT_SIZE)).strip()
        ciphers, l = dte.encode_encrypt(vi, mpw_en)

        # step2
        decoy_sets = {}
        for i in range(PS_DECOYSIZE):
            while True:
                mpw = getline(SOURCE_PATH + "/data/password_dict.txt", random.randint(1, DICT_SIZE)).strip()
                if mpw != mpw_en:
                    break
            decoy_sets[str(i)] = dte.decrypt_decode(ciphers, mpw, l)
            if i < 3:
                print(decoy_sets[str(i)])
        p_decoy = {}
        for fpair in self.features:
            p_decoy[self.dicname(fpair)] = self.pfeature4vault(decoy_sets, fpair)
        return np.array([p_decoy[feat]*self.p_real[feat] for feat in self.p_real]).prod()

    def pfeature4vault(self, vaults, feature):
        """
            calculate proportion of vaults with corresponding feature (actually considering password pair)
        :param vaults: a dict containing lists as items, for statistic
        :param feature: certain feature: a function list (two feature function)
        :return: proportion of vaults hold the feature
        """
        yep = 0
        flag = 0
        for vault in vaults:
            vault = vaults[vault]
            for i in range(len(vault) - 1):
                for j in range(i + 1, len(vault)):
                    if feature[0](vault[i], vault[j]) * (1 - feature[1](vault[i], vault[j])) == 1:
                        yep += 1
                        flag = 1
                        break
                if flag == 1:
                    flag = 0
                    break
        return yep / len(vaults)


class Feature:
    def __init__(self):
        pass

    def lcsstr(self, pw1, pw2):
        return int(pylcs.lcs_string_length(pw1, pw2) >= max(len(pw1), len(pw2))/2)

    def gm(self, pw1, pw2): # this one is not certain!
        res = pylcs.lcs_string_idx(pw1, pw2)
        lcss = ''.join([pw2[i] for i in res if i != -1])  # longest common substring
        sub_len = len(lcss)
        start_pw1 = pw1.index(lcss)
        start_pw2 = pw2.index(lcss)
        HD, TD = start_pw1, len(pw1) - start_pw1 - sub_len
        HA, TA = start_pw2, len(pw2) - start_pw2 - sub_len
        return int((HA+HD) == 0 and (TA == 5 or TD == 5))

    def levenshtein(self, pw1, pw2):
        return int(pylcs.edit_distance(pw1, pw2) <= max(len(pw1), len(pw2)) / 2)

    def lcs(self, pw1, pw2):
        return int(pylcs.lcs_sequence_length(pw1, pw2) >= max(len(pw1), len(pw2)) / 1.5)

    def manhattan(self, pw1, pw2):
        vec1, vec2 = self.pw2vec(pw1, pw2)
        return int(sum([np.abs(a - b) for a, b in zip(vec1, vec2)]) <= len(pw1+pw2)/2.5)

    def overlap(self, pw1, pw2): # change a little, not certain!
        return int(len(set(pw1) - (set(pw1) - set(pw2))) >= len(set(pw1+pw2))/2)

    def pw2vec(self, pw1, pw2):
        x = pw1
        y = pw2
        set1 = set()
        for a in range(0, len(x)):
            set1.add(x[a])
        for a in range(0, len(y)):
            set1.add(y[a])
        vec1 = [None] * len(set1)
        vec2 = [None] * len(set1)
        for counter, each_char in enumerate(set1):
            vec1[counter] = x.count(each_char)
            vec2[counter] = y.count(each_char)
        return vec1, vec2

if __name__ == '__main__':
    weight = Weight()