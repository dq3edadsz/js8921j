from linecache import getline
from random import seed, randint, random, choice
from copy import deepcopy
from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF1
from Crypto.Util import Counter
from publicsuffix import PublicSuffixList
from urllib.parse import urlparse
import logging
import os
import struct
from MSPM.incre_pw_coding import Incremental_Encoder
import numpy as np
import pickle as cPickle
from Vault.utils import set_crypto
from opts import opts
args = opts().parse()

from MSPM.mspm_config import *
if args.victim == 'Golla':
    from Golla.markov_coding import Markov_Encoder
PATH = SOURCE_PATH


def get_exact_domain(url):
    psl = PublicSuffixList()
    url = url.strip()
    u = urlparse(url)
    h = u.hostname
    if not h:
        h = url
    return psl.get_public_suffix(h)


def get_hash(s):
    assert isinstance(s, str)
    h = sha256()
    h.update(s.encode())
    return int(h.hexdigest()[:32], 16)

def PRG(mpw, pin, nums):
    sd = hash(mpw + str(pin))
    seed(sd)
    return [randint(0, nums-1) for x in range(nums)]


def list_shuffle(raw_list, rolls, recover=False):
    assert len(rolls) >= len(raw_list)

    if not recover:
        for i in range(len(raw_list) - 1, -1, -1):
            raw_list[i], raw_list[rolls[i]] = raw_list[rolls[i]], raw_list[i]
    else:
        for i in range(len(raw_list)):
            raw_list[i], raw_list[rolls[i]] = raw_list[rolls[i]], raw_list[i]


class Vault:
    def __init__(self, mpw, pin, dic_size=12000, T=20, dte=None, log_exp=True):
        self.dte = None
        self.args = args
        self.train = args.nottrain
        self.mpw = mpw
        self.pin = pin
        self.T = T
        self.true_vault_no = get_hash(mpw) % T
        self.dic_size = dic_size
        self.pw_seed_list = []
        self.dm_list = []
        self.record = []
        self.concat = []
        self.index = {}
        self.log_exp = log_exp # logical expansion enable
        if dte:
            self.dte = dte
        else:
            self._init_dte()

# ================================================================================================================

    def _init_dte(self):
        if args.victim == 'MSPM':
            if args.model_eval == 'spm':
                self.dte = Incremental_Encoder(self.train).spm
            elif args.model_eval == 'sspm':
                self.dte = Incremental_Encoder(self.train).sspm
            elif args.model_eval == 'mspm':
                self.dte = Incremental_Encoder(self.train)
        elif args.victim == 'Golla':
            self.dte = Markov_Encoder()

    # 随机生成密码
    # sample_prob指采样概率，即从字典中选择密码的概率
    def random_pw(self):
        pw = str()
        while len(pw) < 6:
            pw = getline(PATH + "/data/password_dict.txt", randint(1, self.dic_size)).strip()
        return pw

    def exist_dm(self, dm):
        return get_hash(dm) in self.dm_list

    def _init_index(self):
        if self.log_exp:
            rseq = PRG(self.mpw, self.pin, len(self.dm_list))
            shuffle_dm = deepcopy(self.dm_list)
            list_shuffle(shuffle_dm, rseq)
        else:
            shuffle_dm = self.dm_list
        for i, dm in enumerate(shuffle_dm):
            self.index[dm] = i

# ================================================================================================================
    # 加密pw_list，得到密文数组
    def encrypt_block(self, mpw_list):
        assert len(mpw_list) == len(self.pw_seed_list)
        cipher_list = []
        for i, pw_seed in enumerate(self.pw_seed_list):
            mpw = mpw_list[i]
            aes = set_crypto(mpw)
            #pw_seed = list(np.array(pw_seed) / self.dte.spm.ls_compensa_scale)
            cipher_list.append(aes.encrypt(struct.pack('{}L'.format(len(pw_seed)), *pw_seed)))
        return cipher_list

    # 解密数据库密文，还原pw_list
    def decrypt_block(self, cipher):
        cipher_list = []
        for x in range(0, len(cipher), MAX_PW_LENGTH * 4):
            aes = set_crypto(self.mpw)
            pw_cipher = cipher[x: x + MAX_PW_LENGTH*4]
            seed = aes.decrypt(pw_cipher)
            seed = struct.unpack('{}I'.format(MAX_PW_LENGTH), seed)
            #seed = list(np.array(seed) * self.dte.spm.ls_compensa_scale)
            pw = self.dte.decode_pw(seed)
            cipher_list.append(pw)

        return cipher_list

# ================================================================================================================

    def get_pw(self, dm):
        dm = get_exact_domain(dm)
        dm_h = get_hash(dm)
        if self.index.get(dm_h) is not None:
            i = self.index[dm_h]
            return self.dte.decode_pw(self.pw_seed_list[i])

        logging.debug("mpw {}, pin {}, No password for {}".format(self.mpw, self.pin, dm_h))
        return None

    def add_pw(self, pw, dm):
        dm = get_exact_domain(dm)
        dm_h = get_hash(dm)
        if dm_h in self.dm_list:
            index = self.index[dm_h]
        else:
            logging.debug("new domain")
            index = np.arange(VAULT_SIZE)[np.array(self.record) == -1]
            index = np.random.choice(index)
            key = [k for k, v in self.index.items() if v == index][0]
            del self.index[key]
            self.index[dm_h] = index
        self.record[index] = max(self.record) + 1
        self.save_single_pw(pw, index)  # save pw into file
        self.pw_seed_list[index] = self.concat[-MAX_PW_LENGTH:]
        logging.debug("add password done!")

# ================================================================================================================

    def init_vault(self):
        self.pw_seed_list = [self.dte.spm.encode_pw(self.random_pw(), probfree=True)[0] \
                             for _ in range(VAULT_SIZE)]
        self.record = [-1 for _ in range(VAULT_SIZE)]  # -1 for dummy
        self.save_vault()
        self.load_dm()
        self._init_index()

    def load(self):
        self.load_vault()
        self.load_dm()
        self._init_index()

    def save(self):
        self.save_dm()
        self.save_vault()

    def load_dm(self):
        self.dm_list = []
        with open(PATH + '/data/domain_hash.txt', 'r') as rf:
            for line in rf:
                if len(self.dm_list) == VAULT_SIZE:
                    break
                self.dm_list.append(int(line.strip(), 16))
        # self._init_index()

    def save_dm(self):
        with open(PATH + '/data/domain_hash.txt', 'w') as wf:
            for dm in self.dm_list:
                wf.write(str(hex(dm))[2:]+'\n')

    def load_vault(self):
        self.pw_list = []
        with open(PATH + "/data/vault_data/vault_{}".format(self.true_vault_no), 'rb') as rf:
            cipher = rf.read()
            if (len(cipher) // 4) % MAX_PW_LENGTH:
                raise Exception("Vault cipher incorrect!")

            self.pw_list = self.decrypt_block(cipher)

    def save_vault(self):
        for x in range(self.T):
            # print('save vault_{}'.format(x))
            if x == self.true_vault_no:
                mpw_list = [self.mpw for _ in self.record]
            else:
                mpw_list = []
                while True:
                    mpw = self.random_pw()
                    if get_hash(mpw) % self.T == x:
                        mpw_list.append(mpw)
                    if len(mpw_list) == len(self.record):
                        # print(mpw_list)
                        break
            cipher_list = self.encrypt_block(mpw_list) # encrypt with spm only (all dummy)

            dic = {'cipher_list': cipher_list}
            dic['record'] = self.record # same for every copy
            cPickle.dump(dic, open(PATH + "/data/vault_data/vault_{}".format(x), "wb"))

    def save_single_pw(self, pw, index):

        dic = cPickle.load(open(PATH + "/data/vault_data/vault_{}".format(self.true_vault_no), "rb"))
        cipher_list_true = dic['cipher_list']
        concat = self.record2concat(dic['record'], cipher_list_true)
        self.concat = concat
        for x in range(self.T):
            concat_tmp = self.dte.encode_pw(concat, pw)
            if x == self.true_vault_no:
                mpw = self.mpw
            else:
                while True:
                    mpw = self.random_pw()
                    if get_hash(mpw) % self.T == x:
                        break
            dic = cPickle.load(open(PATH + "/data/vault_data/vault_{}".format(x), "rb"))
            seed = concat_tmp[-MAX_PW_LENGTH:]
            aes = set_crypto(mpw)
            #seed = list(np.array(seed) / self.dte.spm.ls_compensa_scale)
            pw_cipher = aes.encrypt(struct.pack('{}L'.format(MAX_PW_LENGTH), *seed))
            dic['cipher_list'][index] = pw_cipher
            dic['record'][index] = self.record  # same for every copy
            cPickle.dump(dic, open(PATH + "/data/vault_data/vault_{}".format(x), "wb"))

    def record2concat(self, record, cipher_list):
        """

        :param record: shuffled list indicator -1 for dummy, non-negatives for incremental
        :param cipher_list: same order with record
        :return: concatentation of seeds
        """
        record = np.array(record)
        record_valid = record[record > -1]
        cipher_valid_ = [cipher_list[i] for i in np.arange(VAULT_SIZE)[record>-1]]
        cipher_valid = [cipher_valid_[i] for i in np.argsort(record_valid)]
        concat = []
        for pw_cipher in cipher_valid:
            aes = set_crypto(self.mpw)
            pw_plain = aes.decrypt(pw_cipher)
            seed = struct.unpack('{}I'.format(MAX_PW_LENGTH), pw_plain)
            #seed = list(np.array(seed) * self.dte.spm.ls_compensa_scale)
            concat.append(seed)

        return concat


if __name__ == '__main__':
    vault_t = Vault('beeno', '123456')
    vault_t.init_vault()
    vault_t.add_pw('beeno070', 'apple.com')
    print(vault_t.get_pw('apple.com'))