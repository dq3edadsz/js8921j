''' The Markov model
:author: Maximilian Golla
:contact: maximilian.golla@rub.de
:version: 0.7.1, 2019-07-11
'''
from numba import cuda
from cuda_voila import gpu_decode
# External modules
from collections import OrderedDict # storing the alphabet
import os  # load and save / file handling
#import umsgpack  # load and save # pip install u-msgpack-python
import msgpack
import math  # only pow
import logging  # logging debug infos
from tqdm import tqdm  # progress bar while reading the file # pip install tqdm
from random import uniform, randint, shuffle
import datetime
from MSPM.mspm_config import *
import pickle as cPickle
import bz2
import numpy as np
from opts import opts
args = opts().parse()

def is_almost_equal(a, b, rel_tol=1e-09, abs_tol=0.0):
    # print '{0:.16f}'.format(a), '{0:.16f}'.format(b)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class NGramCreator:

    def __init__(self, dict):
        self.name = dict['name']
        self.train = args.nottrain
        self.tag = 'full' if self.train else 'half'
        logging.debug("Constructor started for '{}'".format(self.name))
        self.markovs = len(TRAIN_LENGTH) + 1 # training for each length specified (plus one for all other lengths)
        self.alphabet = ALPHABET
        self.alphabet_len = len(self.alphabet)
        self.alphabet_dict = OrderedDict.fromkeys(self.alphabet)  # a 0, b 1, c 2
        i = 0
        for char in self.alphabet_dict:
            self.alphabet_dict[char] = i
            i += 1
        self.alphabet_list = list(self.alphabet)
        logging.debug("Used alphabet: {}".format(self.alphabet))
        self.ngram_size = dict['ngram_size']
        self.seed_length_rec = self.ngram_size - 2 # pw_length - seed_length_rec = seed_length(without length seed)
        self.laplace_smooth = dict['laplace_smooth']
        self.ls_compensa_scale = dict['ls_compensation']
        logging.debug("NGram size: {}".format(self.ngram_size))
        self.training_file = dict['training_file']
        self.disable_progress = False if dict['progress_bar'] else True
        self.length_dic = {}
        self.ip_list = []
        self.cp_list = []
        self.ep_list = []
        self.no_ip_ngrams = int(math.pow(self.alphabet_len, (self.ngram_size - 1))) # each markov
        self.no_cp_ngrams = int(math.pow(self.alphabet_len, (self.ngram_size))) # each markov
        self.no_ep_ngrams = self.no_ip_ngrams # save one exponentiation :-P
        self._init_lengthdic()
        #self._init_alp_lst()
        logging.debug("training lengths:" + str(TRAIN_LENGTH))
        logging.debug("len(IP) theo: {}".format(self.no_ip_ngrams))
        logging.debug("len(CP) theo: {} => {} * {}".format(self.no_cp_ngrams,
                                                           int(math.pow(self.alphabet_len, (self.ngram_size - 1))),
                                                           self.alphabet_len))
        logging.debug("len(EP) theo: {}".format(self.no_ep_ngrams))

    def __del__(self):
        logging.debug("Destructor started for '{}'".format(self.name))

    def __str__(self):
        return "Hello {}!".format(self.name)

    ########################################################################################################################

    def _is_in_alphabet(self, string):
        for char in string:
            if not char in self.alphabet:
                return False
        return True

    # checks whether two floats are equal like 1.0 == 1.0?
    def _is_almost_equal(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        # print '{0:.16f}'.format(a), '{0:.16f}'.format(b)
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def _countlines(self):
        with open(self.training_file, "rb") as fb:
            f = cPickle.load(fb)
            for count, line in enumerate(f):
                pass
        self.training_file_lines = count + 1

    def _init_halfdic(self):
        print("spliting rockyou into two pieces!")
        if self.train: # train with whole rockyou
            pass
        else: # train with half rockyou, dumping another half for test sampling
            with open(self.training_file, encoding="latin-1") as f:
                lines = f.readlines()
            flatten = []
            for line in lines:
                L = line.lstrip().strip('\r\n').split(' ')
                if len(L) != 2:
                    continue
                cnt, word = int(L[0]), L[1]
                flatten.extend([word] * cnt)
            with open(SOURCE_PATH+"/data/rockyou-flat", "wb") as fp:
                cPickle.dump(flatten, fp, protocol=cPickle.HIGHEST_PROTOCOL)
            lines = flatten
            shuffle(lines)
            train = self.l2l(lines[:len(lines) // 2])
            test = self.l2l(lines[len(lines) // 2:])
            with open(SOURCE_PATH+"/data/rockyou-withcount-train", "wb") as fp:
                cPickle.dump(train, fp, protocol=cPickle.HIGHEST_PROTOCOL)
            with open(SOURCE_PATH+"/data/rockyou-withcount-test", "wb") as fp:
                cPickle.dump(test, fp, protocol=cPickle.HIGHEST_PROTOCOL)
            self.training_file_lines = len(train)
            self.training_file = ROCKY_PATH_TRAIN

    def l2l(self, lst):
        dic = {}
        for pw in lst:
            if pw in dic.keys():
                dic[pw] += 1
            else:
                dic[pw] = 1
        lst = []
        for pw in dic:
            lst.append(str(dic[pw]) + ' ' + pw + '\r\n')
        return lst

    def _init_lengthdic(self):
        dic = cPickle.load(open(SOURCE_PATH + "/MSPM/SPM/trained/" + args.spmdata + "_lengthstatistic", "rb"))
        self.length_dic = {MIN_PW_LENGTH - self.seed_length_rec - 1: 0}
        for length in range(MIN_PW_LENGTH, MAX_PW_LENGTH+1): # TODO: min_pw_len is not identical to ngram length when encoding
            self.length_dic[length - self.seed_length_rec] = \
                self.length_dic[length - self.seed_length_rec - 1] + dic[length]

    def _init_alp_lst(self):
        self.alplst_ip = [] # 4gram for now
        for g0 in self.alphabet_dict:
            for g1 in self.alphabet_dict:
                for g2 in self.alphabet_dict:
                    self.alplst_ip.append(g0+g1+g2)

        self.alplst_cp = []  # 4gram for now
        for g0 in self.alphabet_dict:
            for g1 in self.alphabet_dict:
                for g2 in self.alphabet_dict:
                    for g3 in self.alphabet_dict:
                        self.alplst_cp.append(g0 + g1 + g2 + g3)
    ########################################################################################################################

    # ngram-to-intial-prob-index
    def _n2iIP(self, ngram, markov_id):
        # markov_id => range from [0, self.markovs]
        ngram = list(ngram)
        if self.ngram_size == 5:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[3]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[2]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[1]]) + (
                                    self.alphabet_len ** 3 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_ip_ngrams
        if self.ngram_size == 4:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[2]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_ip_ngrams
        if self.ngram_size == 3:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[1]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_ip_ngrams
        if self.ngram_size == 2:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_ip_ngrams

    # intial-prob-index-to-ngram
    def _i2nIP(self, index):
        _, index = divmod(index, self.no_ip_ngrams)
        if self.ngram_size == 5:
            third, fourth = divmod(index, self.alphabet_len)
            second, third = divmod(third, self.alphabet_len)
            first, second = divmod(second, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second] + self.alphabet_list[third] + \
                   self.alphabet_list[fourth]
        if self.ngram_size == 4:
            second, third = divmod(index, self.alphabet_len)
            first, second = divmod(second, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second] + self.alphabet_list[third]
        if self.ngram_size == 3:
            first, second = divmod(index, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second]
        if self.ngram_size == 2:
            return self.alphabet_list[index]

    # ngram-to-conditial-prob-index
    def _n2iCP(self, ngram, markov_id):
        ngram = list(ngram)
        if self.ngram_size == 5:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[4]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[3]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[2]]) + (
                               self.alphabet_len ** 3 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 4 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_cp_ngrams
        if self.ngram_size == 4:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[3]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[2]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 3 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_cp_ngrams
        if self.ngram_size == 3:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[2]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_cp_ngrams
        if self.ngram_size == 2:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[1]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[0]]) + markov_id * self.no_cp_ngrams

    # conditial-prob-index-to-ngram
    def _i2nCP(self, index):
        _, index = divmod(index, self.no_cp_ngrams)
        if self.ngram_size == 5:
            fourth, fifth = divmod(index, self.alphabet_len)
            third, fourth = divmod(fourth, self.alphabet_len)
            second, third = divmod(third, self.alphabet_len)
            first, second = divmod(second, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second] + self.alphabet_list[third] + \
                   self.alphabet_list[fourth] + self.alphabet_list[fifth]
        if self.ngram_size == 4:
            third, fourth = divmod(index, self.alphabet_len)
            second, third = divmod(third, self.alphabet_len)
            first, second = divmod(second, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second] + self.alphabet_list[third] + \
                   self.alphabet_list[fourth]
        if self.ngram_size == 3:
            second, third = divmod(index, self.alphabet_len)
            first, second = divmod(second, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second] + self.alphabet_list[third]
        if self.ngram_size == 2:
            first, second = divmod(index, self.alphabet_len)
            return self.alphabet_list[first] + self.alphabet_list[second]

    ########################################################################################################################

    # Adds all possible combinations of ngrams to the list with initial count = 1
    def _init_lists(self, kind):
        if kind == "ip_list":
            for i in range(0, int(self.no_ip_ngrams * self.markovs)):
                self.ip_list.append(self.laplace_smooth)  # Smoothing, we initialize every possible ngram with count = 1
        elif kind == "cp_list":
            for i in range(0, int(self.no_cp_ngrams * self.markovs)):
                self.cp_list.append(self.laplace_smooth)  # Smoothing, we initialize every possible ngram with count = 1
        elif kind == "ep_list":
            for i in range(0, int(self.no_ep_ngrams * self.markovs)):
                self.ep_list.append(self.laplace_smooth)  # Smoothing, we initialize every possible ngram with count = 1
        else:
            raise Exception('Unknown list given (required: ip_list, cp_list, or ep_list)')

    def _count(self, kind):
        with open(self.training_file, 'rb') as input_file:
            input_file = cPickle.load(input_file)
            for line in tqdm(input_file, desc=self.training_file, total=self.training_file_lines,
                             disable=self.disable_progress, miniters=int(self.training_file_lines/1000), unit="pw"):
                #line = [it.decode("utf-8") for it in line]
                L = line.lstrip().strip('\r\n').split(' ')
                cnt, word = int(L[0]), L[1]
                # if len(word) != self.length:
                if len(word) < MIN_PW_LENGTH:
                    continue
                if len(word) in TRAIN_LENGTH:
                    markov_id = TRAIN_LENGTH.index(len(word))
                else:
                    markov_id = len(TRAIN_LENGTH)

                if kind == "ip_list":
                    if self._is_in_alphabet(word):  # Filter non-printable
                        ngram = word[0:self.ngram_size - 1]  # Get IP ngram
                        self.ip_list[self._n2iIP(ngram, markov_id)] = self.ip_list[
                                                    self._n2iIP(ngram, markov_id)] + cnt * self.ls_compensa_scale  # Increase IP ngram count by 1

                elif kind == "cp_list":
                    if self._is_in_alphabet(word):  # Filter non-printable
                        old_pos = 0
                        for new_pos in range(self.ngram_size, len(word) + 1,
                                             1):  # Sliding window: pas|ass|ssw|swo|wor|ord
                            ngram = word[old_pos:new_pos]
                            old_pos += 1
                            self.cp_list[self._n2iCP(ngram, markov_id)] = self.cp_list[self._n2iCP(
                                ngram, markov_id)] + cnt * self.ls_compensa_scale  # Increase CP ngram count by 1

                elif kind == "ep_list":
                    if self._is_in_alphabet(word):  # Filter non-printable
                        ngram = word[-self.ngram_size + 1:]  # Get EP ngram
                        self.ep_list[self._n2iIP(ngram, markov_id)] = self.ep_list[
                                self._n2iIP(ngram, markov_id)] + cnt * self.ls_compensa_scale  # Increase EP ngram count by 1

                else:
                    raise Exception("Unknown dictionary given (required: ip_list, cp_list, or ep_list)")

    ########################################################################################################################

    # Determine the probability (based on the counts) of a ngram
    def _prob(self, kind):
        if kind == "ip_list":
            # no_ip_training_ngrams = 0.0  # must be a float
            # for ngram_count in self.ip_list:
            #     no_ip_training_ngrams += ngram_count
            #
            # self.ip_list[0] = self.ip_list[0] / no_ip_training_ngrams
            for mark_idx in range(self.markovs):
                offset = mark_idx * self.no_ip_ngrams
                for index in range(1, self.no_ip_ngrams):
                    self.ip_list[index+offset] = self.ip_list[index-1+offset] + self.ip_list[index+offset]  # count / all
            # Validate that prob sums to 1.0, otherwise coding error. Check for rounding errors using Decimal(1.0) instead of float(1.0)
            logging.debug("IP probability sum: {}".format(self.ip_list[-1]))
            # if not self._is_almost_equal(self.ip_list[-1], 1.0):
            #     raise Exception("ip_list probabilities do not sum up to 1.0! It is only: {}".format(self.ip_list[-1]))
        elif kind == "cp_list":
            for mark_idx in range(self.markovs):
                offset = mark_idx * self.no_cp_ngrams
                for index in range(0, self.no_cp_ngrams, self.alphabet_len):
                    # no_cp_training_ngrams = 0.0  # must be a float
                    # for x in range(index, index + self.alphabet_len):
                    #     no_cp_training_ngrams += self.cp_list[x]  # Count all ngram occurrences within one ngram-1 category
                    #
                    # self.cp_list[index] = self.cp_list[index] / no_cp_training_ngrams
                    for x in range(index + 1, index + self.alphabet_len):
                        self.cp_list[x+offset] = self.cp_list[x-1+offset] + self.cp_list[x+offset]  # count / all (of current [x])
                # Validate that prob sums to 1.0, otherwise coding error. Check for rounding errors using Decimal(1.0) instead of float(1.0)
                '''
                sum = 0.0
                for x in range(index, index+self.alphabet_len):
                    sum += self.cp_list[x]
                #logging.debug("CP probability sum: {0:.16f}".format(sum))
                '''
                # if not self._is_almost_equal(self.cp_list[index + self.alphabet_len - 1], 1.0):
                #     raise Exception("cp_list probabilities do not sum up to 1.0! It is only: {}".format(
                #         self.cp_list[index + self.alphabet_len - 1]))

        elif kind == "ep_list":
            # no_ep_training_ngrams = 0.0  # must be a float
            # for ngram_count in self.ep_list:
            #     no_ep_training_ngrams += ngram_count
            #
            # self.ep_list[0] = self.ep_list[0] / no_ep_training_ngrams
            for mark_idx in range(self.markovs):
                offset = mark_idx * self.no_ep_ngrams
                for index in range(1, self.no_ep_ngrams):
                    self.ep_list[index+offset] = self.ep_list[index - 1+offset] + self.ep_list[index+offset]  # count / all
            # Validate that prob sums to 1.0, otherwise coding error. Check for rounding errors using Decimal(1.0) instead of float(1.0)
            logging.debug("EP probability sum: {}".format(self.ep_list[-1]))
            # if not self._is_almost_equal(self.ep_list[-1], 1.0):
            #     raise Exception("ep_list probabilities do not sum up to 1.0! It is only: {}".format(self.ep_list[-1]))
        else:
            raise Exception("Unknown dictionary given (required: ip_dict, cp_dict, or ep_dict)")


    def save(self, kind):
        start = datetime.datetime.now()
        logging.debug("Start: Writing result to disk, this gonna take a while ...")
        path, file = os.path.split(self.training_file)
        directory = os.path.join('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/MSPM/SPM/trained', self.tag, str(self.ngram_size)+'gram')
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, file[:-4] + '_' + kind + '_' + \
                                str(self.ngram_size) + '_' + str(6) + '.pack'), 'wb') as fp:
            if kind == "ip_list":
                msgpack.dump(self.ip_list, fp)
            elif kind == "cp_list":
                msgpack.dump(self.cp_list, fp)
            elif kind == "ep_list":
                msgpack.dump(self.ep_list, fp)
            else:
                raise Exception("Unknown list given (required: ip_list, cp_list, or ep_list)")
        logging.debug("Done! Everything stored on disk.")
        logging.debug("Storing the data on disk took: {}".format(datetime.datetime.now() - start))

    def load(self, kind):
        start = datetime.datetime.now()

        with open(os.path.join(SPM_PATH, 'trained', self.tag, str(self.ngram_size)+'gram', args.spmdata + '-withc_' + kind + '_' + str(self.ngram_size) + '_' + str(6) + '.pack'), 'rb') as fp:
            if kind == "ip_list":
                self.ip_list = np.array(msgpack.load(fp))
                #self.ip_list_handle = cuda.to_device(self.ip_list).get_ipc_handle()
            elif kind == "cp_list":
                self.cp_list = np.array(msgpack.load(fp))
                assert (self.cp_list == self.cp_list.astype(np.int32)).prod() == 1
                self.cp_list = self.cp_list.astype(np.int32) # transform into int32 to reduce memory cost
                #self.cp_list_handle = cuda.to_device(self.cp_list).get_ipc_handle()
            elif kind == "ep_list":
                self.ep_list = np.array(msgpack.load(fp))
            else:
                raise Exception("Unknown list given (required: ip_list, cp_list, or ep_list)")
        logging.debug("Done! Everything loaded from disk.")
        logging.debug("Loading the data from disk took: {}".format(datetime.datetime.now() - start))

    ########################################################################################################################
    # seed = [length, IP, CP, CP, CP, ... ]

    def encode(self, kind, ngram, markov_id=None):
        if kind == 'ip_list':
            offset = (markov_id+1) * self.no_ip_ngrams
            index = self._n2iIP(ngram, markov_id)
            raw_seed = randint(0 if index%self.no_ip_ngrams == 0 else self.ip_list[index-1], self.ip_list[index]-1)
            range = self.ip_list[index] if index%self.no_ip_ngrams == 0 else (self.ip_list[index] - self.ip_list[index-1])
            fill = self.ip_list[offset-1]
            while True:
                seed = randint(0, 0xFFFF) * fill + raw_seed
                if seed < SEED_MAX_RANGE: # decimal 4294967295
                    break
            return seed, range / fill

        elif kind == 'cp_list':
            index = self._n2iCP(ngram, markov_id)
            _, alph_idx = divmod(index, self.no_cp_ngrams)
            raw_seed = randint(0 if (alph_idx % self.alphabet_len) == 0 else self.cp_list[index - 1].astype(np.int64), (self.cp_list[index]-1).astype(np.int64))
            range = self.cp_list[index].astype(np.int64) if (alph_idx % self.alphabet_len) == 0 else (self.cp_list[index] - self.cp_list[index - 1]).astype(np.int64)
            fill = self.cp_list[index + (self.alphabet_len - (alph_idx % self.alphabet_len)) - 1].astype(np.int64)
            while True:
                seed = randint(0, 0xFFFF) * fill + raw_seed
                if seed < SEED_MAX_RANGE:
                    break
            return seed, range / fill

        elif kind == 'length':
            fill = self.length_dic[MAX_PW_LENGTH-self.seed_length_rec]
            #assert ngram >= MIN_PW_LENGTH and ngram < MAX_PW_LENGTH
            range = self.length_dic[ngram] - self.length_dic[ngram-1]
            raw_seed = randint(self.length_dic[ngram - 1], self.length_dic[ngram] - 1)
            while True:
                seed = randint(0, 0xFFFF) * fill + raw_seed
                if seed < SEED_MAX_RANGE:
                    break
            return seed, range/fill

        else:
            raise Exception("Unknown dictionary given (required: ip_dict, cp_dict, or ep_dict)")

    # pw -> seed
    def encode_pw(self, pw, probfree=False):
        if len(pw) > MAX_PW_LENGTH:
            raise Exception("password length no bigger than {}".format(MAX_PW_LENGTH))
        if len(pw) in TRAIN_LENGTH:
            markov_id = TRAIN_LENGTH.index(len(pw))
        else:
            markov_id = len(TRAIN_LENGTH)

        ip = pw[:self.ngram_size - 1]
        sequence_prob = []
        ip_seed, p = self.encode('ip_list', ip, markov_id)
        sequence_prob.append(p)
        seed = [ip_seed]
        old_pos = 0
        for new_pos in range(self.ngram_size, len(pw) + 1, 1):
            cp = pw[old_pos:new_pos]
            cp_seed, p = self.encode('cp_list', cp, markov_id)
            sequence_prob.append(p)
            seed.append(cp_seed)
            old_pos += 1

        length = len(seed)
        l_seed, l_prob = self.encode('length', length)
        seed.insert(0, l_seed)
        sequence_prob.append(l_prob)
        leng = SEED_LEN - 1 if not probfree else SEED_LEN
        while len(seed) < leng:
            seed.append(randint(0, SEED_MAX_RANGE))
        return seed, sequence_prob

    def decode_ip(self, seed, markov_id, idxonly=False):
        offset = markov_id * self.no_ip_ngrams
        seed = seed % self.ip_list[(markov_id+1) * self.no_ip_ngrams-1]
        threadsperblock = 32
        blockspergrid = (self.ip_list.size + threadsperblock) // threadsperblock
        #idx = np.array([0])
        #gpu_decode[blockspergrid, threadsperblock](seed, offset, self.ip_list[offset:(markov_id+1) * self.no_ip_ngrams], idx)
        idx = np.array([np.searchsorted(self.ip_list[offset:(markov_id + 1) * self.no_ip_ngrams], seed, side='right') + offset])

        if idxonly:
            return idx[0]
        return self._i2nIP(idx[0])

    def decode_cp(self, seed, pre, markov_id):
        ngram = pre + self.alphabet_list[0]
        start = self._n2iCP(ngram, markov_id)
        seed = seed % self.cp_list[start+self.alphabet_len-1]

        threadsperblock = 32
        blockspergrid = (self.alphabet_len + threadsperblock) // threadsperblock
        #idx = np.array([0])
        #gpu_decode[blockspergrid, threadsperblock](seed, start, self.cp_list[start:start+self.alphabet_len], idx)
        idx = np.array([np.searchsorted(self.cp_list[start:start + self.alphabet_len], seed, side='right') + start])

        return self._i2nCP(idx[0])

    def decode_len(self, seed):
        seed = seed % self.length_dic[MAX_PW_LENGTH-self.seed_length_rec]
        for x in self.length_dic:
            if seed < self.length_dic[x]:
                return x

    # seed -> pw
    def decode_pw(self, seed):
        if len(seed) != SEED_LEN and len(seed) != (SEED_LEN - 1):
            raise Exception("seed length must be {}, now is {}".format(SEED_LEN, len(seed)))
        len_seed = seed[0]
        length = self.decode_len(len_seed)
        pw_len = length + self.seed_length_rec
        if pw_len in TRAIN_LENGTH:
            markov_id = TRAIN_LENGTH.index(pw_len)
        else:
            markov_id = len(TRAIN_LENGTH)
        ip_seed = seed[1]
        ip = self.decode_ip(ip_seed, markov_id)
        pw = ip
        # print(length)
        for x in range(2, length + 1):
            cp = self.decode_cp(seed[x], pre=pw[-self.ngram_size + 1:], markov_id=markov_id)
            pw += cp[-1]
        return pw