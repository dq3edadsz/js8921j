#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

''' The Markov model
:author: Maximilian Golla
:contact: maximilian.golla@rub.de
:version: 0.7.1, 2019-07-11
'''

# External modules
from collections import OrderedDict  # storing the alphabet
import os  # load and save / file handling
import umsgpack  # load and save # pip install u-msgpack-python
import math  # only pow
import logging  # logging debug infos
from tqdm import tqdm  # progress bar while reading the file # pip install tqdm
from random import uniform, randint
import datetime
from MSPM.mspm_config import *
import _pickle as cPickle

def is_almost_equal(a, b, rel_tol=1e-09, abs_tol=0.0):
    # print '{0:.16f}'.format(a), '{0:.16f}'.format(b)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class NGramCreator:

    def __init__(self, dict):
        self.name = dict['name']
        logging.debug("Constructor started for '{}'".format(self.name))
        self.alphabet = dict['alphabet']
        self.alphabet_len = len(self.alphabet)
        self.alphabet_dict = OrderedDict.fromkeys(self.alphabet)  # a 0, b 1, c 2
        i = 0
        for char in self.alphabet_dict:
            self.alphabet_dict[char] = i
            i += 1
        self.alphabet_list = list(self.alphabet)
        logging.debug("Used alphabet: {}".format(self.alphabet))
        self.length = MIN_PW_LENGTH
        logging.debug("Model string length: {}".format(self.length))
        self.ngram_size = dict['ngram_size']
        self.laplace_smooth = dict['laplace_smooth']
        self.ls_compensa_scale = dict['ls_compensation']
        assert self.ngram_size >= 2, "n-gram size < 2 does not make any sense! Your configured n-gram size is {}".format(
            self.ngram_size)
        logging.debug("NGram size: {}".format(self.ngram_size))
        self.training_file = dict['training_file']
        with open(SOURCE_PATH+self.training_file, encoding='latin1') as f:
            for count, line in enumerate(f):
                pass
        self.training_file_lines = count + 1
        self.disable_progress = False if dict['progress_bar'] else True
        self.length_dic = {}
        self.ip_list = []
        self.cp_list = []
        self.ep_list = []
        self.no_ip_ngrams = int(math.pow(self.alphabet_len, (self.ngram_size - 1)))
        self.no_cp_ngrams = int(math.pow(self.alphabet_len, (self.ngram_size)))
        self.no_ep_ngrams = self.no_ip_ngrams  # save one exponentiation :-P
        self._init_lengthdic()
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

    def _init_lengthdic(self):
        dic = cPickle.load(open(SOURCE_PATH + "/MSPM/SPM/trained/lengthstatistic", "rb"))
        self.length_dic = {}
        for length in range(MIN_PW_LENGTH, MAX_PW_LENGTH):
            if length == MIN_PW_LENGTH:
                self.length_dic[length] = dic[length]
            else:
                self.length_dic[length] = self.length_dic[length-1] + dic[length]

    ########################################################################################################################

    # ngram-to-intial-prob-index
    def _n2iIP(self, ngram):
        ngram = list(ngram)
        if self.ngram_size == 5:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[3]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[2]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 3 * self.alphabet_dict[ngram[0]])
        if self.ngram_size == 4:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[2]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[0]])
        if self.ngram_size == 3:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[1]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[0]])
        if self.ngram_size == 2:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[0]])

    # intial-prob-index-to-ngram
    def _i2nIP(self, index):
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
    def _n2iCP(self, ngram):
        ngram = list(ngram)
        if self.ngram_size == 5:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[4]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[3]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[2]]) + (
                               self.alphabet_len ** 3 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 4 * self.alphabet_dict[ngram[0]])
        if self.ngram_size == 4:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[3]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[2]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 3 * self.alphabet_dict[ngram[0]])
        if self.ngram_size == 3:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[2]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[1]]) + (
                               self.alphabet_len ** 2 * self.alphabet_dict[ngram[0]])
        if self.ngram_size == 2:
            return (self.alphabet_len ** 0 * self.alphabet_dict[ngram[1]]) + (
                        self.alphabet_len ** 1 * self.alphabet_dict[ngram[0]])

    # conditial-prob-index-to-ngram
    def _i2nCP(self, index):
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
            for i in range(0, int(math.pow(self.alphabet_len, self.ngram_size - 1))):
                self.ip_list.append(self.laplace_smooth)  # Smoothing, we initialize every possible ngram with count = 1
        elif kind == "cp_list":
            for i in range(0, int(math.pow(self.alphabet_len, self.ngram_size))):
                self.cp_list.append(self.laplace_smooth)  # Smoothing, we initialize every possible ngram with count = 1
        elif kind == "ep_list":
            for i in range(0, int(math.pow(self.alphabet_len, self.ngram_size - 1))):
                self.ep_list.append(self.laplace_smooth)  # Smoothing, we initialize every possible ngram with count = 1
        else:
            raise Exception('Unknown list given (required: ip_list, cp_list, or ep_list)')

    def _count(self, kind):
        with open(self.training_file, encoding='latin1') as input_file:
            for line in tqdm(input_file, desc=self.training_file, total=self.training_file_lines,
                             disable=self.disable_progress, miniters=int(self.training_file_lines/1000), unit="pw"):
                L = line.lstrip().strip('\r\n').split(' ')
                cnt, word = int(L[0]), L[1]
                # if len(word) != self.length:
                if len(word) < self.length:
                    continue

                if kind == "ip_list":
                    if self._is_in_alphabet(word):  # Filter non-printable
                        ngram = word[0:self.ngram_size - 1]  # Get IP ngram
                        self.ip_list[self._n2iIP(ngram)] = self.ip_list[
                                                               self._n2iIP(ngram)] + cnt * self.ls_compensa_scale  # Increase IP ngram count by 1

                elif kind == "cp_list":
                    if self._is_in_alphabet(word):  # Filter non-printable
                        old_pos = 0
                        for new_pos in range(self.ngram_size, len(word) + 1,
                                             1):  # Sliding window: pas|ass|ssw|swo|wor|ord
                            ngram = word[old_pos:new_pos]
                            old_pos += 1
                            self.cp_list[self._n2iCP(ngram)] = self.cp_list[self._n2iCP(
                                ngram)] + cnt * self.ls_compensa_scale  # Increase CP ngram count by 1

                elif kind == "ep_list":
                    if self._is_in_alphabet(word):  # Filter non-printable
                        ngram = word[-self.ngram_size + 1:]  # Get EP ngram
                        self.ep_list[self._n2iIP(ngram)] = self.ep_list[
                                                               self._n2iIP(ngram)] + cnt * self.ls_compensa_scale  # Increase EP ngram count by 1

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
            for index in range(1, len(self.ip_list)):
                self.ip_list[index] = self.ip_list[index - 1] + self.ip_list[index]  # count / all
            # Validate that prob sums to 1.0, otherwise coding error. Check for rounding errors using Decimal(1.0) instead of float(1.0)
            logging.debug("IP probability sum: {}".format(self.ip_list[-1]))
            # if not self._is_almost_equal(self.ip_list[-1], 1.0):
            #     raise Exception("ip_list probabilities do not sum up to 1.0! It is only: {}".format(self.ip_list[-1]))
        elif kind == "cp_list":
            for index in range(0, len(self.cp_list), self.alphabet_len):
                # no_cp_training_ngrams = 0.0  # must be a float
                # for x in range(index, index + self.alphabet_len):
                #     no_cp_training_ngrams += self.cp_list[x]  # Count all ngram occurrences within one ngram-1 category
                #
                # self.cp_list[index] = self.cp_list[index] / no_cp_training_ngrams
                for x in range(index + 1, index + self.alphabet_len):
                    self.cp_list[x] = self.cp_list[x - 1] + self.cp_list[x]  # count / all (of current [x])
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
            for index in range(1, len(self.ep_list)):
                self.ep_list[index] = self.ep_list[index - 1] + self.ep_list[index]  # count / all
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
        directory = os.path.join('trained', str(self.ngram_size)+'gram')
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, file[:-4] + '_' + kind + '_' + \
                                str(self.ngram_size) + '_' + str(MIN_PW_LENGTH) + '.pack'), 'wb') as fp:
            if kind == "ip_list":
                umsgpack.dump(self.ip_list, fp)
            elif kind == "cp_list":
                umsgpack.dump(self.cp_list, fp)
            elif kind == "ep_list":
                umsgpack.dump(self.ep_list, fp)
            else:
                raise Exception("Unknown list given (required: ip_list, cp_list, or ep_list)")
        logging.debug("Done! Everything stored on disk.")
        logging.debug("Storing the data on disk took: {}".format(datetime.datetime.now() - start))

    def load(self, kind):
        start = datetime.datetime.now()
        path, file = os.path.split(self.training_file)
        with open(os.path.join(SPM_PATH, 'trained', str(self.ngram_size)+'gram', file[:-4] + '_' + kind + '_' + \
                                str(self.ngram_size) + '_' + str(MIN_PW_LENGTH) + '.pack'), 'rb') as fp:
            if kind == "ip_list":
                self.ip_list = umsgpack.load(fp)
            elif kind == "cp_list":
                self.cp_list = umsgpack.load(fp)
            elif kind == "ep_list":
                self.ep_list = umsgpack.load(fp)
            else:
                raise Exception("Unknown list given (required: ip_list, cp_list, or ep_list)")
        logging.debug("Done! Everything loaded from disk.")
        logging.debug("Loading the data from disk took: {}".format(datetime.datetime.now() - start))

    ########################################################################################################################
    # seed = [length, IP, CP, CP, CP, ... ]

    def encode(self, kind, ngram):
        if kind == 'ip_list':
            index = self._n2iIP(ngram)
            raw_seed = randint(0 if index == 0 else self.ip_list[index - 1], self.ip_list[index]-1)
            range = self.ip_list[index] if index == 0 else (self.ip_list[index] - self.ip_list[index - 1])
            fill = self.ip_list[-1]
            while True:
                seed = randint(0, 0xFFFF) * fill + raw_seed
                if seed < 0xFFFFFFFF:
                    break
            return seed, range / fill

        elif kind == 'cp_list':
            index = self._n2iCP(ngram)
            raw_seed = randint(0 if (index % self.alphabet_len) == 0 else self.cp_list[index - 1], self.cp_list[index]-1)
            range = self.cp_list[index] if (index % self.alphabet_len) == 0 else (self.cp_list[index] - self.cp_list[index - 1])
            fill = self.cp_list[index + (self.alphabet_len - (index % self.alphabet_len)) - 1]
            while True:
                seed = randint(0, 0xFFFF) * fill + raw_seed
                if seed < 0xFFFFFFFF:
                    break
            return seed, range / fill

        elif kind == 'length':
            fill = self.length_dic[MAX_PW_LENGTH-1]
            assert ngram >= MIN_PW_LENGTH and ngram < MAX_PW_LENGTH
            raw_seed = randint(0 if ngram == MIN_PW_LENGTH else self.length_dic[ngram - 1], \
                               self.length_dic[ngram] - 1)
            while True:
                seed = randint(0, 0xFFFF) * fill + raw_seed
                if seed < 0xFFFFFFFF:
                    break
            return seed

        else:
            raise Exception("Unknown dictionary given (required: ip_dict, cp_dict, or ep_dict)")

    # pw -> seed
    def encode_pw(self, pw, probfree=False):
        if len(pw) > MAX_PW_LENGTH:
            raise Exception("password length no bigger than {}".format(MAX_PW_LENGTH))

        ip = pw[:self.ngram_size - 1]
        sequence_prob = []
        ip_seed, p = self.encode('ip_list', ip)
        sequence_prob.append(p)
        seed = [ip_seed]
        old_pos = 0
        for new_pos in range(self.ngram_size, len(pw) + 1, 1):
            cp = pw[old_pos:new_pos]
            cp_seed, p = self.encode('cp_list', cp)
            sequence_prob.append(p)
            seed.append(cp_seed)
            old_pos += 1

        length = len(seed)
        seed.insert(0, self.encode('length', length))
        leng = MAX_PW_LENGTH - 1 if not probfree else MAX_PW_LENGTH
        while len(seed) < leng:
            seed.append(randint(0, 0xFFFFFFFF))
        return seed, sequence_prob

    def decode_ip(self, seed):
        seed = seed % self.ip_list[-1]
        for x in range(len(self.ip_list)):
            if seed < self.ip_list[x]:
                return self._i2nIP(x)
        raise Exception("Cant decode IP seed {}, something wrong!".format(seed))

    def decode_cp(self, seed, pre):
        ngram = pre + self.alphabet_list[0]
        start = self._n2iCP(ngram)
        seed = seed % self.cp_list[start+self.alphabet_len-1]
        for x in range(start, start+self.alphabet_len):
            if seed < self.cp_list[x]:
                return self._i2nCP(x)
        raise Exception("Cant decode CP seed {}, something wrong!".format(seed))

    def decode_len(self, seed):
        seed = seed % self.length_dic[MAX_PW_LENGTH-1]
        for x in self.length_dic:
            if seed < self.length_dic[x]:
                return x

    # seed -> pw
    def decode_pw(self, seed):
        if len(seed) != MAX_PW_LENGTH and len(seed) != (MAX_PW_LENGTH-1):
            raise Exception("seed length must be {}, now is {}".format(MAX_PW_LENGTH, len(seed)))
        len_seed = seed[0]
        length = self.decode_len(len_seed)
        ip_seed = seed[1]
        ip = self.decode_ip(ip_seed)
        pw = ip
        #print(length)
        for x in range(2, length+1):
            cp = self.decode_cp(seed[x], pre=pw[-self.ngram_size+1:])
            pw += cp[-1]
        return pw
