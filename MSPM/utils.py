from MSPM.unreused_prob import unreuse_p
import os
import sys
from os.path import expanduser
import struct
# opens file checking whether it is bz2 compressed or not.
import gzip
import string
import numpy as np
import random as orig_random
BASE_DIR = os.getcwd()
from MSPM.mspm_config import MAX_PW_NUM, MAX_PW_LENGTH, SEED_LEN, SCALAR

home = expanduser("~")
pwd = os.path.dirname(os.path.abspath(__file__))
regex = r'([A-Za-z_]+)|([0-9]+)|(\W+)'
char_group = string.printable


class random:
    @staticmethod
    def randints(s, e, n=1):
        """
        returns n uniform random numbers from [s, e] (including both ends)
        """
        if e == s:
            return [s]*n
        assert e > s, "Wrong range: [{}, {}]".format(s, e)
        n = max(1, n)
        #orig_random.seed(0)
        arr = [orig_random.randint(s, e) for _ in range(n)]
        return arr

    @staticmethod
    def randint(s, e):
        """
        returns one random integer between s and e. Try using @randints in case you need
        multiple random integer. @randints is more efficient
        """
        return random.randints(s, e, 1)[0]

    @staticmethod
    def choice(arr):
        i = random.randint(0, len(arr) - 1)
        assert i < len(arr), "Length exceeded by somehow! Should be < {}, but it is {}" \
            .format(len(arr), i)
        return arr[i]

    @staticmethod
    def sample(arr, n):
        return [arr[i] for i in random.randints(0, len(arr) - 1, n)]


def gen_gilst():
    """
    :return: cumulated list of gi_lst for encoding and decoding
    """
    gi = [unreuse_p(i) for i in range(MAX_PW_NUM)]
    gi[0] = 1.
    gilst = []
    for m in range(MAX_PW_NUM):
        cum = [gi[m]]
        cum.extend([(1-cum[0])/(m+1)]*m)
        gilst.append(np.array([round((np.array(cum[:i]).sum()) * SCALAR) for i in range(len(cum)+1)]))
    return gilst

def main():
    gilst = gen_gilst()
    print(gilst)

if __name__ == '__main__':
    main()