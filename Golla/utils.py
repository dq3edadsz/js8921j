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
    :return: cumulated list of reuse list for encoding and decoding
    """
    gi = [0.66, 0.06, 0.02, 0.01, 0.015] # according to golla ccs16 paper (M=(0.66, 0.06, 0.02, 0.01, 0.015) for pb), reuse rate for related passwords of edit distance of 0, 1, 2, 3, 4
    gi.append(1 - sum(gi)) # add reuse rate for unrelated passwords
    cum = np.array([round((np.array(gi[:i]).sum()) * SCALAR) for i in range(len(gi)+1)])
    return cum

def main():
    gilst = gen_gilst()
    print(gilst)

if __name__ == '__main__':
    main()