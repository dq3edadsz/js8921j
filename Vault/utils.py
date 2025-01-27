import random
import numpy as np
from cryptography.hazmat.primitives import hashes
#import torch
from hashlib import sha256
from Crypto.Hash import SHA512, SHA256
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF1, PBKDF2
from Crypto.Util import Counter
import struct
import secrets
from linecache import getline


def permutation(raw_lst, pw_num, pin, mpw, recover=False, kernel=3, stride=1, MAX_PWS=500, MAX_LEN=10000):

    digest = hashes.Hash(hashes.SHA256())
    mpw = mpw if isinstance(mpw, str) else str(mpw)
    pin = pin if isinstance(pin, str) else str(pin)
    digest.update(bytes((mpw+pin).encode("ascii")))
    random.seed(digest.finalize())
    seq = [random.randint(0, MAX_PWS) for _ in range(MAX_LEN)]
    rolls = [vali for vali in seq if vali < pw_num][:pw_num+kernel-1]
    rolls = minpooling1d(rolls)
    if not recover:
        for i in range(pw_num - 1, -1, -1):
            raw_lst[i], raw_lst[rolls[i]] = raw_lst[rolls[i]], raw_lst[i]
    else:
        for i in range(pw_num):
            raw_lst[i], raw_lst[rolls[i]] = raw_lst[rolls[i]], raw_lst[i]
    return raw_lst

'''def minpooling1d_torch(inp, size=3, stride=1):

    inp = -torch.from_numpy(np.array(inp))[None, None].float()
    inp = -torch.nn.functional.max_pool1d(inp, kernel_size=size, stride=stride)
    return list(inp.squeeze().long().numpy())'''

def minpooling1d(feature_map, size=3, stride=1):
    #Preparing the output of the pooling operation.
    pool_out = np.zeros((np.uint16((len(feature_map)-size)/stride+1)))
    r2 = 0
    for r in np.arange(0,len(feature_map)-size+1, stride):
        pool_out[r2] = np.min([feature_map[r:r+size]])
        r2 = r2 +1
    return list(pool_out.astype(np.long))


#salt = 0x12345678.to_bytes(8, 'little')
def set_crypto(mpw, salt, ctr):
    # salt = hash(pin).to_bytes(8, 'little')
    key = PBKDF2(mpw, salt, 32, hmac_hash_module=SHA256)
    aes = AES.new(key, AES.MODE_CTR, counter=ctr)
    return aes


def main():
    mpw_tmp1 = 'awearwE2EA24'
    mpw_tmp2 = '131easd12'
    mpw_tmp3 = 'QERQ3DCXxca$e'
    msg1 = list(range(40))
    msg2 = list(range(40,80))
    msg3 = list(range(80,120))
    cpht = bytearray()
    salt = secrets.token_hex(8)

    ctr = Counter.new(8 * 16, initial_value=int(1))
    aes = set_crypto(mpw_tmp1, salt=salt, ctr=ctr)
    cpht += aes.encrypt(struct.pack('{}L'.format(len(msg1)), *msg1))
    ctr = Counter.new(8 * 16, initial_value=int(21))
    aes = set_crypto(mpw_tmp2, salt=salt, ctr=ctr)
    cpht += aes.encrypt(struct.pack('{}L'.format(len(msg2)), *msg2))
    ctr = Counter.new(8 * 16, initial_value=int(41))
    aes = set_crypto(mpw_tmp3, salt=salt, ctr=ctr)
    cpht += aes.encrypt(struct.pack('{}L'.format(len(msg3)), *msg3))

    ctr = Counter.new(8 * 16, initial_value=int(1))
    aes = set_crypto(mpw_tmp2, salt=salt, ctr=ctr)
    msg = aes.decrypt(cpht[: 80*8])
    msg = list(struct.unpack('{}L'.format(len(msg)//8), msg))[40:80]
    print(msg == msg2)

if __name__ == '__main__':
    #main()
    print(int(sha256(str.encode('123456')).hexdigest(), 16) % int(1e8))
    print(int(sha256(str.encode('123456')).hexdigest(), 16) % int(1e8))
    print(int(sha256(str.encode('123456')).hexdigest(), 16) % int(1e8))
    print(hash('12142412') % int(1e8))
    print(hash('12142412') % int(1e8))
    print(hash('12142412') % int(1e8))