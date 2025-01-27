SOURCE_PATH = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu'
ROCKY_PATH = SOURCE_PATH + '/data/'
ROCKY_PATH_TRAIN = SOURCE_PATH + '/data/rockyou-withcount-train'
ROCKY_PATH_TEST = SOURCE_PATH + '/data/rockyou-withcount-test'
ROCKY_NOCOUNT_PATH = SOURCE_PATH + '/data/rockyou.txt'
SPM_PATH = SOURCE_PATH + '/MSPM/SPM'

### frequently modifed
PASTB_PATH = SOURCE_PATH + '/data/breachcompilation/fold2' #'/data/pastebin/fold2' #
ALPHA = 0.4 #_bc;0.5710 #_pb;
T_PHY = N_EXP_VAULTS = REFRESH_RATE = 4000
PIN_SAMPLE = int(10**6) # 10^x for x-pin
Nitv = 2 # fixed-interval shuffling
###

# encode decode settings
SEED_LEN = 40
MAX_PW_LENGTH = 30 # max length for seed with pathencode
max_pw_length = MAX_PW_LENGTH - 1 # without pathencode
MIN_PW_LENGTH = 4 # minimum pw length
TRAIN_LENGTH = [4, 5, 6, 7, 8, 9, 10, 11, 12]
SEED_MAX_RANGE = 0xFFFFFFFFFF
REPR_SIZE = 3  # number of bytes to represent an integer. normally 4 bytes. But
               # we might go for higher values for better security.
MAX_INT = 256 ** REPR_SIZE  # value of maximum integer in this representation.

# training settings
TEST_FOLD = [1] #4,1,2,3,5VAULT_SIZE = 20 # maximum number of pws in vault (static)
PROCESS_NUM = 1 # 16 multi-thread
REPEAT_NUM = 20
TEST_DATA_ID = 0
PS_DECOYSIZE = 1000 #5000 # decoy vault numbers determining password similarity p_decoy(Vi)
PIN_REFRESH = 1 # 500(T_PHY=500); 1000(T_PHY=2000)
PIN_ELEMENTS = [str(n) for n in range(10)]
SCALE_PROCESSOR = 1
MAX_PW_NUM = 210
SCALAR = 2**20
DICT_SIZE = 10000 # not certain
LOGICAL = False # use logical expansion or not


# online verification setting
N_ol = 5 # online verify at most 3 times and if all succeed, then attacker suggest the guess is real

#   SPM
ALPHABET = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

# single password model evaluation (appendix E)
PRE_DECOYS = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/decoydict/dict1000_100_15.data'
REAL_PW_NUM = 10 # 10000
REAL_PW_NUM = int(int(REAL_PW_NUM/PROCESS_NUM) * PROCESS_NUM)

# single similar password training settings
SAMPLES_PERP = 10 # number of samples per password drawn from rockyou when training sspm (top-k nearest)
SSPM_CHARS_MAX = 14


def lencheck(pw):
    # true for unqualified pw
    return (not ((len(pw) >= MIN_PW_LENGTH) and (len(pw) <= MAX_PW_LENGTH)))

def not_in_alphabet(string):
    for char in string:
        if not char in ALPHABET:
            return True
    return False