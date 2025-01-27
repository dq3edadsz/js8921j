
# encode decode settings
MAX_PW_LENGTH = 30 # max length for seed with pathencode
max_pw_length = MAX_PW_LENGTH - 1 # without pathencode
MIN_PW_LENGTH = 3 # minimum pw length
ALPHA = 0.3135 # in sspm for encoding direct use
SEED_MAX_RANGE = 0xFFFFFFFFFF

REPR_SIZE = 3  # number of bytes to represent an integer. normally 4 bytes. But
               # we might go for higher values for better security.
MAX_INT = 256 ** REPR_SIZE  # value of maximum integer in this representation.

# training settings
TEST_FOLD = 5 # fold id used for testing (pastebin) 1~5
MAX_PW_NUM = 250
SCALAR = 2**20
VAULT_SIZE = 20 # maximum number of pws in vault (static)
SOURCE_PATH = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault'
PASTB_PATH = SOURCE_PATH + '/data/pastebin/fold5'
ROCKY_PATH = SOURCE_PATH + '/data/rockyou-withcount.txt.bz2'
ROCKY_NOCOUNT_PATH = SOURCE_PATH + '/data/rockyou.txt.gz'
SPM_PATH = SOURCE_PATH + '/MSPM/SPM'
DICT_SIZE = 10000 # not certain
PS_DECOYSIZE = 100 # decoy vault numbers determining password similarity p_decoy(Vi)
LOGICAL = False # use logical expansion or not
PROCESS_NUM = 1 # 100 default
LEAKED_DM = 'google.com'
LEAKED_PW = 'GOOGLE'
N_EXP_VAULTS = 5 # number of decoys + a real (typically 999 + 1 = 1000)