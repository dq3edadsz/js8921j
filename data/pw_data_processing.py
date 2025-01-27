import re
from tqdm import tqdm
import pickle as cPickle
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from MSPM.mspm_config import SOURCE_PATH

def clean_pwdata():
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/neopets-plain.txt', 'r') as file:
        lines = [line.rstrip() for line in file]

    ## clean pw dataset

    # step 1: remove non-ascii
    totalpws = len(lines)
    deleted = 0
    for ith in tqdm(range(totalpws)):
        if lines[ith - deleted].isascii() == False:
            lines.pop(ith - deleted)
            deleted += 1
    print('removed non-ascii:', deleted)

    # step 2: remove any with \x sort of
    totalpws = len(lines)
    deleted = 0
    for ith in tqdm(range(totalpws)):
        pw = lines[ith - deleted]
        if re.search(r'\\x',
                     pw) or '\x7f' in pw or '\x03' in pw or '\x00' in pw or '\x01' in pw or '\x02' in pw or '\x04' in pw or '\x05' in pw or '\x06' in pw or '\x07' in pw or '\x08' in pw or '\x0b' in pw or '\x0c' in pw or '\x0e' in pw or '\x0f' in pw or '\x10' in pw or '\x11' in pw or '\x12' in pw or '\x13' in pw or '\x14' in pw or '\x15' in pw or '\x16' in pw or '\x17' in pw or '\x18' in pw or '\x19' in pw or '\x1a' in pw or '\x1b' in pw or '\x1c' in pw or '\x1d' in pw or '\x1e' in pw or '\x1f' in pw or '\t' in pw:
            lines.pop(ith - deleted)
            deleted += 1
    print('removed hex sort of:', deleted)

    # step 3: remove shorter than 4 or longer than 30
    newlines3 = []
    for pw_ in tqdm(lines):
        if not(len(pw_) < 4 or len(pw_) > 30):
            newlines3.append(pw_)
    print('removed length unqualified:', len(lines) - len(newlines3))

    # write
    '''with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/neopets-flat.txt', 'w') as f:
        # write elements of list
        for items in newlines3:
            f.write('%s\n' % items)
        print("File written successfully")
    f.close()'''

def process_raw_neopets():
    # read file names from dir "/home/beeno/Downloads/Neopets_BF/data"
    mypath = '/home/beeno/Downloads/Neopets_BF/data'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    passwords = []
    for file in tqdm(onlyfiles):
        # read txt file
        lines, invalid_count = [], 0
        with open(mypath + '/' + file, 'r', errors='ignore') as file:
            for line in file:
                try:
                    lines.append(line.rstrip().split(':')[1]) #
                except:
                    invalid_count += 1
        print('invalid code:', invalid_count)#[line.rstrip() for line in file]
        print('password count', len(passwords))
        passwords.extend(lines)
    # write passwords to file
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/neopets-plain.txt', 'w') as f:
        for pw in passwords:
            f.write('%s\n' % pw)
        print("File written successfully")


def count_pws():
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/neopets-flat.txt', 'r') as file:
        lines = [line.rstrip() for line in file]
    pw2count = defaultdict(int)
    for pw in tqdm(lines):
        pw2count[pw] += 1
    pw_count_lst = []
    for pw, count in tqdm(pw2count.items()):
        pw_count_lst.append(str(count) + ' ' + pw + '\r\n')
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/neopets-withcount', 'wb') as fb:
        cPickle.dump(pw_count_lst, fb)


if __name__ == '__main__':
    #process_raw_neopets()
    #clean_pwdata()
    #count_pws()

    length_dict = {}
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/neopets-withcount', "rb") as f:
        lines = cPickle.load(f)
    for line in tqdm(lines):
        contents = line.strip().split(' ') # in the format of 'count password'
        if len(contents) != 2:
            continue
        if len(contents[1]) in length_dict:
            length_dict[len(contents[1])] += int(contents[0])
        else:
            length_dict[len(contents[1])] = int(contents[0])
    print(length_dict)
    cPickle.dump(length_dict, open(SOURCE_PATH + "/MSPM/SPM/trained/neopets_lengthstatistic", "wb"))