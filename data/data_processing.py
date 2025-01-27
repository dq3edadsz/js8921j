import copy
from multiprocessing import Pool
from glob import glob
import mgzip
import re
import pickle
import numpy as np
from tqdm import tqdm
import json
from scipy.optimize import curve_fit
from collections import defaultdict
import pylcs
import orjson
import random
import gc
from time import time
import os


def find_clusters(pws_, progress):
    def rulematch(pw1, pw2):
        # 1.identical
        if pw1 == pw2:
            return True
        # 2. substring or 3. capitalization
        if pw1.lower() in pw2.lower() or pw2.lower() in pw1.lower():
            return True
        # 5. reversal
        if pw1[::-1] == pw2:
            return True
        # 7. common substring
        if pylcs.lcs_string_length(pw1.lower(), pw2.lower()) >= int(max(len(pw1), len(pw2)) / 2):
            return True
        # 4. l33t
        if leet_match(pw1, pw2) or leet_match(pw2, pw1):
            return True
        return False
    def leet_match(pw, pw2leet):
        # transform pw2leet into leet form
        # return True if pw is a substring of pw2leet or vice versa
        leet_charset = {'a': ['4', '@'], 'e': ['3'], 'i': ['1', '!', '¡'], 'l': ['1'], 'o': ["0"], 's': ['$', '5'],
                        'b': ['8'], 't': ['7'], 'c': ['('], '9': ['6'], 'z': ['2']}
        new_wordlist = [pw2leet] # to be leeted
        record_wordlist = [] # make sure no redundant words to be leeted
        cnt = 0
        while len(new_wordlist) > 0:
            w = new_wordlist.pop()
            for i, char in enumerate(w):
                if char.lower() in list(leet_charset.keys()):
                    for leeted_char in leet_charset[char.lower()]:
                        leeted_word = w[:i] + leeted_char + w[i + 1:]
                        if leeted_word not in record_wordlist:
                            cnt += 1
                            new_wordlist.append(leeted_word)
                            record_wordlist.append(leeted_word)
                        if pw.lower() in leeted_word.lower() or leeted_word.lower() in pw.lower():
                            return True
                        if cnt > 100:
                            return False

    # input paasswords from a vault
    # output grouped passwords
    #print(progress)
    pws = pws_.copy()
    s_pws = len(set(pws))
    vs = len(pws)
    #  passwords will be groupeed if they follow at least one of the following rules: 1.identical; 2. substring; 3. capitalization; 4. l33t; 5. reversal; 6. common substring
    groups = [[pws.pop()]]
    for pw in pws:
        matched = 0
        for i, group in enumerate(groups):
            for pw_trg in group:
                if rulematch(pw, pw_trg):
                    groups[i].append(pw)
                    matched = 1
                    break
            if matched == 1:
                break
        if matched == 0:
            groups.append([pw])
    assert sum([len(g) for g in groups]) == (len(pws) + 1)
    return groups, s_pws, vs

def find_max_edit_distance(pw_list):
    """
    Finds the maximum edit distance among a list of passwords efficiently.
    Args:
        pw_list: A list of passwords (strings).

    Returns:
        The maximum edit distance found, or None if the list is empty.
    """
    if len(pw_list) <= 1:
        return 0  # Handle empty or single element case

    max_distance = 0
    n = 0
    for i in range(len(pw_list)-1):
        for j in range(i + 1, len(pw_list)):
            # Calculate edit distance between all unique pairs using nested loops
            distance = pylcs.edit_distance(pw_list[i], pw_list[j])
            #max_distance = max(max_distance, distance)
            max_distance += distance
            n += 1
    return max_distance / n

def cluster_modification(pws, i, k):
    print(i)
    clst = find_clusters(pws, 0)[0]
    cmr = np.array([find_max_edit_distance(c) / np.array([len(pw) for pw in c]).mean() for c in clst])
    if (cmr > 0).sum() == 0:
        cmr = 0
    else:
        cmr = cmr[cmr > 0].mean()
    return cmr, pws, round(np.array([len(pw_) for pw_ in pws]).mean(), 4), len(clst) #len(clst)# / len(clst)

def file2emailpws(cnt, path):
    #   such path has subdirectories and text files, processing will iterate through all of them
    #   each line of files has the format of "email:password"
    # return: the list of lines of the file
    print(str(cnt / 1333 * 100) + '%')
    with open(path, 'r', encoding = "ISO-8859-1") as f:
        lines = f.readlines()
    return lines


def dir2file():
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailpws_files/"
    data_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/data"
    cnt = 0
    for fileid, subdir_or_filepath in enumerate([d for d in glob(data_dir + '/*', recursive=True)]):
        pool = Pool(30)
        workers = []
        all_files = [subdir_or_filepath]
        while [os.path.isdir(p) for p in all_files].count(True) > 0:
            toremove, toextend = [], []
            for p in all_files:
                if os.path.isdir(p):
                    toremove.append(p)
                    toextend.extend([f for f in glob(p + '/*', recursive=True)])
            all_files.extend(toextend)
            for p in toremove:
                all_files.remove(p)

        for path_ in all_files:
            cnt += 1
            workers.append(pool.apply_async(file2emailpws, (cnt, path_, )))
        pool.close()
        pool.join()
        total = []
        for w in workers:
            total.extend(w.get())
        with mgzip.open(write_dir + str(fileid) + ".txt.gz", "wb") as f:
            pickle.dump(total, f)

def group_by_email(readpath, write_path):
    # emailpws: list, with each element as string "email:password"
    # return:list of dictionaries, each dict with key as tuple of emails (email is format of address@domain), values as list of passwords (passwords grouped by the same or similar email (the ones with the same address))
    with mgzip.open(readpath, "rb") as f:
        emailpws = pickle.load(f)
    accounts = defaultdict(list)
    nonassci, beyondlength, total = 0, 0, 0
    for emailpw in tqdm(emailpws):
        if '@' in emailpw and ':' in emailpw and emailpw.index('@') < emailpw.index(':') and not (' ' in emailpw):
            total += 1
            if emailpw.isascii():
                email, password = emailpw.split(":", 1)
                password = password.strip()
                if len(password) > 30 or len(password) < 4:
                    beyondlength += 1
                    continue
                accounts[tuple([email])].append(password)
            else:
                nonassci += 1
    with open(write_path, 'wb') as f:
        pickle.dump(accounts, f)
        f.flush()
    return total, nonassci, beyondlength

def file2groups_emailbased():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailpws_files/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailbased_groups/"
    paths = glob(read_dir + '*.txt.gz')
    workers = []
    pool = Pool(4)
    total_num, nonassci, beyondlength = 0, 0, 0
    while len(paths) > 0:
        if len(workers) < 4:
            path = paths.pop(0)
            workers.append(pool.apply_async(group_by_email, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl',)))
        if len(workers) == 4 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                re = w.get()
                total_num += re[0]
                nonassci += re[1]
                beyondlength += re[2]
            pool = Pool(4)
            workers = []
    print('total', total_num, 'nonassci', nonassci, 'beyondlength', beyondlength)

def group_usernamebased(readpath, writepath):
    # email consists of 'username@domain'
    # given path to pkl file consisting of disctionary of email:passwords, which is already grouped based on email
    # write a new dictionary of username:passwords, which is further grouped based on username
    with open(readpath, 'rb') as f:
        dict_emailpws = pickle.load(f)
    dict_usernamepws = defaultdict(list)
    for email, passwords in tqdm(list(dict_emailpws.items())):
        username = email[0].split('@')[0]
        dict_usernamepws[username].extend(passwords)
    with open(writepath, 'wb') as f:
        pickle.dump(dict_usernamepws, f)
        f.flush()

def group_mixed(readpath, writepath):
    # gourp by username, but if the passwords from different email of the same username do not share at least a pair of similar passwords, then do not group them
    with open(readpath, 'rb') as f:
        dict_emailpws = pickle.load(f)
    dict_mixed = defaultdict(list)
    dict_usernamebased = defaultdict(list) # record the username based group and each pw2groupid
    username2times = defaultdict(int) # record the number of usernames of the same but do not be grouped
    emailbased_vaults = len(dict_emailpws)
    for email, passwords in tqdm(list(dict_emailpws.items())):
        username = email[0].split('@')[0]
        if username == 'info':
            continue
        pws_ = list(set(passwords))
        newflag = True # indicating whether create a new username based pws
        if username2times[username] > 0:
            target_pws = dict_usernamebased[username][0]
            for pw in pws_:
                if pw in target_pws: # find the similar group
                    groupid = dict_usernamebased[username][1][target_pws.index(pw)]
                    #groupid = dict_usernamebased[username][1][random.choice(list(set([i for i, x in enumerate(target_pws) if x == pw])))]
                    dict_mixed[tuple([username, groupid])].extend(passwords)
                    dict_usernamebased[username][0].extend(pws_)
                    dict_usernamebased[username][1].extend([groupid] * len(pws_))
                    newflag = False
                    break
        if newflag:
            dict_mixed[tuple([username, username2times[username]])].extend(passwords)
            if len(dict_usernamebased[username]) == 2:
                dict_usernamebased[username][0].extend(pws_)
                dict_usernamebased[username][1].extend([username2times[username]] * len(pws_))
            else:
                dict_usernamebased[username].append(pws_)
                dict_usernamebased[username].append([username2times[username]] * len(pws_))
            username2times[username] += 1
    with open(writepath, 'wb') as f:
        pickle.dump(dict_mixed, f)
        f.flush()
    return emailbased_vaults, len(dict_mixed)

def email2usernamebased_or_mixed():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailbased_groups/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups/"
    paths = glob(read_dir + '*.pkl')
    workers = []
    pool = Pool(1)
    emailbased_num, mixbased_num = 0, 0
    while len(paths) > 0:
        if len(workers) < 1:
            path = paths.pop(0)
            workers.append(pool.apply_async(group_mixed, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl',)))
        if len(workers) == 1 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                re = w.get()
                emailbased_num += re[0]
                mixbased_num += re[1]
            pool = Pool(1)
            workers = []
    print('emailbased_num', emailbased_num, 'mixbased_num', mixbased_num)

def anaylsis_eachshot(path):
    # path to pkl file, which is a dictionary of email:passwords
    # return a list of 3 elements, [number of emails with 2 passwords,
    #                                  number of emails with 3 passwords,
    #                                  number of emails with >=4 passwords]
    with open(path, 'rb') as f:
        dict_emailpws = pickle.load(f)
    stat_dict = defaultdict(int)
    for email, passwords in tqdm(dict_emailpws.items()):
        stat_dict[len(passwords)] += 1
    return stat_dict

def analysis_vaults():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_g2_distinct/"
    paths = glob(read_dir + '*.pkl')
    workers = []
    stat = defaultdict(int)
    max_len_considered = 200
    outside_max = 0
    pool = Pool(19)
    while len(paths) > 0:
        if len(workers) < 19:
            path = paths.pop(0)
            workers.append(pool.apply_async(anaylsis_eachshot, (path,)))
        if len(workers) == 19 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                stat_thread = w.get()
                for k, v in stat_thread.items():
                    if k <= max_len_considered:
                        stat[k] += v
                    else:
                        stat[k] += v
                        outside_max += v
            pool = Pool(19)
            workers = []
    print("ouside max:", outside_max)
    print('max size', max(stat.keys()))
    # save stat to csv
    results = np.zeros((max_len_considered - 1, 2))
    for i in range(2, max_len_considered + 1):
        results[i - 2][0] = i
        results[i - 2][1] = stat[i]
    np.savetxt("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups.csv", results, delimiter=",")


def duplicate_check():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_g2/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_g2_distinct/"
    paths = glob(read_dir + '*.pkl')
    workers = []
    duplicate_num = 0
    pool = Pool(10)
    while len(paths) > 0:
        if len(workers) < 10:
            path = paths.pop(0)
            workers.append(pool.apply_async(duplicate_check_thread, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl')))
        if len(workers) == 10 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                duplicate_num += w.get()
            pool = Pool(10)
            workers = []
    print('duplicate vaults:', duplicate_num)

def duplicate_check_thread(readpath, writepath):
    with open(readpath, 'rb') as f:
        dict_emailpws = pickle.load(f)
    new_dict = {}
    vaults2num = defaultdict(int)
    pw2id = defaultdict(int)
    duplicate_num = 0
    for email, passwords in tqdm(dict_emailpws.items()):
        for pw in passwords:
            if pw2id[pw] == 0:
                pw2id[pw] = len(pw2id) + 1
        passwords_id = [pw2id[pw] for pw in passwords]
        assert 0 not in passwords_id
        if vaults2num[tuple(sorted(passwords_id))] > 0:
            duplicate_num += 1
        else:
            vaults2num[tuple(sorted(passwords_id))] += 1
            new_dict[email] = passwords
    with open(writepath, 'wb') as f:
        pickle.dump(new_dict, f)
        f.flush()
    return duplicate_num

def get_testset_step1(readpath, writepath):
    # step1 of getting the password vault dataset is to retain the accounts with more than 9 passwords
    # readpath: path to pkl file, which is a dictionary of username:passwords, each item is viewed as an account (of password vault)
    # return 1. the accounts with more than 4 passwords in dictionary format; 2. total number of accounts in the path
    with open(readpath, 'rb') as f:
        dict_usernamepws = pickle.load(f)
    total = len(dict_usernamepws)
    dict2write = {}
    # remove the accounts with less than or equal to 4 passwords
    for username, passwords in tqdm(list(dict_usernamepws.items())):
        if len(passwords) <= 50 and len(passwords) >= 2: #
            dict2write[username] = passwords

    with open(writepath, 'wb') as f:
        pickle.dump(dict2write, f)
        f.flush()
    return len(dict2write), total


def get_testset_step2(readpath, writepath):
    # step2 of getting the password vault dataset is to retain the accounts with at least a pair of similar passwords, either can be duplicated or edit distance <= 1
    # write the accounts with at least a pair of similar passwords to a new file
    deleted = 0
    with open(readpath, 'rb') as f:
        dict_usernamepws = pickle.load(f)
    itm_lst = list(dict_usernamepws.items())
    for username, passwords in tqdm(itm_lst):
        if len(passwords) >= 20 and len(passwords) / len(set(passwords)) < 1.84:
            del dict_usernamepws[username]
            deleted += 1
    '''with open(writepath, 'wb') as f:
        pickle.dump(dict_usernamepws, f)
        f.flush()'''
    return len(dict_usernamepws), deleted


def turn2testset():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_51_200/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_51_200_g184/"
    paths = glob(read_dir + '*.pkl')
    testset_size = 0
    vault_total  = 0
    workers = []
    pool = Pool(3)
    while len(paths) > 0:
        if len(workers) < 3:
            path = paths.pop(0)
            workers.append(pool.apply_async(get_testset_step2, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl')))
        if len(workers) == 3 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                re = w.get()
                testset_size += re[0]
                vault_total += re[1]
            pool = Pool(3)
            workers = []
    print('retaining', testset_size, 'vaults deleted,', vault_total)


def statistic_shot(vault, i, total):
    # a vault (list of pws)
    # return a statistic list for vault, [vaultsize (number of pws), number of distinct pws, reuse rate, modification rate]
    print(str(i / total * 100) + '%')

    vs = len(vault)
    distinct_vs = len(set(vault))
    return [vs, distinct_vs, vs / len(find_clusters(vault, 0)[0]), np.array([len(pw) for pw in vault]).mean()]


def testset_statistics():
    read_path = '/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_200_tr.json'#"/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/processed_dataset/Dbc_ts.pkl"
    '''with open(read_path, 'rb') as f:
        vaultdict = pickle.load(f)'''
    with open(read_path, 'r') as f:
        vaultdict = json.load(f)
    total_vaults = len(vaultdict)
    print('vaults in total:', total_vaults)
    stat = []
    workers = []
    pool = Pool(31)
    for i, vault in enumerate(list(vaultdict.values())):
        workers.append(pool.apply_async(statistic_shot, (vault, i, total_vaults)))
    pool.close()
    pool.join()
    for w in workers:
        stat.append(w.get())
    # write to file
    write_path = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_ts_stat.pkl"
    with open(write_path, 'wb') as f:
        pickle.dump(stat, f)

def keepvaultsize():
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset.pkl", 'rb') as f:
        vaultdict = pickle.load(f)
    itms = list(vaultdict.items())
    for k, v in itms:
        if len(v) <= 50 or len(v) > 200 or len(set(v)) == 1:
            del vaultdict[k]
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset_g50.pkl", 'wb') as f:
        pickle.dump(vaultdict, f)

def rrv(k, v, progressbar):
    print(progressbar)
    rr = len(v) / len(set(v))
    mr = len(v) / len(find_clusters(v, 0)[0])
    if rr > 6 and mr > 8.3:
        return 0, k, v
    else:
        return 1, k, v

def split_highlow_reuse():
    pool = Pool(30)
    workers = []
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset_g50.pkl", 'rb') as f:
        vaultdict = pickle.load(f)
    print('vaults in total:', len(vaultdict))
    itms = list(vaultdict.items())
    highreuse = {}
    lowreuse = {}
    for i, (k, v) in enumerate(itms):
        workers.append(pool.apply_async(rrv, (k, v, (i+1) / len(itms) * 100)))
    pool.close()
    pool.join()
    for w in workers:
        flag, k, v = w.get()
        if len(set(v)) == 1:
            continue
        if flag == 0:
            highreuse[k] = v
        else:
            lowreuse[k] = v
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset_g50_highreuse.pkl", 'wb') as f:
        pickle.dump(highreuse, f)
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset_g50_lowreuse.pkl", 'wb') as f:
        pickle.dump(lowreuse, f)

def split2train_test(datasetpath):
    # 50-50 split
    with open(datasetpath, 'r') as f:
        vaultdict = json.load(f)
    todelete_keys = []
    for k, v in tqdm(vaultdict.items()):
        for pw in v:
            if re.search(r'\\x', pw) or '\x7f' in pw or '\x03' in pw or '\x00' in pw or '\x01' in pw or '\x02' in pw or '\x04' in pw or '\x05' in pw or '\x06' in pw or '\x07' in pw or '\x08' in pw or '\x0b' in pw or '\x0c' in pw or '\x0e' in pw or '\x0f' in pw or '\x10' in pw or '\x11' in pw or '\x12' in pw or '\x13' in pw or '\x14' in pw or '\x15' in pw or '\x16' in pw or '\x17' in pw or '\x18' in pw or '\x19' in pw or '\x1a' in pw or '\x1b' in pw or '\x1c' in pw or '\x1d' in pw or '\x1e' in pw or '\x1f' in pw or '\t' in pw:
                print(k, v)
                todelete_keys.append(k)
                break
    for k in todelete_keys:
        del vaultdict[k]
    print('delete', len(todelete_keys), 'vaults with illegal chars')
    '''with open(datasetpath, 'w') as f:
        json.dump(vaultdict, f)'''

    print('vaults in total:', len(vaultdict))

    itms = list(vaultdict.items())
    random.Random(0).shuffle(itms)

    testdict = dict(itms[:int(len(itms) / 2)])
    traindict = dict(itms[int(len(itms) / 2):])
    # shift the key to idx of str starting from 0 to len(vaultdict)
    testdict = dict([(str(int(91566159*3) + i), v) for i, v in enumerate(list(testdict.values()))])
    traindict = dict([(str(int(91566159*3) + i + len(testdict)), v) for i, v in enumerate(list(traindict.values()))])
    # write to json file
    with open(datasetpath[:-5] + '_tr.json', 'w') as f:
        json.dump(traindict, f)
    with open(datasetpath[:-5] + '_ts.json', 'w') as f:
        json.dump(testdict, f)

def pkls2json_51_200(readdir, writepath):
    def clean_vaultsize58(vaults):
        #vaults = list(vaults.values())
        v1 = list(vaults.values())[-1]
        assert 'VQsaBLPzLa' in v1 and 'DIOSESFIEL' in v1
        similar_fraction, todelete_keys = [], []
        for vid, v in vaults.items():
            if 'VQsaBLPzLa' not in v or 'DIOSESFIEL' not in v:
                continue
            # find how many passwords are identical
            v1_copy = copy.deepcopy(v1)
            identicals = 0
            for pw in v:
                if pw in v1_copy:
                    v1_copy.remove(pw)
                    identicals += 1
            if identicals == 56:
                todelete_keys.append(vid)
            similar_fraction.append(identicals)
        for i in range(59):
            print(58-i, 'identicals:', (np.array(similar_fraction) == (58-i)).sum())
        return todelete_keys

    pklfiles = os.listdir(readdir)
    totalvault = {}
    for i, pklfile in enumerate(pklfiles):
        print(i)
        with open(os.path.join(readdir, pklfile), 'rb') as f:
            totalvault.update(pickle.load(f))

    vaults_size58 = {}
    for k, v in totalvault.items():
        if len(v) == 58:
            vaults_size58[k] = v
    todelete_keys = clean_vaultsize58(vaults_size58)
    for todk in todelete_keys:
        del totalvault[todk]

    vaults2num = defaultdict(int)
    pw2id = defaultdict(int)
    duplicate_num, todelete_keys = 0, []
    for email, passwords in tqdm(totalvault.items()):
        for pw in passwords:
            if pw2id[pw] == 0:
                pw2id[pw] = len(pw2id) + 1
        passwords_id = [pw2id[pw] for pw in passwords]
        assert 0 not in passwords_id
        if vaults2num[tuple(sorted(passwords_id))] > 0:
            duplicate_num += 1
            todelete_keys.append(email)
        else:
            vaults2num[tuple(sorted(passwords_id))] += 1
    print('duplicate vaults:', duplicate_num)
    for todk in todelete_keys:
        del totalvault[todk]

    todelete_keys = []
    for k, v in totalvault.items():
        for pw in v:
            if re.search(r'\\x', pw) or '\x7f' in pw or '\x03' in pw or '\x00' in pw or '\x01' in pw or '\x02' in pw or '\x04' in pw or '\x05' in pw or '\x06' in pw or '\x07' in pw or '\x08' in pw or '\x0b' in pw or '\x0c' in pw or '\x0e' in pw or '\x0f' in pw or '\x10' in pw or '\x11' in pw or '\x12' in pw or '\x13' in pw or '\x14' in pw or '\x15' in pw or '\x16' in pw or '\x17' in pw or '\x18' in pw or '\x19' in pw or '\x1a' in pw or '\x1b' in pw or '\x1c' in pw or '\x1d' in pw or '\x1e' in pw or '\x1f' in pw or '\t' in pw:
                print(k, v)
                todelete_keys.append(k)
                break
    for k in todelete_keys:
        del totalvault[k]
    print('delete', len(todelete_keys), 'vaults with illegal chars:')

    totalvault_keystring = {}
    for n, (k, v) in enumerate(totalvault.items()):
        totalvault_keystring[str(n)] = v

    with open(writepath, 'w') as f:
        json.dump(totalvault_keystring, f)

def get_vaults(i, readpath, writepath, pklfile, fraction_):
    # step1 of getting the password vault dataset is to retain the accounts with more than 9 passwords
    # readpath: path to pkl file, which is a dictionary of username:passwords, each item is viewed as an account (of password vault)
    # return 1. the accounts with more than 4 passwords in dictionary format; 2. total number of accounts in the path
    print(i)
    gc.disable()
    with open(readpath, 'rb') as f:
        dict_usernamepws = pickle.load(f)
    gc.enable()
    dict2write = {}
    deleted = 0

    vaults2num = defaultdict(int)
    pw2id = defaultdict(int)
    duplicate_num, todelete_keys = 0, []
    for email, passwords in tqdm(dict_usernamepws.items()):
        for pw in passwords:
            if pw2id[pw] == 0:
                pw2id[pw] = len(pw2id) + 1
        passwords_id = [pw2id[pw] for pw in passwords]
        assert 0 not in passwords_id
        if vaults2num[tuple(sorted(passwords_id))] > 0:
            duplicate_num += 1
            todelete_keys.append(email)
        else:
            vaults2num[tuple(sorted(passwords_id))] += 1
    for todk in todelete_keys:
        deleted += 1
        del dict_usernamepws[todk]

    for username, passwords in tqdm(list(dict_usernamepws.items())):
        if random.random() < fraction_:
            illegal_char = False
            for pw in passwords:
                if re.search(r'\\x', pw) or '\x7f' in pw or '\x03' in pw or '\x00' in pw or '\x01' in pw or '\x02' in pw or '\x04' in pw or '\x05' in pw or '\x06' in pw or '\x07' in pw or '\x08' in pw or '\x0b' in pw or '\x0c' in pw or '\x0e' in pw or '\x0f' in pw or '\x10' in pw or '\x11' in pw or '\x12' in pw or '\x13' in pw or '\x14' in pw or '\x15' in pw or '\x16' in pw or '\x17' in pw or '\x18' in pw or '\x19' in pw or '\x1a' in pw or '\x1b' in pw or '\x1c' in pw or '\x1d' in pw or '\x1e' in pw or '\x1f' in pw or '\t' in pw:
                    illegal_char = True
                    deleted += 1
                    break
            if not illegal_char:
                dict2write[username] = passwords
    print('deleted:', deleted)

    # split train and test by half
    itms = list(dict2write.items())
    random.Random().shuffle(itms)
    testdict = dict(itms[:int(len(itms) / 2)])
    traindict = dict(itms[int(len(itms) / 2):])
    '''with open(os.path.join(writepath+'_train', pklfile), 'wb') as f:
        pickle.dump(traindict, f)
        f.flush()
    with open(os.path.join(writepath+'_test', pklfile), 'wb') as f:
        pickle.dump(testdict, f)
        f.flush()'''

def pkls2json_2_50(readdir, writepath):
    pklfiles = os.listdir(readdir)
    fraction_ = 1
    illegal_deleted = 0
    totalvault = {}
    pool = Pool(1)
    workers = []
    for i, pklfile in enumerate(pklfiles):
        workers.append(pool.apply_async(get_vaults, (i, os.path.join(readdir, pklfile), writepath, pklfile, fraction_,)))
    pool.close()
    pool.join()

def pickles2dict(pickledir, writepath, id_offset=0):
    # load each pickle file (dictionary) and return a whole dictionary
    files = os.listdir(pickledir)
    result = {}
    for file in tqdm(files):
        gc.disable()
        with open(os.path.join(pickledir, file), 'rb') as f:
            dict_ = pickle.load(f)
        gc.enable()
        for k, v in dict_.items():
            result[str(len(result) + id_offset)] = v
    print(len(result))
    with open(writepath, 'w') as f:
        json.dump(result, f)


def main():
    #dir2file()
    #file2groups_emailbased()
    #analysis_vaults()
    #email2usernamebased_or_mixed()
    turn2testset()
    duplicate_check()

    keepvaultsize()
    split_highlow_reuse()
    split2train_test('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_51_200.json')
    pkls2json_51_200(readdir='/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_51_200_g184', writepath='/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_51_200.json')
    pkls2json_2_50(readdir='/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_50_g184', writepath='/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_50_final')


if __name__ == '__main__':
    #main()
    #pickles2dict('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_50_final_train', '/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_50_tr.json', id_offset=91566159)

    '''vaultdict = {}
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2_200_tr.json", 'rb') as f:
        vaultdict.update(orjson.loads(f.read()))
    dataset_statistics = defaultdict(list)  # key: vault size; value: [[np. of unique passwords, average pw length, no. of clusters, cmr],..., []]
    vaultstotal = len(vaultdict)

    print('vaults in total:', vaultstotal)
    pool = Pool(processes=32)
    workers = []
    i = 0
    for k, v in vaultdict.items():
        workers.append(pool.apply_async(cluster_modification, (v, i, k)))
        i = i + 1
        if len(workers) == vaultstotal // 20 or i == vaultstotal:
            pool.close()
            pool.join()
            for w in workers:
                cmr, pws, avgpw_len, nm_clsters = w.get()
                if dataset_statistics[len(pws)] == []:
                    dataset_statistics[len(pws)] = [np.array([len(set(pws)), avgpw_len, nm_clsters, cmr]), 1]
                else:
                    dataset_statistics[len(pws)][0] += np.array([len(set(pws)), avgpw_len, nm_clsters, cmr])
                    dataset_statistics[len(pws)][1] += 1
            workers = []
            pool = Pool(processes=32)

    # write to json
    with open('bc_tr_stat.pkl', 'wb') as f:
        pickle.dump(dataset_statistics, f)'''

    with open('pb_stat.json', 'r') as f:
        data_statistics_pb = json.load(f)
    with open('bc_tr_stat.pkl', 'rb') as f:
        data_statistics_bc_tr = pickle.load(f)
    with open('bc_ts_stat.pkl', 'rb') as f:
        data_statistics_bc_ts = pickle.load(f)
    data_statistics_bc = {}
    for k in range(2,201):
        ts_keys, tr_keys = list(data_statistics_bc_ts.keys()), list(data_statistics_bc_tr.keys())
        if k in ts_keys and k in tr_keys:
            data_statistics_bc[k] = [np.array(data_statistics_bc_tr[k][0]) + np.array(data_statistics_bc_ts[k][0]), data_statistics_bc_tr[k][1] + data_statistics_bc_ts[k][1]]
        elif k in ts_keys:
            data_statistics_bc[k] = [np.array(data_statistics_bc_ts[k][0]), data_statistics_bc_ts[k][1]]
        elif k in tr_keys:
            data_statistics_bc[k] = [np.array(data_statistics_bc_tr[k][0]), data_statistics_bc_tr[k][1]]

    vaultsize_sep = [10, 200]  # [51, 61, 91] #
    final_stat_pb, final_stat_bc = defaultdict(list), defaultdict(list)
    for k,v in data_statistics_pb.items():
        for sep_value in vaultsize_sep:
            if int(k) <= sep_value:
                final_stat_pb[sep_value].extend(v)
                final_stat_pb['vaultsizes'].extend([int(k)] * len(v))
                break
    for k,v in data_statistics_bc.items():
        for sep_value in vaultsize_sep:
            if k <= sep_value:
                if len(final_stat_bc[sep_value]) == 0:
                    final_stat_bc[sep_value] = v
                else:
                    final_stat_bc[sep_value][0] += v[0]
                    final_stat_bc[sep_value][1] += v[1]
                final_stat_bc['vaultsizes'].extend([k] * v[1])
                break
    final_stat_bc = final_stat_bc
    '''groupsize_stat = defaultdict(list)
    for k, v in data_statistics.items():
        idx = [i for i, vs in enumerate(vaultsize_sep) if int(k) >= vs][-1]
        groupsize_stat[vaultsize_sep[idx]].extend(v)
    groupsize_stat = groupsize_stat'''