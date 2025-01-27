from multiprocessing import Pool
from glob import glob
import mgzip
import pickle
import numpy as np
from tqdm import tqdm
import json
from scipy.optimize import curve_fit
from collections import defaultdict
import pylcs
import random
import csv
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
        return 0  # Handle empty or single element list case

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
    return cmr, k, round(np.array([len(pw_) for pw_ in pws]).mean(), 4)#, len(clst) #len(clst)# / len(clst)

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

    '''for emailpw in tqdm(emailpws):
        if '@' in emailpw and ':' in emailpw and emailpw.index('@') < emailpw.index(':') and not (' ' in emailpw) and emailpw.isascii():
            email, password = emailpw.split(":", 1)
            password = password.strip()
            if len(password) > 30 or len(password) < 6:
                continue
            accounts[tuple([email])].append(password)'''
    with open(write_path, 'rb') as f:
        accounts = pickle.load(f)
    return len(emailpws), sum([len(pws) for pws in accounts.values()])

def file2groups_emailbased():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailpws_files/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailbased_groups/"
    paths = glob(read_dir + '*.txt.gz')
    workers = []
    threads = 2
    pool = Pool(threads)
    beforeclean, afterclean = 0, 0
    while len(paths) > 0:
        print(len(paths))
        if len(workers) < threads:
            path = paths.pop(0)
            workers.append(pool.apply_async(group_by_email, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl',)))
        if len(workers) == threads or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                beforeclean += w.get()[0]
                afterclean += w.get()[1]
            pool = Pool(threads)
            workers = []
    print('beforeclean:', beforeclean, 'afterclean:', afterclean)

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
    '''with open(readpath, 'rb') as f:
        dict_emailpws = pickle.load(f)
    dict_mixed = defaultdict(list)
    dict_usernamebased = defaultdict(list) # record the username based group and each pw2groupid
    username2times = defaultdict(int) # record the number of usernames of the same but do not be grouped
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
            username2times[username] += 1'''
    with open(writepath, 'rb') as f:
        dict_mixed = pickle.load(f)
    return [len(pws) for pws in dict_mixed.values()]

def email2usernamebased_or_mixed():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/emailbased_groups/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups/"
    paths = glob(read_dir + '*.pkl')
    workers = []
    stat = []
    pool = Pool(5)
    while len(paths) > 0:
        print(len(paths))
        if len(workers) < 5:
            path = paths.pop(0)
            workers.append(pool.apply_async(group_mixed, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl',)))
        if len(workers) == 5 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                stat.extend(w.get())
            pool = Pool(5)
            workers = []
    # write stat to csv
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_stat.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(stat)

def anaylsis_eachshot(path):
    # path to pkl file, which is a dictionary of email:passwords
    # return a list of 3 elements, [number of emails with 2 passwords,
    #                                  number of emails with 3 passwords,
    #                                  number of emails with >=4 passwords]
    with open(path, 'rb') as f:
        dict_emailpws = pickle.load(f)
    res = [0, 0, 0]
    for email, passwords in tqdm(dict_emailpws.items()):
        if len(passwords) == 2:
            res[0] += 1
        elif len(passwords) == 3:
            res[1] += 1
        elif len(passwords) >= 4:
            res[2] += 1
    return np.array(res)

def analysis_vaults():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/usernamebased_groups/"
    paths = glob(read_dir + '*.pkl')
    workers = []
    stat = np.zeros(3)
    pool = Pool(8)
    while len(paths) > 0:
        if len(workers) < 8:
            path = paths.pop(0)
            workers.append(pool.apply_async(anaylsis_eachshot, (path,)))
        if len(workers) == 8 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                stat += w.get()
            pool = Pool(8)
            workers = []
    print(stat[0]/stat.sum(), stat[1]/stat.sum(), stat[2]/stat.sum(), stat.sum())


def get_testset_step1(readpath, writepath):
    # step1 of getting the password vault dataset is to retain the accounts with more than 9 passwords
    # readpath: path to pkl file, which is a dictionary of username:passwords, each item is viewed as an account (of password vault)
    # return 1. the accounts with more than 4 passwords in dictionary format; 2. total number of accounts in the path
    with open(readpath, 'rb') as f:
        dict_usernamepws = pickle.load(f)
    total = len(dict_usernamepws)
    vs2num = defaultdict(int)
    dict2write = {}
    # remove the accounts with less than or equal to 4 passwords
    for username, passwords in tqdm(list(dict_usernamepws.items())):
        if len(passwords) <= 200 and len(passwords) >= 51:
            vs2num[len(passwords)] += 1
            dict2write[username] = passwords
    return dict2write, total


def get_testset_step2(readpath, writepath):
    # step2 of getting the password vault dataset is to retain the accounts with at least a pair of similar passwords, either can be duplicated or edit distance <= 1
    # write the accounts with at least a pair of similar passwords to a new file
    with open(readpath, 'rb') as f:
        dict_usernamepws = pickle.load(f)
    for username, passwords in tqdm(list(dict_usernamepws.items())):
        if len(passwords) / len(set(passwords)) < 1.84:
            del dict_usernamepws[username]
    with open(writepath, 'wb') as f:
        pickle.dump(dict_usernamepws, f)
        f.flush()
    return len(dict_usernamepws)


def turn2testset():
    read_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_g9/"
    write_dir = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2to50/"
    paths = glob(read_dir + '*.pkl')
    testset_size = 0
    vault_total  = 0
    workers = []
    dict_total = {}
    pool = Pool(10)
    while len(paths) > 0:
        print(len(paths))
        if len(workers) < 10:
            path = paths.pop(0)
            workers.append(pool.apply_async(get_testset_step1, (path, write_dir + path.split('/')[-1].split('.')[0] + '.pkl')))
        if len(workers) == 10 or len(paths) == 0:
            pool.close()
            pool.join()
            for w in workers:
                re = w.get()
                dict_total.update(re[0])
                vault_total += re[1]
            pool = Pool(10)
            workers = []
    print(vault_total, 'vaults in total.')
    # write dict_total to a new file
    with open('testset51_200.pkl', 'wb') as f:
        pickle.dump(dict_total, f)


def statistic_shot(vault, i, total):
    # a vault (list of pws)
    # return a statistic list for vault, [vaultsize (number of pws), number of distinct pws, reuse rate, modification rate]
    print(i, '/', total)

    vs = len(vault)
    distinct_vs = len(set(vault))
    return [vs, distinct_vs, vs / distinct_vs, vs / len(find_clusters(vault, 0)[0]), np.array([len(pw) for pw in vault]).mean()]


def testset_statistics():
    read_path = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2/fold_0_2.json'#"/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/processed_dataset/Dbc_ts.pkl"
    '''with open(read_path, 'rb') as f:
        vaultdict = pickle.load(f)'''
    with open(read_path, 'r') as f:
        vaultdict = json.load(f)
    total_vaults = len(vaultdict)
    print('vaults in total:', total_vaults)
    stat = []
    workers = []
    pool = Pool(32)
    for i, vault in enumerate(list(vaultdict.values())):
        workers.append(pool.apply_async(statistic_shot, (vault, i, total_vaults)))
    pool.close()
    pool.join()
    for w in workers:
        stat.append(w.get())
    # write to file
    write_path = "/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset_stat.pkl"
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

def split2train_test():
    # 50-50 split
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/bc_testset_g50.pkl", 'rb') as f:
        vaultdict = pickle.load(f)
    print('vaults in total:', len(vaultdict))
    itms = list(vaultdict.items())
    random.Random(0).shuffle(itms)

    # remove the redundant vaults
    redundantv2times = defaultdict(int)
    topop = []
    for i, (k, v) in enumerate(itms):
        if tuple(v) in (redundantv2times.keys()) or v in [v_ for k_, v_ in itms[:i]]:
            redundantv2times[tuple(v)] += 1
            topop.append((k, v))
    print(redundantv2times)
    for k, v in topop:
        itms.remove((k, v))
    traindict = dict(itms[:int(len(itms)/2)])
    testdict = dict(itms[int(len(itms)/2):])

    # shift the key to idx of str starting from 0 to len(vaultdict)
    testdict = dict([(str(i), v) for i, v in enumerate(list(testdict.values()))])
    traindict = dict([(str(i + len(testdict)), v) for i, v in enumerate(list(traindict.values()))])
    # write to json file
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/Dbc_tr.json", 'w') as f:
        json.dump(traindict, f)
    with open("/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/Dbc_ts.json", 'w') as f:
        json.dump(testdict, f)

def pkls2json(readdir, writepath):
    pklfiles = os.listdir(readdir)
    totalvault = {}
    for pklfile in pklfiles:
        with open(os.path.join(readdir, pklfile), 'rb') as f:
            totalvault.update(pickle.load(f))
    totalvault_keystring = {}
    for n, (k, v) in enumerate(totalvault.items()):
        totalvault_keystring['00' + str(n)] = v
    with open(writepath, 'w') as f:
        json.dump(totalvault_keystring, f)

def get_stat_range(dic, pins):
    # dic: k is vault size that is threw in the busket of range determined by pins
    # pins: a value list, each value is the upper bound of the range

    group_dic = defaultdict(list)
    for k, v in dic.items():
        for pin in pins:
            if k < pin:
                group_dic[pins.index(pin)].extend(v)
                break
    return group_dic

def get_uniquepws(vault, process):
    # vault: list of pws
    # return: number of unique pws
    print(process)
    return len(vault), len(set(vault))


def main():
    #dir2file()
    #file2groups_emailbased()
    #analysis_vaults()
    #email2usernamebased_or_mixed()
    turn2testset()

    #testset_statistics()
    #keepvaultsize()
    #split_highlow_reuse()
    #split2train_test()
    #pkls2json(readdir='/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2to50', writepath='/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/mixed_groups_2to50.json')


if __name__ == '__main__':
    #main()

    with open('testset51_200.pkl', 'rb') as f:
        vaultdict = pickle.load(f)
    vaultdict = vaultdict

    '''with open('/home/beeno/Dropbox/research_project/pycharm/dataset/BreachCompilation/fold_0_1.json', 'r') as f:
        vaultdict = json.load(f)
    with open('/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/breachcompilation/fold2/fold_0_2.json', 'r') as f:
        vaultdict.update(json.load(f))'''
    '''vaultdict = {}
    for i in range(1, 6):
        with open("/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/pastebin/fold5_Nge5/fold_" + str(i) + ".json", 'r') as f:
            vaultdict.update(json.load(f))

    print('vaults in total:', len(vaultdict))
    pool = Pool(processes=15)
    workers = []                   
    itms = list(vaultdict.items())
    for i in range(len(itms)):
        k, v = itms[i]
        workers.append(pool.apply_async(cluster_modification, (v, i, k)))
    pool.close()
    pool.join()
    dataset_statistics = defaultdict(list) # key: vault size; value: [[np. of unique passwords, average pw length, no. of clusters, cmr],..., []]
    # group vault into different size range, [50, 80); [80, 110), [110, 140); [140, 170); [170, 200)
    for w in workers:
        cmr, k, avgpw_len, nm_clsters = w.get()
        dataset_statistics[len(vaultdict[k])].append([len(set(vaultdict[k])), avgpw_len, cmr, nm_clsters])
    # write to json
    with open('pb_stat.json', 'w') as f:
        json.dump(dataset_statistics, f)'''

    '''vaultsize_sep = [5, 10, 30, 50] #[50, 80, 140, 201] #
    with open('pb_stat.json') as f:
        data_statistics = json.load(f)
    groupsize_stat = defaultdict(list)
    for k, v in data_statistics.items():
        idx = [i for i, vs in enumerate(vaultsize_sep) if int(k) >= vs][-1]
        groupsize_stat[vaultsize_sep[idx]].extend(v)
    groupsize_stat = groupsize_stat'''