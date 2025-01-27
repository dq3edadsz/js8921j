import random
from tqdm import tqdm
import numpy as np
import math
import pickle
import os
import gc
import json
import csv

def main():
    referrank = False # use rank results in acquiring according Me
    referpath = ['/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPMrank/pb/testset/attack_result_1000_testdataid0_cons1_pin3', '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPMrank/pb/testset/attack_result_1000_testdataid0_cons2_Nitv10_pin3'] #  #
    referfold = 4 #

    ## experiment settings
    tstset = 'bc200' # 'pb' for pastebin dataset or 'bc50', 'bc200' for breachcompilation dataset
    dte_model = 'MSPM' # 'Golla' or 'MSPM'
    chengorgolla = 0 # 0 for cheng; 1 for golla, which returns ranks and all results based on the rank method chosen
    n_vaulteachshot = 100 # 231 number of vaults in each shot
    online_thre = 100 # 100  online verification threshold for each website
    max_repeatn = 15 # repeat times for experiment
    ## attack settings
    intersec = True
    nol_ith = -1 # nol_exps=[5, N-1], [1,3,5,7,N-1]
    expermented_settings = [
                            #"fp cons1 t1000 3pin",
                            #"fp cons1 t4000 4pin",
                            #"fp cons1 t4000 5pin",
                            #"fp cons1 t4000 6pin",
                            #"fp cons1 t4000 6pin",

                            #"fp cons2 t1000 4pin Nitv2",
                            #"fp cons2 t1000 4pin Nitv4",
                            #"fp cons2 t1000 4pin Nitv6",
                            #"fp cons2 t1000 4pin Nitv8",
                            "fp cons2 t4000 6pin Nitv10",
                            #"fp cons2 t1000 4pin Nitv12",
                            #"fp cons2 t1000 4pin Nitv14",
                            #"fp cons2 t1000 4pin Nitv10",
                            ]


    root = '/home/beeno/'#'/Users/sunbin/Library/CloudStorage/'
    write_dir = root + 'Dropbox/encryption/writing/bubble/results_handling/table_data/'
    # override the files in write_dir (make the files to 0 bytes)
    for fname in os.listdir(write_dir):
        if '.csv' in fname and 'rrandmr' not in fname:
            with open(os.path.join(write_dir, fname), 'w') as f:
                pass

    for setting_idx, settings in enumerate(expermented_settings):
        '''t = '2000'
        pin = 'pin6'
        cons = 'cons2' # '1': vanilla shuffling, 2: fixed iterval shuffling
        Nitv = 'Nitv8' # fixed interval shuffling interval'''
        # extract settings
        t = '_'+settings.split(' ')[2][1:]+'_'
        pin = 'pin' + settings.split(' ')[3][0]
        cons = settings.split(' ')[1]
        Nitv = settings.split(' ')[4] if len(settings.split(' ')) == 5 else None

        results = []
        read_dir = root + 'Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/' + dte_model + '/' + tstset + '/mversion/notallleak/vg8'# '/testset' #'/hdd1/bubble_experiments/results/'
        # list all directories from read_dir
        read_dir = [d[0] for d in os.walk(read_dir) if t in d[0] and pin in d[0] and ((cons+'_'+Nitv if cons[-1] == '2' else cons) in d[0])]# and 'multidouble_vault' not in d[0]
        if len(read_dir) != 1:
            raise ValueError('read_dir is not unique or not exist!', read_dir)
        else:
            # get 'repeat_times' as lower bound of files//n_vaulteachshot
            repeat_times = min(max_repeatn, len(os.listdir(read_dir[0])) // n_vaulteachshot) #len(os.listdir(read_dir[0])) // n_vaulteachshot
            print('read_dir:', read_dir[0].split('/')[-1], 'repeat_times:', repeat_times)


        gc.disable()
        for shotid in tqdm(range(repeat_times)):#[0,1,3,4]: #
            for vid in range(n_vaulteachshot): #
                fname = 'results_v' + str(vid) + '_shot' + str(shotid) + '.data'
                with open(os.path.join(read_dir[0], fname), 'rb') as f:
                    re_ = pickle.load(f)
                    if referrank:
                        with open(os.path.join(referpath[setting_idx], fname), 'rb') as f:
                            re_referred = pickle.load(f)
                        re_[1][0][0][0]['r_three'] = re_[1][0][0][0]['r_three'][:4] + [referfold*r_ for r_ in re_referred[1][0][0][0]['r_three'][-3:]]
                        re_[1][0][0][1]['r_three'] = re_[1][0][0][1]['r_three'][:4] + [referfold*r_ for r_ in re_referred[1][0][0][1]['r_three'][-3:]]
                    results.append(re_)
        gc.enable()
        #print('fp but not real mpw', np.array([re_[1][0][0][0]['fp'] != re_[1][0][0][0]['fp_butrealmpw'] if re_[1][0][0][0]['Nol_exps'][-1] > 8 else False for re_ in results]).sum())
        print('proportion of vaults with ranks == 0:', np.array([min(re[1][0][0][0]['r_three'])==0 for re in results]).sum()/len(results))
        extract_rank(results, nol_ith=nol_ith, write_dir=write_dir, intersec=intersec, n_vaulteachshot=n_vaulteachshot, chengorgolla=chengorgolla)
        extract_login_distribution(results, nol_ith=nol_ith, write_dir=write_dir, intersec=intersec,  n_vaulteachshot=n_vaulteachshot, delta=online_thre, chengorgolla=chengorgolla)
        #boxplot(results, nol_ith=nol_ith, write_dir=write_dir, intersec=intersec, n_vaulteachshot=n_vaulteachshot, delta=online_thre, chengorgolla=chengorgolla)


def extract_rank(results, nol_ith, write_dir, intersec=False, n_vaulteachshot=75, chengorgolla=0):
    rank = [[] for _ in range(n_vaulteachshot)]
    T_expand = len(results[0][2][-1])
    print('T_expand:', T_expand)
    id_interornot = 0 if not intersec else 1
    for i, vidx in enumerate(range(n_vaulteachshot)):
        for shotid in range(len(results) // n_vaulteachshot):
            if isinstance(results[shotid * n_vaulteachshot + vidx][1][0][0], dict): # old type results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][id_interornot]
            else: # new type of results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][0][id_interornot]
            #assert r_dict['Nol_exps'][nol_ith]+1 == 60
            leakpw, leakid = results[shotid * n_vaulteachshot + vidx][2][0]
            denominator = sum([1 if (leakpw in v) else 0 for v in results[shotid * n_vaulteachshot + vidx][2][1]])
            rank[i].append(np.array(r_dict['r_three']) / denominator)
        rank[i] = np.stack(rank[i]).mean(0)

    #rank = list(np.array(rank).mean(1))# > sorted(list(np.array(rank).mean(1)))[int(n_vaulteachshot*0.20)]).astype(int)

    np.savetxt(write_dir + 'rank.csv', np.array(rank), delimiter=",")


def extract_login_distribution(results, nol_ith, write_dir, intersec=False, n_vaulteachshot=75, delta=10, chengorgolla=0):
    metaid_histog, Fp = [[] for _ in range(n_vaulteachshot)], [[] for _ in range(n_vaulteachshot)]
    id_interornot = 0 if not intersec else 1

    for i, vidx in enumerate(range(n_vaulteachshot)):
        for shotid in range(len(results) // n_vaulteachshot):
            if isinstance(results[shotid * n_vaulteachshot + vidx][1][0][0], dict):  # old type results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][id_interornot]
            else:  # new type of results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][0][id_interornot]
            vault_size = r_dict['Nol_exps'][-1] + 1

            loginsum = getme(r_dict, chengorgolla)

            metaid_histog[i].append(int((loginsum/(vault_size-1)) >= delta)*100) # 0
            Fp[i].append(r_dict['fp'][nol_ith])

        # random generate several results (need to be deleted)!
        '''original_len = len(Fp[i])
        for _ in range(5):
            ridx = random.randint(0, original_len-1)
            metaid_histog[i].append(metaid_histog[i][ridx])
            Fp[i].append(Fp[i][ridx])'''

    fail = (np.array(Fp) * (100-np.array(metaid_histog)) + np.array(metaid_histog))
    Fp = list((np.array(Fp) * 100).mean(1))
    metaid_histog = list(np.array(metaid_histog).mean(1))
    assert (fail.mean(1) <= 100).prod() == 1

    # read the .csv from file and append new data to another column and write back to file
    with open(write_dir + 'blocked' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and metaid_histog
    lines = [((lines[i].strip() + ',') if len(lines) != 0 else '') + str(metaid_histog[i]) + '\n' for i in range(len(metaid_histog))]
    # write back to file
    with open(write_dir + 'blocked' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

    with open(write_dir + 'Fp' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and Fp
    lines = [((lines[i].strip() + ',') if len(lines) != 0 else '') + str(Fp[i]) + '\n' for i in range(len(Fp))]
    # write back to file
    with open(write_dir + 'Fp' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

    with open(write_dir + 'fail' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and Fp
    lines = [((lines[i].strip() + ',') if len(lines) != 0 else '') + str(fail.mean(1)[i]) + '\n' for i in range(len(fail.mean(1)))]
    # write back to file
    with open(write_dir + 'fail' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

def boxplot(results, nol_ith, write_dir, intersec=False, n_vaulteachshot=75, delta=10, chengorgolla=0):
    metaid_histog, Fp = [[] for _ in range(n_vaulteachshot)], [[] for _ in range(n_vaulteachshot)]
    id_interornot = 0 if not intersec else 1

    for i, vidx in enumerate(range(n_vaulteachshot)):
        for shotid in range(len(results) // n_vaulteachshot):
            if isinstance(results[shotid * n_vaulteachshot + vidx][1][0][0], dict):  # old type results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][id_interornot]
            else:  # new type of results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][0][id_interornot]
            vault_size = r_dict['Nol_exps'][-1] + 1

            # sum the login (based on two parts, logins for the decoys before the real rank and the logins for the real)
            loginsum = getme(r_dict, chengorgolla)

            metaid_histog[i].append(int((loginsum/(vault_size-1)) >= delta)*100) # 0
            Fp[i].append(r_dict['fp'][nol_ith])

        # random generate several results (need to be deleted)!
        '''original_len = len(Fp[i])
        for _ in range(2):
            ridx = random.randint(0, original_len - 1)
            metaid_histog[i].append(metaid_histog[i][ridx])
            Fp[i].append(Fp[i][ridx])'''

    fail = (np.array(Fp) * (100-np.array(metaid_histog)) + np.array(metaid_histog))
    Fp = (np.array(Fp) * 100)
    metaid_histog = np.array(metaid_histog)
    assert (fail.mean(1) <= 100).prod() == 1

    with open(write_dir + 'fail_box' + '.csv', 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        prefix_ = []
    else:
        prefix_ = lines[0].strip().split(',')
    lines = prefix_ + [str(fail.mean(0)[i]) for i in range(len(fail.mean(0)))] + [''] * (20 - len(fail.mean(0)))
    # write back to file
    with open(write_dir + 'fail_box' + '.csv', 'w') as f:
        for ith, line in enumerate(lines):
            if ith == (len(lines)-1):
                f.write(line)
            else:
                f.write(line + ',')

    # read the .csv from file and append new data to another column and write back to file
    with open(write_dir + 'blocked_errorplot' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and metaid_histog
    lines += [str(metaid_histog.mean(0).mean()) + ',' + str(0) + '\n']
    # write back to file
    with open(write_dir + 'blocked_errorplot' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

    with open(write_dir + 'Fp_errorplot' + '.csv', 'r') as f:
        lines = f.readlines()
    lines += [str(Fp.mean(0).mean()) + ',' + str(0) + '\n']
    with open(write_dir + 'Fp_errorplot' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

    with open(write_dir + 'fail_errorplot' + '.csv', 'r') as f:
        lines = f.readlines()
    lines += [str(fail.mean(0).mean()) + ',' + str(0) + '\n']
    with open(write_dir + 'fail_errorplot' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

def errorplot(results, nol_ith, write_dir, intersec=False, n_vaulteachshot=75, delta=10, chengorgolla=0):
    metaid_histog, Fp = [[] for _ in range(n_vaulteachshot)], [[] for _ in range(n_vaulteachshot)]
    id_interornot = 0 if not intersec else 1

    for i, vidx in enumerate(range(n_vaulteachshot)):
        for shotid in range(len(results) // n_vaulteachshot):
            if isinstance(results[shotid * n_vaulteachshot + vidx][1][0][0], dict):  # old type results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][id_interornot]
            else:  # new type of results
                r_dict = results[shotid * n_vaulteachshot + vidx][1][0][0][id_interornot]
            vault_size = r_dict['Nol_exps'][-1] + 1

            # sum the login (based on two parts, logins for the decoys before the real rank and the logins for the real)
            loginsum, rank_ = 0, r_dict['r_three'][2 + chengorgolla]
            # part1 sum decoys before the real
            for nth_batch in r_dict['Me_eachbsz']:
                bsz_, me_bsz = nth_batch
                if rank_ >= bsz_:
                    loginsum += me_bsz.get()
                elif rank_ > 0:
                    loginsum += me_bsz.get() / bsz_ * rank_
                    break
                rank_ = rank_ - bsz_
            # part1 sum the real from the end
            total_num =  0
            for nth_batch in r_dict['Me_eachbsz'][::-1]:
                bsz_, me_bsz = nth_batch
                loginsum += me_bsz.get()
                total_num += bsz_
                if total_num >= 1:
                    break

            metaid_histog[i].append(int((loginsum/(vault_size-1)) >= delta)*100) # 0
            Fp[i].append(r_dict['fp'][nol_ith])
    fail = (np.array(Fp) * (100-np.array(metaid_histog)) + np.array(metaid_histog))
    Fp = (np.array(Fp) * 100)
    metaid_histog = np.array(metaid_histog)
    assert (fail.mean(1) <= 100).prod() == 1

    # read the .csv from file and append new data to another column and write back to file
    with open(write_dir + 'blocked' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and metaid_histog
    lines = [((lines[i].strip() + ',') if len(lines) != 0 else '') + str(metaid_histog.mean(0).mean()) + ',' + str(np.std(metaid_histog.mean(0))) + '\n']
    # write back to file
    with open(write_dir + 'blocked' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

    with open(write_dir + 'Fp' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and Fp
    lines = [((lines[i].strip() + ',') if len(lines) != 0 else '') + str(Fp.mean(1)[i]) + '\n' for i in range(len(Fp.mean(0)))]
    # write back to file
    with open(write_dir + 'Fp' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

    with open(write_dir + 'fail' + '.csv', 'r') as f:
        lines = f.readlines()
    # merge line and Fp
    lines = [((lines[i].strip() + ',') if len(lines) != 0 else '') + str(fail.mean(1)[i]) + '\n' for i in range(len(fail.mean(0)))]
    # write back to file
    with open(write_dir + 'fail' + '.csv', 'w') as f:
        for line in lines:
            f.write(line)

def getme(r_dict, chengorgolla):
    # sum the login (based on two parts, logins for the decoys before the real rank and the logins for the real)
    loginsum, rank_ = 0, r_dict['r_three'][2 + chengorgolla]
    # part1 sum decoys before the real
    if rank_ <= r_dict['r']:
        for nth_batch in r_dict['Me_eachbsz']:
            bsz_, me_bsz = nth_batch
            if rank_ >= bsz_:
                loginsum += me_bsz.get()
            elif rank_ > 0:
                loginsum += me_bsz.get() / bsz_ * rank_
                break
            rank_ = rank_ - bsz_
    else:
        total_num = 0
        for nth_batch in r_dict['Me_eachbsz'][::-1]:
            bsz_, me_bsz = nth_batch
            total_num += bsz_
            if total_num <= 1:
                continue
            loginsum += me_bsz.get()
        loginsum = loginsum * rank_ / (total_num - 1)

    # part1 sum the real from the end
    total_num = 0
    for nth_batch in r_dict['Me_eachbsz'][::-1]:
        bsz_, me_bsz = nth_batch
        loginsum += me_bsz.get()
        total_num += bsz_
        if total_num >= 1:
            break

    return loginsum


if __name__ == '__main__':
    main()