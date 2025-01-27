import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import time
import os
import random
import multiprocessing
from multiprocessing import Pool # from multiprocessing.pool import ThreadPool as Pool
import mgzip
from utils import merge_sublist, get_Mefp_candidatelist_para
from opts import opts
args = opts().parse()
from MSPM.mspm_config import *
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
fsize = 10
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

class Measure:
    def __init__(self):
        pass

    def extract_data(self, pnlst, tlst, num_processes, plot_intersec, expdata_path, repeatn=10,
                     destination = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/extracted_results/tmp/'):
        print('experiment settings:')
        print('using', num_processes, 'processes')
        print('plot_intersec: ', plot_intersec)
        print('read data from: ', expdata_path)
        print('number of files per setting: ', repeatn)


        if not os.path.exists(destination):
            os.makedirs(destination)
        filepath_list = list_files(expdata_path)
        # '''
        reads = time.time()

        for repeat_times in range(repeatn):
            destination_file = destination + 'repeat_' + str(repeat_times) + '.data'
            '''if os.path.exists(destination_file):
                print('file', destination_file, 'exists, skip')
                continue'''
            with open(destination_file, 'wb'):
                pass
            with open(destination_file, 'ab') as f:
                for pn in pnlst:
                    for pcor_idx, T in enumerate(tlst):  # , 1000, 2000
                        filepath = [path for path in filepath_list if 'pin4' in path and ('_'+str(T)+'_') in path and ('_'+str(repeat_times)+'_') in path]
                        filepath = filepath[0] if len(filepath) == 1 else None
                        # record the time finishing reading the file
                        time_read = time.time()
                        if filepath is not None:
                            print('reading file: ', filepath)
                            results = []
                            pool = Pool(num_processes[pcor_idx])
                            workers = []
                            ranks, ranks_soft, fps, fps_butrealmpw, vaults, h0s, h0s_intersc = [], [], [], [], [], [], []
                            with mgzip.open(filepath, 'rb') as f_:
                                while True:
                                    try:
                                        results.append(pickle.load(f_))
                                    except EOFError:
                                        break
                                # results = pickle.load(f)
                            # results = assemble_pinweight(results, pin_vector)
                            print('Finishing reading ' + str(T) + ' ' + pn + ', read file time: ', time.time() - time_read, 'repeat times: ', repeat_times+1)
                            for threaid in range(len(results[0][0][0])):
                                workers.append(pool.apply_async(self.rank_r, ([results[0][0][0][threaid]], [results[0][0][1][threaid]], 1, pn == '4', plot_intersec)))
                            pool.close()
                            pool.join()
                            for results_threa in workers:
                                rt = results_threa.get()
                                ranks.append(rt[0])
                                ranks_soft.extend(rt[1])
                                vaults.extend(rt[2])
                            ranks_0 = np.concatenate(ranks, axis=0)  # batch * attack_cls
                            ranks_1 = ranks_soft  # batch * attack_cls
                            pickle.dump([ranks_0, ranks_1, vaults], f)
            if repeat_times >= repeatn - 1:
                break

        # write results_ts to pickle file
        print('read and process time: ', time.time() - reads)

    def read_data(self, directory, ts):
        # 'ts' is the number of choices for T
        # read all files from directory and return variable results_ts
        results = []
        for file in os.listdir(directory):
            with open(directory + file, 'rb') as f:
                while True:
                    try:
                        results.append(pickle.load(f))  # list of results with length 150 * repeatn * len(tlst)
                    except EOFError:
                        break
        results_sorted = []
        for tn in range(ts):
            for i in range(len(results) // ts):
                results_sorted.append(results[ts * i + tn])
        return results_sorted


    def rank_r(self, results_threa, softs_threa=None, pn4=True, reshuedidx=None, gpuid=0, seed_shot=0):
        """

        :param results_avault_aiter:
        :parma pin_select: 1 if not args.intersection else 2
        :return: rank from 0~1
        """
        batch_rank = []
        batch_soft_rank = []
        vaults = [[results_threa[0][1][0][1].leakpw, results_threa[0][1][0][1].leakmetaid]]
        for i, result_batch in enumerate(results_threa): # result_batch: [results_avault_aiter, pin_frequency_lists]
            prior_array, h0_array, vault = [], [], []
            for result in (result_batch if not args.logical else result_batch[0]):
                prior_array.append([result[tp] for tp in result if ('h0' not in tp and 'vault' not in tp)])
                vault.append(result['vault'])  # N_EXP_VAULTS, [v1, v2, ...]
            vaults.append(vault)
            assert [results_threa[0][0][ith][w] for ith in range(len(results_threa[0][0])) for w in ['psp', 'pps', 'phybrid', 'kl', 'wang_single', 'wang_similar', 'wang_hybrid']] == [pr for pr_a in prior_array for pr in pr_a]
            prior_array = np.array(prior_array).T  # => attack_cls x N_EXP_VAULTS
            if softs_threa is not None:
                prior_array_soft = prior_array * np.array(softs_threa[i])[None]
            tmpseed = random.Random(seed_shot).randint(0, np.iinfo(np.int32).max) #np.random.randint(0, np.iinfo(np.int32).max)
            # hard
            prior_array, idx, logis, intraoffset = self.shuffle_prior(prior_array, len(vault[0]), None if not args.logical else result_batch[1], tmpseed)

            # soft
            if softs_threa is not None:
                prior_array_soft, _, _, _ = self.shuffle_prior(prior_array_soft, len(vault[0]), None if not args.logical else result_batch[1], tmpseed)
                self.find_real_candidatelist(prior_array, idx, logis, intraoffset, batch_soft_rank, len(vault[0]), reshuedidx, gpuid)
            # batch_rank.append(np.argsort(-1*prior_array, axis=1)[:, 0] / N_EXP_VAULTS)
        if args.logical:
            return np.array(batch_rank), batch_soft_rank if softs_threa is not None else None, vaults
        return np.array(batch_rank), batch_soft_rank if softs_threa is not None else None, vaults

    def find_real_candidatelist(self, prior_array, idx, logis, intraoffset, batch_rank, vaultsize, reshuedidx=None, gpuid=0):
        sorted_idx = np.argsort(-1 * prior_array) # attack_cls x N_EXP_VAULTS
        rank = list(np.where(sorted_idx == idx)[1])  # attack_cls

        rankid = np.argsort(rank) # return idx of each rank in ascending order
        median_rankid = rankid[-3] # select index for the third largest rank
        rank_thirdlargest = rank[median_rankid]
        if rank_thirdlargest < 40:
            median_rankid = rankid[-2]

        median_rankid = 2

        onlineverification_result = get_Mefp_candidatelist_para([logis[sorted_idx[median_rankid,idx]] for idx in range(len(logis))], rank[median_rankid], intraoffset, vaultsize, reshuedidx, gpuid) #[{'tried_metapwid': [0]},{'tried_metapwid': [0]}] #

        onlineverification_result[0]['r_three'] = rank
        del onlineverification_result[0]['tried_metapwid']
        onlineverification_result[1]['r_three'] = rank
        del onlineverification_result[1]['tried_metapwid']

        '''rank_cheng = get_Mefp_candidatelist_para([logis[sorted_idx[2,idx]] for idx in range(len(logis))], rank[2], intraoffset, vaultsize, reshuedidx, gpuid)
        rank_cheng[0]['r_three'] = rank
        rank_cheng[1]['r_three'] = rank

        rank_golla = [{}, {}]#get_Mefp_candidatelist_para([logis[sorted_idx[3, idx]] for idx in range(len(logis))], rank[3], intraoffset, vaultsize, reshuedidx, gpuid)
        rank_golla[0]['r_three'] = rank
        rank_golla[1]['r_three'] = rank

        rank_wang = [{}, {}]#get_Mefp_candidatelist_para([logis[sorted_idx[4, idx]] for idx in range(len(logis))], rank[4], intraoffset, vaultsize, reshuedidx, gpuid)
        rank_wang[0]['r_three'] = rank
        rank_wang[1]['r_three'] = rank'''

        if batch_rank is not None:
            batch_rank.append([onlineverification_result])

    def reuse(self, vault):
        maxreuse = 0
        for v in set(vault):
            if maxreuse < vault.count(v):
                maxreuse = vault.count(v)
        return maxreuse/len(vault)

    def shuffle_prior(self, prior, vaultsize, logis=None, seed=0):
        """

        :param prior: attack_cls x N_EXP_VAULTS
        :param logis:
        :return: shuffled array and corresponding indexs
        """
        interidx = np.arange(prior.shape[1]) # N_EXP_VAULTS
        random.Random(seed).shuffle(interidx) # shuffle mpw guesses
        intraidx = random.Random(seed+1).randint(0, PIN_SAMPLE-1) # randomly in range [0, PIN_SAMPLE-1] # shuffle pin guesses

        return prior[:, interidx], np.argmin(interidx), [logis[interidx[idx]] for idx in range(prior.shape[1])], intraidx

def list_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            files.append(file_path)
    return files

def main(plot_intersec = False):
    measure = Measure()
    # exp2 plot => hard & soft filter attacks with σ and pair bubble schemes (pin)
    pnlst = ['4'] #, '6'
    tlst = [200, 600, 2000] #, 1000, 2000
    num_processes = [16, 16, 5] # 16, 16, 6 processor number for each T_phy
    expdata_path = "/home/beeno/Dropbox/research_project/pycharm/" +\
                    "incremental_vault_gpu/attack/results/MSPM/bubble_online/" +\
                    "Me_precise(2folds)/dataid0/cons2_Nitv13" # cons2_Nitv13 or cons1
    repeatn = 5
    if len(args.outdatapath) > 0:
        measure.extract_data(pnlst, tlst, num_processes, args.intersection, expdata_path, repeatn=repeatn, destination=args.outdatapath)
    else:
        measure.extract_data(pnlst, tlst, num_processes, plot_intersec, expdata_path, repeatn=repeatn, destination = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/extracted_results/fixeditv_Nitv13/') # vanilla, vanilla_intersec, fixeditv_Nitv13, fixeditv_Nitv13_intersec

if __name__ == '__main__':
    #main()
    import json
    # merge two json files into one
    jsonpath1 = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/synthesis_vaults/vaultsize60/vaults (copy).json'
    jsonpath2 = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/synthesis_vaults/vaultsize60/vaults.json'

    with open(jsonpath1, 'r') as f:
        vaults1 = json.load(f)
    with open(jsonpath2, 'r') as f:
        vaults2 = json.load(f)
    vaults1.update(vaults2)
    with open(jsonpath2, 'w') as f:
        json.dump(vaults1, f)