import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from attack.results2table import getme

def main():
    ## experiment settings
    tstset = 'bc100' #'pb' #
    chengorgolla = 2 # 0 for cheng; 1 for golla; 2 for fei, which returns ranks and all results based on the rank method chosen
    n_vaulteachshot = 120 #231 number of vaults in each shot
    online_thre = 100 # online verification threshold for each website
    max_repeatn = 20 # repeat times for experiment
    ## attack settings
    nol_ith = -1 # nol_exps=[5, N-1], [1,3,5,7,N-1]
    intersec = True
    expermented_settings = [
                            #"fp cons1 t4000 6pin",
                            "fp cons2 t8000 6pin Nitv10",
                            #"fp cons2 t4000 6pin Nitv15",
                            #"fp cons2 t4000 6pin Nitv12",
                            #"fp cons2 t8000 6pin Nitv10",
                            ]


    root = '/home/beeno/'#'/Users/sunbin/Library/CloudStorage/'
    write_dir = root + 'Dropbox/encryption/writing/bubble/results_handling/table_data/'
    # override the files in write_dir (make the files to 0 bytes)
    for fname in os.listdir(write_dir):
        if '.csv' in fname and 'rrandmr' not in fname:
            with open(os.path.join(write_dir, fname), 'w') as f:
                pass

    for settings in expermented_settings:
        '''t = '2000'
        pin = 'pin6'
        cons = 'cons2' # '1': vanilla shuffling, 2: fixed iterval shuffling
        Nitv = 'Nitv8' # fixed interval shuffling interval'''
        # extract settings
        t = settings.split(' ')[2][1:]
        pin = 'pin' + settings.split(' ')[3][0]
        cons = settings.split(' ')[1]
        Nitv = settings.split(' ')[4] if len(settings.split(' ')) == 5 else None

        results = []
        read_dir = root + 'Dropbox/research_project/pycharm/incremental_vault_gpu/attack/results/MSPM/' + tstset + '/vaults_cmr_pl8/' #
        # list all directories from read_dir
        read_dir = [d[0] for d in os.walk(read_dir) if t in d[0] and pin in d[0] and ((cons+'_'+Nitv if cons[-1] == '2' else cons) in d[0])]# and 'multidouble_vault' not in d[0]
        if len(read_dir) != 1:
            raise ValueError('read_dir is not unique or not exist!', read_dir)
        else:
            # get 'repeat_times' as lower bound of files//n_vaulteachshot
            repeat_times = min(max_repeatn, len(os.listdir(read_dir[0])) // n_vaulteachshot) #len(os.listdir(read_dir[0])) // n_vaulteachshot
            print('read_dir:', read_dir[0].split('/')[-1], 'repeat_times:', repeat_times)


        for shotid in range(repeat_times):#[0,1,3,4]: #
            for vid in range(n_vaulteachshot): #:[0,1,2,4,5,6,8,13,15,18,20,23,24,28,32,39,45,50,53,56]:
                fname = 'results_v' + str(vid) + '_shot' + str(shotid) + '.data'
                with open(os.path.join(read_dir[0], fname), 'rb') as f:
                    results.append(pickle.load(f))

        extract_login_distribution(results, nol_ith=nol_ith, write_dir=write_dir, intersec=intersec, n_vaulteachshot=n_vaulteachshot, delta=online_thre, chengorgolla=chengorgolla)

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

            metaid_histog[i].append(int((loginsum / (vault_size - 1)) >= delta) * 100)  # 0
            Fp[i].append(r_dict['fp'][nol_ith])

    fail = np.array(Fp) * (100-np.array(metaid_histog)) + np.array(metaid_histog)
    Fp = np.array(Fp) * 100 # num_vaults x num_shots
    metaid_histog = np.array(metaid_histog) # num_vaults x num_shots

    re = pd.read_csv('/home/beeno/Dropbox/encryption/writing/bubble/results_handling/valuewise_plcmr_bc100_pl8.csv', sep=',', header=None)
    re = re.values
    pls, cmrs = re[:, 0], re[:, 1]
    fig, axs = plt.subplots(1, 2)  # Adjust width and height as desired

    pl_sep = [[5.5, 6.5], [6.5, 7.5], [7.5, 8.5], [8.5, 9.5], [9.5, 10.5]]
    errorplot_x, errorplot_mean, errorplot_std = [], [], []
    stat = {}
    for ith, sep in enumerate(pl_sep):
        sep_l = sep[0]
        sep_h = sep[1]
        idx = (pls > sep_l) * (pls < sep_h)
        stat[sep_l] = idx.sum()
        errorplot_x.append(sep_l)
        area_data = (fail[idx] >= 100).sum(0) / idx.sum() * 100
        errorplot_mean.append(np.mean(area_data))
        errorplot_std.append(np.std(area_data))
    print(stat)
    # create errorbar plot
    axs[0].errorbar([v for v in errorplot_x], errorplot_mean, yerr=errorplot_std, label='lm', fmt='-', capsize=4)
    axs[0].set_xticks(errorplot_x, [str(int(v)) for v in errorplot_x])
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel('Average pw length')
    axs[0].set_ylabel('ratio')
    #axs.legend()
    axs[0].grid(axis='y', linestyle='--', linewidth=0.1)
    # write to csv error data
    np.savetxt("/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/errorplot_pl.csv", np.stack([errorplot_mean, errorplot_std], axis=1), delimiter=",")

    cmr_sep = [[0, 0.05], [0.05, 0.15], [0.15, 0.25], [0.25, 0.35], [0.35, 0.45], [0.45, 0.55]] #[[0, 0.025], [0.025, 0.075], [0.075, 0.125], [0.125, 0.175], [0.175, 0.225], [0.225, 0.275], [0.275, 0.325], [0.325, 0.375], [0.375, 0.425], [0.425, 0.475], [0.475, 0.525], [0.525, 0.575]] #
    errorplot_x, errorplot_mean, errorplot_std = [], [], []
    stat = {}
    for ith, sep in enumerate(cmr_sep):
        sep_l = sep[0]
        sep_h = sep[1]
        idx = (cmrs >= sep_l) * (cmrs < sep_h)
        stat[sep_l] = idx.sum()
        errorplot_x.append(sep_l)
        area_data = (fail[idx] >= 100).sum(0) / idx.sum() * 100
        errorplot_mean.append(np.mean(area_data))
        errorplot_std.append(np.std(area_data))
    print(stat)
    # create errorbar plot
    axs[1].errorbar([v for v in errorplot_x], errorplot_mean, yerr=errorplot_std, label='lm', fmt='-', capsize=4)
    axs[1].set_xticks(errorplot_x, [str(int(v)) for v in errorplot_x])
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('Cmr')
    axs[1].set_ylabel('ratio')
    # axs.legend()
    axs[1].grid(axis='y', linestyle='--', linewidth=0.1)
    # write to csv error data
    np.savetxt("/home/beeno/Dropbox/encryption/writing/bubble/results_handling/table_data/errorplot_cmr.csv",
               np.stack([errorplot_mean, errorplot_std], axis=1), delimiter=",")

    #plt.show()



if __name__ == '__main__':
    main()