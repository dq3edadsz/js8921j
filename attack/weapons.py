#from numba import jit
import copy
from time import time
import tqdm
import random
import subprocess
import math
from multiprocessing import Pool
from linecache import getline
from attack.para import check_logical
import numpy as np
import pickle
from opts import opts
from utils import Digit_Vault, grouprandom_fixeditv
from tqdm import tqdm
args = opts().parse()
from MSPM.mspm_config import *
if args.victim == 'Golla':
    from Golla.markov_coding import generate_batchdecoys
from MSPM.unreused_prob import unreuse_p

def getthreshold(dte, pw_lst, problst, pw): # get the probability of independent encoding of pw
    ith = len(pw_lst) # starting from 1 for f(ith)
    prob = dte.spm.encode_pw(pw)[1]
    thre = 1.
    if ith > 0:
        prob_lst = [np.log(np.array(prob)).sum() + np.log(unreuse_p(ith))]
        for i in range(ith):
            prob = dte.sspm.encode_pw(pw_lst[i], pw, ith)[1]
            prob = np.log(np.array(prob)).sum() + problst[i] + np.log((1 - unreuse_p(ith)) / ith) if len(prob) != 0 else -np.inf
            prob_lst.append(prob)
        thre = np.exp(prob_lst[0]/4.) / np.exp(np.array(prob_lst)/4.).sum()
    return thre

def mspmattack(rn, outputdir, T, batch, x, testset, decoygen, dte, weight, measure, vaultgen, assem_pws, vids):
    """
    str(x * (repeat_id + 1)), self.T, batch, x, self.testset, self.decoygen, self.dte,
    :param data: decoy pws list with length (N_EXP_VAULTS-1)*batch
    :param T: physical expansion
    :param dte: sspm
    :param testset: single pw vault
    :param weight:
    :param assem_pws: args (num, length, pws, pwrules, basepw)
    :return: [[threa1], [threa2], ..., [threan]]
    """
    unique_vids = list(range(len(testset))) # [vss.index(vs) for vs in unique_vss]
    expansion_rate = [1] * len(testset) # [(i+3) for i in range(len(testset))]
    decoy_num = REFRESH_RATE + PS_DECOYSIZE # number of decoy vaults that will be generated
    for n_, tset, in enumerate(tqdm(testset)): # tset is a password vault (list of passwords) from dataset
        #tset = decoy_vaults[0] # test only
        if n_ not in unique_vids:
            continue

        '''if rn < 5:
            continue
        elif rn == 5 and n_ < 48:
            continue'''

        batch_bundle = [] # each element is [[results_avault_aiter, pin_frequency_lists], softs_aiter, reshuedidx]
        results_avault_aiter = [] # list with N_EXP_VAULTS vault "results", starting from real to the rest of fake
        softs_aiter = []
        feat_tmp = weight.features[0] # random.randint(0, len(weight.features) - 1)
        DV = Digit_Vault(tset, int(vids[n_]), int(len(tset)/expansion_rate[n_]) if args.expandtestset else None) # represent vualt as a digit vault for ease of experiment
        pin_gt = random_pin() # randomly drawn from test set
        mpw_gt = random_mpw()
        #print('Reak mpw: ', mpw_gt, 'Real pin: ', pin_gt)
        if not args.intersection:
            dvlist = list(grouprandom_fixeditv(DV.create_dv(tset)[0]).keys()) if args.fixeditv else None
            scrambled_idx = list_shuffle(mpw_gt, pin_gt, len(tset), existing_list=dvlist)
        else:
            assert args.version_gap > 0 and len(tset) > args.version_gap
            seed_ = random.randint(0, MAX_INT)
            # get leaked versions based on setting "--version_gap" and "--isallleaked"
            dvlist_versions = [list(grouprandom_fixeditv(DV.create_dv(tset[:(len(tset)-vg)])[0], seedconst=seed_).keys()) if args.fixeditv else None for vg in range(args.version_gap+1)]
            scrambled_idx = [list_shuffle(mpw_gt, pin_gt, len(tset)-vg, existing_list=dvlist_versions[vg]) for vg in range(args.version_gap+1)]
            if args.isallleaked == 0:
                scrambled_idx = [scrambled_idx[0], scrambled_idx[-1]]

        reshuedidx = getshuffledidx(scrambled_idx, tset, pin_gt, gpuid=args.gpu) # reshuffled mapping in the shape of (PIN_REFRESH, PADDED_TO*PIN_SAMPLE)

        pin_frequency_lists = [] # list of pin freq list for each candidate vault (mpw)
        #ts = time()
        for threa in range(N_EXP_VAULTS if not args.withleak else T_PHY):
            results = {}
            if threa % REFRESH_RATE == 0:
                # one for each vault; probs are log-summed only (without /4)
                if args.victim == 'MSPM':
                    # get the probability of independent encoding of each pw
                    pasteprobs = [np.log(np.array(dte.spm.encode_pw(pw)[1])).sum() for pw in tset] # log-summed only
                    threshold = [getthreshold(dte, tset[:i], pasteprobs[:i], pw) for i, pw in enumerate(tset)]
                    ts = time()
                    # probs_sspm are log-summed only over probs above
                    _, probs_spm_mspm, decoyvaults = vaultgen(len(tset) - 1, REFRESH_RATE, dte=dte, path1pws=None, path1probs=None, pastepws=tset, pasteprobs=pasteprobs, threshold=threshold, possibleidx=DV.possibleidx) #vaultgen(len(tset) - 1, REFRESH_RATE, dte=dte, path1pws=None, path1probs=None, pastepws=tset, pasteprobs=pasteprobs, threshold=[0 for thr in threshold], possibleidx=[1]) #
                    _, probs_spm_mspm_, decoyvaults_ = vaultgen(len(tset) - 1, PS_DECOYSIZE, dte=dte, path1pws=None, path1probs=None, pastepws=tset, pasteprobs=pasteprobs, threshold=[0 for thr in threshold], possibleidx=[1]) # [0:1]
                    print('vaultgen using', time() - ts)
                elif args.victim == 'Golla':
                    decoyvaults, probs_spm_mspm = generate_batchdecoys(tset, None, decoynum=REFRESH_RATE)#generate_batchdecoys(tset, DV.leakpw, decoynum=REFRESH_RATE)
                    decoyvaults_, probs_spm_mspm_ = generate_batchdecoys(tset, None, decoynum=PS_DECOYSIZE)
                probs_spm_mspm.extend(probs_spm_mspm_)
                decoyvaults.extend(decoyvaults_)

                if threa == 0:
                    vault = tset
                    results['psp'] = additionw(weight.singlepass(vault, dte.spm), vault, DV.leakpw, DV.leakmetaid)
                    kl = additionw(weight.kl(vault, dte.spm), vault, DV.leakpw, DV.leakmetaid)
                    wang_singlepass = additionw(weight.wang_singlepass(vault, dte), vault, DV.leakpw, DV.leakmetaid)
            if threa > 0:
                # PS_DECOYSIZE decoy vaults used for similarity weight calculation
                #s_ = time()
                vault = decoyvaults[threa % REFRESH_RATE]#assem_pws((PS_DECOYSIZE + 1) * (threa % REFRESH_RATE), 1, len(tset) - 1, pws, dte=dte, newpws=path1newpws)[0]
                #print('waepons: assemble a vault using', time() - s_)
                prob_tmp = probs_spm_mspm[(threa % REFRESH_RATE) * len(tset): (threa % REFRESH_RATE) * len(tset) + len(tset)]
                results['psp'] = additionw(0, vault, DV.leakpw, DV.leakmetaid)
                kl = additionw(0, vault, DV.leakpw, DV.leakmetaid)
                wang_singlepass = additionw(0, vault, DV.leakpw, DV.leakmetaid)
                if results['psp'] == 0:
                    results['psp'] = weight.singlepass(vault, dte.spm, prob_tmp)
                    kl = weight.kl(vault, dte.spm, prob_tmp)
                    wang_singlepass = weight.wang_singlepass(vault, dte, prob_tmp)
            if args.logical:
                pin_frequency_lists.append([vault, DV])
            #s_ = time()
            results['pps'] = additionw(0, vault, DV.leakpw, DV.leakmetaid)
            if results['pps'] == 0:
                decoy_draws = list(np.random.randint(PS_DECOYSIZE, size=30) + REFRESH_RATE)
                decoys = [decoyvaults[draw_id] for draw_id in decoy_draws]
                decodic = {dvid:decoys[dvid] for dvid in range(len(decoys))}
                # assem_pws((PS_DECOYSIZE + 1) * (threa % REFRESH_RATE) + 1, PS_DECOYSIZE, len(vault) - 1, pws, dte, newpws=path1newpws)
                results['pps'] = weight.passsimi(vault, dte, decodic, feat_tmp, weight.p_real)
            #print('waepons: assemble PS_DECOYSIZE vaults using', time() - s_)
            # start, num, length, pws, basepw, dte, probs=None, p=False
            results['phybrid'] = additionw(results['psp'] * results['pps'], vault, DV.leakpw, DV.leakmetaid)
            results['kl'] = kl
            results['wang_single'] = wang_singlepass[0] if isinstance(wang_singlepass, list) else wang_singlepass
            results['wang_similar'] = 1 if kl != -np.inf else -np.inf # placeholder, will be modified in func 'addition_weight'
            results['wang_hybrid'] = wang_singlepass
            results['vault'] = vault
            if DV.leakpw in vault and args.softfilter:
                softs_aiter.append(1)
            else:
                softs_aiter.append(1)
            results_avault_aiter.append(results)

        batch_bundle.append([[results_avault_aiter, pin_frequency_lists], softs_aiter, reshuedidx])
        #addition_weight(batch_bundle)
        worker = measure.rank_r([batch_bundle[0][0]], [batch_bundle[0][1]], '4' in args.pin, batch_bundle[0][2], args.gpu, rn*len(testset)+n_)
        with open(outputdir + 'results_' + 'v' + str(n_) + '_shot' + str(rn) + '.data', 'wb') as f:
            pickle.dump(worker, f)

def addition_weight(batch_bundle):
    # add weights to each vault in results (dict) by calling aother python script using subprocess
    # step1: write unique password of each vault into a txt file (each pw a row) '/home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/test_files/dataset_ts.txt'
    # step2: call another python script to predict score for each pw (which then will write reults into '/home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/data/pass2path_1667500_dataset_ts.predictions')
    # step3: read the file and update into additional weight for each vault

    # step1
    pws = []
    for i, vault_digitvault in enumerate(batch_bundle[0][0][1]):
        pws_avault_ = list(set(vault_digitvault[0]))
        pws_avault = [pw for pw in pws_avault_ if len(pw)>5]
        pws.extend(pws_avault)
    pws = list(set(pws))
    sfx = '_' + str(T_PHY) + args.pinlength + str(args.gpu) + (str(Nitv) if args.fixeditv else '') + str(args.version_gap) + args.exp_pastebinsuffix + args.victim # avoid conflict
    with open('/home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/test_files/dataset_ts' + sfx + '.txt', 'w') as f:
        for pw in pws:
            f.write(pw + '\n')

    # step2 /home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/score_eachvault.py
    subprocess.call(['/home/beeno/anaconda3/envs/pass2path/bin/python',
                     '/home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/score_eachvault.py',
                     '-gpu', str(args.gpu), '-sfx', sfx])

    # step3 results file '/home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/data/pass2path_1667500_dataset_ts.pkl'
    with open('/home/beeno/Dropbox/research_project/pycharm/credtweak/credTweakAttack/data/pass2path_1667500_dataset_ts'+sfx+'.pkl', 'rb') as f:
        pws_pws2score = pickle.load(f)
    for i, scores_avault in enumerate(batch_bundle[0][0][0]):
        vault_ = scores_avault['vault']
        get_hybridwang(vault_, pws_pws2score, scores_avault)

def get_hybridwang(vault, pws_pws2score, scores_avault):
    # for each pw_i in set(vault), get score of pw_j in pws_pws2score[pw_i] for all pw_j in vault\{pw_i}
    # then get the average of the scores and use it to divide scores_avault['wang_hybrid']
    if not isinstance(scores_avault['wang_hybrid'], list):
        return
    score_wanghybrid = scores_avault['wang_hybrid'][0]
    wang_denom = []
    for pwi in list(set(vault)):
        pws_remaining = [pw for pw in vault if pw != pwi]
        i_pws2score = pws_pws2score[pwi]
        for pwj in pws_remaining:
            if pwj in i_pws2score:
                wang_denom.append(i_pws2score[pwj])
    if len(wang_denom) > 0:
        scores_avault['wang_similar'] = scores_avault['wang_hybrid'][1] / np.mean(np.array(wang_denom)) * len(vault) / (len(vault)+1)
        scores_avault['wang_hybrid'] = score_wanghybrid * scores_avault['wang_similar'] # scores_avault['wang_hybrid'][1] / np.mean(np.array(wang_denom)) * len(vault) / (len(vault)+1)
    else:
        scores_avault['wang_hybrid'] = score_wanghybrid # if args.victim == 'MSPM' else 0

def spmattack(Q, seq, T, batch, x, testset, decoygen, dte, weight):
    """
    Q, str(x * (repeat_id + 1)), self.T, batch, x, self.testset, self.decoygen, self.dte,
    :param Q:
    :param seq:
    :param data: decoy pws list with length (N_EXP_VAULTS-1)*batch
    :param T: physical expansion
    :param dte: mspm
    :param pw_lst: pw list with length batch
    :param weight:
    :return: [[threa1], [threa2], ..., [threan]]
    """
    print('already => '+seq+'%')
    batch_results = []
    pw_lst = testset[x * batch: (x + 1) * batch]
    data, probs = decoygen(num=(N_EXP_VAULTS - 1)*batch*REPEAT_NUM, pre=args.predecoys)
    assert len(probs) == len(data) == (N_EXP_VAULTS - 1) * batch * REPEAT_NUM
    with open('ana.data1', 'wb') as f:
        pickle.dump(probs, f)
    for threa in tqdm(range(len(pw_lst)), miniters=int(len(pw_lst)/10), unit="attack_pw"):
        data_threa_ = pw_lst[threa:threa+1]
        #data_threa.extend(data)
        for repeat_id in range(REPEAT_NUM):
            data_threa = data_threa_.copy()
            data_threa.extend(data[(threa * REPEAT_NUM + repeat_id) * (N_EXP_VAULTS - 1):
                                   (threa * REPEAT_NUM + repeat_id + 1) * (N_EXP_VAULTS - 1)])
            results_avault_aiter = [] # list with N_EXP_VAULTS vault results, starting from real to the rest of fake
            assert len(data_threa_) == 1 and len(data_threa) == N_EXP_VAULTS
            for i in range(len(data_threa)):
                results = {}
                if i != 0:
                    results['psp'] = weight.singlepass(data_threa[i:(i+1)], dte,
                                                   probs[(threa * REPEAT_NUM + repeat_id) * (N_EXP_VAULTS - 1):
                                                         (threa * REPEAT_NUM + repeat_id + 1) * (N_EXP_VAULTS - 1)][i-1:i])
                else:
                    results['psp'] = weight.singlepass(data_threa[0:1], dte)
                results_avault_aiter.append(results)
            batch_results.append(results_avault_aiter)
    with open('ana.data2', 'wb') as f:
        pickle.dump(batch_results, f)
    #Q.put(batch_results)
    #print('eval =>', time() - s)
    return batch_results

def sspmattack(Q, seq, T, batch, x, testset, decoygen, dte, weight, vaultgen, assem_pws):
    """
    Q, str(x * (repeat_id + 1)), self.T, batch, x, self.testset, self.decoygen, self.dte,
    :param Q:
    :param seq:
    :param data: decoy pws list with length (N_EXP_VAULTS-1)*batch
    :param T: physical expansion
    :param dte: sspm
    :param testset: single pw vault
    :param weight:
    :param assem_pws: args (num, length, pws, pwrules, basepw)
    :return: [[threa1], [threa2], ..., [threan]]
    """
    print('already => '+seq+'%')
    batch_results = []
    #data, _ = decoygen(num=N_EXP_VAULTS, pre=args.predecoys)
    '''with open('ana.data1', 'wb') as f:
        pickle.dump(probs, f)'''
    for n_, tset in enumerate(testset):
        results_avault_aiter = []  # list with N_EXP_VAULTS vault results, starting from real to the rest of fake
        for threa in tqdm(range(N_EXP_VAULTS), miniters=int(N_EXP_VAULTS/10), unit="attack_pw"):
            if threa % REFRESH_RATE == 0:
                pws = vaultgen(len(tset)-1, (100 + 1) * REFRESH_RATE,
                               decoygen())  # tuple of (pws, pwrules) for assembling pw
            if threa > 0:
                vault = assem_pws((100+1)*(threa % REFRESH_RATE), 1, len(tset)-1, pws, decoygen())[0]
            else:
                vault = tset
            results = {}
            results['pps'] = weight.passsimi(vault, dte,
                               assem_pws((100+1)*(threa % REFRESH_RATE)+1, 100, len(vault)-1, pws, decoygen()))
            results_avault_aiter.append(results)
        batch_results.append(results_avault_aiter)
    with open('ana.data2', 'wb') as f:
        pickle.dump(batch_results, f)
    #Q.put(batch_results)
    #print('eval =>', time() - s)
    return batch_results

def decoys(seq, decoygen):
    print('already => '+seq+'%')
    return decoygen(N_EXP_VAULTS)

def additionw(value, vault, leakpw, leakmetaid):
    if not args.withleak:
        return value
    if leakpw in vault:
        if args.fixeditv:
            if leakpw in vault[leakmetaid // Nitv * Nitv: (leakmetaid // Nitv + 1) * Nitv]:
                return value
            else:
                return -np.inf
        return value
    else:
        return -np.inf

def softfilter(vault, leakpw, dte):
    ith = len(vault)  # starting from 1 for f(ith)
    prob_lst = []
    sim = 0
    for i in range(ith):
        _, prob = dte.sspm.encode_pw(vault[i], leakpw, ith)
        if len(prob) != 0:
            prob = np.log(np.array(prob)).sum() + np.log((1 - unreuse_p(ith)) / ith) # get the probability
            sim += 1
        else:
            prob = 0
        prob_lst.append(prob)
    return -np.array(prob_lst).sum()/20 #1 - 1 / (1 + math.exp(-np.array(prob_lst).sum()/6))

def list_shuffle(mpw, pin, vault_size, recover=False, existing_list=None):
    """
    passed unit test function 'tes_list_shuffle()'
    shuffle in direct or fixed interval way
    :param mpw:
    :param pin:
    :param vault_size:
    :param recover:
    :return:
    """
    itvsize = Nitv if args.fixeditv else vault_size
    shuffled_list = []
    if existing_list is not None:
        assert len(existing_list) == vault_size
    for ith in range(math.ceil(vault_size / itvsize)):
        shuffled_list_tmp = list(np.arange(0, itvsize)) if existing_list is None else copy.deepcopy(existing_list)[ith*itvsize:(ith+1)*itvsize]
        if args.fixeditv and args.fixeditvmode == 1 and ith==vault_size//itvsize and vault_size%itvsize != 0: # requires no padding
            shuffled_list_tmp = shuffled_list_tmp[:vault_size % itvsize]
        rng = random.Random(hash(pin + str(ith))) if args.fixeditv else random.Random(hash(pin)) # random.Random(hash(mpw + pin))
        rolls = [rng.randint(0, len(shuffled_list_tmp)-1) for _ in range(len(shuffled_list_tmp))] # vault_size
        if not recover:
            for i in range(len(shuffled_list_tmp)-1, -1, -1):
                shuffled_list_tmp[i], shuffled_list_tmp[rolls[i]] = shuffled_list_tmp[rolls[i]], shuffled_list_tmp[i]
        else:
            for i in range(len(shuffled_list_tmp)):
                shuffled_list_tmp[i], shuffled_list_tmp[rolls[i]] = shuffled_list_tmp[rolls[i]], shuffled_list_tmp[i]
        sub_list = list(np.array(shuffled_list_tmp) + ith * itvsize) if existing_list is None else shuffled_list_tmp
        shuffled_list.extend(sub_list)
    return shuffled_list

def getshuffledidx(scrambled_idx, vault, pin, gpuid):
    if args.logical:
        # (PIN_REFRESH, PADDED_TO*PIN_SAMPLE)
        # get "reshuedidx" as reshuffled mapping in the shape of (PIN_REFRESH, PADDED_TO*PIN_SAMPLE)
        if not args.intersection:
            reshuedidx = check_logical(scrambled_idx, PIN_REFRESH, len(vault), gpuid, seed_=random.Random(hash(pin)).randint(0, MAX_INT))
        else:
            reshuedidx = [[] for _ in range(1)]
            shuffle_idx = None
            for i, si in enumerate(scrambled_idx):
                if args.fixeditv:
                    rsidx, shuffle_idx = check_logical(si, PIN_REFRESH, len(si), gpuid, seed_=random.Random(hash(pin)).randint(0, MAX_INT), reshuidx_whole=shuffle_idx, depth=i)
                else:
                    rsidx = check_logical(si, PIN_REFRESH, len(si), gpuid, seed_=random.Random(hash(pin)).randint(0, MAX_INT))
                for i_, rs in enumerate(rsidx):
                    reshuedidx[i_].append(rs)
        return reshuedidx
    return None

def random_mpw(mpw_gt=None, seed_=1):
    pw = str()
    i = 0
    while len(pw) < 10 or pw == mpw_gt:
        pw = getline(SOURCE_PATH + "/data/password_dict.txt", random.Random(seed_+i).randint(0, 120000)).strip()
        i += 1
    return pw

def random_pin(seed_=1):
    # random select a pin from path '/data/'+args.pin
    size = int(1780588*0.199) if '4' in args.pin else int(0.199*2758491)
    return getline(SOURCE_PATH + "/data/pin/" + args.pin.split('.')[0]+'_test.'+args.pin.split('.')[1], random.Random(seed_).randint(1, size)).strip()