import numpy as np
import tqdm
#from Vault.vault import
from time import time, time_ns, sleep
import sys
import cupy as cp
from MSPM.unreused_prob import unreuse_p
from MSPM.incre_pw_coding import Incremental_Encoder
from multiprocessing import Pool
import math
from cuda_voila import convert2seed, convert2seed_shuffle, ngramip_decodepw, ngramcp_decodepw, rule_decode, num_chardecode, pathn_decode, recovershuffle, checkeachvault, checkincrement
from numba.cuda.random import create_xoroshiro128p_states
import random
from numba import cuda
from opts import opts
args = opts().parse()
from MSPM.mspm_config import *
if args.victim == 'MSPM':
    dte_global = Incremental_Encoder(args.nottrain) #None #

offsetmap = np.array([0, 1, 3, SEED_LEN//2+1], dtype=int)


def assemble_idx(idxs, probs, dte):
    """
    spm
    :param idxs: MAX_PW_LENGTH*num [len1, ip1, cp1_1, cp1_2, cp1_3, ..., len2, ip2, cp2_1, cp2_2, cp2_3, ...,]
    :param dte:
    :return:
    """
    pw_lst = []
    prob_lst = []
    for i in range(idxs.size // SEED_LEN):
        pw = dte._i2nIP(int(idxs[i * SEED_LEN + 1]))
        prob_lst.append(probs[i * SEED_LEN: (i + 1) * SEED_LEN].sum())
        for x in range(2, idxs[i * SEED_LEN] + 1):
            cp = dte._i2nCP(idxs[i * SEED_LEN + x])
            pw += cp[-1]
        pw_lst.append(pw)
    return pw_lst, prob_lst


def decoygenspm_para(num=None, pre=False, dte=None):
    """
    spm 
    :param dte:
    :param num:
    :return: list of decoy pws with length num
    """
    assert num is not None
    with cuda.gpus[args.gpu]:
        # generate random seeds
        threadsperblock = 32
        blockspergrid = (SEED_LEN * num + threadsperblock - 1) // threadsperblock
        seed = np.zeros(SEED_LEN * num)
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.randint(0, MAX_INT))
        convert2seed[blockspergrid, threadsperblock](rng_states, 0, 1, SEED_LEN * num, seed, SEED_MAX_RANGE, 1, -1)
        pws = np.ones(SEED_LEN * num, dtype=int)
        probs = np.zeros(SEED_LEN * num, dtype=np.float64)
        markov_ids = np.ones(num)
        #   length
        s_ = time()
        for i in range(num):
            pws[int(i * SEED_LEN)] = dte.spm.decode_len(seed[int(i * SEED_LEN)])  # could have length prob

        #   ip (para)
        for i in range(num):
            pw_len = pws[int(i * SEED_LEN)] + dte.spm.seed_length_rec
            if pw_len in TRAIN_LENGTH:
                markov_id = TRAIN_LENGTH.index(pw_len)
            else:
                markov_id = len(TRAIN_LENGTH)
            markov_ids[i] = markov_id
        print('basepws:length decode using =>', time() - s_)
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil(num / threadsperblock[0])
        blockspergrid_y = math.ceil(dte.spm.no_ip_ngrams / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        s_ = time()
        ngramip_decodepw[blockspergrid, threadsperblock](seed, dte.spm.ip_list, SEED_LEN, pws,
                                                         markov_ids, dte.spm.no_ip_ngrams, probs)
        print('basepws:decode ip using =>', time() - s_)

        #   cp (# parallel decode)
        blockspergrid_x = math.ceil(num / threadsperblock[0])
        blockspergrid_y = math.ceil((len(ALPHABET) - 1) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        s_ = time()
        for i in range(2, MAX_PW_LENGTH):
            ngramcp_decodepw[blockspergrid, threadsperblock](seed, dte.spm.cp_list, SEED_LEN, pws,
                                                             len(ALPHABET), markov_ids, dte.spm.no_cp_ngrams,
                                                             i, dte.spm.ngram_size, probs)
        print('basepws:decode cp using =>', time() - s_)
    # assemble pws (indx of list)
    s_ = time()
    pws, probs = assemble_idx(pws, probs, dte.spm)
    print('basepws:assemble using =>', time() - s_)
    return pws, probs


def decoysspmgen_para(length, num, basepw, dte=None):
    """
    para template: location predicate
                       0 dr_ndr,
                       1 portion (Head_Tail ?)
                       2 portion Head (don't need to decode, default)
                       3 operation (Add, Delete, Delete-then-add)
                       4-5 DN (if any), AN (if any),
                       6-19 chars... (with length of SSPM_CHARS_MAX)
                       25 portion Tail (don't need to decode, default)
                       26 operation (Add, Delete, Delete-then-add)
                       27-28 DN (if any), AN (if any),
                       29-42 chars... (with length of SSPM_CHARS_MAX)

    :param dte:
    :param num:
    :return: list of decoy pws with length num
    """
    # generate random seeds
    threadsperblock = 32
    blockspergrid = (SEED_LEN * (num * length) + threadsperblock - 1) // threadsperblock
    seed = np.zeros(SEED_LEN * (num * length), dtype=int)
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.randint(0, MAX_INT))
    convert2seed[blockspergrid, threadsperblock](rng_states, 0, 1, SEED_LEN * (num * length), seed, SEED_MAX_RANGE, 1, -1)
    pws = np.ones(SEED_LEN * (num * length), dtype=int)
    #   dr or ndr start = decode_seed(modi_ruleset['nDR'], seeds.pop(0))
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil((num * length) * 4 / threadsperblock[0])
    blockspergrid_y = math.ceil(3 / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # s_ = time()
    # seed, an_cum, dn_cum, char_cum, decodelength, seedlength, pws, max_length
    sspmreuse_lst = []
    for rei in range(1, 150):
        right_dictmp = dte.set_sspmreuse(rei)
        counts = [c['count'] for c in list(right_dictmp.values())[1:]]  # exclude 'count' from the right_dic
        sspmreuse_lst.append([sum(counts[:i + 1]) for i in range(len(counts))])
    for i in range(4):
        if i == 0:  # i: 0 -> drornot
            cumlst = [stripcount(dte.modi_ruleset)] * 3  # starting from non-zero
        elif i == 1:  # 1 -> portion
            cumlst = [stripcount(dte.modi_ruleset['nDR'])] * 3
        else:  # i == 2, 3 -> operation (head, tail)
            cumlst = []
            for portion in list(dte.modi_ruleset['nDR'].keys())[1:]:
                if portion == 'Head_Tail':
                    if i == 2:
                        cumlst.append(stripcount(dte.modi_ruleset['nDR']['Head_Tail']['Head']))
                    else:
                        cumlst.append(stripcount(dte.modi_ruleset['nDR']['Head_Tail']['Tail']))
                else:
                    cumlst.append(stripcount(dte.modi_ruleset['nDR'][portion]))
        rule_decode[blockspergrid, threadsperblock](seed, i, 4, SEED_LEN, pws, np.array(cumlst),
                                                    (num * length) * 4,
                                                    offsetmap, np.array(sspmreuse_lst), length)
    #   modifying number and chars decode (# parallel decode)
    blockspergrid_x = math.ceil((num * length) * (SSPM_CHARS_MAX * 2 + 4) / threadsperblock[0])
    blockspergrid_y = math.ceil((len(ALPHABET) - 1) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # s_ = time()
    # seed, an_cum, dn_cum, char_cum, decodelength, seedlength, pws, max_length
    num_chardecode[blockspergrid, threadsperblock](seed, np.array(list(dte.alphabet_dict['AN'].values())),
                                                   np.array(list(dte.alphabet_dict['DN'].values())),
                                                   np.array(list(dte.alphabet_dict['alp'].values())),
                                                   SSPM_CHARS_MAX * 2 + 4, SEED_LEN, pws,
                                                   (num * length) * (SSPM_CHARS_MAX * 2 + 4))
    # print('decode modifying number and chars parallely using =>', time() - s_)
    # assemble pws (indx of list)
    return pws


def stripcount(right_dic):
    counts = [c['count'] for c in list(right_dic.values())[1:]]  # exclude 'count' from the right_dic
    return np.array([sum(counts[:i + 1]) for i in range(len(counts))])  # from 0 to the value of right_dic['count']


def assemble_sspm(start, num, length, pws, basepw, dte=None):
    """

    :param start: as unit of pwvault
    :param num: as unit of pwvault
    :param length: actual length
    :param pws:
    :param basepw:
    :return:
    """
    pwrules = np.array(['Delete-then-add'] * (4 * num * length))
    drornot = list(dte.modi_ruleset.keys())[1:]
    portion = list(dte.modi_ruleset['nDR'].keys())[1:]
    portlst = []
    for port in list(dte.modi_ruleset['nDR'].keys())[1:]:
        if port == 'Head_Tail':
            dictmp = {'Head': list(dte.modi_ruleset['nDR']['Head_Tail']['Head'].keys())[1:]}
            dictmp['Tail'] = list(dte.modi_ruleset['nDR']['Head_Tail']['Tail'].keys())[1:]
            portlst.append(dictmp)
        else:
            portlst.append(list(dte.modi_ruleset['nDR'][port].keys())[1:])
    pws_tmp = pws[start * length * SEED_LEN: SEED_LEN * (length * (start + num))]
    for i in range(num * length):
        pwrules[i * 4] = drornot[pws_tmp[i * SEED_LEN + offsetmap[0]]]
        if pwrules[i * 4] == 'DR':
            continue
        portid = pws_tmp[i * SEED_LEN + offsetmap[1]]
        pwrules[i * 4 + 1] = portion[portid]
        if pwrules[i * 4 + 1] == 'Head_Tail':
            pwrules[i * 4 + 2] = portlst[portid]['Head'][pws_tmp[i * SEED_LEN + offsetmap[2]]]
            pwrules[i * 4 + 3] = portlst[portid]['Tail'][pws_tmp[i * SEED_LEN + offsetmap[3]]]
        else:
            pwrules[i * 4 + 2] = portlst[portid][pws_tmp[i * SEED_LEN + offsetmap[2]]]
            pwrules[i * 4 + 3] = portlst[portid][pws_tmp[i * SEED_LEN + offsetmap[3]]]
    vaults = {}
    alp = np.array(list(dte.alphabet_dict['alp'].keys()))
    an = np.array(list(dte.alphabet_dict['AN'].keys()))
    dn = np.array(list(dte.alphabet_dict['DN'].keys()))
    for vid in range(num):
        vault = [basepw]
        for i in range(length):
            pw = pws_tmp[(vid * length + i) * SEED_LEN: (vid * length + i + 1) * SEED_LEN]
            pwrule = pwrules[(vid * length + i) * 4: (vid * length + i + 1) * 4]
            if pwrule[0] == 'DR':
                vault.append(basepw)
            else:
                newpw = basepw
                if 'Head' in pwrule[1]:
                    if 'Delete' in pwrule[2]:
                        newpw = newpw[int(dn[int(pw[4])]):]
                    if 'dd' in pwrule[2]:
                        newpw = ''.join(alp[pw[6: 6 + int(an[int(pw[5])])]]) + newpw
                if 'Tail' in pwrule[1]:
                    if 'Delete' in pwrule[3]:
                        newpw = newpw[:-int(dn[pw[SEED_LEN // 2 + 2]])]
                    if 'dd' in pwrule[3]:
                        newpw = newpw + ''.join(
                            alp[pw[SEED_LEN // 2 + 4: SEED_LEN // 2 + 4 + int(an[int(pw[SEED_LEN // 2 + 3])])]])
                vault.append(newpw)
        vaults[vid] = vault
    return vaults


def decoymspmgen_para(length, num, dte, path1pws=None, path1probs=None, pastepws=None, pasteprobs=None, threshold=None, possibleidx=None):
    """ len(tset) - 1, (100 + 1) * REFRESH_RATE
    para template: location predicate
                       0 dr_ndr,
                       1 portion (Head_Tail ?)
                       2 portion Head (don't need to decode, default)
                       3 operation (Add, Delete, Delete-then-add)
                       4-5 DN (if any), AN (if any),
                       6-19 chars... (with length of SSPM_CHARS_MAX)
                       25 portion Tail (don't need to decode, default)
                       26 operation (Add, Delete, Delete-then-add)
                       27-28 DN (if any), AN (if any),
                       29-42 chars... (with length of SSPM_CHARS_MAX)
                       ...
                       49 pathnum

    :param dte:
    :param num:
    :return: list of decoy pws with length num
    """
    # generate random seeds
    with cuda.gpus[args.gpu]:
        threadsperblock = 32
        blockspergrid = (SEED_LEN * (num * length) + threadsperblock - 1) // threadsperblock
        seed = np.zeros(SEED_LEN * (num * length), dtype=int)
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=random.randint(0, MAX_INT))
        convert2seed[blockspergrid, threadsperblock](rng_states, 0, 1, SEED_LEN * (num * length), seed, SEED_MAX_RANGE, 1, -1)
        pws = np.ones(SEED_LEN * (num * length), dtype=int) + 100
        probs = np.zeros(SEED_LEN * (num * length), dtype=np.float64)
        # dr or ndr start = decode_seed(modi_ruleset['nDR'], seeds.pop(0))
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil((num * length) * 4 / threadsperblock[0])
        blockspergrid_y = math.ceil(3 / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        # s_ = time()
        # seed, an_cum, dn_cum, char_cum, decodelength, seedlength, pws, max_length
        sspmreuse_lst = []
        for rei in range(1, MAX_PW_NUM):
            right_dictmp = dte.sspm.set_sspmreuse(rei)
            counts = [c['count'] for c in list(right_dictmp.values())[1:]]  # exclude 'count' from the right_dic
            sspmreuse_lst.append([sum(counts[:i + 1]) for i in range(len(counts))])
        s_ = time()
        for i in range(4):
            if i == 0:  # i: 0 -> drornot
                cumlst = [stripcount(dte.sspm.modi_ruleset)] * 3  # starting from non-zero
            elif i == 1:  # 1 -> portion
                cumlst = [stripcount(dte.sspm.modi_ruleset['nDR'])] * 3
            else:  # i == 2, 3 -> operation (head, tail)
                cumlst = []
                for portion in list(dte.sspm.modi_ruleset['nDR'].keys())[1:]:
                    if portion == 'Head_Tail':
                        if i == 2:
                            cumlst.append(stripcount(dte.sspm.modi_ruleset['nDR']['Head_Tail']['Head']))
                        else:
                            cumlst.append(stripcount(dte.sspm.modi_ruleset['nDR']['Head_Tail']['Tail']))
                    else:
                        cumlst.append(stripcount(dte.sspm.modi_ruleset['nDR'][portion]))
            rule_decode[blockspergrid, threadsperblock](seed, i, 4, SEED_LEN, pws, cumlst[0], cumlst[1],
                                                        cumlst[2], (num * length) * 4, offsetmap,
                                                        np.array(sspmreuse_lst), length, probs)
        print('vaultsgeneration: rule decode using', time() - s_)
        # modifying number and chars decode (# parallel decode)
        blockspergrid_x = math.ceil((num * length) * (SSPM_CHARS_MAX * 2 + 4) / threadsperblock[0])
        blockspergrid_y = math.ceil((len(ALPHABET) - 1) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        s_ = time()
        # seed, an_cum, dn_cum, char_cum, decodelength, seedlength, pws, max_length
        num_chardecode[blockspergrid, threadsperblock](seed, np.array(list(dte.sspm.alphabet_dict['AN'].values())),
                                                       np.array(list(dte.sspm.alphabet_dict['DN'].values())),
                                                       np.array(list(dte.sspm.alphabet_dict['alp'].values())),
                                                       SSPM_CHARS_MAX * 2 + 4, SEED_LEN, pws,
                                                       (num * length) * (SSPM_CHARS_MAX * 2 + 4), probs)
        print('vaultsgeneration: num_chardecode using', time() - s_)
        # assemble pws (indx of list)

        # decode pathnum
        s_ = time()
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil(num * length / threadsperblock[0])
        blockspergrid_y = math.ceil(length / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        pathn_decode[blockspergrid, threadsperblock](seed, pws, SEED_LEN, SEED_LEN - 1, length, num * length, np.concatenate(dte.gi_lst))
        print('vaultsgeneration: pathn_decode using', time() - s_)
        s_ = time()
        problst, vaults = assemble_mspm(0, num, length, pws, dte, probs, True, path1pws, path1probs, None, pastepws, pasteprobs, threshold, possibleidx)
        print('vaultsgeneration: assemble_mspm using', time() - s_)
    return pws, problst, vaults

def assemble_mspm(start, num, length, pws, dte, probs=None, p=False, path1pws=None, path1probs=None,
                  newpws=None, pastepws=None, pasteprobs=None, threshold=None, possibleidx=None):
    """

    :param start: as unit of pwvault
    :param num: as unit of pwvault
    :param length: actual length
    :param pws:
    :param basepw:
    :return:
    """
    pws_tmp = pws[start * length * SEED_LEN: SEED_LEN * (length * (start + num))]
    newpwtmp = newpws[start * (length+1): (length+1) * (start + num)] if not p else None
    pwrules, pathns = getrules(dte, pws_tmp, num, length) # rules to decode mspm
    alp = np.array(list(dte.sspm.alphabet_dict['alp'].keys()))
    an = np.array(list(dte.sspm.alphabet_dict['AN'].keys()))
    dn = np.array(list(dte.sspm.alphabet_dict['DN'].keys()))

    #path1newpws = [None] * (num * (length + 1))
    vaults = []
    problst = []  # length of num*(length+1)

    randomseeds = cp.random.randint(low=0, high=SEED_MAX_RANGE, size=SEED_LEN * (length+1) * num).get()
    pool = Pool(8)
    workers = []
    for vid in range(num):
        #vault, problst, path1newpws = getnewvault(vid, length, pws_tmp, pwrules, pathns, path1newpws, probs, p, path1pws, path1probs, problst, alp, dn, an, pastepws, pasteprobs, threshold, newpwtmp, possibleidx)
        idx_s, idx_e = (vid * length) * SEED_LEN, ((vid+1) * length) * SEED_LEN
        workers.append(pool.apply_async(getnewvault, (0, length, pws_tmp[idx_s:idx_e], pwrules[(vid * length)*4:((vid+1) * length)*4], pathns[(vid * length):((vid+1) * length)], [None] * (length + 1), probs[idx_s:idx_e], p, [], alp, dn, an, pastepws, pasteprobs, threshold, newpwtmp, possibleidx, randomseeds[vid * (length+1) * SEED_LEN: ((vid+1) * (length+1)) * SEED_LEN])))
        #vault, problst_, path1newpws = getnewvault(0, length, pws_tmp[idx_s:idx_e], pwrules[(vid * length)*4:((vid+1) * length)*4], pathns[(vid * length):((vid+1) * length)], [None] * (length + 1), probs[idx_s:idx_e], p, [path1pws[id_] for id_ in idxs_], [path1probs[id_] for id_ in idxs_], [], alp, dn, an, pastepws, pasteprobs, threshold, newpwtmp, possibleidx)
    pool.close()
    pool.join()
    for worker in workers:
        vault, problst_, path1newpws = worker.get()
        vaults.append(vault)
        problst.extend(problst_)
    if p:
        assert len(problst) == num * (length + 1)
        return problst, vaults # both length of num*(length+1)
    return vaults

def getnewvault(vid, length, pws_tmp, pwrules, pathns, path1newpws, probs, p, problst, alp, dn, an, pastepws=None, pasteprobs=None, threshold=None, newpwtmp=None, possibleidx=None, randomseeds=None):
    insertidx = []
    if p:
        newpw, newprob = newpw_prob(randomseeds[:SEED_LEN]) # path1pws[vid], np.log(np.array(dte_global.spm.encode_pw(path1pws[vid])[1])).sum() # path1probs[vid]
        if args.physical:
            newpw, newprob, insertidx = getinsertpastebin(length, pastepws, pasteprobs, threshold, pathns, vid, newpw, newprob, possibleidx)
        path1newpws[vid * (length + 1)] = newpw
        problst.append([newprob, newprob])
    vault = [newpwtmp[vid * (length+1)]] if not p else [newpw]
    for i in range(length):
        pr = probs[(vid * length + i) * SEED_LEN: (vid * length + i + 1) * SEED_LEN] if p else []
        prob = [pr[0]] if p else []
        pw = pws_tmp[(vid * length + i) * SEED_LEN: (vid * length + i + 1) * SEED_LEN]
        pwrule = pwrules[(vid * length + i) * 4: (vid * length + i + 1) * 4]
        if pathns[vid * length + i] == -1 or (newpwtmp is not None and newpwtmp[vid * (length+1) + i + 1] is not None) or i in insertidx:
            # create new password independently
            if p:
                if i not in insertidx:
                    ranid = vid * length + i + 1 # random.randint(0, baselen - 1)
                    newpw, newprob = newpw_prob(randomseeds[(i+1)*SEED_LEN : (i+2)*SEED_LEN])# path1pws[ranid], np.log(np.array(dte_global.spm.encode_pw(path1pws[ranid])[1])).sum() # path1probs[ranid]
                else:
                    newpw, newprob = pastepws[i+1], pasteprobs[i+1]
                path1newpws[vid * (length + 1) + i + 1] = newpw
                problst.append([newprob, newprob + np.log(unreuse_p(i+1))])
            else:
                assert newpwtmp[vid * (length+1) + i + 1] is not None
                newpw = newpwtmp[vid * (length+1) + i + 1]
        else:
            newpw = vault[pathns[vid * length + i]]
            if pwrule[0] == 'DR':
                pass
            else:
                if p:
                    prob.append(pr[1])
                if 'Head' in pwrule[1]:
                    if p:
                        prob.append(pr[3])
                    if 'Delete' in pwrule[2]:
                        if p:
                            prob.append(pr[4])
                        newpw = newpw[int(dn[int(pw[4])]):]
                    if 'dd' in pwrule[2]:
                        if p:
                            prob.extend(pr[5: 6 + int(an[int(pw[5])])])
                        newpw = ''.join(alp[pw[6: 6 + int(an[int(pw[5])])]]) + newpw
                if 'Tail' in pwrule[1]:
                    if p:
                        prob.append(pr[SEED_LEN // 2 + 1])
                    if 'Delete' in pwrule[3]:
                        if p:
                            prob.append(pr[SEED_LEN // 2 + 2])
                        newpw = newpw[:-int(dn[pw[SEED_LEN // 2 + 2]])]
                    if 'dd' in pwrule[3]:
                        if p:
                            prob.extend(pr[SEED_LEN // 2 + 3: SEED_LEN // 2 + 4 + int(an[int(pw[SEED_LEN // 2 + 3])])])
                        newpw = newpw + ''.join(
                            alp[pw[SEED_LEN // 2 + 4: SEED_LEN // 2 + 4 + int(an[int(pw[SEED_LEN // 2 + 3])])]])
            if p:
                ind_pr = np.log(np.array(dte_global.spm.encode_pw(newpw)[1])).sum() if len(newpw) > 5 and len(newpw) < MAX_PW_LENGTH else 0
                problst.append([ind_pr, problst[vid * (length+1):][pathns[vid * length + i]][0] + np.array(prob).sum()])
        vault.append(newpw)
    return vault, problst, path1newpws

def newpw_prob(seed):
    #seed = np.random.RandomState(seed=time_ns()%2**32).randint(low=0, high=SEED_MAX_RANGE, size=SEED_LEN) #np.random.randint(low=0, high=SEED_MAX_RANGE, size=SEED_LEN) #
    newpw_ = dte_global.spm.decode_pw(seed)
    newprob_ = np.log(np.array(dte_global.spm.encode_pw(newpw_)[1])).sum()
    return newpw_, newprob_

def getinsertpastebin(length, pastepws, pasteprobs, threshold, pathns, vid, newpw, newprob, possibleidx_):
    insertidx = []
    if not args.withleak:
        firstorelse = random.choice([0, 1])
        # we dynamically adapt probability according to T_PHY value,
        # bigger the T_PHY, more likely to insert pastebin password (only set this for exp1)
        if firstorelse == 0: # independent encoding option
            if random.random() < min(1. / (length + 1) * np.log10(T_PHY)/np.log10(500), 1.): # T=500 is the minimum
                rid = random.randint(0, len(pastepws) - 1)
                newpw, newprob = pastepws[rid], pasteprobs[rid]
        elif (np.array(pathns[vid * length: (vid + 1) * length]) == -1).sum() > 0:
            cans = []
            for i, v in enumerate(pathns[vid * length:(vid + 1) * length]):
                if v == -1:
                    cans.append(i)
            insertidx_tmp = random.choice(cans)
            if random.random() < min(threshold[insertidx_tmp+1] * np.log10(T_PHY)/np.log10(500), 1.):
                insertidx = insertidx_tmp
    else:
        possibleidx = possibleidx_.copy()
        if 0 in possibleidx: # if leak pw is identical to the first one in vault, then it certainly exists (first always independent encode)
            newpw, newprob = pastepws[0], pasteprobs[0]

        # two parts of possibleidx
        part2 = []
        if length in possibleidx:
            possibleidx.remove(length)
            part2.append(length) # dynamic update part
        for idx in sorted(possibleidx):
            if idx == 0:
                continue
            if random.random() < threshold[idx]:
                insertidx.append(idx - 1)
        for idx in part2: # uncertain part, might clash
            alpha = 1
            if random.random() < threshold[idx]:
                # check pws before
                if len(possibleidx) == 0: # no pws before the same as the last one
                    insertidx.append(idx - 1)
                elif random.random() < float(((1 - np.array([threshold[idx_] for idx_ in possibleidx])) / alpha).prod()):
                    if len(insertidx) > 0:
                        print('clash happens!')
                    else:
                        insertidx.append(idx - 1)
    return newpw, newprob, insertidx

def getrules(dte, pws_tmp, num, length):
    pathns = np.zeros(num * length, dtype=int)
    pwrules = np.array(['Delete-then-add'] * (4 * num * length))
    drornot = list(dte.sspm.modi_ruleset.keys())[1:]
    portion = list(dte.sspm.modi_ruleset['nDR'].keys())[1:]
    portlst = []
    for port in list(dte.sspm.modi_ruleset['nDR'].keys())[1:]:
        if port == 'Head_Tail':
            dictmp = {'Head': list(dte.sspm.modi_ruleset['nDR']['Head_Tail']['Head'].keys())[1:]}
            dictmp['Tail'] = list(dte.sspm.modi_ruleset['nDR']['Head_Tail']['Tail'].keys())[1:]
            portlst.append(dictmp)
        else:
            portlst.append(list(dte.sspm.modi_ruleset['nDR'][port].keys())[1:])
    for i in range(num * length):
        pathns[i] = pws_tmp[i * SEED_LEN + SEED_LEN - 1] - 2
        pwrules[i * 4] = drornot[pws_tmp[i * SEED_LEN + offsetmap[0]]]
        if pwrules[i * 4] == 'DR':
            continue
        portid = pws_tmp[i * SEED_LEN + offsetmap[1]]
        pwrules[i * 4 + 1] = portion[portid]
        if pwrules[i * 4 + 1] == 'Head_Tail':
            pwrules[i * 4 + 2] = portlst[portid]['Head'][pws_tmp[i * SEED_LEN + offsetmap[2]]]
            pwrules[i * 4 + 3] = portlst[portid]['Tail'][pws_tmp[i * SEED_LEN + offsetmap[3]]]
        else:
            pwrules[i * 4 + 2] = portlst[portid][pws_tmp[i * SEED_LEN + offsetmap[2]]]
            pwrules[i * 4 + 3] = portlst[portid][pws_tmp[i * SEED_LEN + offsetmap[3]]]
    return pwrules, pathns

def getrules_lite(sspm, pws_tmp, num, length):
    pathns = np.zeros(num * length, dtype=int)
    pwrules = np.array(['Delete-then-add'] * (4 * num * length))
    drornot = list(sspm.modi_ruleset.keys())[1:]
    portion = list(sspm.modi_ruleset['nDR'].keys())[1:]
    portlst = []
    for port in list(sspm.modi_ruleset['nDR'].keys())[1:]:
        if port == 'Head_Tail':
            dictmp = {'Head': list(sspm.modi_ruleset['nDR']['Head_Tail']['Head'].keys())[1:]}
            dictmp['Tail'] = list(sspm.modi_ruleset['nDR']['Head_Tail']['Tail'].keys())[1:]
            portlst.append(dictmp)
        else:
            portlst.append(list(sspm.modi_ruleset['nDR'][port].keys())[1:])
    for i in range(num * length):
        pathns[i] = pws_tmp[i * SEED_LEN + SEED_LEN - 1] - 2
        pwrules[i * 4] = drornot[pws_tmp[i * SEED_LEN + offsetmap[0]]]
        if pwrules[i * 4] == 'DR':
            continue
        portid = pws_tmp[i * SEED_LEN + offsetmap[1]]
        pwrules[i * 4 + 1] = portion[portid]
        if pwrules[i * 4 + 1] == 'Head_Tail':
            pwrules[i * 4 + 2] = portlst[portid]['Head'][pws_tmp[i * SEED_LEN + offsetmap[2]]]
            pwrules[i * 4 + 3] = portlst[portid]['Tail'][pws_tmp[i * SEED_LEN + offsetmap[3]]]
        else:
            pwrules[i * 4 + 2] = portlst[portid][pws_tmp[i * SEED_LEN + offsetmap[2]]]
            pwrules[i * 4 + 3] = portlst[portid][pws_tmp[i * SEED_LEN + offsetmap[3]]]
    return pwrules, pathns

def check_logical(idx_gt_shuffled, loginum, vault_size, gpuid, seed_=random.randint(0, MAX_INT), reshuidx_whole=None, depth=None):
    itvsize = Nitv if args.fixeditv else vault_size
    padded_size = int(math.ceil(vault_size / itvsize) * itvsize) if args.fixeditv and args.fixeditvmode == 0 else vault_size
    loginum_tmp = 1 # loginum if not args.fixeditv else 1
    with cuda.gpus[gpuid]:
        # get reshuffle index
        threadsperblock = 32
        blockspergrid = (loginum_tmp * (PIN_SAMPLE if int(args.pinlength) <=6 else 10**6) * padded_size + threadsperblock - 1) // threadsperblock
        reshuidx = np.zeros(loginum_tmp * PIN_SAMPLE * padded_size, dtype=np.uint8)
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=seed_)
        convert2seed_shuffle[blockspergrid, threadsperblock](rng_states, 0, 1, loginum_tmp*PIN_SAMPLE*padded_size, reshuidx, itvsize, padded_size//itvsize, padded_size if args.fixeditv and args.fixeditvmode==1 else -1)
        if depth is not None and depth > 0:
            assert reshuidx_whole is not None
            # change reshuidx to current usage: from shape (loginum_tmp * PIN_SAMPLE * (padded_size+depth)) to (loginum_tmp * PIN_SAMPLE * padded_size)
            reshuidx = reshuidx.reshape(loginum_tmp * PIN_SAMPLE, -1)[:, padded_size//itvsize*itvsize:]
            reshuidx_identical = reshuidx_whole.reshape(loginum_tmp * PIN_SAMPLE, -1)[:, :padded_size//itvsize*itvsize]

            # concatenate reshuidx and reshuidx_identical alone axis=1 and reshape to (loginum_tmp * PIN_SAMPLE * padded_size)
            reshuidx = np.concatenate((reshuidx_identical, reshuidx), axis=1).reshape(-1)

        # recover shuffle index
        blockspergrid = (loginum_tmp * PIN_SAMPLE + threadsperblock - 1) // threadsperblock
        reshuedidx = cp.tile(cp.asarray(idx_gt_shuffled, dtype=np.uint8), loginum_tmp * PIN_SAMPLE)
        recovershuffle[blockspergrid, threadsperblock](reshuidx, reshuedidx, padded_size)

    if args.fixeditv:
        if depth is not None:
            return reshuedidx[None], reshuidx
        return reshuedidx[None]

    return reshuedidx[None] #reshuedidx.reshape(loginum, padded_size*PIN_SAMPLE)

def check_pin_batch(reshuffled_list, vaults, DVs, real=False, gpuid=0): # 'vaults' is a list (batch) of vault, 'DVs' is a list (batch) of DV
    cp.cuda.Device(gpuid).use()
    if real:
        assert len(vaults) == 1 and len(DVs) == 1
    bsz = len(vaults)
    padded_size = int(math.ceil(len(vaults[0]) / Nitv) * Nitv) if args.fixeditv and args.fixeditvmode == 0 else len(vaults[0]) # bsz
    if not args.intersection:
        reshuffled_list_t = reshuffled_list # cp.array(reshuffled_list) #
    else:
        reshuffled_list_t = reshuffled_list[0] #cp.array(reshuffled_list[0]), cp.array(reshuffled_list[1]) #
    dvts, realleakpwid = zip(*[DV.create_dv(vault) for DV, vault in zip(DVs, vaults)])
    dvts, realleakpwid = list(dvts), set(list(realleakpwid)) # digitvault: {k: pwid, v: edit distance from leakpw}
    assert len(realleakpwid) == 1 # same leaked pw for all vaults
    realleakpwid = list(realleakpwid)[0]
    dv_pwids = [DV.dv2pwid(dvt) for DV, dvt in zip(DVs, dvts)]
    assert all([realleakpwid in dvspwids for dvspwids in dv_pwids])
    dv_pwids = np.array(dv_pwids) # bsz x len(vaults[0])

    ### set up variables
    N_ol_tmp = 4 if (not real and len(vaults[0]) > 20) else (len(vaults[0]) - 1) # if (not real and len(vaults[0]) > 20) else (len(vaults[0]) - 1)
    if not real and len(vaults[0]) > 50:
        N_ol_tmp = 3

    size0 = N_ol_tmp + 2 if args.intersection else N_ol_tmp + 1
    meta_idxs_totry = [online_verify(dvt, DV) for dvt, DV in zip(dvts, DVs)] # metadata idx list (in descending order of password distance)
    # result layout: top 1 (or 2 rows if intersection) is the prerequisit of candidate guess, the next N_ol rows are online verification result (first, second, ...) of each candidate guess
    result = cp.zeros((size0, PIN_SAMPLE*bsz), dtype=bool)
    if N_ol_tmp == (len(vaults[0]) - 1) and N_ol_tmp > 50: # real and large vaults (pb200)
        meta_pw_id = cp.zeros((2, PIN_SAMPLE*N_ol_tmp*bsz), dtype=np.uint8) # uint8 range 0 to 255; int16 range -32,768 to 32,767
    else:
        meta_pw_id = cp.zeros((2, PIN_SAMPLE*N_ol_tmp*bsz), dtype=np.int32) # int32 range -2,147,483,648 to 2,147,483,647


    threadsperblock = (256, 4)
    blockspergrid_y = math.ceil(size0 / threadsperblock[1])
    blockspergrid_x = math.ceil((PIN_SAMPLE if int(args.pinlength) <= 6 else 10**6) * bsz / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    checkeachvault[blockspergrid, threadsperblock](result, reshuffled_list_t, len(vaults[0]), padded_size, dv_pwids, np.array(DVs[0].dv2pwid(DVs[0].digit_realvault)), np.array(meta_idxs_totry), np.array([DVs[0].leakmetaid]), meta_pw_id, N_ol_tmp, bsz)

    if args.intersection:
        ### write checking results to second row of variable "result"
        ### ==> case1 (password reuse does not have any mitigation): for vanilla shuffling, attacker requires
        ###            mappings across version are identical except the newly added (factorial degradation)
        ### ==> case2 (password reuse greatly mitigate degradation): for fixed interval shuffling with group randomization, attacker requires |V^t \ V^{t-1}|==1
        # below two are only used in non fixed interval shuffling (group randmization in construction 2 limits these reveals)
        for vg in range(len(reshuffled_list) - 1):
            reshuffled_list_t, reshuffled_list_t_1 = reshuffled_list[len(reshuffled_list)-vg-2], reshuffled_list[len(reshuffled_list)-vg-1]
            size_t, size_t_1 = len(reshuffled_list_t) // PIN_SAMPLE, len(reshuffled_list_t_1) // PIN_SAMPLE
            blockspergrid_y = math.ceil(size_t / threadsperblock[1])
            blockspergrid_x = math.ceil((PIN_SAMPLE if int(args.pinlength) <= 6 else 10**6) * bsz / threadsperblock[0])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            #ts = time()
            if args.fixeditv:
                checkincrement[blockspergrid, threadsperblock](result, reshuffled_list_t, np.ascontiguousarray(dv_pwids[:, :size_t]), size_t, reshuffled_list_t_1, np.ascontiguousarray(dv_pwids[:, :size_t_1]), size_t_1, bsz)
            else:
                checkincrement[blockspergrid, threadsperblock](result, reshuffled_list_t, np.tile(np.arange(size_t),(bsz,1)), size_t, reshuffled_list_t_1, np.tile(np.arange(size_t_1),(bsz,1)), size_t_1, bsz)

        if real:
            for i in range(size0):
                result[i, 0] = 1
        return result, meta_pw_id#.reshape(-1)

def online_verify(digitvault, DV):
    """
    online_verify sort the distance in descending order and exclude the leakedmetaid-th password,
    return the password id with distance down to the smallest
    :param digitvault: a dictionary
    :param leakmetaid:
    :return: metaid to be tried in descending order (index of the pwid in the key list)
    """
    meta_idxs = np.argsort(np.array([-1*dis for dis in DV.dv2dis(digitvault)])) # sort in descending order referring to the distance
    return meta_idxs #metaids_totry