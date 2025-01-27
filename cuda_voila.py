import math

import numba
import hashlib

import numpy as np

from MSPM.mspm_config import SSPM_CHARS_MAX, SEED_LEN, PIN_SAMPLE, N_ol
from numba import cuda, int32
from numba.cuda.random import xoroshiro128p_uniform_float32
from opts import opts
import random as orig_random
#cuda.select_device(1)

@cuda.jit
def gpu_decode(seed, offset, cum, idx):
    """

    :param seed:
    :param offset:
    :param cum: starting from nonzero
    :param idx: 1 element array
    :return:
    """
    pos = cuda.grid(1)
    if pos < (cum.size-1):  # Check array boundaries
        if (cum[pos]-seed) <= 0 and (cum[pos+1]-seed) > 0:
            idx[0] = pos + offset + 1
        elif (cum[pos]-seed) > 0 and pos == 0:
            idx[0] = pos + offset

@cuda.jit
def ngramip_decodepw(seed, ip_cum, seedlength, pws, markov_ids, no_ip_ngrams, probs):
    """
        only need to decode cps                                     <- seedlength-2 ->
    :param seed: seedlength*num [seed_l, ip_idx, s1, s2, s3, -1, -1, ..., seed_l, ip_idx, s1, s2, s3, s4, -1, ...]
                                 *len_ip*, **ip**, ------cp----------
                                * (1): known before kernel    - (0): known after kernel
    :param cp_cum: alpha_length ** ngramsize * numblocks
    :param markov_ids: num
    :return: pws : same length with seed, indx for each 'gram' instead.
    """
    x, y = cuda.grid(2) # x => [0, num); y => [0, no_ip_ngrams)
    if x < pws.size//seedlength:
        idx_ = int(x * seedlength + 1)
        offset = int(markov_ids[x] * no_ip_ngrams)
        fill = ip_cum[offset + no_ip_ngrams - 1]
        seed_ip = seed[idx_] % fill
        if (ip_cum[offset+y] - seed_ip) <= 0 and (ip_cum[offset+y+1] - seed_ip) > 0:
            pws[idx_] = int(offset + y + 1)
            probs[idx_] = math.log((ip_cum[offset+y+1] - ip_cum[offset+y]) / fill)
        elif (ip_cum[offset+y] - seed_ip) > 0 and y == 0:
            pws[idx_] = offset
            probs[idx_] = math.log(ip_cum[offset] / fill)

@cuda.jit
def ngramcp_decodepw(seed, cp_cum, seedlength, pws, alph_len, markov_ids, no_cp_ngrams, step_id, ngram, probs):
    """
        only need to decode cps                                     <- seedlength-2 ->
    :param seed: seedlength*num [seed_l, ip_idx, s1, s2, s3, -1, -1, ..., seed_l, ip_idx, s1, s2, s3, s4, -1, ...]
                                 *len_ip*, **ip**, ------cp----------
                                * (1): known before kernel    - (0): known after kernel
    :param cp_cum: alpha_length ** ngramsize * numblocks
    :param markov_ids: num
    :return: pws : same length with seed, indx for each 'gram' instead.
    """
    x, y = cuda.grid(2) # x => [0,seedlength); y => [0, alpha_len)
    if x < pws.size//seedlength and y < (alph_len-1):
        idx_ = int(x * seedlength + step_id) # of pws and seed
        offset = int(markov_ids[x] * no_cp_ngrams)
        start = int(alph_len * (pws[idx_-1] % (alph_len ** (ngram-1)))) # without offsets
        fill = cp_cum[offset + start + alph_len - 1]
        seed_cp = seed[idx_] % fill
        if (cp_cum[offset + start + y] - seed_cp) <= 0 and (cp_cum[offset + start + y + 1] - seed_cp) > 0:
            if (step_id - 1 < pws[x * seedlength]):
                pws[idx_] = offset + start + int(y) + 1
                probs[idx_] = math.log((cp_cum[offset + start + y + 1] - cp_cum[offset + start + y]) / fill)
        elif (cp_cum[offset + start + y] - seed_cp) > 0 and y == 0:
            if (step_id - 1 < pws[x * seedlength]):
                pws[idx_] = offset + start
                probs[idx_] = math.log(cp_cum[offset + start] / fill)

@cuda.jit
def rule_decode(seed, i, decodelength, seedlength, pws, cum0, cum1, cum2, max_length, offsetmap, sspmreuse_lst, real_len, probs):
    # pws  0->drornot, 1->portion, 3->head rule, 26->tail rule    offsetmap=>[0,1,3,26]
    x, y = cuda.grid(2)  # x => [0, max_length); y => [0, alph_len)
    if x < max_length and x % decodelength == i: # seedlength * totalseeds
        idx_ = int((x // decodelength) * seedlength + offsetmap[i]) # rule bits on pws
        if i == 0:
            cum = sspmreuse_lst[(x // decodelength) % real_len] # dr or not
        elif i == 1:
            cum = cum0 # portion
        else:
            idxportion = pws[int((x // decodelength) * seedlength + offsetmap[1])]
            if idxportion == 0:
                cum = cum0
            elif idxportion == 1:
                cum = cum1
            else:
                cum = cum2
        seed_ip = seed[idx_] % cum[-1]
        if y < (cum.size - 1) and (cum[y] - seed_ip) <= 0 and (cum[y + 1] - seed_ip) > 0:
            pws[idx_] = y + 1 # index excluding 'count'
            probs[idx_] = math.log((cum[y+1] - cum[y]) / cum[-1])
        elif y == 0 and (cum[y] - seed_ip) > 0:
            pws[idx_] = 0 # index excluding 'count'
            probs[idx_] = math.log(cum[0] / cum[-1])

@cuda.jit
def num_chardecode(seed, an_cum, dn_cum, char_cum, decodelength, seedlength, pws, max_length, probs):
    x, y = cuda.grid(2)  # x => [0, max_length); y => [0, alph_len)
    if x < max_length: # seedlength * totalseeds
        if x % decodelength == 0 or x % decodelength == (decodelength // 2): # delete num decode
            cum = dn_cum
        elif x % decodelength == 1 or x % decodelength == (decodelength // 2 + 1): # add num decode
            cum = an_cum
        else: # char decode
            cum = char_cum
        idx_ = int((x // decodelength) * seedlength + x % decodelength + \
                   (4 if x % decodelength < (decodelength//2) else (SEED_LEN//2-SSPM_CHARS_MAX-2+2)))
        seed_ip = seed[idx_] % cum[-1]
        if y < (cum.size - 1) and (cum[y] - seed_ip) <= 0 and (cum[y + 1] - seed_ip) > 0:
            pws[idx_] = y + 1
            probs[idx_] = math.log((cum[y + 1] - cum[y]) / cum[-1])
        elif y == 0 and (cum[y] - seed_ip) > 0:
            pws[idx_] = 0
            probs[idx_] = math.log(cum[0] / cum[-1])

@cuda.jit
def pathn_decode(seed, pws, seedlength, offset, length, max_length, cumlst):
    # note that pathtop is only 1 index larger than current encoding path,
    # as controlling decoding into the paths after the current
    x, y = cuda.grid(2)  # x => [0, num*length); y => [0, length)
    if x < max_length:
        pathtop = x % length + 1
        cum = cumlst[int(pathtop*(pathtop+3)/2) : int(pathtop*(pathtop+3)/2)+pathtop+2][1:] # len(cum) = pathtop + 2
        idx_ = int(x * seedlength + offset) # rule bits on pws
        seed_ip = seed[idx_] % cum[-1]
        if y < (cum.size - 1) and (cum[y] - seed_ip) <= 0 and (cum[y + 1] - seed_ip) > 0:
            pws[idx_] = y + 2
        elif y == 0 and (cum[y] - seed_ip) > 0:
            pws[idx_] = 1

@cuda.jit
def convert2seed(rng_states, rand_val, fill, n, seeds, seed_max_value, dividenum, fixeditv_vaultsize):
    # fixeditv_vaultsize is parameter used for fixed interval shuffling without padding mode (args.fixeditv==True & args.fixeditvmode==1)
    # fixeditv_vaultsize == -1 is symbolized for None, and run the algorithm for all cases except fixed interval shuffling without padding mode
    pos = cuda.grid(1)
    if pos < n:  # Check array boundaries
        rand01 = xoroshiro128p_uniform_float32(rng_states, pos) - 1e-8 # 1e-8 to make sure rand01 is not 1
        if rand01 < 0:
            rand01 = 0
        if fixeditv_vaultsize == -1: # for all cases except fixed interval shuffling without padding mode
            if dividenum == 1:
                seeds[pos] = rand_val + int(rand01 * ((seed_max_value - rand_val) // fill) * fill)
            else:
                seeds[pos] = rand_val + int(rand01 * ((seed_max_value - rand_val) // fill) * fill +
                             ((pos%(dividenum*seed_max_value))//seed_max_value)*seed_max_value)
        else:
            if (pos % fixeditv_vaultsize) // seed_max_value == dividenum:# and (pos % fixeditv_vaultsize) % seed_max_value != 0:
                adapted_seed_max = fixeditv_vaultsize % seed_max_value
            else:
                adapted_seed_max = seed_max_value # last interval with different seed_max_value
            seeds[pos] = rand_val + int(rand01 * ((adapted_seed_max - rand_val) // fill) * fill +
                            ((pos % fixeditv_vaultsize) // seed_max_value) * seed_max_value)

@cuda.jit
def  convert2seed_shuffle(rng_states, rand_val, fill, n, seeds, seed_max_value, dividenum, fixeditv_vaultsize):
    # fixeditv_vaultsize is parameter used for fixed interval shuffling without padding mode (args.fixeditv==True & args.fixeditvmode==1)
    # fixeditv_vaultsize == -1 is symbolized for None, and run the algorithm for all cases except fixed interval shuffling without padding mode
    x = cuda.grid(1)
    if x < n:
        stridex = cuda.gridsize(1)
        for pos in range(x, n, stridex):
            rand01 = xoroshiro128p_uniform_float32(rng_states, x) - 1e-8 # 1e-8 to make sure rand01 is not 1
            if rand01 < 0:
                rand01 = 0
            if fixeditv_vaultsize == -1:
                if dividenum == 1:
                    seeds[pos] = rand_val + int(rand01 * ((seed_max_value - rand_val) // fill) * fill)
                else:
                    seeds[pos] = rand_val + int(rand01 * ((seed_max_value - rand_val) // fill) * fill +
                                 ((pos%(dividenum*seed_max_value))//seed_max_value)*seed_max_value)
            else:
                if (pos % fixeditv_vaultsize) // seed_max_value == dividenum:# and (pos % fixeditv_vaultsize) % seed_max_value != 0:
                    adapted_seed_max = fixeditv_vaultsize % seed_max_value
                else:
                    adapted_seed_max = seed_max_value # last interval with different seed_max_value
                seeds[pos] = rand_val + int(rand01 * ((adapted_seed_max - rand_val) // fill) * fill +
                                ((pos % fixeditv_vaultsize) // seed_max_value) * seed_max_value)

@cuda.jit
def recovershuffle(reshu_idx, idx_gt_shuffled, vaultsize):
    """

    :param reshu_idx: T_PHY * 10**(PIN_LENGTH+1) * PADDED_TO
    :param reshued_idx: T_PHY * 10**(PIN_LENGTH+1) * PADDED_TO
    :return:
    """
    pos = cuda.grid(1)
    if pos < idx_gt_shuffled.size//vaultsize:  # Check array boundaries
        for i in range(vaultsize):
            idx_gt_shuffled[i+pos*vaultsize], idx_gt_shuffled[reshu_idx[i+pos*vaultsize]+pos*vaultsize] = \
                idx_gt_shuffled[reshu_idx[i+pos*vaultsize]+pos*vaultsize], idx_gt_shuffled[i+pos*vaultsize]

@cuda.jit
def checkeachvault(result, reshuffled_list, vaultsize, padded_size, digitvault_pwids, realdigitvault_pwids, leakmetaids_idx, fixed_metaid, meta_pw_id, N_ol_tmp, bsz):
    # result = np.zeros((size0, bsz*PIN_SAMPLE), dtype=np.int)
    # meta_pw_id = np.zeros((2, bsz*PIN_SAMPLE*N_ol_tmp), dtype=np.int)
    posy, x = cuda.grid(2) # x => [0, size0-1); y => [0, bsz*PIN_SAMPLE)
    if x < result.shape[0] and posy < result.shape[1]: # result: size0 x PIN_SAMPLE
        stridey, _ = cuda.gridsize(2)
        for y in range(posy, result.shape[1], stridey):
            ith_batch = y // PIN_SAMPLE
            y_inbatch = y % PIN_SAMPLE
            if x == 0: # fixed metaid to match password (attacker tries specific metaid)
                for reshuffidx in range(vaultsize):
                    if reshuffled_list[reshuffidx + y_inbatch * padded_size] - fixed_metaid[0] == 0:
                        break
                if digitvault_pwids[ith_batch, reshuffidx] == realdigitvault_pwids[fixed_metaid[0]]: # TODO: inferred metaid has to be online verified to determine if it is correct
                    result[x, y] = 1
            elif x == 1: # inter metaid match pwid
                result[x, y] = 1
                #pass # intersection needs 1 meta-pw id increment which will be checked later specifically (seems numba cuda.jit does not support pass)
            elif x < result.shape[0]: # flexible metaid to match pwid (attack focuses on password-determined metaid)
                increment, findmetaid_thisshot = 0, True
                for ia in range(leakmetaids_idx.shape[1]):
                    for ib in range(fixed_metaid.size):
                        if reshuffled_list[y_inbatch * padded_size + leakmetaids_idx[ith_batch, ia]] == fixed_metaid[ib]:
                            findmetaid_thisshot = False
                    if not findmetaid_thisshot:
                        findmetaid_thisshot = True
                        continue
                    increment += 1
                    if increment == x-1:
                        break
                metaid_idx_ = leakmetaids_idx[ith_batch, ia]
                meta_pw_id[0, y * N_ol_tmp + x - 2] = reshuffled_list[y_inbatch*padded_size + metaid_idx_] # metaid to try
                meta_pw_id[1, y * N_ol_tmp + x - 2] = digitvault_pwids[ith_batch, metaid_idx_] # pwid to try
                if realdigitvault_pwids[reshuffled_list[y_inbatch * padded_size + metaid_idx_]] == digitvault_pwids[ith_batch, metaid_idx_]:
                    result[x, y] = 1

@cuda.jit
def checkincrement(result, reshuffled_list_t, dvt_pwids, dvt_size, reshuffled_list_t_1, dvt_1_pwids, dvt_1_size, bsz):
    # result = np.zeros((size0, PIN_SAMPLE), dtype=np.int) # the second row should be all 1s (intersection prerequisite)
    # reshuffled_list_t: PIN_SAMPLE x dvt_size
    posy, x = cuda.grid(2)  # x => [0, dvt_size); y => [0, PIN_SAMPLE)
    if x < dvt_size and posy < result.shape[1]:  # result: size0 x PIN_SAMPLE
        stridey, _ = cuda.gridsize(2)
        for y in range(posy, result.shape[1], stridey):
            # i-th mapping (metaid-pwid) in dvt_1 and j-th mapping in dvt
            j = x
            y_inbatch = y % PIN_SAMPLE
            ith_batch = y // PIN_SAMPLE
            for i in range(dvt_1_size):
                # check if any mapping in dvt_1 is not in dvt, if it did, mark result[1, y] = 0
                if reshuffled_list_t_1[y_inbatch * dvt_1_size + i] == reshuffled_list_t[y_inbatch * dvt_size + j] and dvt_1_pwids[ith_batch, i] != dvt_pwids[ith_batch, j]:
                    result[1, y] = 0

@cuda.jit
def mark_totry(result_pre, result_ol, nol, nol_max, try_indicator):
    # result_pre: PIN_SAMPLE*bsz; result_ol: nol x (PIN_SAMPLE*bsz)
    # meta_pw_id: 2 x (PIN_SAMPLE*nol_max*bsz); try_indicator (all ones as init): PIN_SAMPLE*nol_max*bsz
    x, y = cuda.grid(2) # x: [0, PIN_SAMPLE*nol*bsz), y: [0, nol)
    if x < try_indicator.size // nol_max * nol:
        # the latter one will be tried if its formers all pass
        if x % nol == 0 and result_pre[x//nol] == 0:
            try_indicator[x // nol * nol_max + x % nol] = 0
        elif y < (x % nol):
            if result_pre[x//nol] == 0 or result_ol[y, x // nol] == 0:
                try_indicator[x // nol * nol_max + x % nol] = 0

@cuda.jit
def tryrectify_previousfindreal(try_idxs, meta_pw_id, cracked_metaid, mask):
    # try_idxs: num of try idx marked
    # meta_pw_id: 2 x (PIN_SAMPLE*Nol_max*batch)
    # cracked_metaid: num of cracked metaid
    x, y = cuda.grid(2) # x: [0, num of try idx marked), y: [0, num of cracked metaid)
    if x < try_idxs.size and y < cracked_metaid.size:
        if meta_pw_id[0, try_idxs[x]] == cracked_metaid[y]:
            mask[try_idxs[x]] = 1

@cuda.jit
def tryrectify_nowfindreal(try_idxs, meta_pw_id, nowcracked_metaid, nowcracked_metaid_idx, mask):
    # try_idxs: num of try idx marked
    # meta_pw_id: 2 x (PIN_SAMPLE*Nol_max*batch)
    # nowcracked_metaid: num of now cracked metaid
    # nowcracked_metaid_idx: num of now cracked metaid
    x, y = cuda.grid(2) # x: [0, num of try idx marked), y: [0, num of now cracked metaid)
    if x < try_idxs.size and y < nowcracked_metaid.size:
        if meta_pw_id[0, try_idxs[x]] == nowcracked_metaid[y] and try_idxs[x] > nowcracked_metaid_idx[y]:
            mask[try_idxs[x]] = 1

@cuda.jit
def tryrectify_previouschecked(try_idxs, meta_pw_id, checked_pairs, mask):
    # try_idxs: num of try idx marked
    # meta_pw_id: 2 x (PIN_SAMPLE*Nol_max*batch)
    # checked_pairs: 3 x (num of checked pairs)
    x, y = cuda.grid(2) # x: [0, num of try idx marked), y: [0, num of checked pairs)
    if x < try_idxs.size and y < checked_pairs.shape[1]:
        if meta_pw_id[0, try_idxs[x]] == checked_pairs[0, y] and meta_pw_id[1, try_idxs[x]] == checked_pairs[1, y]:
            mask[try_idxs[x]] = 1