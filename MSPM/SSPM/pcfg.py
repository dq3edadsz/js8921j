import pylcs
from collections import OrderedDict
from tqdm import tqdm
import json
import orjson
from MSPM.mspm_config import *
from MSPM.utils import random
import random as orig_random
import os
import numpy as np
import math
import collections
from opts import opts
args = opts().parse()

class RulePCFGCreator:

    def __init__(self, args):
        self.modi_ruleset = OrderedDict()
        self.alphabet = ALPHABET
        self.alphabet_len = len(self.alphabet)
        self.alphabet_dict =self._init_alphdic()
        if args.sspmdata == 'pastebin':
            self.train_vault = self.load_fold()
        elif args.sspmdata == 'rockyou':
            self.train_vault = self.load_rock() # dual lst => [0]:pws  [1]:cnts
        if args.pretrained:
            self.load()

    def _init_alphdic(self):
        dic = OrderedDict()
        dig = [str(d) for d in list(np.arange(0, SSPM_CHARS_MAX))]
        for type in ['AN', 'DN']:
            dic.update({type: OrderedDict().fromkeys(dig)})
            for k in dic[type]:
                dic[type][k] = 1
        dic.update({'alp': OrderedDict().fromkeys(self.alphabet)})
        for char in dic['alp']:
            dic['alp'][char] = 1
        return dic

    def load_rock(self):
        train_vault = []
        train_cnt = []
        with open(ROCKY_PATH, encoding='latin-1') as input_file:
            for line in input_file:
                L = line.lstrip().strip('\r\n').split(' ')
                if len(L) != 2:
                    continue
                cnt, word = int(L[0]), L[1]
                if not_in_alphabet(word) or lencheck(word):  # Important to prevent generating "passwor", or "iloveyo", or "babygir"
                    continue
                train_vault.append(word)
                train_cnt.append(cnt)
        return train_vault, train_cnt

    def load_fold(self):
        self.train_vault = {}
        flst = os.listdir(PASTB_PATH + args.exp_pastebinsuffix)
        if not args.multi_train:
            for id in TEST_FOLD:
                for fname in flst:
                    if str(id) in fname:
                        print('train w/o test file:', fname)
                        f = open(os.path.join(PASTB_PATH + args.exp_pastebinsuffix, fname))
                        vaults_tmp = json.load(f)
                        self.train_vault[id] = vaults_tmp
                        continue # comment the code for training all
        else:
            for rn in range(1):
                for id in TEST_FOLD:
                    with open(os.path.join(PASTB_PATH + args.exp_pastebinsuffix, 'fold_' + str(rn) + '_' + str(id) + '.json'), "rb") as f:
                        vaults_tmp = orjson.loads(f.read())
                    self.train_vault[str(rn) + '_' + str(id)] = vaults_tmp
        return self.train_vault

    def encode_pw(self, src_pw, targ_pw, ith):
        """

        :param src_pw:
        :param targ_pw:
        :param ith: i is set for target password (when encoding target password i+1)
        :return:
        """
        code_path = self.reuse_model(src_pw, targ_pw)
        #print('code path:', code_path)
        if len(code_path) == 0:
            return [], []
        h = code_path.pop(0)
        if h == 'DR':
            seed, prob = self.encode_dr('DR', ith) # return both lists
        else:
            assert h == 'nDR'
            seed_s, prob_s = self.encode_dr('nDR', ith)
            if 'Head_Tail' in code_path[0]:
                seed_h, prob_h = self.encode_path(code_path[1], self.modi_ruleset['nDR']['Head_Tail'])
                seed_t, prob_t = self.encode_path(code_path[2], self.modi_ruleset['nDR']['Head_Tail'])
                tot = self.modi_ruleset['nDR']['count']
                i = list(self.modi_ruleset['nDR'].keys()).index('Head_Tail')
                l_pt = sum([c['count'] for c in list(self.modi_ruleset['nDR'].values())[1:i]])
                r_pt = l_pt + self.modi_ruleset['nDR']['Head_Tail']['count'] - 1
                del seed_h[0], seed_t[0]
                seed = [self.convert2seed(random.randint(l_pt, r_pt), tot)]
                seed.extend(seed_h)
                seed.extend(seed_t)
                prob = [self.modi_ruleset['nDR']['Head_Tail']['count'] / self.modi_ruleset['nDR']['count']]
                prob.extend(prob_h)
                prob.extend(prob_t)
            else:
                assert 'Head' in code_path[0] or 'Tail' in code_path[0]
                seed, prob = self.encode_path(code_path[0], self.modi_ruleset['nDR'])
            seed.insert(0, seed_s.pop())
            prob.extend(prob_s)
        extra = SEED_LEN - len(seed) - 1 # -1 is to reserve space for path seed
        seed.extend(self.convert2seed(0, 1, extra))
        return seed, prob

    def encode_pwtest(self, src_pw, targ_pw, ith):
        code_path = self.reuse_model(src_pw, targ_pw)
        #print('code path:', code_path)
        seed_array = np.ones(SEED_LEN, dtype=int)
        if len(code_path) == 0:
            return np.array([]), np.array([])
        h = code_path.pop(0)
        if h == 'DR':
            seed, prob = self.encode_dr('DR', ith)  # return both lists
            seed_array[0] = seed[0]
        else:
            assert h == 'nDR'
            seed_s, prob_s = self.encode_dr('nDR', ith)
            seed_array[0] = seed_s[0]
            if 'Head_Tail' in code_path[0]:
                seed_h, prob_h = self.encode_path(code_path[1], self.modi_ruleset['nDR']['Head_Tail'])
                seed_t, prob_t = self.encode_path(code_path[2], self.modi_ruleset['nDR']['Head_Tail'])
                del seed_h[0], seed_t[0]
                tot = self.modi_ruleset['nDR']['count']
                i = list(self.modi_ruleset['nDR'].keys()).index('Head_Tail')
                l_pt = sum([c['count'] for c in list(self.modi_ruleset['nDR'].values())[1:i]])
                r_pt = l_pt + self.modi_ruleset['nDR']['Head_Tail']['count'] - 1
                seed = [self.convert2seed(random.randint(l_pt, r_pt), tot)]
                seed_array[1] = seed[0]
                seed_array[3] = seed_h.pop(0)
                if 'Delete-then-add' in code_path[1]:
                    seed_array[4] = seed_h.pop(0)
                    seed_array[5] = seed_h.pop(0)
                else:
                    seed_array[4] = seed_h.pop(0)
                    seed_array[5] = seed_array[4]
                for i_ in range(6, 6 + len(seed_h)):
                    seed_array[i_] = seed_h.pop(0)

                seed_array[SEED_LEN//2+1] = seed_t.pop(0)
                if 'Delete-then-add' in code_path[2]:
                    seed_array[SEED_LEN//2+2] = seed_t.pop(0)
                    seed_array[SEED_LEN//2+3] = seed_t.pop(0)
                else:
                    seed_array[SEED_LEN//2+2] = seed_t.pop(0)
                    seed_array[SEED_LEN//2+3] = seed_array[SEED_LEN//2+2]
                for i_ in range(SEED_LEN//2+4, SEED_LEN//2+4 + len(seed_t)):
                    seed_array[i_] = seed_t.pop(0)
                #seed.extend(seed_h)
                #seed.extend(seed_t)
                prob = [self.modi_ruleset['nDR']['Head_Tail']['count'] / self.modi_ruleset['nDR']['count']]
                prob.extend(prob_h)
                prob.extend(prob_t)
            else:
                assert 'Head' in code_path[0] or 'Tail' in code_path[0]
                seed, prob = self.encode_path(code_path[0], self.modi_ruleset['nDR'])
                seed_array[1] = seed.pop(0)
                if 'Head' in code_path[0]:
                    seed_array[3] = seed.pop(0)
                    if 'Delete-then-add' in code_path[0]:
                        seed_array[4] = seed.pop(0)
                        seed_array[5] = seed.pop(0)
                    else:
                        seed_array[4] = seed.pop(0)
                        seed_array[5] = seed_array[4]
                    for i_ in range(6, 6 + len(seed)):
                        seed_array[i_] = seed.pop(0)
                elif 'Tail' in code_path[0]:
                    seed_array[SEED_LEN//2+1] = seed.pop(0)
                    if 'Delete-then-add' in code_path[0]:
                        seed_array[SEED_LEN//2+2] = seed.pop(0)
                        seed_array[SEED_LEN//2+3] = seed.pop(0)
                    else:
                        seed_array[SEED_LEN//2+2] = seed.pop(0)
                        seed_array[SEED_LEN//2+3] = seed_array[SEED_LEN//2+2]
                    for i_ in range(SEED_LEN//2+4, SEED_LEN//2+4 + len(seed)):
                        seed_array[i_] = seed.pop(0)
                    # seed.extend(seed_h)
                    # seed.extend(seed_t)
                    prob = [self.modi_ruleset['nDR']['Head_Tail']['count'] / self.modi_ruleset['nDR']['count']]
            prob.extend(prob_s)
        return seed_array, prob

    def encode_dr(self, rule, i):
        """

        :param rule: 'DR' or 'nDR'
        :param i: for i+1 th password
        :return: direct reuse probability of single similar password model (both lists)
        """
        tot = self.direct_reuse['count']
        dr_dic = self.set_sspmreuse(i)
        ind = list(dr_dic.keys()).index(rule)
        l_pt = sum([c['count'] for c in list(dr_dic.values())[1:ind]])
        r_pt = l_pt + dr_dic[rule]['count'] - 1
        assert l_pt <= r_pt, "Rule with zero freq! rhs_dict[{}] =  {} (right_dic={})" \
            .format(rule, dr_dic[rule], dr_dic)
        seed = self.convert2seed(random.randint(l_pt, r_pt), tot)
        prob = dr_dic[rule]['count'] / tot
        return [seed], [prob]

    def encode_path(self, path, right_dic):
        seed_p, prob = [], []
        for p in path.copy():
            if p in self.alphabet_dict['alp'] or p in self.alphabet_dict['AN']:
                seed_, prob_ = self.encode_dic(p, path)
                if path.count(p) > 1:
                    path[path.index(p)] = None
                seed_p.append(seed_)
                prob.append(prob_)
            else:
                tot = right_dic['count']
                i = list(right_dic.keys()).index(p)
                l_pt = sum([c['count'] for c in list(right_dic.values())[1:i]])
                r_pt = l_pt + right_dic[p]['count'] - 1
                assert l_pt <= r_pt, "Rule with zero freq! rhs_dict[{}] =  {} (right_dic={})" \
                    .format(p, right_dic[p], right_dic)
                seed_p.append(self.convert2seed(random.randint(l_pt, r_pt), tot))
                right_dic = right_dic[p]
                prob.append(right_dic['count'] / tot)

        return seed_p, prob

    def encode_dic(self, p, path):
        tmp_dic = self.alphabet_dict['alp']
        if 'Add' in path:
            if path.index(p) - path.index('Add') == 1:
                tmp_dic = self.alphabet_dict['AN']
        elif 'Delete-then-add' in path:
            if path.index(p) - path.index('Delete-then-add') == 1:
                tmp_dic = self.alphabet_dict['DN']
            elif path.index(p) - path.index('Delete-then-add') == 2:
                tmp_dic = self.alphabet_dict['AN']
        elif 'Delete' in path:
            if path.index(p) - path.index('Delete') == 1:
                tmp_dic = self.alphabet_dict['DN']
        fill = tmp_dic[next(reversed(tmp_dic))]  # get last value
        idx = list(tmp_dic.keys()).index(p)
        former = tmp_dic[list(tmp_dic.keys())[idx - 1]] if idx > 0 else 0
        range = tmp_dic[p] - former
        seed_ = self.convert2seed(random.randint(former, tmp_dic[p]-1), fill)
        prob_ = range / fill
        return seed_, prob_

    def decode_pw(self, seeds, src_pw, i):
        #assert len(seeds) == (MAX_PW_LENGTH - 1)
        drornot = self.decode_seed(self.modi_ruleset, seeds.pop(0), i)
        if drornot == 'DR':
            return src_pw
        start = self.decode_seed(self.modi_ruleset['nDR'], seeds.pop(0))
        right_dict = self.modi_ruleset['nDR'][start] if 'Head_Tail' in start else self.modi_ruleset['nDR']
        if 'Head' in start:
            seeds, src_pw = self.decode_path(seeds, right_dict['Head'], src_pw, head=True)
        if 'Tail' in start:
            _, src_pw = self.decode_path(seeds, right_dict['Tail'], src_pw, head=False)
        return src_pw

    def decode_path(self, seeds, right_dic, src_pw, head=True):
        de_rul_start = self.decode_seed(right_dic, seeds.pop(0))
        if 'elete' in de_rul_start: # 'Delete' or 'Delete-then-add'
            de_rul = self.decode_char(seeds.pop(0), 'DN')
            assert de_rul.isdigit()
            src_pw = src_pw[int(de_rul):] if head else src_pw[:-int(de_rul)]
        if 'dd' in de_rul_start:  # 'Add' or 'Delete-then-add'
            #if 'Add' in de_rul_start:
            de_rul = self.decode_char(seeds.pop(0), 'AN')
            assert de_rul.isdigit()
            add_len = int(de_rul)
            add_string = ''
            for i in range(add_len):
                de_rul = self.decode_char(seeds.pop(0), 'alp')
                add_string += de_rul
            src_pw = add_string+src_pw if head else src_pw+add_string
        return seeds, src_pw

    def decode_seed(self, right_dic, seed, i=None):
        if i is not None:
            right_dic = self.set_sspmreuse(i)
        counts = [c['count'] for c in list(right_dic.values())[1:]] # exclude 'count' from the right_dic
        cum = [sum(counts[:i]) for i in range(len(counts) + 1)] # from 0 to the value of right_dic['count']
        decode = seed % right_dic['count']
        cum = np.array(cum) - decode
        cum = (cum * np.roll(cum, -1))[:-1]  # roll: <--
        loc = cum.argmin()
        if cum[loc] == 0 and (cum==0).sum() == 2:
            loc += 1
        else:
            assert cum[loc] <= 0
        de_rule = list(right_dic.keys())[loc + 1]
        return de_rule

    def decode_char(self, seed, type):
        tmp_dic = self.alphabet_dict[type]
        decode = seed % tmp_dic[next(reversed(tmp_dic))]
        cum = [0]
        cum.extend(list(tmp_dic.values()))
        cum = np.array(cum) - decode
        cum = (cum * np.roll(cum, -1))[:-1]  # roll: <--
        loc = cum.argmin()
        if cum[loc] == 0 and (cum==0).sum() == 2:
            loc += 1
        else:
            assert cum[loc] <= 0
        de_rule = list(tmp_dic.keys())[loc]
        return de_rule

    def set_sspmreuse(self, i):
        self.direct_reuse['DR']['count'] = int(self.direct_reuse['count'] * self.get_sspmreuse(i))
        self.direct_reuse['nDR']['count'] = self.direct_reuse['count'] - int(self.direct_reuse['count'] * self.get_sspmreuse(i))
        return self.direct_reuse

    def get_sspmreuse(self, i):
        return i * ALPHA / (i * ALPHA + 1 - ALPHA)

    def convert2seed(self, rand_val, fill, n=1):
        if n == 1:
            return rand_val + random.randint(0, (SEED_MAX_RANGE - rand_val) // fill) * fill
        else:
            return [
                rand_val + c * fill
                for c in random.randints(0, (SEED_MAX_RANGE - rand_val) // fill, n=n)
            ]

    def load(self, i=None):
        if args.sspmdata == 'pastebin' and not args.pretrained:
            if i is None:
                suffix = ''
            elif type(i) is int:
                suffix = str(i)
            elif type(i) is str:
                suffix = i
        elif args.sspmdata == 'rockyou' or args.pretrained:
            suffix = 'rky'
        with open(SOURCE_PATH + '/MSPM/SSPM/trained_model'+args.exp_pastebinsuffix+'/sspm_rules'+suffix+'.json') as sspm:
            self.modi_ruleset = json.load(sspm, object_pairs_hook=OrderedDict)
            print('loading sspm_rules'+suffix+'.json'+' '+ args.exp_pastebinsuffix)
        with open(SOURCE_PATH + '/MSPM/SSPM/trained_model'+args.exp_pastebinsuffix+'/sspm_dicts'+suffix+'.json') as sspm:
            self.alphabet_dict = json.load(sspm, object_pairs_hook=OrderedDict)
            print('loading sspm_dicts'+suffix+'.json'+' '+ args.exp_pastebinsuffix)
        if not args.pretrained:
            for type_ in list(self.alphabet_dict.keys()):
                tmp_dic = self.alphabet_dict[type_]
                for i, key in enumerate(list(tmp_dic.keys())):
                    tmp_dic[key] += tmp_dic[list(tmp_dic.keys())[i-1]] if i>0 else 0
        self.direct_reuse = OrderedDict({'count': self.modi_ruleset['count']})
        for name in self.modi_ruleset:
            if type(self.modi_ruleset[name]) is OrderedDict:
                self.direct_reuse[name] = OrderedDict({'count': self.modi_ruleset[name]['count']})

    def train(self):
        # create directory (after checking if it exists)
        if not os.path.exists("trained_model"+args.exp_pastebinsuffix):
            os.makedirs("trained_model"+args.exp_pastebinsuffix)
        if args.sspmdata == 'pastebin':
            for testid in self.train_vault:
                self.modi_ruleset = OrderedDict()
                self.alphabet_dict = self._init_alphdic()
                if not args.multi_train:
                    for id in self.train_vault:
                        if id != testid:
                            tset = self.train_vault[id]
                            for vault_id in tset:
                                if len(tset[vault_id]) > 1 and len(tset[vault_id]) < 51:
                                    for i in range(len(tset[vault_id]) - 1):
                                        for j in range(i + 1, len(tset[vault_id])):
                                            self.reuse_model(tset[vault_id][i], tset[vault_id][j], self.modi_ruleset)
                            self.modi_ruleset.update({'count': self.modi_ruleset['DR']['count'] + self.modi_ruleset['nDR']['count']})
                            self.modi_ruleset.move_to_end('count', last=False)
                else:
                    for id in self.train_vault:
                        if id[0] == testid[0] and id != testid: # complementary split under same repeat of pastebin
                            tset = self.train_vault[id]
                            for idx, vault_id in enumerate(tqdm(tset)):
                                for i in range(len(tset[vault_id]) - 1):
                                    for j in range(i + 1, len(tset[vault_id])):
                                        self.reuse_model(tset[vault_id][i], tset[vault_id][j], self.modi_ruleset)
                            self.modi_ruleset.update({'count': self.modi_ruleset['DR']['count'] + self.modi_ruleset['nDR']['count']})
                            self.modi_ruleset.move_to_end('count', last=False)
                file1 = "trained_model"+args.exp_pastebinsuffix+"/sspm_rules"+str(testid)+".json"
                file2 = "trained_model"+args.exp_pastebinsuffix+"/sspm_dicts"+str(testid)+".json"
                # check file existence
                if os.path.isfile(file1) and os.path.isfile(file2):
                    print('sspm_rules'+str(testid)+'.json'+' already exists')
                    print('sspm_dicts'+str(testid)+'.json'+' already exists')
                    continue
                with open(file1, "w") as outfile:
                    json.dump(self.modi_ruleset, outfile)
                with open(file2, "w") as outfile:
                    json.dump(self.alphabet_dict, outfile)
        elif args.sspmdata == 'rockyou':
            # self.train_vault: dual lst => [0]:pws  [1]:cnts
            print('start training sspm with rocky !')
            for i in tqdm(range(len(self.train_vault[0]) - 1), total=len(self.train_vault[0])-1,
                     miniters=int(len(self.train_vault[0]) / 1000), unit="pw"):
                r = SAMPLES_PERP if i+SAMPLES_PERP < len(self.train_vault[0]) else (len(self.train_vault[0]) - i)
                for j in range(0, r):
                    if j == 0 and self.train_vault[1][i] > 1:
                        cnt = self.train_vault[1][i] - 1
                    elif j > 0:
                        cnt = self.train_vault[1][i] * self.train_vault[1][i+j]
                    else:
                        continue
                    self.reuse_model(self.train_vault[0][i], self.train_vault[0][i+j], self.modi_ruleset, cnt=cnt)
            self.modi_ruleset.update({'count': self.modi_ruleset['DR']['count'] + self.modi_ruleset['nDR']['count']})
            self.modi_ruleset.move_to_end('count', last=False)
            with open("trained_model/sspmrky_rules.json", "w") as outfile:
                json.dump(self.modi_ruleset, outfile)
            with open("trained_model/sspmrky_dicts.json", "w") as outfile:
                json.dump(self.alphabet_dict, outfile)

    def get_alpha(self):
        """
        averaged proportion of same password pairs within vaults
        :return:
        """
        alphas = []
        for vault_id in self.train_vault:
            avault = self.train_vault[vault_id]
            if len(avault) > 1 and len(avault) < 51:
                alphas.append(np.array([math.factorial(count-1) for _, count in collections.Counter(avault).items() if count > 1]).sum() / math.factorial(len(avault) - 1))
        return np.array(alphas).mean()

    def reuse_model(self, pw1, pw2, modi_ruleset=None, cnt=1):

        if modi_ruleset is None:
            modi_list = []
            if pw1 == pw2:
                modi_list.append('DR')
                return modi_list

            res = pylcs.lcs_string_idx(pw1, pw2)
            lcss = ''.join([pw2[i] for i in res if i != -1])  # longest common substring
            sub_len = len(lcss)

            start_pw1 = pw1.index(lcss)
            start_pw2 = pw2.index(lcss)
            if (start_pw1 + sub_len) > len(pw1) or (start_pw2 + sub_len) > len(pw2):
                raise Exception('something error with substring and its corresponding location in pw1 or pw2!')

            HD, TD = start_pw1, len(pw1) - start_pw1 - sub_len
            HA, TA = start_pw2, len(pw2) - start_pw2 - sub_len

            if (HD <= len(pw1) / 2 and TD <= (len(pw1) / 2 - HD) and
                HA <= 2 * (len(pw1) - HD - TD) and TA <= (2 * (len(pw1) - HD - TD) - HA) and
                (HA < SSPM_CHARS_MAX and HD < SSPM_CHARS_MAX and TA < SSPM_CHARS_MAX and TD < SSPM_CHARS_MAX) and
                (HA+HD+TA+TD) != 0):

                if HA == 0 and HD == 0:  # tail modification
                    modi_list = self.sequence_prob(TA, TD, pw2[-TA:], sub='Tail')

                elif TD == 0 and TA == 0:  # head modification
                    modi_list = self.sequence_prob(HA, HD, pw2[:HA], sub='Head')

                else:  # head & tail modification
                    modi_list.append('Head_Tail')
                    # first head then tail ('head' or 'tail' itself is not considered in prob or coding)
                    modi_list = self.sequence_prob(HA, HD, pw2[:HA], lst=modi_list, sub='Head')
                    modi_list = self.sequence_prob(TA, TD, pw2[-TA:], lst=modi_list, sub='Tail')
                modi_list.insert(0, 'nDR')
            return modi_list
        else:
            if pw1 == pw2:
                if 'DR' in modi_ruleset: # direct reuse
                    modi_ruleset['DR']['count'] += 1 * cnt
                else:
                    modi_ruleset['DR'] = OrderedDict({'count': 1 * cnt})
            else:
                res = pylcs.lcs_string_idx(pw1, pw2)
                lcss = ''.join([pw2[i] for i in res if i != -1])  # longest common substring
                sub_len = len(lcss)

                start_pw1 = pw1.index(lcss)
                start_pw2 = pw2.index(lcss)
                if (start_pw1 + sub_len) > len(pw1) or (start_pw2 + sub_len) > len(pw2):
                    raise Exception('something error with substring and its corresponding location in pw1 or pw2!')

                HD, TD = start_pw1, len(pw1) - start_pw1 - sub_len
                HA, TA = start_pw2, len(pw2) - start_pw2 - sub_len

                if HD <= len(pw1) / 2 and TD <= (len(pw1) / 2 - HD) and HA <= 2 * (len(pw1) - HD - TD) and TA <= (2 * (len(pw1) - HD - TD) - HA) and HA < SSPM_CHARS_MAX and HD < SSPM_CHARS_MAX and TA < SSPM_CHARS_MAX and TD < SSPM_CHARS_MAX:
                    if 'nDR' in modi_ruleset:  # modify former passwords
                        modi_ruleset['nDR']['count'] += 1 * cnt
                    else:
                        modi_ruleset['nDR'] = OrderedDict({'count': 1 * cnt})
                    #modi_ruleset['nDR']['count'] = (modi_ruleset['nDR']['count'] + 1) if 'count' in modi_ruleset else 1

                    if HA == 0 and HD == 0:  # tail modification
                        if 'Tail' in modi_ruleset['nDR']:
                            modi_ruleset['nDR']['Tail']['count'] += 1 * cnt
                        else:
                            modi_ruleset['nDR']['Tail'] = OrderedDict({'count': 1 * cnt})

                        if TA == 0:  # Delete
                            self.delete(modi_ruleset['nDR'], 'Tail', TD, cnt)
                        elif TD == 0:  # Add
                            self.add(modi_ruleset['nDR'], 'Tail', pw2[-TA:], cnt)
                        else:  # Delete-then-add
                            self.delete_add(modi_ruleset['nDR'], 'Tail', TD, pw2[-TA:], cnt)

                    elif TD == 0 and TA == 0:  # head modification
                        if 'Head' in modi_ruleset['nDR']:
                            modi_ruleset['nDR']['Head']['count'] += 1 * cnt
                        else:
                            modi_ruleset['nDR']['Head'] = OrderedDict({'count': 1 * cnt})

                        if HA == 0:  # Delete
                            self.delete(modi_ruleset['nDR'], 'Head', HD, cnt)
                        elif HD == 0:  # Add
                            self.add(modi_ruleset['nDR'], 'Head', pw2[:HA], cnt)
                        else:  # Delete-then-add
                            self.delete_add(modi_ruleset['nDR'], 'Head', HD, pw2[:HA], cnt)

                    else:  # head & tail modification
                        if 'Head_Tail' in modi_ruleset['nDR']:
                            modi_ruleset['nDR']['Head_Tail']['count'] += 1 * cnt
                            modi_ruleset['nDR']['Head_Tail']['Head']['count'] += 1 * cnt
                            modi_ruleset['nDR']['Head_Tail']['Tail']['count'] += 1 * cnt
                        else:
                            modi_ruleset['nDR']['Head_Tail'] = OrderedDict({'count': 1 * cnt})
                            modi_ruleset['nDR']['Head_Tail']['Head'] = OrderedDict({'count': 1 * cnt})
                            modi_ruleset['nDR']['Head_Tail']['Tail'] = OrderedDict({'count': 1 * cnt})

                        if HA == 0:  # Delete
                            self.delete(modi_ruleset['nDR']['Head_Tail'], 'Head', HD, cnt)
                        elif HD == 0:  # Add
                            self.add(modi_ruleset['nDR']['Head_Tail'], 'Head', pw2[:HA], cnt)
                        else:  # Delete-then-add
                            self.delete_add(modi_ruleset['nDR']['Head_Tail'], 'Head', HD, pw2[:HA], cnt)

                        if TA == 0:  # Delete
                            self.delete(modi_ruleset['nDR']['Head_Tail'], 'Tail', TD, cnt)
                        elif TD == 0:  # Add
                            self.add(modi_ruleset['nDR']['Head_Tail'], 'Tail', pw2[-TA:], cnt)
                        else:  # Delete-then-add
                            self.delete_add(modi_ruleset['nDR']['Head_Tail'], 'Tail', TD, pw2[-TA:], cnt)

    def delete(self, dic, portion, n, cnt):
        """

        :param dic:
        :param portion: Head or Tail or Head_Tail
        :param n: delete number
        :return:
        """

        if 'Delete' in dic[portion]:
            dic[portion]['Delete']['count'] += 1 * cnt
        else:
            dic[portion]['Delete'] = OrderedDict({'count': 1 * cnt})
        self.alphabet_dict['DN'][str(n)] += 1 * cnt

    def add(self, dic, portion, add_str, cnt):
        """

        :param dic:
        :param portion: Head or Tail or Head_Tail
        :param add_str: add string
        :return:
        """

        n = len(add_str)
        if 'Add' in dic[portion]:
            dic[portion]['Add']['count'] += 1 * cnt
        else:
            dic[portion]['Add'] = OrderedDict({'count': 1 * cnt})
        self.alphabet_dict['AN'][str(n)] += 1 * cnt
        for i in range(n):
            self.add_char(add_str[i], cnt)

    def delete_add(self, dic, portion, del_n, add_str, cnt):
        """

        :param dic:
        :param portion: Head or Tail or Head_Tail
        :param del_n: delete char number
        :param add_str: add string
        :return:
        """

        if 'Delete-then-add' in dic[portion]:
            dic[portion]['Delete-then-add']['count'] += 1 * cnt
        else:
            dic[portion]['Delete-then-add'] = OrderedDict({'count': 1 * cnt})
        add_n = len(add_str)
        self.alphabet_dict['DN'][str(del_n)] += 1 * cnt
        self.alphabet_dict['DN'][str(add_n)] += 1 * cnt
        for i in range(add_n):
            self.add_char(add_str[i], cnt)

    def add_char(self, char, cnt):
        self.alphabet_dict['alp'][char] += 1 * cnt

    def sequence_prob(self, a, d, st, lst=None, sub=None):
        '''

        :param a: addition number
        :param d: deletion number
        :param st: addition string
        :param lst: modification list
        :return: action list
        '''
        if lst is None:
            lst = []
        sub_lst = [] if sub is None else [sub]
        if a == 0:  # Delete
            sub_lst.append('Delete')
            sub_lst.append(str(d))
        elif d == 0:  # Add
            sub_lst.append('Add')
            sub_lst.append(str(a))
            for alpha in st:
                sub_lst.append(alpha)
        else:  # Delete-then-add
            sub_lst.append('Delete-then-add')
            sub_lst.append(str(d))
            sub_lst.append(str(a))
            for alpha in st:
                sub_lst.append(alpha)
        lst.append(sub_lst)
        return lst

def main():
    # train the model
    pcfg = RulePCFGCreator(args)
    pcfg.train()

if __name__ == "__main__":
    main()