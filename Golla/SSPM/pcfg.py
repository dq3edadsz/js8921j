import pylcs
from collections import OrderedDict
from tqdm import tqdm
import json
from MSPM.mspm_config import *
from Golla.utils import random
import os
import numpy as np
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
                    f = open(os.path.join(PASTB_PATH + args.exp_pastebinsuffix, 'fold_' + str(rn) + '_' + str(id) + '.json'))
                    vaults_tmp = json.load(f)
                    self.train_vault[str(rn) + '_' + str(id)] = vaults_tmp
        return self.train_vault

    def encode_pw(self, src_pw, targ_pw):
        """

        :param src_pw:
        :param targ_pw:
        :param ith: i is set for target password (when encoding target password i+1)
        :return:
        """
        code_path = self.reuse_model(src_pw, targ_pw)
        #print('code path:', code_path)
        seed, prob = self.encode_path(code_path[0], self.modi_ruleset)
        #extra = SEED_LEN - len(seed) - 1 # -1 is to reserve space for path seed
        #seed.extend(self.convert2seed(0, 1, extra))
        return seed, prob, code_path[0]

    def encode_path(self, path, right_dic):
        seed_p, prob = [], []
        for p in path.copy():
            tot = right_dic['count']
            i = list(right_dic.keys()).index(p)
            l_pt = sum([c['count'] for c in list(right_dic.values())[1:i]])
            r_pt = l_pt + right_dic[p]['count'] - 1
            assert l_pt <= r_pt, "Rule with zero freq! rhs_dict[{}] =  {} (right_dic={})".format(p, right_dic[p], right_dic)
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

    def decode_modipath(self, seeds):
        #assert len(seeds) == (MAX_PW_LENGTH - 1)
        prob_lst = []
        edit_dist, prob = self.decode_seed(self.modi_ruleset, seeds.pop(0))
        prob_lst.append(prob)
        modi_path = [int(edit_dist)]
        if int(edit_dist) > 0 and int(edit_dist) <= 4:
            edit_op, prob = self.decode_seed(self.modi_ruleset[edit_dist], seeds.pop(0))
            modi_path.append(edit_op)
            prob_lst.append(prob)
        return modi_path, prob_lst

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

    def decode_seed(self, right_dic, seed):
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
        return de_rule, right_dic[de_rule]['count'] / right_dic['count']

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
        with open(SOURCE_PATH + '/Golla/SSPM/trained_model'+args.exp_pastebinsuffix+'/sspm_rules'+suffix+'.json') as sspm:
            self.modi_ruleset = json.load(sspm, object_pairs_hook=OrderedDict)
            print('loading sspm_rules'+suffix+'.json'+' '+ args.exp_pastebinsuffix)
        with open(SOURCE_PATH + '/Golla/SSPM/trained_model'+args.exp_pastebinsuffix+'/sspm_dicts'+suffix+'.json') as sspm:
            self.alphabet_dict = json.load(sspm, object_pairs_hook=OrderedDict)
            print('loading sspm_dicts'+suffix+'.json'+' '+ args.exp_pastebinsuffix)
        if not args.pretrained:
            for type_ in list(self.alphabet_dict.keys()):
                tmp_dic = self.alphabet_dict[type_]
                for i, key in enumerate(list(tmp_dic.keys())):
                    tmp_dic[key] += tmp_dic[list(tmp_dic.keys())[i-1]] if i>0 else 0

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
                                        self.reuse_model(tset[vault_id][j], tset[vault_id][i], self.modi_ruleset)
                            self.modi_ruleset.update({'count': sum([self.modi_ruleset[edit_dist_]['count'] for edit_dist_ in self.modi_ruleset.keys() if edit_dist_ != 'count'])})
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

    def check_pw_pair(self, pw1, pw2):
        # determine whether pw1 and pw2 satisfy reuse definition, which includes the following: consider modify pw1 => pw2
        # 1. pw1 and pw2 have a common substring from the head (the first char)
        # 2. pw1 and pw2 have edit distance of less than 5
        if pw2[:len(pw1)-1] != pw1[:-1]:
            return False
        if pylcs.edit_distance(pw1, pw2) > 4: # only considers edit distance of less than 5
            return False
        if len(pw1) < len(pw2) and pw1[-1] in pw2[len(pw1):]: # corner case pw1='bobby1' => pw2='bobbylex1' edit distance = 3 but it takes 5 steps to modify pw1 to pw2
            return False
        return True


    def reuse_model(self, pw1, pw2, modi_ruleset=None, cnt=1):

        if modi_ruleset is None:
            res = pylcs.lcs_string_idx(pw1, pw2)
            lcss = ''.join([pw2[i] for i in res if i != -1])  # longest common substring
            sub_len = len(lcss)

            start_pw1 = pw1.index(lcss)
            start_pw2 = pw2.index(lcss)
            if (start_pw1 + sub_len) > len(pw1) or (start_pw2 + sub_len) > len(pw2):
                raise Exception('something error with substring and its corresponding location in pw1 or pw2!')

            HD, TD = start_pw1, len(pw1) - start_pw1 - sub_len
            HA, TA = start_pw2, len(pw2) - start_pw2 - sub_len

            edit_dist = pylcs.edit_distance(pw1, pw2)
            if self.check_pw_pair(pw1, pw2) and edit_dist>0:
                modi_list = self.sequence_prob(TA, TD, pw2[-TA:], sub='Tail')
                assert 'Tail' in modi_list[0]
                modi_list[0][0] = str(edit_dist)
            elif self.check_pw_pair(pw1, pw2) and edit_dist==0: # edit_dist == 0
                modi_list = [['0']]
            elif not self.check_pw_pair(pw1, pw2):
                modi_list = [['5']]
            return modi_list
        else:
            if self.check_pw_pair(pw1, pw2):
                res = pylcs.lcs_string_idx(pw1, pw2)
                lcss = ''.join([pw2[i] for i in res if i != -1])  # longest common substring
                sub_len = len(lcss)

                start_pw1 = pw1.index(lcss)
                start_pw2 = pw2.index(lcss)
                if (start_pw1 + sub_len) > len(pw1) or (start_pw2 + sub_len) > len(pw2):
                    raise Exception('something error with substring and its corresponding location in pw1 or pw2!')

                HD, TD = start_pw1, len(pw1) - start_pw1 - sub_len
                HA, TA = start_pw2, len(pw2) - start_pw2 - sub_len
                assert HA == 0 and HD == 0 and TD <= 1

                edit_dist = pylcs.edit_distance(pw1, pw2)
                if edit_dist in modi_ruleset:  # modify former passwords
                    modi_ruleset[edit_dist]['count'] += 1 * cnt
                else:
                    modi_ruleset[edit_dist] = OrderedDict({'count': 1 * cnt})
                if edit_dist == 0:
                    return
                #modi_ruleset['nDR']['count'] = (modi_ruleset['nDR']['count'] + 1) if 'count' in modi_ruleset else 1

                if HA == 0 and HD == 0:  # tail modification
                    if TA == 0:  # Delete
                        self.delete(modi_ruleset, edit_dist, TD, cnt)
                    elif TD == 0:  # Add
                        self.add(modi_ruleset, edit_dist, pw2[-TA:], cnt)
                    else:  # Delete-then-add
                        self.delete_add(modi_ruleset, edit_dist, TD, pw2[-TA:], cnt)
            else:
                edit_dist = 5
                if edit_dist in modi_ruleset:  # unrelated password
                    modi_ruleset[edit_dist]['count'] += 1 * cnt
                else:
                    modi_ruleset[edit_dist] = OrderedDict({'count': 1 * cnt})


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
        elif d == 0:  # Add
            sub_lst.append('Add')
        else:  # Delete-then-add
            sub_lst.append('Delete-then-add')
        lst.append(sub_lst)
        return lst

def main():
    # train the model
    pcfg = RulePCFGCreator(args)
    pcfg.train()

if __name__ == "__main__":
    main()

    '''pw1, pw2 = 'abc312', 'abc31245'
    print('pw1:', pw1, 'pw2:', pw2, 'res', pylcs.lcs_string_idx(pw1, pw2))

    pw1, pw2 = 'abc312', 'abc31'
    print('pw1:', pw1, 'pw2:', pw2, 'res', pylcs.lcs_string_idx(pw1, pw2))

    pw1, pw2 = 'abc312', '21abc312'
    print('pw1:', pw1, 'pw2:', pw2, 'res', pylcs.lcs_string_idx(pw1, pw2))

    pw1, pw2 = 'abc312', 'bc312'
    print('pw1:', pw1, 'pw2:', pw2, 'res', pylcs.lcs_string_idx(pw1, pw2))'''