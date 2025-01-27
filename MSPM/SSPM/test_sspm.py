from MSPM.unreused_prob import unreuse_p
from MSPM.SPM.configs.configure import *
from MSPM.SPM.ngram_creator import NGramCreator
from MSPM.SSPM.pcfg import RulePCFGCreator
import json
from collections import OrderedDict
from MSPM.mspm_config import *
from opts import opts
args = opts().parse()

sspm = RulePCFGCreator(args)
sspm.load()
train_vault = {}
data_path = '/home/beeno/Dropbox/research_project/pycharm/incremental_vault_gpu/data/pastebin/fold5'
flst = os.listdir(data_path)
for fname in flst:
    if str(TEST_FOLD) in fname:
        print('test w/ test file:', fname)
        f = open(os.path.join(data_path, fname))
        train_vault.update(json.load(f))
vault = train_vault

# test1: encode modification coding path
'''
for vault_id in vault:
    for i in range(len(vault[vault_id]) - 1):
        for j in range(i + 1, len(vault[vault_id])):
            if vault[vault_id][i] == vault[vault_id][j]:
                continue
            else:
                code_path = sspm.reuse_model(vault[vault_id][i], vault[vault_id][j])
                print(vault[vault_id][i], '->', vault[vault_id][j], ':', code_path)
                if len(code_path) != 0:
                    if 'Head_Tail' in code_path[0]:
                        seed_h, prob_h = sspm.encode_path(code_path[1], sspm.modi_ruleset['Head_Tail'])
                        seed_t, prob_t = sspm.encode_path(code_path[2], sspm.modi_ruleset['Head_Tail'])
                        tot = sspm.modi_ruleset['count']
                        i = list(sspm.modi_ruleset.keys()).index('Head_Tail')
                        l_pt = sum([c['count'] for c in list(sspm.modi_ruleset.values())[1:i]])
                        r_pt = l_pt + sspm.modi_ruleset['Head_Tail']['count'] - 1
                        del seed_h[0], seed_t[0]
                        seed = [sspm.convert2seed(random.randint(l_pt, r_pt), tot)]
                        seed.extend(seed_h)
                        seed.extend(seed_t)
                        prob = [sspm.modi_ruleset['Head_Tail']['count'] / sspm.modi_ruleset['count']]
                        prob.extend(prob_h)
                        prob.extend(prob_t)
                    else:
                        assert 'Head' in code_path[0] or 'Tail' in code_path[0]
                        seed, prob = sspm.encode_path(code_path[0])

                        print('seed:', seed)
                        print('prob:', prob)
                        '''


'''# test2: decoding process of modification path
for vault_id in vault:
    for i in range(len(vault[vault_id]) - 1):
        for j in range(i + 1, len(vault[vault_id])):
            if vault[vault_id][i] == vault[vault_id][j]:
                continue
            else:
                code_path = sspm.reuse_model(vault[vault_id][i], vault[vault_id][j])
                print(vault[vault_id][i], '->', vault[vault_id][j], ':', code_path)
                if len(code_path) != 0:
                    if 'Head_Tail' in code_path[0]:
                        seed_h, prob_h = sspm.encode_path(code_path[1], sspm.modi_ruleset['Head_Tail'])
                        seed_t, prob_t = sspm.encode_path(code_path[2], sspm.modi_ruleset['Head_Tail'])
                        tot = sspm.modi_ruleset['count']
                        i_ = list(sspm.modi_ruleset.keys()).index('Head_Tail')
                        l_pt = sum([c['count'] for c in list(sspm.modi_ruleset.values())[1:i_]])
                        r_pt = l_pt + sspm.modi_ruleset['Head_Tail']['count'] - 1
                        del seed_h[0], seed_t[0]
                        seed = [sspm.convert2seed(random.randint(l_pt, r_pt), tot)]
                        seed.extend(seed_h)
                        seed.extend(seed_t)
                        prob = [sspm.modi_ruleset['Head_Tail']['count'] / sspm.modi_ruleset['count']]
                        prob.extend(prob_h)
                        prob.extend(prob_t)
                    else:
                        assert 'Head' in code_path[0] or 'Tail' in code_path[0]
                        seed, prob = sspm.encode_path(code_path[0])

                    print('     seed:', seed)
                    print('     prob:', prob)

                    pw = sspm.decode_pw(seed, vault[vault_id][i])
                    print('     decoding:', vault[vault_id][i], pw)
                    assert pw == vault[vault_id][j]'''

# test3: decoding process of modification path with self function
for vault_id in vault:
    for i in range(len(vault[vault_id]) - 1):
        for j in range(i + 1, len(vault[vault_id])):
            seed, prob = sspm.encode_pw(vault[vault_id][i], vault[vault_id][j], i+1)
            print(vault[vault_id][i], '->', vault[vault_id][j])
            print('     seed:', seed)
            print('     prob:', prob)
            if len(seed) == 0:
                continue
            pw = sspm.decode_pw(seed, vault[vault_id][i], i+1)
            print('     decoding:', vault[vault_id][i], pw)
            assert pw == vault[vault_id][j]