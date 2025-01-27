## Bubble

Bubble is the first framework for protecting honey vaults from Credential-Guided Attacks, where attackers holding the storage file and a credential leaked from the vault aims to uncover the user's master secret.

## Table of contents

- Requirements
- Running
- Experimenting

## Requirements

pip3 install pycryptodome
pip3 install publicsuffix
pip3 install numpy
pip3 install rainbow_logging_handler
pip install msgpack
pip install tqdm
pip install pylcs
pip install cryptography
pip install mgzip
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==23.12.*

### links to DTE model and datasets

## Running

## Experimenting

### change password vault dataset

1. train pcfg model (markov model is trained based on password dataset)
2. fit unreuse(i) function and parameter "alpha"

### change victim (MSPM or Golla)

1. change args.victim

### experiments shift (between datasets) checklist

0. pcfg model path "--exp_pastebinsuffix" (option: _pb, _bc50, _bc200); bubble or bubble+
1. "PASTB_PATH" setting
2. "ALPHA" and function "unreuse_p(i)" in unreused_prob.py
2.1 dte_global wether under comment
2.2 check whether DTE with or without bubble
3. others following ## experiments shift (between parameters) checklist

### experiments shift (between parameters) checklist

0. check "incremental_vault_gpu/data/breachcompilation/fold2/fold_0_1.json" testset
0.1. check whether only run "multi-version attack" if wants both
1. check parameters: 'mpwbatch',
2. check "weight.py" whether loaded 14000+ vaults instead of 100 for debugging
3. check "outputdir" in attack.py
4. check whether golla's rank & cheng's rank & wang'rank all need to run (wang's weight calculation returns sp_prob #[sp_prob, vault_decoyprob] and #addition_weight(batch_bundle) and #median_rankid=2)
4.1 check online verification runs, 'onlineverification_result' from func 'find_real_candidatelist'
5. settings for T, pin, Nitv

attacker.py
--model_eval mspm --victim MSPM --physical --withleak --softfilter --logical --spmdata neopets --exp_pastebinsuffix _bc100 --pin RockYou-6-digit.txt --pinlength 6 --fixeditv --fixeditvmode 1 --intersection --version_gap 1 --isallleaked 0 --gpu 1

pcfg.py
--multi_train --exp_pastebinsuffix _bc50


how to split dataset pastebin
1. set 'num_files' wanted to split uniformly and randomly, and the 'repeat_num' that will determine how many splits for entire dataset happens; and run train_test_split.py
2. run pcfg.py to train sspm model on splited dataset (randomly copy a dict and rule model with names without number for preloading, it does not matter). Note that for each repeat of splited dataset, an independent model will be trained
    add arguments: --exp_pastebinsuffix 'num_files'_Nge5 --multi_train
    

run attack 1&2 under construction 2 experiment:
--model_eval mspm --victim MSPM --physical --withleak --softfilter --logical --pin RockYou-4-digit.txt --intersection --exp_pastebinsuffix 2_Nge5 --fixeditv --fixeditvmode 1

run attack 1&2 under construction 1 experiment:
--model_eval mspm --victim MSPM --physical --withleak --softfilter --logical --pin RockYou-4-digit.txt --intersection --exp_pastebinsuffix 2_Nge5


attacker.py
--model_eval mspm --victim MSPM --physical --withleak --softfilter --logical --pin RockYou-6-digit.txt --intersection --exp_pastebinsuffix 2_Nge5 --expandtestset --fixeditv --fixeditvmode 1


pcfg.py
--exp_pastebinsuffix 2_Nge5 --multi_train

metric.py
--logical --intersection
