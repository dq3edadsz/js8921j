## Bubble

Bubble is the first framework for protecting honey vaults from Credential-Guided Attacks, where attackers holding the storage file and a credential leaked from the vault aims to uncover the user's master secret.

## Table of contents

- Requirements
- Implementation of Bubble
- Experimenting

## Requirements

- pip3 install pycryptodome
- pip3 install publicsuffix
- pip3 install numpy
- pip3 install rainbow_logging_handler
- pip install msgpack
- pip install tqdm
- pip install pylcs
- pip install cryptography
- pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==23.12.*

### links to the trained DTE model

This code includes implementation of two DTE models: Markov DTE (CCS'16) and Incremental DTE (USENIX'21). We link their single-password model (4gram Markov). Their Single-similar password models (SSPM) are already incorporated in the uploaded code.

- 4gram cumulative probability model (trained on password dataset **Rockyou**)
- 4gram cumulative probability model (trained on password dataset **Neopets**)

download the model above in the directory "/bubble/MSPM/SPM/trained/full/4gram/"

## Implementation of Bubble

Implementation of Bubble is in vault.py under the directory "/Vault_system/"

Specifically, there is a **vault** class, including high-level functions **init_vault**(pwvault), **load_vault**(), **get_pw**(), **add_pw**(pw, dm), where *pwvault* is a password vault in the form of a dictionary with *dm* (website domain) as key and *pw* (website password) as value.

## Experimenting

The experiment simulates the attacker's online verification process. That said, we skip the offline gueessing process, giving attackers the candidate list after brute forcing the space of master secrets. Attackers remove and rank guesses following the CGA attack model and later online verify candidate list.

### links to password dataset

Here we only provide link to Rockyou dataset. Another password dataset, Neopets, seems still not widespread on the Internet. We can provide Neopets when required upon email.

download the dataset under the directory "/data/"

### run the attacks

attacker.py and run in the command
--model_eval mspm --victim MSPM --physical --withleak --softfilter --logical --spmdata rockyou --exp_pastebinsuffix _bc200 --pin RockYou-6-digit.txt --pinlength 6 --intersection --fixeditv --fixeditvmode 1 --version_gap 1 --isallleaked 0 --gpu 0

the command above uses **MSPM** (--victim) to instantiate Bubble, **rockyou** as the password data set (--spmdata), test set **bc200** (--exp_pastebinsuffix), adopt 6-ditgit PIN (--pinlength), bubble+ (--fixeditv --fixeditvmode 1). The attack conducts both single-version attacks and multi-version attacks (based on two consecutive versions, --version_gap 1 --isallleaked 0)

To modify the experiments, feel free to change the command by referring to the comments below.

### change password vault dataset

1. train pcfg model (markov model is trained based on password dataset)
2. fit unreuse(i) function and parameter "alpha"

### change instantiation DTE (MSPM or Golla)

1. change args.victim

### Commands checklist

0. pcfg model path "--exp_pastebinsuffix" (option: _pb, _bc50, _bc200); 
1. bubble (remove --fixeditv --fixeditvmode 1) or bubble+
2. "PASTB_PATH" setting
3. "ALPHA" and function "unreuse_p(i)" in unreused_prob.py
4. check "incremental_vault_gpu/data/breachcompilation/fold2/fold_0_1.json" testset
5. check "outputdir" in attack.py
6. settings for T, pin, Nitv

