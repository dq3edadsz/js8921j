import sys
import os
import json
import random
import math

file_name = 'weir_vaultcleaned.json'
f = open(file_name)
data = json.load(f)
data_filtered = {}
for d in data:
    if (len(data[d]) > 4 and len(data[d]) < 51):  # adjust this to cater need of online verification trials
        data_filtered[d] = data[d]
data = data_filtered

# repeat the process below for "repeat_num" times
repeat_num = 10
num_files = 2 # split number
data_len = [round(len(data)/num_files) if i < (num_files-1) else (len(data)-(num_files-1)*round(len(data)/num_files)) for i in range(num_files)]
len_cu = [sum(data_len[0:x:1]) for x in range(0, num_files+1)] # sum([]) = 0
# create directory (after checking if it exists)
if not os.path.exists('fold' + str(num_files) + '_Nge5'):
    os.makedirs('fold' + str(num_files) + '_Nge5')
# list all the files in the directory
file_list = os.listdir('fold' + str(num_files) + '_Nge5')
for rn in range(repeat_num):
    # initialize 2D array
    key_lst = list(data.keys())
    random.shuffle(key_lst)

    # loop through 2D array
    for i in range(0, num_files):
        # create file when section is complete
        filen = 'fold_'+str(rn) + '_' + str(i+1) + '.json'
        if filen in file_list:
            print('file already exists:', filen, ' skipping...')
            continue
        name = os.path.join('fold' + str(num_files) + '_Nge5', filen)
        sub_key = key_lst[len_cu[i]:len_cu[i+1]]
        sub_data = {sk:data[sk] for sk in sub_key}
        with open(name, 'w') as outfile:
            json.dump(sub_data, outfile)