#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

''' This script manages the training
:author: Maximilian Golla
:contact: maximilian.golla@rub.de
:version: 0.7.1, 2019-07-11
'''

# Load external modules
from configs.configure import *
from numba import jit
''' Generates a new ngram-object via init, count, prob, (save) '''


def get_creator(length=6, progress_bar=True):
    CONFIG = Configure({'name': '5-gram'})
    ngram_creator = NGramCreator({
        "name": ("NGramCreator, Session: {}, Length: {}, Progress bar: {}".format(CONFIG.NAME, length, progress_bar)),
        "ngram_size": CONFIG.NGRAM_SIZE,
        "laplace_smooth": CONFIG.LAPLACE_SMOOTH,
        "ls_compensation": CONFIG.LS_COMPENATE,
        "training_file": CONFIG.TRAINING_FILE,
        "length": length,
        "progress_bar": progress_bar,
        "train": CONFIG.TRAIN,
    })

    ngram_creator.load('ip_list')
    ngram_creator.load('ep_list')
    ngram_creator.load('cp_list')
    return ngram_creator


def worker(data):
    "This data was received by the process:"
    length = data[0]
    progress_bar = data[1]

    ngram_creator = NGramCreator({
        "name": ("NGramCreator, Session: {}, Length: {}, Progress bar: {}".format(CONFIG.NAME, length, progress_bar)),
        "ngram_size": CONFIG.NGRAM_SIZE,
        'laplace_smooth': CONFIG.LAPLACE_SMOOTH,
        "ls_compensation": CONFIG.LS_COMPENATE,
        "training_file": CONFIG.TRAINING_FILE,
        "length": length,
        "progress_bar": progress_bar,
        "train": CONFIG.TRAIN,
    })
    ngram_creator._countlines()
    #ngram_creator._init_halfdic()

    # Initial probability (IP)
    logging.debug("ip_list init() ...")
    ngram_creator._init_lists("ip_list")

    logging.debug("ip_list count() ...")
    ngram_creator._count("ip_list")

    logging.debug("ip_list prob() ...")
    ngram_creator._prob("ip_list")

    logging.debug("ip_list save() ...")
    ngram_creator.save("ip_list")

    logging.debug("Training IP done ...")

    # Conditional probability (CP)
    logging.debug("cp_list init() ...")
    ngram_creator._init_lists("cp_list")

    logging.debug("cp_list count() ...")
    ngram_creator._count("cp_list")

    logging.debug("cp_list prob() ...")
    ngram_creator._prob("cp_list")

    logging.debug("cp_list save() ...")
    ngram_creator.save("cp_list")

    logging.debug("Training CP done ...")

    # End probability (EP)
    logging.debug("ep_list init() ...")
    ngram_creator._init_lists("ep_list")

    logging.debug("ep_list count() ...")
    ngram_creator._count("ep_list")

    logging.debug("ep_list prob() ...")
    ngram_creator._prob("ep_list")

    logging.debug("ep_list save() ...")
    ngram_creator.save("ep_list")

    logging.debug("Training EP done ...")


''' Manages the training '''


def train():
    try:
        if train:
            logging.debug("Training started ...")
        else:
            logging.debug("Training for test started ...")

        # ''' Singleprocessing
        for length in CONFIG.LENGTHS:
            data = [length, CONFIG.PROGRESS_BAR]
            worker(data)
            break
        # '''

        ''' Multiprocessing
        data = []
        for length in CONFIG.LENGTHS:
            data.append([length, CONFIG.PROGRESS_BAR])
        pool = multiprocessing.Pool(processes=CONFIG.NO_CPUS)
        pool.map(worker, data)
        pool.close() # no more tasks can be submitted to the pool
        pool.join() # wait for the worker processes to exit
        '''

    except Exception as e:
        sys.stderr.write("\x1b[1;%dm" % (31) + "Training failed: {}\n".format(e) + "\x1b[0m")
        sys.exit(1)


def main():
    try:
        global CONFIG
        CONFIG = Configure({"name": "My Config"}, train=False)
        train()
    except KeyboardInterrupt:
        print('User canceled')
        sys.exit(1)
    except Exception as e:
        sys.stderr.write("\x1b[1;%dm" % (31) + "Error: {}\n".format(e) + "\x1b[0m")
        sys.exit(1)


if __name__ == '__main__':
    print("{0}: {1:%Y-%m-%d %H:%M:%S}\n".format("Start", datetime.datetime.now()))
    print("Press Ctrl+C to shutdown")
    main()
    print("{0}: {1:%Y-%m-%d %H:%M:%S}".format("Done", datetime.datetime.now()))
