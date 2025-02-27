#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

''' This script configures the Markov model
:author: Maximilian Golla
:contact: maximilian.golla@rub.de
:version: 0.7.1, 2019-07-11
'''

# Load external modules
import sys, logging, json, datetime
from threading import Thread
from MSPM.mspm_config import SOURCE_PATH
# Load own modules
from Golla.SPM.multiprocessinglog import *
from Golla.SPM.ngram_creator import *

# Global variables
mtlog = MultiProcessingLog('foo.log', 'a', 0, 0)
logger = logging.getLogger()
logger.addHandler(mtlog)
logger.setLevel(logging.DEBUG) # DEBUG, INFO, CRITICAL
logger = multiprocessing.log_to_stderr(logging.INFO) # DEBUG, INFO, CRITICAL

class Configure:

    def __init__(self, dict, train=True):
        self.name = dict['name']
        self.train = train
        logging.debug("Constructor started for '{}'".format(self.name))
        self._read_config()
        self.EVAL_FILE = None

    def _read_config(self):
        try:
            with open(SOURCE_PATH + '/MSPM/SPM/configs/dev.json', 'r') as configfile:
                config = json.load(configfile)
                # Those DEFAULTS are used, if the config file is malformed
                self.NAME = config.get("name", "Demo")
                self.EVAL_FILE = config.get("eval_file", "eval.txt")
                self.TRAINING_FILE = config.get("training_file", "training.txt")
                self.ALPHABET = config.get("alphabet", "abcdefghijklmnopqrstuvwxyz")
                self.LENGTHS = config.get("lengths", [6,8])
                self.NGRAM_SIZE = config.get("ngram_size", 3)
                self.LAPLACE_SMOOTH = config.get('laplace_smooth')
                # reciprocal of expected laplace smooth (e.g., 0.001 => 1000 for count when compensation)
                self.LS_COMPENATE = config.get('laplace_smooth_compensation_scaler')
                self.NO_CPUS = config.get("no_cpus", 8)
                self.PROGRESS_BAR = config.get("progress_bar", False)
                self.TRAIN = self.train
        except Exception as e:
            sys.stderr.write("\x1b[1;%dm" % (31) + "Malformed config file: {}\n".format(e) + "\x1b[0m")
            sys.exit(1)
