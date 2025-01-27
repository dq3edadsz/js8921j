import argparse

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # gpu control
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu id')

        # fixed-interval shuffling
        self.parser.add_argument('--fixeditv', action='store_true',
                            help='using another mode: fixed-interval shuffling or not')
        self.parser.add_argument('--fixeditvmode', type=int, default=0,
                                 help='0, for padding dummy to multiple of Nitv \
                                       1, for no padding')

        # attack
        self.parser.add_argument('--nottrain', action='store_false',
                            help='to use the full (train) model or half model')
        self.parser.add_argument('--predecoys', action='store_true',
                            help='to use the pre generated decoys (shuffle and re-sample for fast attacking)')
        self.parser.add_argument('--model_eval', type=str,
                            help='1.spm, for single password model evaluation \
                                      2.sspm, for single similar model evaluation \
                                      3.mspm, for multi-similar model evaluation')
        self.parser.add_argument('--victim', type=str, default='MSPM',
                            help='1.MSPM, \
                                  2.Golla,')
        self.parser.add_argument('--softfilter', action='store_true')
        self.parser.add_argument('--intersection', action='store_true')
        self.parser.add_argument('--version_gap', type=int, default=1,
                                 help='the gap between versions leaked from the start to the end, consecutive version gap is 1')
        self.parser.add_argument('--isallleaked', type=int, default=0,
                                 help='0, only leak two versions across the gap\
                                       1, leak all versions within the gap')

        # online verification
        self.parser.add_argument('--online_exhaust', action='store_true',
                                 help='use a constant cap or exhaust all entries if necessary')

        # train for sspm
        self.parser.add_argument('--multi_train', action='store_true',
                                 help='train multiple models for sspm')
        self.parser.add_argument('--sspmdata', type=str, default='pastebin',
                                 help='1.rockyou, => train on whole rockyou\
                                       2.pastebin, => train following 5-fold')
        self.parser.add_argument('--pretrained', action='store_true',
                                 help='train with pastebin using model from rockyou')

        # spm training data
        self.parser.add_argument('--spmdata', type=str, default='rockyou',
                                 help='1.rockyou, => spm trained on rockyou\
                                        2.neopets')

        # expand
        self.parser.add_argument('--withleak', action='store_true',
                                 help='experiments with cracking with one leak pw domain')
        self.parser.add_argument('--physical', action='store_true',
                                 help='experiments with physically expanded vault')
        self.parser.add_argument('--logical', action='store_true',
                                 help='experiments with logically expanded vault')

        # data
        self.parser.add_argument('--pin', type=str, default='RockYou-4-digit.txt')
        self.parser.add_argument('--pinlength', type=str, default='')
        self.parser.add_argument('--exp_pastebinsuffix', type=str, default='')
        self.parser.add_argument('--expandtestset', action='store_true',
                                 help='use expanded pastebin as test set')

        # result_batchprocessing
        self.parser.add_argument('--indatapath', type=str, default='')
        self.parser.add_argument('--outdatapath', type=str, default='')

    def parse(self):
        opt = self.parser.parse_args()
        return opt