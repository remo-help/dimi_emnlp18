#!/usr/bin/env python3.4
import time
import pickle
import signal
import sys
import torch
import multiprocessing
#import numpy as np
from .WorkDistributerServer import WorkDistributerServer
from .bounded_pcfg_model import Bounded_PCFG_Model, UnBounded_PCFG_Model
#from .init_pcfg_strategies import *
from .pcfg_model import PCFG_model
from .pcfg_translator import *
from .workers import start_local_workers_with_distributer, start_cluster_workers
from .dimi_io import write_linetrees_file, read_gold_pcfg_file
from collections import Counter, defaultdict
from .dimi import EarlyStopper
# trying something
from .cky_sampler_inner import CKY_sampler


# Has a state for every word in the corpus
# What's the state of the system at one Gibbs sampling iteration?
class Sample:
    def __init__(self):
        self.hid_seqs = []
        self.models = None
        self.log_prob = 0


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1") if type(v) is not bool else v


def wrapped_single_beam(*args, **kwargs):
    try:
        single_node_beam(*args, **kwargs)
    except Exception as e:
        # print(e)
        logging.warning('Sampling beam function has errored out!')
        raise e
        exit(0)


# This is the main entry point for this module.
# Arg 1: ev_seqs : a list of lists of integers, representing
# the EVidence SEQuenceS seen by the user (e.g., words in a sentence
# mapped to ints).
def single_node_beam(ev_seqs, params, working_dir, gold_seqs=None,
                word_dict_file=None, resume=False, eval_sequences=None,
                dev_sequences=None):
    global K
    K = int(params.get('k'))
    sent_lens = list(map(len, ev_seqs))

    max_len = max(map(len, ev_seqs))

    if params.get('max_len', 'None') != 'None':
        if max_len < int(params.get('max_len')):
            max_len = int(params.get('max_len'))
    # vocab_size = max(map(max, ev_seqs)) # vocab_size, which is the max index of the word indices

    f = open(word_dict_file, 'r', encoding='utf-8')
    word_dict = {}
    for line in f:
        (word, index) = line.rstrip().split(" ")
        word_dict[int(index)] = word

    vocab_size = len(word_dict)

    num_sents = len(ev_seqs)
    num_tokens = np.sum(sent_lens)

    total_runtime = 0
    num_samples = 0
    ## Set debug first so we can use it during config setting:
    debug = params.get('debug', 'INFO')
    logfile = params.get('logfile', 'log.txt')

    filehandler = logging.FileHandler(os.path.join(working_dir, 'log.txt'))
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    #logging.basicConfig(level=getattr(logging, debug), format='%(asctime)s %(message)s',
    #                    datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    global D
    D = int(params.get('d', 1))
    iters = int(params.get('iters'))
    try:
        num_cpu_workers = int(params.get('cpu_workers', 0))
    except ValueError as err:
        if params.get('cpu_workers', 0) == 'auto':
            # import multiprocessing
            num_cpu_workers = multiprocessing.cpu_count()
        else:
            raise err
    num_gpu_workers = int(params.get('gpu_workers', 0))

    cluster_cmd = params.get('cluster_cmd', None)
    batch_per_update = min(num_sents, int(params.get('batch_per_update', num_sents)))
    batch_per_worker = min(num_sents, int(params.get('batch_per_worker', 1)))
    gpu = bool(int(params.get('gpu', 0)))
    if gpu and num_gpu_workers < 1 and num_cpu_workers > 0:
        logging.warning("Inconsistent config: gpu flag set with %d gpu workers; setting gpu=False"
                        % (num_gpu_workers))
        gpu = False

    resume_iter = int(params.get("resume_iter", -1))

    init_strategy = params.get("init_strategy", '')
    gold_pos_dict_file = params.get("gold_pos_dict_file", '')

    init_alpha = float(params.get("init_alpha", 1))
    # output settings:
    print_out_first_n_sents = int(params.get('first_n_sents', -1))

    if gold_pos_dict_file:
        gold_pos_dict = {}
        with open(gold_pos_dict_file) as g:
            for line in g:
                line = line.strip().split(' = ')
                gold_pos_dict[int(line[0])] = int(line[1])

    if (gold_seqs != None and 'num_gold_sents' in params):
        logging.info('Using gold tags for %s sentences.' % str(params['num_gold_sents']))

    seed = int(params.get('seed', -1))
    if seed > 0:
        logging.info("Using seed %d for random number generator." % (seed))
        rand = np.random.default_rng(seed=int(seed))
    else:
        logging.info("Using default seed for random number generator.")
        rand = np.random.default_rng()

    logging.info("Total number of tokens: {}, number of nodes: {}".format(sum(sent_lens),
                                                                          sum(sent_lens) * 2))

    iter_logprobs = []
    start_ind = 0
    end_ind = min(num_sents, batch_per_update)
    if eval_sequences:
        eval_start_ind = 0
        eval_end_ind = len(eval_sequences)
        eval_interval = int(params.get('eval_interval', 5))
        logging.info(f"Using eval sequences of length: {len(eval_sequences)}")
        eval_logprob = -np.inf
        if str2bool(params.get('save_evals', True)):
            save_evals = True
            save_logprobs = None
        if dev_sequences:
            dev_start_ind = 0
            dev_end_ind = len(dev_sequences)
            logging.info(f"Using dev sequences of length: {len(dev_sequences)}")
        else:
            dev_start_ind = None
            dev_end_ind = None

    else:
        eval_start_ind = None
        eval_end_ind = None
        eval_interval = None
        evalDistributer = None
        dev_start_ind = None
        dev_end_ind = None
        logging.info(f"eval sequs not enabled")

    if str2bool(params.get('early_stopping', True)):
        tolerance = int(params.get('tolerance', 5))
        best_tolerance = int(params.get('best_tolerance', 10))
        early_stopper = EarlyStopper(tolerance=tolerance, best_tolerance=best_tolerance)
        logging.info(f"Early stopping enabled. Tolerances are: {early_stopper.tolerance} "
                     f"and {early_stopper.best_tolerance}")
    else:
        early_stopper = False
    logging.info("Initializing state: K is {}; D is {}; MaxLen is {}".format(K, D, max_len))

    rnn_model_file = os.path.join(working_dir, 'rnn_model.pkl')

    pcfg_model = PCFG_model(K, D, vocab_size, num_sents, num_tokens, log_dir=working_dir,
                            word_dict_file=word_dict_file, random_generator=rand)
    pcfg_model.set_alpha(alpha=init_alpha)

    if D != -1:
        bounded_pcfg_model = Bounded_PCFG_Model(K, D)
    else:
        bounded_pcfg_model = UnBounded_PCFG_Model(K)

    word_dict = pcfg_model.word_dict
    # print(bounded_pcfg_model.K)

########################################################################
# check if resume necessary
########################################################################
    if not resume:

        dnn_obs_model = None

        hid_seqs = [None] * len(ev_seqs)

        pcfg_model.start_logging()
        # initialization: a few controls:
        pcfg_replace_model(hid_seqs, ev_seqs, bounded_pcfg_model, pcfg_model)

        cur_iter = 0
    else:
        try:
            if resume_iter > 0:
                num_iter = resume_iter
            else:
                pcfg_runtime_stats = open(os.path.join(working_dir, 'pcfg_hypparams.txt'))
                num_iter = int(pcfg_runtime_stats.readlines()[-1].split('\t')[0])
            pcfg_model, dnn_obs_model = torch.load(open(os.path.join(working_dir, 'pcfg_model_' + str(
                num_iter) + '.pkl'), 'rb'))
        except:
            pcfg_model, dnn_obs_model = torch.load(open(os.path.join(working_dir, 'pcfg_model_' + str(
                num_iter - 1) + '.pkl'), 'rb'))

        dnn_obs_model = None
        logging.info("Conitinuing from iteration {}".format(num_iter))
        pcfg_model.set_log_mode('a')
        pcfg_model.start_logging()

        pcfg_replace_model(None, None, bounded_pcfg_model, pcfg_model, resume=True, dnn=dnn_obs_model)

        hid_seqs = [None] * num_sents

        cur_iter = pcfg_model.iter

########################################################################
# set some params
########################################################################
    best_log_prob = -np.inf
    best_eval_prob = -np.inf
    best_test_prob = -np.inf
    best_iter = 0
    best_eval_iter = 0
    eval_logprob = -np.inf
    best_test_save_probs = []
    test_logprob = -np.inf
    warming_period = False
    continue_bool = True
    last_model = False
    best_model = False

########################################################################
# start sampling
########################################################################
    while cur_iter < iters and continue_bool:
        sent_list = []
        pcfg_model.iter = cur_iter