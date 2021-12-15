# @Time    : 9/19/17 5:16 PM
# @Author  : Tiancheng Zhao

from __future__ import print_function

import os
import json
import logging
from datetime import datetime
import torch
# from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import sys
from collections import defaultdict
from argparse import Namespace
import numpy as np


# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt


INT = 0
LONG = 1
FLOAT = 2


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

    @staticmethod
    def msg_from_dict(dictionary, tokenize, speaker2id, bos_id, eos_id, include_domain=False):
        pack = Pack()
        for k, v in dictionary.items():
            pack[k] = v
        pack['speaker'] = speaker2id[pack.speaker]
        pack['conf'] = dictionary.get('conf', 1.0)
        utt = pack['utt']
        if 'QUERY' in utt or "RET" in utt:
            utt = str(utt)
            utt = utt.translate(None, ''.join([':', '"', "{", "}", "]", "["]))
            utt = unicode(utt)
        if include_domain:
            pack['utt'] = [bos_id, pack['speaker'], pack['domain']] + tokenize(utt) + [eos_id]
        else:
            pack['utt'] = [bos_id, pack['speaker']] + tokenize(utt) + [eos_id]
        return pack

def tsne(X, Means, y, fig_path, text_label=None):
    '''
    :param X: numpy: data_num x z_dim
    :param Means: numpy: k x z_dim
    :param y: numpy: data_num x z_dim label
    :return:
    '''
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(np.concatenate((X, Means), axis=0))
    if y is not None:
        df_datapoints = pd.DataFrame(data={'x': X_tsne[:X.shape[0], 0], 'y': X_tsne[:X.shape[0], 1], 'label': y})
        g = sns.lmplot(x='x', y='y', hue='label', data=df_datapoints, fit_reg=False,
                       legend=True)
    else:
        df_datapoints = pd.DataFrame(data={'x': X_tsne[:X.shape[0], 0], 'y': X_tsne[:X.shape[0], 1]})
        g = sns.lmplot(x='x', y='y', data=df_datapoints, fit_reg=False,
                       legend=True)
    ax = g.facet_axis(0, 0)
    # center points
    for i in range(Means.shape[0]):
        ax.scatter(X_tsne[X.shape[0] + i, 0], X_tsne[X.shape[0] + i, 1], color="black")
        plt.text(X_tsne[X.shape[0] + i, 0], X_tsne[X.shape[0] + i, 1], i, color='red',
                 fontsize=10)

    if text_label is not None:
        # print(len(text_label), X.shape[0])

        for i in range(X.shape[0]):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], text_label[i], color='black', fontsize=8)

    plt.savefig(fig_path, dpi=300)
    plt.close()
    return X_tsne[:X.shape[0]], X_tsne[X.shape[0]:]

def process_config(config):
    if config.forward_only:
        config_ori = config

        load_sess = config.load_sess
        backawrd = config.backward_size
        beam_size = config.beam_size
        gen_type = config.gen_type
        data_dir = config.data_dir

        load_path = os.path.join(config.log_dir, load_sess, "params.json")
        config = load_config(load_path)
        config.forward_only = True
        config.load_sess = load_sess
        config.backward_size = backawrd
        config.beam_size = beam_size
        config.gen_type = gen_type
        config.batch_size = 50
        config.data_dir = data_dir

    if "latent_size" in config and config.latent_size > 2:
        config.tsne = True
    else:
        config.tsne = False

    if 'anneal_function' not in config:
        config.anneal = False

    return config

def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if config.forward_only:
        return

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    config.time_stamp = get_time()
    config.script = script
    dir_name = "{}-{}".format(config.time_stamp, script) if script else config.time_stamp
    config.session_dir = os.path.join(config.log_dir, dir_name)
    os.mkdir(config.session_dir)

    fileHandler = logging.FileHandler(os.path.join(config.session_dir,
                                                   'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # save config
    param_path = os.path.join(config.session_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def experiment_name(args):
    exp_name = str()
    exp_name += "%s_" % args.script.split(".")[0]
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "EB=%i_" % args.embed_size
    exp_name += "L=%i_" % args.num_layer
    exp_name += "DCS=%i_" % args.dec_cell_size if 'dec_cell_size' in args else ""
    exp_name += "CCS=%i_" % args.ctx_cell_size if 'ctx_cell_size' in args else ""
    exp_name += "UCS=%i_" % args.utt_cell_size if 'utt_cell_size' in args else ""
    exp_name += "K=%i_" % args.k
    exp_name += "YS=%i_" % args.latent_size
    exp_name += "YN=%i_" % args.mult_k if 'mult_k' in args else ""
    # exp_name += "RNNCELL=%s_"%args.rnn_cell.upper()
    exp_name += "LOSS=%s_" % args.loss_type.upper() if 'loss_type' in args else ""
    exp_name += "BCC=%i_"%args.bi_ctx_cell if 'bi_ctx_cell' in args else ""
    exp_name += "ATT=%i_" % args.use_attribute if 'use_attribute' in args else ""
    exp_name += "BETA=%.2f_" % args.beta if 'beta' in args else ""
    exp_name += "MUTUAL_" if ('use_mutual' in args and args.use_mutual == True) else ""
    exp_name += "MeanField=%s_" % str(args.mean_field) if 'mean_field' in args else ""
    exp_name += "MeanZ=%s_" % str(args.mean_z) if 'mean_z' in args else ""
    exp_name += "SemiW=%.2f_" % args.semi_weight if 'semi_weight' in args else ""
    exp_name += "FEEDC=%s_" % str(args.feed_discrete_variable_into_decoder) if 'feed_discrete_variable_into_decoder' in args else ""
    exp_name += "LabelNLL=%s_" % str(args.use_label_nll) if 'use_label_nll' in args else ""
    exp_name += "Item=%s_" % str(args.cluster_item) if 'cluster_item' in args else ""



    exp_name += "TS=%s"%args.time_stamp


    return exp_name

def load_config(load_path):
    data = json.load(open(load_path, "rb"))
    config = Namespace()
    config.__dict__ = data
    return config


def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def str2bool(v):
    return v.lower() in ('true', '1')


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


def get_dekenize(method="none"):
    if method == "none":
        return lambda x: " ".join(x)
    elif method == "moses":
        # acturally, it's Moses detokenizer in nltk 3.4 (i don't know why it is called tokenize...)
        return lambda x: TreebankWordDetokenizer().tokenize(x)


def get_tokenize():
    return nltk.RegexpTokenizer(r'\w+|#\w+|<\w+>|%\w+|[^\w\s]+').tokenize

def get_chat_tokenize():
    return nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize

def kl_anneal_function(anneal_function, step, k, x0, warmup_step=0, warmup_value=0.0):
    if step <= warmup_step:
        return warmup_value
    step -= warmup_step
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, x0 + step / k)
    elif anneal_function == 'const':
        return k

def interpolate(start, end, steps):
    interpolation = np.zeros((start.shape[0], steps + 2))
    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)
    return interpolation.T

def idx2onehot(idx, n):
    idx = idx.view(-1, 1)
    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx.data, 1)
    return onehot

class missingdict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()

