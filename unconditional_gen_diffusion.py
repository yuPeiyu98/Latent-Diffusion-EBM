#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy
import logging
import datetime


import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


import torch.nn.functional as F
import numpy as np
import h5py
import time
# from optim_n2n import OptimN2N
from data import Dataset
# import utils
import logger
import math


# from torch.utils.tensorboard import SummaryWriter

import torch.nn.utils.spectral_norm as spectral_norm
from collections import OrderedDict, Counter
from dataloader_bases import DataLoader
# from dgmvae import get_chat_tokenize

from sklearn.metrics.cluster import homogeneity_score
import subprocess

from ldebm import LEBM

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_from', default='')


# SRI options
parser.add_argument('--z_n_iters', type=int, default=20)
parser.add_argument('--z_step_size', type=float, default=0.5)
parser.add_argument('--z_with_noise', type=int, default=0)
parser.add_argument('--num_z_samples', type=int, default=10)


# EBM
parser.add_argument('--prior_hidden_dim', type=int, default=200)
parser.add_argument('--z_prior_with_noise', type=int, default=1)
parser.add_argument('--prior_step_size', type=float, default=2e-3)
parser.add_argument('--z_n_iters_prior', type=int, default=50)
parser.add_argument('--max_grad_norm_prior', default=10, type=float)
parser.add_argument('--ebm_reg', default=0.001, type=float) # works better with `--ebm_reg=0.0`
parser.add_argument('--ref_dist', default='gaussian', type=str, choices=['gaussian', 'uniform'])
parser.add_argument('--ref_sigma', type=float, default=1.)
parser.add_argument('--init_factor', type=float, default=1.)



# Model options
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--dec_word_dim', default=512, type=int)
parser.add_argument('--dec_h_dim', default=512, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.2, type=float)
parser.add_argument('--model', default='abp', type=str, choices = ['abp', 'vae'])
parser.add_argument('--train_n2n', default=1, type=int)
parser.add_argument('--train_kl', default=1, type=int)

# Optimization options
parser.add_argument('--checkpoint_dir', default='models/ptb')
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--min_epochs', default=15, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--decay', default=0, type=int)
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--prior_lr', default=0.0001, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--seed', default=8888, type=int)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--sample_every', type=int, default=1000)
parser.add_argument('--kl_every', type=int, default=100)
parser.add_argument('--compute_kl', type=int, default=1)
parser.add_argument('--max_kl_weight', type=float, default=0.1)
parser.add_argument('--mutual_weight', type=float, default=0.05)
parser.add_argument('--test', type=int, default=0)


# corpus config
parser.add_argument('--max_utt_len', type=int, default=40)
parser.add_argument('--data_dir', type=str, default='data/daily_dialog')
parser.add_argument('--max_vocab_cnt', type=int, default=10000)
parser.add_argument('--fix_batch', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--backward_size', type=int, default=5)
parser.add_argument('--num_cls', type=int, default=125)
parser.add_argument('--n_per_cls', type=int, default=3)
parser.add_argument('--debug', type=bool, default=False)

# KL-annealing
parser.add_argument('--anneal', type=bool, default=False) #TODO (bp): anneal KL weight
parser.add_argument('--anneal_function', type=str, default='logistic')
parser.add_argument('--anneal_k', type=float, default=0.0025)
parser.add_argument('--anneal_x0', type=int, default=2500)
parser.add_argument('--anneal_warm_up_step', type=int, default=0)
parser.add_argument('--anneal_warm_up_value', type=float, default=0.000)
parser.add_argument('--pretrain_ae_step', type=int, default=0)
parser.add_argument('--ae_epochs', type=int, default=8)
parser.add_argument('--dim_target_kl', type=float, default=1.0)


##------------------------------------------------------------------------------------------------------------------##
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"



class Pack(OrderedDict):
# class Pack(dict):
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


class DailyDialogCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'train'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'validation'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'test'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(os.path.join(path, 'dialogues.txt'), 'r') as f:
            txt_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_act.txt'), 'r') as f:
            da_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_emotion.txt'), 'r') as f:
            emotion_lines = f.readlines()

        combined_data = [(t, d, e) for t, d, e in zip(txt_lines, da_lines, emotion_lines)]

        return self._process_dialog(combined_data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        for raw_dialog, raw_act, raw_emotion in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]

            # raw_dialog = raw_dialog.decode('ascii', 'ignore').encode()
            raw_dialog = raw_dialog.split('__eou__')[0:-1]
            raw_act = raw_act.split()
            raw_emotion = raw_emotion.split()

            for t_id, turn in enumerate(raw_dialog):
                utt = turn
                utt = [BOS] + self.tokenize(utt.lower()) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=t_id%2,
                                   meta={'emotion': raw_emotion[t_id], 'act': raw_act[t_id]}))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class DailyDialogSkipLoaderLabel(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogSkipLoaderLabel, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array


        z_labels = np.zeros((self.batch_size, 2), dtype=np.int32)
        for b_id in range(self.batch_size):
            z_labels[b_id][0] = int(metas[b_id]["emotion"])
            z_labels[b_id][1] = int(metas[b_id]["act"])

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens,
                    z_labels=z_labels)

##------------------------------------------------------------------------------------------------------------------##


class LM(nn.Module):
    def __init__(self, vocab_size=10000, word_dim=512, h_dim=1024, num_layers=1):
        super(LM, self).__init__()
        self.word_vecs = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.LSTM(word_dim, h_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(*[nn.Linear(h_dim, vocab_size), nn.LogSoftmax(dim=-1)])

    def forward(self, sent, training=True):
        word_embed = F.dropout(self.word_vecs(sent[:, :-1]), training=training, p=0.5)
        rnn_out, _ = self.rnn(word_embed)
        rnn_out = F.dropout(rnn_out, training=training, p=0.5).contiguous()
        preds = self.linear(rnn_out)
        return preds


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


##------------------------------------------------------------------------------------------------------------------##
class RNNVAE(nn.Module):
    def __init__(self, args, vocab_size=10000,
                 enc_word_dim=512,
                 enc_h_dim=1024,
                 enc_num_layers=1,
                 dec_word_dim=512,
                 dec_h_dim=1024,
                 dec_num_layers=1,
                 dec_dropout=0.5,
                 latent_dim=32,
                 max_sequence_length=40):
        super(RNNVAE, self).__init__()
        self.args = args
        self.enc_h_dim = enc_h_dim
        self.enc_num_layers = enc_num_layers
        self.dec_h_dim = dec_h_dim
        self.dec_num_layers = dec_num_layers
        self.embedding_size = dec_word_dim
        self.dropout = dec_dropout
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

        # encoder
        self.enc_word_vecs = nn.Embedding(vocab_size, enc_word_dim)
        self.enc_latent_linear_mean = nn.Linear(enc_h_dim*2, latent_dim)
        self.enc_latent_linear_logvar = nn.Linear(enc_h_dim*2, latent_dim)
        self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers,
                                batch_first=True, bidirectional=True)
        self.enc = nn.ModuleList([self.enc_word_vecs, self.enc_rnn,
                                    self.enc_latent_linear_mean, self.enc_latent_linear_logvar])

        # decoder
        self.dec_word_vecs = nn.Embedding(vocab_size, dec_word_dim)
        dec_input_size = dec_word_dim
        dec_input_size += latent_dim
        self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers=dec_num_layers,
                               batch_first=True)
        self.dec_linear = nn.Sequential(*[nn.Linear(dec_h_dim, vocab_size),
                                          nn.LogSoftmax(dim=-1)])
        self.dec = nn.ModuleList([self.dec_word_vecs, self.dec_rnn, self.dec_linear])

        # decoder hidden state init
        if latent_dim > 0:
            self.latent_hidden_linear_h = nn.Linear(latent_dim, dec_h_dim)
            self.latent_hidden_linear_c = nn.Linear(latent_dim, dec_h_dim)
            self.dec.append(self.latent_hidden_linear_h)
            self.dec.append(self.latent_hidden_linear_c)

        # ebm prior
        self.prior_dim = self.latent_dim
        self.prior_hidden_dim = args.prior_hidden_dim

        self.prior_network = LEBM(
                feat_dim=latent_dim,
                ebm_hdim=args.prior_hidden_dim,
                emb_dim=args.prior_hidden_dim,
                emb_hdim=args.prior_hidden_dim,
                num_cls=args.num_cls,
                num_blocks=12,
                max_T=6,
                use_spc_norm=True,
                e_l_steps=args.z_n_iters_prior,
                e_l_step_size=args.prior_step_size,
                langevin_noise_scale=1.,
            )    

    def sample_amortized_posterior_sample(self, mean, logvar, z=None, sample=True):
        if sample:
            std = logvar.mul(0.5).exp()    
            if z is None:
                z = torch.cuda.FloatTensor(std.size()).normal_(0, 1)
            return z.mul(std) + mean
        else:
            return mean

    def encoder(self, x, args):
        word_vecs = self.enc_word_vecs(x)
        h0 = torch.zeros(self.enc_num_layers*2, word_vecs.size(0), self.enc_h_dim).type_as(word_vecs.data)
        c0 = torch.zeros(self.enc_num_layers*2, word_vecs.size(0), self.enc_h_dim).type_as(word_vecs.data)
        enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
        enc_h_states_last = enc_h_states[:, -1]
        mean = self.enc_latent_linear_mean(enc_h_states_last)
        logvar = self.enc_latent_linear_logvar(enc_h_states_last)
        return mean, logvar    

    def decoder(self, sent, q_z, init_h=True, training=True, dropout=0.2):
        self.word_vecs = F.dropout(self.dec_word_vecs(sent[:, :-1]), training=training, p=dropout)
        if init_h:
            self.h0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
            self.c0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        if q_z is not None:
            q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0),
                                                 self.word_vecs.size(1), q_z.size(1))
            dec_input = torch.cat([self.word_vecs, q_z_expand], 2)
        else:
            dec_input = self.word_vecs
        if q_z is not None:
            self.h0[-1] = self.latent_hidden_linear_h(q_z)
            self.c0[-1] = self.latent_hidden_linear_c(q_z)

        memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
        dec_linear_input = memory.contiguous()
        dec_linear_input = F.dropout(dec_linear_input, training=training, p=dropout)
        preds = self.dec_linear(dec_linear_input.view(
            self.word_vecs.size(0) * self.word_vecs.size(1), -1)).view(
            self.word_vecs.size(0), self.word_vecs.size(1), -1)
        return preds

    def inference(self, device, sos_idx, max_len=None, z=None, init_h=True, training=False):
        batch_size = z.size(0)

        if init_h:
            self.h0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
            self.c0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        self.h0[-1] = self.latent_hidden_linear_h(z)
        self.c0[-1] = self.latent_hidden_linear_c(z)

        if max_len is None:
            max_len = self.max_sequence_length
        generations = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        preds_sequence = torch.zeros(batch_size, max_len, self.vocab_size, dtype=torch.float, device=device)
        input_sequence = torch.tensor([sos_idx]*batch_size, dtype=torch.long, device=device)

        hidden = (self.h0, self.c0)
        for i in range(max_len):
            input_embedding = F.dropout(self.dec_word_vecs(input_sequence).view(batch_size, 1, self.embedding_size), training=training)
            dec_input = torch.cat([input_embedding, z.view(batch_size, 1, self.latent_dim)], dim=2)  #TODO: project z to embedding space before concat?
            output, hidden = self.dec_rnn(dec_input, hidden)
            dec_linear_input = output.contiguous()
            dec_linear_input = F.dropout(dec_linear_input, training=training) #TODO: this dropout is necessary?
            preds = self.dec_linear(dec_linear_input.view(batch_size, self.dec_h_dim))
            probs = F.softmax(preds, dim=1)
            samples = probs.argmax(dim=1)
            generations[:, i] = samples.view(-1).data
            preds_sequence[:, i, :] = preds
            input_sequence = samples.view(-1)

        return generations, preds_sequence

##--------------------------------------------------------------------------------------------------------------------##
def main(args, output_dir):
    set_seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)


    corpus_client = DailyDialogCorpus(args)
    corpus = corpus_client.get_corpus()
    train_dial = corpus['train']
    test_dial = corpus['test']
    train_feed = DailyDialogSkipLoaderLabel("Train", train_dial, args)
    test_feed = DailyDialogSkipLoaderLabel("Test", test_dial, args)

    vocab_size = len(corpus_client.vocab)

    checkpoint_dir = output_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    suffix = "%s_%s.pt" % (args.model, 'bl')
    checkpoint_path = os.path.join(checkpoint_dir, suffix)

    writer = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    reverse_lm = LM(vocab_size=vocab_size, word_dim=args.dec_word_dim, h_dim=args.dec_h_dim, num_layers=args.dec_num_layers)
    reverse_lm.cuda()

    if args.train_from == '':
        model = RNNVAE(args, vocab_size = vocab_size,
                       dec_word_dim = args.dec_word_dim,
                       dec_h_dim = args.dec_h_dim,
                       dec_num_layers = args.dec_num_layers,
                       dec_dropout = args.dec_dropout,
                       latent_dim = args.latent_dim)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
    else:
        logger.info('loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from)
        model = checkpoint['model']

    logger.info("model architecture")
    print(model)

    prior_params = [p[1] for p in model.named_parameters() if 'prior' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'prior' not in p[0] and p[1].requires_grad is True]


    optimizer = torch.optim.Adam([
                                    {'params': prior_params, 'lr': args.prior_lr, 'weight_decay': args.ebm_reg},
                                    {'params': likelihood_params},
                                 ],
                                 lr=args.lr)


    if args.warmup == 0:
        args.beta = 1.
    else:
        args.beta = 0.001

    criterion = nn.NLLLoss(ignore_index=corpus_client.rev_vocab[PAD], reduction='sum')
    model.cuda()
    # criterion.cuda()
    model.train()

    if args.test == 1:
        args.beta = 1
        test_data = Dataset(args.test_file)
        eval(test_data, model)
        exit()

    t = 0
    best_val_nll = 1e5
    best_epoch = 0
    val_stats = []
    epoch = 0
    z_means = torch.zeros(5, args.latent_dim, device=device, dtype=torch.float)

    i = 0
    vae_kl_weights = frange_cycle_zero_linear(30*2905, start=0.0, stop=args.max_kl_weight,
                                                n_cycle=10,
                                                ratio_increase=0.25,
                                                ratio_zero=0.5)

    while epoch < args.num_epochs:
        start_time = time.time()
        epoch += 1
        logger.info('Starting epoch %d' % epoch)
        train_nll_abp = 0.
        num_sents = 0
        num_words = 0
        b = 0

        train_feed.epoch_init(args, shuffle=True)

        while True:
            # i += 1
            data_feed = train_feed.next_batch()
            if data_feed is None:
                break

            if args.debug and b > 100:
                break

            batch_size = len(data_feed['output_lens'])
            sents = torch.tensor(data_feed['outputs'])

            if args.gpu >= 0:
                sents = sents.cuda().long()
            vae_kl_weight = args.max_kl_weight # vae_kl_weights[i]
            ae_train = True if vae_kl_weight == 0.0 else False
            b += 1
            i += 1

            # generator update
            optimizer.zero_grad()

            mu, log_var = model.encoder(sents, args)
            if ae_train:
                z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=False)
            else:
                z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=True)
            preds = model.decoder(sents, z_samples, dropout=args.dec_dropout)
            labels = sents[:, 1:].contiguous()
            nll_abp =  criterion(preds.transpose(1, 2), labels) / batch_size          

            x_pos, x_neg, x_T_1, x_T, t = model.prior_network.sample_latent_training_pairs(
                                            z_samples)

            # KL(q(z|x) || p(z))
            # E_q(z|x) (f(z))
            log_pos, neg_normal_ll = model.prior_network.get_prior_loss(x_pos, x_T_1, t)
            loss_g_kl = - 0.5 * (1 + log_var)
            kl_mask = (loss_g_kl > args.dim_target_kl).float()
            zkl = (kl_mask * loss_g_kl).sum(dim=1).mean()                        
            zkl += log_pos + neg_normal_ll

            # E_p(z) (f(z))
            cd = model.prior_network.get_ebm_loss(x_pos, x_neg, x_T_1, x_T, t)

            # IB
            t0 = torch.zeros(
                    z_samples.size(0), dtype=torch.long, device=z_samples.device)
            z0 = model.prior_network._q_sample(
                    x_start=z_samples,
                    t=t0
                ) * model.prior_network._extract(model.prior_network.a_s_prev, t0 + 1, z_samples.shape)
            mi = model.prior_network.compute_mi(z0, t0)

            results = Pack(nll=nll_abp, zkl=zkl, cd=cd, mi=mi)

            if ae_train:
                abp_loss = nll_abp
            else:
                abp_loss = nll_abp + vae_kl_weight * (zkl + cd) \
                         - args.mutual_weight * mi

            optimizer.zero_grad()

            abp_loss.backward()

            if args.max_grad_norm > 0:
                llhd_grad_norm = torch.nn.utils.clip_grad_norm_(likelihood_params, args.max_grad_norm)
            else:
                llhd_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.clone().detach()) for p in likelihood_params]))

            if args.max_grad_norm_prior > 0:
                prior_grad_norm = torch.nn.utils.clip_grad_norm_(prior_params, args.max_grad_norm_prior)
            else:
                prior_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.clone().detach()) for p in prior_params]))

            optimizer.step()

            num_sents += batch_size

            if b % args.print_every == 0:
                logger.info(
                    'batch/max_batch/ep: {:5d}/ {:5d}/ {:5d} '.format(
                        b, train_feed.num_batch, epoch) +
                    'rec: {:8.3f} '.format(nll_abp) +
                    'zkl: {:8.3f} '.format(zkl) + 
                    'cd: {:8.3f} '.format(cd) +
                    'mi: {:10.8f} '.format(mi) + 
                    'kl_weight: {:8.3f} '.format(vae_kl_weight) +
                    'do_ae_train: {}'.format(str(not vae_kl_weight > 0.0))
                )

        epoch_train_time = time.time() - start_time
        logger.info('Time Elapsed: %.1fs' % epoch_train_time)

        if epoch > 0:
            logger.info('---')
            logger.info('---')
            logger.info('---')
            compute_homogeneity(model, test_feed, args)
            logger.info('---')
            logger.info('---')
            logger.info('---')

        # get_cluster_examples(args, model, test_feed, corpus_client, epoch=epoch)
        logger.info('--------------------------------')
        logger.info('Checking train perf...')
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Mode', 'Train')
        logger.record_tabular('LR', args.lr)
        logger.record_tabular('Epoch Train Time', epoch_train_time)
        train_nll = eval(args, train_feed, model, corpus_client, mode='train')

        logger.info('--------------------------------')
        logger.info('Checking test perf...')
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Mode', 'Test')
        logger.record_tabular('LR', args.lr)
        logger.record_tabular('Epoch Train Time', epoch_train_time)
        test_nll = eval(args, test_feed, model, corpus_client, epoch=epoch)

##--------------------------------------------------------------------------------------------------------------------##
##--------------------------------------------------------------------------------------------------------------------##

def get_sent(ids, vocab, stop_eos=True, stop_pad=True):
    ws = []
    for w_id in ids:
        w = vocab[w_id]
        if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
            if w == EOT:
                ws.append(w)
            break
        if w != PAD:
            ws.append(w)
    return " ".join(ws)

    

def eval(args, test_feed, model, corpus_client, epoch=0, mode='test'):
    model.eval()
    rev_vocab = corpus_client.rev_vocab
    vocab = corpus_client.vocab
    criterion = nn.NLLLoss(ignore_index=rev_vocab[PAD], reduction='sum')
    total_nll_abp = 0.
    num_sents = 0

    test_feed.epoch_init(args, shuffle=False)
    i = 0

    hyps_f = open(os.path.join(output_dir, "hyp-" + mode), "w")
    refs_f = open(os.path.join(output_dir, "ref-" + mode), "w")
    if mode == 'test':
        hyps_prior_f = open(os.path.join(output_dir, "hyp-prior-" + mode), "w")

    root_dir = os.getcwd()
    perl_path = os.path.join(root_dir, "multi-bleu.perl")

    while True:
        i += 1
        data_feed = test_feed.next_batch()
        if data_feed is None:
            break
        if mode != 'test' and i > 200:
            break
        
        batch_size = len(data_feed['output_lens'])
        sents = torch.tensor(data_feed['outputs'])
        sents = sents.cuda().long()
        num_sents += batch_size
        device = sents.device

        mu, log_var = model.encoder(sents, args)
        z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=False)
        preds = model.decoder(sents, z_samples, dropout=args.dec_dropout)
        labels = sents[:, 1:].contiguous()
        nll_abp =  criterion(preds.transpose(1, 2), labels) / batch_size
        total_nll_abp += nll_abp.item()*batch_size        

        recons_sample_ids, _ = model.inference(device, rev_vocab[BOS], z=z_samples)
        
        if mode == 'test':
            z_e_0 = torch.randn(
                    *[batch_size, args.latent_dim]
                ).cuda()

            zs = model.prior_network.p_sample_progressive(z_e_0)            
            prior_sample_ids, _ = model.inference(device, rev_vocab[BOS], z=zs)

        for s, pred_ids in enumerate(recons_sample_ids):
            pred_str = get_sent(pred_ids.cpu().numpy(), vocab)
            true_str = get_sent(labels[s].cpu().numpy(), vocab)

            hyps_f.write(pred_str.strip() + "\n")
            refs_f.write(true_str.strip() + "\n")

            if mode == 'test':
                prior_str = get_sent(prior_sample_ids[s].cpu().numpy(), vocab)
                hyps_prior_f.write(prior_str.strip() + "\n")

    rec_abp = total_nll_abp / num_sents
    logger.record_tabular('ABP REC', rec_abp)
    logger.dump_tabular()
    
    # compute bleu
    hyps_f.close()
    refs_f.close()
    p = subprocess.Popen(["perl", perl_path, os.path.join(output_dir, "ref-"+mode)], stdin=open(os.path.join(output_dir, "hyp-"+mode), "r"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    while p.poll() == None:
        pass
    print("multi-bleu.perl return code: ", p.returncode)

    # fout = open(file_name, "a")
    for line in p.stdout:
        line = line.decode("utf-8")
        if line[:4] == "BLEU":
            # sys.stdout.write(line)
            # fout.write(line)
            logger.info('---')
            logger.info('---')
            logger.info(line)
            logger.info('---')
            logger.info('---')


    if mode == 'test':
        # compute bleu
        hyps_prior_f.close()
        refs_f.close()
        p = subprocess.Popen(["perl", perl_path, os.path.join(output_dir, "ref-"+mode)], stdin=open(os.path.join(output_dir, "hyp-prior-"+mode), "r"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        while p.poll() == None:
            pass
        print("multi-bleu.perl return code: ", p.returncode)

        # fout = open(file_name, "a")
        for line in p.stdout:
            line = line.decode("utf-8")
            if line[:4] == "BLEU":
                # sys.stdout.write(line)
                # fout.write(line)
                logger.info('---')
                logger.info('---')
                logger.info('---prior sample---')
                logger.info(line)
                logger.info('---')
                logger.info('---')    

    model.train()
    return rec_abp


##--------------------------------------------------------------------------------------------------------------------##
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

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.25, ratio_zero=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L

def get_chat_tokenize():
    import nltk
    return nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize

def get_cluster_examples(args, model, test_feed, corpus_client, epoch=0, max_samples=5):
    from collections import defaultdict

    model.eval()
    vocab = corpus_client.vocab
    test_feed.epoch_init(args, shuffle=False)    
    cluster_dict = defaultdict(list)
    
    while True:
        data_feed = test_feed.next_batch()
        if data_feed is None:
            break

        true_labels = data_feed['z_labels']
        act_labels = true_labels[:, 0]
        emt_labels = true_labels[:, 1]

        sents = torch.tensor(data_feed['outputs'])
        sents = sents.cuda().long()

        mu, log_var = model.encoder(sents, args)
        z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=True)
        t0 = torch.zeros(
                z_samples.size(0), dtype=torch.long, device=z_samples.device
            )
        pred_logits = model.prior_network.ebm_prior(z_samples, t0, use_cls_output=True)
        pred_labels = pred_logits.argmax(dim=-1).cpu()

        for b_id, sent in enumerate(sents):
            act_label = act_labels[b_id]
            emt_label = emt_labels[b_id]
            pred_label = '{:0>3d}'.format(pred_labels[b_id])
            sent_text = get_sent(sent.cpu().numpy(), vocab)
            save_str = 'act: {:0>2d}, emt: {:0>2d}, text: {}'.format(act_label, emt_label, sent_text)
            cluster_dict[pred_label].append(save_str)

    keys = cluster_dict.keys()
    keys = sorted(keys)
    logger.info("Find {} clusters".format(len(keys)))

    cluster_f = open(os.path.join(output_dir, "cluster-examples-{:0>2d}".format(epoch)), "w")
    for symbol in keys:
        sents = cluster_dict[symbol]
        if len(sents) < max_samples:
            subset_ids = list(range(len(sents)))
            np.random.shuffle(subset_ids)
        else:
            subset_ids = np.random.choice(range(len(sents)), max_samples, replace=False)
        
        cluster_f.write('<--------------{}-------------->'.format(symbol) + "\n")
        for s_id in subset_ids:
            cluster_f.write(sents[s_id] + "\n")
    cluster_f.close()


def compute_homogeneity(model, test_feed, args):
    test_feed.epoch_init(args, shuffle=False)
    act_scores = []
    emt_scores = []
    batch_sizes = []
    while True:
        data_feed = test_feed.next_batch()
        if data_feed is None:
            break

        labels = data_feed['z_labels']
        act_labels = labels[:, 0]
        emt_labels = labels[:, 1]
        batch_size = labels.shape[0]

        sents = torch.tensor(data_feed['outputs'])
        sents = sents.cuda().long()

        mu, log_var = model.encoder(sents, args)
        z_samples = model.sample_amortized_posterior_sample(mu, log_var, sample=True)
        t0 = torch.zeros(
                z_samples.size(0), dtype=torch.long, device=z_samples.device
            )
        pred_logits = model.prior_network.ebm_prior(z_samples, t0, use_cls_output=True)
        pred_labels = pred_logits.argmax(dim=-1).cpu()

        act_score = homogeneity_score(act_labels, pred_labels)
        emt_score = homogeneity_score(emt_labels, pred_labels)

        act_scores.append(act_score * batch_size)
        emt_scores.append(emt_score * batch_size)
        batch_sizes.append(batch_size)

    act_scores = sum(act_scores)
    emt_scores = sum(emt_scores)
    total_size = sum(batch_sizes)

    act_scores = act_scores / total_size
    emt_scores = emt_scores / total_size

    logger.info('------ act homogeneity {} '.format(act_scores)+ 
                'emt homogeneity {}'.format(emt_scores))

def sample_p_0(x, args):
    if args.ref_dist == 'gaussian':
        return args.init_factor * torch.randn(*[x.size(0), args.latent_dim], device=x.device)
    else:
        return torch.Tensor(*[x.size(0), args.latent_dim]).uniform_(-1, 1).to(x.device)

def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]

def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

def set_seed(seed):
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parser.parse_args()
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)

    set_gpu(args.gpu)

    with logger.session(dir=output_dir, format_strs=['stdout', 'csv', 'log']):
        main(args, output_dir)
