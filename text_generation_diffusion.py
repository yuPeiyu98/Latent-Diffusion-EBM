from __future__ import print_function

import logging
import os
import json
import random

from dgmvae import evaluators, utt_utils
from dgmvae.dataset import corpora
from dgmvae.dataset import data_loaders
from dgmvae.models.sent_models import *
from dgmvae.utils import get_time, Pack
from dgmvae.multi_bleu import multi_bleu_perl
from dgmvae.options import get_parser

from ldebm import LEBM

logger = logging.getLogger()


#------------------------------ utils -----------------------------------#
import math
def log_gaussian(z, mean=None, log_var=None):
    assert len(z.size()) == 2
    if mean is None:
        mean = torch.zeros_like(z)
    if log_var is None:
        log_var = torch.zeros_like(z)
    
    log_p = - (z - mean) * (z - mean) / (2 * torch.exp(log_var) - 0.5 * log_var - 0.5 * math.log(math.pi * 2))
    
    return log_p.sum(dim=-1)



def set_seed(seed, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


#------------------------------ model -----------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgmvae.dataset.corpora import PAD, BOS, EOS, UNK
from torch.autograd import Variable
from dgmvae import criterions
from dgmvae.enc2dec.decoders import DecoderRNN
from dgmvae.enc2dec.encoders import EncoderRNN
from dgmvae.utils import INT, FLOAT, LONG, cast_type
from dgmvae import nn_lib
import numpy as np
from dgmvae.models.model_bases import BaseModel
from dgmvae.enc2dec.decoders import GEN, TEACH_FORCE
from dgmvae.utils import Pack, kl_anneal_function, interpolate, idx2onehot
import itertools
import math

from dgmvae.models.model_bases import summary

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class GMVAE(BaseModel):
    def __init__(self, corpus, config):
        super(GMVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.pad_id = self.rev_vocab[PAD]
        self.num_layer_enc = config.num_layer_enc
        self.num_layer_dec = config.num_layer_dec
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])
        self.dec_embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                          padding_idx=self.rev_vocab[PAD])
        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)
        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size + self.config.latent_size if self.concat_decoder_input else self.embed_size,
                                  self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer_dec, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  tie_output_embed=config.tie_output_embed,
                                  embedding=self.dec_embedding)
        self.q_y_mean = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.enc_out_size, config.latent_size),
            # nn.LeakyReLU(),
            # nn.Linear(config.latent_size, config.latent_size)
        )
        self.q_y_logvar = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.enc_out_size, config.latent_size),
            # nn.LeakyReLU(),
            # nn.Linear(config.latent_size, config.latent_size)
        )
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)

        self.ebm = LEBM(
                feat_dim=config.latent_size,
                ebm_hdim=config.ebm_hidden,
                emb_dim=config.ebm_hidden,
                emb_hdim=config.ebm_hidden,
                num_cls=config.num_cls,
                num_blocks=12,
                max_T=6,
                use_spc_norm=True,
                e_l_steps=config.e_l_steps,
                e_l_step_size=config.e_l_step_size,
                langevin_noise_scale=1.,
            )


        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids', 'z')
        self.kl_w = 0.0

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=40, help="The latent size of continuous latent variable.")
        parser.add_argument('--mult_k', type=int, default=20, help="The number of discrete latent variables.")
        parser.add_argument('--k', type=int, default=5, help="The dimension of discrete latent variable.")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=True)
        parser.add_argument('--num_layer_enc', type=int, default=1)
        parser.add_argument('--num_layer_dec', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Dispersed GMVAE settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--beta', type=float, default=0.2)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=True)
        parser.add_argument('--klw_for_ckl', type=float, default=1.0)
        parser.add_argument('--klw_for_zkl', type=float, default=1.0)
        parser.add_argument('--pretrain_ae_step', type=int, default=0)

        # lsebm
        parser.add_argument('--ebm_hidden', type=int, default=200)
        parser.add_argument('--e_l_steps', type=int, default=50)
        parser.add_argument('--e_prior_sig', type=float, default=1.)
        parser.add_argument('--e_l_step_size', type=float, default=2e-3)
        parser.add_argument('--e_l_with_noise', type=bool, default=True)

        parser.add_argument('--dim_target_kl', type=float, default=1.0)
        parser.add_argument('--mutual_weight', type=float, default=0.0)
        parser.add_argument('--num_cls', type=int, default=20)
        parser.add_argument('--max_kl_weight', type=float, default=0.1)
        parser.add_argument('--n_cycle', type=int, default=10)
        parser.add_argument('--ratio_increase', type=float, default=0.25)
        parser.add_argument('--ratio_zero', type=float, default=0.5)


        return parser

    def _set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_optimizer(self, config):
        if config.op == 'adam':
            return torch.optim.Adam([
                                     {'params': [p[1] for p in self.named_parameters() if 'ebm' not in p[0] and p[1].requires_grad]},
                                     {'params': [p[1] for p in self.named_parameters() if 'ebm' in p[0] and p[1].requires_grad], 'lr': 0.0001},
                                     ],
                                     lr=config.init_lr)

    def model_sel_loss(self, loss, batch_cnt):
        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        if "sel_metric" in self.config and self.config.sel_metric == "elbo":
            return loss.elbo
        return self.valid_loss(loss)

    def freeze_recognition_net(self):
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.x_encoder.parameters():
            param.requires_grad = False
        for param in self.q_y_mean.parameters():
            param.requires_grad = False
        for param in self.q_y_logvar.parameters():
            param.requires_grad = False
        for param in self.post_c.parameters():
            param.requires_grad = False
        for param in self.dec_init_connector.parameters():
            param.requires_grad = False

    def freeze_generation_net(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.gaussian_mus.requires_grad = False
        self.gaussian_logvar.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def backward(self, batch_cnt, loss, step=None, vae_kl_weight=1.):
        total_loss = self.valid_loss(loss, batch_cnt, step=step, vae_kl_weight=vae_kl_weight)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None, step=None, vae_kl_weight=1.):
        if batch_cnt is not None:
            step = batch_cnt

        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        if step == self.config.pretrain_ae_step:
            self.flush_valid = True

        mi_weight = self.config.mutual_weight

        if vae_kl_weight > 0.0:
            total_loss = loss.nll + vae_kl_weight * (loss.zkl + loss.cd) \
                       - mi_weight * loss.mi
        else:
            total_loss = loss.nll

        return total_loss

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu    

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, batch_cnt=1, return_latent=False, vae_kl_weight=1.):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        if vae_kl_weight > 0.0:
            sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                        qz_logvar.repeat(posterior_sample_n, 1),
                                        sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        else:
            sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                        qz_logvar.repeat(posterior_sample_n, 1),
                                        sample=False)  # batch x (latent_size*mult_k)

        # Prepare for decoding
        dec_init_state = self.dec_init_connector(sample_z)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) * (dec_inputs.data - self.eos_id) == 0] = 1
            dec_inputs_copy = dec_inputs.clone()
            dec_inputs_copy[prob < self.config.word_dropout_rate] = self.unk_id
            dec_inputs = dec_inputs_copy

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))

            x_pos, x_neg, x_T_1, x_T, t = self.ebm.sample_latent_training_pairs(
                                            sample_z)

            # KL(q(z|x) || p(z))
            # E_q(z|x) (f(z))
            log_pos, neg_normal_ll = self.ebm.get_prior_loss(x_pos, x_T_1, t)
            loss_g_kl = - 0.5 * (1 + qz_logvar)
            kl_mask = (loss_g_kl > self.config.dim_target_kl).float()
            zkl = (kl_mask * loss_g_kl).sum(dim=1).mean()                        
            zkl += log_pos + neg_normal_ll

            # E_p(z) (f(z))            
            cd = self.ebm.get_ebm_loss(x_pos, x_neg, x_T_1, x_T, t)

            # IB
            t0 = torch.zeros(
                    sample_z.size(0), dtype=torch.long, device=sample_z.device)
            z0 = self.ebm._q_sample(
                x_start=sample_z, 
                t=t0
            ) * self.ebm._extract(self.ebm.a_s_prev, t0 + 1, sample_z.shape)
            mi = self.ebm.compute_mi(z0, t0)

            results = Pack(nll=nll, zkl=zkl, cd=cd, mi=mi)

            if return_latent:
                results['dec_init_state'] = dec_init_state
                results['z'] = sample_z

            return results

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL",
                                ):
        # Importance sampling for estimating the log-likelihood
        assert sample_type in ("LL", "logLL")

        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)

        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.view(self.num_layer_enc, 1 + int(self.bi_enc_cell), -1, self.enc_cell_size)[-1]
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # q(z|x)
        qz_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qz_logvar = self.q_y_logvar(x_last)
        sample_z = self.reparameterization(qz_mean, qz_logvar, sample=True)        

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sample_z)
        dec_outs, dec_last, outputs = self.decoder(sample_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0), -1)

        nll = torch.sum(nll, dim=-1)
        
        f_z = self.ebm._prior_lb(sample_z, qz_mean, qz_logvar)
        log_q_z = log_gaussian(sample_z.double(), mean=qz_mean.double(), log_var=qz_logvar.double())
        assert len(f_z.size()) == 1
        log_D = f_z.double() - log_q_z # log_D.size() = (batch size, )    
        
        ll = (-nll.double() + log_D).exp()
        ll = ll.view(-1, sample_num)
                
        return ll

    def sampling(self, batch_size):        
        z_e_0 = torch.randn(
                    *[batch_size, self.config.latent_size]
                ).to(self.config.DEVICE)        

        self.ebm.eval()
        zs = self.ebm.p_sample_progressive(z_e_0)
        self.ebm.train()

        dec_init_state = self.dec_init_connector(zs)
        _, _, outputs = self.decoder(zs.size(0),
                                     None, dec_init_state,
                                     mode=GEN, gen_type="greedy",
                                     beam_size=self.config.beam_size,
                                     latent_variable=zs if self.concat_decoder_input else None)
        return outputs

#------------------------------ parser -----------------------------------#
from dgmvae.utils import str2bool
import argparse
import logging
# import dgmvae.models.sent_models as sent_models
# import dgmvae.models.sup_models as sup_models
# import dgmvae.models.dialog_models as dialog_models

def add_default_training_parser(parser):
    parser.add_argument('--op', type=str, default='adam')
    parser.add_argument('--backward_size', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--prior_grad_clip', type=float, default=10.0)
    parser.add_argument('--init_w', type=float, default=0.08)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--lr_hold', type=int, default=3)
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--improve_threshold', type=float, default=0.996)
    parser.add_argument('--patient_increase', type=float, default=2.0)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('--max_epoch', type=int, default=60)
    # parser.add_argument('--max_epoch', type=int, default=2)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=1)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--use_small_batch', type=str2bool, default=True)
    parser.add_argument('--small_batch_size', type=int, default=6)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=False)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=300)
    parser.add_argument('--model_file', type=str, default='ckpts/ptb/model_ckpt.pt')
    return parser

def add_default_variational_training_parser(parser):
    # KL-annealing
    parser.add_argument('--anneal', type=str2bool, default=True)
    parser.add_argument('--anneal_function', type=str, default='logistic')
    parser.add_argument('--anneal_k', type=float, default=0.0025)
    parser.add_argument('--anneal_x0', type=int, default=2500)
    parser.add_argument('--anneal_warm_up_step', type=int, default=0)
    parser.add_argument('--anneal_warm_up_value', type=float, default=0.000)

    # Word dropout & posterior sampling number
    parser.add_argument('--word_dropout_rate', type=float, default=0.0)
    parser.add_argument('--post_sample_num', type=int, default=1)
    parser.add_argument('--sel_metric', type=str, default="elbo", help="select best checkpoint base on what metric.",
                        choices=['elbo', 'obj'],)

    # Other:
    parser.add_argument('--aggressive', type=str2bool, default=False)
    return parser

def add_default_data_parser(parser):
    # Data & logging path
    parser.add_argument('--data', type=str, default='ptb')
    parser.add_argument('--data_dir', type=str, default='data/ptb')
    parser.add_argument('--log_dir', type=str, default='logs/ptb/dgmvae_diffusion')
    # Draw points
    parser.add_argument('--fig_dir', type=str, default='figs')
    parser.add_argument('--draw_points', type=str2bool, default=False)
    return parser

# def process_config(config):
#     if config.forward_only:
#         config_ori = config

#         load_sess = config.load_sess
#         backawrd = config.backward_size
#         beam_size = config.beam_size
#         gen_type = config.gen_type
#         data_dir = config.data_dir

#         load_path = os.path.join(config.log_dir, load_sess, "params.json")
#         config = load_config(load_path)
#         config.forward_only = True
#         config.load_sess = load_sess
#         config.backward_size = backawrd
#         config.beam_size = beam_size
#         config.gen_type = gen_type
#         config.batch_size = 50
#         config.data_dir = data_dir

#     if "latent_size" in config and config.latent_size > 2:
#         config.tsne = True
#     else:
#         config.tsne = False

#     if 'anneal_function' not in config:
#         config.anneal = False

#     return config


def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # if config.forward_only:
    #     return

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


def get_parser(model_class="sent_models"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GMVAE")
    parser = add_default_data_parser(parser)
    parser = add_default_training_parser(parser)
    parser = add_default_variational_training_parser(parser)

    config, unparsed = parser.parse_known_args()

    try:
        model_name = config.model
        model_class = eval(model_name)
        parser = model_class.add_args(parser)
    except Exception as e:
        raise ValueError("Wrong model" + config.model)

    config, _ = parser.parse_known_args()
    print(config)
    # config = process_config(config)
    return config


#------------------------------ engine -----------------------------------# 
import numpy as np
from dgmvae.models.model_bases import summary
import torch
from dgmvae.dataset.corpora import PAD, EOS, EOT
from dgmvae.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from dgmvae.utils import get_dekenize, experiment_name, kl_anneal_function
import os
from collections import defaultdict
import logging
from dgmvae import utt_utils

logger = logging.getLogger()

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



class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key and 'PPL' not in self.losses:
                str_losses.append("PPL {:.3f}".format(np.exp(avg_loss)))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def return_dict(self, window=None):
        ret_losses = {}
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            ret_losses[key] = avg_loss.item()
            if 'nll' in key and 'PPL' not in self.losses:
                ret_losses[key.split("nll")[0] + 'PPL'] = np.exp(avg_loss).item()
        return ret_losses

    def avg_loss(self):
        return np.mean(self.backward_losses)

def adjust_learning_rate(optimizer, last_lr, decay_rate=0.5):
    lr = last_lr * decay_rate
    print('New learning rate=', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate  # all decay half
    return lr

def get_sent(model, de_tknize, data, b_id, attn=None, attn_ctx=None, stop_eos=True, stop_pad=True):
    ws = []
    attn_ws = []
    has_attn = attn is not None and attn_ctx is not None
    for t_id in range(data.shape[1]):
        w = model.vocab[data[b_id, t_id]]
        if has_attn:
            a_val = np.max(attn[b_id, t_id])
            if a_val > 0.1:
                a = np.argmax(attn[b_id, t_id])
                attn_w = model.vocab[attn_ctx[b_id, a]]
                attn_ws.append("{}({})".format(attn_w, a_val))
        if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
            if w == EOT:
                ws.append(w)
            break
        if w != PAD:
            ws.append(w)

    att_ws = "Attention: {}".format(" ".join(attn_ws)) if attn_ws else ""
    if has_attn:
        return de_tknize(ws), att_ws
    else:
        try:
            return de_tknize(ws), ""
        except:
            return " ".join(ws), ""

def train(model, train_feed, valid_feed, test_feed, config, evaluator, gen=None):
    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    valid_loss_record = []
    learning_rate = config.init_lr
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    epoch = 0
    training_step = 0
    train_loss = LossManager()
    model.train()
    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    num_batch = train_feed.data_size // model.config.batch_size
    vae_kl_weights = frange_cycle_zero_linear(model.config.max_epoch * num_batch,
                                              start=0.0,
                                              stop=config.max_kl_weight,
                                              n_cycle=config.n_cycle,
                                              ratio_increase=config.ratio_increase,
                                              ratio_zero=config.ratio_zero)

    prior_params = [p[1] for p in model.named_parameters() if 'ebm' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'ebm' not in p[0] and p[1].requires_grad is True]


    while epoch < model.config.max_epoch:
        train_feed.epoch_init(config, verbose=done_epoch == 0, shuffle=True)
        epoch += 1
        batch_cnt = 0
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if model.config.debug and batch_cnt > 200:
                break

            optimizer.zero_grad()

            vae_kl_weight = config.max_kl_weight 
            # vae_kl_weight = vae_kl_weights[training_step]

            loss = model(batch, mode=TEACH_FORCE, batch_cnt=batch_cnt, vae_kl_weight=vae_kl_weight)
            model.backward(batch_cnt, loss, step=batch_cnt, vae_kl_weight=vae_kl_weight)

            torch.nn.utils.clip_grad_norm_(prior_params, config.prior_grad_clip, norm_type=2.)
            torch.nn.utils.clip_grad_norm_(likelihood_params, config.grad_clip, norm_type=2.)
            optimizer.step()
            batch_cnt += 1
            training_step += 1
            train_loss.add_loss(loss)

            if batch_cnt % config.print_step == 0:
                logger.info('batch/max_batch/ep: {:5d}/ {:5d}/ {:5d} '.format(batch_cnt, train_feed.num_batch, epoch) +
                'rec: {:8.3f} '.format(loss.nll) +                 
                'zkl: {:8.3f} '.format(loss.zkl) + 
                'cd: {:8.3f} '.format(loss.cd) +
                'mi: {:10.8f} '.format(loss.mi) + 
                'kl_weight: {:8.3f} '.format(vae_kl_weight) +
                'do_ae_train: {}'.format(str(not vae_kl_weight > 0.0)) 
                )
        
        # do bleu eval
        evaluation(model, test_feed, train_feed, evaluator, epoch, vae_kl_weight)
        # save model
        torch.save(model.state_dict(), os.path.join(config.session_dir, "model_ckpt_{}.pt".format(epoch)))

def validate(model, valid_feed, config, batch_cnt=None, outres2file=None):
    model.eval()
    valid_feed.epoch_init(config, shuffle=False, verbose=True)
    losses = LossManager()
    while True:
        batch = valid_feed.next_batch()
        if batch is None:
            break
        loss = model(batch, mode=TEACH_FORCE)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()
    if outres2file:
        outres2file.write(losses.pprint(valid_feed.name))
        outres2file.write("\n")
        outres2file.write("Total valid loss {}".format(valid_loss))

    logger.info(losses.pprint(valid_feed.name))
    logger.info("Total valid loss {}".format(valid_loss))

    res_dict = losses.return_dict()

    return valid_loss, res_dict


#------------------------------ bleu -----------------------------------#
import os
import subprocess
import sys

root_dir = os.getcwd()
perl_path = os.path.join(root_dir, "multi-bleu.perl")

def multi_bleu_perl(file_name, dir):
    print("Runing multi-bleu.perl")
    # dir = os.path.join(root_dir, dir_name)
    # for file in os.listdir(dir):
    #     if "-test-greedy.txt.txt" in file or file[-16:] == "-test-greedy.txt":
    # file_name = os.path.join(dir, file)
    f = open(file_name, "r")

    hyps_f = open(os.path.join(dir, "hyp"), "w")
    refs_f = open(os.path.join(dir, "ref"), "w")

    # if not file:
    #     print("Open file error!")
    #     exit()
    for line in f:
        if line[:7] == "Target:":
            refs_f.write(line[7:].strip() + "\n")
        if line[:8] == "Predict:":
            hyps_f.write(line[8:].strip() + "\n")
            # hyps += line[8:]

    hyps_f.close()
    refs_f.close()
    f.close()
    p = subprocess.Popen(["perl", perl_path, os.path.join(dir, "ref")], stdin=open(os.path.join(dir, "hyp"), "r"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    while p.poll() == None:
        pass
    print("multi-bleu.perl return code: ", p.returncode)
    # os.remove(os.path.join(dir, "hyp"))
    os.remove(os.path.join(dir, "ref"))

    fout = open(file_name, "a")
    for line in p.stdout:
        line = line.decode("utf-8")
        if line[:4] == "BLEU":
            sys.stdout.write(line)
            fout.write(line)
            logger.info('--- bleu: {}'.format(line))


#------------------------------ main -----------------------------------# 

def get_corpus_client(config):
    if config.data.lower() == "ptb":
        corpus_client = corpora.PTBCorpus(config)
    elif config.data.lower() == "daily_dialog":
        corpus_client = corpora.DailyDialogCorpus(config)
    elif config.data.lower() == "stanford":
        corpus_client = corpora.StanfordCorpus(config)
    elif config.data.lower() == "jericho_world":
        corpus_client = corpora.JerichoWorldCorpus(config)
    elif config.data.lower() == "wikitext_103":
        corpus_client = corpora.WikiText103Corpus(config)
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")
    return corpus_client

def get_dataloader(config, corpus):
    if config.data.lower() == "ptb":
        dataloader = data_loaders.PTBDataLoader
    elif config.data.lower() == "daily_dialog":
        dataloader = data_loaders.DailyDialogSkipLoader
    elif config.data.lower() == "stanford":
        dataloader = data_loaders.SMDDataLoader
    elif config.data.lower() == "jericho_world":
        dataloader = data_loaders.JerichoWorldDataLoader
    elif config.data.lower() == "wikitext_103":
        dataloader = data_loaders.WikiText103DataLoader
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")

    train_dial, valid_dial, test_dial = corpus['train'], \
                                        corpus['valid'], \
                                        corpus['test']

    train_feed = dataloader("Train", train_dial, config)
    valid_feed = dataloader("Valid", valid_dial, config)
    test_feed = dataloader("Test", test_dial, config)

    return train_feed, valid_feed, test_feed

def get_model(corpus_client, config):
    try:
        model = eval(config.model)(corpus_client, config)
        if config.forward_only:
            model.load_state_dict(torch.load(config.model_file))
        else:
            for param in model.parameters():
                param.data.uniform_(-0.1, 0.1)
    except Exception as e:
        raise NotImplementedError("Fail to build model %s" % (config.model))
    if config.use_gpu:
        model.to(config.DEVICE)
    return model

def evaluation(model, test_feed, train_feed, evaluator, epoch, kl_weight):        
    if config.forward_only:
        test_file = os.path.join(config.session_dir, "eval-{}-test-{}.txt".format(get_time(), config.gen_type))
        sampling_file = os.path.join(config.session_dir, "eval-{}-sampling.txt".format(get_time()))
    else:
        test_file = os.path.join(config.session_dir, "epoch-{:0>3d}-test-{}.txt".format(epoch, config.gen_type))
        sampling_file = os.path.join(config.session_dir, "ep-{:0>3d}-sampling.txt".format(epoch))


    if hasattr(model, "sampling_for_likelihood") and kl_weight >= config.max_kl_weight:
        with torch.no_grad():
            ll = utt_utils.calculate_likelihood(model, test_feed, 500, config)  # must
            logger.info('log-likelihood: {:10.3f}'.format(ll))

    if kl_weight >= config.max_kl_weight:
        with open(os.path.join(sampling_file), "w") as f:
            print("Saving test to {}".format(sampling_file))
            utt_utils.exact_sampling(model, 46000, config, dest_f=f, sampling_batch_size=100)

    with open(os.path.join(test_file), "w") as f:
        with torch.no_grad():
            print("Saving test to {}".format(test_file))
            utt_utils.generate(model, test_feed, config, evaluator, num_batch=None, dest_f=f)

    multi_bleu_perl(test_file, config.session_dir)
    model.train()

def main(config):
    set_seed(config.seed)
    prepare_dirs_loggers(config, os.path.basename(__file__))

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")    
    else:
        config.DEVICE = torch.device("cpu")

    corpus_client = get_corpus_client(config)
    dial_corpus = corpus_client.get_corpus()
    evaluator = evaluators.BleuEvaluator("CornellMovie")
    train_feed, valid_feed, test_feed = get_dataloader(config, dial_corpus)
    model = get_model(corpus_client, config)

    if config.forward_only is False:
        train(model, train_feed, valid_feed, test_feed, config, evaluator, gen=utt_utils.generate)
    else:
        evaluation(model, test_feed, train_feed, evaluator, epoch=0, kl_weight=config.max_kl_weight)

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = get_parser()    
    main(config)
