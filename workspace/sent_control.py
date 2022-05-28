from __future__ import print_function

import logging
import os
import json

from dgmvae import evaluators, utt_utils
from dgmvae.dataset import corpora
from dgmvae.dataset import data_loaders
from dgmvae.models.sent_models import *
from dgmvae.utils import prepare_dirs_loggers, get_time
from dgmvae.multi_bleu import multi_bleu_perl
from dgmvae.options import get_parser

from utils import yelp_datautil

from modules.ldebm_sent import LEBM

logger = logging.getLogger()

ID_PAD = 0
ID_UNK = 1
ID_BOS = 2
ID_EOS = 3


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



#------------------------------ model -----------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgmvae.dataset.corpora import PAD, BOS, EOS, UNK
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



class Classifier(nn.Module):
    def __init__(self, vocab_size, id_to_word, config):
        super().__init__()
        self.config=config
        self.vocab = id_to_word
        # self.rev_vocab = corpus.rev_vocab
        self.vocab_size = vocab_size
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = ID_BOS
        self.eos_id = ID_EOS
        self.unk_id = ID_UNK
        self.pad_id = ID_PAD
        self.eos = id_to_word[3]
        self.pad = id_to_word[0]
        self.bos = id_to_word[2]
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
                                      padding_idx=self.pad_id)
        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)
        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size)


        self.ebm = nn.Sequential(
            nn.Linear(config.latent_size, config.ebm_hidden),
            GELU(),
            nn.Linear(config.ebm_hidden, config.ebm_hidden),
            GELU(),
            nn.Linear(config.ebm_hidden, config.num_cls)
        )

    def ebm_prior(self, z, cls_output=False, temperature=1.):
        assert len(z.size()) == 2
        if cls_output:
            return self.ebm(z)
        else:
            return temperature * (self.ebm(z.squeeze()) / temperature).logsumexp(dim=1)

    def forward(self, data_feed):
        out_utts = data_feed[2]
        batch_size = out_utts.size(0)

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
        # qz_logvar = self.q_y_logvar(x_last)

        logits = self.ebm_prior(qz_mean, cls_output=True)

        # sentiment classification
        cls_labels = data_feed[1].long().squeeze(-1)
        cls_loss = F.cross_entropy(logits, cls_labels)

        with torch.no_grad():
            cls_acc = (logits.argmax(dim=-1) == cls_labels).float().mean()
            cls_acc_pos = ((logits.argmax(dim=-1) == cls_labels).to(torch.uint8) & (cls_labels == 1).to(torch.uint8)).float().sum() / ((cls_labels == 1).sum() + 10e-10)
            cls_acc_neg = ((logits.argmax(dim=-1) == cls_labels).to(torch.uint8) & (cls_labels == 0).to(torch.uint8)).float().sum() / ((cls_labels == 0).sum() + 10e-10)

        return cls_loss, cls_acc, cls_acc_pos, cls_acc_neg

class GMVAE(BaseModel):
    def __init__(self, vocab_size, id_to_word, config):
        super(GMVAE, self).__init__(config)
        self.vocab = id_to_word
        # self.rev_vocab = corpus.rev_vocab
        self.vocab_size = vocab_size
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = ID_BOS
        self.eos_id = ID_EOS
        self.unk_id = ID_UNK
        self.pad_id = ID_PAD
        self.eos = id_to_word[3]
        self.pad = id_to_word[0]
        self.bos = id_to_word[2]
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
                                      padding_idx=self.pad_id)
        self.dec_embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                          padding_idx=self.pad_id)
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
        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size)
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, self.config)

        self.ebm = LEBM(
            feat_dim=config.latent_size,
            ebm_hdim=config.ebm_hidden,
            emb_dim=config.ebm_hidden,
            emb_hdim=config.ebm_hidden,
            num_cls=config.num_cls,
            num_blocks=config.num_blocks,
            max_T=config.max_T,
            beta_st=config.beta_st,
            beta_ed=config.beta_ed,
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
        parser.add_argument('--max_utt_len', type=int, default=55)
        parser.add_argument('--max_dec_len', type=int, default=55)
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
        parser.add_argument('--mutual_weight', type=float, default=50.0)
        parser.add_argument('--cls_weight', type=float, default=200.0)
        parser.add_argument('--num_cls', type=int, default=2)
        parser.add_argument('--max_kl_weight', type=float, default=50.0)
        parser.add_argument('--n_cycle', type=int, default=4)
        parser.add_argument('--ratio_increase', type=float, default=0.2)
        parser.add_argument('--ratio_zero', type=float, default=0.2)

        # diffusion
        parser.add_argument('--max_T', type=int, default=6)
        parser.add_argument('--beta_st', type=float, default=1e-4)
        parser.add_argument('--beta_ed', type=float, default=2e-2)
        parser.add_argument('--num_blocks', type=int, default=12)

        # new
        parser.add_argument('--word_dict_max_num', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--max_sequence_length', type=int, default=60)
        parser.add_argument('--task', type=str, default='yelp')
        parser.add_argument('--data_path', type=str, default='data/yelp/processed_files/')


        # new
        parser.add_argument('--pretrain_cls_path', type=str, default='ckpts/yelp/pretrained/cls.pt')
        parser.add_argument('--cls_max_epoch', type=int, default=2)
        parser.add_argument('--cls_eval_step', type=int, default=4000)

        return parser

    def get_optimizer(self, config):
        if config.op == 'adam':
            return torch.optim.Adam([
                                     {'params': [p[1] for p in self.named_parameters() if 'ebm' not in p[0] and p[1].requires_grad]},
                                     {'params': [p[1] for p in self.named_parameters() if 'ebm' in p[0] and p[1].requires_grad], 'lr': 0.0001},
                                     ],
                                     lr=config.init_lr)

    def sample_p_0(self, n):
        return torch.randn(*[n, self.config.latent_size]).cuda()    

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu    

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
        cls_weight = self.config.cls_weight

        if vae_kl_weight > 0.0:
            total_loss = loss.nll + vae_kl_weight * (loss.zkl + loss.cd) - mi_weight * loss.mi + cls_weight * loss.cls_loss
        else:
            total_loss = loss.nll + cls_weight * loss.cls_loss

        return total_loss    
    
    def param_var(self, tgt_probs):
        # Weighted variance of natural parameters
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)

        var_eta1 = torch.sum(tgt_probs_ * (eta1 * eta1), dim=-2) - torch.sum(tgt_probs_ * eta1, dim=-2).pow(2)
        var_eta2 = torch.sum(tgt_probs_ * (eta2 * eta2), dim=-2) - torch.sum(tgt_probs_ * eta2, dim=-2).pow(2)

        return torch.sum(var_eta1 + var_eta2) / tgt_probs.size(0)

    def st_forward(self, data_feed, mode, gen_type='greedy', sample_n=1, batch_cnt=1, return_latent=False, vae_kl_weight=1.):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        # if type(data_feed) is tuple:
        #     data_feed = data_feed[0]
        # batch_size = len(data_feed['output_lens'])
        # out_utts = self.np2var(data_feed['outputs'], LONG)
        out_utts = data_feed[2]
        batch_size = out_utts.size(0)

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
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                    qz_logvar.repeat(posterior_sample_n, 1),
                                    sample=True)  # batch x (latent_size*mult_k)
        
        return qz_mean

    def st_sampling(self, z):
        dec_init_state = self.dec_init_connector(z)
        _, _, outputs = self.decoder(z.size(0),
                                     None, dec_init_state,
                                     mode=GEN, gen_type="greedy",
                                     beam_size=self.config.beam_size,
                                     latent_variable=z if self.concat_decoder_input else None)
        return outputs


    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, batch_cnt=1, return_latent=False, vae_kl_weight=1.):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        out_utts = data_feed[2]
        batch_size = out_utts.size(0)

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
        # labels = out_utts[:, 1:].contiguous()
        # dec_inputs = out_utts[:, 0:-1]
        labels = data_feed[5]
        dec_inputs = data_feed[4]

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
            zkl_s = zkl + log_pos + neg_normal_ll

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

            # sentiment classification
            logits = self.ebm.ebm_prior(z0, t0, use_cls_output=True)
            cls_labels = data_feed[1].long().squeeze(-1)
            cls_loss = F.cross_entropy(logits, cls_labels)            

            with torch.no_grad():
                cls_acc = (logits.argmax(dim=-1) == cls_labels).float().mean()
            results = Pack(nll=nll, mi=mi, ent=zkl, lpo=log_pos, zkl=zkl_s, cd=cd, 
                           cls_loss=cls_loss, cls_acc=cls_acc)

            if return_latent:
                # results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                # results['y_ids'] = c_ids
                results['z'] = sample_z

            return results    

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
from dgmvae.utils import str2bool, process_config
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
    parser.add_argument('--prior_grad_clip', type=float, default=1)
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
    parser.add_argument('--max_epoch', type=int, default=24)
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=1)
    parser.add_argument('--seed', default=3435, type=int)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=3500)
    # parser.add_argument('--eval_step', type=int, default=3)
    parser.add_argument('--num_batch', type=int, default=3500)
    parser.add_argument('--fix_batch', type=str2bool, default=False)
    parser.add_argument('--ckpt_step', type=int, default=2000)
    # parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--preview_batch_num', type=int, default=1)
    parser.add_argument('--gen_type', type=str, default='greedy')
    parser.add_argument('--avg_type', type=str, default='seq')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--forward_only', type=str2bool, default=False)
    parser.add_argument('--load_sess', type=str, default="", help="Load model directory.")
    parser.add_argument('--debug', type=bool, default=False)
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
    parser.add_argument('--data', type=str, default='yelp')
    parser.add_argument('--data_dir', type=str, default='data/yelp')
    parser.add_argument('--log_dir', type=str, default='logs/yelp/dgmvae')
    # Draw points
    parser.add_argument('--fig_dir', type=str, default='figs')
    parser.add_argument('--draw_points', type=str2bool, default=False)
    return parser



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
    config = process_config(config)
    return config


#------------------------------ engine -----------------------------------# 
import numpy as np
from dgmvae.models.model_bases import summary
import torch
# from dgmvae.dataset.corpora import PAD, EOS, EOT
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
    ids = []
    attn_ws = []
    has_attn = attn is not None and attn_ctx is not None
    for t_id in range(data.shape[1]):
        w = model.vocab[data[b_id, t_id]]
        _id = data[b_id, t_id]
        if has_attn:
            a_val = np.max(attn[b_id, t_id])
            if a_val > 0.1:
                a = np.argmax(attn[b_id, t_id])
                attn_w = model.vocab[attn_ctx[b_id, a]]
                attn_ws.append("{}({})".format(attn_w, a_val))
        if (stop_eos and w in model.eos) or (stop_pad and w == model.pad):
            # if w == EOT:
            #     ws.append(w)
            break
        if w != model.pad:
            ws.append(w)
            ids.append(_id)

    att_ws = "Attention: {}".format(" ".join(attn_ws)) if attn_ws else ""
    if has_attn:
        return de_tknize(ws), att_ws
    else:
        try:
            return de_tknize(ws), "", ids
        except:
            return " ".join(ws), "", ids

def train(model, train_feed, valid_feed, test_feed, config, evaluator, classifier, gen=None):
    if gen is None:
        gen = generate

    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    valid_loss_record = []
    learning_rate = config.init_lr
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    epoch = 0
    train_loss = LossManager()
    model.train()
    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    # num_batch = train_feed.num_batch
    num_batch = config.num_batch
    vae_kl_weights = frange_cycle_zero_linear(model.config.max_epoch * num_batch,
                                              start=0.0,
                                              stop=config.max_kl_weight,
                                              n_cycle=config.n_cycle,
                                              ratio_increase=config.ratio_increase,
                                              ratio_zero=config.ratio_zero)



    prior_params = [p[1] for p in model.named_parameters() if 'ebm' in p[0] and p[1].requires_grad is True]
    likelihood_params = [p[1] for p in model.named_parameters() if 'ebm' not in p[0] and p[1].requires_grad is True]


    batch_cnt = 0
    while batch_cnt < (config.max_epoch * config.eval_step):
        batch = train_feed.next_batch()
        if batch is None:
            break
        if model.config.debug and batch_cnt > 200:
            break

        optimizer.zero_grad()

        # vae_kl_weight = vae_kl_weights[batch_cnt]
        vae_kl_weight = config.max_kl_weight
        loss = model(batch, mode=TEACH_FORCE, batch_cnt=batch_cnt, vae_kl_weight=vae_kl_weight)

        model.backward(batch_cnt, loss, step=batch_cnt, vae_kl_weight=vae_kl_weight)
        torch.nn.utils.clip_grad_norm_(prior_params, config.prior_grad_clip, norm_type=2.)
        torch.nn.utils.clip_grad_norm_(likelihood_params, config.grad_clip, norm_type=2.)
        optimizer.step()
        batch_cnt += 1
        train_loss.add_loss(loss)

        if batch_cnt % config.print_step == 0:
            logger.info('batch/max_batch/ep: {:5d}/ {:5d}/ {:5d} '.format(batch_cnt, train_feed.num_batch, epoch) +
            'rec: {:8.3f} '.format(loss.nll) + 
            'mi: {:10.8f} '.format(loss.mi) + 
            'zkl: {:8.3f} '.format(loss.zkl) + 
            'cls_loss: {:8.3f} '.format(loss.cls_loss) + 
            'cls_acc: {:8.3f} '.format(loss.cls_acc) + 
            'cd: {:8.3f} '.format(loss.cd) + 
            'ent: {:8.3f} '.format(loss.ent) +
            'lpo: {:8.3f} '.format(loss.lpo) +
            'kl_weight: {:8.3f}'.format(vae_kl_weight)
            )
    
        # do bleu eval
        if batch_cnt % config.eval_step == 0:
        # if batch_cnt % 200 == 0:
            evaluation(model, test_feed, train_feed, evaluator, batch_cnt, vae_kl_weight, classifier)

            torch.save(model.state_dict(), os.path.join(config.session_dir, "model_ckpt_{}.pt".format(batch_cnt)))        



def train_cls(model, train_feed, test_feed, config, optimizer):
    batch_cnt = 0
    while batch_cnt < (config.cls_max_epoch * config.cls_eval_step):
        model.train()
        batch = train_feed.next_batch()
        if model.config.debug and batch_cnt > 200:
            break

        optimizer.zero_grad()
        loss, acc = model(batch)
        loss.backward()
        optimizer.step()
        batch_cnt += 1

        if batch_cnt % config.print_step == 0:

            logger.info('batch/max_batch/ep: {:5d}/ {:5d}/ {:5d} '.format(batch_cnt, train_feed.num_batch, 0) +
            'cls loss: {:8.3f} '.format(loss) + 
            'acc loss: {:8.3f} '.format(acc)
            )
        if batch_cnt % config.cls_eval_step == 0:
            eval_cls(model, test_feed)

def eval_cls_batch(model, test_feed):
    model.eval()
    loss, acc, acc_pos, acc_neg = model(test_feed)
    return loss, acc, acc_pos, acc_neg

def eval_cls(model, test_feed):
    losses = []
    accs = [] 
    accs_pos = []
    accs_neg = []
    batch_sizes = []
    while True:
        batch = test_feed.next_batch()
        batch_size = batch[1].shape[0]
        loss, acc, acc_pos, acc_neg = eval_cls_batch(model, batch)
        losses.append(loss * batch_size)
        accs.append(acc * batch_size)
        accs_pos.append(acc_pos * batch_size)
        accs_neg.append(acc_neg * batch_size)
        batch_sizes.append(batch_size)
        if test_feed.pointer == 0:
            break
    
    loss = torch.tensor(losses).sum() / sum(batch_sizes)
    acc = torch.tensor(accs).sum() / sum(batch_sizes)
    acc_pos = torch.tensor(accs_pos).sum() / sum(batch_sizes)
    acc_neg = torch.tensor(accs_neg).sum() / sum(batch_sizes)


    logger.info('----->| clc loss:{:8.4f} | acc:{:8.4f} | acc_pos:{:8.4f} | acc_neg:{:8.4f}'.format(loss, acc, acc_pos, acc_neg))


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
    if config.data.lower() == "news":
        corpus_client = corpora.NewsCorpus(config)
    elif config.data.lower() == "ptb":
        corpus_client = corpora.PTBCorpus(config)
    elif config.data.lower() == "daily_dialog":
        corpus_client = corpora.DailyDialogCorpus(config)
    elif config.data.lower() == "stanford":
        corpus_client = corpora.StanfordCorpus(config)
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")
    return corpus_client

def get_dataloader(config, corpus):
    if config.data.lower() == "news":
        dataloader = data_loaders.NewsDataLoader
    elif config.data.lower() == "ptb":
        dataloader = data_loaders.PTBDataLoader
    elif config.data.lower() == "daily_dialog":
        dataloader = data_loaders.DailyDialogSkipLoader
    elif config.data.lower() == "stanford":
        dataloader = data_loaders.SMDDataLoader
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")

    train_dial, valid_dial, test_dial = corpus['train'], \
                                        corpus['valid'], \
                                        corpus['test']

    train_feed = dataloader("Train", train_dial, config)
    valid_feed = dataloader("Valid", valid_dial, config)
    test_feed = dataloader("Test", test_dial, config)

    return train_feed, valid_feed, test_feed

def get_model(vocab_size, id_to_word, config):
    try:
        model = eval(config.model)(vocab_size, id_to_word, config)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

    except Exception as e:
        raise NotImplementedError("Fail to build model %s" % (config.model))
    if config.use_gpu:
        model.cuda()
    return model


def get_dekenize():
    return lambda x: " ".join(x)


def set_seed(seed):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size

    if num_batch != None:
        config.batch_size = 3

    de_tknize = get_dekenize()
    # data_feed.epoch_init(config, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

    evaluator.initialize()
    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    batch_cnt = 0
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        batch_cnt += 1
        
        if batch_cnt > 100:
            break

        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)  # todo: config.gen_type

        if DecoderRNN.KEY_LATENT in outputs:
            key_latent = outputs[DecoderRNN.KEY_LATENT]
            key_latent = key_latent.cpu().data.numpy()
        else:
            key_latent = None

        if DecoderRNN.KEY_CLASS in outputs:
            key_class = outputs[DecoderRNN.KEY_CLASS].cpu().data.numpy()
        else:
            key_class = None

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        # get attention if possible
        pred_attns = None

        for b_id in range(pred_labels.shape[0]):
            pred_str, attn, _ = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            true_str, _, _ = get_sent(model, de_tknize, true_labels, b_id)
            evaluator.add_example(true_str, pred_str)
            if dest_f is None:
                logger.info("Target: {}".format(true_str))
                logger.info("Predict: {}".format(pred_str))
                if key_latent is not None and key_class is not None:
                    key_latent_str = "-".join(map(str, key_latent[b_id]))
                    logger.info("Key Latent: {}\n".format(str(key_class[b_id]) + "\t" + key_latent_str))
                logger.info("\n")
            else:
                dest_f.write("Target: {}\n".format(true_str))
                dest_f.write("Predict: {}\n\n".format(pred_str))

    if dest_f is None:
        logging.info(evaluator.get_report(include_error=dest_f is not None))
    else:
        dest_f.write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")
    return evaluator.get_report(include_error=dest_f is not None, get_value=True)



def eval_sentiment_control(model, data_feed, config, evaluator, classifier, ae_train, num_batch=1, dest_f=None):
    model.train()

    de_tknize = get_dekenize()

    batch_cnt = 0
    true_accs = []
    true_accs_pos = []
    true_accs_neg = []
    transferred_accs = []
    transferred_accs_pos = []
    transferred_accs_neg = []
    batch_sizes = []
    while True:
        batch = data_feed.next_batch()        
        if batch_cnt > 9:
            break

        batch_cnt += 1
        logger.info('---> eval batch {}'.format(batch_cnt))

        cls_labels = batch[1]
        batch_size = cls_labels.shape[0]
        z_e_0 = model.sample_p_0(n=batch_size)
        z_st = model.ebm.p_sample_progressive(z_e_0, ref_lb=cls_labels.long().squeeze(-1))[0]
        labels = batch[5]
        pred_labels = model.st_sampling(z_st)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in pred_labels[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        pred_attns = None

        pred_ids_list = []
        true_ids_list = []
        
        for b_id in range(batch_size):
            pred_str, _, pred_ids = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            _, _, true_ids = get_sent(model, de_tknize, true_labels, b_id)
            dest_f.write("Label: {}\n".format(cls_labels.cpu().squeeze()[b_id].item()))
            dest_f.write("Predict: {}\n\n".format(pred_str))
            pred_ids_list.append(pred_ids)
            true_ids_list.append(true_ids)
        
        with torch.no_grad():
            padding = lambda x, max_len: [el + (max_len - len(el)) * [ID_PAD] for el in x]
            list_max_len = lambda x: max([len(el) for el in x])

            pred_ids = torch.tensor(padding(pred_ids_list, list_max_len(pred_ids_list)))
            true_ids = torch.tensor(padding(true_ids_list, list_max_len(true_ids_list)))

            ids_2_cls_input = lambda x, y: (None, y, x.long().cuda(), None, 
                                        None, None, None, None)

            cls_eval_labels = cls_labels
            cls_pred_batch = ids_2_cls_input(pred_ids, cls_eval_labels)
            _, acc_pred, acc_pred_pos, acc_pred_neg = eval_cls_batch(classifier, cls_pred_batch)

            cls_true_batch = ids_2_cls_input(true_ids, cls_eval_labels)
            _, acc_true, acc_true_pos, acc_true_neg = eval_cls_batch(classifier, cls_true_batch)
            

            true_accs.append(acc_true.cpu() * batch_size)
            true_accs_pos.append(acc_true_pos.cpu() * batch_size)
            true_accs_neg.append(acc_true_neg.cpu() * batch_size)

            transferred_accs.append(acc_pred.cpu() * batch_size)
            transferred_accs_pos.append(acc_pred_pos.cpu() * batch_size)
            transferred_accs_neg.append(acc_pred_neg.cpu() * batch_size)

            batch_sizes.append(batch_size)

    list_mean = lambda x: torch.tensor(x).sum() / sum(batch_sizes)

    logger.info('sentiment control eval: | true acc: {:8.4f} | true pos acc: {:8.4f} | true neg acc: {:8.4f} | tran acc: {:8.4f} |  tran pos acc: {:8.4f} | tran neg acc: {:8.4f} |'.format(
        list_mean(true_accs), list_mean(true_accs_pos), list_mean(true_accs_neg), 
        list_mean(transferred_accs), list_mean(transferred_accs_pos), list_mean(transferred_accs_neg)
    ))


def evaluation(model, test_feed, train_feed, evaluator, epoch, kl_weight, classifier):
        
    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
        sampling_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-sampling.txt".format(get_time()))
    else:
        test_file = os.path.join(config.session_dir,
                                 "epoch-{:0>3d}-test-{}.txt".format(epoch, config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")
        sampling_file = os.path.join(config.session_dir, "ep-{:0>6d}-sampling.txt".format(epoch))
        transfer_file = os.path.join(config.session_dir, "ep-{:0>6d}-transfer.txt".format(epoch))


    ae_train = kl_weight > 0.2
    with open(os.path.join(transfer_file), "w") as f:
        print("Saving test to {}".format(transfer_file))
        eval_sentiment_control(model, test_feed, config, evaluator, classifier, ae_train, num_batch=None, dest_f=f)


    with open(os.path.join(test_file), "w") as f:
        print("Saving test to {}".format(test_file))
        generate(model, train_feed, config, evaluator, num_batch=None, dest_f=f)

    model.train()

def main(config):
    set_seed(config.seed)

    prepare_dirs_loggers(config, os.path.basename(__file__))

    evaluator = evaluators.BleuEvaluator("CornellMovie")


    # train data
    id_to_word, vocab_size, train_file_list, train_label_list = yelp_datautil.prepare_data(
        data_path=config.data_path, max_num=config.word_dict_max_num, task_type=config.task
    )
    train_feed = yelp_datautil.non_pair_data_loader(
        batch_size=config.batch_size, id_bos=ID_BOS, id_eos=ID_EOS, id_unk=ID_UNK,
        max_sequence_length=config.max_sequence_length, vocab_size=vocab_size
    )
    train_feed.create_batches(train_file_list, train_label_list, if_shuffle=True)    


    # eval data
    test_feed = yelp_datautil.non_pair_data_loader(
        batch_size=200, id_bos=ID_BOS,
        id_eos=ID_EOS, id_unk=ID_UNK,
        max_sequence_length=config.max_sequence_length, vocab_size=vocab_size
    )
    eval_file_list = [
        config.data_path + 'sentiment.test.0',
        config.data_path + 'sentiment.test.1',
    ]
    eval_label_list = [
        [0],
        [1],
    ]
    test_feed.create_batches(eval_file_list, eval_label_list, if_shuffle=False)


    classifer = Classifier(vocab_size, id_to_word, config)
    for param in classifer.parameters():
        param.data.uniform_(-0.1, 0.1)
    classifer.cuda()


    if not config.pretrain_cls_path:
        classifier_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifer.parameters()), lr=config.init_lr)
        train_cls(classifer, train_feed, test_feed, config, classifier_optimizer)
        save_path = os.path.join(config.session_dir, config.load_sess, 'cls.pt')
        torch.save({
            'model_state_dict': classifer.state_dict(),
            }, save_path)
        train_feed.create_batches(train_file_list, train_label_list, if_shuffle=True)    
        test_feed.create_batches(eval_file_list, eval_label_list, if_shuffle=False)
        logger.info('saved pretrained classifier')
    else:
        ckpt = torch.load(config.pretrain_cls_path, map_location=torch.device('cuda:{}'.format(config.gpu_idx)))
        classifer.load_state_dict(ckpt['model_state_dict'])
        eval_cls(classifer, test_feed)
        test_feed.create_batches(eval_file_list, eval_label_list, if_shuffle=False)
        logger.info('loaded pretrained classifier')
    


    # eval data batch size 1
    test_feed = yelp_datautil.non_pair_data_loader(
        batch_size=config.batch_size, id_bos=ID_BOS,
        id_eos=ID_EOS, id_unk=ID_UNK,
        max_sequence_length=config.max_sequence_length, vocab_size=vocab_size
    )
    eval_file_list = [
        config.data_path + 'sentiment.test.0',
        config.data_path + 'sentiment.test.1',
    ]
    eval_label_list = [
        [0],
        [1],
    ]
    test_feed.create_batches(eval_file_list, eval_label_list, if_shuffle=True)


    model = get_model(vocab_size, id_to_word, config)

    if config.forward_only is False:
        train(model, train_feed, train_feed, test_feed, config, evaluator, classifer, gen=utt_utils.generate)

if __name__ == "__main__":
    config = get_parser()
    with torch.cuda.device(config.gpu_idx):
        main(config)
