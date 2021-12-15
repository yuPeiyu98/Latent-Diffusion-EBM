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

class SVAE(BaseModel):
    def __init__(self, corpus, config):
        super(SVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
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

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)

        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_c = nn.Linear(self.enc_out_size, config.k * config.mult_k)

        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size + config.k * config.mult_k,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)

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
                                  tie_output_embed=config.tie_output_embed if "tie_output_embed" in config else False,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()

        self.log_py = nn.Parameter(torch.log(torch.ones(self.config.latent_size,
                                                        self.config.k) / config.k),
                                   requires_grad=True)

        self.register_parameter('log_py', self.log_py)

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids')

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

        # Other settings
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=False)

        return parser

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def model_sel_loss(self, loss, batch_cnt):
        return loss.elbo

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt
        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        mi_weight = 0.0 if self.config.use_mutual else 1.0
        total_loss = loss.nll + vae_kl_weight * (
                loss.agg_ckl + mi_weight * loss.mi + loss.zkl)

        return total_loss

    def zkl_loss(self, qy_mean, qy_logvar):
        KL_loss = -0.5 * torch.mean(torch.sum((1 + qy_logvar - qy_mean.pow(2) - qy_logvar.exp()), dim=1))
        return KL_loss

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1
        if isinstance(data_feed, tuple):
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

        # posterior network
        qy_mean = self.q_y_mean(x_last)  # batch x (latent_size*mult_k)
        qy_logvar = self.q_y_logvar(x_last)
        q_z = self.reparameterization(qy_mean.repeat(posterior_sample_n, 1),
                                      qy_logvar.repeat(posterior_sample_n, 1),
                                      sample=gen_type != "greedy" or mode != GEN)  # batch x latent_size
        qc_logits = self.q_c(x_last).view(-1, self.config.k)  # batch*mult_k x k
        log_qc = F.log_softmax(qc_logits, qc_logits.dim() - 1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qc_logits.repeat(posterior_sample_n, 1),
                                             1.0, self.use_gpu,
                                             hard=not self.training, return_max_id=True)
        # sample_y: [batch*mult_k, k], y_ids: [batch*mult_k, 1]
        sample_y = sample_y.view(-1, self.config.mult_k * self.config.k)
        y_ids = y_ids.view(-1, self.config.mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(torch.cat((sample_y, q_z), dim=1))

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=q_z)

        dec_ctx[DecoderRNN.KEY_LATENT] = y_ids
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            avg_log_qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))
            avg_log_qc = torch.log(torch.mean(avg_log_qc, dim=0) + 1e-15)
            agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.sum(agg_ckl)
            ckl_real = self.cat_kl_loss(log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.mult_k), dim=0))
            zkl = self.zkl_loss(qy_mean, qy_logvar)  # [batch_size x mult_k]
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / batch_size

            results = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, real_ckl=ckl_real, elbo=nll+ckl_real+zkl, zkl=zkl, PPL=ppl)
            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids
            return results

    def sampling(self, batch_size):
        sample_y = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        zs = self.torch2var(torch.randn(batch_size, self.config.latent_size))
        cs = self.torch2var(idx2onehot(sample_y.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)
        dec_init_state = self.dec_init_connector(torch.cat((cs, zs), dim=1))
        _, _, outputs = self.decoder(cs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type=self.config.gen_type,
                                      beam_size=self.config.beam_size,
                                      latent_variable=zs
                                      )
        return outputs

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
        assert sample_type in ("LL", "logLL")

        # just for calculating log-likelihood
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

        qy_mean = self.q_y_mean(x_last)  # [batch_size * sample_num, latent_size]
        qy_logvar = self.q_y_logvar(x_last)
        q_z = self.reparameterization(qy_mean, qy_logvar, sample=True)
        # [batch_size * sample_num, latent_size]
        log_qzx = torch.sum(
            - (q_z - qy_mean) * (q_z - qy_mean) / (2 * torch.exp(qy_logvar)) - 0.5 * qy_logvar - 0.5 * math.log(
                math.pi * 2),
            dim=-1)
        log_pz = torch.sum(
            - (q_z) * (q_z) / 2 - 0.5 * math.log(math.pi * 2),
            dim=-1)

        qc_logits = self.q_c(x_last).view(-1, self.config.k)  # batch*mult_k x k
        log_qcx = F.log_softmax(qc_logits, qc_logits.dim() - 1)

        sample_c = torch.multinomial(torch.exp(log_qcx), 1) # .view(-1, self.config.mult_k)  # [batch_size, mult_k]
        log_qcx = torch.sum(torch.gather(log_qcx, 1, sample_c).view(-1, self.config.mult_k), dim=-1)
        sample_c = self.torch2var(idx2onehot(sample_c.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)

        log_pc = math.log(1.0 / self.config.k) * self.config.mult_k


        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(torch.cat((sample_c, q_z), dim=1))

        dec_outs, dec_last, outputs = self.decoder(sample_c.size(0),
                                                   dec_inputs,
                                                   dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=q_z)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)
        ll = torch.exp(-nll.double() + log_pz.double() + log_pc - log_qzx.double() - log_qcx.double())
        if sample_type == "logLL":
            return (-nll.double() + log_pz.double() + log_pc - log_qzx.double() - log_qcx.double()).view(-1, sample_num)
        else:
            ll = ll.view(-1, sample_num)
        return ll

class DiVAE(BaseModel):
    def __init__(self, corpus, config):
        super(DiVAE, self).__init__(config)
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
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.use_kl = getattr(config, "use_kl", True)

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)

        self.q_y = nn.Linear(self.enc_out_size, config.mult_k * config.k)
        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.mult_k * config.k,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size,
                                  self.max_dec_len,
                                  self.embed_size + self.config.mult_k * self.config.k if self.concat_decoder_input else self.embed_size,
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
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        self.log_py = nn.Parameter(torch.log(torch.ones(self.config.mult_k,
                                                        self.config.k)/config.k),
                                   requires_grad=True)
        self.register_parameter('log_py', self.log_py)

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

        self.return_latent_key = ("dec_init_state", "log_qy", "y_ids")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--k', type=int, default=5, help="Latent size of discrete latent variable")
        parser.add_argument('--mult_k', type=int, default=20, help="The number of discrete latent variables.")

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

        # Other settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--use_kl', type=str2bool, default=True)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=False)

        return parser

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0
        if self.config.use_mutual or self.config.anneal is not True:
            vae_kl_weight = 1.0

        total_loss = loss.nll

        if not self.use_kl:
            return total_loss

        if self.config.use_mutual:
            total_loss += (vae_kl_weight * loss.agg_ckl)
        else:
            total_loss += (vae_kl_weight * loss.ckl_real)

        return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        if not self.use_kl:  # DAE
            return loss.nll
        else:
            if "sel_metric" in self.config and self.config.sel_metric == "elbo":
                return loss.elbo
            return self.valid_loss(loss)
            # return loss.elbo

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        posterior_sample_n = self.posterior_sample_n if self.training else 1

        if isinstance(data_feed, tuple):
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


        # posterior network
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, qy_logits.dim()-1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qy_logits.repeat(posterior_sample_n, 1),
                                             1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        sample_y = sample_y.view(-1, self.config.k * self.config.mult_k)
        y_ids = y_ids.view(-1, self.config.mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=sample_y)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            if self.config.avg_type == "seq":
                ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))

            # regularization
            log_qy = log_qy.view(-1, self.config.mult_k, self.config.k)
            avg_log_qc = torch.log(torch.mean(torch.exp(log_qy), dim=0) + 1e-15)
            agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.sum(agg_ckl)

            ckl_real = self.cat_kl_loss(log_qy, self.log_uniform_y, batch_size, unit_average=True, average=False)
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.k), dim=0))
            # H(C) - H(C|X)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qy) * log_qy) / batch_size

            results = Pack(nll=nll, mi=mi, ckl_real=ckl_real,
                           elbo=nll+ckl_real, agg_ckl=agg_ckl)
            if self.config.avg_type == "seq":
                results['PPL'] = ppl

            if return_latent:
                results['log_qy'] = log_qy
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids

            return results

    def sampling(self, batch_size):
        sample_y = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        cs = self.torch2var(idx2onehot(sample_y.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)
        dec_init_state = self.dec_init_connector(cs)

        _, _, outputs = self.decoder(cs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type=self.config.gen_type,
                                      beam_size=self.config.beam_size,
                                     latent_variable=cs)
        return outputs

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
        # just for calculating log-likelihood
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

        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, -1)
        sampling_c = torch.multinomial(torch.exp(log_qy), 1) # .view(-1, self.config.mult_k)  # [batch_size * mult_k, 1]
        log_qcx = torch.sum(torch.gather(log_qy, 1, sampling_c).view(-1, self.config.mult_k), dim=-1)
        sampling_c = self.torch2var(idx2onehot(sampling_c.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)

        # print(log_qcx.size())
        log_pc = math.log(1.0 / self.config.k) * self.config.mult_k

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sampling_c)
        dec_outs, dec_last, outputs = self.decoder(sampling_c.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sampling_c if self.concat_decoder_input else None)

        # nll = self.nll_loss(dec_outs, labels)
        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)

        ll = torch.exp(-nll.double() + log_pc - log_qcx.double())  # log (p(z)p(x|z) / q(z|x))

        ll = ll.view(-1, sample_num)
        # nll_per = torch.log(torch.mean(ll, dim=-1))  #
        # batch_size = nll_per.size(0)
        # nll_per = torch.sum(nll_per)
        return ll

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
                                  self.embed_size + self.config.mult_k * self.config.latent_size if self.concat_decoder_input else self.embed_size,
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
        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        self.post_c = nn.Sequential(
            nn.Linear(self.enc_out_size, self.enc_out_size),
            nn.ReLU(),
            nn.Linear(self.enc_out_size, self.config.mult_k * self.config.k),
        )
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size * config.mult_k,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)
        self.cat_connector = nn_lib.GumbelConnector()

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)

        self.init_gaussian()

        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids', 'z')
        self.kl_w = 0.0

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=2, help="The latent size of continuous latent variable.")
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
        return parser

    def init_gaussian(self):
        self._log_uniform_y = Variable(torch.log(torch.ones(1) / self.config.k))
        if self.use_gpu:
            self._log_uniform_y = self.log_uniform_y.cuda()

        mus = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        logvar = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        if torch.cuda.is_available():
            mus = mus.cuda()
            logvar = logvar.cuda()
        self._gaussian_mus = torch.nn.Parameter(mus, requires_grad=True)  # change: False
        self._gaussian_logvar = torch.nn.Parameter(logvar, requires_grad=True)  # change: False

    @property
    def gaussian_mus(self):
        return self._gaussian_mus

    @property
    def gaussian_logvar(self):
        return self._gaussian_logvar

    @property
    def log_uniform_y(self):
        return self._log_uniform_y

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

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        if step == self.config.pretrain_ae_step:
            self.flush_valid = True

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step - self.config.pretrain_ae_step,
                                               self.config.anneal_k, self.config.anneal_x0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_step" in self.config else 0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_value" in self.config else 0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        mi_weight = 0.0 if self.config.use_mutual else 1.0

        total_loss = loss.nll + vae_kl_weight * (self.config.klw_for_ckl * (loss.agg_ckl + mi_weight * loss.mi) +
                                                 self.config.klw_for_zkl * (loss.zkl + self.config.beta * loss.dispersion)
                                                )
        return total_loss

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def zkl_loss(self, tgt_probs, mean, log_var, mean_prior=True):
        mean = mean.view(-1, self.config.mult_k, self.config.latent_size)
        log_var = log_var.view(-1, self.config.mult_k, self.config.latent_size)
        if mean_prior:
            tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
            eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
            eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
            Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
            Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
            Emu = -0.5 * Eeta1 / Eeta2
            Evar = -0.5 * torch.pow(Eeta2, -1)
            # [batch_size, mult_k, latent_size]
            kl = 0.5 * (
                    torch.sum(log_var.exp().div(Evar), dim=-1)
                    + torch.sum((Emu - mean).pow(2) / Evar, dim=-1)
                    - mean.size(-1)
                    + torch.sum(Evar.log() - log_var, dim=-1)
            )
            # [batch_size, mult_k]
            return kl

        mu_repeat = mean.unsqueeze(-2).expand(-1, -1, self.config.k, -1)  # batch_size x k x z_dim
        logvar_repeat = log_var.unsqueeze(-2).expand(-1, -1, self.config.k, -1)
        gaussian_logvars = self.gaussian_logvar

        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussian_logvars.exp()), dim=-1)
                + torch.sum((self.gaussian_mus - mu_repeat).pow(2) / gaussian_logvars.exp(), dim=-1)
                - mean.size(-1)
                + torch.sum((gaussian_logvars - logvar_repeat), dim=-1)
        )  # batch_size x mult_k x k

        return torch.sum(kl * tgt_probs, dim=-1)  # batch_size*mult_k

    def dispersion(self, tgt_probs):
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
        Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2) # [batch_size, mult_k, latent_size]
        Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
        AE = -0.25 * Eeta1 * Eeta1 / Eeta2 - 0.5 * torch.log(-2 * Eeta2) # [batch_size, mult_k, latent_size]
        AE = torch.mean(torch.sum(AE, dim=(-1, -2)))

        EA = torch.sum(-0.25 * eta1 * eta1 / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1) # [mult_k, k]
        EA = torch.mean(torch.sum(tgt_probs * EA, dim=(-1,-2)))
        return EA-AE
    
    def param_var(self, tgt_probs):
        # Weighted variance of natural parameters
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)

        var_eta1 = torch.sum(tgt_probs_ * (eta1 * eta1), dim=-2) - torch.sum(tgt_probs_ * eta1, dim=-2).pow(2)
        var_eta2 = torch.sum(tgt_probs_ * (eta2 * eta2), dim=-2) - torch.sum(tgt_probs_ * eta2, dim=-2).pow(2)

        return torch.sum(var_eta1 + var_eta2) / tgt_probs.size(0)

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
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
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                      qz_logvar.repeat(posterior_sample_n, 1),
                                      sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        # q(c|x)
        qc_logits = self.post_c(x_last).view(-1, self.config.k)
        log_qc = F.log_softmax(qc_logits, -1)  # [batch*mult_k, k]
        sample_c, c_ids = self.cat_connector(qc_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # sample_c: [batch*mult_k, k], c_ids: [batch*mult_k, 1]
        # sample_c = sample_c.view(-1, self.config.mult_k * self.config.k)
        c_ids = c_ids.view(-1, self.config.mult_k)

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
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))
            # ZKL & dispersion term
            zkl = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=True)  # [batch_size x mult_k]
            zkl_real = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=False)  # [batch_size x mult_k]
            zkl = torch.sum(torch.mean(zkl, dim=0))
            zkl_real = torch.sum(torch.mean(zkl_real, dim=0))
            dispersion = self.dispersion(qc)
            # CKL & MI term
            avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)
            agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.sum(agg_ckl)
            ckl_real = self.cat_kl_loss(log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.mult_k), dim=0))
            # H(C) - H(C|X)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / batch_size

            results = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, zkl=zkl, dispersion=dispersion, PPL=ppl,
                           real_zkl=zkl_real, real_ckl=ckl_real, elbo=nll + ckl_real + zkl_real,
                           param_var=self.param_var(tgt_probs=qc))

            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = c_ids
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

        log_qzx = torch.sum(
            - (sample_z - qz_mean) * (sample_z - qz_mean) / (2 * torch.exp(qz_logvar)) - 0.5 * qz_logvar - 0.5 * math.log(
                math.pi * 2),
            dim=-1)
        sample_z_repeat = sample_z.view(-1, self.config.mult_k, 1, self.config.latent_size).repeat(1, 1, self.config.k, 1)
        log_pzc = torch.sum(
            - (sample_z_repeat - self.gaussian_mus) * (sample_z_repeat - self.gaussian_mus) / (2 * torch.exp(self.gaussian_logvar))
            - 0.5 * self.gaussian_logvar - 0.5 * math.log(math.pi * 2),
            dim=-1)  # [batch_size, mult_k, k]
        log_pz = torch.log(torch.mean(torch.exp(log_pzc.double()), dim=-1))  #
        log_pz = torch.sum(log_pz, dim=-1)

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sample_z)
        dec_outs, dec_last, outputs = self.decoder(sample_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)
        if sample_type == "logLL":
            return (-nll.double() + log_pz - log_qzx.double()).view(-1, sample_num)
        else:
            ll = torch.exp(-nll.double() + log_pz - log_qzx.double())  # exp ( log (p(z)p(x|z) / q(z|x)) )
            ll = ll.view(-1, sample_num)
        return ll

    def sampling(self, batch_size):
        sample_c = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        index = (self.torch2var(torch.arange(self.config.mult_k) * self.config.k) + sample_c).view(-1)
        mean = self.gaussian_mus.view(-1, self.config.latent_size)[index].squeeze()
        sigma = torch.exp(self.gaussian_logvar * 0.5).view(-1, self.config.latent_size)[index].squeeze()
        zs = self.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        zs = zs.view(-1, self.config.mult_k * self.config.latent_size)
        dec_init_state = self.dec_init_connector(zs)
        _, _, outputs = self.decoder(zs.size(0),
                                     None, dec_init_state,
                                     mode=GEN, gen_type="greedy",
                                     beam_size=self.config.beam_size,
                                     latent_variable=zs if self.concat_decoder_input else None)
        return outputs

class GMVAE_fb(GMVAE):
    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=2, help="The latent size of continuous latent variable.")
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
        parser.add_argument('--pretrain_ae_step', type=int, default=0)

        # Free bits setting:
        parser.add_argument('--max_fb_c', type=float, default=5.0)
        parser.add_argument('--max_fb_z', type=float, default=10.0)

        return parser

    def model_sel_loss(self, loss, batch_cnt):  # return albo
        if batch_cnt is not None and batch_cnt < self.config.pretrain_ae_step:
            return loss.nll
        if "sel_metric" in self.config and self.config.sel_metric == "elbo":
            return loss.elbo
        return self.valid_loss(loss)

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step < self.config.pretrain_ae_step:
            return loss.nll  # AE
        if step == self.config.pretrain_ae_step:
            self.flush_valid = True

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step - self.config.pretrain_ae_step,
                                               self.config.anneal_k, self.config.anneal_x0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_step" in self.config else 0,
                                               self.config.anneal_warm_up_value if "anneal_warm_up_value" in self.config else 0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        total_loss = loss.nll + vae_kl_weight * (loss.agg_ckl + loss.zkl)

        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
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
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                           qz_logvar.repeat(posterior_sample_n, 1),
                                           sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        # q(c|x)
        qc_logits = self.post_c(x_last).view(-1, self.config.k)
        log_qc = F.log_softmax(qc_logits, -1)  # [batch*mult_k, k]
        sample_c, c_ids = self.cat_connector(qc_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # sample_c: [batch*mult_k, k], c_ids: [batch*mult_k, 1]
        # sample_c = sample_c.view(-1, self.config.mult_k * self.config.k)
        c_ids = c_ids.view(-1, self.config.mult_k)

        # Prepare for decoding
        dec_init_state = self.dec_init_connector(sample_z)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) * (
                        dec_inputs.data - self.eos_id) == 0] = 1
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
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))
            # ZKL & dispersion term
            # zkl = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=True)  # [batch_size x mult_k]
            zkl_real = self.zkl_loss(qc, qz_mean, qz_logvar, mean_prior=False)  # [batch_size x mult_k]
            zkl = torch.gt(zkl_real, self.config.max_fb_z / self.config.mult_k).float() * zkl_real
            zkl = torch.sum(torch.mean(zkl, dim=0))
            zkl_real = torch.sum(torch.mean(zkl_real, dim=0))
            dispersion = self.dispersion(qc)
            # CKL & MI term
            avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)
            # agg_ckl = self.cat_kl_loss(avg_log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            # agg_ckl = torch.sum(agg_ckl)
            ckl_real = self.cat_kl_loss(log_qc, self.log_uniform_y, batch_size, unit_average=True, average=False)
            agg_ckl = torch.gt(ckl_real, self.config.max_fb_c / self.config.mult_k).float() * ckl_real
            ckl_real = torch.sum(torch.mean(ckl_real.view(-1, self.config.mult_k), dim=0))
            agg_ckl = torch.sum(torch.mean(agg_ckl.view(-1, self.config.mult_k), dim=0))
            # H(C) - H(C|X)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / batch_size

            results = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, zkl=zkl, dispersion=dispersion, PPL=ppl,
                           real_zkl=zkl_real, real_ckl=ckl_real, elbo=nll + ckl_real + zkl_real,
                           param_var=self.param_var(tgt_probs=qc))

            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = c_ids
                results['z'] = sample_z

            return results

class GMVAE_MoP(BaseModel):
    def __init__(self, corpus, config):
        super(GMVAE_MoP, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[corpus.unk]
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
        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc)
        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size + self.config.mult_k * self.config.latent_size if self.concat_decoder_input else self.embed_size,
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
                                  embedding=self.embedding)
        self.q_y_mean = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        self.q_y_logvar = nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        self.post_c = nn.Sequential(
            nn.Linear(self.enc_out_size, self.enc_out_size),
            nn.ReLU(),
            nn.Linear(self.enc_out_size, self.config.mult_k * self.config.k),
        )
        self.dec_init_connector = nn_lib.LinearConnector(
            config.latent_size * config.mult_k,
            self.dec_cell_size,
            self.rnn_cell == 'lstm',
            has_bias=False)
        self.cat_connector = nn_lib.GumbelConnector()

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)

        self.init_gaussian()

        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids', 'z')
        self.kl_w = 0.0

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=2, help="The latent size of continuous latent variable.")
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

        return parser

    def init_gaussian(self):
        self._log_uniform_y = Variable(torch.log(torch.ones(1) / self.config.k))
        if self.use_gpu:
            self._log_uniform_y = self.log_uniform_y.cuda()

        mus = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        logvar = torch.randn(self.config.mult_k, self.config.k, self.config.latent_size)
        if torch.cuda.is_available():
            mus = mus.cuda()
            logvar = logvar.cuda()
        self._gaussian_mus = torch.nn.Parameter(mus, requires_grad=True)  # change: False
        self._gaussian_logvar = torch.nn.Parameter(logvar, requires_grad=True)  # change: False

    @property
    def gaussian_mus(self):
        return self._gaussian_mus

    @property
    def gaussian_logvar(self):
        return self._gaussian_logvar

    @property
    def log_uniform_y(self):
        return self._log_uniform_y

    def model_sel_loss(self, loss, batch_cnt):  # return albo
        if "sel_metric" in self.config and self.config.sel_metric == "elbo":
            return loss.elbo
        return self.valid_loss(loss)

    def valid_loss(self, loss, batch_cnt=None, step=None):
        # loss = Pack(nll=nll, agg_ckl=agg_ckl, mi=mi, zkl=zkl, mean_var=mean_var, PPL=ppl,
        #                            real_zkl=zkl_real, real_ckl=ckl_real, elbo=nll + ckl_real + zkl_real,
        #                            param_var=self.param_var(tgt_probs=qc))

        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_step" in self.config else 0,
                                               self.config.anneal_warm_up_step if "anneal_warm_up_value" in self.config else 0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        mi_weight = 0.0 if self.config.use_mutual else 1.0
        total_loss = loss.nll + vae_kl_weight * loss.zkl
        return total_loss

    def reparameterization(self, mu, logvar, sample=True):
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def zkl_loss(self, tgt_probs, mean, log_var, mean_prior=True):
        mean = mean.view(-1, self.config.mult_k, self.config.latent_size)
        log_var = log_var.view(-1, self.config.mult_k, self.config.latent_size)
        if mean_prior:
            tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
            eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
            eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
            Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
            Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
            Emu = -0.5 * Eeta1 / Eeta2
            Evar = -0.5 * torch.pow(Eeta2, -1)
            # [batch_size, mult_k, latent_size]
            kl = 0.5 * (
                    torch.sum(log_var.exp().div(Evar), dim=-1)
                    + torch.sum((Emu - mean).pow(2) / Evar, dim=-1)
                    - mean.size(-1)
                    + torch.sum(Evar.log() - log_var, dim=-1)
            )
            # [batch_size, mult_k]
            return kl

        mu_repeat = mean.unsqueeze(-2).expand(-1, -1, self.config.k, -1)  # batch_size x k x z_dim
        logvar_repeat = log_var.unsqueeze(-2).expand(-1, -1, self.config.k, -1)
        gaussian_logvars = self.gaussian_logvar

        kl = 0.5 * (
                torch.sum(logvar_repeat.exp().div(gaussian_logvars.exp()), dim=-1)
                + torch.sum((self.gaussian_mus - mu_repeat).pow(2) / gaussian_logvars.exp(), dim=-1)
                - mean.size(-1)
                + torch.sum((gaussian_logvars - logvar_repeat), dim=-1)
        )  # batch_size x mult_k x k

        return torch.sum(kl * tgt_probs, dim=-1)  # batch_size*mult_k

    def dispersion(self, tgt_probs):
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)
        Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)  # [batch_size, mult_k, latent_size]
        Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)
        AE = -0.25 * Eeta1 * Eeta1 / Eeta2 - 0.5 * torch.log(-2 * Eeta2)  # [batch_size, mult_k, latent_size]
        AE = torch.mean(torch.sum(AE, dim=(-1, -2)))

        EA = torch.sum(-0.25 * eta1 * eta1 / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1)  # [mult_k, k]
        EA = torch.mean(torch.sum(tgt_probs * EA, dim=(-1, -2)))
        return EA - AE

    def param_var(self, tgt_probs):
        # Weighted variance of natural parameters
        # tgt_probs: batch_size x mult_k x k
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, self.config.latent_size)
        eta1 = self.gaussian_mus / torch.exp(self.gaussian_logvar)  # eta1 = \Sigma^-1 * mu
        eta2 = -0.5 * torch.pow(torch.exp(self.gaussian_logvar), -1)

        var_eta1 = torch.sum(tgt_probs_ * (eta1 * eta1), dim=-2) - torch.sum(tgt_probs_ * eta1, dim=-2).pow(2)
        var_eta2 = torch.sum(tgt_probs_ * (eta2 * eta2), dim=-2) - torch.sum(tgt_probs_ * eta2, dim=-2).pow(2)

        return torch.sum(var_eta1 + var_eta2) / tgt_probs.size(0)

    def _get_pzc(self, sample_z):
        # sample_z: [batch_size, latent_size * multi_k]
        # Prior: [multi_k, k, latent_size]
        bsz = sample_z.size(0)
        multi_k, k, ls = self.gaussian_mus.size()
        gaussian_mus = self.gaussian_mus.unsqueeze(0).expand(bsz, multi_k, k, ls)
        gaussian_logvar = self.gaussian_logvar.unsqueeze(0).expand(bsz, multi_k, k, ls)
        sample_z = sample_z.view(-1, multi_k, 1, ls).expand(bsz, multi_k, k, ls)
        log_pz = - 0.5 * (sample_z - gaussian_mus) * (sample_z - gaussian_mus) / \
                 torch.exp(gaussian_logvar) - 0.5 * math.log(math.pi * 2) - 0.5 * gaussian_logvar
        return torch.sum(log_pz, dim=-1)


    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
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
        sample_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                           qz_logvar.repeat(posterior_sample_n, 1),
                                           sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)
        # q(c|x)
        qc_logits = self.post_c(x_last).view(-1, self.config.k)
        log_qc = F.log_softmax(qc_logits, -1)  # [batch*mult_k, k]
        sample_c, c_ids = self.cat_connector(qc_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        # sample_c: [batch*mult_k, k], c_ids: [batch*mult_k, 1]
        # sample_c = sample_c.view(-1, self.config.mult_k * self.config.k)
        c_ids = c_ids.view(-1, self.config.mult_k)

        # Prepare for decoding
        dec_init_state = self.dec_init_connector(sample_z)
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]
        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) * (
                        dec_inputs.data - self.eos_id) == 0] = 1
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
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            # Regularization terms
            # ZKL:
            log_qz = - 0.5 * (sample_z - qz_mean.repeat(posterior_sample_n, 1)) \
                     * (sample_z - qz_mean.repeat(posterior_sample_n, 1)) / torch.exp(qz_logvar.repeat(posterior_sample_n, 1)) \
                     - 0.5 * qz_logvar.repeat(posterior_sample_n, 1) - 0.5 * math.log(math.pi * 2)
            log_qz = torch.sum(log_qz, dim=-1)
            log_pzc = self._get_pzc(sample_z) # [batch_size x multi_k x k]
            log_pz = torch.sum(torch.log(torch.mean(torch.exp(log_pzc), dim=-1) + 1e-15), dim=-1)
            zkl = torch.mean(log_qz - log_pz)
            # qc = q(z|x) * p(c|z)
            log_qc = F.log_softmax(log_pzc, dim=-1)
            qc = torch.exp(log_qc.view(-1, self.config.mult_k, self.config.k))

            dispersion = self.dispersion(qc)
            # MI term
            avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)
            mi = - torch.sum(torch.exp(avg_log_qc) * avg_log_qc) + torch.sum(torch.exp(log_qc) * log_qc) / log_qc.size(0)

            results = Pack(nll=nll, mi=mi, zkl=zkl, dispersion=dispersion, PPL=ppl,
                           elbo=nll + zkl,
                           param_var=self.param_var(tgt_probs=qc))

            if return_latent:
                results['log_qy'] = log_qc
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = c_ids
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

        log_qzx = torch.sum(
            - (sample_z - qz_mean) * (sample_z - qz_mean) / (
                        2 * torch.exp(qz_logvar)) - 0.5 * qz_logvar - 0.5 * math.log(
                math.pi * 2),
            dim=-1)
        sample_z_repeat = sample_z.view(-1, self.config.mult_k, 1, self.config.latent_size).repeat(1, 1, self.config.k,
                                                                                                   1)
        log_pzc = torch.sum(
            - (sample_z_repeat - self.gaussian_mus) * (sample_z_repeat - self.gaussian_mus) / (
                        2 * torch.exp(self.gaussian_logvar))
            - 0.5 * self.gaussian_logvar - 0.5 * math.log(math.pi * 2),
            dim=-1)  # [batch_size, mult_k, k]
        log_pz = torch.log(torch.mean(torch.exp(log_pzc.double()), dim=-1))  #
        log_pz = torch.sum(log_pz, dim=-1)

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sample_z)
        dec_outs, dec_last, outputs = self.decoder(sample_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)
        if sample_type == "logLL":
            return (-nll.double() + log_pz - log_qzx.double()).view(-1, sample_num)
        else:
            ll = torch.exp(-nll.double() + log_pz - log_qzx.double())  # exp ( log (p(z)p(x|z) / q(z|x)) )
            ll = ll.view(-1, sample_num)
        return ll

    def sampling(self, batch_size):
        sample_c = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        index = (self.torch2var(torch.arange(self.config.mult_k) * self.config.k) + sample_c).view(-1)
        mean = self.gaussian_mus.view(-1, self.config.latent_size)[index].squeeze()
        sigma = torch.exp(self.gaussian_logvar * 0.5).view(-1, self.config.latent_size)[index].squeeze()
        zs = self.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        zs = zs.view(-1, self.config.mult_k * self.config.latent_size)
        dec_init_state = self.dec_init_connector(zs)
        _, _, outputs = self.decoder(zs.size(0),
                                     None, dec_init_state,
                                     mode=GEN, gen_type="greedy",
                                     beam_size=self.config.beam_size,
                                     latent_variable=zs if self.concat_decoder_input else None)
        return outputs

class VAE(BaseModel):
    def __init__(self, corpus, config):
        super(VAE, self).__init__(config)
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
        self.posterior_sample_n = config.post_sample_num if "post_sample_num" in config else 1
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False
        self.use_kl = getattr(config, "use_kl", True)

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell,
                                    n_layers=self.num_layer_enc
                                )

        self.q_z_mean = nn.Linear(self.enc_out_size, config.latent_size)
        self.q_z_logvar = nn.Linear(self.enc_out_size, config.latent_size)

        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.latent_size,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

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
                                  embedding=self.embedding,
                                  softmax_temperature=self.config.softmax_temperature if "softmax_temperature" in self.config else 1.0)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()

        if 'bow_loss' in self.config and self.config.bow_loss:
            self.bow_mlp = nn.Linear(config.latent_size, self.vocab_size)
            self.bow_loss = True
            self.bow_entropy = criterions.BowEntropy(self.rev_vocab[PAD], self.config)
        else:
            self.bow_loss = False

        self.kl_w = 0.0

        self.return_latent_key = ("dec_init_state", "qz_mean", "qz_logvar", "q_z")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=40, help="The latent size of continuous latent variable.")

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

        parser.add_argument('--use_kl', type=str2bool, default=True, help="use_kl=False: AE; use_kl=True, VAE.")
        parser.add_argument('--bow_loss', type=str2bool, default=False, help="adding bow loss to objective.")
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)
        parser.add_argument('--gmm', type=str2bool, default=False)

        return parser

    def valid_loss(self, loss, batch_cnt=None, step = None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0

        if not self.use_kl:
            loss.KL_loss = 0.0
        total_loss = loss.nll + vae_kl_weight * loss.KL_loss

        if self.bow_loss and self.training:
            total_loss += loss.bow_loss

        return total_loss

    def model_sel_loss(self, loss, batch_cnt): # return albo
        if not self.use_kl:
            return loss.nll
        return loss.ELBO

    def reparameterization(self, mu, logvar, batch=False, sample=True):
        if not self.use_kl:
            sample = False
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
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

        # posterior network
        qz_mean = self.q_z_mean(x_last)
        qz_logvar = self.q_z_logvar(x_last)
        q_z = self.reparameterization(qz_mean.repeat(posterior_sample_n, 1),
                                      qz_logvar.repeat(posterior_sample_n, 1), batch=True,
                                      sample=gen_type != "greedy" or mode != GEN)  # batch x (latent_size*mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(q_z)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        if self.config.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(dec_inputs.size())
            prob[(dec_inputs.data - self.go_id) * (dec_inputs.data - self.pad_id) == 0] = 1
            decoder_input_sequence = dec_inputs.clone()
            decoder_input_sequence[prob < self.config.word_dropout_rate] = self.unk_id
            # input_embedding = self.embedding(decoder_input_sequence)
            dec_inputs = decoder_input_sequence

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size * posterior_sample_n,
                                                   dec_inputs.repeat(posterior_sample_n, 1),
                                                   dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size,
                                                   latent_variable=q_z if self.concat_decoder_input else None)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels.repeat(posterior_sample_n, 1))
            ppl = self.ppl(dec_outs, labels.repeat(posterior_sample_n, 1))
            KL_loss = -0.5 * torch.mean(torch.sum((1 + qz_logvar - qz_mean.pow(2) - qz_logvar.exp()), dim=1))

            if not self.use_kl:
                KL_loss = torch.zeros([]).cuda()

            if self.bow_loss:
                bow_logits = self.bow_mlp(q_z)
                bow_loss = self.bow_entropy(F.log_softmax(bow_logits), labels)
            else:
                bow_loss = torch.zeros([]).cuda()

            results = Pack(nll=nll, KL_loss=KL_loss, ELBO=nll+KL_loss, PPL=ppl, bow_loss=bow_loss)

            if return_latent:
                for key in self.return_latent_key:
                    results[key] = eval(key)
            return results

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
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

        qz_mean = self.q_z_mean(x_last)  # [batch_size * sample_num, latent_size]
        qz_logvar = self.q_z_logvar(x_last)
        q_z = self.reparameterization(qz_mean, qz_logvar, batch=True, sample=True)

        log_qzx = torch.sum(
            - (q_z - qz_mean) * (q_z - qz_mean) / (2 * torch.exp(qz_logvar)) -0.5 * qz_logvar - 0.5 * math.log(math.pi * 2),
            dim=-1)

        dec_init_state = self.dec_init_connector(q_z)
        dec_outs, dec_last, outputs = self.decoder(q_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=q_z if self.concat_decoder_input else None)

        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0), -1)
        nll = torch.sum(nll, dim=-1)

        log_pz = torch.sum(- 0.5 * q_z * q_z - 0.5 * math.log(math.pi * 2), dim=-1) # [batch_size * sample_num, ]

        ll = torch.exp(-nll.double() + log_pz.double() - log_qzx.double())  # log (p(z)p(x|z) / q(z|x))

        if sample_type == "logLL":
            return (-nll.double() + log_pz.double() - log_qzx.double()).view(-1, sample_num)
        else:
            ll = ll.view(-1, sample_num)
        return ll

    def sampling(self, batch_size):
        zs = self.torch2var(torch.randn(batch_size, self.config.latent_size))
        dec_init_state = self.dec_init_connector(zs)

        dec_outs, dec_last, outputs = self.decoder(zs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type="greedy",
                                      beam_size=self.config.beam_size,
                                      latent_variable=zs)

        return outputs

class RNNLM(BaseModel):
    def __init__(self, corpus, config):
        super(RNNLM, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size, self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=config.num_layer, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=False,
                                  # attn_size=self.enc_cell_size,
                                  # attn_mode='cat',
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()

        # self.kl_w = 0.0

        for para in self.parameters():
            nn.init.uniform_(para.data, -0.1, 0.1)

        # self.return_latent_key = ("dec_init_state", "qy_mean", "qy_logvar", "q_z")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--num_layer', type=int, default=1)
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)
        return parser

    def valid_loss(self, loss, batch_cnt=None, step = None):
        return loss.nll

    def model_sel_loss(self, loss, batch_cnt):
        return loss.nll

    def reparameterization(self, mu, logvar, batch=False, sample=False):
        if 'use_KL' in self.config and not self.config.use_KL:
            sample = False
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            z = self.torch2var(torch.randn(mu.size()))
            z = z * std + mu
            return z
        else:
            return mu

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # map sample to initial state of decoder
        # dec_init_state = self.dec_init_connector(q_z)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, None,  # dec_init_state
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels)
            ppl = self.ppl(dec_outs, labels)

            results = Pack(nll=nll, PPL=ppl)

            if return_latent:
                for key in self.return_latent_key:
                    results[key] = eval(key)
            return results

    def sampling(self, batch_size):
        _, _, outputs = self.decoder(batch_size,
                                     None, None,  # dec_init_state
                                     mode=GEN, gen_type="sample",
                                     beam_size=self.beam_size)

        return outputs
