import sys
import os

import argparse
import json
import random
import shutil
import copy
import logging
import datetime
import pickle
import itertools
import time
import math

import numpy as np

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

# import torchvision
# import torchvision.transforms as transforms
from torch.nn.utils import weight_norm

from ldebm import LEBM

import pygrid


##########################################################################################################
## Parameters

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/agnews')
    parser.add_argument('--precomputed_bg_file', type=str, default='data/agnews/vampire.bgfreq')
    parser.add_argument('--vocab_size', type=int, default=30001)



    parser.add_argument('--seed', default=8888, type=int)
    parser.add_argument('--n_labels', type=int, default=2500, help='number of labels per class')
    parser.add_argument('--labeled_weight', default=200., type=float)
    parser.add_argument('--y_z_update', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--z_y_update', default=False, type=bool, help='noise term of langevin')
    parser.add_argument('--ebm_update_unlabeled', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--noise_factor', default=1., type=float)

    parser.add_argument('--max_kl_weight', type=float, default=0.1)
    parser.add_argument('--mutual_weight', type=float, default=0.05)

    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')

    parser.add_argument('--dataset', type=str, default='agnews_bow', choices=['agnews_bow'])
    parser.add_argument('--img_size', default=30001, type=int)
    parser.add_argument('--batch_size', default=200, type=int)

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', default=3)

    parser.add_argument('--nez', default=4, help='size of the output of ebm')
    parser.add_argument('--ngf', default=64, help='feature dimensions of generator')
    parser.add_argument('--nif', default=200, help='feature dimensions of inference network')

    parser.add_argument('--ndf', default=200, help='feature dimensions of ebm')

    parser.add_argument('--e_prior_sig', type=float, default=1, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=1, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=50, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=2e-3, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')

    parser.add_argument('--i_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--i_activation_leak', type=float, default=0.2)
    parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='prior of factor analysis')
    parser.add_argument('--g_activation', type=str, default='lrelu')
    parser.add_argument('--g_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin')
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--g_batchnorm', default=False, type=bool, help='batch norm')

    parser.add_argument('--e_lr', default=0.00001, type=float)
    parser.add_argument('--g_lr', default=0.0001, type=float)
    parser.add_argument('--i_lr', default=0.0001, type=float)

    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--i_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')


    parser.add_argument('--e_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--i_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--e_decay', default=1e-3, help='weight decay for ebm')
    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')
    parser.add_argument('--i_decay',  default=0.002, help='weight decay for gen')


    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--g_gamma', default=0.998, help='lr decay for gen')
    parser.add_argument('--i_gamma', default=0.998, help='lr decay for gen')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--e_beta1', default=0.9, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)

    parser.add_argument('--i_beta1', default=0.9, type=float)
    parser.add_argument('--i_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
    # parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=100, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')
    parser.add_argument('--n_ckpt', type=int, default=40, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=500, help='fid each n epochs')
    parser.add_argument('--n_cls_validataion', type=int, default=1, help='classification each n epochs')
    # parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')

    parser.add_argument('--n_fid_samples', type=int, default=50000) # TODO(nijkamp): we used 40,000 in short-run inference
    # parser.add_argument('--n_fid_samples', type=int, default=1000)


    # wide network
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--widen_factor', default=3, type=int)
    parser.add_argument('--dropRate', default=0.3, type=float)
    parser.add_argument('--leakyRate', default=0.1, type=float)


    return parser.parse_args()


def create_args_grid():
    # TODO add your enumeration of parameters here

    e_lr = [0.00002]
    e_l_step_size = [0.4]
    e_init_sig = [1.0]
    e_l_steps = [60]
    e_activation = ['lrelu']

    g_llhd_sigma = [0.3]
    g_lr = [0.0001]
    g_l_steps = [20]
    g_activation = ['lrelu']

    ngf = [64]
    ndf = [100]

    args_list = [e_lr, e_l_step_size, e_init_sig, e_l_steps, e_activation, g_llhd_sigma, g_lr, g_l_steps, g_activation, ngf, ndf]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'e_lr': args[0],
            'e_l_step_size': args[1],
            'e_init_sig': args[2],
            'e_l_steps': args[3],
            'e_activation': args[4],
            'g_llhd_sigma': args[5],
            'g_lr': args[6],
            'g_l_steps': args[7],
            'g_activation': args[8],
            'ngf': args[9],
            'ndf': args[10],
        }
        # TODO add your result metric here
        opt_result = {'fid_best': 0.0, 'fid': 0.0, 'mse': 0.0}

        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['fid_best'] = job_stats['fid_best']
    job_opt['fid'] = job_stats['fid']
    job_opt['mse'] = job_stats['mse']


##########################################################################################################
## Data
def compute_background_log_frequency(vocab_size, precomputed_bg_file):
    """
    Load in the word counts from the JSON file and compute the
    background log term frequency w.r.t this vocabulary.
    """
    # precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
    log_term_frequency = torch.FloatTensor(vocab_size)
    log_term_frequency[0] = 1e-12
    with open(precomputed_bg_file, "r") as file_:
        precomputed_bg = json.load(file_)
    for i, token in enumerate(precomputed_bg.keys(), start=1):
        if precomputed_bg[token] == 0:
            log_term_frequency[i] = 1e-12
        else:
            log_term_frequency[i] = precomputed_bg[token]
    log_term_frequency = torch.log(log_term_frequency)
    return log_term_frequency




class AgNewsBowDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, _min_sequence_length=3):
        from scipy import sparse

        npy = np.load(data_path)
        coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
        coo_matrix = coo_matrix.tocsc()
        coo_matrix = coo_matrix.tolil()
        labels = load_labels(label_path)
        self.coo_matrix = [(el, l) for el, l in zip(coo_matrix, labels) if el.toarray().sum() > _min_sequence_length]

    def __len__(self):
        return len(self.coo_matrix)
    
    def __getitem__(self, idx):
        # return {'sentence': torch.tensor(self.coo_matrix[idx][0].toarray().squeeze(), dtype=torch.float),
        #         'label': torch.tensor(self.coo_matrix[idx][1], dtype=torch.long)} 
        return (torch.tensor(self.coo_matrix[idx][0].toarray().squeeze(), dtype=torch.float),
                torch.tensor(self.coo_matrix[idx][1], dtype=torch.long))

def load_labels(data_path):
    labeles = []
    with open(data_path, "r")as f:
        for line in f:
            example = json.loads(line)
            labeles.append(example['label'] - 1)
    return labeles



def get_dataset(args):

    if args.dataset == 'agnews_bow':
        train_path = os.path.join(args.data_path, 'train.npz')
        dev_path = os.path.join(args.data_path, 'dev.npz')
        train_label_path = os.path.join(args.data_path, 'train.jsonl')
        dev_label_path = os.path.join(args.data_path, 'dev.jsonl')
        ds_train = AgNewsBowDataset(train_path, train_label_path)
        ds_val = AgNewsBowDataset(dev_path, dev_label_path)
        return ds_train, ds_val


        


 

##########################################################################################################
## Model

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

def get_activation(name, args):
    return {'gelu': GELU(), 'lrelu': nn.LeakyReLU(args.e_activation_leak), 'mish': Mish(), 'swish': Swish()}[name]


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, args):
        super().__init__()

        f = get_activation(args.g_activation, args)
        
        self.gen = nn.Sequential(
            nn.Linear(args.nz, 2000),
            f,
            nn.Linear(2000, 2000),
            f,
            nn.Linear(2000, 2000),
            f,
            nn.Linear(2000, args.img_size) #TODO (BP): how to normalize the data?
        )

        self.bow_bn = torch.nn.BatchNorm1d(args.img_size, eps=0.001, momentum=0.001, affine=True)
        self.bow_bn.weight.data.copy_(torch.ones(args.img_size, dtype=torch.float64))
        self.bow_bn.weight.requires_grad = False

    
    def forward(self, z):
        return self.bow_bn(self.gen(z))


class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        apply_sn = sn if args.e_sn else lambda x: x

        f = get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(args.nz, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.nez))
        )

    def compute_mi(self, z, eps=1e-15):
        z = z.squeeze()
        assert len(z.size()) == 2
        batch_size = z.size(0)
        log_p_y_z = F.log_softmax(self.forward(z, cls_output=True), dim=-1)
        p_y_z = torch.exp(log_p_y_z)
        
        # H(y)
        log_p_y = torch.log(torch.mean(p_y_z, dim=0) + eps)
        H_y = - torch.sum(torch.exp(log_p_y) * log_p_y)

        # H(y|z)
        H_y_z = - torch.sum(log_p_y_z * p_y_z) / batch_size

        mi = H_y - H_y_z

        return mi


    def forward(self, z, cls_output=False):
        if cls_output:
            return self.ebm(z.squeeze()).view(-1, self.args.nez)
        else:
            return self.ebm(z.squeeze()).view(-1, self.args.nez).logsumexp(dim=1)


class _netI(nn.Module):
    def __init__(self, args):
        super(_netI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(args.img_size, 2000),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Dropout(.5),
        )
        self.mu_proj = nn.Linear(2000, args.nz)
        self.log_sigma_proj = nn.Linear(2000, args.nz)



    def forward(self, x):
        out = self.net(x)
        mu = self.mu_proj(out)
        log_sigma = self.log_sigma_proj(out)
        return mu, log_sigma


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, leakyRate=0.01, actBeforeRes=True):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(leakyRate, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(leakyRate, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activateBeforeResidual= actBeforeRes
    def forward(self, x):
        if not self.equalInOut and self.activateBeforeResidual:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu1(self.bn1(x)))
        #out = self.conv1(out if self.equalInOut else x)
        #out = self.conv1(self.equalInOut and out or x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        res = self.convShortcut(x) if not self.equalInOut else x
        return torch.add(res, out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, leakyRate=0.01, actBeforeRes=True):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, leakyRate, actBeforeRes)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, leakyRate, actBeforeRes):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, leakyRate, actBeforeRes))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, args):
        super(WideResNet, self).__init__()
        depth = args.depth
        widen_factor = args.widen_factor
        dropRate = args.dropRate
        leakyRate = args.leakyRate

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, leakyRate, False)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, leakyRate, True)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, leakyRate, True)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(leakyRate, inplace=True)
        self.fc_mu = nn.Linear(nChannels[3], nChannels[3])
        self.fc_sd = nn.Linear(nChannels[3], nChannels[3])

        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc_mu(out).unsqueeze(-1).unsqueeze(-1), self.fc_sd(out).unsqueeze(-1).unsqueeze(-1)


##########################################################################################################

def train(args_job, output_dir_job, output_dir, return_dict):

    #################################################
    ## preamble

    args = parse_args()
    args = pygrid.overwrite_opt(args, args_job)
    args = to_named_dict(args)

    set_gpu(args.gpu)
    set_seed(args.seed)

    makedirs_exp(output_dir)

    job_id = int(args['job_id'])

    logger = setup_logging('job{}'.format(job_id), output_dir, console=True)
    logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #################################################
    ## data
    def extract_labeled_data_agnews(ds_train, num_sample=1000, num_cat=4):
        all_labels = np.array([l.item() for _, l in ds_train])
        _images = []
        _labels = []
        for category in range(num_cat):
            indexes = np.where(all_labels == category)[0][:num_sample]
            for index in indexes:
                _images.append(ds_train[index][0])
                _labels.append(ds_train[index][1])
        images = torch.stack(_images)
        labels = torch.stack(_labels)
        return (images, labels)

    def sample_labeled_data(labeled_data, batch_size=100):
        images, labels = labeled_data
        assert images.size(0) == labels.size(0)
        rand_index = torch.LongTensor(batch_size).random_(0, images.size(0))
        image_batch = images[rand_index].clone().detach()
        label_batch = labels[rand_index].clone().detach()
        return image_batch, label_batch

    ds_train, ds_val = get_dataset(args)
    logger.info('len(ds_train)={}'.format(len(ds_train)))
    logger.info('len(ds_val)={}'.format(len(ds_val)))

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    labeled_data = extract_labeled_data_agnews(ds_train, num_sample=args.n_labels, num_cat=args.nez)
    bg_freq = compute_background_log_frequency(vocab_size=args.vocab_size, precomputed_bg_file=args.precomputed_bg_file)

    #################################################
    ## model

    netG = _netG(args)
    netE = LEBM(
        feat_dim=args.nz,
        ebm_hdim=args.ndf,
        emb_dim=args.ndf,
        emb_hdim=args.ndf,
        num_cls=args.nez,
        num_blocks=12,
        max_T=6,
        use_spc_norm=True,
        e_l_steps=args.e_l_steps,
        e_l_step_size=args.e_l_step_size,
        langevin_noise_scale=1.,
    )
    # _netE(args)
    netI = _netI(args)

    netG.apply(weights_init_xavier)
    netE.apply(weights_init_xavier)
    netI.apply(weights_init_xavier)

    netG = netG.to(device)
    netE = netE.to(device)
    netI = netI.to(device)

    logger.info(netG)
    logger.info(netE)
    logger.info(netI)

    def eval_flag():
        netG.eval()
        netE.eval()
        netI.eval()

    def train_flag():
        netG.train()
        netE.train()
        netI.train()

    def energy(score):
        if args.e_energy_form == 'tanh':
            energy = F.tanh(-score.squeeze())
        elif args.e_energy_form == 'sigmoid':
            energy = F.sigmoid(score.squeeze())
        elif args.e_energy_form == 'identity':
            energy = - score.squeeze()
        elif args.e_energy_form == 'softplus':
            energy = F.softplus(score.squeeze())
        return energy

    # mse = nn.MSELoss(reduction='sum')

    def mse(x_hat, x):
        log_x_hat = F.log_softmax(x_hat + 1e-10, dim=-1)
        reconstruction_loss = - torch.sum(x * log_x_hat, dim=-1)
        return reconstruction_loss.sum()

    #################################################
    ## optimizer

    optE = torch.optim.Adam(netE.parameters(), lr=args.e_lr, weight_decay=args.e_decay, betas=(args.e_beta1, args.e_beta2))
    optG = torch.optim.Adam(netG.parameters(), lr=args.g_lr, weight_decay=args.g_decay, betas=(args.g_beta1, args.g_beta2))
    optI = torch.optim.Adam(netI.parameters(), lr=args.i_lr, weight_decay=args.i_decay, betas=(args.i_beta1, args.i_beta2))


    lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optE, args.e_gamma)
    lr_scheduleG = torch.optim.lr_scheduler.ExponentialLR(optG, args.g_gamma)
    lr_scheduleI = torch.optim.lr_scheduler.ExponentialLR(optI, args.i_gamma)


    #################################################
    ## sampling

    def sample_amortized_post_z(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_p_0(n=args.batch_size, sig=args.e_init_sig):
        return sig * torch.randn(*[n, args.nz]).to(device)    

    #################################################
    ## classification
    def classification(dataloader, netI, netE):
        accs = []
        for i, (x, y) in enumerate(dataloader):
            eval_flag()
            x = x.to(device)
            y = y.to(device)
            batch_size = x.shape[0]
            mu, _ = netI(x)
            t0 = torch.zeros(
                mu.size(0), dtype=torch.long, device=mu.device
            )
            scores = netE.ebm_prior(mu, t0, use_cls_output=True)
            assert scores.size() == (x.size(0), args.nez)
            predictions = scores.argmax(dim=1)
            acc = (predictions == y).float().mean()
            accs.append(acc)
        accs = torch.stack(accs)

        return accs.mean()

    #################################################
    ## train

    train_flag()

    fid = 0.0
    fid_best = math.inf

    # z_fixed = sample_p_0()
    # x_fixed = next(iter(dataloader_train))[0].to(device)

    stats = {
        'loss_g':[],
        'loss_e':[],
        'en_neg':[],
        'en_pos':[],
        'grad_norm_g':[],
        'grad_norm_e':[],
        # 'z_e_grad_norm':[],
        # 'z_g_grad_norm':[],
        'z_e_k_grad_norm':[],
        'fid':[],
        'cls_loss':[],
        'cls_acc':[],
    }
    interval = []
    val_acc = torch.tensor(0.)
    best_val_acc = torch.tensor(0.)
    bg_freq = bg_freq.clone().detach().to(device)

    for epoch in range(args.n_epochs):

        cls_losses = []
        train_acces = []
        for i, (x, _) in enumerate(dataloader_train, 0):

            # if i > 2:
            #     break

            train_flag()

            x = x.to(device)
            batch_size = x.shape[0]
            target = x.clone().detach()

            # labled data
            labeled_x, labeled_y = sample_labeled_data(labeled_data, batch_size=batch_size)
            labeled_x = labeled_x.to(device)
            labeled_y = labeled_y.to(device)


            #------------------------------------
            if args.y_z_update:
                mu, log_var = netI(labeled_x)                
                t0 = torch.zeros(
                        mu.size(0), dtype=torch.long, device=mu.device
                    )
                scores = netE.ebm_prior(mu, t0, use_cls_output=True)
                assert scores.size() == (batch_size, args.nez)
                optE.zero_grad()
                optI.zero_grad()
                cls_loss = args.labeled_weight * F.cross_entropy(scores, labeled_y, reduction='mean')
                cls_loss.backward()
                grad_norm_e_labeled = get_grad_norm(netE.parameters())
                grad_norm_i_labeled = get_grad_norm(netI.parameters())
                if args.e_is_grad_clamp:
                    torch.nn.utils.clip_grad_norm_(netE.parameters(), args.e_max_norm)
                    torch.nn.utils.clip_grad_norm(netI.parameters(), args.g_max_norm)
                optE.step()
                optI.step()
            else:
                cls_loss = torch.tensor(0.)
            cls_losses.append(cls_loss / args.labeled_weight)
            with torch.no_grad():
                predictions = scores.argmax(dim=1)
                train_acc = (predictions == labeled_y).float().mean()
                train_acces.append(train_acc)

            #---------------------------training with unlabeled data--------------------------#
            # Langevin posterior and prior

            optG.zero_grad()
            optI.zero_grad()
            optE.zero_grad()

            mu, log_var = netI(x)
            z_g_k = sample_amortized_post_z(mu, log_var)            

            x_hat = netG(z_g_k)
            loss_g_recon = mse(x_hat + bg_freq, x) / batch_size # TODO (BP): 0.5 * g_llhd_sigma ** 2

            x_pos, x_neg, x_T_1, x_T, t = netE.sample_latent_training_pairs(
                                            z_g_k)

            # KL(q(z|x) || p(z))
            # E_q(z|x) (f(z))
            log_pos, neg_normal_ll = netE.get_prior_loss(x_pos, x_T_1, t)
            loss_g_kl = - 0.5 * (1 + log_var)
            # kl_mask = (loss_g_kl > args.dim_target_kl).float()
            # zkl = (kl_mask * loss_g_kl).sum(dim=1).mean()
            zkl = loss_g_kl.sum(dim=1).mean()
            zkl += log_pos + neg_normal_ll

            # E_p(z) (f(z))
            cd = netE.get_ebm_loss(x_pos, x_neg, x_T_1, x_T, t)

            # IB
            t0 = torch.zeros(
                    z_g_k.size(0), dtype=torch.long, device=z_g_k.device)
            z0 = netE._q_sample(
                    x_start=z_g_k,
                    t=t0
                ) * netE._extract(netE.a_s_prev, t0 + 1, z_g_k.shape)
            mi = netE.compute_mi(z0, t0)

            abp_loss = loss_g_recon + args.max_kl_weight * (zkl + cd) \
                     - args.mutual_weight * mi

            abp_loss.backward()
                        
            optG.step()
            optI.step()
            optE.step()            

            # Printout
            if i % args.n_printout == 0:
                logger.info(
                    'batch/max_batch/ep: {:5d}/ {:5d}/ {:5d} '.format(
                        i, len(dataloader_train), epoch) +
                    'rec: {:8.3f} '.format(loss_g_recon) +
                    'zkl: {:8.3f} '.format(zkl) + 
                    'cd: {:8.3f} '.format(cd) +
                    'mi: {:10.8f} '.format(mi) + 
                    'kl_weight: {:8.3f}'.format(args.max_kl_weight)
                )


        # classification accuracy
        if epoch % args.n_cls_validataion == 0:
            val_acc = classification(dataloader_val, netI, netE)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                save_dict = {
                    'epoch': epoch,
                    'netE': netE.state_dict(),
                    'optE': optE.state_dict(),
                    'netG': netG.state_dict(),
                    'optG': optG.state_dict(),
                }
                torch.save(
                    save_dict, '{}/ckpt/ckpt_{:>06d}_{:.4f}.pth'.format(
                        output_dir, epoch, best_val_acc)
                )

            logger.info(
                'validation classification accuracy={:8.6f}|{:8.6f}'.format(
                    val_acc, best_val_acc)
            )

        # Schedule
        lr_scheduleE.step()
        lr_scheduleG.step()
        lr_scheduleI.step()            

    return_dict['stats'] = {'fid_best': fid_best, 'fid': fid}
    logger.info('done')



##########################################################################################################
## Metrics

# from fid_v2_tf_cpu import fid_score

def is_xsede():
    import socket
    return 'psc' in socket.gethostname()




##########################################################################################################
## Plots

import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    # f = plt.figure(figsize=(20, len(content) * 5))
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        axs[j].plot(interval, v)
        axs[j].set_ylabel(k)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)



##########################################################################################################
## Other

def get_grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


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


def set_gpu(gpu, deterministic=True):
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

  if torch.cuda.is_available():
    torch.cuda.set_device(0)

    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


##########################################################################################################
## Main

def makedirs_exp(output_dir):
    os.makedirs(output_dir + '/samples')
    os.makedirs(output_dir + '/ckpt')

def main():

    fs_prefix = './'

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)

    # run
    copy_source(__file__, output_dir)
    opt = {'job_id': int(0), 'status': 'open'}
    train(opt, output_dir, output_dir, {})


if __name__ == '__main__':
    main()
