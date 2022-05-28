from math import log
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import argparse

import torch
import torch.distributions as tdist
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import datetime
import shutil
import os
import logging
import sys
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.ldebm import LEBM

from modules import clustering

# data loader
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()


    if data == "16gaussians":
        scale = 4.

        centers = []
        for x in [1, 1 - 2/3, -1 + 2/3, -1]:
            for y in [1, 1 - 2/3, -1 + 2/3, -1]:
                centers.append((x, y))        
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(16)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 10
        num_per_class = batch_size // num_classes
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
                 * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    else:
        assert False

def sample_data(args, batch_size):
    x = inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32) * 4.
    return x


# model
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class TrueG(nn.Module):
    def __init__(self, args):
        super(TrueG, self).__init__()
        self.dec_hidden = args.dec_hidden
        self.x_dim = args.x_dim
        self.z_dim = args.z_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.dec_hidden),
            nn.ReLU(),            
            nn.Linear(self.dec_hidden, self.x_dim)
        )

    def forward(self, z):
        x = self.decoder(z)
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        self.proj_head = nn.Linear(args.x_dim, args.enc_hidden)
        self.layers = nn.ModuleList([
                        nn.Sequential(
                            nn.ReLU(),                            
                            nn.Linear(args.enc_hidden, args.enc_hidden),
                        ) for __ in range(2)
                    ])

    def forward(self, x):
        x = self.proj_head(x)
        for layer in self.layers:
            x = layer(x) + x
        return x

class IBEBM(nn.Module):
    def __init__(self, args):
        super(IBEBM, self).__init__()
        self.args = args
        self.x_dim = args.x_dim
        self.dec_hidden = args.dec_hidden
        self.enc_hidden = args.enc_hidden
        self.ebm_hidden = args.ebm_hidden
        self.z_dim = args.z_dim
        self.num_cls = args.num_cls

        self.mi_weight = args.mutual_weight
        self.cls_weight = args.cls_weight
        self.vae_kl_weight = args.max_kl_weight

        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, self.enc_hidden),
            nn.ReLU(),            
            nn.Linear(self.enc_hidden, self.enc_hidden),
        )
        self.mu_proj = nn.Linear(self.enc_hidden, self.z_dim)
        self.log_var = nn.Linear(self.enc_hidden, self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.dec_hidden),
            nn.ReLU(),            
            nn.Linear(self.dec_hidden, self.x_dim)
        )

        self.ebm = LEBM(
            feat_dim=args.z_dim,
            ebm_hdim=args.ebm_hidden,
            emb_dim=args.ebm_hidden,            
            emb_hdim=args.ebm_hidden,
            num_cls=args.num_cls,
            num_blocks=args.num_blocks,
            max_T=args.max_T,            
            use_spc_norm=True,
            e_l_steps=args.e_l_steps,
            e_l_step_size=args.e_l_step_size,
            langevin_noise_scale=1.,
        )

        self.mse = nn.MSELoss()
    
    def inference_forward(self, x):
        x = self.encoder(x)
        mu = self.mu_proj(x)
        log_var = self.log_var(x)
        return mu, log_var

    def sample_posterior(self, mu, log_var, sample=True):
        if sample:
            std = torch.exp(0.5 * log_var)
            z = torch.randn_like(mu)
            z = z * std + mu
            return z
        else:
            return mu    

    def sample_init(self, n):
        return torch.randn(*[n, self.z_dim])    

    def valid_loss(self, loss):
        total_loss = loss.nll + self.vae_kl_weight * (loss.zkl + loss.cd) \
                   - self.mi_weight * loss.mi \
                   + self.cls_weight * loss.cls_loss
        return total_loss

    def forward(self, x, pseudo_labels, batch_cnt=1):
        batch_size = x.size(0)
        mu, log_var = self.inference_forward(x)
        z_post = self.sample_posterior(mu, log_var)
        x_hat = self.decoder(z_post)

        # recon
        recon_loss = self.mse(x, x_hat)

        x_pos, x_neg, x_T_1, x_T, t = self.ebm.sample_latent_training_pairs(z_post)

        # KL(q(z|x) || p(z))
        # E_q(z|x) (f(z))
        log_pos, neg_normal_ll = self.ebm.get_prior_loss(x_pos, x_T_1, t)
        loss_g_kl = - 0.5 * (1 + log_var)        
        # kl_mask = (loss_g_kl > self.args.dim_target_kl).float()
        # zkl = (kl_mask * loss_g_kl).sum(dim=1).mean()
        zkl = loss_g_kl.sum(dim=1).mean()
        zkl += log_pos + neg_normal_ll

        # E_p(z) (f(z))
        cd = self.ebm.get_ebm_loss(x_pos, x_neg, x_T_1, x_T, t)

        # IB
        t0 = torch.zeros(z_post.size(0), dtype=torch.long, device=z_post.device)
        z0 = self.ebm._q_sample(x_start=z_post, t=t0) * self.ebm._extract(self.ebm.a_s_prev, t0 + 1, z_post.shape)
        mi = self.ebm.compute_mi(z0, t0)

        # pseudo-label classification
        logits = self.ebm.ebm_prior(z0, t0, use_cls_output=True)
        cls_labels = pseudo_labels.long() # .squeeze(-1)
        cls_loss = F.cross_entropy(logits, cls_labels)

        loss_dict = Pack(nll=recon_loss, zkl=zkl, cd=cd, mi=mi, cls_loss=cls_loss)

        total_loss = self.valid_loss(loss_dict)

        return total_loss, loss_dict


# utils
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

def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]

def get_output_dir(exp_id, fs_prefix='./'):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(fs_prefix + 'output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def set_seed(seed):
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def plt_samples(samples, ax, iter, output_dir, npts=100, title="$x ~ p(x)$", low=-4, high=4, kde=False, kde_bw=None, divided_sum=False, log_scale=False, log_scale_minus_max=False, divide_sum_log_scale_minus_max=False):
    from scipy.stats import gaussian_kde
    if kde:
        if kde_bw:
            kernel = gaussian_kde(samples.T, bw_method=kde_bw)
        else:
            kernel = gaussian_kde(samples.T)
        # side = np.linspace(low, high, npts)
        # xx, yy = np.meshgrid(side, side)
        # x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        X, Y = np.mgrid[low:high:100j, low:high:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        if divide_sum_log_scale_minus_max:
            Z = np.log(Z + 1e-10)
            Z = Z - Z.max()
            Z = np.exp(Z)
            Z = Z / Z.sum()
        if log_scale_minus_max:
            Z = np.log(Z + 1e-10)
            Z = Z - Z.max()
        if log_scale:
            Z = np.log(Z + 1e-10)
        if divided_sum:
            Z = Z / Z.sum()
        ax.imshow(Z, cmap='viridis', extent=[low, high, low, high])

        fig = plt.figure(figsize=(8, 8))
        plt.xlim([low, high])
        plt.ylim([low, high])
        plt.imshow(Z, cmap='viridis', extent=[low, high, low, high])
        plt.axis('off')
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(fname=os.path.join(output_dir, title + '_{}.png'.format(iter)), bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    else:
        ax.hist2d(samples[:, 0], samples[:, 1], range=[[low, high], [low, high]], bins=280)

        fig = plt.figure(figsize=(8, 8))
        plt.xlim([low, high])
        plt.ylim([low, high])
        plt.hist2d(samples[:, 0], samples[:, 1], range=[[low, high], [low, high]], bins=280)
        plt.axis('off')
        plt.gcf().set_size_inches(8, 8)
        plt.savefig(fname=os.path.join(output_dir, title + '_{}.png'.format(iter)), bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title, fontsize=15)

def visualize_samples(samples, sample_names, iter, output_dir, n_col=3, n_row=2,
    npts=100, memory=100, device="cpu", low=-4, high=4, kde=False, kde_bw=None, divided_sum=False, log_scale=False, log_scale_minus_max=False, divide_sum_log_scale_minus_max=False):
    """Produces visualization for the model density and samples from the model."""
    n_samples = len(samples)
    # n_col = n_samples
    # n_row = n_samples // n_col
    for i, name in zip(range(n_samples), sample_names):
        if i == 0:
            plt.clf()
            kde = False
        else:
            kde = True
        if i + 1 >= 4:
            ax = plt.subplot(n_row, n_col, i + 2, aspect="equal")
        else:
            ax = plt.subplot(n_row, n_col, i + 1, aspect="equal")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        plt_samples(samples[i], ax, iter, output_dir, npts=npts, title=name, low=low, high=high, kde=kde, kde_bw=kde_bw, divided_sum=divided_sum, log_scale=log_scale, log_scale_minus_max=log_scale_minus_max, divide_sum_log_scale_minus_max=divide_sum_log_scale_minus_max)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        choices=['16gaussians', 'pinwheel'],
        type=str, default='16gaussians'
    )
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--niters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--viz_freq', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--x_dim', type=int, default=2)
    parser.add_argument('--dec_hidden', type=int, default=200)
    parser.add_argument('--enc_hidden', type=int, default=200)
    parser.add_argument('--ebm_hidden', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--num_cls', type=int, default=16)
    parser.add_argument('--seed', default=3434, type=int)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ebm_weight_decay', type=float, default=0)

    parser.add_argument('--e_l_steps', type=int, default=50, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=2e-3, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_prior_sig', type=float, default=1, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=1, help='sigma of initial distribution')

    parser.add_argument('--max_T', type=int, default=6)
    parser.add_argument('--beta_st', type=float, default=1e-4)
    parser.add_argument('--beta_ed', type=float, default=2e-2)
    parser.add_argument('--num_blocks', type=int, default=12)

    parser.add_argument('--dim_target_kl', type=float, default=1)
    parser.add_argument('--max_kl_weight', type=float, default=1)
    parser.add_argument('--mutual_weight', type=float, default=0.05)
    parser.add_argument('--cls_weight', type=float, default=0.05)

    # KMeans
    parser.add_argument('--reassign', type=float, default=1., 
                        help="""how many epochs of training between two consecutive reassignments of clusters (default: 1)""")
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    set_seed(args.seed)
    set_gpu(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = IBEBM(args)
    model.to(device)
    model.train()

    data_model = TrueG(args)
    data_model.to(device)
    data_model.eval()

    params_ae = [p[1] for p in model.named_parameters() if 'ebm' not in p[0] and p[1].requires_grad is True]
    params_ebm = [p[1] for p in model.named_parameters() if 'ebm' in p[0] and p[1].requires_grad is True]
    optimizer_ae = optim.Adam(params_ae, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_ebm = optim.Adam(params_ebm, lr=args.emb_lr, weight_decay=args.ebm_weight_decay)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.num_cls)

    # collect data
    data_gathered = np.zeros((args.batch_size * args.niters, args.x_dim), dtype='float32')
    st_time = time.time()
    for itr in range(args.niters):
        # sample data
        _x = sample_data(args, args.batch_size)
        aux_x = _x.cpu().numpy()
        data_gathered[itr * args.batch_size: (itr + 1) * args.batch_size] = aux_x        
    print('Done data collection. Time Elapsed: {}'.format(time.time() - st_time))

    for epoch in range(args.nepochs):

        # compute features for clustering
        features = np.zeros((args.batch_size * args.niters, args.z_dim), dtype='float32')

        st_time = time.time()
        for itr in range(args.niters):
            # sample data
            _x = data_gathered[itr * args.batch_size: (itr + 1) * args.batch_size]
            _x = torch.from_numpy(_x).to(device).detach()
            mu, __ = model.inference_forward(_x)
                        
            aux_z = mu.detach().cpu().numpy()
            features[itr * args.batch_size: (itr + 1) * args.batch_size] = aux_z
        print('Done feature extraction. Time Elapsed: {}'.format(time.time() - st_time))

        # cluster the features
        print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, pca_dim=2, verbose=True)

        # assign pseudo-labels
        print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.data_clusters,
                                                  data_gathered)

        # uniformly sample per target
        sampler = clustering.UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.data_clusters)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        model.train()

        print('Start training')        
        for itr, (x, pseudo_labels) in enumerate(train_dataloader):    
            x = x.to(device)
            pseudo_labels = pseudo_labels.to(device)

            optimizer_ae.zero_grad()
            optimizer_ebm.zero_grad()

            total_loss, loss_dict = model(x, pseudo_labels, batch_cnt=itr)

            total_loss.backward()
            optimizer_ae.step()
            optimizer_ebm.step()


            if itr % args.log_freq == 0:
                logger.info(
                    'itr:{:0>5d} '.format(itr) +
                    'recon_loss:{:>8.6f} '.format(loss_dict.nll) +
                    'kl:{:>8.6f} '.format(loss_dict.zkl) +                
                    'cd:{:>8.6f} '.format(loss_dict.cd) +
                    'mi:{:>8.6f} '.format(loss_dict.mi) + 
                    'cls_loss:{:>8.6f}'.format(loss_dict.cls_loss)
                )

            if itr % args.viz_freq == 0:
                model.eval()
                # model.cpu()
                # data_model.cpu()

                # sample data
                npts = 1000
                batch_size = npts ** 2
                _x = sample_data(args, batch_size=batch_size)
                x = _x.detach().clone().to(device)

                # posterior samples
                mu, log_var = model.inference_forward(x)
                inferred_z = model.sample_posterior(mu, log_var, sample=True)
                x_hat_posterior = model.decoder(inferred_z)

                # scale inferred_z
                inferred_z = inferred_z.detach().cpu().numpy()
                z_trans = inferred_z.min()
                z_scale = inferred_z.max() - inferred_z.min()
                inferred_z = (inferred_z - z_trans) / z_scale * 8 - 4

                # prior samples
                z_prior_0 = model.sample_init(batch_size).to(device)
                z_prior = model.ebm.p_sample_progressive(z_prior_0)
                x_hat_prior = model.decoder(z_prior)

                # scale prior z
                z_prior = z_prior.detach().cpu().numpy()
                z_prior = (z_prior - z_trans) / z_scale * 8 - 4

                # scale x
                x = x.detach().cpu().numpy()
                x_trans = x.min()
                x_scale = x.max() - x.min()
                x = (x - x_trans) / x_scale * 8 - 4

                # scale x_hat_prior
                x_hat_prior = x_hat_prior.detach().cpu().numpy()
                x_hat_prior = (x_hat_prior - x_trans) / x_scale * 8 - 4

                # scale x_hat_posterior
                x_hat_posterior = x_hat_posterior.detach().cpu().numpy()
                x_hat_posterior = (x_hat_posterior - x_trans) / x_scale * 8 - 4


                plt.clf()
                plt.figure(figsize=(6, 4))
                visualize_samples([x, x_hat_posterior, x_hat_prior, inferred_z, z_prior], 
                ["true_x", "posterior_x", "prior_x", "posterior_z", "prior_z"], iter=epoch * args.niters + itr, output_dir=output_dir, npts=npts, kde=True, kde_bw=0.15)
                fig_filename = os.path.join(output_dir, '{:04d}_vanilla_kde_bw.png'.format(epoch * args.niters + itr))
                plt.savefig(fig_filename)
                plt.close()


                model.to(device)
                model.train()
                data_model.to(device)

        torch.save(model.state_dict(), os.path.join(output_dir, "ckpt_{}.pt".format(epoch)))

if __name__ == "__main__":
    # logger
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)
    logger = setup_logging('main', output_dir)

    main()


