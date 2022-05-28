import numpy as np

import types
import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_T,
        emb_dim,
        emb_hdim,
        use_spc_norm=True
    ):
        """
        :param emb_dim: embedding dimension
        :param max_len: maximum diffusion step
        """
        super(PositionalEmbedding, self).__init__()
        
        # pos embedding table
        pe = torch.zeros(max_T, emb_dim)
        position = torch.arange(0, max_T).unsqueeze(1) # size=(max_T, 1)
        div_term = torch.exp(
                         torch.arange(0, emb_dim, 2) * # 10000 **（-2i / emb_dim)
                       -(np.log(10000.0) / emb_dim)
                    )
        pe[:, 0::2] = torch.sin(position * div_term) # even num. indx
        pe[:, 1::2] = torch.cos(position * div_term) # odd num. indx
        
        # size=(max_T, emb_dim)
        self.register_buffer('pe', pe)  # pos emb. is fixed during training

        self.model = nn.Sequential(
                        spectral_norm(
                            nn.Linear(emb_dim, emb_hdim),
                            mode=use_spc_norm
                        ),
                        nn.LeakyReLU(0.2),
                        spectral_norm(
                            nn.Linear(emb_hdim, emb_hdim),
                            mode=use_spc_norm
                        )
                    )

    def forward(self, t):
        """
        query the pre-computed embedding table, and project the pos-emb. 
        to latent space
        """                
        return self.model(self.pe[t]) # size=(batch, emb_dim)

class LEBM(nn.Module):
    def __init__(
        self,
        feat_dim,
        ebm_hdim,
        emb_dim,
        emb_hdim,
        num_cls=1,
        num_blocks=16,
        max_T=6,
        beta_st=1e-4,
        beta_ed=2e-2,
        use_spc_norm=True,
        e_l_steps=30,
        e_l_step_size=2e-4,
        langevin_noise_scale=1.,
        init_weights=True # False
    ):
        """
        :param feat_dim: dim. of the input latent code
        :param ebm_hdim: output dim. of the input embedding
        """
        super(LEBM, self).__init__()            
        ### network arch.
        self.feat_dim = feat_dim

        # input embedding for latent code
        self.ze = nn.Sequential(
                    spectral_norm(
                        nn.Linear(feat_dim, ebm_hdim),
                        mode=use_spc_norm
                    ),
                    nn.LeakyReLU(0.2),
                    spectral_norm(
                        nn.Linear(ebm_hdim, ebm_hdim),
                        mode=use_spc_norm
                    )
                )        

        # pos embedding for timestep
        self.te = PositionalEmbedding(                    
                    emb_dim=emb_dim,
                    emb_hdim=emb_hdim,
                    max_T=max_T,
                    use_spc_norm=use_spc_norm
                    )

        self.concat_proj = nn.Sequential(                        
                        nn.LeakyReLU(0.2),
                        spectral_norm(
                            nn.Linear(emb_hdim + ebm_hdim, ebm_hdim),
                            mode=use_spc_norm
                        )
                    )

        self.layers = nn.ModuleList([
                        nn.Sequential(                            
                            nn.LeakyReLU(0.2),
                            spectral_norm(
                                nn.Linear(ebm_hdim, ebm_hdim),
                                mode=use_spc_norm
                            ) 
                        ) for __ in range(num_blocks)
                    ])

        self.out_z = nn.Sequential(
                        nn.LeakyReLU(0.2),
                        spectral_norm(
                            nn.Linear(ebm_hdim, num_cls),
                            mode=use_spc_norm
                        )
                    )

        ### diffusion
        self.max_T = max_T

        self.sigmas, self.a_s = self._get_sigma_schedule(
                beta_start=beta_st, 
                beta_end=beta_ed, 
                num_diffusion_timesteps=max_T
            )
        self.a_s_cum = np.cumprod(self.a_s)
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.copy()
        self.a_s_prev[-1] = 1

        self.is_recovery = np.ones(self.max_T + 1, dtype=np.float32)
        self.is_recovery[-1] = 0

        ### sampling
        self.e_l_steps = e_l_steps
        self.e_l_step_size = e_l_step_size
        self.langevin_noise_scale = langevin_noise_scale

        if init_weights:
            self._init_weights()

    def _init_weights(self, init_type='orthogonal', gain=1.):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/
        9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    @staticmethod
    def _set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=False for all the networks to 
           avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks 
                                     require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad        

    @staticmethod
    def _get_sigma_schedule(beta_start, beta_end, num_diffusion_timesteps):
        """
        Get the noise level schedule
        :param beta_start: begin noise level
        :param beta_end: end noise level
        :param num_diffusion_timesteps: number of timesteps
        :return:
        -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
        -- a_s: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
        """
        betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
        betas = np.append(betas, 1.)
        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        sqrt_alphas = np.sqrt(1. - betas)
        idx = torch.tensor(
            np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]]), dtype=torch.int32)
        a_s = np.concatenate(
          [[np.prod(sqrt_alphas[: idx[0] + 1])],
           np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
        sigmas = np.sqrt(1 - a_s ** 2)

        return sigmas, a_s    

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(x_shape[0], dtype=torch.long, device=t.device) * t
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.tensor(a, dtype=torch.float32, device=t.device)[t]
        assert list(out.shape) == [bs]
        return out.reshape([bs] + ((len(x_shape) - 1) * [1]))    

    def _q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(size=x_start.shape, device=t.device)
        assert noise.shape == x_start.shape
        x_t = self._extract(self.a_s_cum, t, x_start.shape) * x_start + \
              self._extract(self.sigmas_cum, t, x_start.shape) * noise

        return x_t

    def _q_sample_inc(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(size=x_start.shape, device=t.device)
        assert noise.shape == x_start.shape
        x_t = self._extract(self.a_s, t, x_start.shape) * x_start + \
              self._extract(self.sigmas, t, x_start.shape) * noise

        return x_t

    def _q_sample_pairs(self, x_start, t):
        """
        Generate a pair of disturbed images for training
        :param x_start: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        noise = torch.randn(size=x_start.shape, device=t.device)
        x_t = self._q_sample(x_start, t)
        x_t_plus_one = self._extract(self.a_s, t + 1, x_start.shape) * x_t + \
                       self._extract(self.sigmas, t + 1, x_start.shape) * noise

        return x_t, x_t_plus_one

    def _get_training_lss(self, x_pos, x_neg, t):
        """
        Training loss calculation
        """
        a_s = self._extract(self.a_s_prev, t + 1, x_pos.shape)
        y_pos = a_s * x_pos
        y_neg = a_s * x_neg
        pos_f = self.ebm_prior(y_pos, t, use_cls_output=False, temperature=1.)

        # do not run spectral_norm in this forward pass
        neg_f = self.ebm_prior(y_neg, t, use_cls_output=False, temperature=1.)
        loss = - (pos_f - neg_f) # + (pos_f ** 2) + (neg_f ** 2) * 1e-6

        loss_scale = 1.0 / (
            torch.tensor(
                self.sigmas, 
                dtype=torch.float64, 
                device=t.device
            )[t + 1] / self.sigmas[1])
        loss = loss_scale * loss.squeeze(1)

        return loss.mean()

    def _log_prob(self, y, t, ref_lb, tilde_x, b0, sigma, is_recovery):
        bs = y.size(0)

        return self.ebm_prior(y, t, ref_lb, use_cls_output=False, temperature=1.).squeeze(-1) / b0.view(-1) - \
               torch.sum(
                    ((y - tilde_x * is_recovery) ** 2 / (2 * sigma ** 2)).view(bs, -1), 
                    dim=-1
                )

    def _prior_lb(self, sample_z, qz_mean, qz_logvar):
        def log_gaussian(z, mean=None, log_var=None):
            import math

            assert len(z.size()) == 2
            if mean is None:
                mean = torch.zeros_like(z)
            if log_var is None:
                log_var = torch.zeros_like(z)
            
            log_p = - (z - mean) * (z - mean) / (2 * torch.exp(log_var) \
                    - 0.5 * log_var - 0.5 * math.log(math.pi * 2))
            
            return log_p.sum(dim=-1)

        def standard_normal_logprob(z):
            dim = z.size(-1)
            log_z = -0.5 * dim * np.log(2 * np.pi)
            return log_z - 0.5 * z.pow(2).sum(dim=1)

        # diffused samples
        diff_seq_one_step = []
        diff_seq_progress = []

        x = sample_z.detach()
        for t in range(0, self.max_T + 1):
            if isinstance(t, int) or len(t.shape) == 0:
                t_ = torch.ones(
                        sample_z.size(0), 
                        dtype=torch.long, 
                        device=sample_z.device
                    ) * t

            # q(z_t | z_0) for estimating partition func
            diff_seq_one_step.append(self._q_sample(sample_z.detach(), t_))

            # q(z_t | z_{t - 1})
            x = self._q_sample_inc(x, t_)
            diff_seq_progress.append(x)

        # calculate log-lkhd
        lkhd = 0.0        

        for t in range(1, self.max_T + 1):
            if isinstance(t, int) or len(t.shape) == 0:
                t_ = torch.ones(
                        sample_z.size(0), 
                        dtype=torch.long, 
                        device=sample_z.device
                    ) * t
            sigma_ = self._extract(self.sigmas, t_ - 1, sample_z.shape)
            sigma = self._extract(self.sigmas, t_, sample_z.shape)            
            sigma_cum_ = self._extract(self.sigmas_cum, t_ - 1, sample_z.shape)
            sigma_cum = self._extract(self.sigmas_cum, t_, sample_z.shape)
            a_s_ = self._extract(self.a_s_prev, t_ - 1, sample_z.shape)
            a_s = self._extract(self.a_s_prev, t_, sample_z.shape)            
            a_s_cum_ = self._extract(self.a_s_cum, t_ - 1, sample_z.shape)
            a_s_cum = self._extract(self.a_s_cum, t_, sample_z.shape)
            is_recovery = self._extract(self.is_recovery, t_, sample_z.shape)            
            
            b0 = torch.ones(sample_z.size(0), device=sample_z.device)
            m0 = torch.ones(sample_z.shape, device=sample_z.device)

            lkhd += self._log_prob(
                        a_s * diff_seq_progress[t - 1], 
                        t_ - 1, 
                        diff_seq_progress[t], 
                        b0, sigma, is_recovery) - \
                    (self._log_prob(
                        a_s * diff_seq_one_step[t - 1], 
                        t_ - 1, 
                        diff_seq_one_step[t],
                        b0, sigma, is_recovery) - \
                        log_gaussian(
                            diff_seq_one_step[t - 1],
                            mean=sample_z * a_s_cum_,
                            log_var=2 * sigma_cum_.double().log() * m0)
                        ).exp().mean().log() - \
                    log_gaussian(
                        diff_seq_progress[t],
                        mean=diff_seq_progress[t - 1] * a_s \
                                if t < self.max_T \
                                else diff_seq_progress[t - 1] * self.a_s[-1],
                        log_var=2 * sigma.double().log() * m0)
        
        lkhd += standard_normal_logprob(diff_seq_progress[-1])   

        return lkhd

    def _grad_f(self, y, t, ref_lb, tilde_x, b0, sigma, is_recovery):        
        log_p_y = self._log_prob(y, t, ref_lb, tilde_x, b0, sigma, is_recovery)
        grad_y = torch.autograd.grad(log_p_y.sum(), y)[0]
        return grad_y, log_p_y

    # === Sampling ===
    def _p_sample_langevin(self, tilde_x, t, ref_lb=None):
        """
        Langevin sampling function
        """

        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(tilde_x.size(0), dtype=torch.long, device=tilde_x.device) * t
        sigma = self._extract(self.sigmas, t + 1, tilde_x.shape)
        sigma_cum = self._extract(self.sigmas_cum, t, tilde_x.shape)
        is_recovery = self._extract(self.is_recovery, t + 1, tilde_x.shape)
        a_s = self._extract(self.a_s_prev, t + 1, tilde_x.shape)

        c_t_square = sigma_cum / self.sigmas_cum[0]
        step_size_square = c_t_square * self.e_l_step_size * sigma ** 2

        y = tilde_x.clone().detach().requires_grad_(True)
        for __ in range(self.e_l_steps):
            noise = torch.randn_like(y)

            grad_y, log_p_y = self._grad_f(
                y, t, ref_lb, tilde_x, step_size_square, sigma, is_recovery)
            y = y + 0.5 * step_size_square * grad_y \
                  + torch.sqrt(step_size_square) * noise * self.langevin_noise_scale

        x = y / a_s

        return x

    def p_sample_progressive(self, noise, ref_lb=None):
        """
        Sample a sequence of images with the sequence of noise levels
        """
        bs = noise.size(0)
        x_neg_t = noise.clone().detach()
        x_neg = torch.zeros(
            [self.max_T, bs, self.feat_dim], dtype=torch.float32, device=noise.device)
        x_neg = torch.cat([x_neg, noise[None, ...]], dim=0)

        for t in range(self.max_T - 1, -1, -1):
            x_neg_t = self._p_sample_langevin(
                x_neg_t, torch.tensor(t, dtype=torch.int32, device=noise.device),
                ref_lb if t == 0 else None)
            
            x_neg_t = x_neg_t.view(bs, self.feat_dim)
            insert_mask = (torch.tensor(
                    range(0, self.max_T + 1), dtype=torch.int32, device=noise.device
                ) == t).float()
            insert_mask = insert_mask.view([-1, *([1] * len(noise.shape))])
            x_neg = insert_mask * x_neg_t[None, ...] + (1. - insert_mask) * x_neg

        return x_neg_t, x_neg

    def forward(self, z, t):
        z_emb = self.ze(z) # size=(batch, ebm_hdim)
        t_emb = self.te(t) # size=(batch, emb_hdim)        
        f_emb = self.concat_proj(torch.cat([z_emb, t_emb], dim=-1))

        for layer in self.layers:
            f_emb = layer(f_emb) + f_emb

        return self.out_z(f_emb)        

    def sample_latent_training_pairs(self, sample_z):
        bs, n_feat = sample_z.size(0), sample_z.size(-1)
        sample_z = sample_z.view(bs, n_feat)

        t = torch.randint(
            low=0, high=self.max_T - 1, size=(bs,), device=sample_z.device)
        x_pos, x_neg = self._q_sample_pairs(sample_z, t)

        x_T_1, x_T = self._q_sample_pairs(
                        sample_z, 
                        torch.tensor(
                          self.max_T - 1,
                          device=t.device
                        )
                     )

        return x_pos, x_neg, x_T_1, x_T, t

    def get_ebm_loss(self, x_pos, x_neg, x_T_1, x_T, t):
        x_pos, x_neg = x_pos.detach(), x_neg.detach()
        
        self._set_requires_grad(
            [self.ze, self.te, self.concat_proj, self.layers, self.out_z], 
            False
        )
        x_neg = self._p_sample_langevin(x_neg, t)
        self._set_requires_grad(
            [self.ze, self.te, self.concat_proj, self.layers, self.out_z], 
            True
        )
        loss = self._get_training_lss(x_pos, x_neg, t)

        ##### loss_T
        x_pos_T, x_neg_T = x_T_1.detach(), x_T.detach()
        self._set_requires_grad(
            [self.ze, self.te, self.concat_proj, self.layers, self.out_z], 
            False
        )
        x_neg_T = self._p_sample_langevin(
                    x_neg_T, 
                    torch.tensor(
                        self.max_T - 1,
                        device=t.device
                    )
                )
        self._set_requires_grad(
            [self.ze, self.te, self.concat_proj, self.layers, self.out_z], 
            True
        )
        loss_T = self._get_training_lss(
                    x_pos_T, 
                    x_neg_T, 
                    torch.ones(
                        size=(x_pos_T.size(0),),
                        dtype=torch.long,
                        device=t.device
                    ) * (self.max_T - 1)
                )
        
        return loss + loss_T        

    def get_prior_loss(self, x_pos, x_T_1, t):
        # E_q(z|x) (f(z))
        self._set_requires_grad(
            [self.ze, self.te, self.concat_proj, self.layers, self.out_z], 
            False
        )
       
        # uniformly sampled t ~ [0, T-2]
        a_s = self._extract(self.a_s_prev, t + 1, x_pos.shape)
        y_pos = a_s * x_pos        
        prob_pos = - self.ebm_prior(y_pos, t, use_cls_output=False, temperature=1.)
        loss_scale = 1.0 / (
            torch.tensor(
                self.sigmas, 
                dtype=torch.float64, 
                device=t.device
            )[t + 1] / self.sigmas[1])
        prob_pos = (loss_scale * prob_pos.squeeze(1)).mean()

        # t = T - 1
        T_1_scale = 1.0 / (
            torch.tensor(
                self.sigmas, 
                dtype=torch.float64, 
                device=t.device
            )[self.max_T] / self.sigmas[1])

        a_s = self._extract(
                self.a_s_prev, 
                torch.tensor(self.max_T, device=t.device),
                x_T_1.shape
              )
        y_pos_T_1 = a_s * x_T_1
        prob_pos_T_1 = - self.ebm_prior(
                        y_pos_T_1,
                        torch.ones(
                          size=(y_pos_T_1.size(0),),
                          dtype=torch.long,
                          device=t.device
                        ) * (self.max_T - 1),
                        use_cls_output=False, temperature=1.
                     )
        prob_pos_T_1 = (T_1_scale * prob_pos_T_1.squeeze(1)).mean()
        
        neg_normal_ll = T_1_scale * 0.5 * x_T_1.pow(2).sum(dim=-1).mean()

        self._set_requires_grad(
            [self.ze, self.te, self.concat_proj, self.layers, self.out_z], 
            True
        )
        
        return prob_pos + prob_pos_T_1, neg_normal_ll 
    
    def ebm_prior(self, z, t, ref_lb=None, use_cls_output=False, temperature=1.):
        assert len(z.size()) == 2
        if use_cls_output:
            return self.forward(z, t)
        elif ref_lb != None:
            return self.forward(z, t)[range(z.size(0)), ref_lb].view(-1, 1)
        else:
            return temperature * (
                        self.forward(z, t) / temperature
                    ).logsumexp(dim=1, keepdim=True)

    def compute_mi(self, z, t, eps=1e-15):
        z = z.squeeze() # .detach()
        assert len(z.size()) == 2
        batch_size = z.size(0)        

        log_p_y_z = F.log_softmax(
            self.ebm_prior(z, t, use_cls_output=True), dim=-1)
        p_y_z = torch.exp(log_p_y_z)        
        
        # H(y)
        log_p_y = torch.log(torch.mean(p_y_z, dim=0) + eps)
        H_y = - torch.sum(torch.exp(log_p_y) * log_p_y)

        # H(y|z)
        H_y_z = - torch.sum(log_p_y_z * p_y_z) / batch_size

        mi = H_y - H_y_z

        return mi
