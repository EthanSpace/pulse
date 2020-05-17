import torch
from loss import LossBuilder
from model import StyleGan
from utils import *
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from functools import partial
import time
import numpy as np
from pathlib import Path


class SR(torch.nn.Module):
    def __init__(self, loss_str, eps,
                 image_size=32,
                 use_mask=True,
                 noise_type='fixed',  # zero, fixed, trainable, basis
                 trainable_noise=7,  # number of noise layers to gradient descent on
                 basis_size=None,
                 monte_carlo=False,
                 noise_norm=1,
                 latent_norm=1,
                 latent_window=0,
                 tile_latent=False,
                 spherical_noise=True,
                 layers_vgg=None,
                 bad_noise_layers=[17],
                 opt_name='adam',
                 learning_rate=0.2,
                 steps=1500,
                 lr_schedule='linear1cycledrop',  # fixed, linear1cycledrop, linear1cycle
                 ):
        super(SR, self).__init__()
        self.loss_str = loss_str
        self.eps = eps
        self.image_size = image_size
        self.use_mask = use_mask
        self.noise_type = noise_type
        self.trainable_noise = trainable_noise
        self.basis_size = basis_size
        self.monte_carlo = monte_carlo
        self.noise_norm = noise_norm
        self.latent_norm = latent_norm
        self.tile_latent = tile_latent
        self.spherical_noise = spherical_noise
        self.layers_vgg = layers_vgg
        self.bad_noise_layers = bad_noise_layers
        self.opt_name = opt_name
        self.learning_rate = learning_rate
        self.steps = steps
        self.lr_schedule = lr_schedule
        self.latent_window = latent_window

        self.sr = StyleGan(use_mapping=False).cuda()
        for param in self.sr.parameters():
            param.requires_grad = False

    def forward(self, ref_im):
        batch_size = ref_im.shape[0]

        # Generate latent tensor
        if(self.tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        # Generate list of noise tensors
        noise = []
        noise_vars = []  # stores the noise tensors that we want to gradient descent on
        noise_vars_idx = []  # stores the indices of the tensors in var_latent

        if(self.noise_type == 'basis'):
            noise_basis = [torch.randn((1, 1, 2**(i//2+2), 2**(i//2+2), self.basis_size), dtype=torch.float, device='cuda') for i in range(18)]
            for x in noise_basis:
                x.requires_grad=False

        for i in range(18):
            # dimension of the ith noise tensor
            res = (1, 1, 2**(i//2+2), 2**(i//2+2))
            if(self.noise_type == 'zero' or i in self.bad_noise_layers):
                new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif(self.noise_type == 'fixed'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif(self.noise_type == 'trainable'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                if(i < self.trainable_noise):
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                    noise_vars_idx.append(i)
                else:
                    new_noise.requires_grad = False
            elif(self.noise_type == 'basis'):
                linear_comb = torch.tensor(np.random.normal(
                    0, 1, self.basis_size), dtype=torch.float, device='cuda')
                linear_comb.requires_grad = True
                new_noise = normalize_noise(noise_basis[i].matmul(
                    linear_comb), self.noise_norm, inplace=False)
                noise_vars.append(linear_comb)
                noise_vars_idx.append(i)
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        var_list = [latent]+noise_vars

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[self.opt_name]
        opt = opt_func(var_list, lr=self.learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/self.steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*self.steps)-1/2)*2)+1)/10 if x < 0.9*self.steps else 1/10 + (x-0.9*self.steps)/(0.1*self.steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[self.lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule_func)

        min_loss = np.inf
        best_summary = ""
        start_t = time.time()

        normalize_latent(latent, self.latent_norm, inplace=True, window=self.latent_window)

        if(self.spherical_noise and self.noise_type != "basis"):
            [normalize_noise(x, self.noise_norm, inplace=True) for x in noise]

        elapsed_t = np.inf

        loss_builder = LossBuilder(
            ref_im, self.image_size, self.loss_str, self.eps, self.layers_vgg, self.use_mask).cuda()
        for param in loss_builder.parameters():
            param.requires_grad = False
        toPIL(ref_im[0].cpu().detach().clamp(0, 1)).resize((1024, 1024), Image.NEAREST).save(
            Path('runsLR') / "TRUE.jpg")
        for j in range(self.steps):
            opt.zero_grad()
            if(self.noise_type == 'basis'):
                noise_in = []
                for i in range(18):
                    if i not in self.bad_noise_layers:
                        noise_in.append(normalize_noise(noise_basis[i].matmul(
                            noise_vars[i]), self.noise_norm, inplace=False))
                    else:
                        noise_in.append(noise[i])
            else:
                noise_in = noise

            latent_in = normalize_latent(
                latent, self.latent_norm, inplace=False, window=self.latent_window)
            gen_im = self.sr(latent_in, noise_in)

            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            #tmp code - save SR and LR at every step
            # toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
            #     Path('runs') / f"{j:02}.jpg")
            # toPIL(loss_builder.D(gen_im)[0].cpu().detach().clamp(0, 1)).resize((1024,1024),Image.NEAREST).save(
            #     Path('runsLR') / f"{j:02}.jpg")

            loss_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y:.3f}' for x, y in loss_dict.items()])
            if(loss < min_loss):
                min_loss = loss
                best_summary = loss_summary
                best_im = gen_im.clone()

            loss.backward()
            opt.step()
            scheduler.step()
            normalize_latent(latent, self.latent_norm, inplace=True, window=self.latent_window)

            if(self.spherical_noise and self.noise_type != 'basis'):
                [normalize_noise(x, self.noise_norm, inplace=True)
                 for x in noise]
            if(self.monte_carlo):
                for i in range(18):
                    if(i not in var_idx):
                        res = (1, 1, 2**(i//2+2), 2**(i//2+2))
                        if(self.noise_type == 'zero' or i in bad_noise_layers):
                            new_noise = torch.zeros(
                                res, dtype=torch.float, device='cuda')
                        elif(self.noise_type == 'fixed'):
                            new_noise = torch.randn(
                                res, dtype=torch.float, device='cuda')
                        elif(self.noise_type == 'trainable'):
                            new_noise = torch.randn(
                                res, dtype=torch.float, device='cuda')
                        else:
                            raise Exception(
                                f"cannot use monte-carlo with {noise_type}")
                        noise[i].data = new_noise.data

        total_t = time.time()-start_t
        current_info = f' | {j+1}/{self.steps} | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        print(best_summary+current_info)
        return best_im
