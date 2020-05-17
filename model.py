import torch
import numpy as np
from stylegan import G_synthesis,G_mapping


class StyleGan(torch.nn.Module):
    def __init__(self,use_mapping=False):
        super(StyleGan, self).__init__()
        self.use_mapping = use_mapping
        if(use_mapping):
            self.MAP = G_mapping().cuda()
            self.MAP.load_state_dict(torch.load('mapping.pt'))
        else:
            gauss_mean, gauss_scale = torch.tensor(
                np.load("gaussian_fit.npy"), dtype=torch.float, device='cuda')
            self.register_buffer('gauss_mean', gauss_mean)
            self.register_buffer('gauss_scale', gauss_scale)

        self.Gs = G_synthesis().cuda()
        self.Gs.load_state_dict(torch.load('synthesis.pt'))

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, latent, noise):
        if(latent.shape[1] == 1):
            dlatent = latent.expand(-1, 18, -1)
        elif(latent.shape[1] == 18):
            dlatent = latent
        else:
            raise Exception(
                "Latent has to have dimension (?,18,512) or (?,1,512)")

        if(self.use_mapping):
            corrected_latent = self.MAP(dlatent)
        else:
            corrected_latent = dlatent*self.gauss_scale+self.gauss_mean
            corrected_latent = self.lrelu(corrected_latent)
        out = self.Gs(corrected_latent, noise)
        out = (out+1)/2

        return out
