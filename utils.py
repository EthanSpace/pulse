import torchvision
import numpy as np
import os
from torch.nn.functional import relu
import matplotlib.pyplot as plt

def save_vars(best_vars, path):
    image_dict = {x: y for (x, y) in best_vars.items() if 'image' in x}
    np_dict = {x: y for (x, y) in best_vars.items() if 'image' not in x}
    for x, y in image_dict.items():
        toPIL(y).save(os.path.join(path, f'{x}.png'))
    np.save(os.path.join(path, 'bestlatent.npy'), np_dict)


def normalize_noise(x, noise_norm, inplace=False):
    y = x-x.mean()
    y = noise_norm*y/(y.pow(2).mean(2, keepdim=True)+1e-9).sqrt()
    if(not inplace):
        return y
    else:
        x.data = y.data


def normalize_latent(x, latent_norm, inplace=False, window=0):
#     y = x-x.mean(2, keepdim=True)
    y=x
    norm = (y.pow(2).mean(2,keepdim=True)+1e-9).sqrt()
    factor = 1 + relu(norm-(1+window))/(1+window) - relu(1-window-norm)/(1-window)
    y = latent_norm*y/factor
    if(not inplace):
        return y
    else:
        x.data = y.data


def toTensor(x):
    return torchvision.transforms.ToTensor()(x).unsqueeze(0).cuda()


def toPIL(x):
    return torchvision.transforms.ToPILImage()(x)


def toGRID(x):
    if(x.is_cuda):
        x = x.cpu()
    nrow = int(np.ceil(np.sqrt(x.shape[0])))
    return torchvision.utils.make_grid(x, nrow=nrow).clamp(0, 1)


def plotTensor(x):
    x = toGRID(x)
    plt.figure(figsize=(10, 10))
    plt.imshow(x.permute(1, 2, 0).detach().numpy())
