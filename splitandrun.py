from SR import SR
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
from utils import *
from PIL import Image
from PIL import ImageFilter

class Images(Dataset):
    def __init__(self, root_dir, num_images):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.num_images = num_images

    def __len__(self):
        return self.num_images*len(self.image_list)

    def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        img_path = self.image_list[idx//self.num_images]
#         image = torchvision.transforms.ToTensor()(Image.open(img_path).filter(ImageFilter.GaussianBlur(radius=1)))
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        # noise = torch.randn(image.shape)
        # nimage = image+noise*0.1
        # toPIL(nimage.cpu().detach().clamp(0,1)).save(Path('./aligned_noisy') / img_path.name)
        return image,img_path.stem+f"{idx % self.num_images}.png"


# dataset = Images('aligned_realpics_32')
dataset = Images('./justone', 1)
out_path = Path('runs')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
model = SR(loss_str='3*L2+0.05*GEOCROSS',
           eps=1e-4,
           image_size=16,
           use_mask=False,
           noise_type='trainable',
           trainable_noise=5,  # number of noise layers to gradient descent on
           monte_carlo=False,
           noise_norm=0.5,
           latent_norm=0.7,
           latent_window=0,
           tile_latent=False,
           spherical_noise=True,
           bad_noise_layers=[17],
           opt_name='adam',
           learning_rate=0.4,
           steps=100,
           lr_schedule='linear1cycledrop'
           )
model = torch.nn.DataParallel(model)
for ref_im, ref_im_name in dataloader:
    out_im = model(ref_im)
    for i in range(len(out_im)):
        # toPIL(ref_im[i].cpu().detach().clamp(0, 1)).save(
            # out_path / ref_im_name[i])
        toPIL(out_im[i].cpu().detach().clamp(0, 1)).save(
            out_path / ref_im_name[i])
