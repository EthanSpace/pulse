import torch
import torchvision.models.vgg as models
from bicubic import BicubicDownSample
from segment import BiSeNet


class VGG16(torch.nn.Module):
    def __init__(self, layers=(3, 8, 15, 22), requires_grad=False):
        super(VGG16, self).__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.feature_layers = layers
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, im_size, loss_str, eps, layers_vgg=None, use_mask=False):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2]==ref_im.shape[3]
        factor=1024//im_size
        factor_ref=ref_im.shape[2]//im_size
        assert im_size*factor_ref==ref_im.shape[2]
        assert im_size*factor==1024
        self.D = BicubicDownSample(factor=factor)
        
        self.ref_im_lr = ref_im

        self.parsed_loss = [loss_term.split('*')
                            for loss_term in loss_str.split('+')]
        self.eps = eps

        if('VGG' in loss_str):
            self.vgg16 = VGG16(layers=layers_vgg).cuda()

        vgg16mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float, device='cuda').view(-1, 1, 1)
        vgg16std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float, device='cuda').view(-1, 1, 1)
        self.register_buffer('vgg16mean', vgg16mean)
        self.register_buffer('vgg16std', vgg16std)

        if(use_mask):
            with torch.no_grad():
                masknet = BiSeNet(19).cuda().eval()
                masknet.load_state_dict(torch.load('segment.pt'))
                preprocessed_HR_image = self.preprocess(ref_im)
                labels = masknet(preprocessed_HR_image).argmax(1)
                HRmask = torch.zeros(
                    (ref_im.shape[0], ref_im.shape[2], ref_im.shape[3]), dtype=torch.float, device='cuda')
                # bad_features = [0,14,16] # Use to exclude shoulders and neck
                bad_features = [0]  # Use to exclude background
                for i in range(19):
                    if i not in bad_features:
                        HRmask[labels == i] = 1

                self.mask = self.Dref(HRmask.unsqueeze(1).expand(-1, 3, -1, -1))
        else:
            self.mask = torch.ones(self.ref_im_lr.shape,
                                   dtype=torch.float, device='cuda')

    def preprocess(self, x):
        return (x-self.vgg16mean)/self.vgg16std

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im_lr, **kwargs):
        return 100*(self.mask*(gen_im_lr - ref_im_lr)).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum()

    def _loss_l1(self, gen_im_lr, ref_im_lr, **kwargs):
        return 10*(self.mask*(gen_im_lr - ref_im_lr)).abs().mean((1, 2, 3)).clamp(min=self.eps).sum()

    def _loss_vgg(self, gen_im_lr, ref_im_lr, **kwargs):
        preprocessed_generated_image = self.preprocess(gen_im_lr)
        preprocessed_reference_image = self.preprocess(ref_im_lr)
        generated_image_features = self.flatcat(
            self.vgg16(preprocessed_generated_image))
        reference_image_features = self.flatcat(
            self.vgg16(preprocessed_reference_image))
        return (generated_image_features-reference_image_features).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum()

    # Only use if you do not normalize the latent, otherwise will return 1
    def _loss_reg(self, latent, **kwargs):
        return latent.pow(2).mean()

    # Uses euclidian distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_cross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            dlatent = latent.expand(-1, 18, -1)
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            D = (X-Y).pow(2).sum()
            return D.sum()/2500.0

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D

    def forward(self, latent, gen_im):
        var_dict = {'latent': latent,
                    'gen_im_lr': self.D(gen_im),
                    'ref_im_lr': self.ref_im_lr,
                    }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'VGG': self._loss_vgg,
            'REG': self._loss_reg,
            'CROSS': self._loss_cross,
            'GEOCROSS': self._loss_geocross,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        return loss, losses
