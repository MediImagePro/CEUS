import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from networks import *
from util import *

# Constructor for BaseModel. Initializes model properties and sets up GPU if available.
class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    # Static method to modify command line options for the model.
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    # Abstract method to unpack and preprocess input data.
    def set_input(self, input):
        pass

    @abstractmethod
    # Abstract method to implement the forward pass of the network.
    def forward(self):
        pass

    @abstractmethod
    # Abstract method to calculate losses, gradients, and update network weights.
    def optimize_parameters(self):
        pass

    # Prepares the model for training or testing, sets up schedulers, and loads networks.
    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain:
            load_suffix = '%s' % opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(True)

    # Sets the model to evaluation mode, typically used during testing.
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # Defines the forward function used during testing without gradient computation.
    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    # Placeholder method for computing additional output images for visualization.
    def compute_visuals(self):
        pass

    # Returns the paths of images currently being processed.
    def get_image_paths(self):
        return self.image_paths

    # Updates learning rates for all networks, usually called at end of an epoch.
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # Retrieves current visualization images for display or saving.
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # Returns current training losses/errors.
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # Saves all network models to the disk.
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # Fixes InstanceNorm checkpoints incompatibility issues in older PyTorch versions.
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # Loads all network models from the disk.
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # Prints total number of parameters and network architecture if verbose.
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # Sets `requires_grad` to `False` for all networks to avoid unnecessary computations.
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


# Specific model class for a GAN model, presumably for image enhancement tasks.
class EnhanceGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    # Constructor for BaseModel. Initializes model properties and sets up GPU if available.
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['GA_GAN', 'GT_GAN', 'G_L1', 'D_A', 'D_T']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_AT', 'real_BT', 'fake_BT']
        if self.isTrain:
            self.model_names = ['G', 'DA', 'DT']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = UnetGenerator(1, 3, 8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True)
        self.netG = init_net(self.netG, gpu_ids=opt.gpu_ids)

        if self.isTrain:
            self.netDA = NLayerDiscriminator(1 + 3, 64, 3, norm_layer=nn.BatchNorm2d)
            self.netDA = init_net(self.netDA, gpu_ids=opt.gpu_ids)

            if self.opt.EnhanceT:
                self.netDT = NLayerDiscriminator(1 + 3, 64, 3, norm_layer=nn.BatchNorm2d)
                self.netDT = init_net(self.netDT, gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = GANLoss('vanilla').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_DA = torch.optim.Adam(self.netDA.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            if self.opt.EnhanceT:
                self.optimizer_DT = torch.optim.Adam(self.netDT.parameters(), lr=opt.lr, betas=(0.5, 0.999))
                self.optimizers.append(self.optimizer_DT)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DA)
            # Default backup dic for fasten training process
            self.real_backup = dict()
            self.fake_backup = dict()

    # Abstract method to unpack and preprocess input data.
    def set_input(self, input):
        AtoB = True
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.mask = input['M']
        self.bbox = input['bbox'][0]

    # Abstract method to implement the forward pass of the network.
    def forward(self):
        self.fake_B = self.netG(self.real_A)
        # crop tumor area
        self.real_AT = self.real_A[:, :, self.bbox[0]: self.bbox[2], self.bbox[1]: self.bbox[3]]
        self.real_BT = self.real_B[:, :, self.bbox[0]: self.bbox[2], self.bbox[1]: self.bbox[3]]
        self.fake_BT = self.fake_B[:, :, self.bbox[0]: self.bbox[2], self.bbox[1]: self.bbox[3]]

    # Helper method to compute the basic backward pass for the discriminator.
    def backward_D_basic(self, netD, realA, realB, fakeB):
        real_AB = torch.cat((realA, realB), 1)
        pred_real = netD.forward(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        fake_AB = torch.cat((realA, fakeB), 1)
        pred_fake = netD.forward(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_DA(self):
        self.loss_D_A = self.backward_D_basic(self.netDA, self.real_A, self.real_B, self.fake_B)
        self.loss_D_A.backward()

    def backward_DT(self):
        self.loss_D_T = self.backward_D_basic(self.netDT, self.real_AT, self.real_BT, self.fake_BT)
        self.loss_D_T.backward()

    # Calculates the GAN and L1 loss for the generator.
    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_A = self.netDA(fake_AB)
        self.loss_GA_GAN = self.criterionGAN(pred_fake_A, True)
        # Tumor enhance
        if self.opt.EnhanceT:
            fake_ABT = torch.cat((self.real_AT, self.fake_BT), 1)
            pred_fake_AT = self.netDT(fake_ABT)
            self.loss_GT_GAN = self.criterionGAN(pred_fake_AT, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100
        self.loss_G = self.loss_GA_GAN + self.loss_G_L1 + self.loss_GT_GAN

        self.loss_G.backward()

    # Abstract method to calculate losses, gradients, and update network weights.
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update DA
        self.set_requires_grad(self.netDA, True)  # enable backprop for D
        self.optimizer_DA.zero_grad()     # set D's gradients to zero
        self.backward_DA()                # calculate gradients for D
        self.optimizer_DA.step()          # update D's weights
        if self.opt.EnhanceT:
            # update DT
            self.set_requires_grad(self.netDT, True)  # enable backprop for D
            self.optimizer_DT.zero_grad()     # set D's gradients to zero
            self.backward_DT()                # calculate gradients for D
            self.optimizer_DT.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netDA, False)
        self.set_requires_grad(self.netDT, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights





