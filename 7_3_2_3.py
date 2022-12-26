import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
# %matplotlib nbagg
# %matplotlib inline
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])
from toolbox_02450 import mcnemar
import sklearn.datasets 
from PIL import Image


import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST,CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from functools import reduce

from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
import os
# The digit classes to use, these need to be in order because
# we are using one-hot representation
classes = np.arange(10)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
def one_hot(labels):
    y = torch.eye(len(classes)) 
    return y[labels]

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

# BATCH SIZE && IMAGE SIZE
batch_size = 64
image_size = 28
ds = 'CIFAR10'
if ds == 'CIFAR10':
    transform = transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ]
    )
    dset_train = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dset_test = CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # The loaders perform the actual work
    train_loader = DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)
    test_loader  = DataLoader(dset_test, batch_size=batch_size, pin_memory=cuda)
    channels = 3
# Define the train and test sets
elif ds == 'MNIST':
    dset_train = MNIST("./", train=True,  download=True, transform=transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    ), target_transform=one_hot)
    dset_test  = MNIST("./", train=False, transform=transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    ), target_transform=one_hot)
    # The loaders perform the actual work
    train_loader = DataLoader(dset_train, batch_size=batch_size,
                              sampler=stratified_sampler(dset_train.train_labels), pin_memory=cuda)
    test_loader  = DataLoader(dset_test, batch_size=batch_size, 
                              sampler=stratified_sampler(dset_test.test_labels), pin_memory=cuda)
    channels = 1
# train_loader,_ = torch.utils.data.random_split(train_loader, [897,41])
from torch import nn

latent_dim = 100

# The generator takes random `latent` noise and
# turns it into an MNIST image.
generator = nn.Sequential(
    # nn.ConvTranspose2d can be seen as the inverse operation
    # of Conv2d, where after convolution we arrive at an
    # upscaled image.
    nn.ConvTranspose2d(latent_dim, 256, kernel_size=3, stride=2),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.ConvTranspose2d(64, channels, kernel_size=2, stride=2),
    nn.Sigmoid() # Image intensities are in [0, 1]
).to(device)



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
ds_size = image_size // 2 ** 4
ks = 4
st = 2
pa = 0
# The discriminator takes an image (real or fake)
# and decides whether it is generated or not.
discriminator = nn.Sequential(
    SpectralNorm(nn.Conv2d(channels, 64, kernel_size=ks, stride=st, padding = pa)),
    nn.LeakyReLU(0.2),
    SpectralNorm(nn.Conv2d(64, 128, kernel_size=ks, stride=st, padding = pa)),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2),
    SpectralNorm(nn.Conv2d(128, 512, kernel_size=ks, stride=st, padding = pa)),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2),
    Flatten(),
    nn.Linear(512, 1),
    nn.Sigmoid()
).to(device)

loss = nn.BCELoss()
print("Using device:", device)

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))

tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []
batch_n = 0
num_epochs = 2
clip_value = 0.01

for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    k = 0
    for x, _ in train_loader:
        if x.size(0) == batch_size:
            # True data is given label 1, while fake data is given label 0
            true_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            discriminator.zero_grad()
            generator.zero_grad()
            
            # Step 1. Send real data through discriminator
            #         and backpropagate its errors.
            x_true = Variable(x).to(device)        
            output = discriminator(x_true)
            
            error_true = -torch.mean(output)
            error_true.backward()
            
            # Step 2. Generate fake data G(z), where z ~ N(0, 1)
            #         is a latent code.
            z = torch.randn(batch_size, latent_dim, 1, 1)
            z = Variable(z, requires_grad=False).to(device)
            
            x_fake = generator(z)
                
            # Step 3. Send fake data through discriminator
            #         propagate error and update D weights.
            # --------------------------------------------
            # Note: detach() is used to avoid compounding generator gradients
            output = discriminator(x_fake.detach()) 
            
            error_fake = torch.mean(output)
            error_fake.backward()
            discriminator_optim.step()
            
            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)
            
            # Step 4. Send fake data through discriminator _again_
            #         propagate the error of the generator and
            #         update G weights.
            output = discriminator(x_fake)
            
            error_generator = -torch.mean(output)
            error_generator.backward()
            generator_optim.step()
            
            batch_d_loss.append((error_true/(error_true + error_fake)).item())
            batch_g_loss.append(error_generator.item())
            batches_done = epoch * len(train_loader) + x.size(0)
            if epoch == num_epochs - 1:
                save_image(x_fake.data[1], "wimages/%d.png" % k, nrow=1, normalize=True)
                save_image(x_true.data[1], "wimages2/%d.png" % k, nrow=1, normalize=True)
            k = k + 1
    
    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    
    
    # -- Plotting --
    f, axarr = plt.subplots(1, 2, figsize=(18, 7))

    # Loss
    ax = axarr[0]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.plot(np.arange(epoch+1), discriminator_loss)
    ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
    ax.legend(['Discriminator', 'Generator'])
    
    # Latent space samples
    ax = axarr[1]
    ax.set_title('Samples from generator')
    ax.axis('off')

    rows, columns = 8, 8
    
    # Generate data
    with torch.no_grad():
        z = torch.randn(rows*columns, latent_dim, 1, 1)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)
    
    if ds == 'MNIST':
        canvas = np.zeros((28*rows, columns*28))
    
    # if ds == 'CIFAR10':
    #     for i in range(rows):
    #         for j in range(columns):
    #             idx = i % columns + rows * j
    #             r_array = x_fake.data[idx,0].cpu().numpy()
    #             g_array = x_fake.data[idx,1].cpu().numpy()
    #             b_array = x_fake.data[idx,2].cpu().numpy()
    #             channel_r = Image.fromarray(r_array).convert('L')
    #             channel_g = Image.fromarray(g_array).convert('L')
    #             channel_b = Image.fromarray(b_array).convert('L')
    #             image = Image.merge("RGB",(channel_r, channel_g, channel_b))
    #             canvas[i*28:(i+1)*28, j*28:(j+1)*28] = image.resize((28,28))
    #     ax.imshow(canvas, cmap='gray')

        # for i in range(rows):
        #     for j in range(columns):
        #         idx = i % columns + rows * j
        #         canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_fake.data[idx,0].cpu()
        # ax.imshow(canvas, cmap='gray')
    
        # plt.savefig(tmp_img)
        # plt.close(f)
        # display(Image(filename=tmp_img))
        # clear_output(wait=True)
    
        # os.remove(tmp_img)

d_mu = np.mean(batch_d_loss)
g_mu = np.mean(batch_g_loss)
d_sigma = np.std(batch_d_loss)
g_sigma = np.std(batch_g_loss)
CI_d = np.zeros(2)
CI_g = np.zeros(2)
CI_d[0] = d_mu - 1.96 * d_sigma
CI_d[1] = d_mu + 1.96 * d_sigma
CI_g[0] = g_mu - 1.96 * g_sigma
CI_g[1] = g_mu + 1.96 * g_sigma