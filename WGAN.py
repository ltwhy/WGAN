import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import Image, display, clear_output
import numpy as np
# %matplotlib nbagg
# %matplotlib inline
plt.style.use(["seaborn-deep", "seaborn-whitegrid"])
from toolbox_02450 import mcnemar
import sklearn.datasets 
from scipy import linalg
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
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
from torch.autograd import Variable
import os
from scipy.stats import entropy
from tqdm import tqdm
from fid_score2 import *
from torch import nn
from torch.nn import Parameter

# The digit classes to use, these need to be in order because
# we are using one-hot representation
classes = np.arange(10)

mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]


# def imread(filename):
#     """
#     Loads an image file into a (height, width, 3) uint8 ndarray.
#     """
#     return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

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

def inception_score(batch_size=32, resize=True, splits=8, CRI='IS'):
    # Set up dtype
    device = torch.device("cuda:0")  # you can change the index of cuda
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
    
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    # print('Computing predictions using inception v3 model')
    
    N = 64
    if CRI == 'IS':

        preds = np.zeros((N, 1000))
        preds = get_pred(x_fake)
    
        assert batch_size > 0
        assert N > batch_size
    
        # Now compute the mean KL Divergence
        # print('Computing KL Divergence')
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
            py = np.mean(part, axis=0)  # marginal probability
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]  # conditional probability
                scores.append(entropy(pyx, py))  # compute divergence
            split_scores.append(np.exp(scores))
    
        return np.max(split_scores), np.mean(split_scores)
    else:
        preds_fake = np.zeros((N,2048))
        preds_fake = get_pred(x_fake)
        preds_true = np.zeros((N, 2048))
        preds_true = get_pred(x_true)
        
        mu1 = np.mean(preds_fake,axis=0)
        mu2 = np.mean(preds_true,axis=0)
        sigma1 = np.cov(preds_fake,rowvar=False)
        sigma2 = np.cov(preds_true,rowvar=False)    
        
        eps=1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1+offset)@(sigma2+offset))
        diff = mu1 - mu2
        tr_covmean = np.trace(covmean.real)
        return(diff@diff + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
def readDir():
    dirPath = r"D:\deep_learning\02456-deep-learning-with-PyTorch-master\7_Unsupervised\images_temp"
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            f = dirPath+'/'+f
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles
            else:
                allFiles.append(f)
        return allFiles
    else:
        return 'Error,not a dir'



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
ds = 'MNIST'
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


latent_dim = 1000

# The generator takes random `latent` noise and
# turns it into an MNIST image.
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_dim, 512, kernel_size=3, stride=2),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.ConvTranspose2d(64, channels, kernel_size=2, stride=2),
    nn.Tanh()
).to(device)

generator2 = nn.Sequential(
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
    
ks = 4
st = 2
pa = 0

discriminator2 = nn.Sequential(
    nn.Conv2d(channels, 64, kernel_size=ks, stride=st, padding = pa),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, kernel_size=ks, stride=st, padding = pa),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2),
    nn.Conv2d(128, 512, kernel_size=ks, stride=st, padding = pa),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2),
    Flatten(),
    nn.Linear(512, 1),
    nn.Sigmoid()
).to(device)

# The discriminator takes an image (real or fake)
# and decides whether it is generated or not.
discriminator = nn.Sequential(
    SpectralNorm(nn.Conv2d(channels, 64, kernel_size=ks, stride=st, padding = pa)),
    nn.LayerNorm(13),
    nn.ReLU(),
    SpectralNorm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding = 1)),
    nn.LayerNorm(7),
    nn.ReLU(),
    SpectralNorm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding = pa)),
    nn.LayerNorm(3),
    nn.ReLU(),
    SpectralNorm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding = pa)),
    nn.ReLU(),
    Flatten(),
    SpectralNorm(nn.Linear(512, 1)),
).to(device)




# Optimizers
# generator_optim = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
# discriminator_optim = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
generator_optim2 = torch.optim.Adam(generator2.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim2 = torch.optim.Adam(discriminator2.parameters(), 2e-4, betas=(0.5, 0.999))
# discriminator_optim3 = torch.optim.Adam(discriminator3.parameters(), 2e-4, betas=(0.5, 0.999))
# discriminator_optim4 = torch.optim.Adam(discriminator4.parameters(), 2e-4, betas=(0.5, 0.999))

# discriminator_optim = torch.optim.SGD(discriminator.parameters(), lr=0.05)

# # use an exponentially decaying learning rate
# scheduler_d = optim.lr_scheduler.ExponentialLR(generator_optim, gamma=0.99)
# scheduler_g = optim.lr_scheduler.ExponentialLR(discriminator_optim, gamma=0.99)

tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []
batch_n = 0
num_epochs = 25
disc_iters = 3
IS_temp = 0
FID_temp = 0
d_loss_temp = 0
g_loss_temp = 0
CRI = 'FID'
clip_value = 0.01
LOSS = 'wasserstein' 
initial = True
badstart = False

if LOSS == 'BCEWLS':
    # loss = nn.BCELoss()
    loss = nn.BCEWithLogitsLoss()
    print("Using device:", device)

#FID args
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.path = ["D:\\deep_learning\\02456-deep-learning-with-PyTorch-master\\7_Unsupervised\\images"
             ,"D:\\deep_learning\\02456-deep-learning-with-PyTorch-master\\7_Unsupervised\\images2"]
checkpoint = 100
FID = np.zeros(num_epochs)
MAX = np.zeros(num_epochs)
IS = np.zeros(num_epochs)

for epoch in range(num_epochs):
        batch_d_loss, batch_g_loss = [], []
        dt = torch.zeros((2,batch_size,channels,image_size,image_size))
        k = 0
    
        for x, _ in train_loader:
            if x.size(0) == batch_size:
                # True data is given label 1, while fake data is given label 0
                true_label = torch.ones(batch_size, 1).to(device)
                fake_label = torch.zeros(batch_size, 1).to(device)
                
                if LOSS == 'BCEWLS':
                    discriminator2.zero_grad()
                    generator2.zero_grad()
                    
                    # Step 1. Send real data through discriminator
                    #         and backpropagate its errors.
                    x_true = Variable(x).to(device)        
                    output = discriminator2(x_true)
                    
                    error_true = loss(output, true_label)                    
                    error_true.backward()
                    
                    # Step 2. Generate fake data G(z), where z ~ N(0, 1)
                    #         is a latent code.
                    z = torch.randn(batch_size, latent_dim, 1, 1)
                    z = Variable(z, requires_grad=False).to(device)
                    
                    x_fake = generator2(z)
                        
                    # Step 3. Send fake data through discriminator
                    #         propagate error and update D weights.
                    # --------------------------------------------
                    # Note: detach() is used to avoid compounding generator gradients
                    output = discriminator2(x_fake.detach()) 
                    error_fake = loss(output, fake_label)                  
                    error_fake.backward()
                    discriminator_optim2.step()
                    
                    # Step 4. Send fake data through discriminator _again_
                    #         propagate the error of the generator and
                    #         update G weights.
                    output = discriminator2(x_fake)
                    error_generator = loss(output, true_label)              
                    error_generator.backward()
                    generator_optim2.step()
                    
                    batch_d_loss.append((error_true/(error_true + error_fake)).item())
                    batch_g_loss.append(error_generator.item())
                    batches_done = epoch * len(train_loader) + x.size(0)
                    
                if LOSS == 'wasserstein':
                    
                   
                    discriminator.zero_grad()              
                    generator.zero_grad()
                    
                    z = torch.randn(batch_size, latent_dim, 1, 1)
                    z= Variable(z, requires_grad=False).to(device)  

                    x_true = Variable(x).to(device) 
                    x_fake = generator(z)   

                    error_true = -discriminator(x_true).mean()+discriminator(x_fake).mean() 
                    error_true.backward()
                    

                    discriminator_optim.step()  
                    # discriminator_optim2.step()  
                    # for p in discriminator.parameters():
                    #     p.data.clamp_(-clip_value, clip_value)
                    # for p in discriminator2.parameters():
                    #     p.data.clamp_(-clip_value, clip_value)
                    # discriminator_optim3.step()  
                    # discriminator_optim4.step()  
                    # for p in discriminator3.parameters():
                    #     p.data.clamp_(-clip_value, clip_value) 
                    # for p in discriminator4.parameters():
                    #     p.data.clamp_(-clip_value, clip_value)
                        
                    # if epoch > 0:
                    #     if k//100 % 2 == 0:
                    #         discriminator_optim.step()  
                    #         discriminator_optim2.step()  
                    #         for p in discriminator.parameters():
                    #             p.data.clamp_(-clip_value, clip_value)
                    #         for p in discriminator2.parameters():
                    #             p.data.clamp_(-clip_value, clip_value)
                        
                    #     elif k//100 % 2 == 1:
                    #         discriminator_optim3.step()  
                    #         discriminator_optim4.step()  
                    #         for p in discriminator3.parameters():
                    #             p.data.clamp_(-clip_value, clip_value) 
                    #         for p in discriminator4.parameters():
                    #             p.data.clamp_(-clip_value, clip_value)
                    
                    for i in range(disc_iters):
                        discriminator.zero_grad()
                        generator.zero_grad()
                        z = torch.randn(batch_size, latent_dim, 1, 1)
                        z = Variable(z, requires_grad=False).to(device)                
                        x_fake = generator(z)
                        error_fake = - discriminator(generator(z)).mean()
                        error_fake.backward()
                        generator_optim.step()
                    
                    
                    batch_d_loss.append(error_true.item())
                    batch_g_loss.append(error_fake.item())
                    batches_done = epoch * len(train_loader) + x.size(0)
                
                # if k == 100 and batch_d_loss[k]>-0.499 or batch_d_loss[k]<-0.5:
                #     badstart = True    
                #     break
                # elif k == 100:
                #         initial = False
                if ds == 'CIFAR10' and CRI == 'IS'and k % checkpoint == 0:
                    # j = k//100
                    # MAX[j], IS[j]= inception_score(splits=8,CRI=CRI)
                    # if IS[j] > IS_temp:
                    #     IS_temp = IS[j]
                    #     for i in range(batch_size):
                    #         save_image(x_true.data[i], "images2_IS/%d.png" % i, nrow=1, normalize=True)
                    #         save_image(x_fake.data[i], "images_IS/%d.png" % i, nrow=1, normalize=True)
                    #     best = [epoch,k,MAX,IS] 
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, num_epochs, k , len(train_loader)-1, batch_d_loss[k], batch_g_loss[k])
                    ) 
                    save_image(x_fake.data[1], "images/%d_%d.png" % (epoch,k), nrow=1, normalize=True)
                    if epoch < 10:
                        save_image(x_true.data[1], "images2/%d_%d.png" % (epoch,k), nrow=1, normalize=True)
                elif ds == 'CIFAR10' and CRI == 'FID'and k % checkpoint == 0:
                    # dt[0] = x_fake
                    # dt[1] = x_true
                    # j = k//100
                    # FID[j] =  calculate_fid_given_paths(dt,
                    #                                       args.batch_size,
                    #                                       args.gpu != '',
                    #                                       args.dims)
                    # if FID[j] < FID_temp or FID_temp == 0:
                    #     FID_temp = FID[j]
                    #     for i in range(batch_size):
                    #         save_image(x_true.data[i], "images_fid2/%d.png" % i, nrow=1, normalize=True)
                    #         save_image(x_fake.data[i], "images_fid1/%d.png" % i, nrow=1, normalize=True)
                    #     best = [epoch,k,FID] 
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, num_epochs, k , len(train_loader)-1, batch_d_loss[k], batch_g_loss[k])
                    )   
                    save_image(x_fake.data[1], "images/%d_%d.png" % (epoch,k), nrow=1, normalize=True)
                    if epoch < 10:
                        save_image(x_true.data[1], "images2/%d_%d.png" % (epoch,k), nrow=1, normalize=True)
                elif k % checkpoint == 0:
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, num_epochs, k , len(train_loader)-1, batch_d_loss[k], batch_g_loss[k])
                    ) 
                    save_image(x_fake.data[1], "images/%d_%d.png" % (epoch,k), nrow=1, normalize=True)
                    if epoch < 10:
                        save_image(x_true.data[1], "images2/%d_%d.png" % (epoch,k), nrow=1, normalize=True)
                
                k = k + 1
                      
                                
        # if badstart == True:
        #     break
        # scheduler_d.step()
        # scheduler_g.step()
        discriminator_loss.append(np.mean(batch_d_loss))
        generator_loss.append(np.mean(batch_g_loss))
        
        if ds =='CIFAR10' and CRI == 'IS':
            MAX[epoch], IS[epoch]= inception_score(splits=8,CRI=CRI)
            if IS[epoch] > IS_temp:
                IS_temp = IS[epoch]
                for i in range(batch_size):
                    save_image(x_true.data[i], "images2_IS/%d.png" % i, nrow=1, normalize=True)
                    save_image(x_fake.data[i], "images_IS/%d.png" % i, nrow=1, normalize=True)
                best = [epoch,MAX[epoch],IS[epoch]] 
                print(best)
            # for i in range(3):
            #     # save_image(x_true.data[i], "images2/%d.png" % i, nrow=1, normalize=True)
            #     h = epoch * 10 + i + 1
            #     save_image(x_fake.data[i], "images/%d.png" % h, nrow=1, normalize=True)

        elif ds == 'CIFAR10' and CRI == 'FID':
            dt[0] = x_fake
            dt[1] = x_true
            FID[epoch] =  calculate_fid_given_paths(dt,
                                                  args.batch_size,
                                                  args.gpu != '',
                                                  args.dims)
            if FID[epoch] < FID_temp or FID_temp == 0:
                FID_temp = FID[epoch]
                for i in range(batch_size):
                    save_image(x_true.data[i], "images_F2/%d.png" % i, nrow=1, normalize=True)
                    save_image(x_fake.data[i], "images_F/%d.png" % i, nrow=1, normalize=True)
                best = [epoch,FID[epoch]]
                print(best)
        if ds =='CIFAR10':     
            # for i in range(batch_size):
            #     if epoch == 0:
            #         save_image(x_true.data[i], "fid_true_CIFAR/%d_%d.png" % (epoch,i), nrow=1, normalize=True)
            #     if not os.path.exists('fid_fake_CIFAR/%d/'%epoch):
            #         os.makedirs('fid_fake_CIFAR/%d/'%epoch)
            #     save_image(x_fake.data[i], "fid_fake_CIFAR/%d/%d.png" % (epoch,i), nrow=1, normalize=True)
            
            
            fig = plt.figure(figsize=(8, 8),facecolor='black')
            gs = gridspec.GridSpec(8, 8)
            gs.update(wspace=0.05, hspace=0.05)
            x_fake = x_fake.cpu().detach().numpy()
            x_true = x_true.cpu().detach().numpy()
            for i, sample in enumerate(x_true):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)
    
            if not os.path.exists('result_CIFAR/'):
                os.makedirs('result_CIFAR/')
    
            plt.savefig('result_CIFAR/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            
            f, axarr = plt.subplots(1, 1, figsize=(7, 7))
        
            # Loss
            ax = axarr
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
        
            ax.plot(np.arange(epoch+1), discriminator_loss)
            ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
            ax.legend(['Discriminator', 'Generator'])
            plt.savefig('result_CIFAR_loss/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(f)
            
        if ds == 'MNIST':
            
            for i in range(batch_size):
                if epoch == 0:
                    save_image(x_true.data[i], "fid_true_MNIST/%d_%d.png" % (epoch,i), nrow=1, normalize=True)
                if not os.path.exists('fid_fake_MNIST/%d/'%epoch):
                    os.makedirs('fid_fake_MNIST/%d/'%epoch)
                save_image(x_fake.data[i], "fid_fake_MNIST/%d/%d.png" % (epoch,i), nrow=1, normalize=True)
            
            # -- Plotting --
            f, axarr = plt.subplots(1, 2, figsize=(18, 7))
        
            # Loss
            ax = axarr[0]
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
        
            ax.plot(np.arange(epoch+1), discriminator_loss)
            ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
            ax.legend(['Discriminator', 'Generator'])
            # ax.legend(['Discriminator'])
            # Latent space samples
            ax = axarr[1]
            # ax.set_title('Samples from generator')
            ax.axis('off')
        
            rows, columns = 8, 8
            
            # Generate data
            with torch.no_grad():
                z = torch.randn(rows*columns, latent_dim, 1, 1)
                z = Variable(z, requires_grad=False).to(device)
                x_fake = generator2(z)
            
            canvas = np.zeros((image_size*rows, columns*image_size))
        
            for i in range(rows):
                for j in range(columns):
                    idx = i % columns + rows * j
                    canvas[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size] = x_fake.data[idx,0].cpu()
            ax.imshow(canvas, cmap='gray')
            plt.savefig('result_MNIST/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(f)
            # display(Image(filename=str(epoch)))
            clear_output(wait=True)
        
            # os.remove(str(epoch))

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


# inception_score(splits=8,CRI = "FID")
