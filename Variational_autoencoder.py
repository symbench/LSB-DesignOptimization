__author__ = 'Sherlock Holmes'

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shutil

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

a_l=100; a_h=500;
b_l=70; b_h=300;
c_l=70; c_h=300;
#r_l=100; r_h=750;
#v_l=0.5;v_h=5;
n=20000
dim=3
ranges=[a_l,a_h,b_l,b_h,c_l,c_h]             

input_size=dim                            
output_size=dim                          



device='cpu'
num_epochs = 500
batch_size = 128
learning_rate = 1e-3
_data= gen_test_data(n,dim,ranges)
print('_data shape:',_data.shape)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

loss_train=[]; loss_validate=[]


flag_first=0
train_data,validation_data= data_split(_data,proportion=0.1)
print('Train data:',train_data.shape,'validate data:',validation_data.shape)
copied_train_data=np.copy(train_data)
copied_validation_data=np.copy(validation_data)
        
fitted_train_data= data_preperation(copied_train_data,np.array(ranges))
fitted_validation_data= data_preperation(copied_validation_data,np.array(ranges))
train_data = SimDataset(fitted_train_data)
validate_data = SimDataset(fitted_validation_data)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(3, 400)
        self.fc21 = nn.Linear(400, 2)
        self.fc22 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device=='cuda':
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if device=='cuda':
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    print('epoch is:',epoch)
    model.train()
    train_loss = 0
    if epoch > num_epochs:
           break    
    try:
                dataloader = DataLoader(train_data, batch_size, True)
                correct = 0
                for x,_ in dataloader:
                	#print(x)
                	x= x.to(device) 
                	recon_batch, mu, logvar = model(x)
                	loss = loss_function(recon_batch,x, mu, logvar)

                	optimizer.zero_grad(); 
                	loss.backward() 
                	optimizer.step();
                	correct+= loss.item()
                train_loss=correct/len(train_data); loss_train.append(train_loss)
                #train_loss=correct; loss_train.append(train_loss)
                
                
               
    except KeyboardInterrupt:
                break
           
                     
fig=plt.figure(figsize=(9,6))
plt.plot(loss_train,label='training')
plt.legend()
plt.show()

torch.save(model.state_dict(), './vae.pt')


# write sample from normal dist 
# checker function to ensure the generated data is within the bound ( take it from earlier RRCF work) 