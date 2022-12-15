import os
import glob
import pathlib
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from utils import *
#from trainer import model_trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#from lp import load_N_predict
import shutil

from ./arch/en_dec import *

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


max_epoch = 500
at_least_epoch=25
batch_size=64
device='cuda'
loss_fn=nn.L1Loss()
num_co=[]
flag_first=0


nnstorage=glob.glob("./models/*.pt")
print('nns are:',nnstorage) 

_data= gen_test_data(n,dim,ranges)
#print('_data shape:',_data)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

print('Test data:',_data.shape)



def run(): 
      flag_first=0
      train_data,validation_data= data_split(_data,proportion=0.1)
      print('Train data:',train_data.shape,'validate data:',validation_data.shape)
      copied_train_data=np.copy(train_data)
      copied_validation_data=np.copy(validation_data)
        
      fitted_train_data= data_preperation(copied_train_data,np.array(ranges))
      fitted_validation_data= data_preperation(copied_validation_data,np.array(ranges))
      train_data = SimDataset(fitted_train_data)
      validate_data = SimDataset(fitted_validation_data)

     
      
        
     
      en_model= encoder().cuda()
      dec_model= decoder().cuda()
        
      
           
      
      optimizer_en = optim.Adam(en_model.parameters(), lr=0.0001)
      optimizer_dec = optim.Adam(dec_model.parameters(), lr=0.0001)
      epoch=0; loss_train=[];loss_validate=[]     
      while True: 
            print('training epoch:',epoch)   
            if epoch > max_epoch:
                break    
            try:
                dataloader = DataLoader(train_data, batch_size, True)
                correct = 0
                for x,_ in dataloader:
                	#print(x)
                	x= x.to(device) 
                	output_en = en_model(x); output = dec_model(output_en)
                	loss = loss_fn(output, x)

                	optimizer_en.zero_grad(); optimizer_dec.zero_grad()
                	loss.backward() 
                	optimizer_en.step();optimizer_dec.step()
                	correct+= loss.item()
                train_loss=correct/len(train_data); loss_train.append(train_loss)
                #train_loss=correct; loss_train.append(train_loss)
                
                with torch.no_grad(): 
                  dataloader = DataLoader(validate_data, batch_size, True)
                  correct = 0
                  for y, _ in dataloader:
                    #print(y)
                    y = y.to(device)
                    output_en = en_model(x); output = dec_model(output_en)
                    loss = loss_fn(output, x)					
                    correct+= loss.item()
                validate_loss= correct/len(validate_data); loss_validate.append(validate_loss) 
                
                if epoch <= at_least_epoch:
                  whichmodel=epoch  
                  #torch.save(model.state_dict(), path)
                #if epoch%20==0:
                   #print('epoch is:',epoch)
                if epoch> at_least_epoch:
                 diff_loss=np.absolute(train_loss-validate_loss)
                 if flag_first==0: 
                   #torch.save(model.state_dict(), path)
                   whichmodel=epoch 
                   flag_first=1
                   last_diff_loss=diff_loss

                 elif flag_first==1:
                  if last_diff_loss>diff_loss:
                   #	torch.save(model.state_dict(), path); whichmodel=epoch ;
                   last_diff_loss=diff_loss
               
            except KeyboardInterrupt:
                break
           
            epoch+=1
                     
      fig=plt.figure(figsize=(9,6))
      plt.plot(loss_train,label='training')
      plt.plot(loss_validate,label='validate')
      plt.legend()
      plt.show()
      #		print('--> Saved model is from', whichmodel , ' epoch')
      #print('model is:',model) 
      torch.save(en_model.state_dict(), './encoder.pt')
      torch.save(dec_model.state_dict(), './decoder.pt')
      
    

  

if __name__ == "__main__":
        run()  
        
