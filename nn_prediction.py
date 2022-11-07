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

from en_dec import *

a_l=100; a_h=500;
b_l=70; b_h=300;
c_l=70; c_h=300;
#r_l=100; r_h=750;
#v_l=0.5;v_h=5;
n=10000
dim=3
ranges=[a_l,a_h,b_l,b_h,c_l,c_h]             



input_size=dim                            
output_size=dim                          



max_epoch = 100
at_least_epoch=25
batch_size = 16
device='cuda'
loss_fn=nn.L1Loss()
num_co=[]
flag_first=0

_data= gen_test_data(n,dim,ranges)
#print('_data shape:',_data)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

print('Test data:',_data.shape)



nnstorage=glob.glob("./models/*.pt")
print('nns are:',nnstorage) 

test_data= _data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

print('Test data:',test_data.shape)




def run_encoder(): 
    first_run=1
    result_file_name= './data/prediction_result_encoder.csv'
    copied_test_data=np.copy(test_data)
    fitted_test_data= data_preperation(copied_test_data,np.array(ranges))
        
    testing_data = SimDataset(fitted_test_data)
    fitted_text_X= fitted_test_data; 
    print('fitted X:',fitted_text_X.shape)
        
    
      
    neuralNet= encoder()
        
    try: 
        neuralNet.load_state_dict(torch.load('encoder.pt'))       
        print("Loaded earlier trained model successfully")
    except: 
        print('Couldnot find weights of NN')  
           
    with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              
    output=output.cpu().detach().numpy()
    #estimate_accuracy(test_data,output)
      
    multi_runresults=np.concatenate((test_data,np.array(output)),axis=1)
    np.savetxt(result_file_name,multi_runresults,  delimiter=',')  


def run_decoder(): 
    first_run=1
    result_file_name= './data/prediction_result_decoder.csv'
    __data_= np.loadtxt('./data/prediction_result_encoder.csv',delimiter=',')
    test_data= __data_[:,3:5]
    test_data= test_data
    print('test data shape:', test_data.shape)
    copied_test_data=np.copy(test_data)
    #fitted_test_data= data_preperation(copied_test_data,np.array(ranges))
    fitted_test_data=test_data   
    testing_data = SimDataset(fitted_test_data)
    fitted_text_X= fitted_test_data; 
    print('fitted X:',fitted_text_X.shape)
          
    neuralNet= decoder()
        
    try: 
        neuralNet.load_state_dict(torch.load('decoder.pt'))       
        print("Loaded earlier trained model successfully")
    except: 
        print('Couldnot find weights of NN')  
           
    with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              
    output=output.cpu().detach().numpy()
    #estimate_accuracy(test_data,output)
    output=rescale_data(output,ranges)  
    multi_runresults=np.concatenate((test_data,np.array(output)),axis=1)
    np.savetxt(result_file_name,multi_runresults,  delimiter=',')  


if __name__ == "__main__":
        run_encoder()
        run_decoder()  
        			



