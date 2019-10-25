import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from onet import net
from vgg import vgg 
from torch.autograd import Variable
import numpy as np
import argparse
import csv
import pickle
import os
import matplotlib.pylab as plt
from resnet import * 
from torch.utils.data.sampler import SubsetRandomSampler
import h5py
import bisect 

'''
Classes:
0) 4-QAM,
1) 8-QAM
'''

def get_loss_optimizer(net,learning_rate=0.001):
        #Loss
        loss = torch.nn.CrossEntropyLoss()
        #Optimizer
        optimizer  = optim.Adam(net.parameters(),lr = learning_rate)
        return(loss, optimizer)

def test_net(test_loader = None, path= 'model.pt', batch_size= 128, fname=None):  

    n_batches = len(test_loader)

    model = torch.load(path)
    net = model['model']
    net.load_state_dict(model['state_dict'])
    for par in net.parameters():
        par.requires_grad = False
    net.eval()
    net = net.float()
    net = net.to('cuda') 
    #writing results to spreadsheet
    if fname is None:
        fname = 'test_pred.csv'
    f_out = open(fname,"w")
    wrt = csv.writer(f_out)

    #testing metrics
    corr_cnt = 0
    total_iter = 0
    
    for data in test_loader:
        [inputs,labels,snr] = data
        inputs,labels = Variable(inputs).to('cuda'), Variable(labels)
        pred = net(inputs.float())
            
        snr = snr.numpy()
        pred = np.argmax(pred.cpu(),axis =1).numpy()
        labels = np.argmax(labels.numpy(),axis=1)
        for s,p,l in zip(snr,pred,labels):
            #wrt.writerow([s,p,l]) 
            if(p == l):
                corr_cnt += 1
            
            total_iter +=1         
            
    print("Test done, accr = :" + str(corr_cnt/total_iter))
    f_out.close()


def train_net(train_loader=None, net=None, batch_size=128, n_epochs=5 ,learning_rate = 0.001,saved_model=None,fname=None):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    

    #Get training and test data 
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    loss, optimizer = get_loss_optimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    

    f_out = open(fname,"w")
    wrt = csv.writer(f_out)
    
    total_train_loss = 0
    
    scheduler = StepLR(optimizer, step_size=250, gamma=0.1)
    net = net.float()
    net = net.to('cuda')
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()

        wrt.writerow([epoch,total_train_loss])

        total_train_loss = 0
         
        
        if (((epoch+1) % 250) == 0):
            checkpoint = {'model':net,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()}
            file_name = 'checkpoint.pt'
            torch.save(checkpoint, file_name)    
        
        i = 0
        
        for data in train_loader:
            
            [inputs,labels,snr] = data
            #print(inputs.shape)
            #Wrap them in a Variable object
            
            inputs,labels,snr = Variable(inputs).to('cuda'), Variable(labels).to('cuda'), Variable(snr).to('cuda')
            
            #inputs,labels,snr = Variable(inputs), Variable(labels), Variable(snr)
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            #Forward pass, backward pass, optimize
            outputs = net(inputs.float())
            labels = labels.squeeze_().cpu()
            loss_size = loss(outputs.cpu(), np.argmax(labels,axis=1))
            #loss_size = loss(outputs, np.argmax(labels,axis=1))
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            #Print loss from every 10% (then resets to 0) of a batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.4f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), total_train_loss/print_every , time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
                
            i += 1
        scheduler.step()
    
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    final = {'model':net,
          'state_dict': net.state_dict(),
          'optimizer' : optimizer.state_dict()}
    
    torch.save(final, saved_model)
    f_out.close()
    

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true",help="Train the model")
        parser.add_argument("--steps", type=int, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, help="Batch Size")
        parser.add_argument("--file_train", type=str, help="Path to input data")
        parser.add_argument("--learning_rate", type=float, help="Learning Rate")
        parser.add_argument("--model_path", type=str, help="Saved model")
        parser.add_argument("--classes", type=int, help= "Number of output classes")
        parser.add_argument("--results", type=str, help= "Store results")
        parser.add_argument("--filts", type=str, help= "Number of filts")
        parser.add_argument("--inter", type=str, help= "Intervals")
        
        args = parser.parse_args()
        

        if args.filts is not None:
            filts = [int(x) for x in args.filts.split(",")]

        
        
        args = parser.parse_args()
        
        file_name = args.file_train 
        data = h5py.File(file_name,'r')
        np.random.seed(2019)
        
        print(data['Y'].value)

        n_ex = data['X'].shape[0]
        
        #print(n_ex)
        n_train = int(n_ex*(7/8))

        idx = np.random.permutation(n_ex)

        train_idx = idx[:n_train].tolist()
        test_idx = idx[n_train:n_ex].tolist()

        train_samp = SubsetRandomSampler(train_idx)
        test_samp = SubsetRandomSampler(test_idx)
        #print(data['Y'].shape)
        data = torch.utils.data.TensorDataset(torch.from_numpy(data['X'].value),torch.from_numpy(data['Y'].value), torch.from_numpy(data['Z'].value))
        
        if(args.train):
            
            #Training data
            '''
            model = torch.load(args.model_path)
            nn = model['model']
            nn.load_state_dict(model['state_dict'])
            '''
            #nn = ResNet34(args.classes)
            #nn = vgg(filts)
            nn = net(args.classes)
            train_loader = torch.utils.data.DataLoader(data,batch_size =args.batch_size, sampler=train_samp)    
            train_net(train_loader,nn,args.batch_size,args.steps, args.learning_rate,args.model_path,args.results)
                    
        else:
            #Testing Data
            test_loader = torch.utils.data.DataLoader(data,batch_size=args.batch_size,sampler=test_samp)
            path = args.model_path
            test_net(test_loader,path,args.batch_size,args.results)
            
        
        

        
    
