import torch
import torch.optim as optim
from Net import Net
import time
#import Variable
import h5py
from vgg import vgg 
from torch.autograd import Variable
import numpy as np
import argparse
import csv


LR=0.001
BATCH_SIZE=128

'''
Classes:
0) OOK,
1) 4ASK,
2) 8ASK,
3) BPSK,
4) QPSK,
5) 8PSK,
6) 16PSK,
7) 32PSK,
8) 16APSK,
9) 32APSK,
10) 64APSK,
11) 128APSK,
12) 16QAM,
13) 32QAM,
14) 64QAM,
15) 128QAM,
16) 256QAM,
17) AM-SSB-WC,
18) AM-SSB-SC,
19) AM-DSB-WC,
20) AM-DSB-SC,
21) FM,
22) GMSK,
23) OQPSK
'''



def get_loss_optimizer(net,learning_rate=0.001):
    #We change optimizer here - Euclidean vs. Hyperbolic vs. Product Space
        #Loss
        loss = torch.nn.CrossEntropyLoss()
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr= learning_rate)

        return(loss, optimizer)

def test_net(test_set, path, batch_size= BATCH_SIZE):
    #run test loop here
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, num_workers=2)
    n_batches = len(test_loader)
    net = vgg(filts)
    net.load_state_dict(torch.load(path))
    net.eval()
    #writing results to spreadsheet
    if fname is None:
        fname = "test_pred.csv"
    f_out = open(fname,"w")
    
    #loading labels
    label_load = torch.utils.data.DataLoader(label, batch_size=batch_size, num_workers=2)
    #loading snrs
    snr_load = torch.utils.data.DataLoader(snr,batch_size=batch_size, num_workers=2)
    iter_lab = iter(label_load)
    iter_snr = iter(snr_load)
    
    #testing metrics
    corr_cnt = 0
    total_iter = 0
    for data in iter(test_loader):
        label  = iter_lab.next()
        snr = iter_snr.next()
        data, labels,snr = Variable(data), Variable(labels), Variable(snr)
        pred = net(data)
        for s,p,l in zip(snr,pred,label):
            if(p == l):
                corr_cnt += 1
            wrt.writerow([s,p,l])
            total_iter +=1 
    print("Test done, accr = :" + str(corr_cnt/total_iter))
    f_out.close()

def train_net(train_set=None,label=None, snr= None,net=None, batch_size=BATCH_SIZE, n_epochs=5 ,learning_rate = LR):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training and test data 
    #train_loader,test_loader,val_loader = get_data(batch_size,x,y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, num_workers=2)
    n_batches = len(train_loader)
    
    #val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, num_workers=2)
    
    #loading labels
    label_load = torch.utils.data.DataLoader(label, batch_size=batch_size, num_workers=2)
    #loading snrs
    snr_load = torch.utils.data.DataLoader(snr,batch_size=batch_size, num_workers=2)



    #Create our loss and optimizer functions
    loss, optimizer = get_loss_optimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
          
        iter_lab = iter(label_load)  
        iter_snr = iter(snr_load)

        for i,inputs in enumerate(train_loader,0):
            
            #Get inputs
            #print(inputs.shape)
            labels = iter_lab.next()
            snr = iter_snr.next()
            #Wrap them in a Variable object
            
            inputs, labels,snr = Variable(inputs), Variable(labels), Variable(snr)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            #print("outputs", outputs.shape)
            #print("labels",labels.shape)
            loss_size = loss(outputs, np.argmax((labels), axis=1))
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            #running_loss += loss_size.data
            running_loss = loss_size.data
            #total_train_loss += loss_size.data
            total_train_loss = loss_size.data

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        '''
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
     
        
        for inputs, labels, snr in enumerate(val_loader,0):
            print("invalidation")  
            #Wrap tensors in Variables
            inputs, labels,snr = Variable(inputs), Variable(labels), Variable(snr)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]

            #Do anything with this result?
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        '''     
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    torch.save(net.state_dict(),'/home/kiran/radio_modulation/pytorch/model.pt')
    
    

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true",help="Test the model on this dataset")
        args = parser.parse_args()
        filename = '/home/kiran/radio_modulation/pytorch/qam_train.hdf5'
        x = h5py.File(filename, 'r')
        
            
        # List all groups
        #print("Keys: %s" % f.keys())
        sig = x['X']   
        label = x['Y']
        snr = x['Z'] 

        
        nn = vgg(filts)
        path = '/home/kiran/radio_modulation/pytorch/model.pt'
        if(args.train):
            train_net(sig,label, snr,nn,BATCH_SIZE, 5, LR)
        else:
            test_net(test,path,BATCH_SIZE)
            
        

        
    
