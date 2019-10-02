import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
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

'''
Classes:
0) 4-QAM,
1) 8-QAM
'''

def get_loss_optimizer(net,learning_rate=0.001):
        #Loss
        loss = torch.nn.CrossEntropyLoss()
        #loss = torch.nn.MSELoss()
        #Optimizer
        #optimizer  = optim.Adam(net.parameters(),lr = learning_rate)
        optimizer = optim.SGD(net.parameters(),lr=learning_rate)
        return(loss, optimizer)

def test_net(test_set = None,mods = None,snrs = None, path= 'model.pt', batch_size= 128, fname=None):  
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2)
    n_batches = len(test_loader)
     
    lbl_loader = torch.utils.data.DataLoader(mods, batch_size= batch_size, num_workers=2) 
    snr_loader = torch.utils.data.DataLoader(snrs, batch_size= batch_size, num_workers=2)

    #net = ResNet18()
    model = torch.load(path)
    net = model['model']
    net.load_state_dict(model['state_dict'])
    for par in net.parameters():
        par.requires_grad = False
    net.eval()
    
    #writing results to spreadsheet
    if fname is None:
        fname = 'test_pred.csv'
    f_out = open(fname,"w")
    wrt = csv.writer(f_out)

    #testing metrics
    corr_cnt = 0
    total_iter = 0
    iter_lbl = iter(lbl_loader)
    iter_snr = iter(snr_loader)
    
    for i,inputs in enumerate(test_loader,0):
        labels = iter_lbl.next()
        snr = iter_snr.next()
        inputs,labels,snr = Variable(inputs), Variable(labels), Variable(snr)
        pred = net(inputs)

        c = 0
        if (pred.shape[1] > 2):
            c = 1
        
        pred = np.argmax(pred,axis =1)
        labels = np.argmax(labels.numpy(),axis=1)
        for s,p,l in zip(snr,pred,labels):
            
            wrt.writerow([s,p,l])
            
            if(c == 1):
                if(p > 8):
                    p = 1
                else:
                    p = 0
            
            if(p == l):
                corr_cnt += 1

            total_iter +=1 
    print("Test done, accr = :" + str(corr_cnt/total_iter))
    f_out.close()

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


def train_net(train_set=None,lbl=None,snr=None, net=None, batch_size=128, n_epochs=5 ,learning_rate = 0.001,fname = None,saved_model=None):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    
    dataset_size = train_set.shape[0]
    
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

    #Get training and test data 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size,sampler=train_sampler, num_workers=2)
    n_batches = len(train_loader)

    lbl_loader = torch.utils.data.DataLoader(lbl, batch_size= batch_size, num_workers=2,sampler =train_sampler)
    snr_loader = torch.utils.data.DataLoader(snr, batch_size= batch_size, num_workers=2,sampler=train_sampler)
    #Create our loss and optimizer functions
    loss, optimizer = get_loss_optimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    f_out = open(fname,"w")
    wrt = csv.writer(f_out)
    
    total_train_loss = 0

    scheduler = StepLR(optimizer, step_size=250, gamma=0.1)
    
    net = net.double()
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()

        wrt.writerow([epoch,total_train_loss])

        total_train_loss = 0
         
        iter_lbl = iter(lbl_loader)  
        iter_snr = iter(snr_loader) 
        
        if (((epoch+1) % 250) == 0):
            checkpoint = {'model':net,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()}
            file_name = 'checkpoint.pt'
            torch.save(checkpoint, file_name)    
        
        for i,inputs in enumerate(train_loader,0):
            labels = iter_lbl.next()
            snr = iter_snr.next()
            #Wrap them in a Variable object
            inputs,labels,snr = Variable(inputs), Variable(labels), Variable(snr)
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            #Forward pass, backward pass, optimize
            outputs = net(inputs.double())
            labels = labels.squeeze_()
            #labels = labels.float()
            loss_size = loss(outputs, np.argmax(labels,axis=1))
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            #Print loss from every 10% (then resets to 0) of a batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.4f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), total_train_loss , time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
                
        
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        val_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, num_workers=2,sampler=valid_sampler)
        val_lbl_loader = torch.utils.data.DataLoader(lbl, batch_size= batch_size, num_workers=2,sampler = valid_sampler)
        iter_lblval = iter(val_lbl_loader)
        
        for i,inputs in enumerate(val_loader,0):
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            labels = iter_lblval.next()
            #Forward pass
            val_outputs = net(inputs.double())
            val_loss_size = loss(val_outputs, np.argmax(labels,axis=1))
            total_val_loss += val_loss_size.data

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
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
        parser.add_argument("--file_test", type=str, help="Path to input test data")
        parser.add_argument("--filts", type=str, help="Filters")
        parser.add_argument("--learning_rate", type=float, help="Learning Rate")
        parser.add_argument("--results", type=str, help="Path to save results")
        parser.add_argument("--model_path", type=str, help="Saved model")
        parser.add_argument("--gpus",type=str, help = "GPU's to use")
        parser.add_argument("--classes", type=int, help= "Number of output classes")
        args = parser.parse_args()
        
        if args.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
        
        args = parser.parse_args()
        
        filts = []

        if args.filts is not None:
            filts = [int(x) for x in args.filts.split(",")]
        #Set up neural network

        
        if(args.train):
            ''' 
            #Training data
            model = torch.load('checkpoint.pt')
            nn = model['model']
            nn.load_state_dict(model['state_dict'])
            '''
            nn = ResNet18(args.classes)
            #nn = vgg(filts)
            file_name = args.file_train 
            data = pickle.load(open(file_name, "rb"), encoding ='latin1')
            mods,snrs = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [0,1])
            x = []  
            lbl = []
            classes = []
            sns = []
            
            for mod in mods:
                for snr in snrs:
                    x.append(data[(mod,snr)])
                    for i in range(data[(mod,snr)].shape[0]):  
                        lbl.append((mod,snr))
                        classes.append(mod)
                        sns.append(snr)

            x = np.vstack(x)
            classes = to_onehot(classes)
            print(x.shape)         
            train_net(x,classes,sns,nn,args.batch_size, args.steps, args.learning_rate,args.results,args.model_path)
                    
        else:
            #Testing Data
            
            data_test = pickle.load(open(os.path.expanduser(args.file_test), "rb"), encoding ='latin1')
            mods,snrs = map(lambda j: sorted(list(set(map(lambda x: x[j], data_test.keys())))), [0,1])
            #print(mods) 
            x = []  
            lbl = []
            classes= []
            sns = []
            for mod in mods:
                for snr in snrs:
                    x.append(data_test[(mod,snr)])
                    for i in range(data_test[(mod,snr)].shape[0]):  
                        lbl.append((mod,snr))
                        classes.append(mod)
                        sns.append(snr)
            classes = to_onehot(classes)
            x = np.vstack(x)
            
            path = args.model_path
            test_net(x,classes,sns,path,args.batch_size,'test_pred.csv')
        
        

        
    
