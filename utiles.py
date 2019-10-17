import numpy as np
import h5py
import torch
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler

def create_label():
#    mods = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC',
#            '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK',
#            '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
    mods = range(24)
    mo = []
    for m in mods:
        mo.append([m] * 4096)

    mo = np.expand_dims(np.hstack(mo), axis=1)
    print (np.shape(mo))

    return mo


def grab_per_snr(x, snr):

    for n in range(24):
        mod = x['X'][n * 4096 * 26:(n + 1) * 4096 * 26, :, :]
        s1 = mod[snr * 4096:(snr + 1) * 4096, :, :]
        #y = np.transpose(s1, (0, 2, 1))
        y = s1
        
        print (np.shape(y))

        if n == 0:
            sub = y
        else:
            sub = np.append(sub, y, axis=0)

    print (np.shape(sub))
    return sub


def data_generator(dataset, snr):
    print('loading data...')
    x = h5py.File(dataset, 'r+')

    sub = grab_per_snr(x, snr)
    labels = create_label()

    n_examples = labels.shape[0]
    n_train = int(n_examples * (7/8))                              # set the train size
    #print (n_train)
    index = np.random.permutation(n_examples)
    #print (index)
    train_idx = index[:n_train].tolist()
    test_idx = index[n_train:n_examples].tolist()
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    dataset = utils.TensorDataset(torch.from_numpy(sub).float(), torch.from_numpy(labels).long())
    
    train_loader = utils.DataLoader(dataset, batch_size=64, sampler=train_sampler, num_workers=2)
    test_loader = utils.DataLoader(dataset, batch_size=64, sampler=test_sampler, num_workers=2)

    return train_loader, test_loader