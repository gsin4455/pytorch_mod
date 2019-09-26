import torch
import h5py
import numpy as np

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


if __name__== '__main__':
        filename = '/opt/datasets/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
        x = h5py.File(filename, 'r')

        # List all groups
        #print("Keys: %s" % f.keys())
        sig = x['X']   
        label = x['Y']
        snr = x['Z']   

        #get qam portion of dataset
        idx = np.arange(12*4096*26,17*4096*26)
        len = idx.shape[0]
        print(len)
        n_train = int(len*0.5)

        train_idx = np.sort(np.random.choice(idx, size=n_train, replace=False))
        print(train_idx.shape[0])
        test_idx = np.sort(list(set(idx)-set(train_idx)))
        print("Finished separating indices")

        qam_sig = sig[train_idx,:,:]
        qam_lab = label[train_idx,:]
        qam_snr = snr[train_idx,:]
        print("Finished seperating training data")
        
        qam_sig_t = sig[test_idx,:,:]
        qam_lab_t = label[test_idx,:]
        qam_snr_t = snr[test_idx,:]
        print("Finished separating testing data")



        hf_train = h5py.File('qam_train.hdf5', 'w')
        hf_train.create_dataset('X',data=qam_sig)
        hf_train.create_dataset('Y',data=qam_lab)
        hf_train.create_dataset('Z',data=qam_snr)
        print("Written training data")

        hf_test = h5py.File('qam_test.hdf5', 'w')
        hf_test.create_dataset('X',data=qam_sig_t)
        hf_test.create_dataset('Y',data=qam_lab_t)
        hf_test.create_dataset('Z',data=qam_snr_t)
        print("Written testing data")

        print("Successfully written data")
        hf_test.close()
        hf_train.close()


