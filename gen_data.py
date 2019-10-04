from __future__ import division

import numpy as np
import random
import matplotlib.pylab as plt
import h5py

'''
Classes:
0) 4-QAM,
1) 8-QAM
'''

sps = 128
spv = 8
#vec_len = sps*spv #1024

A = np.array([-1.0-1.0j,1.0-1.0j, -1.0+1.0j, 1.0+1.0j], dtype=np.complex64)
B = np.array([-2.0-2.0j, 2.0-2.0j, -1.0-1.0j,1.0-1.0j, -1.0+1.0j, 1.0+1.0j, 2.0+2.0j, -2.0+2.0j], dtype=np.complex64)
C = np.array([-2.0-2.0j, 2.0-2.0j,2.0+2.0j, -2.0+2.0j], dtype=np.complex64)


#generate vector with certain proportion of 8-QAM\4-QAM and level of noise

def vec(p=None, snr= 0,binary =True):
    #vector of shape [1,spv]
    #generate collection of symbols

    if(binary == True):
        if (p == 0):
            vec = random.choices(A,k=spv*sps)
        elif(p == 1):
            vec = random.choices(B,k=spv*sps)
    
    else:
        a = random.choices(A,k=(spv-p)*sps)
        c = random.choices(C,k=p*sps)
        vec = [x.pop(0) for x in random.sample([a]*len(a) + [c]*len(c), len(a)+len(c))]

    #add noise
    s = 10**(snr/10)
    pn = 1.0/s
    noise_real = np.sqrt(pn/4)*np.random.randn(spv*sps)
    noise_imag = np.sqrt(pn/4)*np.random.randn(spv*sps)
    vec += noise_real + noise_imag*1j

    return vec
    
if __name__ == "__main__":
    mod = [0,1]
    

    loops = int(spv*512)
    
    
    dataset = np.zeros([loops*2*26,1024,2],dtype=np.float64)
    labels = np.zeros([loops*2*26,2], dtype=int)
    snrs = np.zeros([loops*2*26,1],dtype=int)
    
    dataset_p = np.zeros([loops*2*26,1024,2],dtype=np.float64)
    labels_p = np.zeros([loops*2*26,2], dtype=int)
    snrs_p = np.zeros([loops*2*26,1],dtype=int)

    #f,ax = plt.subplots(2,1, constrained_layout=True)
    #Generating normal labelled dataset for binary classifer
    print("Generating Binary Data")
    j = 0
    for i in range(loops):
        for m in mod:
            for s in range(-20,32,2):
                out = vec(m,s,True)
                dataset[j,:,0] = np.real(out)
                dataset[j,:,1] = np.imag(out)
                labels[j,m] = 1
                snrs[j,:] = s
                j += 1
                '''
                if(s == 18 and m == 1 and i == 0):
                    #print(dataset[(m,s)][i,0,:].shape)
                    #print(dataset[(m,s)][i,1,:].shape)
                    ax[0].scatter(dataset[(m,s)][i,0,:],dataset[(m,s)][i,1,:])
                
                if(s == -20 and m == 1 and i == 0):
                    ax[1].scatter(dataset[(m,s)][i,0,:],dataset[(m,s)][i,1,:])
                '''

    #plt.show()
    #Generate probability labelled dataset

    print("Generating P-Value Data")
    j = 0
    for i in range(int(loops/4)):
        for p in range(0,spv):
            for s in range (-20,32,2):
                out = vec(p,s,False)
                dataset_p[j,:,0] = np.real(out)
                dataset_p[j,:,1] = np.imag(out)
                labels_p[j,m] = 1
                snrs_p[j,:] = s
                j += 1

    hf_train = h5py.File('qam_train.hdf5', 'w')
    hf_train.create_dataset('X',data=dataset)
    hf_train.create_dataset('Y',data=labels)
    hf_train.create_dataset('Z',data=snrs)
    print("Written training data")

    hf_train_p = h5py.File('qam_train_p.hdf5', 'w')
    hf_train_p.create_dataset('X',data=dataset_p)
    hf_train_p.create_dataset('Y',data=labels_p)
    hf_train_p.create_dataset('Z',data=snrs_p)
    print("Written training-p data")
                
    print("Successfully written data")
    hf_train.close()
    hf_train_p.close()
    print("Writing data to file")
    

    
