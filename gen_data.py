from __future__ import division

from sklearn.preprocessing import normalize
import numpy as np
import random
import matplotlib.pylab as plt
import h5py
'''
Classes:
0) 4-QAM,
1) 8-QAM,
2) 16-QAM,
3) 32-QAM,
'''

sps = 8
spv = 64

A = np.array([-1.0-1.0j,1.0-1.0j, -1.0+1.0j, 1.0+1.0j,
    -2.0-2.0j, 2.0-2.0j,  2.0+2.0j, -2.0+2.0j,
    -1.0-2.0j, -1.0+2.0j, -2.0-1.0j,-2.0+1.0j, 1.0-2.0j, 1.0+2.0j, 2.0-1.0j, 2.0+1.0j,
    -1.0-3.0j, -1.0+3.0j, -2.0-3.0j,-2.0+3.0j, 1.0-3.0j, 1.0+3.0j, 2.0-3.0j, 2.0+3.0j,
    -3.0-1.0j, -3.0+1.0j, -3.0-2.0j,-3.0+2.0j, 3.0-1.0j, 3.0+1.0j, 3.0-2.0j, 3.0+2.0j], dtype=np.complex64)

Ar  = normalize(np.real(A)[:,np.newaxis],axis=0).ravel()
Ai  = normalize(np.imag(A)[:,np.newaxis],axis=0).ravel()
A = Ar + 1.0j*Ai

#print(A)
#reassign 
E  = A[4:8]

#print(A.shape)
#print(E.shape)



def vec(p=None, snr= 0,binary =True):
    #generate collection of symbols

    if(binary == True):
        if (p == 0):
            vec = random.choices(A[0:4],k=spv)
        elif(p == 1):
            vec = random.choices(A[0:8],k=spv)
        elif(p == 2):
            vec = random.choices(A[0:16],k=spv)
        else:
            vec = random.choices(A[0:32],k=spv)
    
        
    else:
        outset = int(spv*p)
        inset = spv-outset
        a = random.choices(A[0:4],k=inset)
        b = random.choices(E,k=outset)
        #vec = [x.pop(0) for x in random.sample([a]*len(a) + [b]*len(b), len(a)+len(b))]
        vec = np.concatenate((a,b),axis=0)
        np.array(random.shuffle(vec))  
    

    
    #add noise
    s = 10.0**(snr/10.0)
    pn = 1.0/s
    noise_real = np.sqrt(pn/64)*normalize(np.random.randn(spv)[:,np.newaxis],axis=0).ravel()
    noise_imag = np.sqrt(pn/64)*normalize(np.random.randn(spv)[:,np.newaxis],axis=0).ravel()
    
    vec += noise_real + noise_imag*1.0j
    vec = np.repeat(vec,sps)

    return vec
    
if __name__ == "__main__":
    mod = [0,1]
    
    loops = 4096
    
    dataset = np.zeros([loops*2*26,2,spv*sps],dtype=np.float64)
    labels = np.zeros([loops*2*26,2], dtype=np.float64)
    snrs = np.zeros([loops*2*26,1],dtype=np.float64)
    
    
    dataset_p = np.zeros([loops*20*26,2,spv*sps],dtype=np.float64)
    labels_p = np.zeros([loops*20*26,20], dtype=np.float64)
    snrs_p = np.zeros([loops*20*26,1],dtype=np.float64)
    
    #f,ax = plt.subplots(2,1, constrained_layout=True)
    #Generating normal labelled dataset for binary classifer
    
    print("Generating Binary Data")
    j = 0
    for i in range(loops):
        for m in mod:
            for s in range(-20,32,2):
                out = vec(m,s,True)
                dataset[j,0,:] = np.real(out)
                dataset[j,1,:] = np.imag(out)

                labels[j,m] = 1
                snrs[j,:] = s
                j += 1
                
                        
    #plt.show()
    #Generate probability labelled dataset
    
    
    print("Generating P-Value Data")
    j = 0
    for s in range(-20,32,2):
        print(s)
        for p in range(0,100,5):
            print(p/100)
            for i in range (loops):
                out = vec((p/100),s,False)
                dataset_p[j,0,:] = np.real(out)
                dataset_p[j,1,:] = np.imag(out)
                
                labels_p[j,int(p/5)] = 1 
                snrs_p[j,:] = s
                j += 1
    
    hf_train = h5py.File('qam_train2.hdf5', 'w')
    hf_train.create_dataset('X',data=dataset)
    hf_train.create_dataset('Y',data=labels)
    hf_train.create_dataset('Z',data=snrs)
    print("Written binary training data")
    
    
    
    hf_train_p = h5py.File('qam_train2_p.hdf5', 'w')
    hf_train_p.create_dataset('X',data=dataset_p)
    hf_train_p.create_dataset('Y',data=labels_p)
    hf_train_p.create_dataset('Z',data=snrs_p)
    print("Written training-p data")
               
    print("Successfully written data")
    hf_train.close()
    hf_train_p.close()
    print("Writing data to file")
    

    
