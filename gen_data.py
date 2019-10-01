from __future__ import division

import numpy as np
import random
import pickle
import matplotlib.pylab as plt



'''
Classes:
0) 4-QAM,
1) 8-QAM
'''

sps = 8
spv = 16
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
            vec = random.choices(A,k=spv)
        elif(p == 1):
            vec = random.choices(B,k=spv)
    
    else:
        a = random.choices(A,k=spv-p)
        c = random.choices(C,k=p)
        vec = [x.pop(0) for x in random.sample([a]*len(a) + [c]*len(c), len(a)+len(c))]

    #add noise
    s = 10**(snr/10)
    pn = 1.0/s
    noise_real = np.sqrt(pn/4)*np.random.randn(spv)
    noise_imag = np.sqrt(pn/4)*np.random.randn(spv)
    vec += noise_real + noise_imag*1j

    return vec
    
if __name__ == "__main__":
    mod = [0,1]
    
    dataset = {}
    dataset_p = {}
    loops = int(spv*16)
    #f,ax = plt.subplots(2,1, constrained_layout=True)
    #Generating normal labelled dataset for binary classifer
    for i in range(loops):
        for m in mod:
            for s in range(-20,20,2):
                dataset[(m, s)] = np.zeros([loops, 2, spv], dtype=np.float32)
                out = vec(m,s,True)
                dataset[(m,s)][i,0,:] = np.real(out)
                dataset[(m,s)][i,1,:] = np.imag(out)
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
    for i in range(int(loops/8)):
        for p in range(0,spv):
            for s in range (-20,20,2):
                dataset_p[(p, s)] = np.zeros([loops, 2, spv], dtype=np.float32)
                out = vec(p,s,False)
                dataset_p[(p,s)][i,0,:] = np.real(out)
                dataset_p[(p,s)][i,1,:] = np.imag(out)
                #print(p,s)

    print("Writing data to file")
    pickle.dump( dataset, open("qam_data.pkl", "wb" ) )
    pickle.dump( dataset_p, open("qam_data_p.pkl", "wb" ) )


       

    
