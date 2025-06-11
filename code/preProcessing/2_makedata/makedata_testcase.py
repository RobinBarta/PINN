'''
    create t x y z u v w T p dataset
'''

import os, sys
import numpy as np

from tqdm import tqdm

# %%

class Parameter:
    casename, filename, Zeros = 'RBC_PTV_1E6_07', 'Points_1E6_07_{time}.npz', 5
    t0, t1 = 0, 10

# %%

def LoadParticles(t,params):
    data = np.load(params.raw_data.format(time=str(t).zfill(params.Zeros)))['data']
    t, x, y, z = data[:,0], data[:,1], data[:,2], data[:,3]
    u, v, w, T, p = data[:,4], data[:,5], data[:,6], data[:,7], data[:,8]
    return t, x, y, z, u, v, w, T, p 

def main(): 
    params = Parameter()
    params.raw_data = '../../../data/'+params.casename+'/input/raw_data/'+params.filename
    params.data_path = '../../../data/'+params.casename+'/input/'
    
    times = np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int)
    
    outdata = np.empty([0,9])
    for t in tqdm(times):
        outdata = np.append(outdata,np.vstack([LoadParticles(t,params)]).T,axis=0)
    
    np.savez(params.data_path+params.casename+'_t_'+str(len(times))+'.npz', inputs=outdata[:,:4], outputs=outdata[:,4:])
if __name__ == "__main__":
    main()