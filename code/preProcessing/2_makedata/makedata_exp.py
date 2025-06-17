'''
    create t x y z u v w T p dataset
'''

import os, sys
import numpy as np

# %%

class Parameter:
    casename, folder, filename, Zeros = 'RBC_PTVexp_1E9_7', 'Lagrange', 'Lagrange_{time}.txt', 0
    t0, t1 = 0, 100
    L = 300 # mm
    f = 5 # Hz
    dT = 6.6 #K
    lamb = 207e-6 # 1/K

# %%


def LoadParticles(i,t,params):
    u_ff = np.sqrt(params.lamb*9810*params.L*params.dT) # mm/s
    t_ff = params.L/u_ff # s
    data = np.loadtxt(params.raw_data.format(time=str(t).zfill(params.Zeros)),skiprows=1)
    tt = data[:,0]/t_ff
    xt, yt, zt = data[:,1]/params.L, data[:,2]/params.L, data[:,3]/params.L
    ut, vt, wt = data[:,4]/u_ff, data[:,5]/u_ff, data[:,6]/u_ff
    Tt , pt = data[:,7], data[:,8]
    return tt,xt,yt,zt,ut,vt,wt,Tt,pt

def main(): 
    params = Parameter()
    params.raw_data = '../../../data/'+params.casename+'/input/raw_data/'+params.folder+'/'+params.filename
    params.data_path = '../../../data/'+params.casename+'/input/'
    
    times = np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int)
    
    outdata = np.empty([0,9])
    for i,t in enumerate(times):
        t0,x0,y0,z0,u0,v0,w0,T0,p0 = LoadParticles(i,t,params)
        print(i,t0[0])
        out = np.vstack([t0,x0,y0,z0,u0,v0,w0,T0,p0]).T
        outdata = np.append(outdata,out,axis=0)
    
    np.savez(params.data_path+params.casename+'_t_'+str(len(times))+'.npz', inputs=outdata[:,:4], outputs=outdata[:,4:])
if __name__ == "__main__":
    main()