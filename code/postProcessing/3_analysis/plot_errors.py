'''
    predict fields, errors and derivatives
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys, matplotlib, logging, keras
logging.disable(logging.WARNING)
import numpy as np
import importlib.util
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras import layers
from keras import backend as K
from matplotlib import tri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator

os.chdir('../../main')
sys.path.append(os.getcwd())

from PINN import *

os.chdir('../../data')


# %%

class Parameter:
    casename, filename, runname = 'RBC_EXP_1E9_7', 'RBC_EXP_1E9_7_t_145.npz', 'run1'      
    weight = 2990
    # physical parameter
    Ra, Pr = 1e9, 7
    # plot parameter
    t_steps = 1
    delta = 0.04
    mode = 'diag' # offdiag, diag, mid
    x0, x1, Nx = 0, 1, 80
    y0, y1, Ny = 0, 1, 80
    z0, z1, Nz = 0, 1, 80
    
# %%


def main(): 
    params = Parameter()
    params.case_path = params.casename+'/output/'+params.runname
    params.data_path = params.casename+'/input/'+params.filename
    
    # load data
    data = np.load(params.data_path)
    inputs, outputs = data['inputs'], data['outputs']
    
    # load config
    spec = importlib.util.spec_from_file_location("config", params.case_path+'/config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    params2 = config.Parameter()
    
    # build PINN model 
    pinn = PINN(params2)
    _ = pinn(tf.zeros((1, 4)))  
    pinn.load_weights(params.case_path+'/weights/weights_epoch_'+f"{int(params.weight):04d}"+'.weights.h5')
    
    # make output folder
    os.makedirs(params.case_path+'/prediction', exist_ok = True)
    os.makedirs(params.case_path+'/prediction/errors', exist_ok = True)
    
    # select plane for the plot
    X, Z = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.z0,params.z1,params.Nz),indexing='ij')
    X, Z = np.ravel(X), np.ravel(Z)
    if params.mode == 'offdiag':
        Y = params.x1-X.copy()
        os.makedirs(params.case_path+'/prediction/errors/offdiag', exist_ok = True)
        params.plot_path = params.case_path+'/prediction/errors/offdiag/'
    elif params.mode == 'diag': 
        os.makedirs(params.case_path+'/prediction/errors/diag', exist_ok = True)
        params.plot_path = params.case_path+'/prediction/errors/diag/'
        Y = X.copy()
    elif params.mode == 'mid':
        Y = np.linspace(params.y0,params.y1,params.Ny)[params.Ny//2]*np.ones_like(X)
        os.makedirs(params.case_path+'/prediction/errors/mid', exist_ok = True)
        params.plot_path = params.case_path+'/prediction/errors/mid/'
        
    # plot prediction
    p_cor, p_mae = [], []
    for i, t in enumerate(tqdm(np.unique(inputs[:,0])[:params.t_steps],desc='Prediction: ',position=0,leave=True)):
        # prediction at slice
        inputs_plot = np.vstack([t*np.ones_like(X),X,Y,Z]).T
        pred = pinn(inputs_plot).numpy() 
        # get ground truth
        # shift truth p around 0
        #outputs[:,-1] = outputs[:,-1]-np.mean(outputs[:,-1])
        ID_t = np.argwhere(inputs[:,0]==t)[:,0]
        if params.mode == 'offdiag':
            ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-(params.x1-inputs[ID_t,1]))<params.delta)[:,0]
        elif params.mode == 'diag': 
            ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-inputs[ID_t,1])<params.delta)[:,0]
        elif params.mode == 'mid':
            ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-np.linspace(params.y0,params.y1,params.Ny)[params.Ny//2])<params.delta)[:,0]
        truth = np.append(inputs[ID_t][ID_diag],outputs[ID_t][ID_diag],axis=1)
        # calculate error uvw
        pred_err = pinn(inputs[ID_t][ID_diag]).numpy()
        err1, err2, err3 = np.abs(truth[:,4]-pred_err[:,0]), np.abs(truth[:,5]-pred_err[:,1]), np.abs(truth[:,6]-pred_err[:,2])
        maxerru = np.max(err1)
        maxerrv = np.max(err2)
        maxerrw = np.max(err3)
        # calculate error T
        err4 = np.abs(truth[:,7]-pred_err[:,3])
        maxerrT = np.max(err4)
        # calculate error p
        # correct pressure by offset
        p, pp = truth[:,8], pred_err[:,4]
        p = p - np.mean(p)
        p_cor.append(pearson_correlation(pp,p).numpy())
        p_mae.append(keras.losses.mean_absolute_error(pp,p).numpy())
        err5 = np.abs(p-pp)
        maxerrp = np.max(err5)
        
        # plot UVW with error
        Fontsize, Cbarsize, Tickwidth, Pad, DPI = 18, 12, 2, 3, 100
        fig, ax = plt.subplots(3, 3, figsize=(16,13), dpi=DPI, sharex=True, sharey=True)
        # format
        ax[0,0].set_title(r'u$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,0].set_title(r'u', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,0].set_title(r'|u$_{DNS}$- u|', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,1].set_title(r'v$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,1].set_title(r'v', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,1].set_title(r'|v$_{DNS}$- v|', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,2].set_title(r'w$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,2].set_title(r'w', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,2].set_title(r'|w$_{DNS}$- w|', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[2,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
        ax[2,0].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[2,1].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[2,2].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
        for k in range(3):
            for l in range(3):
                ax[k,l].tick_params(axis='both', which='major', width=Tickwidth)
                ax[k,l].tick_params(axis='both', which='minor', width=Tickwidth)
                ax[k,l].yaxis.set_minor_locator(AutoMinorLocator(2))
                ax[k,l].xaxis.set_minor_locator(AutoMinorLocator(2))
                ax[k,l].set_xlim(0,1), ax[k,l].set_ylim(0,1)
                ax[k,l].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
                ax[k,l].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
                [spine.set_linewidth(3) for spine in ax[k,l].spines.values()]
        # plots
        triang = tri.Triangulation(truth[:,1],truth[:,3])
        #u0 = ax[0,0].tricontourf(triang,truth[:,-5],levels=np.linspace(-0.3,0.3,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3))
        #v0 = ax[0,1].tricontourf(triang,truth[:,-4],levels=np.linspace(-0.3,0.3,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3))
        #w0 = ax[0,2].tricontourf(triang,truth[:,-3],levels=np.linspace(-0.3,0.3,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3))
        u0 = ax[0,0].scatter(truth[:,1],truth[:,3],c=truth[:,-5],cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3),s=5)
        v0 = ax[0,1].scatter(truth[:,1],truth[:,3],c=truth[:,-4],cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3),s=5)
        w0 = ax[0,2].scatter(truth[:,1],truth[:,3],c=truth[:,-3],cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3),s=5)
        u1 = ax[1,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,0].reshape(params.Nx,params.Nz),levels=np.linspace(-0.3,0.3,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3))
        v1 = ax[1,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,1].reshape(params.Nx,params.Nz),levels=np.linspace(-0.3,0.3,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3))
        w1 = ax[1,2].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,2].reshape(params.Nx,params.Nz),levels=np.linspace(-0.3,0.3,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.3,vmax=0.3))
        u2 = ax[2,0].tricontourf(triang,err1,levels=np.linspace(0,maxerru,801),cmap='viridis',norm=TwoSlopeNorm(maxerru/2,vmin=0,vmax=maxerru))
        v2 = ax[2,1].tricontourf(triang,err2,levels=np.linspace(0,maxerrv,801),cmap='viridis',norm=TwoSlopeNorm(maxerrv/2,vmin=0,vmax=maxerrv))
        w2 = ax[2,2].tricontourf(triang,err3,levels=np.linspace(0,maxerrw,801),cmap='viridis',norm=TwoSlopeNorm(maxerrw/2,vmin=0,vmax=maxerrw))
        # colorbars
        u0bar, v0bar, w0bar = plt.colorbar(u0), plt.colorbar(v0), plt.colorbar(w0)  
        u1bar, v1bar, w1bar = plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1) 
        u2bar, v2bar, w2bar = plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  
        cbars = [u0bar,v0bar,w0bar, u1bar, v1bar, w1bar]
        for bar in cbars:
            bar.set_ticks([-0.30,-0.15,0.0,0.15,0.30],labels=[-0.30,-0.15,0.0,0.15,0.30])
            bar.ax.set_yticklabels([-0.3,-0.15,0,0.15,0.3], fontsize=Cbarsize, fontweight='bold')
            bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in bar.ax.spines.values()]
            bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
        # err 
        n = 4
        ticku = np.floor(np.linspace(0, maxerru, 5) * 10**n) / 10**n
        u2bar.set_ticks(ticku,labels=ticku)
        u2bar.ax.set_yticklabels(ticku, fontsize=Cbarsize, fontweight='bold')
        u2bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        [spine.set_linewidth(2) for spine in u2bar.ax.spines.values()]
        u2bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
        # err v
        tickv = np.floor(np.linspace(0, maxerrv, 5) * 10**n) / 10**n
        v2bar.set_ticks(tickv,labels=tickv)
        v2bar.ax.set_yticklabels(tickv, fontsize=Cbarsize, fontweight='bold')
        v2bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        [spine.set_linewidth(2) for spine in v2bar.ax.spines.values()]
        v2bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
        # err w
        tickw = np.floor(np.linspace(0, maxerrw, 5) * 10**n) / 10**n
        w2bar.set_ticks(tickw,labels=tickw)
        w2bar.ax.set_yticklabels(tickw, fontsize=Cbarsize, fontweight='bold')
        w2bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        [spine.set_linewidth(2) for spine in w2bar.ax.spines.values()]
        w2bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
        # output
        plt.tight_layout()
        plt.savefig(params.plot_path+'uvw_err_'+str(i).zfill(4)+'.jpg',dpi=DPI)
        plt.close('all') 
        
        # plot Tp with error
        if np.min(truth[:,-1]) != np.max(truth[:,-1]):
            Fontsize, Tickwidth, Pad, DPI = 18, 2, 3, 100
            fig, ax = plt.subplots(3, 2, figsize=(12,14), dpi=DPI, sharex=True, sharey=True)
            # format
            ax[0,0].set_title(r'T$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,0].set_title(r'T', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,0].set_title(r'|T$_{DNS}$- T|', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,1].set_title(r'p$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,1].set_title(r'p', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,1].set_title(r'|p$_{DNS}$- p|', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[2,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
            ax[2,0].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[2,1].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
            for k in range(3):
                for l in range(2):
                    ax[k,l].tick_params(axis='both', which='major', width=Tickwidth)
                    ax[k,l].tick_params(axis='both', which='minor', width=Tickwidth)
                    ax[k,l].yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax[k,l].xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax[k,l].set_xlim(0,1), ax[k,l].set_ylim(0,1)
                    ax[k,l].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
                    ax[k,l].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
                    [spine.set_linewidth(3) for spine in ax[k,l].spines.values()]
            # plots
            T0 = ax[0,0].scatter(truth[:,1],truth[:,3],c=truth[:,-2],cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
            p0 = ax[0,1].scatter(truth[:,1],truth[:,3],c=p,cmap='seismic',norm=TwoSlopeNorm(0,-0.03,0.03),s=5)
            #T0 = ax[0,0].tricontourf(triang,truth[:,-2],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            #p0 = ax[0,1].tricontourf(triang,p,levels=np.linspace(np.min(p),np.max(p),801),cmap='seismic',norm=TwoSlopeNorm(np.mean(p),vmin=np.min(p),vmax=np.max(p)))
            T1 = ax[1,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,3].reshape(params.Nx,params.Nz),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            p1 = ax[1,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,4].reshape(params.Nx,params.Nz),levels=np.linspace(-0.03,0.03,801), cmap='seismic', norm=TwoSlopeNorm(-0.0003,-0.03,0.03))
            T2 = ax[2,0].tricontourf(triang,err4,levels=np.linspace(0,maxerrT,801),cmap='viridis',norm=TwoSlopeNorm(maxerrT/2,vmin=0,vmax=maxerrT))
            p2 = ax[2,1].tricontourf(triang,err5,levels=np.linspace(0,maxerrp,801),cmap='viridis',norm=TwoSlopeNorm(maxerrp/2,vmin=0,vmax=maxerrp))
            # cbar
            T0bar, p0bar = plt.colorbar(T0), plt.colorbar(p0)    
            T1bar, p1bar = plt.colorbar(T1), plt.colorbar(p1) 
            T2bar, p2bar = plt.colorbar(T2), plt.colorbar(p2)  
            # cbar T
            T0bar.set_ticks([-0.50,-0.25,0.0,0.25,0.50],labels=[-0.50,-0.25,0.0,0.25,0.50])
            T0bar.ax.set_yticklabels([-0.5,-0.25,0,0.25,0.5], fontsize=Cbarsize, fontweight='bold')
            T0bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in T0bar.ax.spines.values()]
            T0bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
            T1bar.set_ticks([-0.50,-0.25,0.0,0.25,0.50],labels=[-0.50,-0.25,0.0,0.25,0.50])
            T1bar.ax.set_yticklabels([-0.5,-0.25,0,0.25,0.5], fontsize=Cbarsize, fontweight='bold')
            T1bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in T1bar.ax.spines.values()]
            T1bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
           
            p0bar.set_ticks([-0.03,-0.015,0.0,0.015,0.03],labels=[-0.03,-0.015,0.0,0.015,0.03])
            p0bar.ax.set_yticklabels([-0.03,-0.015,0.0,0.015,0.03], fontsize=Cbarsize, fontweight='bold')
            p0bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in p0bar.ax.spines.values()]
            p0bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
            p1bar.set_ticks([-0.03,-0.015,0.0,0.015,0.03],labels=[-0.03,-0.015,0.0,0.015,0.03])
            p1bar.ax.set_yticklabels([-0.03,-0.015,0.0,0.015,0.03], fontsize=Cbarsize, fontweight='bold')
            p1bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in p1bar.ax.spines.values()]
            p1bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
            
            # cbar err T
            n = 2
            tickT = np.floor(np.linspace(0, maxerrT, 5) * 10**n) / 10**n
            T2bar.set_ticks(tickT,labels=tickT)
            T2bar.ax.set_yticklabels(tickT, fontsize=Cbarsize, fontweight='bold')
            T2bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in T2bar.ax.spines.values()]
            T2bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)
            # cbar err p
            n = 4
            tickp = np.floor(np.linspace(0, maxerrp, 5) * 10**n) / 10**n
            p2bar.set_ticks(tickp,labels=tickp)
            p2bar.ax.set_yticklabels(tickp, fontsize=Cbarsize, fontweight='bold')
            p2bar.ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            [spine.set_linewidth(2) for spine in p2bar.ax.spines.values()]
            p2bar.ax.tick_params(labelsize=Cbarsize, axis='both', which='major', width=Tickwidth)

            # output
            plt.tight_layout()
            plt.savefig(params.plot_path+'Tp_err_'+str(i).zfill(4)+'.jpg',dpi=DPI)
            plt.close('all') 
                
    print('\np_cor:', np.mean(p_cor))
    print('p_mae:', np.mean(p_mae))
if __name__ == "__main__":
    main()