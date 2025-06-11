'''
    predict fields, errors and derivatives
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys, matplotlib, logging
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
    t_steps = 140    
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
    os.makedirs(params.case_path+'/prediction/pred', exist_ok = True)
    
    # select plane for the plot
    X, Z = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.z0,params.z1,params.Nz),indexing='ij')
    X, Z = np.ravel(X), np.ravel(Z)
    if params.mode == 'offdiag':
        Y = params.x1-X.copy()
        os.makedirs(params.case_path+'/prediction/pred/offdiag', exist_ok = True)
        os.makedirs(params.case_path+'/prediction/pred/offdiag/derivatives', exist_ok = True)
        params.plot1_path = params.case_path+'/prediction/pred/offdiag/'
        params.plot2_path = params.case_path+'/prediction/pred/offdiag/derivatives/'
    elif params.mode == 'diag': 
        os.makedirs(params.case_path+'/prediction/pred/diag', exist_ok = True)
        os.makedirs(params.case_path+'/prediction/pred/diag/derivatives', exist_ok = True)
        params.plot1_path = params.case_path+'/prediction/pred/diag/'
        params.plot2_path = params.case_path+'/prediction/pred/diag/derivatives/'
        Y = X.copy()
    elif params.mode == 'mid':
        Y = (params.x1-params.x0)/2*np.ones_like(X)
        os.makedirs(params.case_path+'/prediction/pred/mid', exist_ok = True)
        os.makedirs(params.case_path+'/prediction/pred/mid/derivatives', exist_ok = True)
        params.plot1_path = params.case_path+'/prediction/pred/mid/'
        params.plot2_path = params.case_path+'/prediction/pred/mid/derivatives/'
        
    # plot prediction
    for i, t in enumerate(tqdm(np.unique(inputs[:,0])[:params.t_steps],desc='Prediction: ',position=0,leave=True)):
        # prediction at slice
        inputs_plot = np.vstack([t*np.ones_like(X),X,Y,Z]).T
        pred = pinn(inputs_plot).numpy() 
        # get ground truth
        ID_t = np.argwhere(inputs[:,0]==t)[:,0]
        if params.mode == 'offdiag':
            ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-(params.x1-inputs[ID_t,1]))<params.delta)[:,0]
        elif params.mode == 'diag': 
            ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-inputs[ID_t,1])<params.delta)[:,0]
        elif params.mode == 'mid':
            ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-((params.x1-params.x0)/2))<params.delta)[:,0]
        truth = np.append(inputs[ID_t][ID_diag],outputs[ID_t][ID_diag],axis=1)
        
        # plot UVWTP
        Fontsize, Tickwidth, Pad, DPI, Size = 18, 2, 3, 100, 3
        fig, ax = plt.subplots(2, 5, figsize=(25,10), sharex=True, sharey=True)
        ax[0,0].set_title(r'u$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,0].set_title(r'u', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,1].set_title(r'v$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,1].set_title(r'v', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,2].set_title(r'w$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,2].set_title(r'w', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,3].set_title(r'T$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,3].set_title(r'T', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,4].set_title(r'p$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,4].set_title(r'p', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,1].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,2].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,3].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,4].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
        [spine.set_linewidth(2) for spine in ax[0,0].spines.values()],[spine.set_linewidth(2) for spine in ax[0,1].spines.values()],[spine.set_linewidth(2) for spine in ax[0,2].spines.values()],[spine.set_linewidth(2) for spine in ax[0,3].spines.values()],[spine.set_linewidth(2) for spine in ax[0,4].spines.values()],[spine.set_linewidth(2) for spine in ax[1,0].spines.values()],[spine.set_linewidth(2) for spine in ax[1,1].spines.values()],[spine.set_linewidth(2) for spine in ax[1,2].spines.values()],[spine.set_linewidth(2) for spine in ax[1,3].spines.values()],[spine.set_linewidth(2) for spine in ax[1,4].spines.values()]
        ax[0,0].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,1].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,2].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,3].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,4].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,0].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,1].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,2].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,3].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,4].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,0].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,1].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,2].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,3].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,4].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,0].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,1].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,2].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,3].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,4].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0,0].tick_params(axis='both', which='major', width=Tickwidth),ax[0,0].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,0].tick_params(axis='both', which='major', width=Tickwidth),ax[1,0].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,1].tick_params(axis='both', which='major', width=Tickwidth),ax[0,1].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,1].tick_params(axis='both', which='major', width=Tickwidth),ax[1,1].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,2].tick_params(axis='both', which='major', width=Tickwidth),ax[0,2].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,2].tick_params(axis='both', which='major', width=Tickwidth),ax[1,2].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,3].tick_params(axis='both', which='major', width=Tickwidth),ax[0,3].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,4].tick_params(axis='both', which='major', width=Tickwidth),ax[0,4].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,3].tick_params(axis='both', which='major', width=Tickwidth),ax[1,3].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,4].tick_params(axis='both', which='major', width=Tickwidth),ax[1,4].tick_params(axis='both', which='minor', width=Tickwidth)
        ax[0,0].set_ylim(0,1),ax[0,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,0].set_ylim(0,1),ax[1,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,0].set_xlim(0,1),ax[1,0].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,1].set_xlim(0,1),ax[1,1].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,2].set_xlim(0,1),ax[1,2].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,3].set_xlim(0,1),ax[1,3].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,4].set_xlim(0,1),ax[1,4].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
        triang = tri.Triangulation(truth[:,1],truth[:,3])
        u0 = ax[0,0].tricontourf(triang,truth[:,-5],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        v0 = ax[0,1].tricontourf(triang,truth[:,-4],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        w0 = ax[0,2].tricontourf(triang,truth[:,-3],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        if np.min(truth[:,-1]) != np.max(truth[:,-1]):
            T0 = ax[0,3].tricontourf(triang,truth[:,-2],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            p0 = ax[0,4].tricontourf(triang,truth[:,-1],levels=np.linspace(np.min(truth[:,-1]),np.max(truth[:,-1]),801),cmap='seismic',norm=TwoSlopeNorm(np.mean(truth[:,-1]),vmin=np.min(truth[:,-1]),vmax=np.max(truth[:,-1])))
        u1 = ax[1,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,0].reshape(params.Nx,params.Nz),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        v1 = ax[1,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,1].reshape(params.Nx,params.Nz),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        w1 = ax[1,2].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,2].reshape(params.Nx,params.Nz),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        T1 = ax[1,3].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,3].reshape(params.Nx,params.Nz),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        p1 = ax[1,4].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz),pred[:,4].reshape(params.Nx,params.Nz),levels=np.linspace(np.min(pred[:,4]),np.max(pred[:,4]),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(pred[:,4]),vmax=np.max(pred[:,4])))
        plt.colorbar(u0), plt.colorbar(v0), plt.colorbar(w0)   
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)           
        if np.min(truth[:,-1]) != np.max(truth[:,-1]):
            plt.colorbar(T0)  , plt.colorbar(p0)  
            plt.colorbar(T1)  , plt.colorbar(p1)
        plt.tight_layout()
        plt.savefig(params.plot1_path+'uvwTp_'+str(i).zfill(4)+'.jpg',dpi=DPI)
        plt.close('all')
        
        # plot derivatives
        U, V, W, Ti, P = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Ut, Ux, Uy, Uz = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Vt, Vx, Vy, Vz = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Wt, Wx, Wy, Wz = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Tt, Tx, Ty, Tz = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Px, Py, Pz = np.empty(0), np.empty(0), np.empty(0)
        Uxx, Uyy, Uzz = np.empty(0), np.empty(0), np.empty(0)
        Vxx, Vyy, Vzz = np.empty(0), np.empty(0), np.empty(0)
        Wxx, Wyy, Wzz = np.empty(0), np.empty(0), np.empty(0)
        Txx, Tyy, Tzz = np.empty(0), np.empty(0), np.empty(0)
        # make sure the slice fits into the memory
        for ij in range(int(len(inputs_plot)/10000)+1):
            x = tf.Variable(inputs_plot[ij*10000:(ij+1)*10000,:])
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                data_watch = pinn(x)
                u, v, w, T, p = tf.unstack(data_watch, axis=1)
                # first derivatives
                u_t, u_x, u_y, u_z = tf.unstack(tape.gradient(u, x), axis=-1)
                v_t, v_x, v_y, v_z = tf.unstack(tape.gradient(v, x), axis=-1)
                w_t, w_x, w_y, w_z = tf.unstack(tape.gradient(w, x), axis=-1)
                T_t, T_x, T_y, T_z = tf.unstack(tape.gradient(T, x), axis=-1)
                p_t, p_x, p_y, p_z = tf.unstack(tape.gradient(p, x), axis=-1)
                # second derivatives
                u_xx = tape.gradient(u_x, x)[...,1]
                u_yy = tape.gradient(u_y, x)[...,2]
                u_zz = tape.gradient(u_z, x)[...,3]
                v_xx = tape.gradient(v_x, x)[...,1]
                v_yy = tape.gradient(v_y, x)[...,2]
                v_zz = tape.gradient(v_z, x)[...,3]
                w_xx = tape.gradient(w_x, x)[...,1]
                w_yy = tape.gradient(w_y, x)[...,2]
                w_zz = tape.gradient(w_z, x)[...,3]
                T_xx = tape.gradient(T_x, x)[...,1]
                T_yy = tape.gradient(T_y, x)[...,2]
                T_zz = tape.gradient(T_z, x)[...,3]    
                del tape
            U, V, W, Ti, P = np.append(U,u), np.append(V,v), np.append(W,w), np.append(Ti,T), np.append(P,p)
            Ut, Ux, Uy, Uz = np.append(Ut,u_t), np.append(Ux,u_x), np.append(Uy,u_y), np.append(Uz,u_z)
            Vt, Vx, Vy, Vz = np.append(Vt,v_t), np.append(Vx,v_x), np.append(Vy,v_y), np.append(Vz,v_z)
            Wt, Wx, Wy, Wz = np.append(Wt,w_t), np.append(Wx,w_x), np.append(Wy,w_y), np.append(Wz,w_z)
            Tt, Tx, Ty, Tz = np.append(Tt,T_t), np.append(Tx,T_x), np.append(Ty,T_y), np.append(Tz,T_z)
            Px, Py, Pz = np.append(Px,p_x), np.append(Py,p_y), np.append(Pz,p_z)
            Uxx, Uyy, Uzz = np.append(Uxx,u_xx), np.append(Uyy,u_yy), np.append(Uzz,u_zz)
            Vxx, Vyy, Vzz = np.append(Vxx,v_xx), np.append(Vyy,v_yy), np.append(Vzz,v_zz)
            Wxx, Wyy, Wzz = np.append(Wxx,w_xx), np.append(Wyy,w_yy), np.append(Wzz,w_zz)
            Txx, Tyy, Tzz = np.append(Txx,T_xx), np.append(Tyy,T_yy), np.append(Tzz,T_zz)
        T_pred = Wt + U*Wx + V*Wy + W*Wz + Pz - np.sqrt(params.Pr/params.Ra)*(Wxx+Wyy+Wzz)
        # NSE & conti & EE
        fig, ax = plt.subplots(4, 5, figsize=(25,19), sharex=True, sharey=True)
        ax[0,0].set_title(r'$\partial_t u$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,0].set_title(r'$\partial_t v$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,0].set_title(r'$\partial_t w$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[3,0].set_title(r'$\partial_t T$', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,1].set_title(r'$\vec{u}\cdot\nabla u$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,1].set_title(r'$\vec{u}\cdot\nabla v$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,1].set_title(r'$\vec{u}\cdot\nabla w$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[3,1].set_title(r'$\vec{u}\cdot\nabla T$', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,2].set_title(r'$-(Pr/Ra)^{0.5} \Delta u$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,2].set_title(r'$-(Pr/Ra)^{0.5} \Delta v$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,2].set_title(r'$-(Pr/Ra)^{0.5} \Delta w$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[3,2].set_title(r'$-(Pr*Ra)^{-0.5} \Delta T$', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,3].set_title(r'$\partial_x p$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,3].set_title(r'$\partial_y p$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,3].set_title(r'$\partial_z p$', fontsize=Fontsize, pad=9, fontweight='bold')
        ax[0,4].set_title(r'$\nabla\cdot\vec{u}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,4].set_title(r'$T$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[2,4].set_title(r'$\Sigma$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[3,4].set_title(r'$\Sigma$', fontsize=Fontsize, pad=9, fontweight='bold')
        for k in range(4):
            for l in range(5):
                ax[k,l].tick_params(axis='both', which='major', width=Tickwidth)
                ax[k,l].tick_params(axis='both', which='minor', width=Tickwidth)
                ax[k,l].yaxis.set_minor_locator(AutoMinorLocator(2))
                ax[k,l].xaxis.set_minor_locator(AutoMinorLocator(2))
                [spine.set_linewidth(2) for spine in ax[k,l].spines.values()]
        ax[0,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[2,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[3,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
        ax[3,0].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[3,1].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[3,2].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[3,3].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[3,4].set_xlabel('X', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
        ax[0,0].set_ylim(0,1),ax[0,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,0].set_ylim(0,1),ax[1,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[2,0].set_ylim(0,1),ax[2,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[3,0].set_ylim(0,1),ax[3,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
        ax[1,0].set_xlim(0,1),ax[3,0].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,1].set_xlim(0,1),ax[3,1].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,2].set_xlim(0,1),ax[3,2].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,3].set_xlim(0,1),ax[3,3].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,4].set_xlim(0,1),ax[3,4].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
        #
        u0 = ax[0,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Ut.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Ut),np.max(Ut),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Ut),vmax=np.max(Ut)))
        v0 = ax[0,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), (U*Ux + V*Uy + W*Uz).reshape(params.Nx,params.Nz), levels=np.linspace(np.min((U*Ux + V*Uy + W*Uz)),np.max((U*Ux + V*Uy + W*Uz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Ux + V*Uy + W*Uz)),vmax=np.max((U*Ux + V*Uy + W*Uz))))
        w0 = ax[0,2].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), -(np.sqrt(params.Pr/params.Ra) * (Uxx+Uyy+Uzz)).reshape(params.Nx,params.Nz), levels=np.linspace(np.min(-(np.sqrt(params.Pr/params.Ra) * (Uxx+Uyy+Uzz))),np.max(-(np.sqrt(params.Pr/params.Ra) * (Uxx+Uyy+Uzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(params.Pr/params.Ra) * (Uxx+Uyy+Uzz))),vmax=np.max(-(np.sqrt(params.Pr/params.Ra) * (Uxx+Uyy+Uzz)))))
        T0 = ax[0,3].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Px.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Px),np.max(Px),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Px),vmax=np.max(Px)))
        p0 = ax[0,4].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), (Ux+Vy+Wz).reshape(params.Nx,params.Nz), levels=np.linspace(np.min((Ux+Vy+Wz)),np.max((Ux+Vy+Wz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Ux+Vy+Wz)),vmax=np.max((Ux+Vy+Wz))))
        #
        u1 = ax[1,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Vt.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Vt),np.max(Vt),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Vt),vmax=np.max(Vt)))
        v1 = ax[1,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), (U*Vx + V*Vy + W*Vz).reshape(params.Nx,params.Nz), levels=np.linspace(np.min((U*Vx + V*Vy + W*Vz)),np.max((U*Vx + V*Vy + W*Vz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Vx + V*Vy + W*Vz)),vmax=np.max((U*Vx + V*Vy + W*Vz))))
        w1 = ax[1,2].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), -(np.sqrt(params.Pr/params.Ra) * (Vxx+Vyy+Vzz)).reshape(params.Nx,params.Nz), levels=np.linspace(np.min(-(np.sqrt(params.Pr/params.Ra) * (Vxx+Vyy+Vzz))),np.max(-(np.sqrt(params.Pr/params.Ra) * (Vxx+Vyy+Vzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(params.Pr/params.Ra) * (Vxx+Vyy+Vzz))),vmax=np.max(-(np.sqrt(params.Pr/params.Ra) * (Vxx+Vyy+Vzz)))))
        T1 = ax[1,3].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Py.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Py),np.max(Py),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Py),vmax=np.max(Py)))
        p1 = ax[1,4].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Ti.reshape(params.Nx,params.Nz), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        #
        u2 = ax[2,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Wt.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Wt),np.max(Wt),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Wt),vmax=np.max(Wt)))
        v2 = ax[2,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), (U*Wx + V*Wy + W*Wz).reshape(params.Nx,params.Nz), levels=np.linspace(np.min((U*Wx + V*Wy + W*Wz)),np.max((U*Wx + V*Wy + W*Wz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Wx + V*Wy + W*Wz)),vmax=np.max((U*Wx + V*Wy + W*Wz))))
        w2 = ax[2,2].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), -(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)).reshape(params.Nx,params.Nz), levels=np.linspace(np.min(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),np.max(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),vmax=np.max(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))))
        T2 = ax[2,3].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Pz.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Pz),np.max(Pz),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Pz),vmax=np.max(Pz)))
        p2 = ax[2,4].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), T_pred.reshape(params.Nx,params.Nz), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        #
        u3 = ax[3,0].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), Tt.reshape(params.Nx,params.Nz), levels=np.linspace(np.min(Tt),np.max(Tt),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Tt),vmax=np.max(Tt)))
        v3 = ax[3,1].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), (U*Tx + V*Ty + W*Tz).reshape(params.Nx,params.Nz), levels=np.linspace(np.min((U*Tx + V*Ty + W*Tz)),np.max((U*Tx + V*Ty + W*Tz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Tx + V*Ty + W*Tz)),vmax=np.max((U*Tx + V*Ty + W*Tz))))
        w3 = ax[3,2].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), -(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)).reshape(params.Nx,params.Nz), levels=np.linspace(np.min(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))),np.max(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))),vmax=np.max(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))))
        T3 = ax[3,4].contourf(X.reshape(params.Nx,params.Nz),Z.reshape(params.Nx,params.Nz), (Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))).reshape(params.Nx,params.Nz), levels=np.linspace(np.min((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))),np.max((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))),vmax=np.max((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))))))
        #
        plt.colorbar(u0), plt.colorbar(v0), plt.colorbar(w0)  , plt.colorbar(T0)  , plt.colorbar(p0)     
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)  , plt.colorbar(p1)              
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)  , plt.colorbar(p2)             
        plt.colorbar(u3), plt.colorbar(v3), plt.colorbar(w3)  , plt.colorbar(T3)            
        plt.tight_layout()
        plt.savefig(params.plot2_path+'NSE_conti_EE_'+str(i).zfill(4)+'.jpg',dpi=DPI)
        plt.close('all')   
if __name__ == "__main__":
    main()