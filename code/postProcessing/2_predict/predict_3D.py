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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator

os.chdir('../../main')
sys.path.append(os.getcwd())

from PINN import *

os.chdir('../../data')


# %%

class Parameter:
    casename, filename, runname = 'RBC_DNS_1E6_07', 'RBC_DNS_1E6_07_t_11.npz', 'run1'    
    weight = 200
    # plot parameter
    t_steps = 1
    x0, x1, Nx = 0, 1, 64
    y0, y1, Ny = 0, 1, 64
    z0, z1, Nz = 0, 1, 64
    
# %%


def main(): 
    params = Parameter()
    params.case_path = params.casename+'/output/'+params.runname
    params.data_path = params.casename+'/input/'+params.filename
    
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
    
    # load data
    data = np.load(params.data_path)
    inputs = data['inputs']
    
    # create PINN inputs
    X, Y, Z = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.y0,params.y1,params.Ny),np.linspace(params.z0,params.z1,params.Nz),indexing='ij')
    X, Y, Z = np.ravel(X), np.ravel(Y), np.ravel(Z)
    os.makedirs(params.case_path+'/prediction/3D', exist_ok = True)
    params.plot_path = params.case_path+'/prediction/3D/'
        
    # plot profile
    for i, t in enumerate(tqdm(np.unique(inputs[:,0])[:params.t_steps],desc='Prediction: ',position=0,leave=True)):
        # prediction
        inputs_plot = np.vstack([t*np.ones_like(X),X,Y,Z]).T
        pred = np.empty([0,5])
        for ij in range(int(len(inputs_plot)/10000)+1):
            pred = np.append(pred,pinn(inputs_plot[ij*10000:(ij+1)*10000,:]).numpy(),axis=0)
        
        # save
        np.savez(params.plot_path+'pred_3D_'+str(i).zfill(4)+'.npz',inputs=inputs_plot,outputs=pred)
        
        # plot
        #fig = plt.figure(figsize=figsize)
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(inputs[:,1],inputs[:,2],inputs[:,3],c=pred[:,-2])
        #ax.set_xlim(params.x0,params.x1),ax.set_ylim(params.y0,params.y1),ax.set_zlim(params.z0,params.z1)
        #ax.set_xlabel('X'),ax.set_ylabel('Y'),ax.set_zlabel('Z')
        #plt.show()
        #sys.exit()
if __name__ == "__main__":
    main()