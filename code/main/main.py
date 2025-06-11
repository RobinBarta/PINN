'''
---------------------------------------------------------------------------------
|    Project     :  PINN for RBC in Boussinesq Approximation                    |
|    Authors     :  Michael Mommert, Robin Barta, Christian Bauer, Marie Volk   |
---------------------------------------------------------------------------------

    ∂_t u + (u*𝜵)u = -𝜵p + sqrt(Pr/Ra)*Δu + T*e_z
    ∂_t T + (u*𝜵)T = sqrt(1/Pr/Ra)*ΔT
               𝜵*u = 0
'''

# %% used libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys, datetime, shutil, logging, random
logging.disable(logging.WARNING)
import numpy as np
import tensorflow as tf
import keras, importlib.util

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from keras import layers, initializers, Input, Model 

from PINN import *

# %%

# set random seed
tf.random.set_seed(3), np.random.seed(2), random.seed(2204)
# clear all previously registered custom objects
tf.keras.utils.get_custom_objects().clear()

def main(): 
    # memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    print('Running on: ', physical_devices[0])
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # ---------------------------------------
    # initialize parameters from config.py
    # ---------------------------------------
    if len(sys.argv) != 2:
        print("Usage: python ./main.py config.py")
        sys.exit()
    config_path = sys.argv[1]
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    params = config.Parameter()
    params.data_path = '../../data/'+params.casename+'/input/'+params.filename
    params.output_path = '../../data/'+params.casename+'/output/run_' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    params.load_path = '../../data/'+params.casename+'/output/'+params.initial_path+'/weights/weights_epoch_'+f"{int(params.initial_weight):04d}"+'.weights.h5'
   
    # ---------------------------------------------------------------------------
    # create output folder, save the current config file, creat weights folder
    # ---------------------------------------------------------------------------
    os.mkdir(params.output_path)
    shutil.copy(config_path, params.output_path)
    os.mkdir(params.output_path + '/weights')
    os.mkdir(params.output_path + '/plot_epochs')
    
    # -----------------------------------
    # load data
    # -----------------------------------
    data = np.load(params.data_path)
    inputs, outputs = data['inputs'], data['outputs']
    # get test dataset random over all datapoints
    #_, inputs_val, _, outputs_val = train_test_split(inputs, outputs, test_size = (3*params.batch_size)/len(inputs))
    # get test dataset equal to the middle time step
    t_test = np.unique(inputs[:,0])
    ID_t = np.argwhere(inputs[:,0]==t_test[len(t_test)//2])[:,0]
    inputs_val, outputs_val = inputs[ID_t], outputs[ID_t]
    # get truth for plot output in the LSC diagonal of the middle time step
    ID_diag = np.argwhere(np.abs(inputs[ID_t,2]-(1-inputs[ID_t,1]))<params.delta)[:,0]
    truth = np.append(inputs[ID_t][ID_diag],outputs[ID_t][ID_diag],axis=1)
        
    # -------------------
    # build PINN model
    # -------------------   
    initial_epoch = 0
    pinn = PINN(params)
    _ = pinn(tf.zeros((1, 4)))  
    pinn.model.summary()
    if params.load_initial == True:
        print('Transferring weights from initial model.')
        pinn.load_weights(params.load_path)
        initial_epoch = params.initial_weight
    
    # ---------------------
    # compile PINN model
    # ---------------------
    pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate))
    
    # -------------------
    # train PINN model
    # -------------------
    # checkpoint for saving weights after each epoch
    save_weights = tf.keras.callbacks.ModelCheckpoint(params.output_path+'/weights/weights_epoch_{epoch:04d}.weights.h5', save_weights_only=True, save_freq=int(params.save_period*len(inputs)/params.batch_size))
    # custom logging
    custom_log = CustomLoggingCallback(test_data=[inputs_val,outputs_val], data_plot_truth=truth, model=pinn, N_epochs=(initial_epoch+params.epochs), log_dir=params.output_path+'/logs', param=params)
    # lr reduction
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=params.reduction_factor,patience=params.reduction_epochs,min_lr=params.min_lr,min_delta=params.min_delta)
    # train PINN model
    pinn.fit(x=inputs, y=outputs, batch_size=params.batch_size, epochs=initial_epoch+params.epochs, initial_epoch=initial_epoch, validation_data=(inputs_val, outputs_val), verbose=0, callbacks=[custom_log, save_weights, reduce_lr])
if __name__ == "__main__":
    main()