# %% used libraries
import os, time, keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from keras import layers
from matplotlib import tri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator

# %%


def pearson_correlation(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_mean = K.mean(x)
    y_mean = K.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = K.sum(x_centered * y_centered)
    denominator = K.sqrt(K.sum(K.square(x_centered)) * K.sum(K.square(y_centered)))   
    pearson_corr = numerator / (denominator + K.epsilon())
    return pearson_corr


class CustomLoggingCallback(keras.callbacks.Callback):
    def __init__(self, test_data, data_plot_truth, model, N_epochs, log_dir, param):
        super(CustomLoggingCallback, self).__init__()
        self.test_data = test_data  # Store test dataset
        self.data_plot_truth = data_plot_truth
        self._model = model         # Use a protected attribute for the model
        self.N_epochs = N_epochs
        self.epoch_start_time = None 
        self.log_dir = log_dir
        self.param = param
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Create or open the log file
        self.log_file = open(os.path.join(log_dir, 'training_log.txt'), 'a')
        # Write header to the log file
        header = "Epoch,Time,LR,Loss,Loss_data,Loss_T_bound,Loss_NSE,Loss_EE,Loss_conti,MAE_u,MAE_v,MAE_w,MAE_T,MAE_p,COR_u,COR_v,COR_w,COR_T,COR_p\n"
        self.log_file.write(header)
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        inputs, outputs = self.test_data
        #y_pred = self.model(inputs, training=False)
        #u, v, w, T, p = tf.unstack(y_pred, axis=1)
        # make sure the slice fits into the memory
        u, v, w, T, p = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        for ij in range(int(len(inputs)/10000)+1):
            x_data = tf.Variable(inputs[ij*10000:(ij+1)*10000,:])
            y_data = self.model(x_data)
            Uij, Vij, Wij, Tij, Pij = tf.unstack(y_data, axis=1)
            u, v, w, T, p = np.append(u,Uij), np.append(v,Vij), np.append(w,Wij), np.append(T,Tij), np.append(p,Pij)
        # unstack output and ground truth
        u_true, v_true, w_true, T_true, p_true = tf.unstack(outputs, axis=1)
        # estimate evaluation metrics
        mae = { 'MAE_u': keras.losses.mean_absolute_error(u_true, u),
                'MAE_v': keras.losses.mean_absolute_error(v_true, v),
                'MAE_w': keras.losses.mean_absolute_error(w_true, w),
                'MAE_T': keras.losses.mean_absolute_error(T_true, T),
                'MAE_p': keras.losses.mean_absolute_error(p_true, p)}
        cor = { 'COR_u': pearson_correlation(u_true, u),
                'COR_v': pearson_correlation(v_true, v),
                'COR_w': pearson_correlation(w_true, w),
                'COR_T': pearson_correlation(T_true, T),
                'COR_p': pearson_correlation(p_true, p)}
        # write output
        l1, l2, l3, l4, l5, l6 = logs['loss'], logs['loss_data'], logs['loss_bounds'], logs['loss_NSE'], logs['loss_EE'], logs['loss_conti']
        l7 = logs['learning_rate']
        eu, ev, ew, eT, ep = float(mae['MAE_u'].numpy()), float(mae['MAE_v'].numpy()), float(mae['MAE_w'].numpy()), float(mae['MAE_T'].numpy()), float(mae['MAE_p'].numpy())
        cu, cv, cw, cT, cp = float(cor['COR_u'].numpy()), float(cor['COR_v'].numpy()), float(cor['COR_w'].numpy()), float(cor['COR_T'].numpy()), float(cor['COR_p'].numpy())
        epoch_time = time.time() - self.epoch_start_time
        # Prepare log string
        log_str = f"{epoch+1},{epoch_time:.2f},{l7:.6f},{l1:.7f},{l2:.7f},{l3:.7f},{l4:.7f},{l5:.7f},{l6:.7f},"
        log_str += f"{eu:.7f},{ev:.7f},{ew:.7f},{eT:.7f},{ep:.7f},{cu:.7f},{cv:.7f},{cw:.7f},{cT:.7f},{cp:.7f}\n"
        # Write to log file
        self.log_file.write(log_str)
        self.log_file.flush()  # Ensure the data is written immediately
        # Print to console
        print(f"Epoch: {epoch+1}/{self.N_epochs} - Time: {epoch_time:.2f}s - LR: {l7:.6f}")
        print(f"  loss: {l1:.7f} - loss_data: {l2:.7f} - loss_bounds: {l3:.7f} - loss_NSE: {l4:.7f} - loss_EE: {l5:.7f} - loss_conti: {l6:.7f}")
        print(f"  mae_u: {eu:.7f} - mae_v: {ev:.7f} - mae_w: {ew:.7f} - mae_T: {eT:.7f} - mae_p: {ep:.7f}")
        print(f"  cor_u: {cu:.7f} - cor_v: {cv:.7f} - cor_w: {cw:.7f} - cor_T: {cT:.7f} - cor_p: {cp:.7f}\n")
        # plot current fields
        if epoch % self.param.save_period == 0:
            t_test = np.unique(inputs[:,0])
            t_test = t_test[len(np.unique(inputs[:,0]))//2]
            X, Z = np.meshgrid(np.linspace(0,1,80),np.linspace(0,1,80),indexing='ij')
            X, Z = np.ravel(X), np.ravel(Z)
            Y = 1-X.copy()
            #Y = X.copy()
            #Y = 0.5*np.ones_like(X)
            inputs_plot = np.vstack([t_test*np.ones_like(X),X,Y,Z]).T
            outputs_plot = self.model(inputs_plot, training=False).numpy()
            x_plot, y_plot, z_plot = inputs_plot[:,1].reshape(80,80), inputs_plot[:,2].reshape(80,80), inputs_plot[:,3].reshape(80,80)
            x0_plot, y0_plot, z0_plot = self.data_plot_truth[:,1], self.data_plot_truth[:,2], self.data_plot_truth[:,3]
            triang = tri.Triangulation(x0_plot, z0_plot)
            Fontsize, Tickwidth, Pad, DPI = 18, 2, 3, 100
            fig, ax = plt.subplots(2, 5, figsize=(18,8), sharex=True, sharey=True)
            ax[0,0].set_title(r'u$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,0].set_title(r'u', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,1].set_title(r'v$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,1].set_title(r'v', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,2].set_title(r'w$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,2].set_title(r'w', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,3].set_title(r'T$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,3].set_title(r'T', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,4].set_title(r'p$_{DNS}$', fontsize=Fontsize, pad=9, fontweight='bold'),ax[1,4].set_title(r'p', fontsize=Fontsize, pad=9, fontweight='bold')
            ax[0,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_ylabel('Z', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,0].set_xlabel('X=Y', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,1].set_xlabel('X=Y', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,2].set_xlabel('X=Y', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,3].set_xlabel('X=Y', fontsize=Fontsize, labelpad=Pad, fontweight='bold'),ax[1,4].set_xlabel('X=Y', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
            [spine.set_linewidth(2) for spine in ax[0,0].spines.values()],[spine.set_linewidth(2) for spine in ax[0,1].spines.values()],[spine.set_linewidth(2) for spine in ax[0,2].spines.values()],[spine.set_linewidth(2) for spine in ax[0,3].spines.values()],[spine.set_linewidth(2) for spine in ax[0,4].spines.values()],[spine.set_linewidth(2) for spine in ax[1,0].spines.values()],[spine.set_linewidth(2) for spine in ax[1,1].spines.values()],[spine.set_linewidth(2) for spine in ax[1,2].spines.values()],[spine.set_linewidth(2) for spine in ax[1,3].spines.values()],[spine.set_linewidth(2) for spine in ax[1,4].spines.values()]
            ax[0,0].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,1].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,2].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,3].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,4].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,0].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,1].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,2].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,3].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,4].yaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,0].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,1].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,2].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,3].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[0,4].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,0].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,1].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,2].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,3].xaxis.set_minor_locator(AutoMinorLocator(2)),ax[1,4].xaxis.set_minor_locator(AutoMinorLocator(2))
            ax[0,0].tick_params(axis='both', which='major', width=Tickwidth),ax[0,0].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,0].tick_params(axis='both', which='major', width=Tickwidth),ax[1,0].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,1].tick_params(axis='both', which='major', width=Tickwidth),ax[0,1].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,1].tick_params(axis='both', which='major', width=Tickwidth),ax[1,1].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,2].tick_params(axis='both', which='major', width=Tickwidth),ax[0,2].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,2].tick_params(axis='both', which='major', width=Tickwidth),ax[1,2].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,3].tick_params(axis='both', which='major', width=Tickwidth),ax[0,3].tick_params(axis='both', which='minor', width=Tickwidth),ax[0,4].tick_params(axis='both', which='major', width=Tickwidth),ax[0,4].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,3].tick_params(axis='both', which='major', width=Tickwidth),ax[1,3].tick_params(axis='both', which='minor', width=Tickwidth),ax[1,4].tick_params(axis='both', which='major', width=Tickwidth),ax[1,4].tick_params(axis='both', which='minor', width=Tickwidth)
            ax[0,0].set_ylim(0,1),ax[0,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,0].set_ylim(0,1),ax[1,0].set_yticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,0].set_xlim(0,1),ax[1,0].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,1].set_xlim(0,1),ax[1,1].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,2].set_xlim(0,1),ax[1,2].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,3].set_xlim(0,1),ax[1,3].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold'),ax[1,4].set_xlim(0,1),ax[1,4].set_xticks([0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0], fontsize=Fontsize-2,fontweight='bold')
            ax[0,0].tricontourf(triang,self.data_plot_truth[:,-5],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            ax[0,1].tricontourf(triang,self.data_plot_truth[:,-4],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            ax[0,2].tricontourf(triang,self.data_plot_truth[:,-3],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            if np.min(self.data_plot_truth[:,-1]) != np.max(self.data_plot_truth[:,-1]):
                ax[0,3].tricontourf(triang,self.data_plot_truth[:,-2],levels=np.linspace(-0.5,0.5,801),cmap='seismic',norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
                ax[0,4].tricontourf(triang,self.data_plot_truth[:,-1],levels=np.linspace(np.min(self.data_plot_truth[:,-1]),np.max(self.data_plot_truth[:,-1]),801),cmap='seismic',norm=TwoSlopeNorm(np.mean(self.data_plot_truth[:,-1]),vmin=np.min(self.data_plot_truth[:,-1]),vmax=np.max(self.data_plot_truth[:,-1])))
            ax[1,0].contourf(x_plot,z_plot,outputs_plot[:,0].reshape(80,80),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            ax[1,1].contourf(x_plot,z_plot,outputs_plot[:,1].reshape(80,80),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            ax[1,2].contourf(x_plot,z_plot,outputs_plot[:,2].reshape(80,80),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
            if np.min(outputs_plot[:,4])<0 and np.max(outputs_plot[:,4])>0:
                ax[1,3].contourf(x_plot,z_plot,outputs_plot[:,3].reshape(80,80),levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
                ax[1,4].contourf(x_plot,z_plot,outputs_plot[:,4].reshape(80,80),levels=np.linspace(np.min(outputs_plot[:,4]),np.max(outputs_plot[:,4]),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(outputs_plot[:,4]),vmax=np.max(outputs_plot[:,4])))
            plt.tight_layout()
            plt.savefig(self.param.output_path+'/plot_epochs/t_0_epoch_{epoch}.jpg'.format(epoch=str(epoch).zfill(5)),dpi=DPI)
            plt.close('all')
    def on_train_end(self, logs=None):
        # Close the log file when training ends
        self.log_file.close()


class PINN(keras.Model):
    def __init__(self, params):
        super(PINN, self).__init__() # Pass kwargs to the parent class           
        self.params = params
        # network architecture
        self.N_layer = self.params.N_layer
        self.N_neuron = self.params.N_neuron
        # dimensionless numbers
        self.Pr = self.params.Pr
        self.Ra = self.params.Ra
        # weights
        self.lambda_data = self.params.lambda_data
        self.lambda_conti = self.params.lambda_conti
        self.lambda_NSE = self.params.lambda_NSE
        self.lambda_EE = self.params.lambda_EE
        self.lambda_bounds = self.params.lambda_bounds
        # scaling factors
        self.Nu = self.params.Nu
        # functions
        self.activation_function = lambda a: tf.sin(a)
        self.model = self.build_model()
            
    def build_model(self):
        model = keras.Sequential()
        # input layer
        model.add(layers.Input(shape=(4,)))  # 4 inputs: t, x, y, z
        # hidden layers
        for i in range(self.N_layer):
            model.add(layers.Dense(self.N_neuron, activation=self.activation_function))
        # output layer
        model.add(layers.Dense(5))   # Output is u,v,w,T_fluc,p
        return model
    
    def call(self, inputs):
        y_pred = self.model(inputs)
        #return y_pred
        t, x, y, z = tf.unstack(inputs, axis=1)
        u, v, w, T_fluc, p = tf.unstack(y_pred, axis=1)
        if self.Nu == 0.0:
            T = T_fluc
        else:  
            Tmean = tf.where(z<0.5, (tf.exp(-z*2*self.Nu)-tf.exp(-self.Nu))/(2-2*tf.exp(-self.Nu)), (-tf.exp((z-1)*2*self.Nu)+tf.exp(-self.Nu))/(2-2*tf.exp(-self.Nu)))
            T = Tmean + (z**2-z)*T_fluc
        return tf.stack([u, v, w, T, p], axis=1) 

    def loss_function(self, inputs, y_true):
        # get current learning rate
        lr = float(self.optimizer.learning_rate)
        
        # --------------------------------
        # data loss: MSE of u,v,w
        # --------------------------------
        # load data
        t, x, y, z = tf.unstack(inputs, axis=1)
        u_true, v_true, w_true, T_true, p_true = tf.unstack(y_true, axis=1)
        
        # predict data
        with tf.GradientTape(persistent=True) as tape_data:
            tape_data.watch(inputs)
            y_pred_data = self(inputs, training=True)
            u_pred, v_pred, w_pred, T_pred, p_pred = tf.unstack(y_pred_data, axis=1)
            del tape_data
        
        # data loss
        loss_data = keras.losses.mean_squared_error(tf.concat([u_true, v_true, w_true], axis=-1), tf.concat([u_pred, v_pred, w_pred], axis=-1))
        
        # centering for the pressure field
        loss_data_pcent = tf.abs(tf.reduce_mean(p_pred))
        
        
        # --------------------------------
        # PDE losses:
        # --------------------------------
        # start with random collocation points
        N = tf.shape(inputs)[0]
        t_col = tf.reshape(t, (N,1))
        x_col = tf.random.uniform(shape=(N, 1), minval=0.0, maxval=1.0)
        y_col = tf.random.uniform(shape=(N, 1), minval=0.0, maxval=1.0)
        z_col = tf.random.uniform(shape=(N, 1), minval=0.0, maxval=1.0)
        inputs_col = tf.concat([t_col, x_col, y_col, z_col], axis=1)
        
        # automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs_col)
            # Predict the fields
            y_pred = self(inputs_col, training=True)
            u, v, w, T, p = tf.unstack(y_pred, axis=1)
                
            # first derivatives
            u_t, u_x, u_y, u_z = tf.unstack(tape.gradient(u, inputs_col), axis=-1)
            v_t, v_x, v_y, v_z = tf.unstack(tape.gradient(v, inputs_col), axis=-1)
            w_t, w_x, w_y, w_z = tf.unstack(tape.gradient(w, inputs_col), axis=-1)
            T_t, T_x, T_y, T_z = tf.unstack(tape.gradient(T, inputs_col), axis=-1)
            p_t, p_x, p_y, p_z = tf.unstack(tape.gradient(p, inputs_col), axis=-1)

            # second derivatives
            u_xx = tape.gradient(u_x, inputs_col)[...,1]
            u_yy = tape.gradient(u_y, inputs_col)[...,2]
            u_zz = tape.gradient(u_z, inputs_col)[...,3]
            
            v_xx = tape.gradient(v_x, inputs_col)[...,1]
            v_yy = tape.gradient(v_y, inputs_col)[...,2]
            v_zz = tape.gradient(v_z, inputs_col)[...,3]
            
            w_xx = tape.gradient(w_x, inputs_col)[...,1]
            w_yy = tape.gradient(w_y, inputs_col)[...,2]
            w_zz = tape.gradient(w_z, inputs_col)[...,3]
            
            T_xx = tape.gradient(T_x, inputs_col)[...,1]
            T_yy = tape.gradient(T_y, inputs_col)[...,2]
            T_zz = tape.gradient(T_z, inputs_col)[...,3]    
            
            #p_xx = tape.gradient(p_x, inputs_col)[...,1]
            #p_yy = tape.gradient(p_y, inputs_col)[...,2]
            #p_zz = tape.gradient(p_z, inputs_col)[...,3] 
            
            #u_xt, u_xx, u_xy, u_xz = tf.unstack(tape.gradient(u_x, inputs_col), axis=-1)
            #v_xt, v_xx, v_xy, v_xz = tf.unstack(tape.gradient(v_x, inputs_col), axis=-1)
            #w_xt, w_xx, w_xy, w_xz = tf.unstack(tape.gradient(w_x, inputs_col), axis=-1)
            
            #u_yt, u_yx, u_yy, u_yz = tf.unstack(tape.gradient(u_y, inputs_col), axis=-1)
            #v_yt, v_yx, v_yy, v_yz = tf.unstack(tape.gradient(v_y, inputs_col), axis=-1)
            #w_yt, w_yx, w_yy, w_yz = tf.unstack(tape.gradient(w_y, inputs_col), axis=-1)
            
            #u_zt, u_zx, u_zy, u_zz = tf.unstack(tape.gradient(u_z, inputs_col), axis=-1)
            #v_zt, v_zx, v_zy, v_zz = tf.unstack(tape.gradient(v_z, inputs_col), axis=-1)
            #w_zt, w_zx, w_zy, w_zz = tf.unstack(tape.gradient(w_z, inputs_col), axis=-1)
            
            # third derivatives
            #u_yxx = tape.gradient(u_yx, inputs_col)[...,1]
            #u_zxx = tape.gradient(u_zx, inputs_col)[...,1]
            #u_yyy = tape.gradient(u_yy, inputs_col)[...,2]
            #u_zyy = tape.gradient(u_zy, inputs_col)[...,2]
            #u_yzz = tape.gradient(u_yz, inputs_col)[...,3]
            #u_zzz = tape.gradient(u_zz, inputs_col)[...,3]
            
            #v_xxx = tape.gradient(v_xx, inputs_col)[...,1]
            #v_zxx = tape.gradient(v_zx, inputs_col)[...,1]
            #v_xyy = tape.gradient(v_xy, inputs_col)[...,2]
            #v_zyy = tape.gradient(v_zy, inputs_col)[...,2]
            #v_xzz = tape.gradient(v_xz, inputs_col)[...,3]
            #v_zzz = tape.gradient(v_zz, inputs_col)[...,3]
            
            #w_xxx = tape.gradient(w_xx, inputs_col)[...,1]
            #w_yxx = tape.gradient(w_yx, inputs_col)[...,1]
            #w_xyy = tape.gradient(w_xy, inputs_col)[...,2]
            #w_yyy = tape.gradient(w_yy, inputs_col)[...,2]
            #w_xzz = tape.gradient(w_xz, inputs_col)[...,3]
            #w_yzz = tape.gradient(w_yz, inputs_col)[...,3]
            del tape
          
        # loss NSE
        NSE_u = u_t + u*u_x + v*u_y + w*u_z + p_x - np.sqrt(self.Pr/self.Ra)*(u_xx + u_yy + u_zz)
        NSE_v = v_t + u*v_x + v*v_y + w*v_z + p_y - np.sqrt(self.Pr/self.Ra)*(v_xx + v_yy + v_zz)
        NSE_w = w_t + u*w_x + v*w_y + w*w_z + p_z - np.sqrt(self.Pr/self.Ra)*(w_xx + w_yy + w_zz) - T
        loss_NSE = tf.reduce_mean(tf.square(tf.stack([NSE_u, NSE_v, NSE_w])))
        
        # loss VOR
        #VOR_u = (w_yt - v_zt) + u*(w_yx-v_zx) + v*(w_yy-v_zy) + w*(w_yz-v_zz) - (w_y-v_z)*u_x - (u_z-w_x)*u_y - (v_x-u_y)*u_z - tf.sqrt(self.Pr/self.Ra)*(w_yxx-v_zxx + w_yyy-v_zyy + w_yzz-v_zzz) - T_y
        #VOR_v = (u_zt - w_xt) + u*(u_zx-w_xx) + v*(u_zy-w_xy) + w*(u_zz-w_xz) - (w_y-v_z)*v_x - (u_z-w_x)*v_y - (v_x-u_y)*v_z - tf.sqrt(self.Pr/self.Ra)*(u_zxx-w_xxx + u_zyy-w_xyy + u_zzz-w_xzz) + T_x
        #VOR_w = (v_xt - u_yt) + u*(v_xx-u_yx) + v*(v_xy-u_yy) + w*(v_xz-u_yz) - (w_y-v_z)*w_x - (u_z-w_x)*w_y - (v_x-u_y)*w_z - tf.sqrt(self.Pr/self.Ra)*(v_xxx-u_yxx + v_xyy-u_yyy + v_xzz-u_yzz)
        #loss_VOR = tf.reduce_mean(tf.square(tf.stack([VOR_u, VOR_v, VOR_w])))
        
        # loss Energy Equation
        EE = T_t + u*T_x + v*T_y + w*T_z - np.sqrt(1/(self.Pr*self.Ra))*(T_xx + T_yy + T_zz)
        loss_EE = tf.reduce_mean(tf.square(EE))
        
        # loss continuity equation
        divU = u_x + v_y + w_z
        loss_conti = tf.reduce_mean(tf.square(divU))
        
        # pressure poisson equation
        #PP = p_xx + p_yy + p_zz + (u_x*u_x + v_y*v_y + w_z*w_z + 2*(u_y*v_x + v_z*w_y + w_x*u_z)) - T_z
        #loss_pp = tf.reduce_mean(tf.square(PP))
        
        # centering for the pressure field
        loss_col_pcent = tf.abs(tf.reduce_mean(p))
          
        
        # --------------------------------
        # boundary loss
        # --------------------------------
        # start with random collocation points
        N = tf.shape(inputs)[0]
        t_b = tf.reshape(t, (N,1))
        # boundaries in x
        x_bx = tf.cast(tf.random.uniform(shape=(N,1), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        y_bx = tf.random.uniform(shape=(N,1), minval=0., maxval=1.)
        z_bx = tf.random.uniform(shape=(N,1), minval=0., maxval=1.)
        inputs_bx = tf.concat([t_b, x_bx, y_bx, z_bx], axis=1)
        # boundaries in y
        x_by = tf.random.uniform(shape=(N,1), minval=0., maxval=1.)
        y_by = tf.cast(tf.random.uniform(shape=(N,1), minval=0, maxval=2, dtype=tf.int32), tf.float32) 
        z_by = tf.random.uniform(shape=(N,1), minval=0., maxval=1.)
        inputs_by = tf.concat([t_b, x_by, y_by, z_by], axis=1)
        # boundaries at z0
        x_bz = tf.random.uniform(shape=(N,1), minval=0., maxval=1.)
        y_bz = tf.random.uniform(shape=(N,1), minval=0., maxval=1.)
        z_bz0 = tf.cast(tf.random.uniform(shape=(N,1), minval=0, maxval=1, dtype=tf.int32), tf.float32)
        inputs_bz0 = tf.concat([t_b, x_bz, y_bz, z_bz0], axis=1)
        # boundaries at z1
        z_bz1 = tf.cast(tf.random.uniform(shape=(N,1), minval=1, maxval=2, dtype=tf.int32), tf.float32)
        inputs_bz1 = tf.concat([t_b, x_bz, y_bz, z_bz1], axis=1)    
          
        # loss boundaries in x
        with tf.GradientTape(persistent=True) as tape_bx:
            tape_bx.watch(inputs_bx)
            y_pred_bx = self(inputs_bx, training=False)
            # prediction
            u_bx, v_bx, w_bx, T_bx, p_bx = tf.unstack(y_pred_bx, axis=1)
            # first derivatives
            T_bx_t, T_bx_x, T_bx_y, T_bx_z = tf.unstack(tape_bx.gradient(T_bx, inputs_bx), axis=-1)
            p_bx_t, p_bx_x, p_bx_y, p_bx_z = tf.unstack(tape_bx.gradient(p_bx, inputs_bx), axis=-1)
            del tape_bx
        # Neumann pressure x
        loss_Neumann_p_x = tf.reduce_mean(tf.square(p_bx_x))
        # Neumann temperature x
        loss_Neumann_T_x = tf.reduce_mean(tf.square(T_bx_x))
        # Dirichlet velocity x
        loss_Dirichlet_u_x = keras.losses.mean_squared_error(tf.zeros_like(u_bx),u_bx)
        loss_Dirichlet_v_x = keras.losses.mean_squared_error(tf.zeros_like(v_bx),v_bx)
        loss_Dirichlet_w_x = keras.losses.mean_squared_error(tf.zeros_like(w_bx),w_bx)
        
        # loss boundaries in y
        with tf.GradientTape(persistent=True) as tape_by:
            tape_by.watch(inputs_by)
            y_pred_by = self(inputs_by, training=False)
            # prediction
            u_by, v_by, w_by, T_by, p_by = tf.unstack(y_pred_by, axis=1)
            # first derivatives
            T_by_t, T_by_x, T_by_y, T_by_z = tf.unstack(tape_by.gradient(T_by, inputs_by), axis=-1)
            p_by_t, p_by_x, p_by_y, p_by_z = tf.unstack(tape_by.gradient(p_by, inputs_by), axis=-1)
            del tape_by
        # Neumann pressure y
        loss_Neumann_p_y = tf.reduce_mean(tf.square(p_by_y))
        # Neumann temperature y
        loss_Neumann_T_y = tf.reduce_mean(tf.square(T_by_y))
        # Dirichlet velocity y
        loss_Dirichlet_u_y = keras.losses.mean_squared_error(tf.zeros_like(u_by),u_by)
        loss_Dirichlet_v_y = keras.losses.mean_squared_error(tf.zeros_like(v_by),v_by)
        loss_Dirichlet_w_y = keras.losses.mean_squared_error(tf.zeros_like(w_by),w_by)
        
        # loss boundaries in z0
        with tf.GradientTape(persistent=True) as tape_bz0:
            tape_bz0.watch(inputs_bz0)
            y_pred_bz0 = self(inputs_bz0, training=False)
            # prediction
            u_bz0, v_bz0, w_bz0, T_bz0, p_bz0 = tf.unstack(y_pred_bz0, axis=1)
            # first derivatives
            p_bz0_t, p_bz0_x, p_bz0_y, p_bz0_z = tf.unstack(tape_bz0.gradient(p_bz0, inputs_bz0), axis=-1)
            del tape_bz0
        # Dirichlet pressure z0
        loss_Dirichlet_p_z0 = keras.losses.mean_squared_error(0.5*tf.ones_like(p_bz0_z),p_bz0_z)
        # Dirichlet temperature z0
        loss_Dirichlet_T_z0 = keras.losses.mean_squared_error(0.5*tf.ones_like(T_bz0),T_bz0)
        # Dirichlet velocity z0
        loss_Dirichlet_u_z0 = keras.losses.mean_squared_error(tf.zeros_like(u_bz0),u_bz0)
        loss_Dirichlet_v_z0 = keras.losses.mean_squared_error(tf.zeros_like(v_bz0),v_bz0)
        loss_Dirichlet_w_z0 = keras.losses.mean_squared_error(tf.zeros_like(w_bz0),w_bz0)
        
        # loss boundaries in z1
        with tf.GradientTape(persistent=True) as tape_bz1:
            tape_bz1.watch(inputs_bz1)
            y_pred_bz1 = self(inputs_bz1, training=False)
            # prediction
            u_bz1, v_bz1, w_bz1, T_bz1, p_bz1 = tf.unstack(y_pred_bz1, axis=1)
            # first derivatives
            p_bz1_t, p_bz1_x, p_bz1_y, p_bz1_z = tf.unstack(tape_bz1.gradient(p_bz1, inputs_bz1), axis=-1)
            del tape_bz1
        # Dirichlet pressure z1
        loss_Dirichlet_p_z1 = keras.losses.mean_squared_error(-0.5*tf.ones_like(p_bz1_z),p_bz1_z)
        # Dirichlet temperature z1
        loss_Dirichlet_T_z1 = keras.losses.mean_squared_error(-0.5*tf.ones_like(T_bz1),T_bz1)
        # Dirichlet velocity z1
        loss_Dirichlet_u_z1 = keras.losses.mean_squared_error(tf.zeros_like(u_bz1),u_bz1)
        loss_Dirichlet_v_z1 = keras.losses.mean_squared_error(tf.zeros_like(v_bz1),v_bz1)
        loss_Dirichlet_w_z1 = keras.losses.mean_squared_error(tf.zeros_like(w_bz1),w_bz1)
        
        # combine boundary loss
        loss_bounds = (
                       self.params.lambda_uz1*loss_Dirichlet_u_z1 + self.params.lambda_vz1*loss_Dirichlet_v_z1 + self.params.lambda_wz1*loss_Dirichlet_w_z1 +
                       self.params.lambda_uz0*loss_Dirichlet_u_z0 + self.params.lambda_vz0*loss_Dirichlet_v_z0 + self.params.lambda_wz0*loss_Dirichlet_w_z0 +
                       self.params.lambda_uy*loss_Dirichlet_u_y + self.params.lambda_vy*loss_Dirichlet_v_y + self.params.lambda_wy*loss_Dirichlet_w_y +
                       self.params.lambda_ux*loss_Dirichlet_u_x + self.params.lambda_vx*loss_Dirichlet_v_x + self.params.lambda_wx*loss_Dirichlet_w_x +
                       self.params.lambda_Tx*loss_Neumann_T_x + self.params.lambda_Ty*loss_Neumann_T_y + self.params.lambda_Tz0*loss_Dirichlet_T_z0 + self.params.lambda_Tz1*loss_Dirichlet_T_z1 +
                       self.params.lambda_px*loss_Neumann_p_x + self.params.lambda_py*loss_Neumann_p_y + self.params.lambda_pz0*loss_Dirichlet_p_z0 + self.params.lambda_pz1*loss_Dirichlet_p_z1
                       )
        
        
        # --------------------------------
        # total loss
        # --------------------------------
        total_loss = (# data loss
                      self.lambda_data*loss_data +
                      # PDE losses
                      self.lambda_NSE*loss_NSE +
                      self.lambda_EE*loss_EE +
                      self.lambda_conti*loss_conti +
                      # pcenter losses
                      self.params.lambda_pcent*loss_data_pcent +
                      #self.params.lambda_pcent*loss_col_pcent +
                      # bounds losses
                      self.lambda_bounds*self.params.lambda_Tz1*loss_Dirichlet_T_z1 +
                      self.lambda_bounds*self.params.lambda_Tz0*loss_Dirichlet_T_z0 +
                      self.lambda_bounds*self.params.lambda_Tx*loss_Neumann_T_x + 
                      self.lambda_bounds*self.params.lambda_Ty*loss_Neumann_T_y + 
                      self.lambda_bounds*self.params.lambda_pz0*loss_Dirichlet_p_z0 + 
                      self.lambda_bounds*self.params.lambda_pz1*loss_Dirichlet_p_z1 + 
                      self.lambda_bounds*self.params.lambda_px*loss_Neumann_p_x + 
                      self.lambda_bounds*self.params.lambda_py*loss_Neumann_p_y + 
                      self.lambda_bounds*self.params.lambda_uz1*loss_Dirichlet_u_z1 + 
                      self.lambda_bounds*self.params.lambda_uz0*loss_Dirichlet_u_z0 + 
                      self.lambda_bounds*self.params.lambda_uy*loss_Dirichlet_u_y +
                      self.lambda_bounds*self.params.lambda_ux*loss_Dirichlet_u_x +
                      self.lambda_bounds*self.params.lambda_vz1*loss_Dirichlet_v_z1 + 
                      self.lambda_bounds*self.params.lambda_vz1*loss_Dirichlet_v_z0 + 
                      self.lambda_bounds*self.params.lambda_vy*loss_Dirichlet_v_y +
                      self.lambda_bounds*self.params.lambda_vx*loss_Dirichlet_v_x +
                      self.lambda_bounds*self.params.lambda_wz1*loss_Dirichlet_w_z1 + 
                      self.lambda_bounds*self.params.lambda_wz0*loss_Dirichlet_w_z0 + 
                      self.lambda_bounds*self.params.lambda_wy*loss_Dirichlet_w_y +
                      self.lambda_bounds*self.params.lambda_wx*loss_Dirichlet_w_x
                      )
        
        # return dictionary of losses
        return {'loss': total_loss, 
                'loss_data': loss_data, 
                'loss_NSE': loss_NSE, 
                'loss_EE': loss_EE,
                'loss_conti': loss_conti, 
                'loss_bounds': loss_bounds, 
                'learning_rate': lr}
    
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape_train:
            loss_dict = self.loss_function(inputs, outputs)
        loss = loss_dict['loss']
        # compute gradient
        gradients = tape_train.gradient(loss, self.trainable_variables)
        # apply gradient to update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_dict
    
    # dummy function -> see customLogging for testing
    @tf.function
    def test_step(self, data):
        inputs, y_true = data
        #y_pred = self(inputs, training=False)
        return {}