class Parameter:
    # ------------------
    # data parameters
    # ------------------
    casename, filename = 'RBC_PTV_1E6_07', 'RBC_DNS_1E6_07_t_11.npz'
    Ra, Pr = 1e6, 0.7
    
    # ------------------
    # initialization
    # ------------------
    load_initial = False
    initial_path, initial_weight = '', 0
    
    # ------------------
    # PINN parameters
    # ------------------
    N_layer, N_neuron = 10, 256
    batch_size = 4096
    epochs, save_period = 3000, 10
    learning_rate, min_lr, reduction_factor = 1e-3, 1e-4, 0.8
    min_delta, reduction_epochs = 5e-6, 100
    
    # ----------
    # weights
    # ----------
    lambda_data = 1.0
    lambda_NSE = 1e-1
    lambda_EE = 1e-2
    lambda_conti = 1e-3
    lambda_pcent = 0 # 1e-3
    lambda_bounds = 1e-4
    
    lambda_ux, lambda_uy, lambda_uz0, lambda_uz1 = 0, 0, 0, 0 # 0 or 1
    lambda_vx, lambda_vy, lambda_vz0, lambda_vz1 = 0, 0, 0, 0 # 0 or 1 
    lambda_wx, lambda_wy, lambda_wz0, lambda_wz1 = 0, 0, 0, 0 # 0 or 1 
    lambda_px, lambda_py, lambda_pz0, lambda_pz1 = 0, 0, 0, 0 # 0 or 1 
    lambda_Tx, lambda_Ty, lambda_Tz0, lambda_Tz1 = 0, 0, 1, 1 # 0 or 1 
    
    # ----------
    # fine tuning , only if Nu>0 is selected
    # ----------
    Nu = 0.0
    
    # ----------
    # plotting
    # ----------
    delta = 1/62