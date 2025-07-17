1. Juni 2025
Version: 1.0

## Project description

The RBC-PINN project is used to reconstruct temperature and pressure fields based on velocity data in thermal convective flows. Details on the architecture and the method are found in [1]. We are happy to help with modifying the PINN to reconstruct only pressure and velocity gradients in non-thermal flows. Also, The proposed PINN is compatible with PTV data generated with proPTV: https://github.com/RobinBarta/proPTV.

The project comes with a test case involving a DNS generated Rayleigh-Bénard convection in a cubic cell with Ra=1E6 and Pr=0.7.

## How to install?

#### Requirements:

RBC-PINN requires at least Python 3.10 installed with pip. It was developed and tested on Python 3.10.4 which can be downloaded at: https://www.python.org/downloads/release/python-3104/
We recommand using a Linux machine for the installation steps below and for code execution.
The code was tested with a Nvidia RTX 4090 GPU.
Make sure you installed the latest GPU drivers.

The code was also tested on a Windows 11 machine but in this case cuda12.3 and cudnn8.9.7 must be installed manually and it is needed to comment out the two pip packages (nvidia-nccl-cu12==2.19.3 and tensorflow-io-gcs-filesystem==0.37.1) in the requirements.txt file.

#### Linux Installation:

1) download the RBC-PINN project to a desired location

2) install python 3.10.4 and pip

3) open your terminal, navigate into the PINN project folder location and set up a venv with:

  `python -m venv venv`

4) active the venv on Linux with:

  `source venv/bin/activate`

5) install the required python packages

  `pip install -r requirements.txt`
  
## How to use?

1) Set up the data folder structure for your case using code/preprocessing/1_createCase/createCase.py by inserting a case name in the parameter class. 

-> ignore this step for testing, see the already created test case folder data/RBC_PTV_1E6_07

2) Create a dataset using code/preprocessing/2_makedata/makedata_exp.py for your own measurment data, e.g. using the raw PTV data generated with proPTV under code/postProcessing/10_PINN/get_lagrange_data.py. The dataformat is a long list stored as .npz file with rows: t x y z u v w T p. In the case of experimental data set T and p everywhere to zero.

-> In the case of processing the test case run code/preprocessing/2_makedata/makedata_testcase.py to generate the dataset stored at data/RBC_PTV_1E6_07/input

3) set up the config file at code/main/config.py for your dataset

-> ignore this step for testing, the config.py file is already set up for the test case

4) run the code in your venv and navigate to code/main/ and run the following command in your terminal:
   
  `python main.py config.py`
  
5) Each 10th epoch of training a picture of the reconstructed variables u, v, w, T, p compared with the ground truth along the LSC diagonal of the flow is automatically saved at: data/RBC_PTV_1E6_07/output/run_{datetime}/plot_epochs/. Also the weights of each 10th epoch are stored at data/RBC_PTV_1E6_07/output/run_{datetime}/weights/ and the training log is stored at data/RBC_PTV_1E6_07/output/run_{datetime}/logs/

## How to cite?

When PINN is useful for your scientific work and you use it or parts of it, you need to cite us:

[1] R.Barta, M.-C. Volk, C. Bauer, C. Wagner and M. Mommert. Temperature and pressure reconstruction in turbulent Rayleigh-Bénard convection by Lagrangian velocities using PINN. Measurements, Science & Technology, 2025. https://doi.org/10.1088/1361-6501/adee38

[2] M.-C. Volk, A. Sergent, D. Lucor, M. Mommert, C. Bauer, C. Wagner. A PINN Methodology for Temperature Field Reconstruction in the PIV Measurement Plane: Case of Rayleigh-Bénard Convection. International Communications in Heat and Mass Transfer, 2025. https://doi.org/10.1016/j.icheatmasstransfer.2025.109284

[3] M. Mommert, R. Barta, C. Bauer, M. C. Volk, and C. Wagner. Periodically activated physics-informed neural networks for assimilation tasks for three-dimensional Rayleigh-Bénard convection. Computers and Fluids, 2024. https://doi.org/10.1016/j.compfluid.2024.106419


and include the licence file in all copies with modifications or other code that uses parts of the LPINN framework.

## Contact

If you have a question or need help installing PINN or fixing a bug you have found, please contact us via: michael.mommert@dlr.de

We are happy to help and look forward to meeting you.
