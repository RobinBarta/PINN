1. Juni 2025
Version: 1.0

## Project description

The RBC-PINN project is used to reconstruct temperature and pressure fields based on velocity data in thermal convective flows. Details on the architecture and the method are found in [1]. We are happy to help with modifying the PINN to reconstruct only pressure and velocity gradients in non-thermal flows. Also, The proposed PINN is compatible with PTV data generated with proPTV: https://github.com/RobinBarta/proPTV.

The project comes with a test case involving a DNS generated Rayleigh-Bénard convection with Ra=1E6 and Pr=0.7.

## How to install?

#### Requirements:

RBC-PINN requires Python 3.10 installed with pip. It was developed and tested on Python 3.10.4 which can be downloaded at: https://www.python.org/downloads/release/python-3104/

#### Installation:

1) download the RBC-PINN project to a desired location

2) install python 3.10.4 and pip on Windows or Linux

3) install cuda12.3 and cudnn8.9.7 for your gpu

4) open your terminal, navigate into the PINN project folder location and set up a venv with:

  `python -m venv venv`

5) active the venv one Windows with 

  `cd venv/Scripts activate`
   
   and on Linux with:

  `source venv/bin/activate`

6) install the required python packages

  `pip install -r requirements.txt`
  
## How to use?

1) Set up a data folder using code/preprocessing/1_createDataset/createdataset.py by inserting a data name. 

-> ignore this step for testing, see the test case folder data/RBC_DNS_1E6_07

2) Create a dataset using the code/preprocessing/2_makedata/makedata_exp.py file for your own measurment data, e.g. using the raw PTV data generated with proPTV under code/postProcessing/10_PINN/get_lagrange_data.py. The dataformat is a long list stored as .npz file with rows: t x y z u v w T p. In the case of experimental data set T and p everywhere to zero.

-> In the case of processing the test case run code/preprocessing/2_makedata/makedata_testcase.py to generate the dataset stored at data/RBC_DNS_1E6_07/input

3) set up the config file at code/main/config.py for your dataset

-> ignore this step for testing, the config.py file is already set up for the test case

4) run the code in your venv and navigate to code/main/ and run the following command in your terminal:
   
  `python main.py config.py`

## How to cite?

When PINN is useful for your scientific work and you use it or parts of it, you need to cite us:

[1] R.Barta, M.-C. Volk, C. Bauer, C. Wagner and M. Mommert. Temperature and pressure reconstruction in turbulent Rayleigh-Bénard convection by Lagrangian velocities using PINN, 2025. Preprint: https://doi.org/10.48550/arXiv.2505.02580

[2] Marie-Christine Volk, Anne Sergent, Didier Lucor, Michael Mommert, Christian Bauer, Claus Wagner. A PINN Methodology for Temperature Field Reconstruction in the PIV Measurement Plane: Case of Rayleigh-Bénard Convection, 2025. Preprint: https://doi.org/10.48550/arXiv.2503.23801

[3] M. Mommert, R. Barta, C. Bauer, M. C. Volk, and C. Wagner. Periodically activated physics-informed neural networks for assimilation tasks for three-dimensional Rayleigh-Bénard convection. Computers and Fluids, 2024. https://doi.org/10.1016/j.compfluid.2024.106419


and include the licence file in all copies with modifications or other code that uses parts of the LPINN framework.

## Contact

If you have a question or need help installing PINN or fixing a bug you have found, please contact me: michael.mommert@dlr.de

I am happy to help and look forward to meeting you.
