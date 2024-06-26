# Silicon Carbide Machine Learned Interatomic Potential 
This folder documents the development of a machine learned interatomic potential (MLIP) for silicon carbide (SiC), including but not limited to scripts, model files, and examples.
## Data
This folder includes data featurization scripts. The input data file for featurization can be found on the [Dropbox](https://www.dropbox.com/scl/fo/q3i0kfc37l0tygk4vk6ng/h?rlkey=bf0m9bd6375wxrks2wo8is0su&dl=0).
## Training
This folder contains the model training script to reproduce the developed model. The specific training and testing keys, *'train_data.pkl'* and *'test_data.pkl'*, respectively, have been provided.    
The model hyperparameters include ridge and curvature regularizers, and a energy/force weight term. The regularizer hyperparameter values can be found in the *'regularizer'* variable (line 281) and the weight hyperparameter value can be found in the *'model.fit_from_file'* function call (line 286) within [*'/training/model_train.py'*](https://github.com/SubhashUFlorida/SiC-MLIP/blob/main/training/model_train.py).  
**Note:** Reproducing the model is not necessary for use, one can simply use the model files found in the folder [*'/model_files/potential_files/'*](https://github.com/michaelmacisaac/MLIPs/tree/main/SiC/model_files/model_coeffs).     

## Model Files
This folder contains the trained model saved in a json file and the LAMMPS model files.  
## VASP Validation  
This folder includes a VASP relaxation of a carbon-terminated SiC-3C (001) surface exhibiting the dimer-reconstruction  
## LAMMPS 
This folder serves as a resource for using the developed MLIP in LAMMPS simulations, including simulations used during MLIP validation.  
For lammps installation and compilation details please refer to [UF3 LAMMPS plug-in](https://github.com/uf3/uf3/tree/master/lammps_plugin).
### Pair style and pair coeff 
Below are examples of pair_style and pair_coeff commands to use for modeling SiC with CPU and GPU resources. Please see above link for more detailed information.
* CPU
```
    pair_style uf3 3 2
    pair_coeff 2 2 potential_files/Si_Si
    pair_coeff 1 1 potential_files/C_C
    pair_coeff 1 1 1 potential_files/C_C_C
    pair_coeff 1 1 2 potential_files/C_C_Si
    pair_coeff 1 2 potential_files/C_Si
    pair_coeff 1 2 2 potential_files/C_Si_Si
    pair_coeff 2 1 1 potential_files/Si_C_C
    pair_coeff 2 1 2 potential_files/Si_C_Si
    pair_coeff 2 2 2 potential_files/Si_Si_Si
```
* GPU (Kokkos)
```
    pair_style uf3 3 2
    pair_coeff 2 2 potential_files/Si_Si
    pair_coeff 1 1 potential_files/C_C
    pair_coeff 3b 1 1 1 potential_files/C_C_C
    pair_coeff 3b 1 1 2 potential_files/C_C_Si
    pair_coeff 1 2 potential_files/C_Si
    pair_coeff 3b 1 2 2 potential_files/C_Si_Si
    pair_coeff 3b 2 1 1 potential_files/Si_C_C
    pair_coeff 3b 2 1 2 potential_files/Si_C_Si
    pair_coeff 3b 2 2 2 potential_files/Si_Si_Si
```



