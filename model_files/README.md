# Model Files
* *'model.json'* has the saved model information following training.
* The folder *'model_coeffs'* contains the model files used for LAMMPS simulations.

**Note:** To account for the non-uniformly spaced knots used during model training, the potential files were modified.  
More specifically, 'uk' (i.e., uniform knots) was replaced with 'nk' (i.e., non-uniform knots) in the second line
of each potential file. See *'model_train.py'* in the [training folder](https://github.com/michaelmacisaac/MLIPs/tree/main/SiC/training) for more details.
