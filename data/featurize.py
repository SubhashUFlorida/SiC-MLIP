from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pickle
from uf3.representation import process
import os
import numpy as np
import matplotlib.pyplot as plt
from uf3.data import io
from uf3.data import geometry
import scipy.special as ss
from uf3.data import composition
from uf3.representation import bspline 
from uf3.util import parallel
from uf3.util import plotting

#User Inputs
n_cores = 32 #Number of cores to use for parallel processing
data_path = 'df_all.pkl' #Path to the data file

element_list = ['C', 'Si']
degree = 3
chemical_system = composition.ChemicalSystem(element_list=element_list,
                                             degree=degree)
def segmented_knots_2b(average,width,num1,num2,num3,rmin,rmax,min_endpoint_spacing=0.2):

    """
    Generates knot postions for a given interaction, featuring a gaussian like
    distribution of knots about the first nearest neighbor distance, with adjacent
    linear regions.

    Args:
	average (float): First nearest neighbor distance.
        width (float): Width of the gaussian-like distribution.
        num1 (int): number of knots in the first segment
        num2 (int): number of knots in the second segment
        num3 (int): number of knots in the third segment
        rmin (float): minimum pair distance
        rmax (float): maximum pair distance
        min_endpoint_spacing (float): minimum spacing between rmin
            and the first knot, and rmax and the last knot.

    Returns:
	knots (array): array of segmented knots
    """


    gaussian_begin = average - width/2 #startpoint of gaussian dist
    gaussian_end = average + width/2 #endpoint of gaussian dist
    erf = ss.erf(1.0)
    x2=np.linspace(-erf, erf, num2)
    e2=ss.erfinv(x2)
    gaussian_knots=average+e2*width/2 #centering dist about r0
    linear_begin=np.linspace(rmin, gaussian_begin, num1, endpoint=False) #adjacent linear segments
    linear_final=np.linspace(rmax, gaussian_end, num3, endpoint=False)
    if (linear_begin[0]-rmin)<min_endpoint_spacing:
        linear_begin=linear_begin[1:]
    if (rmax-linear_final[-1])<min_endpoint_spacing:
        linear_final=linear_final[:-1]
    knots=list(set(np.concatenate((np.array([rmin]),linear_begin, gaussian_knots, linear_final,np.array([rmax])))))
    knots=np.sort(np.concatenate((np.array([rmin,rmin,rmin]),knots,np.array([rmax,rmax,rmax])))) #padding with end knots

    return knots


def segmented_knots_3b(average_ij,average_ik,average_jk,width_ij,width_ik,
                        width_jk,num_ij_1,num_ij_2,num_ij_3,num_ik_1,num_ik_2,
                        num_ik_3,num_jk_1,num_jk_2,num_jk_3,rmin_ij,rmax_ij,
                        rmin_ik,rmax_ik,rmin_jk,rmax_jk,min_endpoint_spacing=0.2):

    """
    Generates knot postions for a given interaction, featuring a gaussian like
    distribution of knots about the first nearest neighbor distance, with adjacent
    linear regions.

    Args:
	average_ij (float): First nearest neighbor distance for ij interaction.
        average_ik (float): First nearest neighbor distance for ik interaction.
        average_jk (float): First nearest neighbor distance for jk interaction.
        width_ij (float): Width of the gaussian-like distribution for ij interaction.
        width_ik (float): Width of the gaussian-like distribution for ik interaction.
        width_jk (float): Width of the gaussian-like distribution for jk interaction.
        num_ik_1 (int): number of knots in the first segment for ik interaction.
        num_ik_2 (int): number of knots in the second segment for ik interaction.
        num_ik_3 (int): number of knots in the third segment for ik interaction.
        num_ij_1 (int): number of knots in the first segment for ij interaction.
        num_ij_2 (int): number of knots in the second segment for ij interaction.
        num_ij_3 (int): number of knots in the third segment for ij interaction.
        num_jk_1 (int): number of knots in the first segment for jk interaction.
        num_jk_2 (int): number of knots in the second segment for jk interaction.
        num_jk_3 (int): number of knots in the third segment for jk interaction.
        rmin_ik (float): minimum pair distance for ik interaction.
        rmax_ik (float): maximum pair distance for ik interaction.
        rmin_ij (float): minimum pair distance for ij interaction.
        rmax_ij (float): maximum pair distance for ij interaction.
        rmin_jk (float): minimum pair distance for jk interaction.
        rmax_jk (float): maximum pair distance for jk interaction.
        min_endpoint_spacing (float): minimum spacing between rmin and the first knot, and rmax and the last knot.

    Returns:
	knots_ijk (array): array of segmented knots
    """


    knots_ij=segmented_knots_2b(average_ij,width_ij,num_ij_1,num_ij_2,num_ij_3,rmin_ij,rmax_ij)
    knots_ik=segmented_knots_2b(average_ik,width_ik,num_ik_1,num_ik_2,num_ik_3,rmin_ik,rmax_ik)
    knots_jk=segmented_knots_2b(average_jk,width_jk,num_jk_1,num_jk_2,num_jk_3,rmin_jk,rmax_jk)
    knots_ijk=[knots_ij,knots_ik,knots_jk]

    return knots_ijk

#2 body terms
rmin2=0.001
width2=1.5
rmax2=6.0
#species_species_2body_num1/2/3
Si_Si_2_1=9
Si_Si_2_2=20
Si_Si_2_3=14
C_C_2_1=4
C_C_2_2=20
C_C_2_3=18
C_Si_2_1=6
C_Si_2_2=20
C_Si_2_3=15
#3 body terms
rmin3=0.5
rmax3_1=5.0
rmax3_2=10.0
width_ij=1.25
width_ik=1.25
width_jk=1.5
#species_species_3body_ij/ik(1)jk(2)_num1/2/3
C_C_3_1_1=1
C_C_3_1_2=7
C_C_3_1_3=5
C_C_3_2_1=1
C_C_3_2_2=10
C_C_3_2_3=14
Si_Si_3_1_1=3
Si_Si_3_1_2=7
Si_Si_3_1_3=4
Si_Si_3_2_1=3
Si_Si_3_2_2=9
Si_Si_3_2_3=13
C_Si_3_1_1=2
C_Si_3_1_2=6
C_Si_3_1_3=5
C_Si_3_2_1=2
C_Si_3_2_2=9
C_Si_3_2_3=14

knots_map={}

knots_map[('Si','Si')]=segmented_knots_2b(average=2.37,width=width2,
                            num1=Si_Si_2_1,num2=Si_Si_2_2,num3=Si_Si_2_3,
                            rmin=rmin2,rmax=rmax2,min_endpoint_spacing=0.2)

knots_map[('C','C')]=segmented_knots_2b(average=1.41,width=width2,
                            num1=C_C_2_1,num2=C_C_2_2,num3=C_C_2_3,
                            rmin=rmin2,rmax=rmax2,min_endpoint_spacing=0.2)

knots_map[('C','Si')]=segmented_knots_2b(average=1.88,width=width2,
                            num1=C_Si_2_1,num2=C_Si_2_2,num3=C_Si_2_3,
                            rmin=rmin2,rmax=rmax2,min_endpoint_spacing=0.2)

knots_map[('C','C','C')]=segmented_knots_3b(average_ij=1.41,average_ik=1.41,average_jk=1.41,
                            width_ij=width_ij,width_ik=width_ik,width_jk=width_jk,
                            num_ij_1=C_C_3_1_1,num_ij_2=C_C_3_1_2,num_ij_3=C_C_3_1_3,
                            num_ik_1=C_C_3_1_1,num_ik_2=C_C_3_1_2,num_ik_3=C_C_3_1_3,
                            num_jk_1=C_C_3_2_1,num_jk_2=C_C_3_2_2,num_jk_3=C_C_3_2_3,
                            rmin_ij=rmin3,rmax_ij=rmax3_1,rmin_ik=rmin3,rmax_ik=rmax3_1,rmin_jk=rmin3,rmax_jk=rmax3_2)

knots_map[('C','C','Si')]=segmented_knots_3b(average_ij=1.41,average_ik=1.88,average_jk=1.88,
                            width_ij=width_ij,width_ik=width_ik,width_jk=width_jk,
                            num_ij_1=C_C_3_1_1,num_ij_2=C_C_3_1_2,num_ij_3=C_C_3_1_3,
                            num_ik_1=C_Si_3_1_1,num_ik_2=C_Si_3_1_2,num_ik_3=C_Si_3_1_3,
                            num_jk_1=C_Si_3_2_1,num_jk_2=C_Si_3_2_2,num_jk_3=C_Si_3_2_3,
                            rmin_ij=rmin3,rmax_ij=rmax3_1,rmin_ik=rmin3,rmax_ik=rmax3_1,rmin_jk=rmin3,rmax_jk=rmax3_2)

knots_map[('C','Si','Si')]=segmented_knots_3b(average_ij=1.88,average_ik=1.88,average_jk=2.37,
                                width_ij=width_ij,width_ik=width_ik,width_jk=width_jk,
                                num_ij_1=C_Si_3_1_1,num_ij_2=C_Si_3_1_2,num_ij_3=C_Si_3_1_3,
                                num_ik_1=C_Si_3_1_1,num_ik_2=C_Si_3_1_2,num_ik_3=C_Si_3_1_3,
                                num_jk_1=Si_Si_3_2_1,num_jk_2=Si_Si_3_2_2,num_jk_3=Si_Si_3_2_3,
                                rmin_ij=rmin3,rmax_ij=rmax3_1,rmin_ik=rmin3,rmax_ik=rmax3_1,rmin_jk=rmin3,rmax_jk=rmax3_2)

knots_map[('Si','Si','Si')]=segmented_knots_3b(average_ij=2.37,average_ik=2.37,average_jk=2.37,
                                width_ij=width_ij,width_ik=width_ik,width_jk=width_jk,
                                num_ij_1=Si_Si_3_1_1,num_ij_2=Si_Si_3_1_2,num_ij_3=Si_Si_3_1_3,
                                num_ik_1=Si_Si_3_1_1,num_ik_2=Si_Si_3_1_2,num_ik_3=Si_Si_3_1_3,
                                num_jk_1=Si_Si_3_2_1,num_jk_2=Si_Si_3_2_2,num_jk_3=Si_Si_3_2_3,
                                rmin_ij=rmin3,rmax_ij=rmax3_1,rmin_ik=rmin3,rmax_ik=rmax3_1,rmin_jk=rmin3,rmax_jk=rmax3_2)

knots_map[('Si','C','C')]=segmented_knots_3b(average_ij=1.88,average_ik=1.88,average_jk=1.41,
                                width_ij=width_ij,width_ik=width_ik,width_jk=width_jk,
                                num_ij_1=C_Si_3_1_1,num_ij_2=C_Si_3_1_2,num_ij_3=C_Si_3_1_3,
                                num_ik_1=C_Si_3_1_1,num_ik_2=C_Si_3_1_2,num_ik_3=C_Si_3_1_3,
                                num_jk_1=C_C_3_2_1,num_jk_2=C_C_3_2_2,num_jk_3=C_C_3_2_3,
                                rmin_ij=rmin3,rmax_ij=rmax3_1,rmin_ik=rmin3,rmax_ik=rmax3_1,rmin_jk=rmin3,rmax_jk=rmax3_2)

knots_map[('Si','C','Si')]=segmented_knots_3b(average_ij=1.88,average_ik=2.37,average_jk=1.88,
                                width_ij=width_ij,width_ik=width_ik,width_jk=width_jk,
                                num_ij_1=C_Si_3_1_1,num_ij_2=C_Si_3_1_2,num_ij_3=C_Si_3_1_3,
                                num_ik_1=Si_Si_3_1_1,num_ik_2=Si_Si_3_1_2,num_ik_3=Si_Si_3_1_3,
                                num_jk_1=C_Si_3_2_1,num_jk_2=C_Si_3_2_2,num_jk_3=C_Si_3_2_3,
                                rmin_ij=rmin3,rmax_ij=rmax3_1,rmin_ik=rmin3,rmax_ik=rmax3_1,rmin_jk=rmin3,rmax_jk=rmax3_2)

bspline_config = bspline.BSplineBasis(chemical_system=chemical_system,knots_map=knots_map)

# Load data
with open(data_path, 'rb') as file: #Assuming df_all.pkl is the training data and is in the current directory
    df_all = pickle.load(file)

# Compute energy and force features
filename = 'features.h5'
if os.path.isfile(filename):
    print("File already exists.")
else:
    representation = process.BasisFeaturizer(bspline_config)
    client = ProcessPoolExecutor(max_workers=n_cores)
    table_template = "features_{}"
    representation.batched_to_hdf(filename,
                                df_all,
                                client,
                                n_jobs = n_cores,
                                batch_size=512,
                                progress="bar",
                                table_template=table_template)
    











