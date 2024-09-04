
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd
from pathlib import Path
import sys
import glob as glob
import itertools
import time
import scipy.special as ss
import logging
import shutil 
import subprocess
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from uf3.data import analyze
from uf3.data import io, geometry, composition
from uf3.representation import bspline, process
from uf3.regression import least_squares
from uf3.util import parallel


def file_copier(dirname):
    """
    Purpose: makes new directories for each model and copies files necessary 
             for elastic constant predictions.
    Arguments:
            dirname: int
                description: integer to specify directory 
            files: list of files
                description: list of files to copy
    Return:
            newdir: str
                path to newly created directory
    """
    files=['init.mod','displace.mod', 'potential.mod',
           'in.elastic','generate_uf3_lammps_pots.py',
           'data.SiC_cell','in.lattice','data.6H.lmp','init6h.mod','in.elastic6h','in.lattice6h'] 
    current_dir=os.getcwd()
    newdir=os.path.join(current_dir,'data',f'G{dirname}')
    os.makedirs(newdir, exist_ok=True)
    for file in files:
        src=file
        dst=os.path.join(newdir, file)
        shutil.copy(src,dst)
    return newdir


def model_fit(filename,ridge1,ridge2,ridge3,curve2,curve3,weight,sample_weights,training_keys,bspline_config):
    """
    Purpose: Function fits model with specified parameters and training keys.

    Arguments:
        filename: str
            description: h5 file with featurized data
        ridge3: float
            description: weight of three-body ridge regularizer
        curve2: float
            description: weight of two-body curvature parameter
        curve3: float
            description: weight of three-body curvature parameter
        weight: float, between 0 and 1
            Weight of energy in model training 
        training_keys: list
            description: list of indices to select from filename
                         to be used during model training.
        bspline_config: object
            description: instance of bspline object
    Return:
        model: model object
            description: model object to later be tested and saved
    """
    
    regularizer = bspline_config.get_regularization_matrix(ridge_1b=ridge1,
        ridge_2b=ridge2,ridge_3b=ridge3,curvature_2b=curve2,curvature_3b=curve3)

    model = least_squares.WeightedLinearModel(bspline_config, regularizer=regularizer)
    
    model.fit_from_gram(gram_ordinate_file=filename)
    return model


def model_save(model,directory):
    """
    Purpose: Function saves a fitted model

    Arguments:
        model: object
            description: fitted model object
        directory: str
            description: directory to save model to.
    """

    model.to_json(os.path.join(directory,"model.json"))
    subprocess.run(["python","generate_uf3_lammps_pots.py","model.json","model_coeffs"],cwd=directory)

def pot_file_mod(directory):
    """
    Purpose: Function modifies UF3 generated potential file, replacing
    'uk' with 'nk' in the second line.
    Arguments:
        directory: str
            description: directory to modify potential files.
    """
    files=glob.glob(os.path.join(directory,'*'))
    for file in files:
        with open(file,'r') as f:
            lines=f.readlines()
        with open(file,'w') as f:
            for line in lines:
                if 'uk' in line:
                    line=line.replace('uk','nk')
                f.write(line)

def init_mod(directory,value):
    path=os.path.join(directory,'init.mod')
    with open(path,'r') as file:
        lines=file.readlines()
    with open (path,'w') as file:
        for line in lines:
            words = line.split()
            if (('variable' in words) and ('a' in words) and ('equal' in words)):
                file.write(f'variable a equal {value}\n')
            else:
                file.write(line)
                
def elastic_3C(directory,lammps_path):
    """
    Purpose: Function evaluates the elastic constants of a model given a directory to run
            necessary lammps input file.
            Directories should include the relevant files with information as follows:
                in.elastic : as is
                init.mod : should include desired atom arangement
                diplace.mod : as is 
                potential.mod : should have the pair style and pair coeffs relevant to the potential
                                used. Whatever files are copied in 'file_copier' will be used.
    Arguments:
        directory: str
            desription : directory that code should be executed in
    
    Returns: elastic constants in list
    """
    start_time=time.time()
    subprocess.run([lammps_path, "-i", "in.elastic"],cwd=directory)
    flag=1
    while flag==1:
        elapsed_time=time.time()-start_time
        constants=np.zeros(9)
        path=os.path.join(directory,'log.lammps')
        if os.path.exists(path):
            with open(path) as f:
                lines=f.readlines()
                constants_inds=[1,3,5,8,10,12,15,17,19] #constants indices in log.lammps parsing
                if len(lines)>300:
                    for i in np.arange(-200,0,1):
                        words=lines[i].split()
                        if (('print' in words) and ("C11all" in words)):
                            constants=[float(lines[i+b].split()[4]) for b in constants_inds]
                        if constants[8]!=0:
                            flag=2
        if elapsed_time>600:
            constants=['N/A' for i in range(9)]
            flag=2
    return constants

def elastic_6H(directory,lammps_path):
    """
    Purpose: Function evaluates the elastic constants of a model given a directory to run
            necessary lammps input file.
            Directories should include the relevant files with information as follows:
                in.elastic : as is
                init.mod : should include desired atom arangement
                diplace.mod : as is 
                potential.mod : should have the pair style and pair coeffs relevant to the potential
                                used. Whatever files are copied in 'file_copier' will be used.
    Arguments:
        directory: str
            desription : directory that code should be executed in
    
    Returns: elastic constants in list
    """
    start_time=time.time()
    subprocess.run([lammps_path, "-i", "in.elastic6h"],cwd=directory)
    flag=1
    while flag==1:
        elapsed_time=time.time()-start_time
        constants=np.zeros(9)
        path=os.path.join(directory,'log.lammps')
        if os.path.exists(path):
            with open(path) as f:
                lines=f.readlines()
                constants_inds=[1,3,5,8,10,12,15,17,19] #constants indices in log.lammps parsing
                if len(lines)>300:
                    for i in np.arange(-200,0,1):
                        words=lines[i].split()
                        if (('print' in words) and ("C11all" in words)):
                            constants=[float(lines[i+b].split()[4]) for b in constants_inds]
                        if constants[8]!=0:
                            flag=2
        if elapsed_time>600:
            constants=['N/A' for i in range(9)]
            flag=2
    return constants

def cohesive3C(directory,lammps_path):
    """
    Purpose: Function evaluates the SiC-3C cohesive energy of a model given a directory to run
            necessary lammps input file.
    Arguments:
        directory: str
            desription : directory that code should be executed in
    
    Returns: elastic constants in list
    """
    start_time=time.time()
    subprocess.run([lammps_path, "-i", "in.lattice"],cwd=directory)
    flag=1
    while flag==1:
        elapsed_time=time.time()-start_time
        constants=np.zeros(9)
        path=os.path.join(directory,'log.lammps')
        if os.path.exists(path):
            with open(path) as f:
                lines=f.readlines()
                ecoh=0
                lattice_constant=0
                if len(lines)>50:
                    for i in np.arange(-40,0,1):
                        words=lines[i].split()
                        if (('print' in words) and ('energy' in words) and ('(eV)' in words)):
                            ecoh=float(lines[i+1].split()[4][:-1])
                        if (('print' in words) and ('"Lattice' in words) and ('constant' in words)):
                            lattice_constant=float(lines[i+1].split()[4][:-1])
                        if ((ecoh!=0) and (lattice_constant!=0)):
                            flag=2
                        if (('ERROR:' in words) and ('Lost' in words) and ('atoms:' in words)):
                            ecoh='N/A'
                            lattice_constant='N/A'
                            flag=2
        if elapsed_time>600:
            ecoh='N/A'
            lattice_constant='N/A'
            flag=2

    return ecoh,lattice_constant
def cohesive6H(directory,lammps_path):
    """
    Purpose: Function evaluates the SiC-3C cohesive energy of a model given a directory to run
            necessary lammps input file.
    Arguments:
        directory: str
            desription : directory that code should be executed in
    
    Returns: elastic constants in list
    """
    start_time=time.time()
    subprocess.run([lammps_path, "-i", "in.lattice6h"],cwd=directory)
    flag=1
    while flag==1:
        elapsed_time=time.time()-start_time
        path=os.path.join(directory,'log.lammps')
        if os.path.exists(path):
            with open(path) as f:
                lines=f.readlines()
                ecoh=0
                if len(lines)>50:
                    for i in np.arange(-40,0,1):
                        words=lines[i].split()
                        if (('print' in words) and ('energy' in words) and ('(eV)' in words)):
                            ecoh=float(lines[i+1].split()[4][:-1])
                        if ecoh!=0:
                            flag=2
                        if (('ERROR:' in words) and ('Lost' in words) and ('atoms:' in words)):
                            ecoh='N/A'
                            flag=2
        if elapsed_time>600:
            ecoh='N/A'
            flag=2
    return ecoh

def file_delete(file):
    """
    Purpose: Function deletes a file if it exists.
    Arguments:
        file: str
            description: path to file to be deleted.
    """
    if os.path.exists(file):
        os.remove(file)

def moduli(C,polytype):
    cij=np.array([[C[0],C[3],C[4],0,0,0],
                [C[3],C[1],C[5],0,0,0],
                [C[4],C[5],C[2],0,0,0],
                [0,0,0,C[6],0,0],[0,0,0,0,C[7],0],
                [0,0,0,0,0,C[8]]])
    sij=np.linalg.inv(cij)
    K_V=((cij[0,0]+cij[1,1]+cij[2,2])+2*(cij[0,1]+cij[1,2]+cij[2,0]))/9
    K_R=1/((sij[0,0]+sij[1,1]+sij[2,2])+2*(sij[0,1]+sij[1,2]+sij[2,0]))
    G_V=((cij[0,0]+cij[1,1]+cij[2,2])-(cij[0,1]+cij[1,2]+cij[2,0])+3*(cij[3,3]+cij[4,4]+cij[5,5]))/15
    G_R=15/(4*(sij[0,0]+sij[1,1]+sij[2,2])-4*(sij[0,1]+sij[1,2]+sij[2,0])+3*(sij[3,3]+sij[4,4]+sij[5,5]))
    K_VRH=(K_V+K_R)/2
    G_VRH=(G_V+G_R)/2
    poisson=(3*K_VRH-2*G_VRH)/(6*K_VRH+2*G_VRH)
    if polytype=='3C':
        K_VRHerror=(K_VRH-211)/211*100
        G_VRHerror=(G_VRH-187)/187*100
    if polytype=='6H':
        K_VRHerror=(K_VRH-213)/213*100
        G_VRHerror=(G_VRH-187)/187*100

    return K_VRH, G_VRH

def R2(true,predicted):
    """
    This function calculates the R2 and MAE values given true
    and predicted values.
    Arguments:
        true: array
            description: array of reference values
        predicted: array
            description: array of predicted vales
    Returns:
        R2: R-squared
        MAE: Mean average error
    """
    mae = mean_absolute_error(true,predicted)
    R2 = r2_score(true, predicted)

    return R2, mae

def task_function(input_data):
    logging.info(f" Job {input_data['dirname']} started")
    newdir=file_copier(dirname=input_data['dirname'])
    logging.info(f" Job {input_data['dirname']} directory created")
    model=model_fit(filename=input_data['filename'],ridge1=input_data['ridge1'],
                    ridge2=input_data['ridge2'],ridge3=input_data['ridge3'],
                    curve2=input_data['curve2'],curve3=input_data['curve3'],
                    weight=input_data['weight'],sample_weights=input_data['sample_weights'],
                    training_keys=input_data['training_keys'],
                    bspline_config=input_data['bspline_config'])
    logging.info(f" Job {input_data['dirname']} model fit")
    for pair in [("C","Si"),("Si","Si"),("C","C")]: 
        r_target = input_data['analysis']["lower_bounds"][pair]
        model.fix_repulsion_2b(pair, 
                       r_target=r_target,
                       min_curvature=0.0)   
    y_e, p_e, y_f, p_f, rmse_e, rmse_f = model.batched_predict(filename=input_data['holdoutfilename'], 
                                keys=input_data['holdout_keys'])
    logging.info(f" Job {input_data['dirname']} model predicted")
    energyr2, energymae = R2(y_e,p_e)
    forcer2, forcemae = R2(y_f,p_f)
    if ((rmse_e < .5) and (rmse_f<.7) and (input_data['lammps_run']==True)):
        logging.info(f" Job {input_data['dirname']} model passed RMSE criteria")
        model_save(model=model,directory=newdir)
        if input_data['uniform_knots']==False:
            pot_file_mod(os.path.join(newdir,'model_coeffs'))
        try:
            ecoh3C,lattice_constant=cohesive3C(newdir,lammps_path=input_data['lammps_path'])
        except:
            ecoh3C,lattice_constant='N/A','N/A'
        file_delete(os.path.join(newdir,'log.lammps'))
        init_mod(directory=newdir,value=lattice_constant)
        try:
            constants3C=elastic_3C(directory=newdir,lammps_path=input_data['lammps_path'])
            bulk3c_1,shear3c_1=moduli(constants3C,polytype='3C')
        except:
            constants3C=['N/A' for i in range(9)]
            bulk3c_1,shear3c_1='N/A','N/A'
        file_delete(os.path.join(newdir,'log.lammps'))
        init_mod(directory=newdir,value=(lattice_constant-.04))
        try:
            constants3C_2=elastic_3C(directory=newdir,lammps_path=input_data['lammps_path'])
            bulk3c_2,shear3c_2=moduli(constants3C_2,polytype='3C')
        except:
            constants3C_2=['N/A' for i in range(9)]
            bulk3c_2,shear3c_2='N/A','N/A'
        file_delete(os.path.join(newdir,'log.lammps'))
        init_mod(directory=newdir,value=(lattice_constant-.08))
        try:
            constants3C_3=elastic_3C(directory=newdir,lammps_path=input_data['lammps_path'])
            bulk3c_3,shear3c_3=moduli(constants3C_3,polytype='3C')
        except:
            constants3C_3=['N/A' for i in range(9)]
            bulk3c_3,shear3c_3='N/A','N/A'
        try:
            constants6H=elastic_6H(directory=newdir,lammps_path=input_data['lammps_path'])
            bulk6h,shear6h=moduli(constants6H,polytype='6H')
        except:
            constants6H=['N/A' for i in range(9)]
            bulk6h,shear6h='N/A','N/A'
        file_delete(os.path.join(newdir,'log.lammps'))
        try:
            ecoh6H=cohesive6H(newdir,lammps_path=input_data['lammps_path'])
        except: 
            ecoh6H='N/A'
    else:
        constants3C=['N/A' for i in range(9)]
        constants6H=['N/A' for i in range(9)]
        ecoh3C='N/A'
        ecoh6H='N/A'
        lattice_constant='N/A'
        bulk3c_1='N/A'
        shear3c_1='N/A'
        bulk3c_2='N/A'
        shear3c_2='N/A'
        bulk3c_3='N/A'
        shear3c_3='N/A'
        bulk6h='N/A'
        shear6h='N/A'
    try:
        file_delete(os.path.join(newdir,'log.lammps'))
        file_delete(os.path.join(newdir,'displace.mod'))
        file_delete(os.path.join(newdir,'potential.mod'))
        file_delete(os.path.join(newdir,'init.mod'))
        file_delete(os.path.join(newdir,'data.SiC_cell'))
        file_delete(os.path.join(newdir,'in.elastic'))
        file_delete(os.path.join(newdir,'generate_uf3_lammps_pots.py'))
        file_delete(os.path.join(newdir,'in.lattice'))
        file_delete(os.path.join(newdir,'data.6H.lmp'))
        file_delete(os.path.join(newdir,'init6h.mod'))
        file_delete(os.path.join(newdir,'in.elastic6h'))
        file_delete(os.path.join(newdir,'in.lattice6h'))
    except:
        pass
    output_data={'weight':input_data['weight'],'curve2':input_data['curve2'],
                 'curve3':input_data['curve3'], 'ridge1':input_data['ridge1'],
                 'ridge2':input_data['ridge2'], 'ridge3':input_data['ridge3'],
                 'RMSE_E':rmse_e, 'MAE_E':energymae,'R2_E':energyr2,
                'RMSE_F':rmse_f, 'MAE_F':forcemae, 'R2_F':forcer2,
                '3CC11':constants3C[0],'3CC12':constants3C[3], '3CC44':constants3C[6],
                '3Cbulk1':bulk3c_1,'3Cshear':shear3c_1,'3Cbulk2':bulk3c_2,'3Cbulk3':bulk3c_3,
                '3Cshear2':shear3c_2,'3Cshear3':shear3c_3,
                '6HC11':constants6H[0],'6HC33':constants6H[2],'6HC12':constants6H[3],
                '6HC23':constants6H[5],'6HC44':constants6H[7],'6HC66':constants6H[8],
                '6Hbulk':bulk6h,'6Hshear':shear6h,
                'cohesive3C':ecoh3C,'cohesive6H':ecoh6H,'lattice_constant':lattice_constant}
    with open(os.path.join(newdir,'output.json'),"w") as file:
        json.dump(output_data,file,indent=4)
    logging.info(f" Job {input_data['dirname']} completed. Output Data: {output_data}")

    return 1


def main():
    script_name = Path(__file__).stem
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=f"{script_name}.log",
        filemode="w",
        level=logging.INFO,
        )
    #defining fitting parameters
    ridge1=[1e-1,1e-2,1e-3,1e-4]
    ridge2=[1e-5,1e-6,1e-7,1e-8,1e-9]
    ridge3=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    curve2=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    curve3=[1e-5,1e-6,1e-7,1e-8,1e-9]
    weight=np.arange(0.0,0.13,0.01)
    combinations=list(itertools.product(ridge1,ridge2,ridge3,curve2,curve3,weight))
    np.random.seed(42)
    inds=np.random.choice(np.arange(len(combinations)),size=len(combinations),replace=False)
    combinations=np.array(combinations)
    combinations=combinations[inds]
    ridge1s=combinations[:,0]
    ridge2s=combinations[:,1]
    ridge3s=combinations[:,2]
    curve2s=combinations[:,3]
    curve3s=combinations[:,4]
    weights=combinations[:,5]

    #Importing data and splitting
    df_data=pd.read_pickle('/blue/subhash/michaelmacisaac/SiC/uf3pots/featurized_data/feat_29/df_all.pkl')
    test_data=[]
    tempdata=df_data[df_data.index.str.contains('temp')]
    df_data=df_data.drop(tempdata.index)
    prefixes=[]
    for x in range(len(df_data)):
        prefixes.append(f"{df_data.index[x].split('_')[0]}_{df_data.index[x].split('_')[1]}")
    prefixes=list(set(prefixes))
    for i in range(len(prefixes)):
        test_data.append(df_data[df_data.index.str.contains(f"{prefixes[i]}")])
        if len(pd.concat(test_data))>int(len(df_data)*.1):
            break
    test_data=pd.concat(test_data)
    df_data=df_data.drop(test_data.index)
    ##Split data into train and holdout indices
    training_keys = df_data.index
    holdout_keys = test_data.index
    #Upweighting samples
    sample_weights={}

     

    #Define element list and degree of model
    element_list = ['C', 'Si']
    degree = 3
    chemical_system = composition.ChemicalSystem(element_list=element_list,
                                                degree=degree)
    def knots_map_pair_hybrid(average,width,num1,num2,num3,rmin,rmax):
        r1 = average - width/2 #startpoint of gaussian dist
        r2 = average + width/2 #endpoint of gaussian dist
        erf = ss.erf(1.0)
        x2=np.linspace(-erf, erf, num2)
        e2=ss.erfinv(x2) 
        y2=average+e2*width/2 #centering dist about r0
        y1=np.linspace(rmin, r1, num1, endpoint=False)
        y3=np.linspace(rmax, r2, num3, endpoint=False)
        if (y1[0]-rmin)<0.2:
            y1=y1[1:]
        if (rmax-y3[-1])<0.2:
            y3=y3[:-1]
        overall_array=list(set(np.concatenate((np.array([rmin]),y1, y2, y3,np.array([rmax])))))
        overall_array=np.sort(np.concatenate((np.array([rmin,rmin,rmin]),overall_array,np.array([rmax,rmax,rmax]))))

        return overall_array

    def knots_map_trio_hybrid(average_ij,average_ik,average_jk,width_ij,width_jk,num_ij_1,num_ij_2,num_ij_3,num_ik_1,num_ik_2,num_ik_3,num_jk_1,num_jk_2,num_jk_3,rmin,rmax1,rmax2):
        array_ij=knots_map_pair_hybrid(average_ij,width_ij,num_ij_1,num_ij_2,num_ij_3,rmin,rmax1)
        array_ik=knots_map_pair_hybrid(average_ik,width_ij,num_ik_1,num_ik_2,num_ik_3,rmin,rmax1)
        array_jk=knots_map_pair_hybrid(average_jk,width_jk,num_jk_1,num_jk_2,num_jk_3,rmin,rmax2)
        array_ijk=[array_ij,array_ik,array_jk]

        return array_ijk

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

    knots_map_hybrid={}
    knots_map_hybrid[('Si','Si')]=knots_map_pair_hybrid(average=2.37,width=width2,num1=Si_Si_2_1,num2=Si_Si_2_2,num3=Si_Si_2_3,rmin=rmin2,rmax=rmax2)
    knots_map_hybrid[('C','C')]=knots_map_pair_hybrid(average=1.41,width=width2,num1=C_C_2_1,num2=C_C_2_2,num3=C_C_2_3,rmin=rmin2,rmax=rmax2)
    knots_map_hybrid[('C','Si')]=knots_map_pair_hybrid(average=1.88,width=width2,num1=C_Si_2_1,num2=C_Si_2_2,num3=C_Si_2_3,rmin=rmin2,rmax=rmax2)
    knots_map_hybrid[('C','C','C')]=knots_map_trio_hybrid(average_ij=1.41,average_ik=1.41,average_jk=1.41,width_ij=width_ij,width_jk=width_jk,num_ij_1=C_C_3_1_1,num_ij_2=C_C_3_1_2,num_ij_3=C_C_3_1_3,num_ik_1=C_C_3_1_1,num_ik_2=C_C_3_1_2,num_ik_3=C_C_3_1_3,num_jk_1=C_C_3_2_1,num_jk_2=C_C_3_2_2,num_jk_3=C_C_3_2_3,rmin=rmin3,rmax1=rmax3_1,rmax2=rmax3_2)
    knots_map_hybrid[('C','C','Si')]=knots_map_trio_hybrid(average_ij=1.41,average_ik=1.88,average_jk=1.88,width_ij=width_ij,width_jk=width_jk,num_ij_1=C_C_3_1_1,num_ij_2=C_C_3_1_2,num_ij_3=C_C_3_1_3,num_ik_1=C_Si_3_1_1,num_ik_2=C_Si_3_1_2,num_ik_3=C_Si_3_1_3,num_jk_1=C_Si_3_2_1,num_jk_2=C_Si_3_2_2,num_jk_3=C_Si_3_2_3,rmin=rmin3,rmax1=rmax3_1,rmax2=rmax3_2)
    knots_map_hybrid[('C','Si','Si')]=knots_map_trio_hybrid(average_ij=1.88,average_ik=1.88,average_jk=2.37,width_ij=width_ij,width_jk=width_jk,num_ij_1=C_Si_3_1_1,num_ij_2=C_Si_3_1_2,num_ij_3=C_Si_3_1_3,num_ik_1=C_Si_3_1_1,num_ik_2=C_Si_3_1_2,num_ik_3=C_Si_3_1_3,num_jk_1=Si_Si_3_2_1,num_jk_2=Si_Si_3_2_2,num_jk_3=Si_Si_3_2_3,rmin=rmin3,rmax1=rmax3_1,rmax2=rmax3_2)
    knots_map_hybrid[('Si','Si','Si')]=knots_map_trio_hybrid(average_ij=2.37,average_ik=2.37,average_jk=2.37,width_ij=width_ij,width_jk=width_jk,num_ij_1=Si_Si_3_1_1,num_ij_2=Si_Si_3_1_2,num_ij_3=Si_Si_3_1_3,num_ik_1=Si_Si_3_1_1,num_ik_2=Si_Si_3_1_2,num_ik_3=Si_Si_3_1_3,num_jk_1=Si_Si_3_2_1,num_jk_2=Si_Si_3_2_2,num_jk_3=Si_Si_3_2_3,rmin=rmin3,rmax1=rmax3_1,rmax2=rmax3_2)
    knots_map_hybrid[('Si','C','C')]=knots_map_trio_hybrid(average_ij=1.88,average_ik=1.88,average_jk=1.41,width_ij=width_ij,width_jk=width_jk,num_ij_1=C_Si_3_1_1,num_ij_2=C_Si_3_1_2,num_ij_3=C_Si_3_1_3,num_ik_1=C_Si_3_1_1,num_ik_2=C_Si_3_1_2,num_ik_3=C_Si_3_1_3,num_jk_1=C_C_3_2_1,num_jk_2=C_C_3_2_2,num_jk_3=C_C_3_2_3,rmin=rmin3,rmax1=rmax3_1,rmax2=rmax3_2)
    knots_map_hybrid[('Si','C','Si')]=knots_map_trio_hybrid(average_ij=1.88,average_ik=2.37,average_jk=1.88,width_ij=width_ij,width_jk=width_jk,num_ij_1=C_Si_3_1_1,num_ij_2=C_Si_3_1_2,num_ij_3=C_Si_3_1_3,num_ik_1=Si_Si_3_1_1,num_ik_2=Si_Si_3_1_2,num_ik_3=Si_Si_3_1_3,num_jk_1=C_Si_3_2_1,num_jk_2=C_Si_3_2_2,num_jk_3=C_Si_3_2_3,rmin=rmin3,rmax1=rmax3_1,rmax2=rmax3_2)
    bspline_config = bspline.BSplineBasis(chemical_system=chemical_system,knots_map=knots_map_hybrid) 

    analyzer = analyze.DataAnalyzer(chemical_system, 
                                r_cut=10.0,
                                bins=0.01)
    atoms_key = 'geometry'
    histogram_slice = np.random.choice(np.arange(len(df_data)),
                                   min(1000, len(df_data)),
                                   replace=False)
    df_slice = df_data[atoms_key].iloc[histogram_slice]
    analyzer.load_entries(df_slice)
    analysis = analyzer.analyze()
    del df_data
    slurmarrayid=int(sys.argv[1])
    begin=(slurmarrayid-1)*100
    end=(slurmarrayid)*100
    if end>len(combinations):
        end=len(combinations)
    holdoutfilename = '/home/michaelmacisaac/blue_subhash/michaelmacisaac/SiC/uf3pots/featurized_data/feat_29/features.h5'
    for dir in range(begin,end):
        input_datas = {
                'analysis':analysis,
                'dirname':dir,
                'ridge1':ridge1s[dir],
                'ridge2':ridge2s[dir],
                'ridge3':ridge3s[dir],
                'curve2':curve2s[dir],
                'curve3':curve3s[dir],
                'weight':weights[dir],
                'sample_weights':sample_weights,
                'bspline_config':bspline_config,
                'filename':f'/blue/subhash/michaelmacisaac/SiC/uf3pots/featurized_data/feat_29/npz_files/gram_ordinate_w{str(np.round(weights[dir],2)).replace(".","_")}.npz',
                'holdoutfilename':holdoutfilename,
                'training_keys':training_keys,
                'holdout_keys':holdout_keys,
                'uniform_knots':False,
                'lammps_path':'/home/michaelmacisaac/lammps-2Jun2022/build/lmp_mpi',
                'lammps_run':True
                        }
        pot_dir=os.path.join(os.getcwd(),'data',f'G{dir}')
        jsonfile=os.path.join(pot_dir,'output.json')
        if os.path.exists(jsonfile):
            continue
        else:
            if os.path.exists(pot_dir):
                shutil.rmtree(pot_dir)
                pass
            else:
                pass
        task_function(input_data=input_datas)
        

if __name__ == '__main__':
    main()

