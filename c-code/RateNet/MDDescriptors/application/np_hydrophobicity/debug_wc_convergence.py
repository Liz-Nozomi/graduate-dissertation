#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_wc_convergence.py
The purpose of this script is to run the WC interface for multiple trajectories, 
then see if the WC interface converged afterwards. 

Written by: Alex K. Chew (04/08/2020)

==================
=== Pseudocode ===
==================

INPUTS:
    - input prefix for gro and xtc
    - sampling time list that you would like to check
    - contour level desired

OUTPUTS:
    - pdb / pickle file for each sampling time

ALGORITHM:
    - Use trjconv functions to generate a trajectory that would best match the 
    sampling time required
    - Load the trajectory
    - Compute the grid for each sampling time instance (avg densities)
    - Generate grid based on contour levels
    - Output PDB, pickle, etc. for the wc interface
    - After looping through each PDB, etc. generate a sampling time plot for 
    z differences.

RUNNING THE CODE:
    python3.6 /home/akchew/bin/pythonfiles/modules/MDDescriptors/application/np_hydrophobicity/debug_wc_convergence.py

"""

## IMPORTING MODULES
import numpy as np
import os
import mdtraj as md


## IMOPRTING CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing, check_path, check_dir

## IMPORTING TRJCONV
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv

## IMPORTING GRIDDING TOOL
from MDDescriptors.surface.willard_chandler_parallel import compute_wc_grid

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## PICKLE TOOL
import MDDescriptors.core.pickle_tools as pickle_tools

## DEFAULT VARIABLES
FOLDER_NAME = "WC_CONVERGENCE"

### FUNCTION TO GENERATE GRO AND XTC FILE
def generate_gro_xtc_for_wc_interface(path_to_sim,
                                      sampling_time):
    '''
    This function generates the gro and xtc fie wc interface.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        sampling_time: [list]
            sampling time list
        
    OUTPUTS:
        output_gro_file: [str]
            output gro file
        output_xtc_file: [str]
            output xtc file
    '''

    ## DESIGNING TRJCONV COMMANDS
    trjconv_commands = convert_with_trjconv(wd = path_to_sim)
        
    ## GETTING MAXIMA
    max_time = np.max(sampling_time)
        
    ## SEEING IF YOU WANT FROM THE END    
    if want_from_end is True:
        ## FINDING TRAJ LENGTH
        traj_length = trjconv_commands.gmx_check_traj_length(file_prefix=input_prefix)
        ## DEFINING TRAJ TIME
        begin_traj_time = traj_length - max_time
        end_traj_time = traj_length
    else:
        begin_traj_time = 0 
        end_traj_time = max_time
    
    ## DEFINING OUTPUT SUFFIX
    output_suffix = "-wcdebug-%d-%d"%(begin_traj_time, end_traj_time)
    
    ## USING TRJCONV TO CREATE OUTPUT
    output_gro_file, output_xtc_file = trjconv_commands.generate_water_heavy_atoms(
                                                                                   input_prefix = input_prefix,
                                                                                   output_suffix = output_suffix,
                                                                                   water_residue_name = 'SOL',
                                                                                   center_residue_name = None,
                                                                                   only_last_ns = False,           # add gmx check option to find total frames
                                                                                   rewrite = False,
                                                                                   first_frame = begin_traj_time,
                                                                                   last_frame = end_traj_time)
    return output_gro_file, output_xtc_file

### FUNCTION TO RUN GRID AND TO GENERATE PICKLES
def debug_wc_sampling_time(path_to_sim,
                           output_xtc_file,
                           output_gro_file,
                           sampling_time):
    '''
    The purpose of this function is to generate sampling time for different 
    willard chandler interfaces. 
    INPUTS:
        path_to_sim; [str]
            path to simulation
        output_xtc_file: [str]
            path to xtc file
        output_gro_file: [str]
            path to gro file
        sampling_time: [list]
            list of sampling times
    OUTPUTS:
        void -- the output is to pickle files
    '''
    ## DEFINING PATH TO XTC AND GRO
    path_xtc = os.path.join(path_to_sim,
                            output_xtc_file)
    path_gro = os.path.join(path_to_sim,
                            output_gro_file)
    
    ## LOADING TRAJECTORY
    print("Loading trajectory:")
    print(" --> Path: %s" %(path_to_sim))
    print(" --> gro file: %s"%(output_gro_file) )
    print(" --> xtc file: %s"%(output_xtc_file) )
    traj = md.load(path_xtc, 
                   top = path_gro,
                   discard_overlapping_frames = True)
    
    ## LOOPING THROUGH EACH SAMPLING TIME
    time_traj = traj.time
    ## NORMALIZING TIME
    normalize_time = time_traj - time_traj[0]
    
    ## TRUNCATING THE TRAJECTORY
    nearest_times = [ calc_tools.find_nearest(array = normalize_time, value = each_time) for each_time in sampling_time ]
    ## DEFINING INDEXES TO GO UP TO
    indices_list = [ near_time[1] for near_time in nearest_times]
    
    ## DEFINING PATH TO OUTPUT FOLDER
    path_to_output_folder = os.path.join(path_to_sim,
                                         FOLDER_NAME)
    
    ## CHECKING FOLDER
    check_dir(path_to_output_folder)
    
    ## LOOPING THROUGH EACH SAMPLING TIME
    for sampling_idx, each_sampling_time in enumerate(sampling_time):
        ## DEFINING INDEX
        index_of_traj = indices_list[sampling_idx]
        
        print("==== Idx %d of %d: Sampling time: %.3f ps; Index: %d ===="%(sampling_idx,
                                                                 len(sampling_time),
                                                                 each_sampling_time,
                                                                 indices_list[sampling_idx]
                                                                 ))
        
        ## DEFINING CURRENT TRAJECTORY
        current_traj = traj[:index_of_traj]
        print(" --> Initial-final time : %d-%d" %(current_traj.time[0],current_traj.time[-1]))
        
        ## DEFINING FILE NAME
        pickle_name = '-'.join([ str(current_traj.time[0]), str(current_traj.time[-1]) ])
        

        ## STORING ALL POSSIBLE DATA
        _, interface, avg_density_field = compute_wc_grid(traj = current_traj, 
                                                          sigma = alpha, 
                                                          mesh = mesh, 
                                                          contour = None, 
                                                          n_procs = n_procs,
                                                          print_freq = 100,
                                                          want_normalize_c = False)
        
        ## DEFINING PATH
        path_to_pickle = os.path.join(path_to_output_folder, pickle_name)
        
        ## STORING INTERFACE AND DENSITY INTO PICKLE
        pickle_tools.pickle_results(results = [interface, avg_density_field],
                                    pickle_path = path_to_pickle,
                                    verbose = True)
    return
    

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
        
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## RUNNING TESTING    
    # if testing == True:
        
    ## DEFINING PATH TO SIM
    path_to_sim = r"/Volumes/shared/np_hydrophobicity_project/simulations/20200403-Planar_SAMs-5nmbox_vac_with_npt_equil_other/NVTspr_50_Planar_300.00_K_C11CONH2_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps"
    
    ## CHECKING PATH
    path_to_sim = check_path(path_to_sim)
    
    ## DEFINING INPUT PREFIX
    input_prefix="sam_prod"
    
    ## DEFINING SAMPLING TIME
    sampling_time=[500, 1000, 3000, 5000]
    # 500, 1000, 2000, 3000, 4000, 5000
    
    ## DEFINING CONTOUR LEVELS
    contours = [26]

    ## SEEING IF YOU WANT FROM END
    want_from_end = True
    
    ## DEFINING MESH
    mesh = WC_DEFAULTS['mesh']
    
    ## DEFINING OUTPUT FILE
    output_file = "output_files"
    
    ## WILLARD-CHANDLER VARIABLES
    alpha = WC_DEFAULTS['alpha']

    ## GETTING INPUT
    n_procs = 10
    
    #####################
    ### MAIN FUNCTION ###
    #####################
    
    ## GETTING GRO AND XTC FILE
    output_gro_file, output_xtc_file = generate_gro_xtc_for_wc_interface(path_to_sim = path_to_sim,
                                                                         sampling_time = sampling_time)
    
    ## DEBUGGING WC INTERFACE
    debug_wc_sampling_time(path_to_sim = path_to_sim,
                           output_xtc_file = output_xtc_file,
                           output_gro_file = output_gro_file,
                           sampling_time =sampling_time )
    
