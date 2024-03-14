#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loop_traj.py
The purpose of this function is to loop through multiple directories and 
run analysis tools for them. 

Written by: Alex K. Chew (alexkchew@gmail.com, 02/01/2020)

"""
### IMPORTING MODULES
import numpy as np
import pandas as pd
import os
import mdtraj as md
import glob


## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path, check_dir

## FUNCTIONS AVAILABLE
from MDDescriptors.application.np_hydrophobicity.debug_bundling_with_frozen_groups import main_compute_bundles
from MDDescriptors.application.np_hydrophobicity.compute_rmsf_of_ligands import main_compute_gmx_rmsf
from MDDescriptors.application.np_hydrophobicity.compute_wc_grid_for_multiple_times import main_compute_wc_grid_multiple_subset_frames

## DELETION OF FILES
from MDDescriptors.core.delete_files import remove_all_hashtag_files

## PICKLING RESULTS
from MDDescriptors.core.pickle_tools import pickle_results

## IMPORTING FUNCTIONS
from MDDescriptors.traj_tools.loop_traj import loop_traj

## DEFINING REWRITE
rewrite = True    


#%% MAIN SCRIPT
if __name__ == "__main__":
        
    ## DEFINIGN MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
    
    ## DEFINING SIMULATION LIST
    simulation_list = [
            "NP_HYDRO_SI_GNP_DEBUGGING_SPRING",
            "NP_HYDRO_SI_PLANAR_SAMS_DEBUGGING_SPRING",
            ]
    
    ## SEEING IF SIMULATIONS ARE PLANAR SAMS
    planar_sims_list = ["NP_HYDRO_SI_PLANAR_SAMS_DEBUGGING_SPRING"]
    
    ## LOOPING
    for simulation_dir in simulation_list:
        if simulation_dir in planar_sims_list:
            ## DEFINING IF PLANAR SIM
            isplanar = True
        else:
            isplanar = False
    
        ## DEFINING PATH TO LOOP ON
        path_to_main_dir = os.path.join(main_dir,
                                        simulation_dir)
        
        ### DEFINING ANALYSIS FOLDER
        analysis_folder = "analysis"
        
        ## DEFINING PICKLE NAME
        pickle_name="results.pickle"
        
        # True if you want to rewrite
        
        ## CREATING EMPTY LISTS
        function_list, function_input_list = [], []
        
        ## DEFINING JOB_TYPES
        job_type_list = [ 'ligand_rmsf', 'wc_interface_multiple'] # , 'wc_interface_multiple', 'bundling', 
        #  'bundling' 
        
        ## REMOVING BUNDLING FOR NOW
        if isplanar is True:
            if 'bundling' in job_type_list:
                job_type_list.remove('bundling')
        
        ## ADDING BUNDLING
        if 'bundling' in job_type_list:
            ## DEFINING FUNCTIONS
            current_func = main_compute_bundles
            
            ## DEFINING INPUTS 
            func_inputs = {
                    "input_prefix" : "sam_prod",
                    }
            
            ## APPENDING
            function_list.append(current_func)
            function_input_list.append(func_inputs)
            
        ## ADDING RMSF  
        if 'ligand_rmsf' in job_type_list:
            ## DEFINING FUNCTIONS
            current_func = main_compute_gmx_rmsf
            
            ## DEFINING INPUTS 
            func_inputs = {
                    "input_prefix" : "sam_prod",
                    "rewrite" : rewrite,
                    'first_frame': 2000,
                    'last_frame': 12000,
                    }
            if isplanar is True:
                ## INCLUDING GOLD RESIDUE NAME
                func_inputs["center_residue_name"] = "AUI"
                ## INCLUDING SPECIFIC DETAILS
                rmsf_inputs={"input_mdp_file": "nvt_double_prod_gmx5_charmm36_frozen_gold_only.mdp"
                             }
                func_inputs['rmsf_inputs'] = rmsf_inputs
            
            ## APPENDING
            function_list.append(current_func)
            function_input_list.append(func_inputs)
        
        ## ADDING WC INTERFACE
        if "wc_interface_multiple" in job_type_list:
            ## CURRENT FUNCTIONS
            current_func = main_compute_wc_grid_multiple_subset_frames
            
            ## DEFINING trjconv funcs
            trjconv_funcs = {
                    'first_frame': 2000,
                    'last_frame': 12000,
                    }
            
            ## DEFINING INPUTS 
            func_inputs = {
                    "input_prefix" : "sam_prod",
                    'num_frames_iterated': 5000, # 1000,
                    'n_procs': 20,
                    'out_path': None, # This code will update the output path
                    'rewrite': rewrite,
                    'alpha': 0.24,
                    'mesh': [0.1, 0.1, 0.1],
                    'contour': 26,
                    'func_inputs': trjconv_funcs,
                    }
            
            ## APPENDING
            function_list.append(current_func)
            function_input_list.append(func_inputs)
            
            
        ## DEFINING INPUTS TO TRAJ
        inputs = {
                'path_to_main_dir': path_to_main_dir,
                'function_list': function_list,
                'function_input_list' : function_input_list,
                'analysis_folder': analysis_folder,
                'rewrite': rewrite,
                }
        
        ## RUNNING FUNCTION
        traj_analysis = loop_traj(**inputs)
        
        ## RUNNING FUNCTION
        traj_analysis.run()
    