#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loop_traj_np_descriptors.py
This function loops through multiple trajectories and runs np descriptors

Written by: Alex K. Chew (03/31/2020)

USAGE:
    python3.6 /home/akchew/bin/pythonfiles/modules/MDDescriptors/application/np_descriptors/loop_traj_np_descriptors.py

"""
import os
import glob

## DEFINING NPROCS
NPROCS=20
REWRITE=True # True if you want to rewrite

## IMPORTING FUNCTIONS
from MDDescriptors.traj_tools.loop_traj import loop_traj

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_descriptors.global_vars import SIM_DICT, PARENT_SIM_PATH


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING PARENT DIR
    parent_sim_path = PARENT_SIM_PATH
    
    ## DEFINING EMPTY DIRECTORY
    path_main_list = []
    
    dir_list_type = ['ROT_WATER_SIMS',
                     'ROT_DMSO_SIMS']
    
    for dir_type in dir_list_type:
    
        ## GETTING DIRECTORY
        directory = SIM_DICT[dir_type]
        
        ## DEFINING PATH
        path_to_dir = os.path.join(parent_sim_path,
                                   directory)
    
        ## GETTING LIST
        path_main_list.extend(glob.glob(path_to_dir + "/*"))
        
    print(path_main_list)
    
    ## CREATING EMPTY LISTS
    function_list, function_input_list = [], []
    
    ## DEFINING JOB_TYPES
    job_type_list = [
                     'compute_gmx_sasa'
                     ] 
    
    ###########################
    #### DEFAULT VARIABLES ####
    ###########################
    
    ### DEFINING ANALYSIS FOLDER
    analysis_folder = "analysis"
    
    ## DEFINING PICKLE NAME
    pickle_name="results.pickle"
    
    ## COMPUTING DENSITIES FUNCTION
    if 'compute_gmx_sasa' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_descriptors.compute_np_sasa import main_compute_np_sasa
        
        ## DEFINING FUNCTIONS
        current_func = main_compute_np_sasa
        
        ## DEFINING INPUTS FOR MAIN FUNCTION
        func_inputs = {
                'input_prefix': 'sam_prod',
                'rewrite': REWRITE,
                }
        
        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(func_inputs)
    
    ####### RUNNING MD DESCRIPTORS #######
    ## DEFINING INPUTS TO TRAJ
    inputs = {
            'path_to_main_dir': path_main_list,
            'function_list': function_list,
            'function_input_list' : function_input_list,
            'analysis_folder': analysis_folder,
            'pickle_name': pickle_name,
            'rewrite': REWRITE, # True if you want to rewrite
            'remove_hashtag_files': REWRITE, # True if you want to remove all hashtag files
            }
    
    ## RUNNING FUNCTION
    traj_analysis = loop_traj(**inputs)
    
    ## RUNNING FUNCTION
    traj_analysis.run()
