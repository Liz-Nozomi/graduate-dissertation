#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loop_traj_np_hydrophobicity.py

This script loops through multiple trajectories and run analysis on them. 

Written by: Alex K. Chew (04/21/2020)
"""
## IMPORTING MODULES
import os
import glob

### IMPORTING GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, NP_SIM_PATH

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path, check_dir

## IMPORTING FUNCTIONS
from MDDescriptors.traj_tools.loop_traj import loop_traj

## DEFINING NPROCS
NPROCS=20
REWRITE=False # True if you want to rewrite

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DEFINING ANALYSIS FOLDER
    analysis_folder = "analysis"
    
    ## DEFINING PICKLE NAME
    pickle_name="results.pickle"
    
    ## DEFINING MAIN DIRECTORY
    main_dir = NP_SIM_PATH
    
    ## DEFINING PARENT DIRECTORIES
    np_parent_dirs =[
#            "20200420-cosolvent_mapping",
#            "20200420-cosolvent_mapping_dod",
            # '20200422-planar_cosolvent_mapping'
            # "20200427-gnp_cosolvent_mapping_no_formic_acid",
#             "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic"
            # "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_COOH",
            # "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_dod",
            # "20200508-Planar_cosolvent_mapping_sims",
            # "20200508-GNP_cosolvent_mapping_sims",
            #            "20200618-GNP_COSOLVENT_MAPPING",
            "20200625-GNP_cosolvent_mapping_sims_unsat_branched",
            ]
    
    ## DEFINING PLANAR DIRS
    planar_dirs=[
            '20200422-planar_cosolvent_mapping',
            "20200424-planar_cosolvent_mapping_planar_rerun",
            "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic",
            "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_COOH",
            "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_dod",
            "20200508-Planar_cosolvent_mapping_sims",
            ]
    
    ## PATH TO MAIN SIM LIST
    path_main_list = []
    
    ## LOOPING
    for each_parent_list in np_parent_dirs:
        ## GLOBBING ALL POSIBLE CONFIGS
        current_sim_list  = glob.glob( os.path.join(main_dir,
                                                    each_parent_list) + "/*" )
        
        ## APPENDING
        path_main_list.extend(current_sim_list)
        
    ## SORTING LIST
    path_main_list.sort()
    # path_main_list = [path_main_list[0]]
    #%%
    
    
    ## CREATING EMPTY LISTS
    function_list, function_input_list = [], []
    
    ## DEFINING JOB_TYPES
    job_type_list = [
            'compute_np_cosolvent_mapping'
            ]
    
    ## COMPUTING COSOLVENT MAPPING FUNCTION
    if 'compute_np_cosolvent_mapping' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_hydrophobicity.compute_np_cosolvent_mapping import main_compute_np_cosolvent_mapping

        ## DEFINING FUNCTIONS
        current_func = main_compute_np_cosolvent_mapping
        
        ## DEFINING PARENT DIR
        if each_parent_list in planar_dirs:
            parent_wc_folder = "20200403-Planar_SAMs-5nmbox_vac_with_npt_equil"
        else:
            parent_wc_folder = "20200618-Most_likely_GNP_water_sims_FINAL"
        
        ## DEFINING WC INTERFACE LOCATION
        path_to_wc_folder = os.path.join(PARENT_SIM_PATH,
                                         parent_wc_folder)
        
        
    
        np_mapping_inputs = {
                'cutoff': 0.33,
                }
        
        main_np_cosolvent_mapping_inputs={
                # 'path_to_sim': path_to_sim, <-- assumed to be inputted
                'func_inputs': np_mapping_inputs,
                'input_prefix': 'sam_prod',
                'n_procs': NPROCS,
                'path_to_wc_folder': path_to_wc_folder,
                'initial_frame': 2000,
                'final_frame': 12000,
                'rewrite': REWRITE
                }
        
        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(main_np_cosolvent_mapping_inputs)
        
    
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