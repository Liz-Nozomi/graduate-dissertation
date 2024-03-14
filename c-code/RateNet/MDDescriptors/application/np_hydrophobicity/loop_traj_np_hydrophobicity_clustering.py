#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loop_traj_np_hydrophobicity_clustering.py

This script loops through multiple trajectories and run analysis on them. 

Written by: Alex K. Chew (06/16/2020)

## RUNNING CODE
python3.6 /home/akchew/bin/pythonfiles/modules/MDDescriptors/application/np_hydrophobicity/loop_traj_np_hydrophobicity_clustering.py
"""
## IMPORTING MODULES
import os
import glob

### IMPORTING GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, \
    NP_SIM_PATH, PATH_DICT

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path, check_dir

## IMPORTING FUNCTIONS
from MDDescriptors.traj_tools.loop_traj import loop_traj

## DEFINING NPROCS
NPROCS=20
REWRITE=True # True if you want to rewrite

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DEFINING ANALYSIS FOLDER
    analysis_folder = "analysis"
    
    ## DEFINING PICKLE NAME
    pickle_name="results.pickle"

    ## DEFINING SIM KEYS
    sim_keys=['GNP', 'GNP_unsaturated', 'GNP_branched', 'GNP_least_likely', ]
    
    ## PATH TO MAIN SIM LIST
    path_main_list = []
    
    ## LOOPING THROUGH EACH AND GENERATING SIM LIST
    for each_key in sim_keys:
        ## DEFINING PATH TO SIM
        parent_sim_path=os.path.join(PARENT_SIM_PATH,
                                 PATH_DICT[each_key],
                                 )
        ## LOOKING INTO LIST
        sim_list = glob.glob(parent_sim_path + "/*")
        
        ## SORTING
        sim_list.sort()
        
        ## EXTENDING
        path_main_list.extend(sim_list)
        
    ## CREATING EMPTY LISTS
    function_list, function_input_list = [], []
    
    ## DEFINING JOB_TYPES
    job_type_list = [
            'compute_mu_clustering'
            ]
    
    ## COMPUTING COSOLVENT MAPPING FUNCTION
    if 'compute_mu_clustering' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_hydrophobicity.analyze_spatial_heterogeniety import main_mu_clustering

        ## DEFINING FUNCTIONS
        current_func = main_mu_clustering
        
        ## DEFINING CLUSTERING INPUTS
        clustering_inputs = {
                'cutoff': 11.25,
                'min_samples': 6, # 5
                'eps': 0.40,
                "clustering_type": "dbscan",
                }
        
        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(clustering_inputs)
        
    
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