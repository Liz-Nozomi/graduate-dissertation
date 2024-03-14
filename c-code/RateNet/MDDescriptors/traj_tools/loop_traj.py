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
#from MDDescriptors.application.np_hydrophobicity.debug_bundling_with_frozen_groups import main_compute_bundles
#from MDDescriptors.application.np_hydrophobicity.compute_rmsf_of_ligands import main_compute_gmx_rmsf
#from MDDescriptors.application.np_hydrophobicity.compute_wc_grid_for_multiple_times import main_compute_wc_grid_multiple_subset_frames

## DELETION OF FILES
from MDDescriptors.core.delete_files import remove_all_hashtag_files

## PICKLING RESULTS
from MDDescriptors.core.pickle_tools import pickle_results


### FUNCTION TO LOOP TRAJ
class loop_traj:
    '''
    The purpose of this function is to loop through multiple trajectories 
    and run the same function. The main idea is that the function could 
    run gmx trjconv (if necessary), then apply python code to analyze the system. 
    This trajectory function will take the output class and store it within an 
    "analysis" folder. By doing so, we could extract multiple trajectories 
    at a time. 
    INPUTS:
        path_to_main_dir: [str]
            path to the main directory. If this is a list, we will skip the 
            searching of the main directories and run this list. 
        function_list: [list]
            list of functions
        function_input_list: [list]
            function input list
        analysis_folder: [str]
            list of analysis folders
        rewrite: [logical]
            True if you want to rewrite analysis
        remove_hashtag_files: [logical, default = False]
            True if you want to remove all hashtag files after each step. This 
            is useful if you want to clean your directories.
    FUNCTIONS:
        get_dir_list:
            gets list of directories
        run: 
            loops through function list and runs it 
    '''
    ## INITIALIZING
    def __init__(self,
                 path_to_main_dir,
                 function_list,
                 function_input_list,
                 analysis_folder = "analysis",
                 pickle_name = 'results.pickle',
                 rewrite = False,
                 remove_hashtag_files = False
                 ):
        ## STORING INITIAL VARIABLES
        self.path_to_main_dir = path_to_main_dir
        self.function_list = function_list
        self.function_input_list = function_input_list
        self.analysis_folder = analysis_folder
        self.pickle_name = pickle_name
        self.rewrite = rewrite
        self.remove_hashtag_files = remove_hashtag_files
    
        ## GETTING LIST OF DIRECTORIES
        if type(self.path_to_main_dir) is str:
            self.list_of_dir = self.get_dir_list()
        else:
            self.list_of_dir = self.path_to_main_dir
        return

    ### FUNCTION TO GET LIST OF DIRECTORIES
    def get_dir_list(self,):
        '''
        This function gets the main directory listing. 
        INPUTS:
            self:
                class object
        OUTPUTS:
            list_of_dir: [list]
                list of directories
        '''
        
        ## FINDING ALL FILES
        list_of_files = glob.glob(self.path_to_main_dir + "/*")
        ## GETTING ONLY DIRECTORIES
        list_of_dir = [ each_file for each_file in list_of_files 
                                  if os.path.isdir(each_file) ]
        
        return list_of_dir
    
    ### FUNCTION TO LOOP THROUGH ALL SIMULATION TRAJECTORIES AND RUN THE COMMAND
    def run(self, rewrite = False):
        '''
        The purpose of this function is to run each functoin for a path, then
        output the results into a pickle.
        INPUTS:
            self:
                class object
            rewrite: [logical]
                True if you want to overwrite pickles
        OUTPUTS:
            pickle file for each class
        '''
        
        ## LOOPING THROUGH EACH DIRECTORY AND RUNNING
        for idx_dir,path_to_sim in enumerate(self.list_of_dir):
            
            ## LOOPING THROUGH EACH FUNCTION
            for idx, current_func in enumerate(self.function_list):
                ## DEFINING FUNCTION INPTUS
                func_inputs = self.function_input_list[idx].copy()
                ## ADDING PATH TO FILE
                func_inputs['path_to_sim'] = path_to_sim
                
                ## DEFINING PATH
                path_to_storage_folder = os.path.join(path_to_sim,
                                                      self.analysis_folder,
                                                      current_func.__name__)
                
                ## CHECK IF KEY IS INSIDE
                if 'out_path' in func_inputs.keys():
                    ## UPDATING THE PATH
                    func_inputs['out_path'] = path_to_storage_folder
                    
                ## CHECKING IF VARYING VARIABLES IS PRESENT
                if 'VARYING_VARIABLES' in func_inputs.keys():
                    ## GETTING VARIABLES
                    varying_variables_keys = func_inputs['VARYING_VARIABLES'].keys()
                    current_vars = {each_key: func_inputs['VARYING_VARIABLES'][each_key][idx_dir] 
                                    for each_key in varying_variables_keys}
                    ## REMOVING FROM INPUTS
                    func_inputs.pop('VARYING_VARIABLES', None)
                    ## ADDING THE VARS INTO INPUTS
                    func_inputs = {**func_inputs, **current_vars}
                
                ## CHECKING DIRECTORY
                check_dir(path_to_storage_folder)
                
                ## DEFINING PATH TO PICKLE
                path_pickle = os.path.join(path_to_storage_folder,
                                           self.pickle_name)
                
                ## CHECKING PATH
                if os.path.isfile(path_pickle) is False or self.rewrite is True:
                
                    ## RUNNING FUNCTION
                    results = current_func(**func_inputs)
            
                    ## STORING THE RESULTS
                    pickle_results(results = results,
                                   pickle_path = path_pickle,
                                   verbose = True)
                else:
                    print("Job is complete!")
                    print("Pickle path is: %s"%(path_pickle) )
                    
            ## AT THE END, REMOVE HASHTAGS IF NECESSARY
            if self.remove_hashtag_files is True:
                remove_all_hashtag_files(wd = path_to_sim)
        
        return
    
#%% MAIN SCRIPT
if __name__ == "__main__":
        
    ## DEFINIGN MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
    
    ### DIRECTORY TO WORK ON    
    # simulation_dir=r"20200213-Debugging_GNP_spring_constants_heavy_atoms_more_springs"
    simulation_dir= r"20200217-planar_SAM_frozen_debug"
    # r"20200212-Debugging_GNP"
    # r"20200217-planar_SAM_frozen_debug"
    # r"20200213-Debugging_GNP_spring_constants_heavy_atoms_more_springs"
    # r"20200212-Debugging_GNP_spring_constants_heavy_atoms"
    # r"20200129-Debugging_GNP_spring_constants_heavy_atoms"
    
    ## DEFINING IF PLANAR SIM
    isplanar = True
    # True
    
    ## DEFINING PATH TO LOOP ON
    path_to_main_dir = os.path.join(main_dir,
                                    simulation_dir)
    
    ### DEFINING ANALYSIS FOLDER
    analysis_folder = "analysis"
    
    ## DEFINING PICKLE NAME
    pickle_name="results.pickle"
    
    ## DEFINING REWRITE
    rewrite = False
    # True if you want to rewrite
    
    ## CREATING EMPTY LISTS
    function_list, function_input_list = [], []
    
    ## DEFINING JOB_TYPES
    job_type_list = [ 'ligand_rmsf', 'wc_interface_multiple', 'bundling' ] # , 'wc_interface_multiple', 'bundling', 
    
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
        
        ## DEFINING INPUTS 
        func_inputs = {
                "input_prefix" : "sam_prod",
                'num_frames_iterated': 1000,
                'n_procs': 20,
                'out_path': None, # This code will update the output path
                'rewrite': False,
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
            }
    
    ## RUNNING FUNCTION
    traj_analysis = loop_traj(**inputs)
    
    #%%
    ## RUNNING FUNCTION
    traj_analysis.run()
    
#    
#    #%%
#        
#    
#    
#
#    
#    ## FINDING ALL FILES
#    list_of_files = glob.glob(path_to_main_dir + "/*")
#    ## GETTING ONLY DIRECTORIES
#    list_of_dir = [ each_file for each_file in list_of_files 
#                              if os.path.isdir(each_file) ]
#    
#    ## LOOPING THROUGH EACH DIRECTORY AND RUNNING
#    for path_to_sim in list_of_dir:
#        ## ADDING PATH TO FILE
#        func_inputs['path_to_sim'] = path_to_sim
#        
#        ## DEFINING PATH
#        path_to_storage_folder = os.path.join(path_to_sim,
#                                              analysis_folder,
#                                              current_func.__name__)
#        ## CHECKING DIRECTORY
#        check_dir(path_to_storage_folder)
#        
#        ## DEFINING PATH TO PICKLE
#        path_pickle = os.path.join(path_to_storage_folder,
#                                   pickle_name)
#        
#        ## CHECKING PATH
#        if os.path.isfile(path_pickle) is False or rewrite is True:
#        
#            ## RUNNING FUNCTION
#            results = current_func(**func_inputs)
#    
#            ## STORING THE RESULTS
#            pickle_results(results = results,
#                           pickle_path = path_pickle,
#                           verbose = True)
#        else:
#            print("Job is complete!")
#            print("Pickle path is: %s"%(path_pickle) )

    
    