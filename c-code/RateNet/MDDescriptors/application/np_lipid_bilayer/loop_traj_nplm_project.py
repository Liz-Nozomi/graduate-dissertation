#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loop_traj_nplm_project.py
This function runs loop trajectories for nplm project

Written by: Alex K. Chew (02/15/2020)
"""
import os
import glob

## IMPORTING ALL FUNCTIONS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
        NPLM_SIM_DICT, PARENT_SIM_PATH, nplm_job_types

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path, check_dir

## IMPORTING FUNCTIONS
from MDDescriptors.traj_tools.loop_traj import loop_traj

## DEFINING NPROCS
NPROCS=20
REWRITE=False # True if you want to rewrite


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\nplm_sims")
    
    ## DEFINING SPECIFIC SIMS TO RUN
    sim_list_for_dict=[
#             'us_forward_R12',
#             'us_forward_R01',
#             'us_reverse_R12',
#             'us_reverse_R01',
             'modified_FF_unbiased_after_us_ROT017_1.900nm',
             'modified_FF_us_forward_R17'
#             'unbiased_ROT001_4.700',
#            
#            'unbiased_ROT012_1.300',
#            'unbiased_ROT012_1.900',
#            'unbiased_ROT012_2.100',

#            'unbiased_ROT001_1.700',
#            'unbiased_ROT001_1.900',
#            'unbiased_ROT001_2.100',
#            'unbiased_ROT001_2.300',

#            'unbiased_ROT012_5.300',
#            'unbiased_ROT012_5.300_rev',
            
            
#            'unbiased_ROT012_3.500',
#            'unbiased_ROT001_3.500'
    
#             'pullthenunbias_ROT001',
#             'pullthenunbias_ROT012',
            ]
    
    ## DEFINING IF WANT PART2
    want_part2 = False
    
    if want_part2 is True:
        analysis_folder="analysis_part2"
    else:
        ### DEFINING ANALYSIS FOLDER
        analysis_folder = "analysis"
        
    
    ## GETTING PATH LIST
    path_main_list = []
    
    ## GETTING ITP FILE
    itp_file_list = []
    
    ## LOOPING THROUGH EACH SIM
    for each_sim in sim_list_for_dict:
        ## GETTING MAIN SIM
        main_sim_dir = NPLM_SIM_DICT[each_sim]['main_sim_dir']
        specific_sim = NPLM_SIM_DICT[each_sim]['specific_sim']
        
        ## GETTING ALL JOB TYPES
        job_types = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                   main_sim_dir = main_sim_dir,
                                   specific_sim = specific_sim,)
        
        ## DEFINING CONFIG LIBRARY
        config_library = job_types.config_library
        
        if want_part2 is True:
            config_library=[job_types.config_library[job_types.config_library.index('5.100')]]
        
        ## LOOPING THROUGH CONFIG LIBRARY
        for idx, specific_config in enumerate(config_library):
            ## PATH TO ANALYSIS
            path_to_sim = os.path.join(job_types.path_simulations,
                                       specific_config,
                                       )
        
            ## APPENDING
            path_main_list.append(path_to_sim)
            
            ## APPENDING
            itp_file_list.append(job_types.np_itp_file)
    
    
    #%%

    ## DEFINING PICKLE NAME
    pickle_name="results.pickle"
        
    ## RUNNING EACH
    # for path_to_main_dir in path_main_list:

    ## CREATING EMPTY LISTS
    function_list, function_input_list = [], []
    
    ## DEFINING JOB_TYPES
    job_type_list = [
#                     'compute_contacts',
                     'compute_com_distances',
#                     'compute_densities',
#                     'compute_np_intercalation',
#                     'compute_nplm_distances',
                     ] 
    
    ## COMPUTING DENSITIES FUNCTION
    if 'compute_densities' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_lipid_bilayer.compute_nplm_density_maps import main_compute_nplm_densities
        
        ## DEFINING FUNCTIONS
        current_func = main_compute_nplm_densities
        
        ## DEFINING INPUTS
        nplm_densities_input = {
                'bin_width': 0.2,
                'r_range': (0, 6),
                'z_range': (-9, 9),
                'lm_res_name': "DOPC",
                }
        
        ## DEFINING FUNCTION INPUT
        func_inputs = {'input_prefix': 'nplm_prod',
                       'nplm_densities_input': nplm_densities_input,
                       'last_time_ps': 50000,
                       }
        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(func_inputs)
    
    ## COMPUTING DENSITIES FUNCTION
    if 'compute_contacts' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import main_compute_contacts

        ## DEFINING FUNCTIONS
        current_func = main_compute_contacts
        
        ## DEFINING INPUTS
        contacts_input = {
                'cutoff': 0.5,
                'lm_res_name': 'DOPC',
                }
        
        ## DEFINING FUNCTION INPUT
        func_inputs = {'input_prefix': 'nplm_prod',
                       'func_inputs': contacts_input,
                       'last_time_ps': 50000,
                       'selection': 'non-Water',
                       'gro_output_time_ps': 0,
                       'n_procs': NPROCS,
                       'rewrite': REWRITE
                       }
        
        if want_part2 is True:
            func_inputs['input_prefix'] = 'nplm_prod_part2'
            func_inputs['gro_output_time_ps'] = 50000
        
        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(func_inputs)
        
    ## RUNNING COM DISTANCES
    if 'compute_com_distances' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_lipid_bilayer.compute_com_distances import main_compute_com_distances

        ## DEFINING FUNCTIONS
        current_func = main_compute_com_distances
        
        ## DEFINING FUNCTION INPUT
        func_inputs = {'input_prefix': 'nplm_prod',
                       'group_1_resname': 'DOPC',
                       'group_2_resname': 'AUNP',
                       'rewrite': REWRITE,
                       }
        
        if want_part2 is True:
            func_inputs['input_prefix'] = 'nplm_prod_part2'
            
        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(func_inputs)
        
        
    ## RUNNING NP-INTERCALATION
    if 'compute_np_intercalation' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_lipid_bilayer.compute_np_intercalation import main_compute_np_intercalation

        ## DEFINING FUNCTIONS
        current_func = main_compute_np_intercalation
        
        ## DEFINING FUNCTION INPUT
        func_inputs = {
                'input_prefix': 'nplm_prod',
                'last_time_ps': 50000,
                'selection': 'non-Water',
                'lm_res_name': 'DOPC',
                'rewrite': REWRITE,
                'VARYING_VARIABLES': {
                        'itp_file':itp_file_list
                        }
                }

        ## APPENDING
        function_list.append(current_func)
        function_input_list.append(func_inputs)
    
    ## COMPUTING NPLM DISTANCES FUNCTION
    if 'compute_nplm_distances' in job_type_list:
        ## IMPORTING FUNCTION
        from MDDescriptors.application.np_lipid_bilayer.compute_contacts_for_plumed import main_compute_nplm_distances

        ## DEFINING FUNCTIONS
        current_func = main_compute_nplm_distances
        
        ## DEFINING FUNCTION INPUT
        func_inputs = {'input_prefix': 'nplm_prod',
                       'lipid_membrane_resname': "DOPC",
                       'last_time_ps': 50000,
                       'selection': 'non-Water',
                       'gro_output_time_ps': 0,
                       'n_procs': NPROCS,
                       'rewrite': REWRITE
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
            