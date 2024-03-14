#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nplm_generate_list_of_sims.py

This script simply generates a list of simulations to perform analysis on. 
This will be useful in conjunction with bash scripts that need the path 
of every simulation. 

Written by: Alex K. Chew (05/18/2020)

"""

import os
from optparse import OptionParser # Used to allow commands within command line

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types
## CHECK TESTING FUNCTION
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    ## DEFINING SPECIFIC SIMS TO RUN
    
    ### TURNING TEST ON / OFF
    testing = check_testing() # False if you're running this script on command prompt!!!`
    
    ## DEFINING DEFAULT PROD
    default_prefix="nplm_prod"
    
    ## TESTING
    if testing is True:
        ## DEFINING OUTPUT PATH
        output_path=""
    else:
        # Adding options for command line input (e.g. --ligx, etc.)
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## INPUT FOLDER
        parser.add_option("--output_path", 
                          dest="output_path", 
                          action="store", 
                          type="string", 
                          help="Path to job list", default=".")
        
        ### PARSING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## INPUT FILES
        output_path = options.output_path
        
    
    ## GENERATE THROUGH SIMULATION KEYS
    sim_list_for_dict=[each_key for each_key in NPLM_SIM_DICT if each_key.startswith("modified_FF_unbiased_after_us_ROT017_1.900nm") or each_key == "modified_FF_us_forward_R17"] # _20ns_US
    # each_key.startswith("modified_FF_hydrophobic_pmf_unbiased_ROT017")
    # or (each_key.startswith('plumed_unbiased_') and 'ROT012' in each_key)
    # each_key.startswith("plumed_unbiased") or each_key.startswith("unbiased_after_us_") 
    # each_key.startswith("pullthenunbias_") ]
    # each_key.startswith("plumed_unbiased_after_US")]
    # each_key.startswith("us-PLUMED")
    # each_key.startswith("us") or each_key.startswith("unbiased") or each_key.startswith("plumed") or each_key == 'pullthenunbias_ROT012' 
    # if each_key.startswith("us") or 
    
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
        
        ## CHECKING PREFIX
        if 'prefix' in NPLM_SIM_DICT[each_sim]:
            current_prefix = NPLM_SIM_DICT[each_sim]['prefix']
        else:
            current_prefix = default_prefix
        
        ## SEEING IF REFERENCE IS THERE
        if 'reference_for_index' in NPLM_SIM_DICT[each_sim]:
            ## DEFINING REF DICT
            ref_dict = NPLM_SIM_DICT[each_sim]['reference_for_index']
            ## GENERATING PATH
            path_for_ref = os.path.join(PARENT_SIM_PATH,
                                        NPLM_SIM_DICT[ref_dict['sim_key']]['main_sim_dir'],
                                        NPLM_SIM_DICT[ref_dict['sim_key']]['specific_sim'],
                                        ref_dict['prefix'] + ".gro"
                                        )
        else:
            path_for_ref = ""
            
        
        ## LOOPING THROUGH CONFIG LIBRARY
        for idx, specific_config in enumerate(config_library):
            ## PATH TO ANALYSIS
            path_to_sim = os.path.join(job_types.path_simulations,
                                       specific_config,
                                       )
        
            ## APPENDING
            path_main_list.append([path_to_sim, current_prefix, path_for_ref])
    
    ## SORTING
    path_main_list.sort()
    
    ## CHECKING IF EMPTY STRING
    if len(output_path) > 0:
        ## WRITING TO FILE
        with open(output_path, 'w') as f:
            ## LOOPING AND WRITING
            for each_sim in path_main_list:
                f.write("%s\n"%(' '.join(each_sim) ))
        