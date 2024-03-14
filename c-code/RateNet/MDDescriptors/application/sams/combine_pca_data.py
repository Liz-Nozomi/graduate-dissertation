# -*- coding: utf-8 -*-
"""
pca.py
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT NUMPY
import numpy as np
## IMPORT LOAD AND SAVE PKL FUNCTIONS
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path

##############################################################################
# FUNCTIONS AND CLASSES
############################################################################## 
## FUNCTION TO READ WHAM CSV
def read_csv( csv_file ):
    r'''
    Function to read WHAM csv output file
    '''
    ## OPEN FILE AND READ LINES
    with open( csv_file ) as raw_data:
        data = raw_data.readlines()
    ## REMOVE FIRST THREE LINES
    data = data[3].split(',')[-1].strip()
    return float(data)
    
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\figure_pca\raw_data"   
    ## MAIN DIRECTORY
    main_dir_X = r"R:\simulations\polar_sams\unbiased\sample1"
    main_dir_y = r"R:\simulations\polar_sams\indus\sample1"
    ## FILES
    data_file_X = r"output_files/sam_prod_triplet_distribution.pkl"
    data_file_y = r"output_files/sam_wham_reweighted.csv"
    paths_to_files_X = {
                        # "CH3"         : "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                        # "NH2-k0.0"    : "sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",                           
                        # "NH2-k0.1"    : "sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",
                        # "NH2-k0.2"    : "sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",
                        # "NH2-k0.3"    : "sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",
                        # "NH2-k0.4"    : "sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",
                        # "NH2-k0.5"    : "sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",
                        # "NH2-k0.6"    : "sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",
                        # "NH2-k0.7"    : "sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",
                        # "NH2-k0.8"    : "sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",
                        # "NH2-k0.9"    : "sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",
                        # "NH2-k1.0"    : "sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
                        # "NH2-m0.25"   : "sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                        # "NH2-m0.40"   : "sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
                        # "NH2-m0.50"   : "sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                        # "NH2-m0.75"   : "sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                        # "NH2-s0.25"   : "sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                        # "NH2-s0.40"   : "sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",
                        # "NH2-s0.50"   : "sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                        # "NH2-s0.75"   : "sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.0"  : "sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36",                           
                        # "CONH2-k0.1"  : "sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.2"  : "sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.3"  : "sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.4"  : "sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.5"  : "sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.6"  : "sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.7"  : "sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.8"  : "sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36",
                        # "CONH2-k0.9"  : "sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36",
                        # "CONH2-k1.0"  : "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                        # "CONH2-m0.25" : "sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                        # "CONH2-m0.40" : "sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
                        # "CONH2-m0.50" : "sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                        # "CONH2-m0.75" : "sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                        # "CONH2-s0.25" : "sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                        # "CONH2-s0.40" : "sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",
                        # "CONH2-s0.50" : "sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                        # "CONH2-s0.75" : "sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                        # "OH-k0.0"     : "sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",                           
                        # "OH-k0.1"     : "sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",
                        # "OH-k0.2"     : "sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",
                        # "OH-k0.3"     : "sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",
                        # "OH-k0.4"     : "sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",
                        # "OH-k0.5"     : "sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",
                        # "OH-k0.6"     : "sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",
                        # "OH-k0.7"     : "sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",
                        # "OH-k0.8"     : "sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",
                        # "OH-k0.9"     : "sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",
                        # "OH-k1.0"     : "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                        # "OH-m0.25"    : "sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                        # "OH-m0.40"    : "sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",
                        # "OH-m0.50"    : "sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                        # "OH-m0.75"    : "sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                        # "OH-s0.25"    : "sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                        # "OH-s0.40"    : "sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",
                        # "OH-s0.50"    : "sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                        # "OH-s0.75"    : "sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                      }
    paths_to_files_y = {
                        # "CH3"         : "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.0"    : "sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36_2x2x0.3nm",                           
                        # "NH2-k0.1"    : "sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.2"    : "sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.3"    : "sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.4"    : "sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.5"    : "sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.6"    : "sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.7"    : "sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.8"    : "sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k0.9"    : "sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-k1.0"    : "sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-m0.25"   : "sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-m0.40"   : "sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-m0.50"   : "sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-m0.75"   : "sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-s0.25"   : "sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-s0.40"   : "sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-s0.50"   : "sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "NH2-s0.75"   : "sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.0"  : "sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36_2x2x0.3nm",                           
                        # "CONH2-k0.1"  : "sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.2"  : "sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.3"  : "sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.4"  : "sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.5"  : "sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.6"  : "sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.7"  : "sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.8"  : "sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k0.9"  : "sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-k1.0"  : "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-m0.25" : "sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-m0.40" : "sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-m0.50" : "sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-m0.75" : "sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-s0.25" : "sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-s0.40" : "sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-s0.50" : "sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "CONH2-s0.75" : "sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.0"     : "sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36_2x2x0.3nm",                           
                        # "OH-k0.1"     : "sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.2"     : "sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.3"     : "sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.4"     : "sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.5"     : "sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.6"     : "sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.7"     : "sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.8"     : "sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k0.9"     : "sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-k1.0"     : "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-m0.25"    : "sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-m0.40"    : "sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-m0.50"    : "sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-m0.75"    : "sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-s0.25"    : "sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-s0.40"    : "sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-s0.50"    : "sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36_2x2x0.3nm",
                        # "OH-s0.75"    : "sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36_2x2x0.3nm",
                      }
    ## LOOP THROUGH DIRECTORIES TO GATHER DATA 
    data = {}
    for ii, key in enumerate(paths_to_files_X.keys()):
        path_dir = paths_to_files_X[key]
        path_to_file = check_server_path( os.path.join( main_dir_X, path_dir, data_file_X ) )
        loaded_data = load_pkl( path_to_file )
        data[key] = loaded_data
    save_pkl( data, os.path.join( output_dir, r"triplet_distribution_pca_input_oh.pkl" ) )

    data = {}
    for ii, key in enumerate(paths_to_files_y.keys()):
        path_dir = paths_to_files_y[key]
        path_to_file = check_server_path( os.path.join( main_dir_y, path_dir, data_file_y ) )
        loaded_data = read_csv( path_to_file )
        data[key] = loaded_data
    save_pkl( data, os.path.join( output_dir, r"hydration_fe_pca_input_oh.pkl" ) )