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
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl
## FUNCTION TO COMPUTE TRIPLET ANGLE DISTRIBUTION
from MDDescriptors.application.sams.triplet_angle import compute_triplet_angle_distribution

##############################################################################
# FUNCTIONS AND CLASSES
############################################################################## 
### FUNCTION TO COMPUTE HBOND TRIPLETS
def compute_triplet_entropy( sim_working_dir = "None",
                             input_prefix    = "None",
                             rewrite         = False,
                             **kwargs ):
    r'''
    Function loads gromacs trajectory and computes triplet angle distribution
    '''
    if sim_working_dir  == "None" \
        or input_prefix == "None":
        print( "ERROR: missing inputs. Check inputs and try again" )
        return 1
    else:
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_triplet_entropy.pkl" )
        path_pkl_triplet_angles = os.path.join( sim_working_dir, r"output_files", input_prefix + "_triplet_distribution.pkl" )
        if rewrite is not True and os.path.exists( path_pkl_triplet_angles ):
            ## LOAD TRIPLET DISTRIBUTION DATA
            triplet_distribution = load_pkl( path_pkl_triplet_angles )
        else:
            ## PREPARE DIRECTORY FOR ANALYSIS
            path_to_sim = check_server_path( sim_working_dir )
            ## GET TRIPLET DISTRIBUTION
            triplet_distribution = compute_triplet_angle_distribution( sim_working_dir = path_to_sim,
                                                                       input_prefix    = input_prefix,
                                                                       rewrite         = rewrite,
                                                                       **kwargs )
        ## COMPUTE NUM HBONDS
        entropy = triplet_entropy( triplet_distribution[:,1] )
        ## SAVE PICKLE FILE
        save_pkl( entropy, path_pkl )
        
## TRIPLET ENTROPY FUNCTION
def triplet_entropy( p_theta ):
    r'''
    Function to compute triplet angle entropy
    '''
    ## REMOVE ZEROS FROM DATA
    p_no_zeros = p_theta[p_theta>0.]
    ## COMPUTE ENTROPY
    entropy = -1. * np.sum( p_no_zeros * np.log( p_no_zeros ) )
    ## RETURN RESULTS
    return entropy
    
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    sim_working_dir = r"R:\simulations\polar_sams\unbiased\sample1\sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36"
    input_prefix = "sam_prod"
    test = compute_triplet_entropy( sim_working_dir = sim_working_dir,
                                    input_prefix = input_prefix )
    
