#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_np_bundling_pickles.py
This script will just fix the nanoparticle bundling pickles so it is readable 
by pymol. I am receiving some error about attribute -- which I think is due to
the way the pickle was stored. 


Written by: Alex K. Chew (04/08/2020)
"""
import os
import pandas as pd
import glob
## IMPORTING MD DESCRIPTOR MOST LIKELY CONFIGURATION
from MDDescriptors.application.nanoparticle.np_most_likely_config import find_most_probable_np
import MDDescriptors.core.pickle_tools as pickle_funcs


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING PATH TO SIM
    path_to_sim = "/Volumes/akchew/scratch/nanoparticle_project/simulations/EAM_COMPLETE"
    
    ## LOOKING FOR ALL OBJECTS
    dirs_list = glob.glob(path_to_sim + "/*")
    
    ## DEFINING SIMULATION
    bundling_dir = "most_likely-avg_heavy_atom"
    
    ## LOOPING
    for dir_of_pickle in dirs_list:
        
        ## PRINTING
        print("Working on %s"%(dir_of_pickle))
        
        ## PICKLE NAME
        pickle_name = "np_most_likely.pickle"
        
        ## DEFINING OUTPUT PICKLE NAME
        output_pickle_name = "grp_assignments.pickle"
        
        ## DEFINING PICKLE PATH
        path_pickle = os.path.join(dir_of_pickle,
                                   bundling_dir,
                                   pickle_name)
        
        ## DEFINING PATH OUTPUT
        path_output_pickle = os.path.join(dir_of_pickle,
                                          bundling_dir,
                                          output_pickle_name)
        
        ## RUNNING FOR PICKLE FILES THAT DO NOT EXIST
        if os.path.exists(path_output_pickle) is False:
            ## LOADING THE PICKLE
            most_probable_np = pd.read_pickle(path_pickle)[0]
            
            ## GETTING RESIDUE NAME
            res_name = most_probable_np.bundling_groups.structure_np.ligand_names[0]
            
            ## GETTING ASSIGNMENTS
            grp_assignments = most_probable_np.bundling_groups.lig_grp_list
            
            ## GETTING LIGAND HEAVY ATOMS
            ligand_atom_index_list = most_probable_np.bundling_groups.structure_np.ligand_atom_index_list
            
            
            ## PRINTING
            pickle_funcs.pickle_results(results = [grp_assignments, res_name, ligand_atom_index_list],
                                        pickle_path = path_output_pickle)
        
        else:
            print("Skipping since the pickle already exists")
        
    
    
    
    