# -*- coding: utf-8 -*-
"""
debug_bundling_with_frozen_groups.py
The purpose of this function is to debug the frozen groups found in frozen 
gold simulations. We will quickly compute the number of bundles for a single 
simulation over time, then see how much the bundles actually fluctuate. 

Written by: Alex K. Chew (alexkchew@gmail.com, 01/31/2020)

This module needs hdbscan. To install, load the environment, then:
    pip install hdbscan
"""

### IMPORTING MODULES
import numpy as np
import pandas as pd
import os
import mdtraj as md

## MDDESCRIPTOR MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv, generate_gro_xtc_with_center_AUNP

## GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import LIGAND_RESIDUE_NAMES, MDDESCRIPTORS_DEFAULT_VARIABLES

## IMPROTING BUNDLING GROUPS
from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups, plot_nanoparticle_bundling_groups


#################################################
### CLASS FUNCTION TO COMPUTE BUNDLING GROUPS ###
#################################################
class compute_bundling_grps:
    '''
    The purpose of this function is to compute the bundling groups 
    for the nanoparticle system.
    
    INPUTS:
        traj_data: [obj]
            trajectory data
        itp_file: [str]
            string of the itp file
    '''
    def __init__(self,
                 traj_data,
                 itp_file = 'sam.itp',
                 replace_default_vars_dict = {},
                 ):
        ## LOADING DEFAULT VARIABLES
        self.default_vars = MDDESCRIPTORS_DEFAULT_VARIABLES[calc_nanoparticle_bundling_groups.__name__]
        ## CHANGING ITP FILE
        self.default_vars['itp_file'] = itp_file
        ## REPLACING DEFAULT VARAIBLES
        for keys in replace_default_vars_dict:
            self.default_vars[keys]     = replace_default_vars_dict[keys]
        ## COMPUTING BUNDLING GROUPS
        self.bundling_groups = calc_nanoparticle_bundling_groups( traj_data = traj_data,
                                                                  **self.default_vars )
        return


### MAIN FUNCTION TO COMPUTE BUNDLES
def main_compute_bundles(path_to_sim,
                         input_prefix,
                         output_suffix = None,
                         **bundling_group_kwargs):
    '''
    Main run function to compute bundles. This function will generate 
    gro and xtc based on no gold, and centered. 
    INPUTS:
        path_to_sim: [str]
            main path to simulation
        input_prefix: [str]
            input prefix for gro and xtc
        output_suffix: [str]
            output suffix for gro and xtc. 
    OUTPUTS:
        bundle: [class]
            bundling class
        traj_data: [obj]
            trajectory data
    '''

    ## GETTING TRAJECTORY
    output_gro, output_xtc = generate_gro_xtc_with_center_AUNP(path_to_sim = path_to_sim,
                                                               input_prefix = input_prefix,
                                                               output_suffix = output_suffix,
                                                               rewrite = False,)
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path_to_sim, # Directory to analysis
                                          structure_file = output_gro, # structure file
                                          xtc_file = output_xtc, # trajectories
                                          )
            
    ## DEFINING REPLACE DEFAULT KEYS
    replace_default_vars_dict = {**bundling_group_kwargs}
    
    ## DEFINING INPUTS
    input_vars = {
            'traj_data': traj_data,
            'itp_file': 'sam.itp',
            'replace_default_vars_dict': replace_default_vars_dict,
            }

    ## COMPUTING BUNDLING GROUPS
    bundle = compute_bundling_grps(**input_vars, )
    
    return bundle, traj_data

#%% MAIN SCRIPT
if __name__ == "__main__":
        
    ## DEFINIGN MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
    
    ### DIRECTORY TO WORK ON
    simulation_dir=r"20200129-Debugging_GNP_spring_constants_heavy_atoms"
    
    ## DEFINING SPECIFIC DIR
    specific_dir = r"MostlikelynpNVTspr_25-EAM_300.00_K_2_nmDIAM_C11NH3_CHARMM36jul2017_Trial_1_likelyindex_1"
    specific_dir = r"MostlikelynpNVTspr_25-EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1"
    specific_dir = r"MostlikelynpNVTspr_2000-EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1"
    
    ## DEFINING INPUT GRO AND XTC
    input_prefix = "sam_prod"
    
    ## DEFINING OUTPUT PREFIX
    ## DFEINING OUTPUT SUFFIX
    output_suffix = "_gold_center"
    
    ## DEFINING PATH
    path_to_sim = os.path.join(main_dir,
                               simulation_dir,
                               specific_dir)
    
    ## COMPUTING BUNDLE
    bundle, traj_data = main_compute_bundles(path_to_sim = path_to_sim,
                                             input_prefix = input_prefix,
                                             output_suffix = output_suffix,
                                             save_disk_space = False,
                                             displacement_vector_type = 'avg_heavy_atom')
    
    #%%
    
    ## DEFINING BUNDLE PLOT
    bundle_plotter = plot_nanoparticle_bundling_groups(bundling = bundle.bundling_groups )
    
    #%%
    
    bundle_plotter.plot_sulfur_lig_grp_frame()
    
    
    #%%
    
    ## COMPUTING BUNDLE WITH JUST TERMINAL HEAVY ATOMS
    bundle_2, traj_data = main_compute_bundles(path_to_sim = path_to_sim,
                                             input_prefix = input_prefix,
                                             output_suffix = output_suffix,
                                             save_disk_space = False,
                                             displacement_vector_type = 'terminal_heavy_atom')
    
    #%%
    
    ## DEFINING BUNDLE PLOT
    bundle_plotter_2 = plot_nanoparticle_bundling_groups(bundling = bundle_2.bundling_groups )
    
    #%%
    
    bundle_plotter_2.plot_sulfur_lig_grp_frame()
    
    
    #%%
    import MDDescriptors.core.calc_tools as calc_tools # calc tools
    import mdtraj as md
    
    
    ## DEFINING BUNDLING GROUP
    bundling_groups = bundle.bundling_groups
    
    ## CURRENT LIG DISPLACEMENTS
    current_lig_displacements = bundling_groups.lig_displacements
    
    ## FLATTENING ALL HEAVY ATOM INDEXES
    flattened_heavy_atom_index = calc_tools.flatten_list_of_list(bundling_groups.structure_np.ligand_heavy_atom_index)
    
    ## GENERATE ATOM PAIRS
    # atom_pairs = [ [each_sulfur_index, bundling_groups.terminal_group_index[idx] ] for idx, each_sulfur_index in enumerate(bundling_groups.structure_np.head_group_atom_index) ]
    

    
    ## GETTING LIGAND DISPLACEMENTS
    lig_displacements_array = compute_lig_avg_displacement_array(traj = traj_data.traj,
                                                                 sulfur_atom_index = bundling_groups.structure_np.head_group_atom_index,
                                                                 ligand_heavy_atom_index = bundling_groups.structure_np.ligand_heavy_atom_index)
    
    #%%
    

    

    # ----------------------------------------------------------------------
    ### FUNCTION TO CORRECT ATOM NUMBERS
    def correct_atom_numbers( list_to_convert, conversion_legend ):
        '''
        The purpose of this function is to take a list of list and convert all the numbers according to a conversion list. The conversion list is also a list of list (i.e. [[1,2], [2,3]])
        The way this script works is that it flattens out the list of list, then converts the numbers, then re-capitulates the list of list.
        INPUTS:
            list_to_convert: list of list that you want to fix in terms of numbers (i.e. atom numbers)
                NOTE: zeroth index is the current value and 1st index is the transformed value
            conversion_legend: list of list that has indexes where the first index is the original and the next index is the new value
        OUTPUTS:
            converted_list: [np.array] array with the corrected values
        '''
        # Start by converting the list to a numpy array
        converted_list = np.array(list_to_convert).astype('int')
        ## CONVERTING CONVERSION LEGEND TO NUMPY ARRAY
        conversion_legend_array = np.array(conversion_legend).astype('int')
        # Copying list so we do not lose track of it
        orig_list = converted_list[:]
        # Looping through each conversion list value and replacing
        for legend_values in conversion_legend_array:
            ## FINDING LOCATION OF WHERE IS TRUE
            indexes = np.where( orig_list == legend_values[0] )
            ## CHANGE IF FOUND INDEX
            if len(indexes) > 0:
                converted_list[indexes] = legend_values[1]
        return converted_list

    ### FUNCTION TO CREATE DISTANCE MATRIX BASED ON PAIR DISTANCES
    def create_pair_distance_matrix(atom_index, distances, atom_pairs, total_atoms = None):
        '''
        The purpose of this function is to get the distances and atom pairs to generate a distance matrix for pairs
        INPUTS:
            atom_index: [list] 
                list of atom indexes you want pair distances for
            distances: [np.array, shape=(frames, pairs, 1)] 
                Distances between all pairs
            atom_pairs: [list] 
                list of atom pairs
            total_atoms: [int, optional, default = None (will use atom_index)] 
                total number of atoms used for the pairs
        OUTPUTS:
            distances_matrix: [np.array, shape=(time_frame, total_atoms, total_atoms)] 
                Distance matrix between atoms
        '''
        ## GETTING TOTAL ATOMS
        if total_atoms == None:
            total_atoms = len(atom_index)
        
        ## FINDING TOTAL TIME
        total_time = len(distances)
        
        ## CREATING DICTIONARY TO MAP ATOM PAIRS TO INDICES IN MATRIX
        atom_index_mapping_dist_matrix = np.array([ [current_index, idx] for idx, current_index in enumerate(atom_index)])
        
        ## FIXING ATOM PAIRS NAMES
        dist_matrix_atom_pairs = correct_atom_numbers( atom_pairs, atom_index_mapping_dist_matrix )
        
        ## CREATING ZEROS ARRAY
        distances_matrix = np.zeros( (total_time, total_atoms, total_atoms)  )
        
        ## LOOPING THROUGH EACH FRAME
        for each_frame in range(total_time):
            ## DEFINING CURRENT DISTANCES
            frame_distances = distances[each_frame]
            ## LOOPING THROUGH EACH PAIR
            for idx, each_res_pair in enumerate(dist_matrix_atom_pairs):
                distances_matrix[each_frame][tuple(each_res_pair)] = frame_distances[idx]
        
            ## ADJUSTING FOR SYMMETRIC MATRIX
            distances_matrix[each_frame] = distances_matrix[each_frame] + distances_matrix[each_frame].T
            
        return distances_matrix
    # ----------------------------------------------------------------------
    
    ## COMPUTING PAIR DISTANCES WITH ARBITRARY DISPLACEMENTS
    distances, atom_pairs, total_atoms, atom_indices_to_change = calc_tools.compute_pair_distances_with_arbitrary_displacements(traj = traj_data.traj,
                                                                                                                                 displacements = lig_displacements_array,
                                                                                                                                 periodic=True,
                                                                                                                                 )
                                        
    ## COMPUTING DISTANCE MATRIX
    tail_tail_distance_matrix = create_pair_distance_matrix( atom_index = atom_indices_to_change,
                                                             distances = distances,
                                                             atom_pairs = atom_pairs,
                                                             total_atoms = total_atoms,
                                                             )
    
    
    
    
    
    
    
    #%%
    
    total_frames = len(traj_data.traj)
    total_atom_1= len(atom_1_index)
    total_atom_2 = len(atom_2_index)
    ## RESHAPING THE DISPLACMEENT
    displacements_reshaped = displacements.reshape(total_frames, total_atom_1, total_atom_2, 3)
    
    
    
    
    #%%
    
    
    
    

    
    ## GENERATING ATOM PAIRS
    distances = calc_tools.calc_pair_distances_between_two_atom_index_list(traj = traj_data.traj,
                                                                           atom_1_index = bundling_groups.structure_np.head_group_atom_index,
                                                                           atom_2_index = flattened_heavy_atom_index,
                                                                           periodic = True)
    
    
    
    



    