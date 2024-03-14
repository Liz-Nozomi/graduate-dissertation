#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nplm_generate_index_list.py

The purpose of this script is to generate an index list for the lipid membrane 
project. The idea would be to load a gro file, then identify all the nanoparticle 
atoms and lipid membrane atoms. Then, we will perform a lipid membrane 
extract protocol that was developed recently under the "divide_lipid_membrane_leaflets.py". 

Written by: Alex K. Chew (05/18/2020)


"""
import os
import mdtraj as md
import numpy as np
import time
from optparse import OptionParser # Used to allow commands within command line

## READING FILE AS LINES
from MDDescriptors.core.read_write_tools import import_index_file

## IMPORTING LIPID MEMBRANE DETAILS
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import get_nplm_heavy_atom_details
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups, get_permutation_groups

## CHECK TESTING FUNCTION
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

## LOADING PLOT FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## RENAMING DICTIONARY LABEL
def add_dict_prefix(my_dict,
                    prefix = ''):
    '''
    This adds a prefix to the labels in the dictionary. This is useful when you
    want to clearly distinguish groups, e.g. add "NP" to "GOLD" -> "NP_GOLD"
    
    INPUTS:
        my_dict: [dict]
            dictionary containing information
        prefix: [str]
            prefix to add to each label
    OUTPUTS:
        my_dict: [dict]
            updated dictionary
    '''
    key_list = list(my_dict.keys())
    ## LOOPING THROUGH EACH KEY
    for each_key in key_list:
        ## NEW KEY
        new_key = prefix + "_" + each_key
        ## UPDATING AND REMOVING PREVIOUS KEY
        my_dict[new_key] = my_dict[each_key]
        del my_dict[each_key]
    return my_dict

### MAIN FUNCTION
def main_generate_nplm_groups_with_permutations(path_to_sim,
                                                gro_file,
                                                index_file = None,
                                                permutation_file = None,
                                                group_list = None,
                                                lm_residue_name = "DOPC",
                                                verbose = True,
                                                skip_indexing = False):
    '''
    This is the main function to generate nanoparticle lipid membrane 
    groups for the index files.
    INPUTS:
        path_to_sim: [str]
            path to the simulation file
        permutation_file: [str]
            path to permutation file to store all possible permutations between 
            nanoparticle and lipid membrane
        gro_file: [str]
            gro file
        index_file: [str]
            index file
        lm_residue_name: [str]
            lipid membrane residue name
        verbose: [logical]
            True if you want to be verbose
        group_list: [str]
            file that will store all the groups
        skip_indexing: [logical]
            True or False, depending if you want 
    OUTPUTS:
        This function will update the 'index_file' with the groups.
        This function will also output a permutation file that will inform 
        about the possible combinations between the groups.
        
        np_full_group: [dict]
            dictinoary of the nanoparticle groups
        lm_full_group: [dict]
            dictionary of lipid membrane groups
    '''
    
    ## DEFINING PATH TO INDEX FILE
    path_to_index = os.path.join(path_to_sim,
                                   index_file)
    
    ## LOADING FILE
    path_to_gro = os.path.join(path_to_sim,
                               gro_file
                               )
    ## LOADING TRAJECTORY
    if verbose is True:
        print("---- Generating index groups for nplm ----")
        print("Loading GRO file: %s"%(path_to_gro) )
    traj = md.load(path_to_gro)
    
    
    ## FINDING NANOPARTICLE AND LIPID MEMBRANE ATOM INDICES
    ligand_names, np_heavy_atom_index, np_heavy_atom_names, lm_heavy_atom_index, lm_heavy_atom_names = get_nplm_heavy_atom_details(traj = traj,
                                                                                                                                   lm_res_name = lm_residue_name,
                                                                                                                                   atom_detail_type="all")
    
    ## GENERATING GROUPS
    ## FINDING ATOM INDEX
    lm_groups, lm_groups_indices = generate_lm_groups(traj = traj,
                                                      atom_names = lm_heavy_atom_names,
                                                      lm_heavy_atom_index = lm_heavy_atom_index,
                                                      verbose = False,
                                                      want_atom_index = True,
                                                      )
    
    ## FINDING NANOPARTICLE GROUPS
    np_groups, np_groups_indices = generate_rotello_np_groups(traj = traj,
                                                              np_heavy_atom_index = np_heavy_atom_index,
                                                              atom_names_np = np_heavy_atom_names,
                                                              verbose = False,
                                                              want_atom_index = True,
                                                              combine_alkane_and_R_group = True,
                                                              combine_N_and_R_group = True)
    
    ## FINDING LARGER GROUPS FOR NANOPARTICLE
    np_larger_groups={
            'LIGAND_HEAVY_ATOMS': np.array([each_idx for each_idx in np_heavy_atom_index if traj.topology.atom(each_idx).element.symbol != "Au"]),
            'LIGAND_CARBON_ONLY': np.array([each_idx for each_idx in np_heavy_atom_index if traj.topology.atom(each_idx).element.symbol == "C"]),
            }
    
    ## GETTING FULL GROUPING
    np_full_group = {**np_groups_indices, **np_larger_groups}
    
    ## ADDING NP PREFIX
    np_full_group = add_dict_prefix(np_full_group, prefix="NP")
    lm_full_group = add_dict_prefix(lm_groups_indices, prefix="LM")
    
    ## GETTING INDEX FILE
    ndx_file = import_index_file(path_to_index)
    
    ## WRITING TO INDEX FILE
    ndx_file = import_index_file(path_to_index)
    
    ## LOOPING THROUGH EACH GROUP
    for each_group_dict in [np_full_group, lm_full_group]:
        ## LOOPING THROUGH EACH KEY
        for each_key in each_group_dict.keys():
            ## DEFINING INDICES
            current_indices = each_group_dict[each_key]
            ## WRITING TO INDEX FILE
            ndx_file.write(index_key = each_key,
                           index_list = current_indices,
                           backup_file = False) # Turning off backup of files
            
    ## WRITING GROUP FILE
    if group_list is not None:
        ## DEFINING PATH TO GROUP LIST
        path_to_group_list = os.path.join(path_to_sim,
                                           group_list)
        
        ## WRITING
        with open(path_to_group_list, 'w') as f:
            for each_group_dict in [np_full_group, lm_full_group]:
                for each_key in each_group_dict.keys():
                    f.write("%s\n"%(each_key))
        
    ## WRITING PERMUATION FILE
    if permutation_file is not None:
        ## DEFINING PATH TO PERMUTATION
        path_to_permutation = os.path.join(path_to_sim,
                                           permutation_file)
        ## FINDING ALL PERMUATIONS
        nplm_perm_list = get_permutation_groups(np_full_group,
                                                lm_full_group)
        if verbose is True:
            print("Generating permutation file: %s"%(path_to_permutation))
        with open(path_to_permutation, 'w') as perm:
            ## lOOPING THROUGH AND WRITING
            for each_entry in nplm_perm_list:
                perm.write("%s\n"%(each_entry))

    return np_full_group, lm_full_group

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ### TURNING TEST ON / OFF
    testing = check_testing() # False if you're running this script on command prompt!!!`
    
    ## TESTING
    if testing is True:
    
        ## DEFINING PATH TO SIMULATION
        path_to_sim = "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200120-US-sims_NPLM_rerun_stampede/US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/4_simulations/5.100"
        
        ## DEFINING GRO FILE
        gro_file="nplm_prod-non-Water.gro"
        
        ## DEFINING INDEX FILE
        index_file="nplm_prod-DOPC-AUNP.ndx"
        
        ## DEFINING PERMUTATION FILE
        permutation_file="permutation.dat"
        
        ## DEFINING DOPC LIPID BILAYERS
        lm_residue_name = "DOPC"
        
        ## DEFINING GROUP LIST
        group_list="groups.dat"
    else:
        
        ### DEFINING PARSER OPTIONSn
        # Adding options for command line input (e.g. --ligx, etc.)
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## INPUT FOLDER
        parser.add_option("--path_to_sim", 
                          dest="path_to_sim", 
                          action="store", 
                          type="string", 
                          help="Path to simulation", default=".")
        
        parser.add_option("--gro_file", 
                          dest="gro_file", 
                          action="store", 
                          type="string", 
                          help="Gro file", default=".")
        
        parser.add_option("--index_file", 
                          dest="index_file", 
                          action="store", 
                          type="string", 
                          help="Index file", default=".")
        
        parser.add_option("--lm_residue_name", 
                          dest="lm_residue_name", 
                          action="store", 
                          type="string", 
                          help="Lipid membrane residue name", default="DOPC")
        
        parser.add_option("--output_perm_file", 
                          dest="permutation_file", 
                          action="store", 
                          type="string", 
                          help="Permutation file", default=None)
        
        parser.add_option("--output_group_list", 
                          dest="group_list", 
                          action="store", 
                          type="string", 
                          help="Groups that will be listed", default=None)
        
        ### PARSING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## INPUT FILES
        path_to_sim = options.path_to_sim
        gro_file = options.gro_file
        index_file = options.index_file
        lm_residue_name = options.lm_residue_name
        permutation_file = options.permutation_file
        group_list = options.group_list
        
    
    ## MAIN FUCNTION TO GENERATE NPLM GORUPS WITH PERMUTATIONS
    np_full_group, lm_full_group = main_generate_nplm_groups_with_permutations(path_to_sim = path_to_sim,
                                                                               gro_file = gro_file,
                                                                               index_file = index_file,
                                                                               permutation_file = permutation_file,
                                                                               lm_residue_name = lm_residue_name,
                                                                               group_list = group_list,)
    