#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_ligand_residue_names.py
This function finds all possible ligand residue names

e.g.
    from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj
    

"""
## IMPORTING MODULES
import os
import sys
import numpy as np

## IMPORTING PATHS
from MDDescriptors.core.check_tools import check_path
from MDDescriptors.core.import_tools import read_file_as_line

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## DEFINING PATH TO RESIDUE NAME
PATH_RESIDUE_NAME=check_path(r"R:\scratch\MDLigands\final_ligands\charmm36-jul2017.ff\lig_names.txt")
# "/Volumes/akchew/scratch/MDLigands/final_ligands/charmm36-jul2017.ff/lig_names.txt"

### FUNCTION TO LOAD ALL LIGAND RESIDUE NAMES
def load_all_ligand_residue_names(path_to_file = PATH_RESIDUE_NAME, 
                                  verbose = False,
                                  return_lig_names = False):
    '''
    The purpose of this function is to load all ligand residue names
    INPUTS:
        return_lig_names: [logical]
            True if you awnt to return ligand names
    '''
    ## CHECKING PATH TO SERVER
    path_to_file = check_path( path_to_file )
    
    ## CHECKING IF FILE EXISTS
    if os.path.isfile( path_to_file ):
        ## READING FILE
        lines = read_file_as_line( file_path = path_to_file,
                                   want_clean = True,
                                   verbose = verbose,
                                  )
        ## DEFINING LIGAND RESIDUE LIST
        ligand_residue_list = [each_line.split(', ')[1] for each_line in lines]
        
        ## DEFINING LIGAND RESIDUE LIST
        ligand_name_list = [each_line.split(', ')[0] for each_line in lines]

    else:
        print("Error in load_all_ligand_residue_names function")
        print("Path to file does not exist: %s "%(path_to_file))
        print("Double-check code in MDDescriptors > applications > nanoparticle > core > find_ligand_residue_names.py")
        sys.exit(1)
    if return_lig_names is True:
        return ligand_residue_list, ligand_name_list
    return ligand_residue_list

### FUNCTION TO GET LIGAND NAMES
def get_ligand_names_within_traj(traj, return_lig_names = False):
    '''
    The purpose of this function is to get ligand names within the 
    trajectory. 
    INPUTS:
        traj: [obj]
            tajectory object
        return_lig_names: [logical]
            True if you want to get all lignaen ames
    OUTPUTS:
        ligand_names: [np.array]
            ligand names within the trajectory
    '''
    ## LOADING LIGAND RESIDUE NAMES
    if return_lig_names is False:
        ligand_residue_list = np.array(load_all_ligand_residue_names())
    else:
        ligand_residue_list, ligand_name_list = load_all_ligand_residue_names(return_lig_names = return_lig_names)
        ## CONVERTING TO NUMPY ARRAY
        ligand_residue_list = np.array(ligand_residue_list)
        ligand_name_list = np.array(ligand_name_list)
    
    ## FINDING RESIDUE NAME
    unique_residue_names = np.unique([ each_residue.name for each_residue in traj.topology.residues ])
    
    ## FINDING ALL LIGAND NAME
    ligand_names = np.intersect1d(unique_residue_names, ligand_residue_list)
    
    if return_lig_names is True:
        ## FINDING LOCATION
        ligand_full_name = []
        for each_res in ligand_names:
            res_idx = np.where(ligand_residue_list == each_res)[0]
            ## STORING
            ligand_full_name.append(ligand_name_list[res_idx])
            
        return ligand_names, ligand_full_name
    else:
        return ligand_names

### FUNCTION TO GET ATOM INDICES OF LIGANDS
def get_atom_indices_of_ligands_in_traj( traj ):
    '''
    The purpose of this function is to get all atom indices within traj, specifically 
    all ligand indices
    INPUTS:
        traj: [obj]
            trajectory object
    OUTPUTS:
        ligand_names: [list]
            ligand names
        atom_index: [np.array]
            list of atom indices
    '''
    ## GETTING LIGAND NAMES
    ligand_names = get_ligand_names_within_traj(traj = traj)
    
    ## GETTING ALL ATOM INIDICES
    atom_index = np.array(calc_tools.flatten_list_of_list([ 
                           calc_tools.find_residue_heavy_atoms(traj = traj, 
                                                               residue_name = each_ligand)
                                           for each_ligand in ligand_names]))
    
    return ligand_names, atom_index

#%% MAIN SCRIPT
if __name__ == "__main__":
    ## LOADING ALL RESIDUE NAMES
    ligand_residue_list = load_all_ligand_residue_names()