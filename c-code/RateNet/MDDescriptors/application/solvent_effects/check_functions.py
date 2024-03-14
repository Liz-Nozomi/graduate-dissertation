# -*- coding: utf-8 -*-
"""
check_functions.py
The purpose of this script is to store all check functions

Created on: 01/09/2019

FUNCTIONS:
    check_traj_names: checks traj names to see what is available
    check_solute_solvent_names: checks solute and solvent names
    
    
Author(s):
    Alex K. Chew (alexkchew@gmail.com)
    
** UPDATES **

"""

## SYSTEM TOOLS
import numpy as np
import sys

### FUNCTION TO CHECK IF RESIDUE IN TRAJECTORY AND OUTPUT ONLY RESIDUES IN TRAJECTORY
def check_traj_names( traj_data, residue_names ):
    '''
    The purpose of this function is to check the list of names and find the ones that are in the trajectory
    INPUTS:
        traj_data: [md.traj]
            trajectory from md.traj
        residue_names: [list]
            list of the residues you are interested in
    OUTPUTS:
        residue_names: [list]
            updated residue names
    '''
    residue_names = [ name for name in residue_names if name in traj_data.residues.keys() ]
    return residue_names

### FUNCTION TO CHECK SOLUTE AND SOLVENT NAMES
def check_solute_solvent_names( traj_data, solute_name, solvent_name ):
    '''
    The purpose of this function is to check the solute and solvent names. We are assuming you have one and the other.
    INPUTS:
        traj: [md.traj]
            trajectory from md.traj
        solute_name: [list]
            list of the solutes you are interested in
        solvent_name: [list]
            list of solvents you are interested in.
    OUTPUTS:
        solute_name: [list]
            list of updated solutes
        solvent_names: [list]
            list of updated solvent molecule names
    '''
    ## UPDATING SOLUTE AND SOLVENT NAMES
    solute_name = check_traj_names(traj_data = traj_data, residue_names = solute_name) 
    solvent_name = check_traj_names(traj_data = traj_data, residue_names = solvent_name)
    
    ### CHECK IF SOLUTE EXISTS IN TRAJECTORY
    if len(solute_name) == 0 or len(solvent_name) == 0:
        print("ERROR! Solute or solvent specified not available in trajectory! Stopping here to prevent further errors.")
        print("Residue names available: ")
        print(traj_data.residues.keys())
        print("Input solute names: ")
        print(solute_name)
        print("Input solvent names: ")
        print(solvent_name)
        sys.exit()
    
    return solute_name, solvent_name
