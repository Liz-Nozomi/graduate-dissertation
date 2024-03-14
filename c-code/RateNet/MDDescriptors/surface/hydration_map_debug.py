# -*- coding: utf-8 -*-
"""
hydration_map_debug.py
this script debugs hydration map analysis

CREATED ON: 05/12/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os
import pickle
import numpy as np
import mdtraj as md
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl

##############################################################################
## CLASSES AND FUNCTIONS
##############################################################################

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## OUTPUT DIRECTORY
    output_dir = r""    
    ## INPUT FILES
    wc_grid_pkl = r"remove_grid.pickle"
    input_prefix = "sam_debug"
    ## MAIN DIRECTORY
    main_dir       = r"R:\simulations\np_hydrophobicity\debug_hydration_maps"
    simulation_dir = r"NVTspr_50_Planar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps"
    wc_grid_dir    = r"26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000\grid-45000_50000"
    ## COMBINES PATHS
    path_sim     = os.path.join( main_dir, simulation_dir )
    path_gro     = os.path.join( path_sim, input_prefix + ".gro" )
    path_xtc     = os.path.join( path_sim, input_prefix + ".xtc" )
    path_wc_grid = os.path.join( path_sim, wc_grid_dir, wc_grid_pkl )
    
    ## LOAD TRAJ
    traj = md.load( path_xtc, top = path_gro )
    
    ## LOAD WC GRID PICKLE
    grid = load_pkl( path_wc_grid )