#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_wc_surface.py
This code generates the Willard-chandler surface.

Written by: Alex K. Chew & Bradley C. Dallin
"""
##


## IMPORTING TOOLS
from MDDescriptors.core.import_tools import import_traj

## CHECKING TOOLS
# from MDDescriptors.core.check_tools import check_testing

## GRIDDING TOOLS
from MDDescriptors.geometry.willard_chandler_grid import create_grid, WC_DEFAULTS


#%% MAIN FUNCTIONS
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    # testing = check_testing()
    
    ## DEFINING SIMULATION DIRECTORY
    path_sim="/Volumes/akchew/scratch/nanoparticle_project/simulations/HYDROPHOBICITY_PROJECT_C11/EAM_300.00_K_2_nmDIAM_C11COO_CHARMM36jul2017_Trial_1/NVT_grid_1"
    
    ## DEFINING GRO FILE
    gro_file="sam_prod.gro"
    ## DEFINING XTC FILE
    xtc_file="sam_prod.xtc"
    
    ## DEFINING NUMBER OF PROC
    n_procs=1
    
    ## IMPORTING TRAJECTORY
    traj_data = import_traj( directory = path_sim, 
                             structure_file = gro_file,
                             xtc_file = xtc_file,
                             verbose = True)
    
    #%%
    
    ## RUNNING GRID
    grid = create_grid(traj = traj_data.traj, 
                       out_path, 
                       wcdatfilename, 
                       wcpdbfilename, 
                       alpha = alpha, 
                       mesh = mesh, 
                       contour = contour, 
                       write_pdb = True, 
                       n_procs = n_procs )