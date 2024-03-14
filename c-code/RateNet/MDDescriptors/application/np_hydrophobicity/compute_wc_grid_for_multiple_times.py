# -*- coding: utf-8 -*-
"""
compute_wc_grid_for_multiple_times.py
The purpose of this script is to compute the wc grid for multiple grids. The 
idea is to test whether the willard-chandler grid is correctly converged 
when you have spring constants

Written by: Alex K. Chew (02/13/2020)
"""
### IMPORTING MODULES
import numpy as np
import pandas as pd
import os
import mdtraj as md

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv, generate_gro_xtc_with_center_AUNP

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

## IMPORTING FINDING LIG RESIDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj

## CREATING GRID FUCNTIONS
from MDDescriptors.surface.core_functions import create_grid, get_list_args

from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

### FUNCTION TO GENERATE GRID
def main_compute_wc_grid_multiple_subset_frames(path_to_sim,
                                                input_prefix,
                                                num_frames_iterated,
                                                n_procs = 1,
                                                alpha = WC_DEFAULTS['alpha'],
                                                contour = WC_DEFAULTS['contour'],
                                                mesh = WC_DEFAULTS['mesh'],
                                                out_path = None,
                                                rewrite = False,
                                                func_inputs = {}):
    '''
    The purpose of this function is to compute the wc interface for multiple 
    subsets of instances. Note that this function will generate a gro 
    and xtc file that has only the heavy atoms of water. 
    INPUTS:
        path_to_sim: [str]
            path to simulation folder
        input_prefix: [str]
            input prefix for gro, xtc, tpr, and so on
        num_frames_iterated: [int]
            number of frames to iterate on
        n_procs: [int]
            number of processors to run this code on
        func_inputs: [dict]
            dictionary of function inputs to run this on
    OUTPUTS:
        grid_list: [list]
            list of grid for output willard chandler interfaces
        output_prefix_list: [list]
            list of output prefixes
    '''
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv(wd = path_to_sim)
    
    func_inputs['input_prefix'] = input_prefix
    
    ## CONVERTING TRJ TO HEAVY WATER ATOMS
    output_gro_file, output_xtc_file = trjconv_conversion.generate_water_heavy_atoms(rewrite = rewrite,
                                                                                     **func_inputs)
    
    ## LOADING TRAJECTORY
    traj = md.load(os.path.join(path_to_sim, output_xtc_file),
                   top = os.path.join(path_to_sim, output_gro_file))

    ## GETTING THE NUMBER OF ITERATIONS
    num_iterations = int( len(traj.time) / num_frames_iterated )
    
    ## GETTING FRAME RANGES
    frame_ranges = [ np.arange(num_frames_iterated) + iterations*num_frames_iterated for iterations in range(num_iterations) ]
    
    ## DEFINING OUTPUT PATH
    if out_path is None:
        out_path = path_to_sim
    
    ## DEFINING GRID INPUTS
    wc_grid_inputs = {
            'alpha': alpha,
            'mesh': mesh,
            'contour': contour
            }
    # dict((k, WC_DEFAULTS[k]) for k in ('alpha', 'contour', 'mesh'))
    print("WC parameters:")
    print(wc_grid_inputs)
    
    ## GENERATING GRID LIST
    grid_list = []
    output_prefix_list = []
    
    ## LOOPING THROUGH FRAME RANGES
    for current_range in frame_ranges:
        ## DEFINING DATA FILE NAME
        output_prefix = "wc-%d_%d"%(current_range[0],current_range[-1])
        ## GENERATING GRID
        grid = create_grid( traj = traj[current_range], 
                            out_path = out_path, 
                            wcdatfilename = output_prefix + ".dat", 
                            wcpdbfilename = output_prefix + ".pdb", 
                            write_pdb = True, 
                            n_procs = n_procs,
                            verbose = True,
                            want_rewrite = rewrite,
                            **wc_grid_inputs
                            )
        
        ## STORING GRID
        grid_list.append(grid[:])
        output_prefix_list.append(output_prefix)
        
    return grid_list, output_prefix_list


#%%
## MAIN FOLDER
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
    
    ### DIRECTORY TO WORK ON    
    simulation_dir=r"20200212-Debugging_GNP_spring_constants_heavy_atoms"
    
    ## DEFINING SPECIFIC DIR
    specific_dir = r"MostlikelynpNVTspr_1000-EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1_likelyindex_1"
    
    ## DEFINING PATH
    path_to_sim = os.path.join(main_dir,
                               simulation_dir,
                               specific_dir)
    
    ## DFEINING GRID INPUTS
    grid_inputs = {
            'path_to_sim': path_to_sim,
            'input_prefix': "sam_prod",
            'num_frames_iterated': 1000,
            'n_procs': 1,
            }
    
    ## GETTING GRIDDING
    grid_list, output_prefix_list = main_compute_wc_grid_multiple_subset_frames(**grid_inputs)
    

    
    
    
    
    
    
    
    