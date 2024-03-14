#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_mu_distribution.py
The purpose of this script is to regenerate the mu distribution for a given 
simulation. 

Written by: Alex K. Chew (03/18/2020)
"""
import sys
import os
import glob
## DEFINNIG SIM PATH
from MDDescriptors.application.np_hydrophobicity.global_vars import \
    PARENT_SIM_PATH, COMBINE_NEIGHBORS_DIR, COMBINE_NEIGHBORS_LOG, GRID_LOC, GRID_OUTFILE, \
    OUT_HYDRATION_PDB, MU_PICKLE, PROD_GRO

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing 

## READING MU DISTRIBUTION
from MDDescriptors.application.np_hydrophobicity.analyze_mu_distribution import extract_hydration_maps, compute_mu_from_unnorm_p_N

## IMPORTING NUM DIST FUNCTION
from MDDescriptors.surface.generate_hydration_maps import compute_num_dist

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import MAX_N, MU_MIN, MU_MAX

## CALC MU FUNCTION
from MDDescriptors.surface.core_functions import load_datafile, create_pdb

## IMPORTING GRO READING
from MDDescriptors.core.read_write_tools import extract_gro

## PICKLE FUNCTIONS
from MDDescriptors.core.pickle_tools import pickle_results

## READING FILES
from MDDescriptors.core.import_tools import read_file_as_line

### FUNCTION TO READ NEIGHBOR LOGS
def read_neighbor_logs(path_pickle_log,
                       deliminator=', '):
    '''
    The purpose of this function is to read the logs and output the details
    '''
    ## READING LINES
    lines = read_file_as_line(path_pickle_log)
    
    ## SPLITTING BASED ON DELIMINATOR
    lines = [ each_line.split(deliminator) for each_line in lines ]    
    return lines


### FUNCTION TO COMPUTE MU DISTRIBUTION
def compute_mu_dist_from_neighbor_array(path_to_neighbors = None,
                                        pickle_file_name = None,
                                        max_neighbors = MAX_N,
                                        num_neighbors_array = None,
                                        unnorm_p_N = None,):
    '''
    This function computes the mu distribution using the neighbor array.
    INPUTS:
        path_to_neighbors: [str]
            path to the neighbor pickle location
        pickle_file_name: [str]
            path to the pickle file name
        max_neighbors: [int]
            maximum neighbors used for the number distribution
        num_neighbors_array: [np.array]
            number neighbors array. If this is None, then we will load 
            it from the file.
        unnorm_p_N: [np.array]
            unnormalize p_N array
    OUTPUTS:
        mu_dist: {obj]
            object containing information for mu distribution
        
    '''
    ## LOADING PICKLE
    if num_neighbors_array is None:
        hydration_map = extract_hydration_maps()
        num_neighbors_array = hydration_map.load_neighbor_values(main_sim_list = [path_to_neighbors],
                                                           pickle_name = pickle_file_name)[0]
        
    ## COMPUTING NUMBER DIST
    unnorm_p_N = compute_num_dist(num_neighbors_array = num_neighbors_array,
                                  max_neighbors = max_neighbors)

    ## GETTING MU DIST
    mu_dist = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
    
    return mu_dist, num_neighbors_array, unnorm_p_N

### FUNCTION TO COMPUTE MU FOR A LIST OF DIRECTORIES
def compute_mu_from_neighbors_for_one_directory(full_path_to_dir,
                                              extraction_dir,
                                              combine_neighbors_dir = COMBINE_NEIGHBORS_DIR,
                                              combine_neighbors_log = COMBINE_NEIGHBORS_LOG,
                                              min_mu = MU_MIN,
                                              max_mu = MU_MAX,
                                              output_pdb = OUT_HYDRATION_PDB,
                                              output_mu = MU_PICKLE,
                                              grid_dir = GRID_LOC,
                                              grid_output_file = GRID_OUTFILE,
                                              input_gro_file = PROD_GRO,
                                              max_N = MAX_N,
                                              num_neighbor_array = None,
                                              grid = None,
                                              ):
    '''
    This function computes mu and outputs it into a *.pdb / *.pickle file.
    INPUTS:
        full_path_to_dir: [str]
            full path to the directory you want to work in
        extraction_dir: [str]
            extractiong directory to look into, e.g. 30-0.24-0.1,0.1,0.1-0.33-all_heavy-0-150000
        combine_neighbors_dir: [str]
            combining neighbors directory
        combine_neighbors_log: [str]
            log file of neighbors
        min_mu: [float]
            minimum value for mu to test
        max_mu: [float]
            maximum mu value to test
        output_pdb: [str]
            output pdb file to print to
        output_mu: [str]
            output mu values (mu.pickle)
        grid_dir: [str]
            location of the wc grid information
        grid_output_file: [str]
            location of the output file for the grid
        input_gro_file: [str]
            input gro file
        num_neighbors_array: [np.array]
            number neighbors array. If this is None, then we will load 
            it from the file.
        grid: [np.array]
            grid with N X 3 points
    OUTPUTS:
        mu_dist: [obj]
            object for the mu distribution
        mu: [np.array]
            mu distribution for each grid point
        renorm_mu: [np.array]
            renormalized mu values
        grid: [np.array]
            xyz coordinates for the grid
        
    '''
    ## DEFINING DIRECTORY
    path_to_current_dir = os.path.join(full_path_to_dir,
                                       extraction_dir)
    
    ## PATH TO NEIGHBORS
    path_to_neighbors = os.path.join(path_to_current_dir,
                                         combine_neighbors_dir,
                                         )
    
    ## LOADING THE GRID POINTS
    path_to_grid = os.path.join(path_to_current_dir,
                                grid_dir,
                                grid_output_file)
    
    ## FINDING NEIGHBORS
    path_to_neighbors_log = os.path.join(path_to_neighbors,
                                         combine_neighbors_log,
                                         )
    
    ## PATH TO GRO FILE
    path_to_gro = os.path.join(full_path_to_dir,
                               input_gro_file)
    
    ## PATH TO OUTPUT PDB
    path_pdb_file = os.path.join(path_to_current_dir,
                                 output_pdb)
    
    ## DEFINING PATH TO MU PICKLE
    path_mu_file = os.path.join(path_to_current_dir,
                                 output_mu)
    
    ## READING LOG FILE
    log_file = read_neighbor_logs(path_to_neighbors_log)
    
    ## GETTING PICKLE FILES
    pickle_file_name = log_file[0][1]
    
    ## COMPUTING MU DISTRIBUTION
    mu_dist, num_neighbors_array, unnorm_p_N = compute_mu_dist_from_neighbor_array(path_to_neighbors = path_to_neighbors,
                                                                                   pickle_file_name = pickle_file_name,
                                                                                   max_neighbors = max_N,
                                                                                   num_neighbors_array = num_neighbor_array,)
    
    ## GETTING MU VALUE
    mu = mu_dist.mu_storage['mu_value'].to_numpy()
    
    ## PICKLING MU VALUES
    pickle_results( results = [mu],
                    pickle_path = path_mu_file)
    
    print("Outputting mu values to: %s"%(path_mu_file))
    #############################
    ### LOADING THE GRID FILE ###
    #############################
    if grid is None:
        grid = load_datafile(path_to_file = path_to_grid)
    
    ## RENOMRALIZE MU VALUE
    renorm_mu = ( mu - min_mu ) / ( max_mu - min_mu ) # red = hydrophobic; blue = hydrophilic
    
    ########################
    ### LOADING GRO FILE ###
    ########################
    ## LOADING GRO FILE
    gro_file = extract_gro(gro_file_path = path_to_gro)

    #########################
    ### CREATING PDB FILE ###
    #########################            
    ## WRITE PDB FILE WITH PROBES COLORED BY MU
    create_pdb(data = grid,
               box_dims = gro_file.Box_Dimensions,
               b_factors = renorm_mu,
               path_pdb_file = path_pdb_file
               )
    
    print("Outputting PDB file to: %s"%(path_pdb_file))
    
    return mu_dist, mu, renorm_mu, grid

### FUNCTION TO FIND A LIST OF DIRECTORIES
def compute_mu_from_neighbors_array_dir_list(path_to_dir, verbose = True, **args):
    '''
    This function runs the neighbors array list aross multiple directories
    INPUTS:
        path_to_dir: [str]
            path to the directory
        verbose: [logical]
            True if you want to print verbosely
        **args: [dict]
            arguments that will be passed on
    '''
    ## GETTING LIST OF DIRECTORY
    list_of_dir = glob.glob(path_to_dir + "/*")
    if len(list_of_dir) == 0:
        print("Error, no directory is available for: %s"%(path_to_dir))
        sys.exit(1)
    for each_dir in list_of_dir: 
        ## DEFINING FULL PATH
        full_path_to_dir = os.path.join(path_to_dir, each_dir)    
        if verbose is True:
            print("Working on:")
            print(full_path_to_dir)
        ## RECOMUTING MU VALUES
        mu_dist, mu, renorm_mu, grid = compute_mu_from_neighbors_for_one_directory(full_path_to_dir = full_path_to_dir, **args)
    return mu_dist, mu, renorm_mu, grid


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":

    
#    ## SEE IF TESTING IS ON
#    testing = check_testing()
    
    ## DEFINING PARENT SIM PATH
    parent_sim_path = PARENT_SIM_PATH
    
#    ## RUNNING TESTING    
#    if testing == True:
    ## DEFINING DIRECTORY NAME
    dir_name = "20200224-GNP_spr_50"
    dir_name = "20200224-planar_SAM_spr50"
    
    dir_list = [
                # "20200224-planar_SAM_spr50",
                # "20200224-GNP_spr_50",
                # "20200326-Planar_SAMs_with_larger_z_frozen_with_vacuum"
                # "20200224-GNP_spr_50",
                "20200328-Planar_SAMs_new_protocol-shorterequil_spr50"
                ]
    
    mu_list = [
            [5, 15],
            [5, 15],
#            [5,12],
#           [2.5,5],
            ]
    
    ## DEFINING EXTACTION DIR
    extraction_dir_list=[
            "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000",
            "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000"
            # "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-50000",
#            "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-150000",
#            "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000",
            ]
    
    ## DEFINING WC INTERFACE
    grid_dir = "grid-49000_50000"
    # GRID_LOC
    grid_output_file = GRID_OUTFILE
    
    ## LOOPING THROUGH TRAJECTORY
    for dir_idx, dir_name in enumerate(dir_list):

        ## DEFINING NAME TO 
        extraction_dir = extraction_dir_list[dir_idx]
        
        ## DEFINING COMBINING NEIGHBORS DIR
        combine_neighbors_dir = COMBINE_NEIGHBORS_DIR
        combine_neighbors_log = COMBINE_NEIGHBORS_LOG
        
        ## GETTING MU VALUES
        min_mu = mu_list[dir_idx][0]
        max_mu = mu_list[dir_idx][1]
        
        print("Min mu: %.2f"%(min_mu) )
        print("Max mu: %.2f"%(max_mu) )

        
        ## DEFINING OUTPUT MU PICKLE
        output_pdb = OUT_HYDRATION_PDB
        output_mu = MU_PICKLE
        
        ## ADDING TO PDB AND MU FILE NAMES 
        str_included= '-%d_%d'%(min_mu, max_mu)
        output_pdb = os.path.splitext(output_pdb)[0] + str_included +  os.path.splitext(output_pdb)[1]
        # output_mu = os.path.splitext(output_mu)[0] + str_included +  os.path.splitext(output_mu)[1]
        
        ## DEFINING INPUT GRO FILE
        input_gro_file = PROD_GRO
    
        ## DEFINING PATH TO DIRECTORY
        path_to_dir = os.path.join(parent_sim_path, dir_name )
        print(path_to_dir)
        
        ## GETTING MU AND GRID
        mu_dist, mu, renorm_mu, grid = compute_mu_from_neighbors_array_dir_list(path_to_dir = path_to_dir,
                                                                                min_mu = min_mu,
                                                                                max_mu = max_mu,
                                                                                extraction_dir = extraction_dir,
                                                                                output_pdb = output_pdb,
                                                                                output_mu = output_mu,
                                                                                num_neighbor_array = None,
                                                                                grid_dir = grid_dir,
                                                                                )

            
            