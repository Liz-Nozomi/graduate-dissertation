# -*- coding: utf-8 -*-
"""
combine_neighbors_array.py
The purpose of this script is to combine all the neighbor array codes into a
single pickle file. This way, we could easily access all the data and clean 
up any miscellaneous files. 

Written by: Alex K. Chew (01/03/2020, alexkchew@gmail.com)

"""

## IMPORTING MODUELS
import os
import pandas as pd
import numpy as np
import sys

## CHECK PATH
from MDDescriptors.core.check_tools import check_path

## READING FILES
from MDDescriptors.core.import_tools import read_file_as_line

## IMPORTING GRO READING
from MDDescriptors.core.read_write_tools import extract_gro

## PICKLE FUNCTIONS
from MDDescriptors.core.pickle_tools import load_pickle_results, pickle_results

## IMPORTING NUM DIST FUNCTION
from MDDescriptors.surface.generate_hydration_maps import compute_num_dist

## IMPORTING FUNCTIONS FROM PARALLEL
from MDDescriptors.surface.generate_hydration_maps_parallel import get_pickle_name_for_itertool_traj, update_neighbor_log

## CHECK TESTING FUNCTIONS
from MDDescriptors.core.check_tools import check_testing 

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import MAX_N, MU_MIN, MU_MAX

## CALC MU FUNCTION
from MDDescriptors.surface.core_functions import calc_mu, load_datafile, create_pdb

## READING MU DISTRIBUTION
from MDDescriptors.application.np_hydrophobicity.regenerate_mu_distribution import read_neighbor_logs, compute_mu_from_neighbors_for_one_directory


### FUNCTION TO EXTRACT PICKLE NAME FROM FILE
def extract_pickle_name_for_itertool_traj(pickle_name):
    '''
    The purpose of this function is to decode the pickle names
    INPUTS:
        pickle_name: [str]
            pickle name to save for itertools
    OUTPUTS:
        name_dict: [dict]
            dictionary clarifying the meaning of the name
            contains initial and final frame
    '''
    ## SPLITTING
    split_list = pickle_name.split('.')[0].split('-')
    # RETURNS: ['0', '50000']
    
    ## GENERATING NAME DICT
    name_dict={
            'initial': split_list[0],
            'final': split_list[1],
            }
    
    
    return name_dict

### FUNCTION TO COMBINE ALL LOGS
def combine_all_pickles(path_pickle_log,
                        path_pickle,
                        save_space=False,):
    '''
    The purpose of this function is to combine all pickles into a single 
    pickle. This is useful when you are trying to load a single pickle 
    file and could save space. 
    INPUTS:
        path_pickle_log: [str]
            path to the pickle log file
        path_pickle: [str]
            path to the pickle folder
        save_space: [logical]
            True if you want to save space by removing extra pickles
            
    OUTPUTS:
        void -- no outputs. 
    '''

    ## READING LOG FILE
    log = read_neighbor_logs(path_pickle_log)
    
    ## RUNNING COMMAND IF LOG IS > 1, MEANING THAT WE HAVE MULTIPLE FILES
    if len(log) > 1:
        ## GETTING DICTIONARY FOR EACH
        name_dicts = [ extract_pickle_name_for_itertool_traj(each_log[1]) for each_log in log ]
        
        ## GETTING DATAFRAME
        name_pd = pd.DataFrame(name_dicts).astype('int32')
        
        ## PRINTING DATABASE
        print(name_pd)
        
        ## GETTING LOW AND HIGH
        min_frame = np.min(name_pd['initial'])
        max_frame = np.max(name_pd['final'])
        
        ## PRINTING MIN AND MAX FRAME
        print("Min frame is: %d"%(min_frame) )
        print("Max frame is: %d"%(max_frame) )
        
        ## GETTING DESIRED PICKLE NAME
        final_pickle_name = get_pickle_name_for_itertool_traj(traj_initial = min_frame,
                                                              traj_final = max_frame)
        
        ## DEFINING PATH TO FINAL PICKLE
        path_final_pickle = os.path.join(path_pickle,
                                         final_pickle_name)
        
        
        ## IF FINAL PICKLE EXISTS, STOP HERE
        if os.path.exists(path_final_pickle):
            print("Pickle file already exists, but log is >1!")
            print("Stopping here to prevent error of overwritting ")
            print("Path final pickle: %s"%(path_final_pickle) )
            sys.exit(1)
        
        ## CREATING EMPTY ARRAY
        results_list = []
        
        ## LOOPING THROUGH EACH LOG
        for each_file in log:
            ## LOADING PICKLE FILE
            pickle_file = each_file[1]
    
            ## PATH TO PICKLE
            path_to_specific_pickle = os.path.join(path_pickle,
                                                   pickle_file)
            
            ## LOADING PICKLE
            results = load_pickle_results(path_to_specific_pickle)[0][0]
            ## APPENDING
            results_list.append(results)
            
        ## CREATING A NUMPY ARRAY OF THE RESULTS
        combined_results = np.concatenate(results_list, axis = 1)
        
        ## STORING PICKLE RESUTLS
        pickle_results(results = [combined_results],
                       pickle_path = path_final_pickle,
                       verbose = True,
                       )
        
        ## SAVING SPACE
        if save_space is True and os.path.exists(path_final_pickle):
            ## LOOPING THROUGH EACH LOG
            for each_file in log:
                ## LOADING PICKLE FILE
                pickle_file = each_file[1]
                ## PATH TO PICKLE
                path_to_specific_pickle = os.path.join(path_pickle,
                                                       pickle_file)
                
                ## REMOVING
                os.remove(path_to_specific_pickle)
                
        ## UPDATING NEIGHBOR LOGS
        if os.path.exists(path_final_pickle):
            update_neighbor_log(path_pickle_log = path_pickle_log,
                                index = 0,
                                pickle_name = final_pickle_name,
                                )
    else:
        print("Since log only has one entry, no changes have been made to combine the pickles")
        
        ## PATH TO PICKLE
        pickle_file = log[0][1]
        path_to_specific_pickle = os.path.join(path_pickle,
                                               pickle_file)
        ## LOADING THE PICKLE FILE
        combined_results = load_pickle_results(path_to_specific_pickle)[0][0]
        
    return combined_results






#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    hydrophobic_side_only = False
    single_sam = False
    
    ## DEFINING MAX N
    max_N = MAX_N
    
    ## DEFINING MIN AND MAX MU
    min_mu = MU_MIN
    max_mu = MU_MAX
    
    ## TYPE
    analysis_type = "new"
    # "old"
    
    ## RUNNING TESTING    
    if testing == True:
    
        ## DEFINING MAIN SIMULATION
        main_sim=check_path(r"R:\simulations\np_hydrophobicity\unbiased")
        ## DEFINING SIM NAME
        sim_name=r"np_planar_300K_dodecanethiol_CHARMM36"
        ## DEFINING WORKING DIRECTORY
        simulation_path = os.path.join(main_sim, sim_name)
        ## DEFINING GRO AND XTC
        gro_file = "sam_prod.gro"
        hydrophobic_side_only = False
        single_sam = True
        
        min_mu = 6.5
        max_mu = 9.5
        
        ## DEFINING ANALYSIS FOLDER
        analysis_folder="30.0-0.24-0.1,0.1,0.1-0.33-all_heavy"
        
        ## DEFINING LOCATION TO STORE PICKLES
        pickle_folder = "compute_neighbors"
        pickle_log = "neighbors.log"
        
        ## DEFINING GRID FILE
        grid_folder="grid-0_1000"
        grid_dat_file="out_willard_chandler.dat"
        
        ## DEFINING OUTPUT PPDB
        output_pdb="mu.pdb"
        
        ## PATH TO PDB
        path_pdb_file = os.path.join(simulation_path, analysis_folder, output_pdb)
        
        
        ## DEFINING PATH TO GRO
        path_gro = os.path.join(simulation_path, gro_file)
        
        ## DEFINING PATH TO GRID
        path_grid= os.path.join(simulation_path,
                                analysis_folder,
                                grid_folder,
                                grid_dat_file
                                )
        
        ## DEFINING PATH TO PICKLE FOLDER
        path_pickle = os.path.join(simulation_path,
                                   analysis_folder,
                                   pickle_folder,
                                   )
        
        ## LOG FILE
        path_pickle_log = os.path.join(path_pickle,
                                       pickle_log)
        
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## DEFINING OUTPUT PICKLE PATH
        parser.add_option('--path_pickle', dest = 'path_pickle', help = 'Path of pickle', default = '.', type=str)
        
        ## DEFINING OUTPUT PICKLE PATH
        parser.add_option('--path_pickle_log', dest = 'path_pickle_log', help = 'Path of pickle log', default = '.', type=str)
        
        ## GRIDING FILE
        parser.add_option('--path_grid', dest = 'path_grid', help = 'Path to willard chandler', default = 'output', type=str)        
        
        ## GRO FILE
        parser.add_option('--path_gro', dest = 'path_gro', help = 'Path of GRO file', default = '.', type=str)
        
        ## OUTPUT PDB FILE
        parser.add_option('--path_pdb_file', dest = 'path_pdb_file', help = 'Path of output PDB file', default = '.', type=str)
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## DEFINING INPUTS
        path_pickle = options.path_pickle
        path_pickle_log = options.path_pickle_log
        path_gro = options.path_gro
        path_pdb_file = options.path_pdb_file
        path_grid = options.path_grid
    
    ## DEFINING MU LOCATION
    mu_pickle_name = "mu.pickle"
    
    ## COMBINING ALL PICKLES
    combined_results = combine_all_pickles(path_pickle_log = path_pickle_log,
                                           path_pickle = path_pickle,
                                           save_space = True, # True if saving space is desired
                                           )
    ## LOADING GRID
    grid = load_datafile(path_to_file = path_grid)
    if analysis_type == "new":
        ## PRINTING
        print("Running analysis on new method -- removal of all shifting functions")
        ## DEFINING PATH TO DIR
        path_to_dir = os.path.dirname(path_gro)
        input_gro_file = os.path.basename(path_gro)
        output_pdb = os.path.basename(path_pdb_file)
        
        ## DEFINING EXTRACTION DIRECTORY
        extraction_dir = os.path.basename(os.path.dirname(path_pickle))
        
        output_mu = mu_pickle_name
        
        ## GETTING MU FROM ONE DIRECTORY
        mu_dist, mu, renorm_mu, grid = compute_mu_from_neighbors_for_one_directory(full_path_to_dir = path_to_dir,
                                                                                   min_mu = min_mu,
                                                                                   max_mu = max_mu,
                                                                                   extraction_dir = extraction_dir,
                                                                                   output_pdb = output_pdb,
                                                                                   output_mu = output_mu,
                                                                                   input_gro_file = input_gro_file,
                                                                                   num_neighbor_array = combined_results,
                                                                                   grid = grid)
        
        
    else:
        #%% GETTING NUMBER DISTRIBUTION
        num_neighbor_dist = compute_num_dist(num_neighbors_array = combined_results,
                                             max_neighbors = max_N)
        ## LOADING GRO FILE
        gro_file = extract_gro(gro_file_path = path_gro)
    
        ## COMPUTE THE DISTRIBUTION FOR THE HYDROPHOBIC SIDE OF A PEPTIDE
        if hydrophobic_side_only is True:
            peptide_atoms = np.array([ ndx for ndx, name in enumerate(gro_file.ResidueName) if name not in [ "SOL", "MET", "HOH", "CL" ] ])
            third_backbone_carbon = [ ndx for ndx, atom_name in enumerate(gro_file.AtomName) if "C" in atom_name and ndx in peptide_atoms ][2]
            last_backbone_carbon = [ ndx for ndx, atom_name in enumerate(gro_file.AtomName) if "C" in atom_name and ndx in peptide_atoms ][-1]
            above_x = grid[:,0] > gro_file.xCoord[third_backbone_carbon]
            below_x = grid[:,0] < gro_file.xCoord[last_backbone_carbon]
            below_z = grid[:,2] < 0.5*gro_file.Box_Dimensions[2]-0.2
            mask = above_x * below_x * below_z
            grid = grid[mask,:]
            num_neighbor_dist = num_neighbor_dist[mask,:]
        
        ## COMPUTING MU
        mu = calc_mu( p_N_matrix = num_neighbor_dist,
                      d = max_N)
        
        ## STORING MU PICKLE
        path_mu = os.path.join( os.path.dirname(path_pdb_file),
                                mu_pickle_name)
        
        ## PICKLING MU VALUES
        pickle_results( results = [mu],
                        pickle_path = path_mu)
        
        ## RENOMRALIZE MU VALUE
        renorm_mu = ( mu - min_mu ) / ( max_mu - min_mu ) # red = hydrophobic; blue = hydrophilic
        
        ## WRITE PDB FILE WITH PROBES COLORED BY MU
        create_pdb(data = grid,
                   box_dims = gro_file.Box_Dimensions,
                   b_factors = renorm_mu,
                   path_pdb_file = path_pdb_file
                   )
        
        x = np.arange( 0, 20.5, 0.5 )
        y = np.histogram( mu, bins = len(x), range = (0,20) )[0]
        y =  y / y.sum()
        print( "mean: %.3f" % np.mean(mu) )
    
    