# -*- coding: utf-8 -*-
"""
publish_images.py
The purpose of this function is to generate publishable images for the nanoparticle 
hydrophobicity project. 

Written by: Alex K. Chew (02/13/2020)

"""


import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import copy

## STATISTICS    
from scipy.stats import kurtosis, skew
    
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path

## PICKLING FUNCTIONS
import MDDescriptors.core.pickle_tools as pickle_tools

## IMPORT TOOLS
import MDDescriptors.core.import_tools as import_tools

## PICKLING FUNCTIONS
from MDDescriptors.core.pickle_tools import load_pickle_results, save_and_load_pickle, pickle_results
import MDDescriptors.core.plot_tools as plot_tools
from MDDescriptors.core.decoder import decode_name

## IMPORTING EXTRACT TRAJ FUNCTOIN
from MDDescriptors.traj_tools.loop_traj_extract import extract_multiple_traj

### FUNCTIONS 
from MDDescriptors.application.np_hydrophobicity.compute_rmsf_of_ligands import main_compute_gmx_rmsf
from MDDescriptors.application.np_hydrophobicity.compute_wc_grid_for_multiple_times import main_compute_wc_grid_multiple_subset_frames

## DISTANCE METRICS
from scipy.spatial import distance

## ADDING PEARSONS R
from scipy.stats import pearsonr

## LOADING DAT FILE FUNCTION
from MDDescriptors.surface.core_functions import load_datafile


## IMPORTING GLOBAL VARIABLES
from MDDescriptors.application.np_hydrophobicity.global_vars import \
    PARENT_SIM_PATH, PATH_DICT, RELABEL_DICT, MU_PICKLE, LIGAND_COLOR_DICT, \
    PURE_WATER_SIM_DICT, PREFIX_SUFFIX_DICT, DEFAULT_WC_ANALYSIS, GRID_LOC, GRID_OUTFILE, \
    NP_SIM_PATH
    
## LOADING MODULES
from MDDescriptors.application.np_hydrophobicity.analyze_mu_distribution import \
    extract_hydration_maps, compute_histogram_for_mu, plot_histogram_data


## DEBUGGING WC INTERFACE
from MDDescriptors.application.np_hydrophobicity.debug_wc_interface import debug_wc_interface

## IMPORTING MODULES FOR REMOVING PLANAR SAM GRIDS
from MDDescriptors.application.np_hydrophobicity.remove_grids_for_planar_SAMs import remove_grid_for_planar_SAMs, extract_new_grid_for_planar_sams

## NANOPARTICLE TOOLS
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure

## IMPORTING GLOBAL VARS
from MDDescriptors.application.nanoparticle.global_vars import GOLD_ATOM_NAME

## IMPORTING NUM DIST FUNCTION
from MDDescriptors.surface.generate_hydration_maps import compute_num_dist

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import R_CUTOFF, MAX_N

## IMPORTING COMPUTE MU
from MDDescriptors.application.np_hydrophobicity.analyze_mu_distribution import \
    compute_mu_from_unnorm_p_N, debug_mu_convergence, plot_convergence_of_mu
    
## COSOLVENT MAPPING
## MAIN FUNCTION
from MDDescriptors.application.np_hydrophobicity.compute_np_cosolvent_mapping import main_compute_np_cosolvent_mapping, find_original_np_hydrophobicity_name

## LOADING MODULES
from MDDescriptors.application.np_hydrophobicity.compute_np_cosolvent_mapping_extract import \
    get_all_paths, load_frac_of_occur, sum_frac_of_occurences, plot_gaussian_kde_scatter, \
    generate_sampling_occur_dict, compute_sampling_time_frac_occur, plot_sampling_time_df, \
    plot_min_sampling_time, compute_min_sampling_time
    
## DEFAULTS
plot_tools.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_tools.FIGURE_SIZES_DICT_CM['1_col_landscape']

#####################################################
### PATH DEFINITIONS
#####################################################

## DEFINING IMAGE LOCATION
IMAGE_LOC = r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_hydrophobicity_manuscript/Figures/svg_output"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200309\images\np_hydro_project"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200224\images"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200217\images\bundling"

## DEFINING MAIN DIRECTORY
MAIN_NP_DIR = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
MAIN_HYDRO_DIR  = PARENT_SIM_PATH
# check_server_path(r"S:\np_hydrophobicity_project\simulations")

## DEFINING PATH TO INDUS
PATH_TO_INDUS_SI = r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_hydrophobicity_manuscript/Excel_Spreadsheet/from_Brad/SI_Indus_data.xlsx"

## DEFINING PATH TO CSV FILE
PATH_TO_CONTACT_ANGLES = r"/Volumes/shared/np_hydrophobicity_project/csv_files/contact_angles.xlsx"

## GENERATING COLORS FOR LIGAND
LIGAND_LINESTYLES = {
        'dodecanethiol': "solid",
        "C11NH3": "dotted",
        }

LIGAND_COLORS = {    
        'dodecanethiol': "k",
        "C11NH3": "b",
        "C11OH": "r",
        }
## GENERATING LINE STYLES
SPRING_CONSTANT_COLORS = {
        "25": "red",
        "50": "black",
        "100": "blue",
        '500': "magenta",
        '1000': 'green',
        "2000": "cyan",
        '10000': 'yellow',
        }

## DEFINING ANALYSIS DIRECTORY
ANALYSIS_FOLDER = "analysis"
## DEFINING PICKLE NAME
RESULTS_PICKLE = "results.pickle"


## DEFINING SIMULATION DIRECTORY
PATH_TO_NP_WATER = os.path.join(NP_SIM_PATH,
                                "EAM_COMPLETE")

## DIRECTORY FOR MOST LIKELY
MOST_LIKELY_DIR="most_likely-avg_heavy_atom"
MOST_LIKELY_PICKLE="np_most_likely.pickle"

## DEFINING DEBUG PICKLE
DEBUG_PICKLE = "debug.pickle"

## DEFINING SAVE FIG
SAVE_FIG = False

## DEFINING DEFAULT SCATTER STYLE
DEFAULT_SCATTER_FOR_ERRORBAR = {
        'fmt': '.',
        'linestyle': None,
        'markersize': 12,
        }

## DEFINIG INDUS DICT
INDUS_RESULTS_DICT = {
        'CF3': {
                'value': 46.3845,
                'error': 2.4205,
                },
        'CH3': {
                'value': 37.1225,
                'error': 0.8355,
                },
        'NH2': {
                'value': 85.759,
                'error': 4.486,
                },
        'CONH2': {
                'value': 94.535,
                'error': 3.872,
                },
        'OH': {
                'value': 94.9315,
                'error': 1.9325,
                },
        'COOH': {
                'value': 111.5855,
                'error': 1.1495,
                },
        }


###############################################################################
### PLOTTING TOOLS 
###############################################################################
    
### FUNCTION TO LOAD
def compute_rmsf_across_traj(traj_output):
    '''
    This function analyzes RMSF across trajectories
    INPUTS:
        traj_output: [obj]
            trajectory output object
    OUTPUTS:
        traj_output: [obj]
            updated object with avg rmsf
            
    '''
    ## STORING VECTOR
    avg_rmsf_list = []
    
    ### LOOPING THROUGH EACH SIM
    for idx, each_sim in enumerate( traj_output.full_sim_list ):
        ## LOOPING THROUGH EACH LIGAND AND GENERATING DATA
        results = traj_output.load_results(idx = idx,
                                           func = main_compute_gmx_rmsf)[0]
        ## APPENDING
        avg_rmsf_list.append(results.avg_rmsf)
        
    ## STORING
    traj_output.decoded_name_df['avg_rmsf'] = avg_rmsf_list
    
    return traj_output


### FUNCTION TO PLOT RMSE VS SPRING CONSTANT
def plot_rmse_vs_spring_constant(df,
                                 fig = None,
                                 ax = None):
    '''
    The purpose of this function is to plot the RMSE versus spring constant.
    INPUTS:
        df: [pd.dataframe]
            dataframe with decoded name information
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''    
    df = traj_output.decoded_name_df
    
    ## PLOTTING
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_xlabel("Spring constant")
    ax.set_ylabel("Avg. ligand RMSF")
    
    ## GETTING UNIQUE LIGANDS
    unique_ligand_list = np.unique(df['ligand'])
    
    ## LOOPING THROUGH LIGAND
    for ligand_type in unique_ligand_list:
        ## GETTING DATAFRAME
        current_df = df[df.ligand == ligand_type]
        
        ## CONVERTING
        current_df = current_df.astype({'spring_constant': float,
                                        'avg_rmsf': float})
        
        ## SORTING
        current_df = current_df.sort_values(by=['spring_constant'])
        
        ## GETTING ALL SPRING CONSTANTS
        spring_constant_array = current_df.spring_constant.to_numpy()
        avg_rmsf_array = current_df.avg_rmsf.to_numpy()
        
        ## DEFINING COLOR
        color = LIGAND_COLORS[ligand_type]
        
        ## PLOTTING
        ax.errorbar(x = spring_constant_array, 
                    y = avg_rmsf_array,
                    color = color,
                    label = ligand_type,
                    **DEFAULT_SCATTER_FOR_ERRORBAR)
    
    ## ADDING LEGEND
    ax.legend()
    
    ## FITTING
    fig.tight_layout()
    return fig, ax

### FUNCTION TO COMPARE TWO GRIDS
def compare_two_grids(ref_grid, 
                      new_grid,
                      compare_type = "min_dist", 
                      verbose = True):
    '''
    This function compares two grids: new and reference grid. It will use 
    many different types to generate multiple comparison between grids.
    INPUTS:
        ref_grid: [np.array]
            grid in x, y, z coordinates. This is the reference grid
        new_grid: [np.array]
            grid in x, y, z coordinates. This is the grid of testing
        compare_type: [str]
            type of comparison you would like. Types are:
                "min_dist": 
                    computes average minimum distance between the grid points
                "z_dist":
                    computes average of each grid z-dimension and outputs the difference
    OUTPUTS:
        diff: [float]
            single value representing the difference
    '''
    ## DEFINING AVAILABLE TYPES
    available_types = ["min_dist", "z_dist"]
    
    if compare_type == "min_dist":
        print("Comparing via min distances")
        ## COMPUTING DISTANCE BETWEEN GRID POINTS
        dist_array = distance.cdist(ref_grid, new_grid)
            
        ## GETTING MINMUM
        min_array = dist_array.min(axis=1)
        
        ## GETTING AVG
        diff = np.mean(min_array)
        
    elif compare_type == "z_dist":
        print("Comparing via z-distances")
        ## COMPUTING MEAN Z VALUES
        mean_z_values = [ np.mean(each_grid, axis = 0)[2] for each_grid in [ref_grid, new_grid]]
        
        ## OUTPUTTING THE DIFFERENCE
        diff = np.abs(mean_z_values[1] - mean_z_values[0])
        
    else:
        print("Error, compare type %s is not available"%(compare_type) )
        print("Available types are: %s "%(", ".join(available_types)) )
    
    return diff

### FUNCTION TO COMPUTE AVG DEVIATION
def compute_avg_deviation(results, 
                          ref_grid = None,
                          compare_type = 'min_dist',
                          use_last_grid_ref = False
                          ):
    '''
    Computes avg deviation when you have the results from willard-chandler gridding
    INPUTS:
        results: [obj]
            results object
        ref_grid: [dict]
            grid in x, y, z dictionary. If this is None, then we will use the first 
            index as a reference
        compare_type: [str]
            type of comparison between the two grids
        use_last_grid_ref: [logical]
            True if you want to use the last grid as a reference
    OUTPUTS:
        avg_deviation: [float]
            average deviation
    '''
    if ref_grid is None:
        ## GETTING REFERENCE
        if use_last_grid_ref is False:
            ref_grid = results[0][0]
            index_array = np.arange(1, len(results[0]))
        else:
            print("Using last grid index as reference")
            ref_grid = results[0][-1]
            index_array = np.arange(0, len(results[0])-1)
    ## STORING RESULTS
    diff_storage = []
    
    ## LOOPING THROUGH GRID
    for index in index_array:
        ## DEFINING NEW GRID
        new_grid = results[0][index]
        
        ## GETTING AVG
        diff = compare_two_grids(ref_grid = ref_grid,
                                 new_grid = new_grid,
                                 compare_type = compare_type)
        
        ## STORING
        diff_storage.append(diff)
    
    ## DEFINING AVG DEVIATION
    avg_deviation = np.mean(diff_storage)
    
    return avg_deviation

### FUNCTION TO PLOT AVG DEVIATION VS. SPRING CONSTANT
def plot_avg_deviation_vs_spring_constant(df,
                                          fig = None,
                                          ax = None,
                                          ):
    '''
    This function plots the data from traj output. In particular, we are 
    interested in the average deviation verses psring constant. 
    INPUTS:
        df: [obj]
            datafrmae object
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''
    
    
    df = traj_output.decoded_name_df
    
    ## PLOTTING
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_xlabel("Spring constant")
    ax.set_ylabel("Avg. deviation")
    
    ## GETTING UNIQUE LIGANDS
    unique_ligand_list = np.unique(df['ligand'])
    
    ## LOOPING THROUGH LIGAND
    for ligand_type in unique_ligand_list:
        ## GETTING DATAFRAME
        current_df = df[df.ligand == ligand_type]
        
        ## CONVERTING
        current_df = current_df.astype({'spring_constant': float,
                                        'avg_deviation': float})
        
        ## SORTING
        current_df = current_df.sort_values(by=['spring_constant'])
        
        ## GETTING ALL SPRING CONSTANTS
        spring_constant_array = current_df.spring_constant.to_numpy()
        y_array = current_df.avg_deviation.to_numpy()
        
        ## DEFINING COLOR
        color = LIGAND_COLORS[ligand_type]
        
        ## PLOTTING
        ax.errorbar(x = spring_constant_array, 
                    y = y_array,
                    color = color,
                    label = ligand_type,
                    **DEFAULT_SCATTER_FOR_ERRORBAR)
    
    ## ADDING LEGEND
    ax.legend()

    ## FITTING
    fig.tight_layout()
    
    return fig, ax


### FUNCTION TO COMPUTE AVG DEVIATION
def compute_avg_deviation_across_traj(traj_output,
                                      ref_grids = None,
                                      compare_type = 'min_dist',
                                      is_planar = False,
                                      verbose = True,
                                      prod_gro="sam_prod",
                                      omit_first = False,
                                      use_last_grid_ref = False):
    '''
    This script computes the average deviation of wc grid points given the 
    path. Note that we ca also compute the deviations relative to a reference 
    grid. This reference is useful when comparing between grids of different 
    simulations. 
    INPUTS:
        traj_output: [obj]
            trajectory object
        ref_grids: [dict]
            dictionary of reference grids. If this is None, then we will 
            just take the first chunk of the willard-chandler interface 
            to be our reference value. 
        compare_type: [str]
            type of comparison between the two grids
        is_planar: [logical]
            True if you have a planar SAM. If so, then we need to remove 
            grid points above and below the SAM.
        prod_gro: [str]
            default gro file to load if "is_planar" is True
        omit_first: [logical]
            True if you want to omit the first index
        use_last_grid_ref: [logical]
            True if you want to use the last grid as reference
    OUTPUTS:
        traj_output: [obj]
            trajectory output object. Note that we will save the deviations 
            into a class object.         
    '''
    ## STORING VECTOR
    storage_list = []
    
    ### LOOPING THROUGH EACH SIM
    for idx, each_sim in enumerate( traj_output.full_sim_list ):
        ## PRINTING
        print("Working on %s"%(each_sim))
        ## PRINTING
        if verbose is True:
            print("Grid comparison type used: %s"%(compare_type) )
        ## DEFINING REFERENCE GRID
        if ref_grids is not None:
            ## GETTING LIGAND NAME
            lig = traj_output.decoded_name_df.loc[idx]['ligand']
            current_ref_grid = ref_grids[lig]
        else:
            current_ref_grid = None
            
        ## GETTING THE RESULTS
        results = traj_output.load_results(idx = idx,
                                           func = main_compute_wc_grid_multiple_subset_frames)
        
        if omit_first is True:
            print("Omitting first trajectory!")
            results = (results[0][1:], results[1][1:])
        
        ## CHECKING IF PLANAR IS TURNED ON
        if is_planar is True:
            print("Since planar is turned on, removing grid points that are above or below waters")
            ## DEFINING NEW ARRAY
            new_grid_array = []
            
            path_gro = os.path.join(each_sim, "sam_prod.gro")
            updated_grid = remove_grid_for_planar_SAMs(path_gro = path_gro,
                                                       grid = results[0][0])
            
            ## FINDING UPPER AND LOWER GRID
            upper_grid, lower_grid = updated_grid.find_upper_lower_grid(grid = updated_grid.new_grid)
            
            ## STORING
            new_grid_array.append([upper_grid[:], lower_grid[:] ])
            
            ## LOOPING THROUGH THE OTHERS
            for each_idx in np.arange(1,len(results[0])):
                ## UPDATING
                new_grid = updated_grid.find_new_grid(grid = results[0][each_idx])
                ## FINDING UPPER AND LOWER GRID
                upper_grid, lower_grid = updated_grid.find_upper_lower_grid(grid = new_grid)
                ## APPENDING
                new_grid_array.append([upper_grid[:], lower_grid[:]])
            
            ## GETTING DIFERENCES
            diff_array = []
            
            ## LOOPING THROUGH EACH
            for idx, each_grid_array in enumerate(new_grid_array):
                if idx == 0:
                    planar_ref_grid_list = [each_grid_array[0], each_grid_array[1]]
                else:
                    new_grids = [ each_grid_array[0], each_grid_array[1] ]
                    ## FINDING DIFFERENCES
                    for ref_idx, planar_ref_grid in enumerate(planar_ref_grid_list):
                        ## COMPUTING DISTANCES
                        diff = compare_two_grids(ref_grid = planar_ref_grid,
                                                 new_grid = new_grids[ref_idx],
                                                 compare_type = 'z_dist',
                                                 )
                        
                        ## STORING
                        diff_array.append(diff)
            
            ## GETTING AVERAGE DEVIATIONS
            avg_deviation = np.mean(diff_array)
            
#            ## LASTLY, UPDATE THE RESULTS
#            grids_summary = (new_grid_array, results[1])
        else:
            grids_summary = results

            ## COMPUTING DEVIATION
            avg_deviation = compute_avg_deviation(results = grids_summary,
                                                  ref_grid = current_ref_grid,
                                                  compare_type = compare_type,
                                                  use_last_grid_ref = use_last_grid_ref)
        ## STORING
        storage_list.append(avg_deviation)
    
    ## STORING
    traj_output.decoded_name_df['avg_deviation'] = storage_list[:]
    
    return traj_output

    
### FUNCTION TO GET MU VALUE FOR BULK WATER
def get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                sim_path = MAIN_HYDRO_DIR):
    '''
    This function gets the mu value for the bulk water. 
    INPUTS:
        water_sim_dict: [dict]
            dictionary containing information of the:
                'main_dir': directory it is inside, e.g. 'PURE_WATER_SIMS'
                'parent_dir': directory that is used, e.g. 'wateronly-6_nm-tip3p-300.00_K'
                'wc_folder': folder that is used for the analysis of wc, e.g. '0.24-0.33-2000-50000',
                'mu_pickle': mu pickle that is within the folder
        sim_path: [str]
            path to the simulation folder
    OUTPUTS:
        mu_value: [list]
            list of mu values
    '''
    ## DEFINING PATH TO MU
    path_to_mu = os.path.join(sim_path,
                              water_sim_dict['main_dir'],
                              water_sim_dict['parent_dir'],
                              water_sim_dict['wc_folder'])
    
    ## LOADING PICKLE
    hydration_map = extract_hydration_maps()
    mu_values = hydration_map.load_neighbor_values(main_sim_list = [path_to_mu],
                                                       pickle_name = water_sim_dict['mu_pickle'])

    return mu_values

### GENERALIZED MU MOMENTS WITHOUT CENTRALIZING
def nmoment(x, n, c = 0, counts = None):
    '''
    This function computes the moments that are non-centralized, in other words, 
    mean is considered at 0
    INPUTS:
        x: [np.array]
            array of x values
        n: [int]
            moment that you desire
        c: [float]
            central point of your moment
        counts: [np.array, default=None]
            weight vector for your x arrays
    '''
    
    ## COUNTING
    if counts is None:
        counts = np.ones(len(x))
    
    return np.sum(counts*(x-c)**n) / np.sum(counts)

### FUNCTION TO COMPUTE STATISTICS FOR MU
def compute_mu_stats(mu,
                     want_moments = True,
                     want_uncentralized_moments = True):
    '''
    This function computs the statistics for mu values.
    
    ##  REFERENCE FOR DIFFERENT MOMENTS:
    # https://medium.com/@praveenprashant/the-four-moments-of-a-probability-distribution-6b900a25d0d8
    # Example of using skew and kurtosis
    # https://stackoverflow.com/questions/45483890/how-to-correctly-use-scipys-skew-and-kurtosis-functions
    
    INPUTS:
        mu: [np.array]
            mu array
        want_moments: [logical]
            True if you want moments 1, 2, 3, 4, etc.
    OUTPUTS:
        stats_dict: [dict]
            dictionary of statistics 
                'mean': mean of data set
                'var': variance of the data set
                'skew': skewness of the data set (3rd moment)
                'kurt': kurtosis of the data set (4th moment)
    '''
    from scipy import stats
    stats_dict = {
            'mean': np.mean(mu),
            'median': np.median(mu),
            'var': np.var(mu),
            'skew': skew(mu),
            'kurt': kurtosis(mu),
            'mode': stats.mode(mu, axis = None),
            }
    moment_array = np.arange(1,5,1)
    if want_moments is True:
        from scipy.stats import moment
        for moment_idx in moment_array:
            stats_dict['moment_%d'%(moment_idx)] = moment(mu, moment = moment_idx)
            
    if want_uncentralized_moments:
        ## LOOPING THROUGH MOMENT INDEX
        for moment_idx in moment_array:
            stats_dict['uncentered_moment_%d'%(moment_idx)] = nmoment(x = mu,
                                                                      n = moment_idx,
                                                                      c = 0 )
            ## NORMALIZED
            stats_dict['normalized_uncentered_moment_%d'%(moment_idx)] = stats_dict['uncentered_moment_%d'%(moment_idx)]**(1/moment_idx)
            
            ## RATIO
            stats_dict['ratio_uncentered_moment_%d'%(moment_idx)] = nmoment(x = mu,
                                                                          n = moment_idx,
                                                                          c = 0 ) / nmoment(x = mu,
                                                                                      n = moment_idx - 1,
                                                                                      c = 0 )
    return stats_dict

### FUNCTION TO LOAD MU VALUES FOR A SET
def load_mu_values_for_multiple_ligands(path_list,
                                        ligands = [],
                                        histogram_range = (0, 20),
                                        histogram_step_size = 0.5,
                                        main_sim_dir=MAIN_HYDRO_DIR,
                                        wc_analysis="26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",
                                        mu_pickle = MU_PICKLE,
                                        want_avg = True,
                                        want_mu_array = False,
                                        want_stats = True,
                                        want_grid = False,
                                        relative_path_to_wc = os.path.join(GRID_LOC, GRID_OUTFILE),
                                        ):
    '''
    This function loads mu values for multiple ligands.
    INPUTS:
        path_list: [dict]
            dictionary with the path prefix information
        ligands: [list]
            list of ligands
        main_sim_dir: [str]
            main simulation directory
        histogram_range: [tuple, 2]
            histogram range
        histogram_step_size: [float]
            histogram step size
        wc_analysis: [str]
            willard chandler analysis folder
        mu_pickle: [str]
            mu value
        want_avg: [logical]
            True if you want average mu value
        want_mu_array: [logical]
            True if you want the actual mu array
        want_stats: [logical]
            True if you want statistics for mu distributions
        want_grid: [logical]
            True if you want the grid values
    OUTPUTS:
       storage_dict: [dict]
           dictionary with each key and ligand, with corresponding histogram, e.g.
              'C11CF3': {'hist': [array([0.        , 0.        , 0.        , 0.        , 0.        ,
                       0.        , 0.        , 0.        , 0.        , 0.        ,
                       0.        , 0.        , 0.00084282, 0.02359882, 0.1963759 ,
                       0.25200169, 0.23303835, 0.24062368, 0.23556679, 0.35018963,
                       0.29498525, 0.13948588, 0.02612727, 0.00589971, 0.00126422,
                       0.        , 0.        , 0.        , 0.        , 0.        ,
                       0.        , 0.        , 0.        , 0.        , 0.        ,
                       0.        , 0.        , 0.        , 0.        , 0.        ])],
               'x': array([ 0.25,  0.75,  1.25,  1.75,  2.25,  2.75,  3.25,  3.75,  4.25,
                       4.75,  5.25,  5.75,  6.25,  6.75,  7.25,  7.75,  8.25,  8.75,
                       9.25,  9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25,
                      13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75,
                      18.25, 18.75, 19.25, 19.75])}}}
    '''

    ## DEFINING STORAGE DICT
    storage_dict = {}
    
    ## STORING LIGANDS
    ligands_stored = ligands[:]
    
    ## LOOPING THROUGH EACH PATH
    for each_path_item in path_list:
        
        ## SEEING IF PATH DICT KEY
        if 'path_dict_key' in path_list[each_path_item].keys():
            path_dict_key = path_list[each_path_item]['path_dict_key']
        else:
            path_dict_key = each_path_item

        ## DEFINING PREFIX
        prefix_key = path_list[each_path_item]['prefix_key']
        
        ## DEFINING KEY LIST
        if type(path_dict_key) is list:
            path_dict_key_list = path_dict_key
            prefix_key_list = prefix_key
            varying_keys = True
        else:
            path_dict_key_list = [path_dict_key]
            prefix_key_list = [prefix_key]
            varying_keys = False
            
        ## CREATING NEW KEY
        storage_dict[each_path_item] = {}
        
        ## LOOPING THROUGH PATH DICT KEY
        for idx, path_dict_key in enumerate(path_dict_key_list):
            
            ## DEFINING PREFIX KEY
            prefix_key = prefix_key_list[idx]
            
            ## DEFINING PATH TO SIMULATION
            path_to_sim = os.path.join(main_sim_dir,
                                       PATH_DICT[path_dict_key])
            
            prefix_dict = PREFIX_SUFFIX_DICT[prefix_key]
            
            ## SEEING IF LIGANDS ARE WITHIN PREFIX
            if 'ligands' in path_list[each_path_item].keys():
                ligands = path_list[each_path_item]['ligands']
            else:
                ligands = ligands_stored[:]
                
            ## LOOPING THROUGH LIGANDS
            for lig_idx,each_ligand in enumerate(ligands):
                ## FINDING SIM NAME
                sim_name = prefix_dict['prefix'] + each_ligand + prefix_dict['suffix']
                ## PATH TO SIM
                full_path_to_sim = os.path.join(path_to_sim,
                                                sim_name)
                
                ## PATH TO MU
                path_to_mu = os.path.join(full_path_to_sim,
                                          wc_analysis)
                
                ## GETTING MU
                hydration_map = extract_hydration_maps()
                mu_list = hydration_map.load_mu_values(main_sim_list = [path_to_mu],
                                                       pickle_name = mu_pickle)

                ## COMPUTING HISTOGRAM DATA
                histogram_data, xs = compute_histogram_for_mu( mu_list = mu_list,
                                                               histogram_range = histogram_range,
                                                               step_size = histogram_step_size,
                                                               )
                
                ## CREATING IDX
                if each_ligand not in storage_dict[each_path_item]:
                    storage_dict[each_path_item][each_ligand] = {}
                    
                ## STORING
                if varying_keys is True:
                    if path_dict_key not in storage_dict[each_path_item][each_ligand]:
                        storage_dict[each_path_item][each_ligand][path_dict_key] = {}
                    ## DEFINING CURRENT DICTIONARY
                    current_dict = storage_dict[each_path_item][each_ligand][path_dict_key]

                else:
                    current_dict = storage_dict[each_path_item][each_ligand]
                    
                ## STORING
                current_dict['hist'] = histogram_data[0]
                current_dict['x'] = xs

                ## SEEING IF YOU WANT AVERAGE
                if want_avg is True:
                    current_dict['avg'] = np.mean(mu_list)
                if want_mu_array is True:
                    current_dict['mu'] = mu_list[0]
                if want_stats is True:
                    current_dict['stats'] = compute_mu_stats(mu = mu_list[0])
                
                
                ## CHECK IF YOU WANT THE GRID
                if want_grid is True:
                    ## PATH TO GRID
                    path_to_grid = os.path.join(path_to_mu,
                                                relative_path_to_wc)
                    
                    ## LOADING GRID
                    grid = load_datafile(path_to_grid)
                    
                    ## STORING
                    current_dict['grid'] = grid[:]
                
                ## PRINTING EMPTY LINE
                print("")
            
    return storage_dict

### FUNCTION TO PLOT THE DISTRIBUTION
def plot_mu_distribution(storage_dict,
                         path_list,
                         water_mu_value = None,
                         line_dict={'linestyle': '-',
                                   'linewidth': 2},
                         figsize = None,
                         xlim = [4, 20],
                         xticks = np.arange(5, 20, 2),
                         want_legend_all=False,
                         want_combined_plot = False,
                         want_default_lig_colors = True,
                         want_title = True,
                         avg_type='avg',
                         fig = None,
                         axs = None,):
    '''
    This function plots the mu value probability distribution.
    INPUTS:
        storage_dict: [dict]
            dictionary with the hist and x value distributions
        water_mu_value: [list]
            list of water mu value.
        path_list: [dict]
            dictionary contianing path information
        line_dict: [dict]
            dictionary of the line
        want_legend_all: [logical]
            True if you want lgend for all axis
        want_combined_plot: [logical]
            True if you want storage dict to be a combined plot
        want_default_lig_colors: [logical]
            True if you want default lig colors (overwriting combined plots colors)
        want_title: [logical]
            True if you want the title
        avg_type: [str]
            type of average you want to plot
                'avg': average of distribution
                'median': median of distribution
    OUTPUTS:
        
    '''
    ## CREATING FIGURE
    if fig is None or axs is None:
        if want_combined_plot is True:
            fig, axs = plot_tools.create_fig_based_on_cm(fig_size_cm = figsize)
        else:
            ## CREATING FIGURES
            fig, axs = plt.subplots(nrows=len(storage_dict), 
                                    sharex=True,
                                    figsize = figsize)
        
    ## RE-DEFINING AXS
    if len(storage_dict)==1 or want_combined_plot is True:
        axs = [axs]
    
    
    ## LOOPING THROUGH DESIRED TYPE
    for idx, each_type in enumerate(storage_dict):
        
        ## DEFINING AXIS
        if want_combined_plot is False:
            ax = axs[idx]
        else:
            ax = axs[0]
            
        ## DEFINING COLOR
        if 'color' in path_list[each_type]:
            color_list = path_list[each_type]['color']
        
        ## DEFINING LIGANDS
        ligands = list(storage_dict[each_type].keys())
        
        ## LOOPING THROUGH EACH LIGAND
        for each_ligand in ligands:
            
            ## SEEING IF LIGANDS HAS HIST
            if 'hist' not in storage_dict[each_type][each_ligand]:
                ## DEFINING LIST
                key_list = list(storage_dict[each_type][each_ligand].keys())
            else:
                key_list = ["None"]
            
            ## LOOPING THORUGH KEY LIST
            for key_idx, current_key in enumerate(key_list):
                
                ## GETTING HIST AND X
                if current_key == "None":
                    current_dict = storage_dict[each_type][each_ligand]
                else:
                    current_dict = storage_dict[each_type][each_ligand][current_key]
                
                ## DEFINNING X AND Y
                hist = current_dict['hist']
                x = current_dict['x']
                
                ## FINDING AVG VALUE
                if avg_type == 'avg':   
                    avg_value = current_dict['avg']
                elif avg_type == 'median':
                    avg_value = current_dict['stats'][avg_type]
                
                if want_combined_plot is False or want_default_lig_colors is True:
                
                    ## REDEFINING LABEL
                    if each_ligand in RELABEL_DICT:
                        label = RELABEL_DICT[each_ligand]
                    else:
                        label = each_ligand
                    
                    if each_ligand in LIGAND_COLOR_DICT:
                        color = LIGAND_COLOR_DICT[each_ligand]
                    else:
                        print("Warning, ligand %s not defined in LIGAND_COLOR_DICT")
                        print("Printing color as black")
                        color = 'k'
                else:
                    color = path_list[each_type]['color']
                    label = each_type
                
                ## FOR WHEN YOU HAVE MULTIPLE COLORS
                if current_key != "None":
                    color = color_list[key_idx]
                    label = current_key
                
                ## PLOTTING
                ax.plot(x, hist, 
                        color = color, 
                        label = label,
                        **line_dict)
                
                ## PLOTTING AVG ON PLOT
                ax.axvline(x = avg_value, linestyle=':', color = color, linewidth = 1.5)
        
        ## SETTING X TICKS
        ax.tick_params(axis="x",direction="out")
        # ax.set_xticklabels([])
        
        ## ADDING TITLE
        if want_combined_plot is False and want_title is True:
            ax.text(.5,.9,each_type,
                horizontalalignment='center',
                transform=ax.transAxes)
        
        ## SETTING Y LABEL
        ax.set_ylabel("$P(\mu)$")
        
    ## ADDING AXIS LABELS
    axs[-1].set_xlabel("$\mu$")
    
    ## ADDING WATER LINE
    if water_mu_value is not None:
        [axes.axvline(x = water_mu_value[0], linestyle='--', color='k', label='Pure water', linewidth = 1.5) for axes in axs]
    
    ## ADDING LEGEND
    if want_legend_all is False:
        axs[0].legend()
    else:
        [axes.legend(loc="upper right") for axes in axs]
    
    ## LOOPING AND SETTING DIMENSIONS
    if want_combined_plot is False:
        for item_idx, each_path_item in enumerate(path_list):
            ## MODULATING LIMITS
            ax = axs[item_idx]
            
            ## X AXIS
            ax.set_xlim(xlim)
            ax.set_xticks(xticks)
            
            ## SETTING Y AXIS
            ax.set_ylim(path_list[each_path_item]['ylim'])
            ax.set_yticks(path_list[each_path_item]['yticks'])
            
    else:
        axs[0].set_xlim(xlim)
        axs[0].set_xticks(xticks)
            ## ADDING GRID
            # ax.grid(axis='x',linestyle=':',linewidth=1)
        
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## ADJUSTING SPACE OF SUB PLOTS
    if want_combined_plot is False:
        plt.subplots_adjust(wspace=0, hspace=0)
        
    return fig, axs

### FUNCTION TO PLOT BAR
def plot_bar_stats(
        ordered_ligs,
        ordered_stats,
        color_ligs,
        current_stats_key = 'mean',
        stats_label = "Statistics",
        fig_size_cm = FIGURE_SIZE,
        want_water_line = True
                   ):
    '''
    This function generates a bar plot for statistics vs. ligands.
    INPUTS:
        ordered_ligs: [list]
            list of ligand names
        ordered_stats: [list]
            listt of statistics
        color_ligs: [list]
            list of the color per ligand
        stats_label: [str]
            statistics label
        fig_size_cm: [tuple]
            figure size
        want_water_line: [logical]
            True if you want water line for mean statistcs
        
    OUTPUTS:
        fig, ax:
            figure and axis outputting the bar stats
    '''
    ## PLOTTING
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_ylabel(stats_label)
    
    ## PLOTTING 
    bar_plot = ax.bar(ordered_ligs,ordered_stats,color='k')
    
    ## COLORING
    for idx, each_bar in enumerate(bar_plot):
        each_bar.set_color(color_ligs[idx])
    
    ## DRAWING y=0
    ax.axhline(y=0, linestyle='-', linewidth=1 , color='k')
    
    ## ADDING WATER LINE FOR MEAN
    if (current_stats_key == 'mean' or current_stats_key == 'uncentered_moment_1') and want_water_line is True:
        water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                     sim_path = MAIN_HYDRO_DIR)
        ax.axhline(y=water_mu_value, linestyle='--', linewidth=1.5, color = 'k', label="Bulk water")
    
        ## ADDING LEGNED
        ax.legend()
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT PARTICULAR STATS GIVEN STORAGE DICT, ETC.
def plot_bar_specific_stats(storage_dict,
                            specific_type,
                            current_stats_key = 'mean',):
    '''
    This function plots a specific type of bar plot for GNP, planar sams, etc.
    INPUTS:
        storage_dict: [dict]
            dictionary containing all storage information
        specific_type: [str]
            specific type to plot, e.g. 'GNP'
        current_stats_key: [str]
            the stats key to plot
    OUTPUTS:
        fig, ax;
            Figure and axis for the plot
    '''
    
    ## EXTRACTING CURRENT DICT
    current_dict = storage_dict[specific_type]
    
    ## RELABEL
    if current_stats_key in STATS_RELABEL:
        stats_label = STATS_RELABEL[current_stats_key]
    else:
        stats_label = current_stats_key
    
    ## EXTRACTION OF DATA
    current_stats = np.array([current_dict[each_key]['stats'][current_stats_key] for each_key in current_dict])
    current_ligands = np.array(list(current_dict.keys()))
    
    ## ORDERING BY STATS
    ordered_idx = np.argsort(current_stats)
    
    ## ORDERED RESULTS
    ordered_ligs = current_ligands[ordered_idx]
    ordered_stats = current_stats[ordered_idx]
    
    ## GETTING COLORSordered_ligs
    color_ligs = [LIGAND_COLOR_DICT[each_lig] for each_lig in ordered_ligs]
    
    ## RELABEL IF NECESSARY
    ordered_ligs = np.array([RELABEL_DICT[each_lig] for each_lig in ordered_ligs])
    
    ## PLOTTING
    fig, ax = plot_bar_stats(
                            ordered_ligs = ordered_ligs,
                            ordered_stats = ordered_stats,
                            current_stats_key = current_stats_key,
                            color_ligs = color_ligs,
                            stats_label = stats_label,
                            fig_size_cm = FIGURE_SIZE,
                            want_water_line = True
                            )

    return fig, ax

### FUNCTION TO EXTRACT MEDIAN VALUES
def extract_stats_values(storage_dict,
                         stat_key="median",
                         stats_word="stats"):
    '''
    This function extracts statistics from a storage dictionary and 
    generates a datafrmae for it.
    INPUTS:
        storage_dict: [dict]
            storage dictionary containing 'stats' and ligands
        stats_key: [str]
            statistics key that you want, e.g. median
        stats_word: [str]
            stats key word, e.g. stats
    OUTPUTS:
        df: [pd.dataframe]
            dataframe with the statistics
             
    '''

    ## STORING
    median_storage_list = []
    
    ## LOOPING THROUGH DICT
    for each_key in storage_dict:
        ## LOOPING THROUGH LIGANDS
        for each_lig in storage_dict[each_key]:
            current_stats = storage_dict[each_key][each_lig][stats_word]
            
            if stat_key is not None:
                current_stats_key = storage_dict[each_key][each_lig][stats_word][stat_key]
            else:
                current_stats_key = current_stats
                
            ## STORING
            median_storage_dict = {
                    'ligand' : each_lig,
                    'type': each_key,
                    'stat_key': current_stats_key,
                    }
            ## STORING
            median_storage_list.append(median_storage_dict)
    ## CREATING DATAFRAME
    df = pd.DataFrame(median_storage_list)
    return df

#############################################
### FUNCTIONS TO GET GRID FOR PLANAR SAMS ###
#############################################

### FUNCTION TO SPLIT GRID ARRAY
def split_grid_planar_SAMs(grid):
    '''
    The purpose of this function is to split the grid from upper and lower 
    planar SAM.
    
    INPUTS:
        grid: [np.array, shape=(N,3)]
            grid array
    OUTPUTS:
        grid_split: [list]
            list of grid values that have been split
        indices: [list]
            list of indices of the top and bottom
    '''
    ## GETTING AVG Z OF GRID
    z_grid = grid[:, -1]
    avg_z = np.mean(z_grid)
    
    ## FINDING ABOVE AND BELOW
    top_indices = np.where(z_grid > avg_z)[0]
    bot_indices = np.where(z_grid < avg_z)[0]
    
    ## GETTING SPLIT GRID
    indices = [top_indices, bot_indices]
    grid_split = [grid[each_indices] for each_indices in indices ]
    
    return grid_split, indices

### FUNCTION TO COMPUTE MU DICT GIVEN GRID
def compute_mu_median_btn_planar_SAMs(grid, mu):
    '''
    This function splits the planar SAM between top and bottom grid
    based on the z-dimension average of the grid. Then, 
    we compute the mu value of top and bottom. Finally, we take an average 
    and standard deviation of the mu values to get ensemble statistics.
    INPUTS:
        grid: [np.array, shape=(N,3)]
            grid array
        mu: [np.array, shape=N]
            mu values for each grid point
    OUTPUTS:
        mu_dict: [dict]
            dictionary of mu values.
                'value' is the average median
                'error' is the standard deviation of the medians
                'mu_split' is the split mu values
    '''

    
    ## GETTING GRID SPLIT
    grid_split, indices = split_grid_planar_SAMs(grid)
    
    ## COMPUTING MU
    mu_split = [ mu[each_indices] for each_indices in indices]
    
    ## COMPUTING MEDIAN FOR EACH SPLIT AND GET STD
    mu_stats = [compute_mu_stats(each_mu)['median'] for each_mu in mu_split]
    mu_dict = {'value': np.mean(mu_stats),
               'error': np.std(mu_stats),
               'mu_split': mu_split}
    
    return mu_dict

### FUNCTION TO GET AVG MU VALUE FOR PLANAR SAMS
def compute_mu_dict_for_planar_SAMs(planar_dict):
    '''
    This function computes the mu_dict for multiple planar SAMs. Then,
    it will check if the ligand has a specific label. If so, the 
    key will be relabeled.
    INPUTS:
        planar_dict: [dict]
            planar dict from 'load_mu_values_for_multiple_ligands'
    OUTPUTS:
        output_dict: [dict]
            dictionary with average / std of median mu values. e.g.
                {'CH3': {'value': 7.198357907671381, 'error': 0.0028448371066178697},
                 'OH': {'value': 12.238207599873416, 'error': 0.007715520815903432},
                 'NH2': {'value': 10.146941716434412, 'error': 0.027331304637264964},
                 'COOH': {'value': 14.171247499688942, 'error': 0.23081202754715502},
                 'CONH2': {'value': 11.956990409459404, 'error': 0.06622783076035077},
                 'CF3': {'value': 9.12305643807354, 'error': 0.05486905810890619}}
    '''

    ## LOOPING THROUGH AND CREATING A PLANAR DICTIONARY OF MU VALUES
    output_dict = {}
    
    ## LOOPING THROUGH EACH KEY
    for each_ligand in planar_dict:
        ## DEFINING CURRENT DICT
        current_dict = planar_dict[each_ligand]
        
        ## GETTING MU AND GRID 
        mu = current_dict['mu']
        grid = current_dict['grid']
        
        ## COMPUTING MU DICT
        mu_dict = compute_mu_median_btn_planar_SAMs(grid = grid, 
                                                    mu = mu)
        
        ## SEEING IF THE LIGAND HAS A LABEL
        if each_ligand in RELABEL_DICT:
            label = RELABEL_DICT[each_ligand]
        else:
            label = each_ligand
            
        ## STORING
        output_dict[label] = mu_dict.copy()
    return output_dict

### FUNCTION TO PLOT INDUS VERSUS UNBIASED
def plot_indus_vs_unbiased_mu(indus_dict,
                              y_dict,
                              x_value_key = 'value',
                              x_error_key = 'error',
                              y_value_key = 'value',
                              y_error_key = 'error',
                              fig = None,
                              ax = None
                              ):
    '''
    The purpose of this function is to plot the indus and unbiased 
    mu distributions.
    INPUTS:
        indus_dict: [dict]
            indus dictionary with 'value' and 'error' for each key
        y_dict: [dict]
            unbiased mu dict, with same keys as indus_dict
        x_value_key , y_value_key: [str]
            value key in x, y dictionaries
        x_error_key , y_error_key: [str]
            error key in x, y dictionaries
        fig, ax:
            figure and axis to add onto
    OUTPUTS:
        fig, ax:
            figure and axis for plot
    '''
    ## DEFINING LABELS
    labels = [each_key for each_key in indus_dict]
    
    x = [indus_dict[each_key][x_value_key] for each_key in indus_dict]
    y = [y_dict[each_key][y_value_key] for each_key in indus_dict]
    
    ## GETTING ERROR
    xerr = [indus_dict[each_key][x_error_key] for each_key in indus_dict]
    yerr = [y_dict[each_key][y_error_key] for each_key in indus_dict]
    
    ## PLOTTING
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
        ## ADDING AXIS
        ax.set_xlabel("INDUS $\mu$ (kT)")
        ax.set_ylabel("Unbiased MD $\mu$ (kT)")
    
    ## ERROR BAR
    (_, caps, _) = ax.errorbar(x = x,
                y = y,
                yerr = yerr,
                xerr = xerr,
                fmt='.',
                markersize=12,
                color='k',
                linestyle = None,
                capsize=2,
                elinewidth = 2,
                )
    
    ## LOOPING THROUGH CAPS
    for cap in caps:
        cap.set_color('black')
        cap.set_markeredgewidth(2)
    
    ## FITTING TO LINE
    fit = np.polyfit(x, y, 1)
    
    ## FINDING EQUATION
    equation = 'y={:.2f}x+{:.2f}'.format(fit[0], fit[1])
    
    ## ADDING BEST FIT LINE
    ax.plot(np.unique(x), 
            np.poly1d(fit)(np.unique(x)),
            color='k',
            linestyle = '--',
            linewidth = 2,
            label = equation)
    ## ANNOTATING TEXT    
    for i, txt in enumerate(labels):
        ax.annotate(labels[i], xy = (x[i], y[i]),
                    xytext=(0, 20), textcoords='offset pixels',
                    horizontalalignment='center', verticalalignment='top'
                    )
    
    ## GETTING PEARSONR
    pear_r = pearsonr(x = x, y = y)[0]
    
    ## ADDING TEXT TO PLOT
    ax.text(0.98, 0.05,"Pearson's r = %.2f"%(pear_r),
        horizontalalignment='right',
        verticalalignment='center',
        transform = ax.transAxes)
        
    
    ## ADD LEGEND
    ax.legend(loc = 'upper left')    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO LOAD CLUSTERING
def load_clustering_multiple_ligs(path_list,
                                  relative_hdbscan_clustering,
                                  main_sim_dir = MAIN_HYDRO_DIR):
    '''
    This function loads the output of the hdbscan clustering algorithm.
    INPUTS:
        path_list: [dict]
            dictionary that has the names of each type and the ligand names.
        relative_hdbscan_clustering: [str]
            relative path to hdbscan clustering from path of simulation
        main_sim_dir: [str]
            path to simulation
    OUTPUTS:
        storage_dict: [dict]
            dictionary storing outputs of the clutering algorithm
    '''
    
    ## DEFINING STORAGE DICT
    storage_dict = {}
    ## LOOPING THROUGH EACH PATH
    for each_path_item in path_list:
        
        ## SEEING IF PATH DICT KEY
        if 'path_dict_key' in path_list[each_path_item].keys():
            path_dict_key = path_list[each_path_item]['path_dict_key']
        else:
            path_dict_key = each_path_item

        ## DEFINING PREFIX
        prefix_key = path_list[each_path_item]['prefix_key']
        
        ## DEFINING KEY LIST
        if type(path_dict_key) is list:
            path_dict_key_list = path_dict_key
            prefix_key_list = prefix_key
        else:
            path_dict_key_list = [path_dict_key]
            prefix_key_list = [prefix_key]
            
        ## CREATING NEW KEY
        storage_dict[each_path_item] = {}
        
        ## LOOPING THROUGH PATH DICT KEY
        for idx, path_dict_key in enumerate(path_dict_key_list):
            
            ## DEFINING PREFIX KEY
            prefix_key = prefix_key_list[idx]
            
            ## DEFINING PATH TO SIMULATION
            path_to_sim = os.path.join(main_sim_dir,
                                       PATH_DICT[path_dict_key])
            
            prefix_dict = PREFIX_SUFFIX_DICT[prefix_key]
            
            ## FINDING ALL LIGANDS
            ligands = path_list[each_path_item]['ligands']
                
            ## LOOPING THROUGH LIGANDS
            for lig_idx,each_ligand in enumerate(ligands):
                ## FINDING SIM NAME
                sim_name = prefix_dict['prefix'] + each_ligand + prefix_dict['suffix']
                ## PATH TO SIM
                full_path_to_sim = os.path.join(path_to_sim,
                                                sim_name)
                
                ## PATH TO PICKLE
                path_pickle = os.path.join(full_path_to_sim,
                                           relative_hdbscan_clustering)
                
                ## LOADING PICKLE
                hdbscan_cluster, labels, clustering, num_clusters, idx_above, idx_below = load_pickle_results(file_path = path_pickle,
                                                                                                              verbose = True)[0]
                
                ## STORING
                storage_dict[each_path_item][each_ligand] = {
                        'hdbscan_cluster': hdbscan_cluster,
                        'labels': labels,
                        'clustering': clustering,
                        'num_clusters': num_clusters,
                        'idx_above': idx_above,
                        'idx_below': idx_below,
                        }
    return storage_dict.copy()

### FUNCTION TO GENERATE DF WITH AVERAGE AND DF
def merge_data_frame_get_avg_std(df_list):
    '''
    This function simply gets the merged dataframe and averages / stds it
    INPUTS:
        df_list: [list]
            list of dataframes, shape = 2
    OUTPUTS:
        merged_df: [df]
            dataframe with 'avg' and 'std' for specific stats key
    '''

    ## MERGING DATAFRAMES
    merged_df = df_list[0].merge(df_list[1], left_on = 'ligand', right_on='ligand')
    
    ## GETTING STATS ARRAY
    stats_array = (merged_df['stat_key_x'].to_numpy(), merged_df['stat_key_y'].to_numpy())
    
    ## ADDING MEAN AND STD
    merged_df['avg'] = np.mean( stats_array, axis = 0 )
    merged_df['std'] = np.std( stats_array, axis = 0 )
    return merged_df


# ORDERED FROM SAT, UNSAT, AND BRANCHED
bunched_dict = {
        'CH3': ["dodecanethiol", "dodecen-1-thiol", 'C11branch6CH3'],
        'CF3': ["C11CF3", "C11double67CF3", 'C11branch6CF3'],
        'NH2': ["C11NH2", "C11double67NH2", 'C11branch6NH2'],
        'CONH2': ["C11CONH2", "C11double67CONH2", 'C11branch6CONH2'],
        'OH': ["C11OH", "C11double67OH", 'C11branch6OH'],
        'COOH': ["C11COOH", "C11double67COOH", 'C11branch6COOH'],
        }

### FUNCTION TO PLOT BAR FOR DIFFERENT STATS
def plot_bar_versus_ligs(storage_dict,
                         merged_df,
                         fig_size_cm,
                         width=0.25,
                         bunched_dict = bunched_dict,
                         colors = ['black', 'red', 'blue'],
                         error_kw=dict(lw=1, capsize=2, capthick=1),
                         ylabel="Median $\mu$",
                         water_mu_value = None):
    '''
    The purpose of this function is to plot the statsitic versus ligand.
    INPUTS:
        storage_dict: [dict]
            dictionary with unique types
        merged_df: [df]
            dataframe with merged information for 'avg' and 'std'
        fig_size_cm: [tuple]
            figure size as a tuple between width and height
        width: [float]
            width of the bar
        bunched_dict: [dict]
            dictionary for bunching data together
        colors: [list]
            list of colors corresponding the bundict_dict
        error_kw: [dict]
            dictionary for error bars for bar
        ylabel: [str]
            y label
        water_mu_value: [tuple]
            mu value for bulk water
        
    OUTPUTS:
        fig, ax:
            figure and axis for plot
    '''

    ## CREATING PLOT
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    ## GETTING TOTAL
    n = len(storage_dict)
    x = np.arange(len(bunched_dict))
    ## GETTING ARRAY
    n_array = np.arange(-n + 2, n - 1)
    
    ## VARIABLES
    for idx, each_key in enumerate(storage_dict):

        ## GETTING LIGAND NAMES
        lig_names = [ bunched_dict[each_key][idx] for each_key in bunched_dict]
        
        ## FINDING MEDIAN VALUES
        current_df = merged_df.loc[merged_df.ligand.isin(lig_names)]
        ## REORDERING BASED ON LIG NAMES
        current_df = current_df.set_index('ligand').reindex(lig_names)
        
        ## DEFINING MEDIAN
        median_values = current_df['avg'].to_numpy()
        err_values = current_df['std'].to_numpy()
        
        ## PLOTTING BAR        
        ax.bar(x + width * n_array[idx] ,
               height = median_values, 
               yerr = err_values,
               width=width, 
               color=colors[idx], 
               align='center',
               label=  each_key,
               error_kw = error_kw,
               edgecolor = 'k',
               linewidth = 1,
               )
        
    ## SETTING X LABELS
    ax.set_xticks(x)
    ax.set_xticklabels( list(bunched_dict.keys()) )
    
    ## DRAWING WATER LINE
    if water_mu_value is not None:
        ax.axhline(y = water_mu_value[0], linestyle='--', color='k', label='Pure water', linewidth = 1.5)
    ## ADDING LEGEND
    ax.legend(loc='upper left')
    
    ## ADDING LABELS
    ax.set_ylabel(ylabel)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax
    
### FUNCTION TO EXTRACT TRAJ INFORMATION
def extract_traj_information(traj_data,
                             output_dict = {},
                             is_planar = False):
    '''
    The purpose of this function is to extract trajectory information 
    for nanoparticle systems. 
    INPUTS:
        traj_data: [obj]
            trajectory data object
        is_planar: [logical]
            True if you have a planar SAM
    OUTPUTS:
        output_dict: [dict]
            dictionary of information
    '''
    ## DEFINING TRAJECTORY
    traj = traj_data.traj
    
    ## GETTING LIGAND NAME
    lig_res_names, lig_full_name = get_ligand_names_within_traj(traj, return_lig_names = True)
    
    ## OUTPUT LIG NAMES
    output_dict['lig_name'] = lig_full_name[0][0]
    
    ## STORING BOX LENGTHS
    for idx, each_value in enumerate(traj.unitcell_lengths[0]):
        output_dict['box_%d'%(idx)] = each_value
        
    if is_planar is True:
        itp_file = None
        separated_ligands = True
    else:
        itp_file = 'match'
        separated_ligands = False
        
    ## RUNNING STRUCTURE
    structure = nanoparticle_structure(traj_data = traj_data,
                                       ligand_names = lig_res_names,
                                       itp_file = itp_file,
                                       structure_types = None,
                                       separated_ligands = separated_ligands)
    
    ## TOTAL LIGANDS
    output_dict['num_ligs'] = structure.total_ligands
#        if is_planar is False:
#            output_dict['num_gold_atoms'] = len(structure.gold_atom_index)
#        else:
    output_dict['num_gold_atoms'] = len([atom.index for atom in traj.topology.atoms if atom.name == GOLD_ATOM_NAME])
    
    ## TOTAL WATER
    output_dict['num_water'] = traj_data.residues['HOH']

    return output_dict, structure
    
### FUNCTION TO TRY TO ADD TO DICT
def try_to_add(output_dict, label_dict, input_key, output_key):
    '''
    This function trys to add data value
    '''
    if input_key in label_dict:
        output_dict[output_key] = label_dict[input_key]
    return output_dict

### FUNCTION TO PLOT MU VERSUS TIME
def plot_subplots_mu_vs_time(sheet_dict,
                             figsize_inches,
                             x_data_key = 'equil time',
                             x_axis_label = "Equil. time (ps)",
                             y_axis_label = "$\mu_A$",
                            errorbar_format={
                                    'linestyle' : "-",
                                    "fmt": ".",
                                    "capthick": 1.5,
                                    "markersize": 8,
                                    "capsize": 3,
                                    "elinewidth": 1.5,
                                    },
                            want_separate = True,
                             ):
    '''
    This function plots mu vs time, used for supplmentary information
    on equilibration and production times
    INPUTS:
        sheet_dict: [dict]
            dictionary containin all the information
        figsize_inches: [tuple]
            figure inches
        x_data_key: [str]
            x data key
        x_axis_label: [str]
            x axis label
        y_axis_label: [str]
            y axis label
        errorbar_format: [dict]
            dictinoary containing error bar format
        want_separate: [logical]
            True if you want separate
    OUTPUTS:
        fig, axs: [obj]
            figure and axis object
    '''
    ## CREATING FIGURES
    fig, axs = plt.subplots(nrows=len(sheet_dict), 
                            ncols = 1,
                            sharex=True,
                            figsize = figsize_inches)
    
    ## LOOPING
    for idx, key in enumerate(sheet_dict):
        ## DEFINING AXIS
        ax = axs[idx]
        
        ## DEFINING LABEL
        current_dict = sheet_dict[key]
        
        ## DEFINING VARIABLE
        each_sheet = current_dict['tab']
        
        ## DEFINING COLOR
        color = current_dict['color']
        
        ## LOADING THE DATA
        data = pd.read_excel(PATH_TO_INDUS_SI,
                             sheet_name = each_sheet)
        
        ## AVGING TOP AND BOTTOM
        top_n_bottom = data[['top','bottom']].to_numpy()
        avg_data = np.mean(top_n_bottom,axis=1)
        std_data = np.std(top_n_bottom,axis=1)
        
        ## DEFINING X AND Y
        x = data[x_data_key]
        if want_separate is False:
            y = avg_data
            y_err = std_data
        
            ## PLOTTING
            ax.errorbar(x = x,
                        y = y,
                        color = color,
                        **errorbar_format
                        )
        
            ## ADDING ERROR BAR AS SHADED
            ax.fill_between(x, y-y_err, y+y_err, color = color, alpha = 0.5)
        else:
            
            ## DEFINING TYPE
            top_bot_styles={
                    'top': {
                        'linestyle': '-',   
                        'fmt': '.',
                        'fillstyle': 'full',
                        'label': 'top',
                            },
                    'bottom': {
                        'linestyle': '--',
                        'fmt': '.',
                        'fillstyle': 'none',
                        'label': 'bottom',
                            }
                    
                    }
            ## LOOPING
            for idx, each_data in enumerate(top_n_bottom.T):
                if idx == 0:
                    current_type = 'top'
                elif idx == 1:
                    current_type = 'bottom'
                
                y = each_data
                
                ## EDITING DICT
                current_top_down_dict = top_bot_styles[current_type]
                for each_key in current_top_down_dict:
                    errorbar_format[each_key] = current_top_down_dict[each_key]
                ## PLOTTING
                ax.errorbar(x = x,
                            y = y,
                            color = color,
                            **errorbar_format
                            )
            ## ADDING LEGEND
            ax.legend()
        
        ## ADDING TITLE
        ax.text(.5,.9,key,
            horizontalalignment='center',
            transform=ax.transAxes)
        
        ## ADDING LABELS
        if idx == len(sheet_dict) - 1:
            ax.set_xlabel(x_axis_label)
            
        ## EDITING LIMITS
        if 'yticks' in current_dict:
            ax.set_yticks(current_dict['yticks'])

        if 'ylim' in current_dict:
            ax.set_ylim(current_dict['ylim'])
            
        if 'xticks' in current_dict:
            ax.set_xticks(current_dict['xticks'])

        if 'xlim' in current_dict:
            ax.set_xlim(current_dict['xlim'])
            
        ## SETTING Y LABEL
        ax.set_ylabel(y_axis_label)
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
    ## REMOVING WHITE SPACE
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, axs

### FUNCTION TO LOAD DATA FOR NEIGHBORS ARRAY GIVEN A DICTIONARY
def load_data_for_neighbors_array(input_dict,
                                  neighbors_pickle = r"2000-50000.pickle",
                                  main_sim_dir=MAIN_HYDRO_DIR):
    '''
    The purpose of this function is to load the neighbors array  array.
    INPUTS:
        input_dict: [dict]
            dictionary containing the following
                'prefix_key': prefix key for prefix/suffix
                'path_dict_key': key for paths
                'ligands': ligands that you are interested in as a list
        neighbors_pickle: [str]
            neighbors pickle
        main_sim_dir: [str]
            path to simulation directory
    OUTPUTS:
        storage_data_dict: [dict]
            dictionary contianing neighbors array for each ligand
    '''

    ## DEFINING STORAGE
    storage_data_dict = {}
    ## LOOPING
    for ligand in input_dict['ligands']:
        print("Working on ligand: %s"%(ligand) )
        ## DEFINING DATA
        prefix_key = input_dict['prefix_key']
        path_dict_key = input_dict['path_dict_key']
    
        ## DEFINING PREFIX
        prefix_dict = PREFIX_SUFFIX_DICT[prefix_key]
        ## FINDING SIM NAME
        sim_name = prefix_dict['prefix'] + ligand + prefix_dict['suffix']
    
        ## DEFINING PATH TO SIMULATION
        path_to_sim = os.path.join(main_sim_dir,
                                   PATH_DICT[path_dict_key])
        ## PATH TO SIM
        full_path_to_sim = os.path.join(path_to_sim,
                                        sim_name)
    
        ## DEFINING RELATIVE PATH    
        relative_path = os.path.join(DEFAULT_WC_ANALYSIS, "compute_neighbors" )
        

    
        ## PATH TO SIMULATION
        path_to_neighbors = os.path.join(full_path_to_sim,
                                   relative_path)
        
        ## LOADING PICKLE
        hydration_map = extract_hydration_maps()
        neighbor_array = hydration_map.load_neighbor_values(main_sim_list = [path_to_neighbors],
                                                           pickle_name = neighbors_pickle)[0]
        
        ## STORING
        storage_data_dict[ligand] = neighbor_array[:]
        
    return storage_data_dict

## FUNCTION TO GET MU OBJECTS
def compute_mu_objects(neighbor_array_dict):
    '''
    The purpose of this function is to loop through  a dictionary containing
    all neighbor arrays, then get mu values
    INPUTS:
        neighbor_array_dict: [dict]
            dictionary containing neighbor arrays
    OUTPUTS:
        mu_dist_obj_dict: [dict]
            dictionary containing muobject
    '''
    mu_dist_obj_dict = {}
    for each_key in neighbor_array_dict:
        print("Computing mu for %s"%(each_key))
        ## DEFINING NEIGHBORS ARRAY
        neighbor_array = neighbor_array_dict[each_key]

        ## COMPUTING UNNORMALIZED NUMBER DIST
        unnorm_p_N = compute_num_dist(num_neighbors_array = neighbor_array, # num_neighbors_array,
                                      max_neighbors = MAX_N)
        
        ## COMPUTING MU DISTRIBUTION
        mu_dist_obj = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
        
        ## STORING
        mu_dist_obj_dict[each_key] = mu_dist_obj
    return mu_dist_obj_dict

### FUNCTION TO COMPUTE CONVERGENCE
def compute_mu_convergence_using_neighbors_dict(neighbor_array_dict,
                                                frame_rate = 6000,
                                                ):
    '''
    This function computes the convergence of mu for varying frame rates and 
    percent errors.
    INPUTS:
        neighbor_array_dict: [dict]
            neighbors array dictionary
        frame_rate: [int]
            frame rate
        percent_error: [float]
            error to check
        convergence_type: [str]
            type of converegence:
                "value" means the value +- the percent error
    OUTPUTS:
        storage_mu_convergence_dict: [dict]
            dictionary containing the following items:
                mu_debug: [obj]
                    debugging mu class
                mu_storage_reverse: [list]
                    list of mu values stored
                frame_list: [list]
                    list of frames that were varied
                
    '''
    ## DEFINING STORAGE DICT
    storage_mu_convergence_dict = {}
    
    ## LOOPING THROUGH EACH NEIGHBOR ARRAY DICT
    for each_key in neighbor_array_dict:
        ## PRINTING EACH KEY
        print("Working on current key: %s"%(each_key))
        
        ## DEFINING NEIGHBORS ARRAY
        neighbor_array = neighbor_array_dict[each_key]
        
        ## GENERATING DEBUGGING MU CLASS
        mu_debug = debug_mu_convergence()
        
        ## COMPUTING CONVERGENCE TIME -- DEBUGGED, CHECKED CONVERGENCE
        mu_storage_reverse, frame_list = mu_debug.compute_mu_convergence_time(num_neighbors_array = neighbor_array,
                                                                              frame_rate = frame_rate,
                                                                              want_reverse = False,
                                                                              method_type = "new") 
        ## STORING EACH KEY
        storage_mu_convergence_dict[each_key] = {
                'mu_debug': mu_debug,
                'mu_storage_reverse': mu_storage_reverse,
                'frame_list': frame_list,
                }
                        
    return storage_mu_convergence_dict

### FUNCTION TO PLOT HISTOGRAM
def plot_histogram_sampling_time(sampling_time_x_converged,
                                 fig = None,
                                 ax = None,
                                 plot_kwargs = {'color': 'k'},
                                 frame_rate_ps = 6000,
                                 total_time_ps = 48000,
                                 convert_style = "ps2ns"
                                 ):
    '''
    This function plots the histogram of sampling itme.
    INPUTS:
        sampling_time_x_converged: [list]
            list of sampling time for convergence
    '''

    ## CREATING PLOT
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## GETTING TIME IN NS
    if convert_style == "ps2ns":
        sampling_time_x_ns = np.array(sampling_time_x_converged) / 1000.0
        frame_rate_ns = frame_rate_ps/1000.0
        total_time_ns = total_time_ps/1000.0
    elif type(convert_style) == float:
        sampling_time_x_ns = np.array(sampling_time_x_converged) * convert_style
        frame_rate_ns = frame_rate_ps * convert_style
        total_time_ns = total_time_ps * convert_style
        
    
    ## PLOTTING HISTOGRAMS FOR SAMPLING TIME CONVERGED
    bins = np.arange(0, total_time_ns + frame_rate_ns, frame_rate_ns)
    
    ax.hist(sampling_time_x_ns, bins = bins, **plot_kwargs)
    ax.set_xticks(bins )
    ax.set_xlabel("Minimum sampling time (ns)")
    ax.set_ylabel("Frequency")
    
    fig.tight_layout()

    return fig, ax

## FUNCTION TO PLOT MINIMUM SAMPLING TIME
def plot_minimum_sampling_time_normalized(storage_mu_convergence_dict,
                                          color_dict,
                                          percent_error = 1,
                                          convergence_type = "value"):
    '''
    This function plots the minimum sampling time required for convergence
    INPUTS:
        storage_mu_convergence_dict: [dict]
            dictionary with mu convergence dictionary
        
    OUTPUTS:
        
    '''

    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = len(storage_mu_convergence_dict), ncols = 1, sharex = True, 
                             figsize = plot_tools.cm2inch( *fig_size_cm ))
    
    ## PLOTTING FOR CONVERGENCE
    for idx, each_lig in enumerate(storage_mu_convergence_dict):
        ax = axs[idx]
        ## DEFINING RESULTS
        current_output_dict = storage_mu_convergence_dict[each_lig]
        mu_debug = current_output_dict['mu_debug']
        mu_storage_reverse = current_output_dict['mu_storage_reverse']
        frame_list = current_output_dict['frame_list']
        
        ## GETTING COLOR
        color = color_dict[each_lig]
    
        ## GETTING CONVERGENCE INFORMATION
        theoretical_bounds, index_converged, sampling_time_x_converged, x \
                        = mu_debug.main_compute_sampling_time_from_reverse(mu_storage_reverse = mu_storage_reverse,
                                                                           frame_list = frame_list,
                                                                           percent_error = percent_error,
                                                                           convergence_type = convergence_type,
                                                                           )
                        
        ## PLOTTING
        fig, ax = plot_histogram_sampling_time(sampling_time_x_converged,
                                     fig = fig,
                                     ax = ax,
                                     plot_kwargs = {'color': color,
                                                    'density': True,
                                                    'linewidth': 1,
                                                    'edgecolor': 'k',
                                                    'align': 'left'}
                                     )
        ## setting y label
        ax.set_ylabel("PDF")
        
        
        ## CHECKING LIMITS
        if 'limits' in storage_mu_convergence_dict[each_lig]:
            limits_dict = storage_mu_convergence_dict[each_lig]['limits']
            ## SETTING LIMITS
            ax.set_ylim(limits_dict['ylim'])
            ax.set_yticks(limits_dict['yticks'])
        
        ## ADDING TITLE
        ax.text(.5,.9,each_lig,
            horizontalalignment='center',
            transform=ax.transAxes)
        
    ## REMOVING WHITE SPACE
    plt.subplots_adjust(wspace=0, hspace=0)
                    
    return fig, axs

### FUNCTION TO PLOT THE STATISTICS
def plot_bar_stats_key_from_df(df,
                               water_mu_value = None,
                               fig = None,
                               ax = None,
                               fig_size_cm = FIGURE_SIZE,
                               width = 0.25,
                               colors = ['red', 'black', 'blue']):
    '''
    This function plots stats key from dataframe.
    INPUTS:
        df: [pd.datafrmae]
            dataframe from pandas on the stats key, e.g.
                       ligand                      type   stat_key
            6   dodecanethiol  Planar_SAMs_unrestrained   7.073721
            0   dodecanethiol               Planar_SAMs   7.198299
        width: [float]
            width of the bar plot
        water_mu_value: [list]
            mu value as a list. The first index will be plotted
        colors: [list]
            list of the colors
    OUTPUTS:
        
    '''
    ## CREATING FIGURE OR AXIS
    if fig is None or ax is None:
        ## CREATING PLOT
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)

    ## RE-ARRANGING LIGANDS BASED ON MU VALUES
    df = df.sort_values(by = 'stat_key')
    
    ## DEFINING LIGANDS
    ligands = pd.unique(df.ligand)    

    ## DEFINING 
    x = np.arange(len(ligands))
    
    ## STORING MEAN AND STD
    statistics_storage = {}
    
    ## GENERATING     
    for idx, each_type in enumerate(pd.unique(df.type)):
        df_type = df.loc[df.type==each_type]
        median_values = [ float(df_type[df_type.ligand == each_lig]['stat_key'] ) for each_lig in ligands]
        
        statistics_storage[each_type] = median_values
        
        ## PLOTTING
        ax.bar(x + width * (idx-.5) ,
               median_values, 
               width=width, 
               color=colors[idx], 
               align='center',
               edgecolor = 'k',
               linewidth = 1,
               label=  each_type)
    
    ## DRAWING WATER LINE
    if water_mu_value is not None:
        ax.axhline(y = water_mu_value[0], linestyle='--', color='k', label='Pure water', linewidth = 1.5)
    ## ADDING LEGEND
    ax.legend(loc='upper left')
    
    ## ADDING LABELS
    ax.set_ylabel("Median $\mu$")
    
    ## RELABELED
    relabeled_ligs = [ RELABEL_DICT[each_lig] for each_lig in ligands ]
    
    ## SETTING X LABELS
    ax.set_xticks(x)
    ax.set_xticklabels( relabeled_ligs )
    
    ## TIGHT LAYOUT
    fig.tight_layout()

    return fig, ax



#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    #%% FIGURE 2B
    # Distribution of one representative example of CH3
    
    ## DEFINING LIGANDS
    ligands = ["dodecanethiol",
               "C11OH"
               ]
    
    ## DEFINING LOCATION
    path_list = {
                    'Planar_SAMs': {
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ligands,
                                        },
                }
                    
    ## DEFINING GRID DETAILS
    wc_analysis=DEFAULT_WC_ANALYSIS
    mu_pickle = MU_PICKLE

    ## LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = ligands,
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = True,
                                                       want_grid = False,
                                                       )
    
    ## FINDING WATR MU VALUE
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
        
    
    ## PLOTTING THE HISTOGRAMS
    # figsize=(9,7) # in cm # <-- publication size
    figsize=(5,4) # in cm
    figsize_inches=plot_tools.cm2inch(*figsize)

    
    ## DEFINING X LIM
    xlim = [6, 17]
    # [5, 17]
    xticks = np.arange(6, xlim[1] + 2 , 2)

    ## PLOTTING
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 1.5},
                                        figsize = figsize,
                                        water_mu_value = water_mu_value,
                                        xlim = xlim,
                                        xticks = xticks,
                                        avg_type='median',
                                        want_title = False,
                                        want_combined_plot = True)
    
    
    ## SETTING Y TICKS
    axs[0].set_yticks(np.arange(0, 2.5, 0.5 ))
    axs[0].set_ylim([0,2])
    
    ## SETTING HEIGHT AND WEIGHT
    fig.set_figheight(figsize_inches[1])
    fig.set_figwidth(figsize_inches[0])
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    

    ## DEFINING FIGURE PREFIX
    fig_prefix = "2B_"
    ## DEFINING FIGURE NAME
    figure_name = fig_prefix + "dodecanethiol"
    
    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', 
                 save_fig=True,)
    
    #%% FIGURE 2C - CORRELATION PLOT
    
    ###########################################################
    ### CORRELATION BETWEEN INDUS AND DENSITY FLUCTUATIONS ####
    ###########################################################
    
    ## DEFINING INDICTS
    indus_dict = INDUS_RESULTS_DICT
    
    ## READING SPREADSHEET
    contact_angles_df = pd.read_excel(PATH_TO_CONTACT_ANGLES)
    

    ## CONVERTING TO DICT
    contact_angles_dict = {}
    for idx,row in contact_angles_df.iterrows():
        ## GETTING EACH R GROUP
        r_group = row['R group']
        value = row['cos_theta_value']
        error = row['cos_theta_error']
        
        ## STORING
        contact_angles_dict[r_group] = {
                'value': value,
                'error': error,
                }    
    
    ### LOADING PLANAR SAMS
    ## DEFINING LIGANDS
    ligands = ["dodecanethiol",
               "C11OH",
               "C11NH2",
               "C11COOH",
               "C11CONH2",
                "C11CF3",                
                ]
    
    ## DEFINING LOCATION
    path_list = {
                'Planar_SAMs': {
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                        },
                 }
    ## LOADING STORAGE DICT
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = ligands,
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = True,
                                                       want_grid = True
                                                       )
    
    ## COMPUTING MU DICT
    unbiased_mu_dict = compute_mu_dict_for_planar_SAMs(planar_dict = storage_dict['Planar_SAMs'])
    
    #%%
    
    ## CLOSING FIGURES
    plt.close('all')
    
    ## DEFINING LIST
    dict_to_plot = {
            'contact angles': {
                    'dict': contact_angles_dict,
                    'ylabel': 'cos $\Theta$',
                    'ylim': [-0.75, 1.25],
                    'yticks': np.arange(-.5, 1.25, 0.5)
                    },
            'unbiased mu': {
                    'dict': unbiased_mu_dict,
                    'ylabel': 'Median $\mu_L$',
                    'ylim': [6, 17],
                    'xlabel': '$\mu_A$',
                    'xlim': [25, 125],
                    'xticks': np.arange(25, 150, 25)
                    },
            }
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 11)
    
    ## CREATING SUBPLOTS
    figs, axs = plt.subplots(nrows = len(dict_to_plot), ncols = 1, sharex = True,
                             figsize = plot_tools.cm2inch( *fig_size_cm ) )
    
    ## LOOPING
    for idx_dict,each_key in enumerate(dict_to_plot):
    
        ## DEFINING CURRNET DICT
        current_dict = dict_to_plot[each_key]
        
        ## GENERATING FIGURE
        fig, ax = plot_indus_vs_unbiased_mu(indus_dict = indus_dict,
                                            y_dict = current_dict['dict'],
                                            fig = figs,
                                            ax = axs[idx_dict]
                                            )
        
        ## ADDING LABELS
        if 'ylabel' in current_dict:
            ax.set_ylabel(current_dict['ylabel'])
    
        ## SETTING YLIM
        if 'ylim' in current_dict:
            ax.set_ylim(current_dict['ylim'])
        if 'yticks' in current_dict:
            ax.set_yticks(current_dict['yticks'])
            
        ## XLABEL
        if 'xlabel' in current_dict:
            ax.set_xlabel(current_dict['xlabel'])
        if 'xlim' in current_dict:
            ax.set_xlim(current_dict['xlim'])
        if 'xticks' in current_dict:
            ax.set_xticks(current_dict['xticks'])
            
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## REMOVING WHITE SPACE
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ## DEFINING FIGURE NAME
    figure_name = "2B_relation_btn_indus_and_unbiased_contacts"
    
    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', 
                 save_fig=True,)
    
    
    #%%


    ## GENERATING FIGURE
    fig, ax = plot_indus_vs_unbiased_mu(indus_dict = indus_dict,
                                        y_dict = unbiased_mu_dict
                                        )
    
    ## SETTING AXIS
    ax.set_ylim([6, 16])
    ax.set_xlim([25, 125])
    
    ax.set_xticks(np.arange(25, 150, 25))
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 6)
    # (7,6)
    
    ## UPDATING FIGURE SIZE
    fig = plot_tools.update_fig_size(fig,
                                     fig_size_cm = fig_size_cm)

    ## DEFINING FIGURE NAME
    figure_name = "2C_relation_btn_indus_and_unbiased"
    
    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', 
                 save_fig=True,)
    
    
    
    #%% FIG 3C MU DISTRIBUTION

    ## DEFINING LIGANDS
    ligands = ["dodecanethiol",
               "C11CF3", 
               "C11NH2", # Turn on for publication image
               "C11CONH2",
               "C11OH",
               "C11COOH",               
                ]
    
    ## DEFINING GRID DETAILS
    wc_analysis=DEFAULT_WC_ANALYSIS
    mu_pickle = MU_PICKLE

    
    ## DEFINING LOCATION
    path_list = {
                'GNP': {
                    'prefix_key': 'GNP',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.1],
                    'ligands': ligands,
                        },
                'Planar_SAMs': {
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ligands,
                                        },

#                 'Planar_SAMs_unrestrained': { # Planar_SAMs
#                    'prefix_key': 'Planar_no_restraint', # 'Planar',
#                    'yticks': np.arange(0, 2.5, 0.5),
#                    'ylim': [-0.1, 2.0],
#                        },
                 }

    # #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = ligands,
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = True,
                                                       )
    
    ## PRINTING
    print("GNP - doecanethiol mu value: %.2f"%(storage_dict['GNP']['dodecanethiol']['stats']['median']))
    print("Planar - doecanethiol mu value: %.2f"%(storage_dict['Planar_SAMs']['dodecanethiol']['stats']['median']))
    
    #%% LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
    #%%
    
    # #%% PLOTTING DISTRIBUTION
    
    ## PLOTTING THE HISTOGRAMS
    # figsize=(9,7) # in cm # <-- publication size
    figsize=(8.5, 7.5)
    # (8.5, 8.5)
    # (11,11) # in cm
    figsize=plot_tools.cm2inch(*figsize)

    
    ## DEFINING X LIM
    xlim = [5, 17]
    xticks = np.arange(5, 18, 2)

    ## PLOTTING
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 1.5},
                                        figsize = figsize,
                                        xlim = xlim,
                                        xticks = xticks,
                                        avg_type='median')
    
    #%%
    # #%% SAVING IMAGE
    
    ## FIGURE NAME
    figure_name = "3_C_mu_distribution_median"
    # "2_P_mu_distribution_median" # <-- publication

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'svg' 
                 save_fig=True,)
        
    #%% FIGURE 4B
    ### Figure on C3OH comparison with other chain lengths
    
    ## DEFINING LOCATION
    path_list = {
                'GNP': {
                    'path_dict_key': 'GNP',
                    'prefix_key': 'GNP',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.1],
                    'ligands': ['C11OH'],
                    'color': 'black',
                        },
                'C3_Planar_SAMs': {
                    'path_dict_key': 'Planar_SAMs',
                    'prefix_key': 'Planar',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.1],
                    'ligands': ['C3OH'],
                    'color': 'blue',
                        },
                'Planar_SAMs': {
                                    'path_dict_key': 'Planar_SAMs',
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ['C11OH'],
                                    'color': 'red',
                 }
                
                }

    
    # #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = ligands,
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = True,
                                                       )
    
    #%% LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
        
    
    ## PLOTTING THE HISTOGRAMS
    figsize=(8, 6)
    # (9, 7) # in cm

    ## PLOTTING
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 1.5},
                                        figsize = figsize,
                                    xlim = [7, 17],
                                    xticks = np.arange(7, 18, 2),
                                    want_combined_plot = True,
                                    want_legend_all = True,
                                    avg_type = 'median',
                                    want_default_lig_colors = False)
    
    ## SETTING Y TICK
    axs[0].set_yticks(np.arange(0,1.2,0.25))
    
    
    ## FIGURE NAME
    figure_name = "4_B_comparing_mu_to_c3"
    # "2_P_mu_distribution_median" # <-- publication

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'svg' 
                 save_fig=True,)
    
    #%% FIGURE 5 - DIFFERENT SIZE DISTRIBUTIONS
    
    ## DEFINING LOCATION
    path_list = {'Dodecanethiol': {
                    'path_dict_key': ['GNP', 'GNP_6nm', 'Planar_SAMs'],
                    'prefix_key': ['GNP', 'GNP_6nm', 'Planar'],
                    'yticks': np.arange(0, 2.5, 0.5),
                    'ylim': [-0.2, 2.25],
                    'ligands': [
                            "dodecanethiol",                       
                            ],
                    'color': ['purple', 'orange', 'black'],                
                        },
                
                'OH': {
                                    'path_dict_key': ['GNP', 'GNP_6nm', 'Planar_SAMs'],
                                    'prefix_key': ['GNP', 'GNP_6nm', 'Planar'],
                                    'yticks': np.arange(0, 1.2, 0.25),
                                    'ylim': [-0.1, 1.1],
                                    'ligands': [
                                            "C11OH",                       
                                            ],
                                    'color': ['purple', 'orange', 'black'],                
                                        },
                        }
                
    #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = [],
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       )
    
    #%% LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
        
    ### FUNCTION PLOT MU DISTRIBUTION ACROSS GROUPS
    ## PLOTTING
    figsize=(9, 8) # in cm
    # FIGURE_SIZE
    # (9, 7) # in cm
    figsize=plot_tools.cm2inch(*figsize)
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 2},
                                        figsize = figsize,
                                    xlim = [5, 17],
                                    xticks = np.arange(5, 18, 2),
                                    want_legend_all = True,
                                    want_combined_plot = False)
    
    ## FIGURE NAME
    figure_name = "5_Comparison_btn_sizes"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg',
                 save_fig = True,
                )
                 
    #%% FIGURE 6B - DISTRIBUTION OF OH'S
    
    ## DEFINING LOCATION
    path_list = {'Saturated': {
                    'path_dict_key': 'GNP',
                    'prefix_key': 'GNP',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.2],
                    'ligands': [
                            "C11OH",
                            ],
                    'color': 'black',
                        },
                'Unsaturated': {    
                                    'path_dict_key': 'GNP_unsaturated',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 1.2, 0.25),
                                    'ylim': [-0.1, 1.2],
                                    'ligands': [
                                            "C11double67OH",
                                            ],
                                    'color': 'red',
                                        },             

                'Branched': {    
                                    'path_dict_key': 'GNP_branched',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 1.2, 0.25),
                                    'ylim': [-0.1, 1.2],
                                    'ligands': [
                                            "C11branch6OH",
                                            ],
                                    'color': 'blue',
                                        },   

                }
    
    ## LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = [],
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       )
    ## LOADING WATER MU VALUE    
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
    
    
    #%% PLOTTING
    ## PLOTTING THE HISTOGRAMS
    figsize=(8, 6) # in cm
    # figsize=plot_tools.cm2inch(*figsize)

    ## PLOTTING
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 1.5},
                                        figsize = figsize,
                                    xlim = [7, 10],
                                    xticks = np.arange(7, 15, 1),
                                    want_combined_plot = True,
                                    want_legend_all = True,
                                    want_default_lig_colors = False,
                                    avg_type = 'median')
    
    ## SETTING Y AXIS
    axs[0].set_ylim([-0.05, 0.60])
    
    ## FIGURE NAME
    figure_name = "6a_distribution_OH"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg',
                 save_fig = True,
                )
                 
    #%% FIGURE 6C - MEDIAN MU DISTRIBUTION
    
    ## DEFINING LOCATION
    path_list = {'Saturated GNPs': {
                    'path_dict_key': 'GNP',
                    'prefix_key': 'GNP',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.2],
                    'ligands': [
                            "dodecanethiol",
                            "C11OH",
                            "C11NH2",
                            "C11COOH",
                            "C11CONH2",
                            "C11CF3",   
                            ],
                        },
                'Unsaturated GNPs': {    
                                    'path_dict_key': 'GNP_unsaturated',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 1.2, 0.25),
                                    'ylim': [-0.1, 1.2],
                                    'ligands': [
                                            "dodecen-1-thiol",
                                            "C11double67OH",
                                            "C11double67NH2",
                                            "C11double67COOH",
                                            "C11double67CONH2",
                                            "C11double67CF3",
                                            ],
                                        },             
                'Branched GNPs': {    
                                    'path_dict_key': 'GNP_branched',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 1.2, 0.25),
                                    'ylim': [-0.1, 1.2],
                                    'ligands': [
                                            "C11branch6OH",
                                            'C11branch6CH3',
                                            'C11branch6NH2',
                                            'C11branch6COOH',
                                            'C11branch6CONH2',
                                            'C11branch6CF3',
                                            ],
                                        },   
                }
    ## LOADING MU VALUES                
    storage_dict_most_likely = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                                   ligands = [],
                                                                   main_sim_dir=MAIN_HYDRO_DIR,
                                                                   )
    
    
    # LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
    
    
    ## LOADING FOR LEAST LIKELY DISTRIBUTION
    path_list_least_likely = copy.deepcopy(path_list)
    
    ## RELABELING PREFIX KEY AND PATHDICT FOR EACH
    for each_key in path_list_least_likely:
        path_list_least_likely[each_key]['path_dict_key'] = 'GNP_least_likely'
        path_list_least_likely[each_key]['prefix_key'] = 'GNP_least_likely'
    
    ## LOADING MU VALUES                
    storage_dict_least_likely = load_mu_values_for_multiple_ligands(path_list = path_list_least_likely,
                                                                    ligands = [],
                                                                    main_sim_dir=MAIN_HYDRO_DIR,
                                                                    )
    
    #%% SYNTHESIZING THE RESULTS FROM TWO STORAGE DICTS
    
    ## DEFINING DICTIONARY

    ## LOOPING EACH DICT AND EXTRACTING INFORMATION
    df_medians = [extract_stats_values(storage_dict = each_dict,
                                       stat_key="median") for each_dict in [storage_dict_most_likely, storage_dict_least_likely] ]
    
    ## FINDING AVG AND STD
    merged_df = merge_data_frame_get_avg_std(df_list = df_medians)
            
    #%% PLOTTING BAR
    
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 6)
    
    ## GETTING FIGURE
    fig, ax = plot_bar_versus_ligs(storage_dict = storage_dict,
                                  merged_df = merged_df,
                                  fig_size_cm = fig_size_cm,
                                  width=0.25,
                                  bunched_dict = bunched_dict,
                                  colors = ['black', 'red', 'blue'],
                                  error_kw=dict(lw=1, capsize=2, capthick=1),
                                  ylabel="Median $\mu$",
                                  water_mu_value = water_mu_value)
    
    ## SETTING Y
    ax.set_ylim([8, 12])
    ax.set_yticks( np.arange(8, 13, 1) )
    
    ## FIGURE NAME
    figure_name = "6c_bar_plot_median_mu"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg',
                 save_fig = True,
                )
    
    #%% 6D Spatially heterogenous clustering of mu values
    
    ## DEFINING LOCATION
    path_list = {'Saturated GNPs': {
                    'path_dict_key': 'GNP',
                    'prefix_key': 'GNP',
                    'ligands': [
                            "dodecanethiol",
                            "C11OH",
                            "C11NH2",
                            "C11COOH",
                            "C11CONH2",
                            "C11CF3",   
                            ],
                        },
                'Unsaturated GNPs': {    
                                    'path_dict_key': 'GNP_unsaturated',
                                    'prefix_key': 'GNP',
                                    'ligands': [
                                            "dodecen-1-thiol",
                                            "C11double67OH",
                                            "C11double67NH2",
                                            "C11double67COOH",
                                            "C11double67CONH2",
                                            "C11double67CF3",
                                            ],
                                        },             
                'Branched GNPs': {    
                                    'path_dict_key': 'GNP_branched',
                                    'prefix_key': 'GNP',
                                    'ligands': [
                                            "C11branch6OH",
                                            'C11branch6CH3',
                                            'C11branch6NH2',
                                            'C11branch6COOH',
                                            'C11branch6CONH2',
                                            'C11branch6CF3',
                                            ],
                                        },   
                }
    
    ## LOADING FOR LEAST LIKELY DISTRIBUTION
    path_list_least_likely = copy.deepcopy(path_list)
    
    ## RELABELING PREFIX KEY AND PATHDICT FOR EACH
    for each_key in path_list_least_likely:
        path_list_least_likely[each_key]['path_dict_key'] = 'GNP_least_likely'
        path_list_least_likely[each_key]['prefix_key'] = 'GNP_least_likely'
    
    ## DEFINING RELATIVE PATH TO CLUSTERING PICKLE
    relative_hdbscan_clustering = os.path.join("analysis",
                                               "main_mu_clustering",
                                               "results.pickle")
        
    ## RUNNING AND LOADING
    storage_dict_most_likely = load_clustering_multiple_ligs(path_list = path_list,
                                      relative_hdbscan_clustering = relative_hdbscan_clustering)


    ## RUNNING AND LOADING
    storage_dict_least_likely = load_clustering_multiple_ligs(path_list = path_list_least_likely,
                                                              relative_hdbscan_clustering = relative_hdbscan_clustering)
        
    
    #%%
    
    ## LOOPING EACH DICT AND EXTRACTING INFORMATION
    df = [extract_stats_values(storage_dict = each_dict,
                                       stat_key=None,
                                       stats_word="num_clusters") for each_dict in [storage_dict_most_likely, storage_dict_least_likely] ]
    
    ## FINDING AVG AND STD
    merged_df = merge_data_frame_get_avg_std(df_list = df)
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 6)
    
    ## GETTING FIGURE
    fig, ax = plot_bar_versus_ligs(storage_dict = storage_dict_most_likely,
                                  merged_df = merged_df,
                                  fig_size_cm = fig_size_cm,
                                  width=0.25,
                                  bunched_dict = bunched_dict,
                                  colors = ['black', 'red', 'blue'],
                                  error_kw=dict(lw=1, capsize=2, capthick=1),
                                  ylabel="# hydrophilic clusters",
                                  water_mu_value = None)
    #%%
    ## SETTING Y LIM
    ax.set_ylim([0, 30])
    ax.set_yticks( np.arange(0, 35, 5) )
    
    ## FIGURE NAME
    figure_name = "6d_bar_plot_hydrophilic_clusters"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg',
                 save_fig = True,
                )
                 
                 
                 
    #%% FIGURE 7C - HYDROPHOBIC MAPPING
    
    ## DEFINING PARENT DIRECTORIES
    np_parent_dirs =[
             "20200618-GNP_COSOLVENT_MAPPING",
            ]

    ## DEFINING MAIN DIRECTORY
    main_dir = NP_SIM_PATH

    ## FINDING ALL PATHS
    full_path_dataframe = get_all_paths(np_parent_dirs = np_parent_dirs,
                                        main_dir = main_dir)
    
    ## DEFINING LIGAND OF INTEREST
    desired_lig = 'C11OH'
    # 'C11OH'
    
    ## GETTING PATH
    path_sim_list = full_path_dataframe.loc[full_path_dataframe.ligand == desired_lig]['path'].to_list()
    
    #%% GENERATING FRACTION OF OCCURENCES
    
    ## SORTING PATH
    path_sim_list.sort()
    
    ## DEFINING INPUTS
    inputs_frac_occur={'path_sim_list' : path_sim_list}
    
    ## EXTRACTING
    storage_frac_occurences, mapping_list = load_frac_of_occur(**inputs_frac_occur)
    
    #%% SUMMING ALL FRACTION OF OCCURENCES
    
    ## DEFINING INPUTS
    inputs_storage={"storage_frac_occurences": storage_frac_occurences}

    ## DEFINING PICKLE PATH
    pickle_path = os.path.join(path_sim_list[0],
                               ANALYSIS_FOLDER,
                               main_compute_np_cosolvent_mapping.__name__,
                               "stored_frac_occur.pickle"
                               )
    
    ## EXTRACTION PROTOCOL WITH SAVING
    fraction_sim_occur_dict = save_and_load_pickle(function = sum_frac_of_occurences, 
                                                                  inputs = inputs_storage, 
                                                                  pickle_path = pickle_path,
                                                                  rewrite = False,
                                                                  verbose = True)
    
    #%% CORRELATING FRACTION OF OCCUPANCY TO MU VALUES
    
    #### LOADING MU VALUES
    
    ## DEFINING PARENT WC FOLDER
    parent_wc_folder = "20200618-Most_likely_GNP_water_sims_FINAL"
    
    ## FINDING ORIGINAL HYDROPHOBICITY NAME
    orig_name = find_original_np_hydrophobicity_name(path_sim_list[0])
    
    ## LOADING WC GRID
    relative_path_to_pickle=os.path.join("26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",
                                          "mu.pickle",
                                          )
    
    
    path_to_mu = os.path.join(PARENT_SIM_PATH,
                              parent_wc_folder,
                              orig_name,
                              relative_path_to_pickle)
    
    ## LOADING RESULTS
    mu_array = load_pickle_results(file_path = path_to_mu,
                                   verbose = True)[0][0]
    
    
    #%% CORRELATING MU VALUES
    
    ## CREATING NEW COLOR MAP
    tmap = plot_tools.create_cmap_with_white_zero(cmap = plt.cm.jet,
                                                  n = 100,
                                                  perc_zero = 0.125)

    ## GENERATING PLOT
    figure_size = (17.1, 6)
    figsize = plot_tools.cm2inch(figure_size)
    # figsize = (figsize[0]*2, figsize[1])

    ## ADDING FIGURE
    fig, ax = plot_gaussian_kde_scatter(fraction_sim_occur_dict = fraction_sim_occur_dict,
                                        mu_array = mu_array,
                                        figsize = figsize,
                                        solvent_list = [ 'PRO', 'HOH' ],
                                        y_range = (-0.2, 1.2),
                                        x_range = (8, 13),
                                        nbins = 300, # 100
                                        vmin = 0,
                                        vmax = 2,
                                        cmap = tmap, # plt.cm.Greens,
                                        )
    
    ## FIGURE NAME
    figure_name = "7B_cosolvent_map"
    


    #%%
    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'pdf', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    
    
    #%% END_POINT MANUSCRIPT
    
    ############################################################################
    ### SUPPORTING INFORMATION 
    ############################################################################

    #%% SI FIGURE 2B - BUNDLING GROUPS
    

    ## DEFINING LIGAND OUTPUT
    ligand = "dodecanethiol"
    
    ## GETTING LIGAND
    dir_name = "EAM_300.00_K_2_nmDIAM_%s_CHARMM36jul2017_Trial_1"%(ligand)
    
    
    ## DEFINING PATH TO PICKLE
    path_pickle = os.path.join(PATH_TO_NP_WATER,
                               dir_name,
                               MOST_LIKELY_DIR,
                               MOST_LIKELY_PICKLE)
    
    ## LOADING PICKLE
    most_probable_np = pd.read_pickle(path_pickle)[0] # load_class_pickle(path_pickle)
    #%% FIGURE 2B
    ## LOADING PLOTTING FUNCTIONS
    from MDDescriptors.application.np_hydrophobicity.extract_np_most_likely import plot_num_bundles_and_nonbundle_vs_frame, plot_prob_in_bundle, plot_cross_entropy_vs_bundles
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 6)
    figsize = plot_tools.cm2inch(fig_size_cm)
    
    ## PLOTTING
    fig, axs = plot_num_bundles_and_nonbundle_vs_frame(most_probable_np,
                                                       figsize = figsize,
                                                       want_nonbundle = False,
                                                       relative_zero = 10)
    ## SETTING LIMIS
    axs[0].set_ylim([0,7.5])
    axs[0].set_xlim([5, 55])
    axs[0].legend()
    
    ## SETTING AXIS
    figure_name = "SI_2B_Bundling_dodecanethiol"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
                 
    #%% FIGURE 2C
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 6)
    ## DEFINING LIGAND INDEX
    ligand_index = np.arange(len(most_probable_np.prob_ligand_in_bundle))
    prob_in_bundle = most_probable_np.prob_ligand_in_bundle

    ## DEFINING AXIS KWARGS
    ax_kwargs = {
            'x_axis_labels': (0, 80, 20),
            'y_axis_labels': (0, 1, 0.2),
            'ax_x_lims'    : [-5, 85 ],
            'ax_y_lims'    : [0, 1.05],
            }
    
    ## PLOTTING FIGURE
    fig, ax = plot_prob_in_bundle(ligand_index = ligand_index,
                                  prob_in_bundle = prob_in_bundle,
                                  figure_size = fig_size_cm,
                                  ax_kwargs = ax_kwargs
                                  )
    ## SETTING AXIS
    figure_name = "SI_2C_probability_within_bundle"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    #%% FIGURE 2D: CROSS ENTROPY
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 6)
    
    ## DEFINING AXIS KWARGS
    ax_kwargs = {
            'x_axis_labels': None, # (0, 80, 20),
            'y_axis_labels': None, # (0, 1, 0.2),
            'ax_x_lims'    : (1.5, 7.5), # [-5, 85 ],
            'ax_y_lims'    : (-5, 110), # [0, 1.1],
            }

    ## GENERATING PLOT
    fig, ax = plot_cross_entropy_vs_bundles(most_probable_np = most_probable_np,
                                            figure_size = fig_size_cm,
                                            ax_kwargs = ax_kwargs)
    
    
    ## SAV ING FIGURE
    figure_name = "SI_2D_cross_entropy_vs_bundles"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    
    
    #%% FIGURE S3: LIGAND FLUCTUATIONS AND WC INTERFACE PROPERTIES
    
    
    ## DEFINING DEFAULT SPRING CONSTANT
    DEFAULT_SPRING_CONSTANT = 50

    #%%

    ## CREATING DICT
    simulation_dict_debug_wc = {
            'GNP':{
                    'simulation_dir': "NP_HYDRO_SI_GNP_DEBUGGING_SPRING",
                    "is_planar" : False,
                    },
            'planar' : {
                    'simulation_dir': "NP_HYDRO_SI_PLANAR_SAMS_DEBUGGING_SPRING",
                    "is_planar" : True,
                    },
            }
    
    #### LOADING DATA FOR LIGAND FLUCTATIONS AND WC POINTS
    for each_key in simulation_dict_debug_wc:
        ## DEFINING INPUTS
        simulation_dir = simulation_dict_debug_wc[each_key]['simulation_dir']
        is_planar = simulation_dict_debug_wc[each_key]['is_planar']
        
        ## DEFINING PATH TO LIST
        path_to_sim_list = os.path.join(MAIN_NP_DIR, simulation_dir)    
        ## GETTING TRAJECTORY OUTPUT
        traj_output = extract_multiple_traj(path_to_sim_list =path_to_sim_list )
        
        ## GETTING DECODED NAME LIST
        traj_output.decode_all_sims(decode_type='nanoparticle')
        
        ## LOADING RMSF
        traj_output = compute_rmsf_across_traj(traj_output)
        
        ## DEFINING Z DISTANCES
        if is_planar is True:
            compare_type = "z_dist"
        else:
            compare_type = "min_dist"
        
        ## COMPUTING AVERAGE DEVIATION
        traj_output = compute_avg_deviation_across_traj(traj_output = traj_output,
                                                        ref_grids = None,
                                                        is_planar = is_planar,
                                                        compare_type = compare_type,
                                                        omit_first = False,
                                                        use_last_grid_ref = True)
        
        ## STORING
        simulation_dict_debug_wc[each_key]['traj_output'] = traj_output
    
    
    
    #%%
    
    ### LOOPING THROUGH EACH SIM
    for idx, each_sim in enumerate( traj_output.full_sim_list ):
        if idx == 0:
#            ## LOOPING THROUGH EACH LIGAND AND GENERATING DATA
#            results = traj_output.load_results(idx = idx,
#                                               func = main_compute_gmx_rmsf)[0]
    
            ## GETTING THE RESULTS
            results = traj_output.load_results(idx = idx,
                                               func = main_compute_wc_grid_multiple_subset_frames)
            
            path_gro = os.path.join(each_sim, "sam_prod.gro")
            updated_grid = remove_grid_for_planar_SAMs(path_gro = path_gro,
                                                       grid = results[0][0])
    
            ## DEFINING UPPER AND LOWER
            upper_grid, lower_grid = updated_grid.find_upper_lower_grid(grid = updated_grid.new_grid)
    
    #%% PLOTTING
    
    
    ## DEFINING FIGURE SIZE    
    figure_size = (17.1/2, 11)
    figsize = plot_tools.cm2inch(figure_size)
    
    ## ADDING TO DICTIONARY Y TICKS
    simulation_dict_debug_wc['GNP']['yaxis'] = {
            'Ligand RMSF': {
                    'ticks': np.arange(0, 0.28, 0.04),
                    'lim': [0, 0.24]
                    },
            'SAM-water interface': {
                    'ticks': np.arange(0, 0.10, 0.02),
                    'lim': [0, 0.08],
            }
            }
            
    simulation_dict_debug_wc['planar']['yaxis'] = {
            'Ligand RMSF': {
                    'ticks': np.arange(0, 0.10, 0.02),
                    'lim': [0, 0.10]
                    },
            'SAM-water interface': {
                    'ticks': np.arange(0, 0.8, 0.2),
                    'lim': [-0.1, 0.7],
            }
            }
    
    ## DEFINING TYPES
    fig_types = ["Ligand RMSF", "SAM-water interface"]
    
    ## LOOPING
    for each_type in fig_types:
    
        ## CREATING SUBPLOTS
        fig, axes = plt.subplots(nrows=len(simulation_dict_debug_wc), ncols=1, sharex = True,
                                 figsize=figsize)
        
        ## PLOTTING EACH ONE
        for sim_idx, each_key in enumerate(simulation_dict_debug_wc):
            ## DEFINING AXIS
            ax = axes[sim_idx]
            
            ## DEFINING TRAJ OUTPUT
            traj_output = simulation_dict_debug_wc[each_key]['traj_output']
            
            ## DRAWING OPTIMAL SPRING CONSTANT LINE
            ax.axvline(x = DEFAULT_SPRING_CONSTANT, color = 'k', linestyle = '--')
            
            ## FOR LIGAND RMSF
            if each_type == "Ligand RMSF":
                fig, ax = plot_rmse_vs_spring_constant(df = traj_output.decoded_name_df,
                                                       fig = fig,
                                                       ax = ax)
            elif each_type == "SAM-water interface":
                fig, ax = plot_avg_deviation_vs_spring_constant(df = traj_output.decoded_name_df,
                                          fig = fig,
                                          ax = ax,
                                          )
            ## REMOVING REDUNDANT LEGEND
            if sim_idx > 0:
                ax.get_legend().remove()
            
            ## EDITING Y LABELS
            yaxis_info = simulation_dict_debug_wc[each_key]['yaxis'][each_type]
            yticks = yaxis_info['ticks']
            ylim = yaxis_info['lim']
            
            ## SETTING LABELS
            ax.set_yticks(yticks)
            ax.set_ylim(ylim)
            
            ## ADDING TITLE
            ax.text(.5,.9,each_key,
                horizontalalignment='center',
                transform=ax.transAxes)
                
        ## REMOVING WHITE SPACE
        plt.subplots_adjust(wspace=0, hspace=0)

        ## SAVING FIGURE
        figure_name = "SI_3_debug_spring_constant%s"%(each_type)
        plot_tools.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'svg', # 'pdf',
                     save_fig = True,
                     dpi=300, # lowering resolution
                    )
                
        
    #%%

    
    ''' DEBUGGING 
    ## GETTING THE RESULTS
    results = traj_output.load_results(idx = 10, # 0,
                                       func = main_compute_wc_grid_multiple_subset_frames)
    
    ## GETTING ALL GRIDS
    grids = [ results[0][each_idx] for each_idx in range(len(results[0])) ]
    
    ## GETTING NEW GRIDS
    new_grids = [ updated_grid.find_new_grid(grid = each_grid) for each_grid in grids]
    
    #%%
    figure = plot_tools.plot_3d_points_with_scalar(xyz_coordinates = new_grids[0],
                                                   )
    ## GETTING AVG DEVIATIONS
    avg_deviations = compute_avg_deviation(results = (new_grids, 0), 
                                           compare_type = 'min_dist',
                                           use_last_grid_ref = True)
    print(avg_deviations)
    
    '''
    
    #%% FIGURE S4: DEBUGGING WC INTERFACE PARAMETERS
    
    plt.close('all')
    
    ## DEFINING LIST OF TYPES
    type_dict = {
            'Planar': 
                {'path_dict_key': 'Planar_SAMs',
                 'prefix_key': 'Planar',
                 },
            'GNP': {
                 'path_dict_key': 'GNP',
                 'prefix_key': 'GNP',
                    }
            }
            

    ## DEFINING GRO FILE NAME
    gro_file_name = "sam_prod_2000_50000-heavyatoms.gro" 
    
    ## DEFINING LIGANDS
    ligands = ["dodecanethiol",
               "C11OH"]
    ## PICKLE FOR REMOVING GRO
    remove_grid_pickle = "remove_grid.pickle"

    ## DEFINING CONTOUR ARRAY AND PROBE RADIUS ARRAY
    contour_array = np.arange(0, 32, 2)
    probe_radius = np.arange(0, 0.40, 0.05)
    
    ## DEFINING PARMAETERS
    selected_params = {
            'c': 26,
            'r': 0.33,
            }
    
    ## CREATING NEW COLOR MAP
    tmap = plot_tools.create_cmap_with_white_zero(cmap = plt.cm.jet,
                                                  n = 100,
                                                  perc_zero = 0.05)
    ## DEFINING IFGURE SIZE    
    figure_size = (17.1, 11)
    figsize = plot_tools.cm2inch(figure_size)
    
    ## CREATING SUBPLOTS
    fig, axes = plt.subplots(nrows=len(type_dict), ncols=len(ligands),
                             figsize=figsize)
    
    ## LOOPING THROUGH EACH ONE
    for type_idx, each_key in enumerate(type_dict):
        
        ## DEFINING PATHS
        path_to_sim_parent = os.path.join(PARENT_SIM_PATH, 
                                      PATH_DICT[type_dict[each_key]['path_dict_key']])
        

        ## LOOPING THROUGH LIGANDS
        for lig_idx, each_lig in enumerate(ligands):
            ## DEFINING DIRECTORY NAME
            dir_name = ''.join([PREFIX_SUFFIX_DICT[type_dict[each_key]['prefix_key']]['prefix'],
                                each_lig,
                                PREFIX_SUFFIX_DICT[type_dict[each_key]['prefix_key']]['suffix'] 
                                ])
            ## PATH TO CURRENT SIM
            path_to_current_sim = os.path.join(path_to_sim_parent,
                                               dir_name)
            
            ## DEFINING PATH TO GRID
            path_to_grid_folder = os.path.join(path_to_current_sim,
                                               DEFAULT_WC_ANALYSIS,
                                               GRID_LOC)
            
            ## LOCATING PICKLE
            path_to_debug = os.path.join(path_to_grid_folder,
                                         DEBUG_PICKLE)
                        
            ## DEFINING PATH TO STORAGE PICKLE
            storage_pickle = debug_wc_interface.__name__ + ".pickle"
            path_to_storage_pickle = os.path.join(path_to_grid_folder, storage_pickle)
            
            ## LOADING TRAJ ONLY IF NEEDED
            if os.path.exists(path_to_storage_pickle) is False:
                    
                ## LOADING THE GRO FILE
                traj_data = import_tools.import_traj(directory = path_to_current_sim,
                                                     structure_file = gro_file_name,
                                                     xtc_file = gro_file_name)
            else:
                traj_data = None
            
            ## LOADING THE PICKLE
            grid, interface, avg_density_field = pickle_tools.load_pickle(file_path = path_to_debug)[0]
            
            ## DEFINING IF PLANAR
            if each_key == 'Planar':
                print("Planar is turned on for 'Planar' key")
                is_planar = True
            else:
                is_planar = False
        
            if is_planar is True:
                remove_grid_path = os.path.join(path_to_grid_folder,
                                                remove_grid_pickle)
                ## REMOVING GRID FUNCTION
                remove_grid_func = pickle_tools.load_pickle(file_path = remove_grid_path)[0][0]
                
                ## REMOVING GRID POINTS
                grid = extract_new_grid_for_planar_sams(grid = grid,
                                                        water_below_z_dim = remove_grid_func.water_below_z_dim,
                                                        water_above_z_dim = remove_grid_func.water_above_z_dim )[0]
            else:
                remove_grid_func = None
        
            ## DEBUGGIN WC NTERFACE
            wc_debug = debug_wc_interface()
            
            ''' Visualize the WC interface
            ## DEFINING CLASS FUNCTION
            fig = wc_debug.plot_wc_density_field_with_traj(traj_data = traj_data,
                                                            avg_density_field = avg_density_field ,
                                                            interface = interface,
                                                            size = (1000, 1000),
                                                         )    
            '''
            

            
            ## DEFINING INPUTS
            input_dict = {
                    'contour_array': contour_array,
                    'probe_radius': probe_radius,
                    'traj_data': traj_data,
                    'avg_density_field': avg_density_field,
                    'remove_grid_func' : remove_grid_func,
                    'interface': interface,
                    }
            
            # bash extract_hydration_maps_with_python.sh > hydration.out 2>&1 &
            ## RUNNING FUNCTION
            storage_frac_grid_points = pickle_tools.save_and_load_pickle(function = wc_debug.loop_to_get_contour_vs_probe_radius,
                                                                         inputs = input_dict,
                                                                         pickle_path =  path_to_storage_pickle,
                                                                         rewrite=False)
            
            ## DEFINING AX
            ax = axes[type_idx][lig_idx]
            ## GENERATING PLOT FOR IT
            fig, ax, cont = wc_debug.plot_contour_map_of_contour_level_vs_probe_radius(x = contour_array,
                                                                                 y = probe_radius,
                                                                                 heatmap = storage_frac_grid_points,
                                                                                 levels = [0, 0.05, 0.25, 0.5, 0.75, 1.0],
                                                                                 cmap = tmap,
                                                                                 fig = fig,
                                                                                 ax = ax,
                                                                                 want_color_bar = False,
                                                                                 return_contourf = True
                                                                                 )
            
            ## ADDING POINT TO PLOT
            ax.plot(selected_params['c'],
                    selected_params['r'],
                    linestyle = "None",
                    marker = "o",
                    color = "green",
                    markersize = 8,
                    markeredgewidth = 1.5,
                    markeredgecolor = 'k',
                    label = "Selected"
                    )
            ## ADDING LEGEND
            ax.legend(loc='lower right')
            
            ## ADDING TITLE
#            ax.set_title(each_key + "_" + each_lig)
            
    ## ADDING COLOR BAR
    fig.subplots_adjust(right=0.85,wspace = 0.3)
    cbar_ax = fig.add_axes([0.9, 0.2 , 0.02, 0.7])
    current_cbar = fig.colorbar(cont, cax=cbar_ax)

    ## SAVING FIGURE
    figure_name = "SI_4_wc_debugging"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    
    #%% TABLE S1
    
    ## DEFINING LIST OF SIMULATIONS
    types_list = [
            'GNP',
            'Planar_C3',
            'Planar_SAMs',
            'GNP_6nm',
            ]
    
    ## DEFINING PROD GRO FILE
    PROD_GRO="sam_prod.gro"
    
    ## CREATING LIST
    storage_list = []
    
    ## LOOPING THROUGH THE LIST
    for each_type in types_list:
        ## FINDING PARENT DIR
        parent_dir = PATH_DICT[each_type]
        ## GLOBBING
        path_to_parent = os.path.join(PARENT_SIM_PATH, 
                                      parent_dir)
        ## GETTING LIST OF DIRS
        list_of_dirs = glob.glob(path_to_parent + "/*") 
        
        ## LOOPING THROUGH EACH DIR
        for dir_idx, each_dir in enumerate(list_of_dirs):
            ## DEFINING OUTPUT DICT
            output_dict = {}
            
            ## FINDING BASENAME
            dir_basename = os.path.basename(each_dir)
            
            ## STORING NAME
            output_dict['Dirname'] = dir_basename
            
            ## DEFINING LABELS
            label_dict = decode_name(dir_basename, decode_type = 'nanoparticle')
            
            ## DEFINING DICT
            input_output_key_dict = {
                    'diameter': 'size',
                    'shape': 'shape',
                    }
            ## STORING INPUT
            for input_key in input_output_key_dict:
                ## STORING LABELS
                output_dict = try_to_add(output_dict = output_dict, 
                                         label_dict = label_dict, 
                                         input_key = input_key, 
                                         output_key = input_output_key_dict[input_key])
            
            ## DEFINING WHETHER PLANAR
            if output_dict['shape'] == "Planar":
                is_planar = True
            else:
                is_planar = False
            
            ## LOADING GRO FILE
            traj_data = import_tools.import_traj(directory = each_dir,
                                                 structure_file = gro_file_name,
                                                 xtc_file = gro_file_name)            
            
            ## EXTRACTING FROM FILE
            output_dict, structure = extract_traj_information(traj_data,
                                                              output_dict = output_dict,
                                                              is_planar = is_planar)
            
            ## GETTING TOTAL WC GRID POINTS
            path_to_grid = os.path.join(each_dir,
                                        DEFAULT_WC_ANALYSIS,
                                        GRID_LOC,
                                        GRID_OUTFILE
                                        )
            
            ## LOADING DATAFILE
            grid_results = load_datafile(path_to_grid)
            
            ## STORING
            output_dict['total_wc_pts'] = len(grid_results)
            
            
            ## STORING
            storage_list.append(output_dict)
            
    ## CREATING DATAFRAME
    df = pd.DataFrame(storage_list)
    
    ## OUTPUT TO CSV
    figure_name = "TABLE_S1_systems_information.csv"
    path = os.path.join(IMAGE_LOC,
                        figure_name)
    
    df.to_csv(path)
    
    
    #%% FIGURE S5: INDUS CALCULATION RESULTS
    
    #### PART A
    sheet_dict = {
            'CH3':{
                    'tab': "INDUS HISTO CH3",
                    'color': 'k',
                    },
            'OH':
                {
                        'tab': "INDUS HISTO OH",
                        'color': 'r',
                        }    
                }
    
    ## DEFINING FIGURE SIZE    
    figure_size = (17.1, 7)
    figsize_inches=plot_tools.cm2inch(*figure_size)
    
    ## CREATING FIGURES
    fig, axs = plt.subplots(nrows=1, 
                            ncols = len(sheet_dict),
                            sharey=True,
                            figsize = figsize_inches)
    
    ## LOOPING
    for idx,key in enumerate(sheet_dict):
        
        ## DEFINING KEY
        ax = axs[idx]
        
        ## Defining COLOR
        color = sheet_dict[key]['color']
        
        ## DEFINING SHEET
        each_sheet = sheet_dict[key]['tab']
        ## LOADING THE DATA
        histogram_data = pd.read_excel(PATH_TO_INDUS_SI,
                                       sheet_name = each_sheet)

        
        ## PLOTTING
        x = histogram_data['N']
        
        ## LOOPING (Removing 'N')
        windows = histogram_data.columns.to_list()[1:]
        
#        cmap = plt.get_cmap('viridis')
#        colors = [cmap(i) for i in np.linspace(0, 1, len(windows))]
        
        ## LOOPING
        for window_idx,each_window in enumerate(windows):
            ## DEFINING Y
            y = histogram_data[each_window]
            ## PLOTTING
            ax.plot(x,y,
                    color = color ) # colors[window_idx]
        
        ## SHOWINGY LABELS
        ax.yaxis.set_tick_params(labelbottom=True)
        
        ## ADDING LABELS
        ax.set_xlabel("N")
        ax.set_ylabel("P(N)")
        
        ## ADDING TITLE
        ax.text(.5,.9,key,
            horizontalalignment='center',
            transform=ax.transAxes)
#        ax.set_title(key)
        
        ## tight layout
        fig.tight_layout()
        

    ## SAVING FIGURE
    figure_name = "SI_5_AB_indus_histograms"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    #%% FIGURE S5, PART C,D

    ## DEFINING FIGURE SIZE    
    figure_size = (17.1/2, 8)
    figsize_inches=plot_tools.cm2inch(*figure_size)
        
    #### PART B
    sheet_dict = {
            'CH3':{
                    'tab': "INDUS_EQUIL_TIME_CH3",
                    'color': 'k',
                    'yticks': np.arange(34,41, 1),
                    'ylim': [34, 40],
                    'xlim': [-250, 2250],
                    'xticks': np.arange(0,2500, 500),
                    },
            'OH':
                {
                    'tab': "INDUS_EQUIL_TIME_OH",
                    'color': 'red',
                    'yticks': np.arange(90, 100, 2),
                    'ylim': [90, 100],
                    'xlim': [-250, 2250],
                    'xticks': np.arange(0,2500, 500),
                        }    
                }
    
    ## PLOTTING
    fig, axs = plot_subplots_mu_vs_time(sheet_dict,
                                 figsize_inches,
                                 x_data_key = 'equil time',
                                 x_axis_label = "Equil. time (ps)",
                                 y_axis_label = "$\mu_A$",
                                errorbar_format={
                                        'linestyle' : "-",
                                        "fmt": ".",
                                        "capthick": 1.5,
                                        "markersize": 8,
                                        "capsize": 3,
                                        "elinewidth": 1.5,
                                        }
                                 )

    ## SAVING FIGURE
    figure_name = "SI_5_C_indus_equiltime"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    
    
    #### PART C
    sheet_dict = {
            'CH3':{
                    'tab': "INDUS_CONVERGE_TIME_CH3",
                    'color': 'k',
                    'yticks': np.arange(34,41, 1),
                    'ylim': [34, 40],
                    'xlim': [-250, 3250],
                    'xticks': np.arange(0,3500, 500),
                    },
            'OH':
                {
                        'tab': "INDUS_CONVERGE_TIME_OH",
                        'color': 'red',
                        'yticks': np.arange(90, 100, 2),
                        'ylim': [90, 100],
                        'xlim': [-250, 3250],
                        'xticks': np.arange(0,3500, 500),
                        }    
                }
    
    ## PLOTTING
    fig, axs = plot_subplots_mu_vs_time(sheet_dict,
                                 figsize_inches,
                                 x_data_key = 'sim time',
                                 x_axis_label = "Sim. time (ps)",
                                 y_axis_label = "$\mu_A$",
                                errorbar_format={
                                        'linestyle' : "-",
                                        "fmt": ".",
                                        "capthick": 1.5,
                                        "markersize": 8,
                                        "capsize": 3,
                                        "elinewidth": 1.5,
                                        }
                                 )
    ## SAVING FIGURE
    figure_name = "SI_5_D_indus_simtime"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    
    #%% FIGURE S6 - COMPUTING MU_L

    #### DEFAULTS FOR FIGURE S6
    ## DEFINING FRAME RATE
    frame_rate = 6000
    
    ## DEFINING COLORS
    color_dict = {
            'dodecanethiol': 'black',
            'C11OH': 'red',
            }
    
    #%%
    
    ### DEFINING DICTIONARY
    planar_sam_mu_L = {
            'path_dict_key': "Planar_SAMs",
            'prefix_key': "Planar",
            'ligands': ['dodecanethiol', "C11OH"]
            }
    
    ## GETTING NEIGHBOR PLANAR ARRAY PICKLES
    neighbor_array_dict =  load_data_for_neighbors_array(input_dict = planar_sam_mu_L,
                                      neighbors_pickle = r"2000-50000.pickle",
                                      main_sim_dir=MAIN_HYDRO_DIR)
            
    ## COMPUTING MU DISTRUBTION OBJECT
    mu_dist_obj_dict = compute_mu_objects(neighbor_array_dict = neighbor_array_dict)
    
    ## PERFORMING CONVERGENCE TEST
    storage_mu_convergence_dict = compute_mu_convergence_using_neighbors_dict(neighbor_array_dict,
                                                                              frame_rate = frame_rate,
                                                                              )

    #%% ## PLOTTING FOR SI_7 A,B
    ## DEFINING FIGURE SIZE
    fig_size_cm = (17.1, 7)

    ## SHOWING FOR POINT
    grid_index = 24 # Used for planar SAM dodecanethiol
    ## GETTING PERCENT ERROR
    percent_error = 10
    convergence_type = "percent_error"

    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = False, 
                             figsize = plot_tools.cm2inch( *fig_size_cm ))
    
    ## PLOTTING FOR P(N)
    ax = axs[0]
    for each_lig in mu_dist_obj_dict:
        
        ## DEFINING MU 
        mu_dist_obj = mu_dist_obj_dict[each_lig]
        
        ## GETTING COLOR
        color = color_dict[each_lig]
        
        ## PLOTTING FIGURE
        fig, ax = mu_dist_obj.plot_log_p_N_for_grid(grid_index = grid_index,
                                                    fig_size_cm = fig_size_cm,
                                                    fig = fig,
                                                    ax = ax,
                                                    scatterkwargs={'color': color,
                                                                   'label': each_lig},
                                                    line_dict={
                                                                'linestyle': '--',
                                                                'color': color
                                                                },
                                                    )
    
    ## ADDING LEGEND
    ax.legend()
    
    ## RE-SIZING
    ax.set_xlim([-0.25,2.5])
    ax.set_ylim([-20,0])
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## DEFINING SECOND AX
    ax = axs[1]
    
    ## PLOTTING FOR CONVERGENCE
    for each_lig in storage_mu_convergence_dict:
        ## DEFINING RESULTS
        current_output_dict = storage_mu_convergence_dict[each_lig]
        mu_debug = current_output_dict['mu_debug']
        mu_storage_reverse = current_output_dict['mu_storage_reverse']
        frame_list = current_output_dict['frame_list']
        
        ## GETTING COLOR
        color = color_dict[each_lig]
        
        ## GETTING CONVERGENCE INFORMATION
        theoretical_bounds, index_converged, sampling_time_x_converged, x \
                        = mu_debug.main_compute_sampling_time_from_reverse(mu_storage_reverse = mu_storage_reverse,
                                                                           frame_list = frame_list,
                                                                           percent_error = percent_error,
                                                                           convergence_type = convergence_type,
                                                                           )
        ## DEFINING X
        x_array = np.array(x)/1000.0
        
        ## PLOTTING CONVERGENCE
        fig, ax = plot_convergence_of_mu( mu_storage_reverse = mu_storage_reverse,  
                                         x = x_array,
                                         grid_index = grid_index,
                                         theoretical_bounds = theoretical_bounds,
                                         index_converged = index_converged,
                                         fig = fig,
                                         ax = ax,
                                         color = color,
                                         xlabel = "Sampling time (ns)"
                                         )
        
    ## SETTING LIMITS
    frame_rate_ns = frame_rate / 1000.0
    ax.set_xlim([x_array[0] - 2, x_array[-1] + 2])
    ax.set_xticks(np.arange(x_array[0], x_array[-1] + frame_rate_ns, frame_rate_ns))
    
    ## SETTING Y LIM
    ax.set_ylim([5, 15])
    ax.set_yticks(np.arange(5,17,2))

    ## SETTING AXIS
    figure_name = "SI_6_Planar_SAM_mu_L_values"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', 
                 save_fig=True,)
    
    
    
    
        
    #%% FIGURE S6 - SAMPLING TIMES
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (17.1/2, 7)
    
    ## DEFINING AXIS LIMITS
    storage_mu_convergence_dict['dodecanethiol']['limits'] = {
            'ylim': [0, 0.20],
            'yticks': np.arange(0,0.24,0.05)
            }
    
    storage_mu_convergence_dict['C11OH']['limits'] = {
            'ylim': [0, 0.20],
            'yticks': np.arange(0,0.20,0.05)
#            'ylim': [0, 0.10],
#            'yticks': np.arange(0,0.10,0.025)
            }
    
    ## PLOTTING
    fig, axs = plot_minimum_sampling_time_normalized(storage_mu_convergence_dict,
                                                     color_dict = color_dict,
                                                     percent_error = percent_error,
                                                     convergence_type = convergence_type)
    
    ## SETTING AXIS
    figure_name = "SI_6_C_sampling_time_mu"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', 
                 save_fig=True,)
    
    #%% REPEAT FIGURE S6 FOR GNPS
    ### DEFINING DICTIONARY
    GNP_mu_L = {
            'path_dict_key': "GNP",
            'prefix_key': "GNP",
            'ligands': ['dodecanethiol', "C11OH"]
            }
    
    ## GETTING NEIGHBOR PLANAR ARRAY PICKLES
    gnp_neighbor_array_dict =  load_data_for_neighbors_array(input_dict = GNP_mu_L,
                                      neighbors_pickle = r"2000-50000.pickle",
                                      main_sim_dir=MAIN_HYDRO_DIR)
    ## COMPUTING MU DISTRUBTION OBJECT
    gnp_mu_dist_obj_dict = compute_mu_objects(neighbor_array_dict = gnp_neighbor_array_dict)
    

    ## PERFORMING CONVERGENCE TEST
    gnp_storage_mu_convergence_dict = compute_mu_convergence_using_neighbors_dict(neighbor_array_dict = gnp_neighbor_array_dict,
                                                                              frame_rate = frame_rate,
                                                                              )
    #%%
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (17.1/2, 7)    
    
    ## DEFINING AXIS LIMITS
    gnp_storage_mu_convergence_dict['dodecanethiol']['limits'] = {
            'ylim': [0, 0.20],
            'yticks': np.arange(0,0.24,0.05)
            }
    
    gnp_storage_mu_convergence_dict['C11OH']['limits'] = {
            'ylim': [0, 0.20],
            'yticks': np.arange(0,0.20,0.05)
            }
    
    
    ## PLOTTING
    fig, axs = plot_minimum_sampling_time_normalized(gnp_storage_mu_convergence_dict,
                                                     color_dict = color_dict,
                                                     percent_error = percent_error,
                                                     convergence_type = convergence_type)
    
    ## SETTING X LIMITS
    axs[0].set_xlim([-3, 51])

    ## SETTING AXIS
    figure_name = "SI_6_D_sampling_time_mu_GNPs"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', 
                 save_fig=True,)
    
    
    #%% FIGURE S7: RESTRAINED VERSUS UNRESTRAINED LIGANDS
    
    ## DEFINING LIGANDS
    ligands = ["dodecanethiol",
               "C11CF3", 
               "C11NH2", # Turn on for publication image
               "C11CONH2",
               "C11OH",
               "C11COOH",               
                ]
    
    ## DEFINING GRID DETAILS
    wc_analysis=DEFAULT_WC_ANALYSIS
    mu_pickle = MU_PICKLE

    
    ## DEFINING LOCATION
    path_list = {
                'Restrained_SAMs': {
                                    'path_dict_key': 'Planar_SAMs',
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ligands,
                                        },
                 'Unrestrained_SAMs': { 
                     'path_dict_key': 'Planar_SAMs_unrestrained',
                    'prefix_key': 'Planar_no_restraint', # 'Planar',
                    'yticks': np.arange(0, 2.5, 0.5),
                    'ylim': [-0.1, 2.25],
                        },
                 }

    # #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = ligands,
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = True,
                                                       )
    
    
    #%% LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
    
    #%% PLOTTING DISTRIBUTION
    
    ## PLOTTING THE HISTOGRAMS
    # figsize=(9,7) # in cm # <-- publication size
    
    ## GETTING FIG SIZE
    figsize=(8.5, 7.5)
    # (8.5, 8.5)
    # (11,11) # in cm
    figsize=plot_tools.cm2inch(*figsize)

    
    ## DEFINING X LIM
    xlim = [5, 17]
    xticks = np.arange(5, 18, 2)

    ## PLOTTING
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 1.5},
                                        figsize = figsize,
                                        xlim = xlim,
                                        xticks = xticks,
                                        avg_type='median')
    
    ## FIGURE NAME
    figure_name = "SI_7A_mu_distribution_restrained_vs_unrestrainted"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'svg' 
                 save_fig=True,)
    
    
    #%% FIGURE S7B
    ## GETTING EXTRACTED STATS
    df = extract_stats_values(storage_dict = storage_dict,
                                       stat_key="median")
    
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (8, 7.5)

    ## CREATING FIGURE
    fig, ax= plot_bar_stats_key_from_df(df,
                                        water_mu_value = water_mu_value,
                                        fig_size_cm = fig_size_cm)
    
    
    ## SETTING Y
    ax.set_ylim([5, 17])
    ax.set_yticks( np.arange(5, 19, 2) )
    
    ## FIGURE NAME
    figure_name = "SI_7B_bar_plot_comparison"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'svg', # 'svg' 
                 save_fig=True,)
    
    #%% FIGURE S8: Sampling time for cosolvent mapping 
    
    ## DEFINING PARENT DIRECTORIES
    np_parent_dirs =[
             "20200618-GNP_COSOLVENT_MAPPING",
            ]

    ## DEFINING MAIN DIRECTORY
    main_dir = NP_SIM_PATH

    ## FINDING ALL PATHS
    full_path_dataframe = get_all_paths(np_parent_dirs = np_parent_dirs,
                                        main_dir = main_dir)
    
    ## DEFINING LIGAND OF INTEREST
    desired_lig = 'C11OH'
    # 'C11OH'
    
    ## GETTING PATH
    path_sim_list = full_path_dataframe.loc[full_path_dataframe.ligand == desired_lig]['path'].to_list()
    
    #%% GENERATING FRACTION OF OCCURENCES
    
    ## SORTING PATH
    path_sim_list.sort()
    
    ## DEFINING INPUTS
    inputs_frac_occur={'path_sim_list' : path_sim_list}
    
    ## EXTRACTING
    storage_frac_occurences, mapping_list = load_frac_of_occur(**inputs_frac_occur)
    
    #%% RUNNING SAMPLING TIME CALCULATIONS
    
    ## DEFINING FIGURE PREFIX
    fig_prefix = "SI_8_"
    
    ## CLOSING ALL FIGURES
    plt.close('all')
    
    ## GETTING OCCURENCES AS A FUNCTION OF TIME
    sampling_time_occur_dict = generate_sampling_occur_dict(storage_frac_occurences)
    
    ## DEFINING SOLVENT LIST
    solvent_list = ['HOH', 'PRO']
    
    ## GETTING LABEL
    solvent_yaxis_lim = {
            'HOH': {
                    'ylim': [0.80, 1.10] ,
                    'yticks': np.arange(0.80, 1.20, 0.10),
                    },
            'PRO': {
                    'ylim': [0.10, 0.40] ,
                    'yticks': np.arange(0.10, 0.50, 0.10),
                    },

            }
    
    ## DEFINING DELTA
    delta = 0.10
    convergence_type = 'value'
    frame_rate = 500 # frames

    frame2ns = 10.0/1000.0
    
    ## DEFINING FIGURE SIZE
    fig_size_cm = (17.1/2, 11)
    
    ## LOOPING THROUGH SOLVENTS
    for solvent in solvent_list:
    
        ## CREATING SUBPLOTS
        fig, axs = plt.subplots(nrows = 2, 
                                ncols = 1, 
                                sharex = True, 
                                figsize = plot_tools.cm2inch( *fig_size_cm ),
                                )
        
        ## COMPUTING SAMPLING TIME OF OCCURENCES
        sampling_time_df = compute_sampling_time_frac_occur(sampling_time_occur_dict,
                                                            frame_rate = 500,
                                                            solvent = solvent,
                                                            )
        

        
        ## CREATING SAMPLING TIME FIGURE
        fig, ax = plot_sampling_time_df(sampling_time_df,
                                        grid_index = 0,
                                        x_scalar = frame2ns,
                                        x_key = "last",
                                        y_key = "frac_occur",
                                        delta = delta,
                                        fig = fig,
                                        ax = axs[0],
                                        convergence_type = convergence_type,
                                        fig_size_cm = FIGURE_SIZE)
        
        ## GETTING LIMITS
        if solvent in solvent_yaxis_lim:
            ax.set_ylim(solvent_yaxis_lim[solvent]['ylim'])
            ax.set_yticks(solvent_yaxis_lim[solvent]['yticks'])
        
        
        ## COMPUTING MIN SAMPLING TIME
        stored_min_sampling = compute_min_sampling_time(sampling_time_df,
                                                        delta = delta,
                                                        convergence_type = convergence_type,
                                                        x_key = 'last',
                                                        y_key = 'frac_occur',
                                                        )
        
        ## PLOTTING
        fig, ax = plot_histogram_sampling_time(stored_min_sampling,
                                               total_time_ps = 5000,
                                               frame_rate_ps = frame_rate,
                                               convert_style = frame2ns,
                                               plot_kwargs = {'color': 'k',
                                                              'density': True,
                                                              'linewidth': 1,
                                                              'edgecolor': 'k',
                                                              'align': 'left'},
                                                fig = fig,
                                                ax = axs[1]
                                     )
        
        ## SETTING AXIS
        ax.set_ylim([0, 0.18])
        ax.set_yticks(np.arange(0,0.24,0.04))
        
        ## SHOWINGY LABELS
        [ax.xaxis.set_tick_params(labelbottom=True) for ax in axs]
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        ## DEFINING FIGURE NAME
        figure_name = fig_prefix + solvent
        
        ## SETTING AXIS
        plot_tools.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'svg', 
                     save_fig=True,)
    
    
    #%% FIGURE S9 - COSOLVENT MAP OF UNSATURATED SAM
    
    ## DEFINING PARENT DIRECTORIES
    np_parent_dirs =[
             "20200625-GNP_cosolvent_mapping_sims_unsat_branched",
            ]

    ## DEFINING MAIN DIRECTORY
    main_dir = NP_SIM_PATH

    ## FINDING ALL PATHS
    full_path_dataframe = get_all_paths(np_parent_dirs = np_parent_dirs,
                                        main_dir = main_dir)
    
    ## DEFINING LIGAND OF INTEREST
    desired_lig = 'C11branch6OH'
    # 'C11OH'
    
    ## GETTING PATH
    path_sim_list = full_path_dataframe.loc[full_path_dataframe.ligand == desired_lig]['path'].to_list()
    
    #%% GENERATING FRACTION OF OCCURENCES
    
    ## SORTING PATH
    path_sim_list.sort()
    
    ## DEFINING INPUTS
    inputs_frac_occur={'path_sim_list' : path_sim_list}
    
    ## EXTRACTING
    storage_frac_occurences, mapping_list = load_frac_of_occur(**inputs_frac_occur)
    
    #%% SUMMING ALL FRACTION OF OCCURENCES
    
    ## DEFINING INPUTS
    inputs_storage={"storage_frac_occurences": storage_frac_occurences}

    ## DEFINING PICKLE PATH
    pickle_path = os.path.join(path_sim_list[0],
                               ANALYSIS_FOLDER,
                               main_compute_np_cosolvent_mapping.__name__,
                               "stored_frac_occur.pickle"
                               )
    
    ## EXTRACTION PROTOCOL WITH SAVING
    fraction_sim_occur_dict = save_and_load_pickle(function = sum_frac_of_occurences, 
                                                                  inputs = inputs_storage, 
                                                                  pickle_path = pickle_path,
                                                                  rewrite = False,
                                                                  verbose = True)
    
    #%% CORRELATING FRACTION OF OCCUPANCY TO MU VALUES
    
    #### LOADING MU VALUES
    
    ## DEFINING PARENT WC FOLDER
    parent_wc_folder = "20200618-Most_likely_GNP_water_sims_FINAL"
    
    ## FINDING ORIGINAL HYDROPHOBICITY NAME
    orig_name = find_original_np_hydrophobicity_name(path_sim_list[0])
    
    ## LOADING WC GRID
    relative_path_to_pickle=os.path.join("26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",
                                          "mu.pickle",
                                          )
    
    
    path_to_mu = os.path.join(PARENT_SIM_PATH,
                              parent_wc_folder,
                              orig_name,
                              relative_path_to_pickle)
    
    ## LOADING RESULTS
    mu_array = load_pickle_results(file_path = path_to_mu,
                                   verbose = True)[0][0]
    
    
    #%% CORRELATING MU VALUES
    
    ## CREATING NEW COLOR MAP
    tmap = plot_tools.create_cmap_with_white_zero(cmap = plt.cm.jet,
                                                  n = 100,
                                                  perc_zero = 0.125)

    ## GENERATING PLOT
    figure_size = (17.1, 6)
    figsize = plot_tools.cm2inch(figure_size)
    # figsize = (figsize[0]*2, figsize[1])

    ## ADDING FIGURE
    fig, ax = plot_gaussian_kde_scatter(fraction_sim_occur_dict = fraction_sim_occur_dict,
                                        mu_array = mu_array,
                                        figsize = figsize,
                                        solvent_list = [ 'PRO', 'HOH' ],
                                        y_range = (-0.2, 1.2),
                                        x_range = (8, 13),
                                        nbins = 300, # 100
                                        vmin = 0,
                                        vmax = 2,
                                        cmap = tmap, # plt.cm.Greens,
                                        )
    
    #%%
    
    ## FIGURE NAME
    figure_name = "SI_9B_cosolvent_map"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'pdf', # 'pdf',
                 save_fig = True,
                 dpi=300, # lowering resolution
                )
    
    #%% COMPARING tO INDUS SPHERICAL
    
    
    ## DEFINING LIGANDS
    ligands = ["dodecanethiol",
               "C11OH",
                ]
    
    ## DEFINING GRID DETAILS
    wc_analysis=DEFAULT_WC_ANALYSIS
    mu_pickle = MU_PICKLE

    
    ## DEFINING LOCATION
    path_list = {

                'Planar_SAMs': {
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ligands,
                                        },

                 }

    # #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = ligands,
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = True,
                                                       want_grid = True,
                                                       )
    
    
    #%%
    
    ## DEFINIG DICT
    dict_for_points = {
            'dodecanethiol': {
                    'top': (2.163, 2.598, 14.026),
                    'bot': (2.163, 2.598, 9.692),
                    },
            'C11OH': {
                    'top': (2.163, 2.298, 13.719),
                    'bot': (2.163, 2.298, 9.657),
                    },
            
            }
    
    ## defining current dict
    current_specific_dict = storage_dict['Planar_SAMs']
    
    ## LOOPING
    for each_key in dict_for_points:
        ## GETTING DICT
        current_dict = current_specific_dict[each_key]
        
        ## GETTING GRID
        grid = current_dict['grid']
        mu_list = current_dict['mu']
        
        ## LOOPING
        for top_bot in dict_for_points[each_key]:
            ## GETTING LOCATION
            location = dict_for_points[each_key][top_bot]
            
            ## GETTING CLOSEST
            closest_distances = np.linalg.norm(grid - location, axis = 1)
            closest_dis_idx = np.argmin(closest_distances)
            
            ## GETTING MIN
            closest_min_value = closest_distances[closest_dis_idx]
            
            print("Index: %d, min dist: %.3f"%(closest_dis_idx,closest_min_value ))
            ## gETTING MU
            mu_value = mu_list[closest_dis_idx]
            print("Mu value for %s, %s: %.3f"%(each_key,top_bot, mu_value ))
            
            
    
    
    
    
    
    
    #%% -- TRASH BELOW 
    
    
    #%% LARGE SAM DISTRIBUTION -- OH
    
    ## DEFINING LOCATION
    path_list = {'2nm GNP': {
                    'path_dict_key': 'GNP',
                    'prefix_key': 'GNP',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.1],
                    'ligands': [
                            "C11OH",                       
                            ],
                    'color': 'blue',
                        },
                '6nm GNP': {
                                    'path_dict_key': 'GNP_6nm',
                                    'prefix_key': 'GNP_6nm',
                                    'ligands': [
                                            "C11OH",                       
                                            ],
                                    'color': 'red',
                        
                        },
                'Planar SAMs': {    'path_dict_key': 'Planar_SAMs',
                                    'prefix_key': 'Planar',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.0],
                                    'ligands': [
                                            "C11OH"
                                            ],
                                    'color': 'black',
                                        },

                }
    
    #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = [],
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       )
    
    #%% LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
        
    ### FUNCTION PLOT MU DISTRIBUTION ACROSS GROUPS
    
    
    ## PLOTTING
    figsize=(11,10)
    # FIGURE_SIZE
    # (9, 7) # in cm
    figsize=plot_tools.cm2inch(*figsize)
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 2},
                                        figsize = FIGURE_SIZE,
                                    xlim = [6, 18],
                                    xticks = np.arange(7, 18, 2),
                                    want_legend_all = True,
                                    want_combined_plot = True)
    
    
    
    ## FIGURE NAME
    figure_name = "Comparison_btn_sizes_OH"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png',
                 save_fig = True,
                )
    
    
    
    #%% LEAST LIKELY DISTRIBUTIONS
    
    ## DEFINING LOCATION
    path_list = {'Most likely GNPs': {
                    'path_dict_key': 'GNP',
                    'prefix_key': 'GNP',
                    'yticks': np.arange(0, 1.2, 0.25),
                    'ylim': [-0.1, 1.2],
                    'ligands': [
                            "dodecanethiol",
                            "C11OH",
                            "C11NH2",
                            "C11COOH",
                            "C11CONH2",
                            "C11CF3",   
                            ],
                        },
                'Least likely GNPs': {    
                                    'path_dict_key': 'GNP_least_likely',
                                    'prefix_key': 'GNP_least_likely',
                                    'yticks': np.arange(0, 1.2, 0.25),
                                    'ylim': [-0.1, 1.2],
                                    'ligands': [
                                        "dodecanethiol",
                                        "C11OH",
                                        "C11NH2",
                                        "C11COOH",
                                        "C11CONH2",
                                        "C11CF3",   
                                            ],
                                        },
                        }
    #%% LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = [],
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       )
    
    
    #%% LOADING WATER MU VALUES
    water_mu_value = get_mu_value_for_bulk_water(water_sim_dict = PURE_WATER_SIM_DICT,
                                                 sim_path = MAIN_HYDRO_DIR)
        
    
    #%% PLOTTING DISTRIBUTION
    
    ## PLOTTING THE HISTOGRAMS
    figsize=(9,7) # in cm
    figsize=(11,11) # in cm
    figsize=plot_tools.cm2inch(*figsize)

    
    ## DEFINING X LIM
    xlim = [7, 15]
    xticks = np.arange(7, 15, 2)
    # np.arange(5, 20, 2)

    ## PLOTTING
    fig, axs = plot_mu_distribution(storage_dict = storage_dict,
                                    path_list = path_list,
                                    water_mu_value = water_mu_value,
                                     line_dict={'linestyle': '-',
                                               'linewidth': 2},
                                        figsize = figsize,
                                        xlim = xlim,
                                        xticks = xticks,
                                        want_legend_all = True)
    
    ## FIGURE NAME
    figure_name = "most_vs_least_likely"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png',
                 save_fig = True,
                 )
    #%% PLOTTING BAR PLOT OF MEDIAN DISTRIBUTIONS
    
    
    ## CREATING DATAFRAME
    df = extract_stats_values(storage_dict = storage_dict,
                              stat_key = "median")
    
    #%%
    
    ## CREATING PLOT
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## DEFINING WIDTH
    width=0.2
    
    n = len(storage_dict)
    n_array = np.arange(n)

    ## RE-ARRANGING LIGANDS BASED ON MU VALUES
    df = df.sort_values(by = 'stat_key')
    
    ## DEFINING LIGANDS
    ligands = pd.unique(df.ligand)    

    ## DEFINING 
    x = np.arange(len(ligands))
    
    ## DEFINING COLORS
    colors = ['black', 'red', 'blue']
    
    ## STORING MEAN AND STD
    statistics_storage = {}
    
    ## GENERATING     
    for idx, each_type in enumerate(pd.unique(df.type)):
        df_type = df.loc[df.type==each_type]
        median_values = [ float(df_type[df_type.ligand == each_lig]['stat_key'] ) for each_lig in ligands]
        
        statistics_storage[each_type] = median_values
        
        ## PLOTTING
        ax.bar(x + width * (idx-.5) ,
               median_values, 
               width=width, 
               color=colors[idx], 
               align='center',
               label=  each_type)
    
    ## DRAWING WATER LINE
    ax.axhline(y = water_mu_value[0], linestyle='--', color='k', label='Pure water', linewidth = 1.5)
    ## ADDING LEGEND
    ax.legend(loc='upper left')
    
    ## SETTING Y
    ax.set_ylim([8, 12])
    ax.set_yticks( np.arange(8, 13, 1) )
    
    ## ADDING LABELS
    ax.set_ylabel("Median $\mu$")
    
    ## RELABELED
    relabeled_ligs = [ RELABEL_DICT[each_lig] for each_lig in ligands ]
    
    ## SETTING X LABELS
    ax.set_xticks(x)
    ax.set_xticklabels( relabeled_ligs )
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## FIGURE NAME
    figure_name = "comparison_mostlikely_vs_leastlikely_median"

    ## SETTING AXIS
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png',
                 save_fig = True,
                 )
    
    
    
    #%%
    
    #### PLOTTING SPRING CONSTANTS
    ## DEFINING SIMULATION DIRECTORY
    # simulation_dir=r"20200212-Debugging_GNP_spring_constants_heavy_atoms"
    simulation_dir="20200217-planar_SAM_frozen_debug"
    # "20200212-Debugging_GNP"
    # "20200217-planar_SAM_frozen_debug"
    
    ## DEFINING SPECIFIC DIRECTORY
    specific_dir = r"MostlikelynpNVTspr_2000-EAM_300.00_K_2_nmDIAM_C11NH3_CHARMM36jul2017_Trial_1_likelyindex_1"

    ## DEFINING PATH TO LIST
    path_to_sim_list = os.path.join(MAIN_NP_DIR, simulation_dir)    
    
    
    #%%
    ## GETTING TRAJECTORY OUTPUT
    traj_output = extract_multiple_traj(path_to_sim_list =path_to_sim_list )
    
    ## GETTING DECODED NAME LIST
    traj_output.decode_all_sims(decode_type='nanoparticle')
    
    ## PLOTTING
    fig, ax = plot_rmse_vs_spring_constant(traj_output)

    ## DEFINING FIGURE NAME
    figure_name = simulation_dir + '-' +"RMSE_vs_spring_constant"    

    ## STORING FIGURE
    plot_tools.save_fig_png(fig = fig,
                            label = os.path.join(IMAGE_LOC,
                                              figure_name),
                             save_fig = SAVE_FIG)


    #%% WC INTERFACE

    
    ## DEFINING SIMULATION DIRECTORY
    simulation_dir="20200217-planar_SAM_frozen_debug"
    # "20200212-Debugging_GNP"
    # 
    # r"20200212-Debugging_GNP"
    
    ## DEFINING SPECIFIC DIRECTORY
    specific_dir = r"MostlikelynpNVTspr_2000-EAM_300.00_K_2_nmDIAM_C11NH3_CHARMM36jul2017_Trial_1_likelyindex_1"

    ## DEFINING PATH TO LIST
    path_to_sim_list = os.path.join(MAIN_NP_DIR, simulation_dir)    
    
    ## GETTING TRAJECTORY OUTPUT
    traj_output = extract_multiple_traj(path_to_sim_list =path_to_sim_list )
    
    ## GETTING DECODED NAME LIST
    traj_output.decode_all_sims(decode_type='nanoparticle')
    
    ## COMPUTING AVERAGE DEVIATION
    traj_output = compute_avg_deviation_across_traj(traj_output = traj_output,
                                                    ref_grids = None)
    
    
    
    #%%
    
    
    
    ## STORING VECTOR
    storage_list = []
    
    ### LOOPING THROUGH EACH SIM
    for idx, each_sim in enumerate( traj_output.full_sim_list ):
        print("Working on %s"%(each_sim))
        results = traj_output.load_results(idx = idx,
                                           func = main_compute_wc_grid_multiple_subset_frames)
        ## COMPUTING DEVIATION
        avg_deviation = compute_avg_deviation(results = results)
        ## STORING
        storage_list.append(avg_deviation)
    
    ## STORING
    traj_output.decoded_name_df['avg_deviation'] = storage_list[:]
    
    #%%
    

    
    ## PLOTTING AVERAGE DEVIATION VERSUS SPRING CONSTANT
    fig, ax = plot_avg_deviation_vs_spring_constant(traj_output)
    
    ## DEFINING FIGURE NAME
    figure_name = simulation_dir + '-' + "Avg_deviation_vs_spring_constant"    

    ## STORING FIGURE
    plot_tools.save_fig_png(fig = fig,
                            label = os.path.join(IMAGE_LOC,
                                              figure_name),
                             save_fig = SAVE_FIG)


    
    #%% LOADING PLANAR SIM FROZEN
    

    ## DEFINING PLANAR SIMS THAT ARE FROZEN
    frozen_planar_sim_dict = {
            'main_sim_folder': '20200114-NP_HYDRO_FROZEN',
            'ligand_folders':
                {
                        'dodecanethiol': r"FrozenPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps",
                        'C11OH': r"FrozenPlanar_300.00_K_C11OH_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps",
                        },         
            }
    
    ## UNFROZEN
    planar_sim_dict = {
            'main_sim_folder': 'PLANAR',
            'ligand_folders':
                {
                        'dodecanethiol': r"FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps",
                        'C11OH': r"FrozenGoldPlanar_300.00_K_C11OH_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps",
                        },         
            }
    
    ##  DEFINING RELATIVE PAATH TO GRID
    relative_path_to_grid = os.path.join(r"25.6-0.24-0.1,0.1,0.1-0.33-all_heavy",
                                         r"grid-0_1000",
                                         r"out_willard_chandler.dat")
    
    ## LOADING FOR EACH
    ligands = ['dodecanethiol', 'C11OH']
    
    ## GENERATING EMPTY DICTIONARY
    ref_grids = {}
    
    for each_ligand in ligands:
        ## DEFINING PATH
        path_to_dat = os.path.join(MAIN_HYDRO_DIR,
                                   planar_sim_dict['main_sim_folder'],
                                   planar_sim_dict['ligand_folders'][each_ligand],
                                   relative_path_to_grid)
        
        ## LOADING THE GRID
        grid_results = load_datafile(path_to_dat)
        ## STORING
        ref_grids[each_ligand] =  grid_results[:]

    
    #%% GETTING DEVIATIONS FROM A REFERENCE
    
    ## GETTING TRAJECTORY OUTPUT
    traj_output = extract_multiple_traj(path_to_sim_list =path_to_sim_list )
    
    ## GETTING DECODED NAME LIST
    traj_output.decode_all_sims(decode_type='nanoparticle')
    
    ## COMPUTING AVERAGE DEVIATION
    traj_output = compute_avg_deviation_across_traj(traj_output = traj_output,
                                                    ref_grids = ref_grids)
    
    #%% PLOTTING THE DIFFERENCE
    ## PLOTTING AVERAGE DEVIATION VERSUS SPRING CONSTANT
    fig, ax = plot_avg_deviation_vs_spring_constant(traj_output)
    
    ## DEFINING FIGURE NAME
    figure_name = simulation_dir + '-' + "Avg_deviation_vs_spring_constant_relative_to_unfrozen"    

    ## STORING FIGURE
    plot_tools.save_fig_png(fig = fig,
                            label = os.path.join(IMAGE_LOC,
                                              figure_name),
                             save_fig = SAVE_FIG)

    #%%
    
    ## GETTING TRAJECTORY OUTPUT
    traj_output = extract_multiple_traj(path_to_sim_list =path_to_sim_list )
    
    ## GETTING DECODED NAME LIST
    traj_output.decode_all_sims(decode_type='nanoparticle')
    
    ## COMPUTING AVERAGE DEVIATION
    traj_output = compute_avg_deviation_across_traj(traj_output = traj_output,
                                                    ref_grids = ref_grids,
                                                    compare_type = "z_dist")

    # #%% PLOTTING THE DIFFERENCE
    ## PLOTTING AVERAGE DEVIATION VERSUS SPRING CONSTANT
    fig, ax = plot_avg_deviation_vs_spring_constant(traj_output)
    
    ## DEFINING FIGURE NAME
    figure_name = simulation_dir + '-' + "Avg_z_deviation_vs_spring_constant_relative_to_unfrozen"    

    ## STORING FIGURE
    plot_tools.save_fig_png(fig = fig,
                            label = os.path.join(IMAGE_LOC,
                                              figure_name),
                             save_fig = SAVE_FIG)
    #%% ARCHIVE
    
    
    
    ## DEFINING STATS RELABLE
    STATS_RELABEL={
            'mean': 'Mean',
            'var': 'Variance',
            'skew': 'Skewness',
            'kurt': 'Kurtosis',
            'mode': 'Mode',
            'median': 'Median',
            
            
            'moment_1': 'Moment 1',
            'moment_2': 'Moment 2',
            'moment_3': 'Moment 3',
            'moment_4': 'Moment 4',
            
            'uncentered_moment_1': 'Non-center Moment 1',
            'uncentered_moment_2': 'Non-center Moment 2',
            'uncentered_moment_3': 'Non-center Moment 3',
            'uncentered_moment_4': 'Non-center Moment 4',
            
            'normalized_uncentered_moment_1': 'Norm. Non-center Moment 1',
            'normalized_uncentered_moment_2': 'Norm. Non-center Moment 2',
            'normalized_uncentered_moment_3': 'Norm. Non-center Moment 3',
            'normalized_uncentered_moment_4': 'Norm. Non-center Moment 4',
            
            'ratio_uncentered_moment_1': 'Ratio. Non-center Moment 1',
            'ratio_uncentered_moment_2': 'Ratio. Non-center Moment 2',
            'ratio_uncentered_moment_3': 'Ratio. Non-center Moment 3',
            'ratio_uncentered_moment_4': 'Ratio. Non-center Moment 4',
            }
    
    ## DEFINING STATS KEYS
    stats_keys = [
                    # 'mean', 'var', 'skew', 'kurt',
                    # 'moment_1', 'moment_2', 'moment_3', 'moment_4',
                  # 'uncentered_moment_1',
                  
                  'ratio_uncentered_moment_1',
                  'ratio_uncentered_moment_2',
                  'ratio_uncentered_moment_3',
                  'ratio_uncentered_moment_4'
                  
                  # 'median',
#                  'normalized_uncentered_moment_1',
#                  'normalized_uncentered_moment_2',
#                  'normalized_uncentered_moment_3',
#                  'normalized_uncentered_moment_4',
                  ]
                  # 'uncentered_moment_2', 'uncentered_moment_3', 'uncentered_moment_4',]
    
    ## LOOPING THROUGH ALL POSSIBLE TYPES
    for specific_type in storage_dict:
        
        ## LOOPING THROUGH ALL STATS
        for current_stats_key in stats_keys:
            
            ## CLOSING
            plt.close('all')
            
            ## PLOTTING 
            fig, ax = plot_bar_specific_stats(storage_dict = storage_dict,
                                        specific_type = specific_type,
                                        current_stats_key = current_stats_key)
            
            ## DEFINING FIGURE NAME
            figure_name = "_".join([specific_type, current_stats_key])
            
            ## SETTING AXIS
            plot_tools.store_figure(fig = fig, 
                         path = os.path.join(IMAGE_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)

