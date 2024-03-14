# -*- coding: utf-8 -*-
"""
analyze_mu_distribution.py
The purpose of this function is to analyze mu distributions outputted 
from hydration map analysis

Written by: Alex K. Chew (01/22/2020)

FUNCTIONS AND CLASSES:
    debug_mu_convergence:
        debugs mu convergence 

"""
## IMPORTING MODULES
import os
import mdtraj as md
import numpy as np


## POLYNOMIAL FUNCTION
import numpy.polynomial.polynomial as poly

## IMPORTING EXTRACTION TOOLS
from MDDescriptors.application.np_hydrophobicity.extract_hydration_maps import extract_name,get_all_paths, plot_histogram_for_prob_dist, plot_gaussian_fit_to_find_mu
## CHECK PATH
from MDDescriptors.core.check_tools import check_path, check_dir, check_spyder

## CALC FUNCTIONS
import MDDescriptors.core.calc_tools as calc_tools
from MDDescriptors.core.calc_tools import find_theoretical_error_bounds, get_converged_value_from_end
## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_tools

## PICKLE FUNCTIONS
from MDDescriptors.core.pickle_tools import load_pickle_results, pickle_results

## DEFINING DESIRED DICTIONARY
import matplotlib.pyplot as plt
    
import glob as glob
## IMPORTING PANDAS
import pandas as pd

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import MAX_N, MU_MIN, MU_MAX

## IMPORTING NUM DIST FUNCTION
from MDDescriptors.surface.generate_hydration_maps import compute_num_dist
## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import R_CUTOFF, MAX_N
## GETTTING FUNCTION TO GET P VALUES
from MDDescriptors.surface.core_functions import get_x_y_from_p_N, compute_mu_from_num_dist_for_one_grid_point, calc_mu
## LOADING DAT FILE FUNCTION
from MDDescriptors.surface.core_functions import load_datafile

## LOADING MD LIGANDS
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_atom_indices_of_ligands_in_traj

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools

plot_tools.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_tools.FIGURE_SIZES_DICT_CM['1_col_landscape']

## IMPORITNG GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, IMAGE_LOCATION

## DEFINING FIGURE NAME
STORE_FIG_LOC = IMAGE_LOCATION
STORE_FIG_LOC = check_path(STORE_FIG_LOC)
## SAVE FIG
SAVE_FIG = True

## DEFINING HISTOGRAM STEP SIZE
HISTOGRAM_STEP_SIZE=0.4

## DEFINING MAIN DIRECTORY
MAIN_SIM_FOLDER=r"S:\np_hydrophobicity_project\simulations"

## CHECKING PATH
MAIN_SIM_FOLDER = check_path(MAIN_SIM_FOLDER)

# COLORS: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
LIGAND_COLOR_DICT = {
        'dodecanethiol': 'black',
        'C11OH': 'red',
        'C11COOH': 'magenta',
        'C11NH2': 'blue',
        'C11CONH2': 'orange',
        'C11CF3': 'cyan',
        'C11COO': 'orange',
        'C11NH3': 'green',
        'ROT001': 'purple',
        'ROT002': 'olive',
        'ROT003': 'lightblue',
        'ROT004': 'goldenrod',
        'ROT005': 'magenta',
        'ROT006': 'maroon',
        'ROT007': 'slategrey',
        'ROT008': 'indigo',
        'ROT009': 'chocolate',
        'ROTPE1': 'greenyellow',
        'ROTPE2': 'powderblue',
        'ROTPE3': 'darksalmon',
        'ROTPE4': 'deeppink',
        'C11double67OH': 'gray',
        'dodecen-1-thiol':  'darkviolet',
        }
    
## DEFINING DIRECTORIES
DIRECTORY_DICT = {
        'Planar_sims': '20200114-NP_HYDRO_FROZEN',
        'NP_sims': 'NP_SPHERICAL_REDO',
        'Planar_sims_unfrozen': 'PLANAR',
        'GNP_SPR_600': '20200215-GNPspr600',
        'GNP_SPR_600_extended': '20200215-GNPspr600_extended',
        'GNP_SPR_50' : '20200224-GNP_spr_50',
        'Planar_SPR_600': '20200215-planar_SAM_frozen_600spring',
        'Planar_SPR_600_extended': '20200215-planar_SAM_frozen_600spring_extended',
        'Planar_SPR_50': '20200224-planar_SAM_spr50',
        'GNP_SPR_50_double_bonds': '20200227-50_doub',
        'GNP_SPR_50_nptequil': '20200401-Renewed_GNP_sims_with_equil',
        'Water_only': 'PURE_WATER_SIMS',
        'Planar_SPR_50_long_z': '20200326-Planar_SAMs_with_larger_z_frozen',
        'Planar_SPR_50_long_z_8nm': '20200326-Planar_SAMs_with_larger_z_8nm_frozen',
        'Planar_SPR_50_long_z_vacuum': '20200326-Planar_SAMs_with_larger_z_frozen_with_vacuum',
        'Planar_SPR_50_SHORTEQUIL': '20200328-Planar_SAMs_new_protocol-shorterequil_spr50',
        'Planar_SPR_50_notempanneal': '20200331-Planar_SAMs_no_temp_anneal_or_vacuum_withsprconstant',
        'Planar_SPR_50_equil_with_vacuum_5nm': '20200403-Planar_SAMs-5nmbox_vac_with_npt_equil'
        }

### FUNCTION TO EXTRACT FILE INFORMATION
def extract_name(name):
    '''
    The purpose of this function is to extract the most likely name.
    INPUTS:
        name: [str]
            name of your interest, e.g.
                MostlikelynpNVT_EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1_likelyindex_1
    OUTPUTS:
        name_dict: [dict]
            dictionary of the name details
    '''
    ## SPLITITNG
    name_split = name.split('_')
    # RETURNS:
    #   ['MostlikelynpNVT', 'EAM', '300.00', 'K', '2', 'nmDIAM', 'C11CONH2', 'CHARMM36jul2017', 'Trial', '1', 'likelyindex', '1']
    if name_split[0] != 'FrozenPlanar' and \
       name_split[0] != 'FrozenGoldPlanar' and \
       name_split[0] != 'NVTspr':
         
       ## SPHERICAL
       if name_split[0] == 'MostlikelynpNVT': 
           current_type = 'spherical_frozen'
       elif name_split[0] == "MostlikelynpNVTspr":
           current_type = 'gnp_spr_' + name_split[1].split('-')[0]
       else:
            print("Error for name split:")
            print(name_split)   
           
       if current_type != 'spherical_frozen':
            name_dict = {
                    'type': current_type,
                    'temp': float(name_split[2]),
                    'diam': float(name_split[4]),
                    'ligand': name_split[6],
                    'index' : name_split[-1],
                    }
       else:
            name_dict = {
                    'type': current_type,
                    'temp': float(name_split[2]),
                    'diam': float(name_split[4]),
                    'ligand': name_split[6],
                    'index' : name_split[-1],
                    }
    else:
        ## DEFINING TYPE
        if name_split[0] == 'FrozenPlanar':
            current_type = 'planar_frozen'
        elif name_split[0] == 'FrozenGoldPlanar':
            current_type = 'planar_unfrozen'
        elif name_split[0] + '_' + name_split[1] == 'NVTspr_600':
            current_type = 'planar_spr_600'
        elif name_split[0] + '_' + name_split[1] == 'NVTspr_50':
            current_type = 'planar_spr_50'
        else:
            print("Error for name split:")
            print(name_split)   
        if current_type != 'planar_spr_50' and current_type != 'planar_spr_600':
            name_dict = {
                    'type': current_type,
                    'temp': float(name_split[1]),
                    'diam': float(name_split[4].split('x')[0]),
                    'ligand': name_split[3],
                    'index' : '1'
                    }
        else:
            name_dict = {
                    'type': current_type,
                    'temp': float(name_split[3]),
                    'diam': float(name_split[6].split('x')[0]),
                    'ligand': name_split[5],
                    'index' : '1'
                    }
        
    return name_dict


###################################################
### CLASS FUNCTION TO KEEP TRACK OF DIRECTORIES ###
###################################################
class track_directories:
    '''
    The purpose of this function is to keep track of directories. 
    INPUTS:
        main_path_list: [list]
            main path list
        verbose: [logical]
            True/False if you want to verbosely print detaeils
    OUTPUTS:
        self.path_list: [list]
            list of all the paths
        self.file_basename: [list]
            list of all the file basenames
        self.extracted_basename: [list]
            list of extracted base names in directionary form
        self.df: [pd.DataFrame]
            Dataframe of extracted files
    '''
    def __init__(self,
                 main_path_list,
                 verbose = True):
        ## STORING
        self.verbose = verbose
        
        ## LOOKING THROUGH DIRECTORIES
        files_in_dir =[ glob.glob(os.path.join(each_path, "*")) for each_path in main_path_list ]
        ## GETTING FILE LIST
        self.path_list = calc_tools.flatten_list_of_list(files_in_dir)
        
        ## GETTING BASENAME
        self.file_basename = [ os.path.basename(each_file) for each_file in self.path_list ]
        
        ## RUNNING EXTRACTION TOOLS
        self.extracted_basename = [ extract_name(each_file) for each_file in self.file_basename ]
        
        ## GENERATING DATAFRAME
        self.df = pd.DataFrame(self.extracted_basename)
        
        ## ADDING THE PATH TO THE DATAFRAME
        self.df['PATH'] = self.path_list[:]
        
        ## PRINTING
        if self.verbose is True:
            print("Total main directories given: %d"%(len(main_path_list)) )
            print("Total directories found: %d"%(len(self.path_list)) )
        
        return
    
    ## FUNCTION TO GET NEW DATAFRAME BASED ON INPUTS
    def get_desired_sims(self,
                         desired_sims):
        '''
        The purpose of this function is to get the desired sims based on inputs. 
        This code loops through your given dictionary and finds all matching 
        constraints.
        INPUTS:
            self: [obj]
                class property
            desired_sims: [dict]
                dictionary with a list for each name of the desired sim, e.g.
                    desired_sims = {
                            'type'   : [ 'planar' ],
                            }
        OUTPUTS:
            rslt_df: [dataframe]
                output dataframe that matches your constraints
        '''
        ## LOCATING SPECIFIC DIRECTORIES WITHIN DATAFRAMES
        ## DEFINING NEW DATAFRAME
        rslt_df = self.df.copy()
        for each_key in desired_sims:
            rslt_df = rslt_df[rslt_df[each_key].isin(desired_sims[each_key])]
        ## PRINTING
        if self.verbose is True:
            print("Total desired simulations: %d"%( len(rslt_df) ) )
        return rslt_df

### FUNCTION TO GET ALL SIMULATION DIRECTORIES
def get_all_paths( main_path, desired_folders, folder_dict ):
    '''
    The purpose of this function is to get all paths given a main path, 
    desired folders, and a dictionary. The desired_folders variable should 
    be a list of keys used in folder_dict.
    
    INPUTS:
        main_path: [str]
            main path to simulation folder
        desired_folders: [list]
            list of desired folders to look into
        folder_dict: [dict]
            dictionary of folder keys
    OUTPUTS:
        main_path_list: [list]
            list of main path files
    '''
    ## MAKING EMPTY LIST
    main_path_list = []
    
    ## LOOPING THROUGH EACH DESIRED FOLDER
    for each_folder in desired_folders:
        ## ADDING EACH FILE
        main_path_list.append( os.path.join(main_path, folder_dict[each_folder]) )
    return main_path_list

################################################
### CLASS FUNCTION TO EXTRACT HYDRATION MAPS ###
################################################
class extract_hydration_maps:
    '''
    The purpose of this function is to extract hydration maps 
    INPUTS:
        
    OUTPUTS:
        
    '''
    def __init__(self):
        
        return
    
    ## FUNCTION TO LOAD PICKLE
    @staticmethod
    def load_pickle(path_to_pickle,
                    pickle_name):
        '''
        The purpose of this function is to load the pickle
        INPUTS:
            path_to_pickle: [str]
                path to pickle
            pickle_name: [str]
                name of the pickle
        OUTPUTS:
            results: [obj]
                output of the pickle
        '''
    
        ## DEFINING FULL PATH
        full_pickle_path = os.path.join(path_to_pickle,
                                        pickle_name)
        ## LOADING MU
        results = load_pickle_results(full_pickle_path)[0][0]
        
        return results
    
    ## FUNCTION TO LOAD ALL MU VALUES
    @staticmethod
    def load_mu_values(main_sim_list,
                       pickle_name = "mu.pickle",
                       verbose = True):
        '''
        The purpose of this function is to load all mu values
        INPUTS:
            main_sim_list: [list]
                full path to the mu value folder
            pickle_name: [str]
                mu pickle file name
        OUTPUTS:
            mu_list: [list]
                mu list
        '''
        ## CREATING LIST
        mu_list = []
        
        ## LOOPING THROUGH EACH SIMULATION
        for each_sim in main_sim_list:
            if verbose is True:
                print("Loading mu from: %s"%(each_sim) )
            ## DEFINING FULL PATH
            path_to_mu = os.path.join(each_sim,
                                      pickle_name)
            ## LOADING MU
            mu_values = load_pickle_results(path_to_mu)[0][0]
            
            ## APPENDING
            mu_list.append(mu_values)
            
        return mu_list
    
    ### FUNCTION TO LOAD NEIGHBORS PICKLE
    def load_neighbor_values(self,
                             main_sim_list,
                             pickle_name):
        '''
        The purpose of this function is to load neighbor values
        INPUTS:
            main_sim_list: [list]
                full path to the mu value folder
            pickle_name: [str]
                relative pickle file name
        OUTPUTS:
            neighbor_list: [list]
                list of neighbor arrays
        '''
        ## LOADING NEIGHBOR LIST
        neighbor_list = self.load_mu_values(main_sim_list = main_sim_list,
                                            pickle_name = pickle_name)
        return neighbor_list

### FUNCTION TO GET HISTOGRAM INFORMATION FOR MU LIST
def compute_histogram_for_mu( mu_list,
                              histogram_range,
                              step_size = 0.2,
                              ):
    '''
    The purpose of this function is to compute histogram for mu values. 
    INPUTS:
        mu_list: [list]
            list of mu arrays
        histogram_range: [list, shape = 2]
            minimima and maxima of histogram range
        step_size: [float, default = 0.2]
            step size for the histogram
    OUTPUTS:
        histogram_data: [list]
            list of histogram data for each mu list
        xs: [np.array]
            x values for the histogram data        
    '''

    ## DEFINING X VALUES
    xs = np.arange( histogram_range[0], histogram_range[1], step_size ) + 0.5 * step_size
    
    ## STORING HISTOGRAM
    histogram_data = []
    
    ## LOOPING THROUGH MU LIST
    for mu in mu_list:
        ## DEFINING HISTOGRAM
        ps = np.histogram( mu, bins = xs.size, 
                          range = ( histogram_range[0], histogram_range[1] ),
                          density = True )[0]
        ## STORING HISTOGRAM
        histogram_data.append(ps)
    return histogram_data, xs

### FUNCTION TO PLOT THE HISTOGRAM DATA
def plot_histogram_data(histogram_data,
                        xs,
                        legend_list = [],
                        color_list = [],
                        linestyle = '-',
                        fig = None,
                        ax = None,
                        xlabel="$\mu$",
                        ylabel="$P(\mu)$",
                        ):
    '''
    The purpose of this function is to plot the histogram data.
    INPUTS:
        histogram_data: [list]
            list of histogram data
        xs: [np.array]
            x values for the histogram
        color_list: [list]
            list of colors
        legend_list: [list]
            list of legend names
        fig: [obj]
            figure object
        ax: [obj]
            axis object
    OUTPUTS:
        fig, ax: [obj]
            figure and axis objects
    '''
    ## CREATING PLOT
    if fig is None or ax is None:
        fig, ax = plot_tools.create_plot()
        ## ADDING LABEL
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    ## LOOPING THROUGH HISTOGRAM DATA
    for idx, hist in enumerate(histogram_data):
        if len(color_list) == 0:
            color = None
        else:
            color = color_list[idx]
            
        if len(legend_list)==0:
            label = None
        else:
            label = legend_list[idx]
            
        ## PLOTTING
        ax.plot(xs, hist, 
                linestyle = linestyle, 
                linewidth=2, 
                color = color, 
                label = label)
        
    return fig, ax

### FUNCTION TO GET HISTOGRAM DATA FOR EACH
def extract_histogram_multiple_ligands(main_path,
                                       folder_dict,
                                       shape_types,
                                       desired_folders,
                                       ligands,
                                       relative_path,
                                       mu_pickle_name,
                                       histogram_range = [5, 30],
                                       step_size = 0.2,
                                       ):
    '''
    The purpose of this function is to extract histogram data from multiple 
    ligands.
    INPUTS:
        shape_types: [list]
            shapetype desired from extraction names, e.g.
            'planar_frozen', 'spherical', or 'planar_unfrozen'
        main_path: [str]
            path to main simulation folder
        folder_dict: [dict]
            dictionary of the folders within the main simulation folder
        desired_folders: [list]
            list of desired folders
        mu_pickle_name: [str]
            pickle name for mu
        relative_path: [str]
            relative path to mu distribution
        ligands: [list]
            list of ligands that you want to extract from
        histogram_range: [list]
            list of histogram ranges
        step_size: [float]
            step size
    OUTPUTS:
        fig, ax:
            figure and axis
    
    '''
    ## CREATING DICTIONARY
    histogram_data_storage = {}
    ## LOOPING THROGUH DESIRED TYPE
    for idx, each_type in enumerate(shape_types):
        ## LOOPING THROUGH DESIRED SIMS
        if type(ligands[0]) == list:
            current_ligs = ligands[idx]
        else:
            current_ligs = ligands
        desired_sims = {
                'type'   : [each_type], 
                'ligand' : current_ligs,
                }
        
        ## GETTING MAIN PATH LIST
        main_path_list = get_all_paths(main_path = MAIN_SIM_FOLDER, 
                                       desired_folders = desired_folders, 
                                       folder_dict = DIRECTORY_DICT )
        ## TRACTING DIRECTORIES
        directories = track_directories( main_path_list = main_path_list )
        
        ## GETTING DATAFRAMES THAT YOU DESIRE
        df = directories.get_desired_sims(desired_sims = desired_sims)
        
        if type(relative_path) is list:
            current_rel_path = relative_path[idx]
        
        ## GETTING PATH
        path_mu_list = [ os.path.join(each_path, current_rel_path) for each_path in list(df['PATH']) ]
        
        ## LOADING PICKLE
        hydration_map = extract_hydration_maps()
        mu_list = hydration_map.load_mu_values(main_sim_list = path_mu_list,
                                               pickle_name = mu_pickle_name)
        
        ## COMPUTING HISTOGRAM DATA
        histogram_data, xs = compute_histogram_for_mu( mu_list = mu_list,
                                                       histogram_range = histogram_range,
                                                       step_size =step_size,
                                                       )
        
        ## DEFINING LIGANDS
        ligand_list = list(df.ligand)
        
        ## DEFINING COLOR LIST
        color_list = [ LIGAND_COLOR_DICT[each_lig] for each_lig in ligand_list]
        
        ## STORING
        if each_type in histogram_data_storage.keys():
            each_type += "_%d"%(idx+1)
        histogram_data_storage[each_type]={
                'legend_list': ligand_list,
                'color_list': color_list,
                'histogram_data': histogram_data,
                'xs': xs,
                }
    
    # if len(shape_types) != len(histogram_data_storage):
        
        ## REDEFINING SHAPE TYPES
    shape_types= list(histogram_data_storage.keys())
    
    ### PLOTTING THE GNP AND PLANAR TOGETHER
    
    ## CREATING FIGURES
    fig, axs = plt.subplots(nrows=len(shape_types), sharex=True)
    
    ## FIXING FOR WHEN AXS IS ONE
    if len(shape_types)==1:
        axs = [axs]
    
    ## LOOPING THROUGH DESIRED TYPE
    for idx, each_type in enumerate(shape_types):
        ## PLOTTING HISTOGRAM
        fig, ax = plot_histogram_data(linestyle = '-',
                                      fig = fig,
                                      ax = axs[idx],
                                      **histogram_data_storage[each_type]
                                      )
        ## ADDING TITLE
        ax.text(.5,.9,each_type,
            horizontalalignment='center',
            transform=ax.transAxes)
    
        ## ADDING LEGEND
        ax.legend()
        ## ADDING RID
        ax.grid()
        
        ## SETTING Y LABEL
        ax.set_ylabel("$P(\mu)$")
    
    ## ADJUSTING SPACE OF SUB PLOTS
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ## ADDING AXIS LABELS
    axs[len(shape_types)-1].set_xlabel("$\mu$")

    return fig, axs, histogram_data_storage


### CLASS FUNCTION TO COMPUTE THE MU CONVERGENCE
class debug_mu_convergence:
    '''
    The purpose of this function is to test the convergence of mu. 
    INPUTS:
        neighbor_list: [np.array]
            neighbor list array
    OUTPUTS:
        
    FUNCTIONS:
        compute_mu_convergence_time:
            computes mu convergence over time
        compute_mu_for_various_frames:
            computes mu for various frames
        compute_mu_equil_time:
            computes equilibration time for mu
        compute_bounds_for_convergence_mu:
            computes bounds for the convergence
        plot_convergence_of_mu:
            plots convergence of mu for a single grid point
        plot_sampling_time_vs_grid_index:
            plots sampling time versus grid index
        main_compute_sampling_time_from_reverse:
            computes sampling time from the reverse
    USAGE EXAMPLE:
        ## GENERATING DEBUGGING MU CLASS
        mu_debug = debug_mu_convergence()
    
        ## COMPUTING CONVERGENCE TIME
        mu_storage_reverse, frame_list = mu_debug.compute_mu_convergence_time(num_neighbors_array = num_neighbors_array,
                                                                              frame_rate = frame_rate,
                                                                              want_reverse = True)
        
        ## GETTING CONVERGENCE INFORMATION
        theoretical_bounds, index_converged, sampling_time_x_converged, x \
                        = mu_debug.main_compute_sampling_time_from_reverse(mu_storage_reverse = mu_storage_reverse,
                                                                           frame_list = frame_list,
                                                                           percent_error = percent_error,
                                                                           )
        ## PLOTTING CONVERGENCE
        fig, ax = mu_debug.plot_convergence_of_mu( mu_storage_reverse = mu_storage_reverse,  
                                                   x = x,
                                                   grid_index = 0,
                                                   theoretical_bounds = theoretical_bounds,
                                                   index_converged = index_converged
                                                   )
        
     
        ## PLOTTING SAMPLING TIME VERSUS GRID
        fig, ax = mu_debug.plot_sampling_time_vs_grid_index(sampling_time_x_converged)
        
        ## GETTING MAXIMUM SAMPLING TIME
        max_sampling_time = np.max(sampling_time_x_converged)
    '''
    def __init__(self):
        
        return

    ### FUNCTION TO COMPUTE SAMPLING MU
    def compute_mu_convergence_time(self,
                                    num_neighbors_array,
                                    frame_rate,
                                    want_reverse = False,
                                    method_type = "new",):
        '''
        The purpose of this function is to compute mu across different sets of 
        time frames.
        INPUTS:
            num_neighbors_array: [np.array, shape=(num_points, frame)]
                number of neighbors array.
            frame_rate: [int]
                frame rate that you want to compute convergence for
            want_reverse: [logical]
                True if you want convergence from the reverse
        OUTPUTS:
            mu_storage: [np.array, shape = (num_split, num_grid points)]
                mu values for each grid value
            final_sampling_values: [np.array]
                final values for sampling
        '''
        ## GETTING NUMBER OF FRAMES
        num_frames = num_neighbors_array.shape[1]
        ## COPYING
        neighbor_array = num_neighbors_array[:]
        
        ## COMPUTING SAMPLING TIME
        final_sampling_values  = np.arange(frame_rate, num_frames, frame_rate)
        
        ## IF WANT REVERSE IS TRUE
        if want_reverse is True:
            print("Since reverse is turned on, sampling time will be computed from the end of the trajectory")
            frame_list = [ np.arange( num_frames - each_final - 1, num_frames ) for each_final in final_sampling_values]
            # neighbor_array = neighbor_array[:, ::-1]
        else:
            ## GETTING FRAME LIST
            frame_list = [ np.arange(0, each_final + 1)  for each_final in final_sampling_values]
        
        ## COMPUTING EQUILIBRATION TIME
        mu_storage = self.compute_mu_for_various_frames(neighbor_array = neighbor_array,
                                                        frame_list = frame_list,
                                                        method_type = method_type)
        
        return mu_storage, frame_list

    ### FUNCTION TO COMPUTE MU FOR VARIOUS FRAMES
    @staticmethod
    def compute_mu_for_various_frames(neighbor_array,
                                      frame_list,
                                      max_N = MAX_N,
                                      method_type="new"):
        '''
        The purpose of this function is to compute mu for varying frames.
        INPUTS:
            neighbor_array: [np.array]
                neighbors array
            frame_list: [list]
                list of frames you want to compute mu for
            max_N: [int]
                maximum number counted for the probability distribution
            method_type: [str]
                method for computing mu: either "new" or "old". The "old" method 
                uses translations of the probability distribution, which may be
                erroneous.
        OUTPUTS:
            mu_storage: [np.array =  (num_iter, num_grids)]
                mu storage per grid piont
        '''
        ## STORING MU VALUES
        mu_storage = []
        
        ## LOOPING THROUGH FRAMES
        for idx, frame_array in enumerate(frame_list):
            ## DEFINING FINAL FRAMES
            initial_frame = frame_array[0]
            final_frame = frame_array[-1]
            print("Computing mu from %d to %d"%( initial_frame, final_frame ) )
            
            ## COMPUTING NUMBER DIST
            unnorm_p_N = compute_num_dist(num_neighbors_array = neighbor_array[:, frame_array],
                                          max_neighbors = MAX_N)
            
            ## DEFINING OLD METHOD
            if method_type == "old":
                mu = calc_mu(p_N_matrix = unnorm_p_N,
                             d = MAX_N)
            ## DEFINING NEW METHOD
            elif method_type == "new":
                ## GETTING MU DIST
                mu_dist = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
                ## DEFINING MU
                mu = mu_dist.mu_storage['mu_value'].to_numpy()
            
            ## APPENDING
            mu_storage.append(mu)
        
        ## CONVERTING TO NUMPY ARRAY
        mu_storage = np.array(mu_storage)
        
        return mu_storage

    ### FUNCTION TO COMPUTE MU EQUIL TIME
    def compute_mu_equil_time(self,
                              num_neighbors_array,
                              frame_for_convergence,
                              inc,
                              want_reverse = False,
                              method_type = "new",
                              ):
        '''
        The purpose of this function is to compute mu equilibration time. 
        The idea is that we start from the beginning of the trajectory, then 
        compute mu for separate chunks, and finally output the mu value per 
        chunk. The equilibration time is when this chunk converges to a single 
        value. 
        INPUTS:
            num_neighbors_array: [np.array]
                number of neighbors array
            frame_for_convergence: [int]
                frames necessary for convergence
            inc: [int]
                increments to move the frame
        OUTPUTS:
            mu_storage: [np.array, shape = (num_split, num_grid points)]
                mu values for each grid value
            frame_list: [list]
                frame list used for mu storage
        '''
        ## GETTING NUMBER OF FRAMES
        num_frames = num_neighbors_array.shape[1]
        
        ## COPYING
        neighbor_array = num_neighbors_array.copy()
        ## IF WANT REVERSE IS TRUE
        if want_reverse is True:
            print("Since reverse is turned on, equil time will be computed from the end of the trajectory")
            neighbor_array = neighbor_array[:, ::-1]
            
        
        ## FINDING TOTAL ITERATIONS
        num_iter = int((num_frames - frame_for_convergence) / inc) + 1
        
        ## GETTING ARRAY
        frame_list = [ np.arange(0,frame_for_convergence + 1) + inc * each_iter for each_iter in range(num_iter)]
        
        ## COMPUTING EQUILIBRATION TIME
        mu_storage = self.compute_mu_for_various_frames(neighbor_array = neighbor_array,
                                                        frame_list = frame_list,
                                                        method_type = method_type)
        return mu_storage, frame_list

    ### FUNCTION TO GET CONVERGENCE
    @staticmethod
    def compute_bounds_for_convergence_mu(mu_storage_reverse,
                                          percent_error = 5,
                                          convergence_type = "percent_error"):
        '''
        The purpose of this function is to get the convergence for all mu values 
        taken for convergence calculations
        INPUTS:
            mu_storage_reverse: [np.array, (num_sampling, num_grid_pts)]
                mu array for each sampling sets. Note that this function assumes 
                that the last sampling takes into account the entire trajectory. 
                Therefore, we will focus on getting that value. 
            percent_error: [float]
                percent error that you allow mu to vary
            convergence_type: [str]
                type of convergence.
                    percent_error:
                        This will look for the error based on the percent error. 
        OUTPUTS:
            theoretical_bounds: [list, shape = num_grid_pts]
                list of theoretical bounds based on the mu distribution
            index_converged: [list, num_grid_pts]
                list of converged index
        '''
        ## GETTING THEORETICAL BOUND
        theoretical_bounds =  [ find_theoretical_error_bounds(value = each_value, 
                                                              percent_error = percent_error,
                                                              convergence_type = convergence_type) 
                                for each_value in mu_storage_reverse[-1] ]
        
        ## STORING VALUES
        index_converged = []
        
        ## LOOPING THROUGH EACH INDEX
        for grid_index in range(mu_storage_reverse.shape[1]):
    
            ## DEFINING THEORETICAL BOUND
            current_bound = theoretical_bounds[grid_index][1]
            ## DEFINING CURRENT Y
            y = mu_storage_reverse[:,grid_index]
        
            ## FINDING CONVERGED VALUE FROM THE END
            index = get_converged_value_from_end(y_array = y, 
                                                 desired_y = mu_storage_reverse[-1][grid_index],
                                                 bound = current_bound,
                                                 )
            ## APPENDING
            index_converged.append(index)
        return theoretical_bounds, index_converged
    
    #### FUNCTION TO PLOT SAMPLING TIME FOR A GRID
    @staticmethod
    def plot_convergence_of_mu( **kwargs
                                ):
        '''
        The purpose of this function is to plot mu vs. time for different sampling 
        times
        INPUTS:
            mu_storage_reverse: [np.array, (num_sampling, num_grid_pts)]
                mu array for each sampling sets. Note that this function assumes 
                that the last sampling takes into account the entire trajectory. 
                Therefore, we will focus on getting that value. 
            x: [np.array]
                x array (e.g. frames)
            grid_index: [int]
                grid index that you want to print
            theoretical_bounds: [list]
                list of theoretical bounds
            index_converged: [list]
                list of indices of converged values
        OUTPUTS:
            fig, ax: 
                figure and axis for convergence
        '''
        ## PLOTTING FIGURE AND AXIS
        fig, ax = plot_convergence_of_mu(**kwargs)
        return fig, ax
    
    ### FUNCTION TO PLOT MINIMUM SAMPLING TIME
    @staticmethod
    def plot_sampling_time_vs_grid_index(sampling_time_x_converged):
        '''
        This function plots the minimum sampling time per grid point.
        INPUTS:
            sampling_time_x_converged: [list]
                list of sampling time for convergence
        OUTPUTS:
            fig, ax:
                figure and axis for sampling
        '''
    
        ## PLOTTING
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
        
        ## ADDING AXIS LABELS
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Minimum sampling time")
        
        ## DEFINIG GRID INDEXX
        grid_index = np.arange(len(sampling_time_x_converged))
        
        ## PLOTTING SCATTER
        ax.scatter(grid_index, sampling_time_x_converged, color='k', linewidth = 1, s=10)
        
        ## PLOTTING MAXIMA
        max_sampling_time = np.max(sampling_time_x_converged)
        ax.axhline(y = max_sampling_time, color = 'red', linestyle = '--')
        
        ## FITTING
        fig.tight_layout()
        
        return fig, ax
        
    ### FUNCTION TO GET SAMPLING TIME
    def main_compute_sampling_time_from_reverse(self,
                                                mu_storage_reverse,
                                                frame_list,
                                                percent_error,
                                                convergence_type = "percent_error",
                                                ):
        '''
        This function computes sampling time from the reverse of the trajectory. 
        INPUTS:
            mu_storage_reverse: [np.array:
                mu values stored in reverse
            neighbor_list: [np.array, shape = (num_grid, num_frames)]
                neighbor list array
            frame_rate: [int]
                frame rate to run the sampling time
            percent_error: [float]
                percent error allowed for deviation
            convergence_type: [str]
                convergence type
        OUTPUTS:

            theoretical_bounds: [list]
                theoretical bounds from the percent error
        '''        
        ## GETTING CONVERGENCE
        theoretical_bounds, index_converged = self.compute_bounds_for_convergence_mu(mu_storage_reverse,
                                                                                     percent_error = percent_error,
                                                                                     convergence_type = convergence_type)
        ## GETTING X ARRAY
        x = [ len(each_array) for each_array in frame_list] # each_array[-1]
        
        ## GETTING ALL SAMPLING TIME
        sampling_time_x_converged = [ x[idx] for idx in index_converged]
        
        return theoretical_bounds, index_converged, sampling_time_x_converged, x


### FUNCTION TO FIT TO POLYNOMIAL
def fit_polynomial(x, y, order = 2):
    ''' 
    This function fits the data to a polynomial. 
    INPUTS:
        x: [np.array]
            x array
        y: [np.array]
            y array
        order: [int, default = 2]
            order that you want to fit your data to
    OUTPUTS:
        coeff: [np.array]
            polynomial coefficients, e.g. 
                array([-0.01678549,  0.18287894, -0.271743  ])
        p_func: [obj]
            polynomial function that you could use to 
            output values, e.g. p_func(value)
    '''
    ''' OLD numpy
    ## FITTING TO GAUSSIAN
    coeff = np.polyfit( x, y, order )
    ## GETTING P FUNCTION
    p_func = np.poly1d(coeff)
    '''
    ## FITTING TO GAUSSIAN
    coeff = poly.polyfit( x, y, order )
    ## GETTING P FUNCTION
    p_func = poly.Polynomial(coeff)
    # np.poly1d(coeff)
    return coeff, p_func

### FUNCTION TO GET NORMALIZATION FOR X
def normalize_x(x, P_x):
    '''
    The purpose of this function is to normalize the 
    x distribution. This is done by using some probability 
    of x = N function (P_x).
    INPUTS:
        x: [np.array]
            x value array
        P_x: [np.array]
            probability of x array
    OUTPUTS:
        x_norm: [np.array]
            normalized x values
    '''
    ## GETTING NORMALIZATION CONSTANT
    normalize_constant = np.sum(x*P_x)
    
    ## GETTING NEW AX
    x_norm = x/normalize_constant
    return x_norm

## FUNCTION TO ADD FITTED DISTRIBUTION TO FIGURE
def add_fit_to_plot(ax, p_func,
                    num_points = 100,
                    line_dict={
                            'linestyle': '--',
                            'color': 'blue'
                            }
                    ):
    '''
    The purpose of this function is to add a fit to the 
    plot. 
    INPUTS:
        ax: [obj]
            axis object
        p_func: [obj]
            fitting object from numpy.polyfit and poly1d
        num_points: [int, default = 100]
            number of points to add into the plot. Note that 
            the more the points, the finer the line.
    OUTPUTS:
        void. this will just output an image for the plot
    '''
    ## GETTING X LIMITS
    x_limits = ax.get_xlim()
    
    ## DEFINING LINESPACE
    x_lin = np.linspace(0,x_limits[1],100)
    ## PLOTTING
    ax.plot(x_lin, p_func(x_lin), **line_dict)
    return

## PLOTTING P(N) DISTRIBUTION
def plot_p_N_dist(x, 
                  y,
                  xlabel = 'N',
                  ylabel = 'P(N)',
                  fig_size_cm = FIGURE_SIZE,
                  fig = None,
                  ax = None,
                  scatterkwargs={
                          'color' : 'k'}):
    '''
    The purpose of this function is to plot p(N) distribution 
    for a given data set. 
    INPUTS:
        x: [np.array]
            x value array, which should be an array 
            with integer values. These values should 
            range the data that you have (i.e. probabilities that 
            are greater than zero)
        y: [np.array]
            probability of finding x = n. 
        p_N: [np.array]
            p(N) distribution, which is used to normalize 
            the X distribution. If this is None, then 
            we will simply use the y values. 
        scatterkwargs: [dict]
            dictionary of items for kwargs
    OUTPUTS:
        fig, ax:
            figure and axis
    '''
    ## PLOTTING
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## ADDING AXIS
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ## PLOTTING X AND Y
    ax.scatter(x,y, **scatterkwargs)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT MU VALUES
def plot_scatter_mu_values(mu_values, fig_size = FIGURE_SIZE):
    '''
    The purpose of this function is to plot the scatter 
    mu values across grid.
    INPUTS:
        mu_values: [np.array]
            mu values
    OUTPUTS:
        fig, ax
    '''
    ## CREATING PLOT
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size)
    
    ## DEFINING GRID INDEXES
    grid_index = np.arange(0, len(mu_values))
    
    ## ADDING LABELS
    ax.set_xlabel("Grid index")
    ax.set_ylabel("$\mu$ (kT)")
    
    ## CREATING HISTOGRAM PLOT
    ax.scatter(grid_index, mu_values, color="k")
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax
###############################################################################
### FUNCTION TO GET PROBABILITY DISTRIBUTION FROM UNNORMALIZED DISTRIBUTION ###
###############################################################################
class compute_mu_from_unnorm_p_N:
    '''
    The purpose of this function is to compute mu from 
    unnormalized p_N
    
    INPUTS:
        unnorm_p_N: [np.array, shape = (num_grid, num_max)]
            unnormalized probability distribution
    OUTPUTS:
        self.mu_storage: [df]
            dataframe of the mu values
    FUNCTIONS:
        compute_all_mu_values:
            function that computes mu value for all grid points
        compute_mu_value_for_single_p_N: 
            function that computes mu values for a single grid point
    ACTIVE FUNCTIONS:
        plot_p_N_dist_for_grid:
            plots p(N) vs. N for a specific grid index
    USAGE EXAMPLE:
        ## GETTING MU DIST
        mu_dist = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
        ## DEFINING GRID INDEX
        grid_index = 1
        ## PLOTTING P(N)
        fig, ax = mu_dist.plot_p_N_dist_for_grid(grid_index = grid_index)
        ## PLOTTING LOG P(N)
        fig, ax = mu_dist.plot_log_p_N_for_grid(grid_index = grid_index)
    
    '''
    def __init__(self,
                 unnorm_p_N):
        ## STORING UNNORMALIZED DISTRIBUTION
        self.unnorm_p_N = unnorm_p_N
        
        ## COMPUTING FOR ALL MU
        self.mu_storage = self.compute_all_mu_values()
        
        return
    
    ## FUNCTION THAT COMPUTES ALL MU VALUES
    def compute_all_mu_values(self):
        '''
        The purpose of this function is to compute mu 
        values for all possible grid points.
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            mu_storage: [df]
                list of dictionaries containing mu value 
                information for each grid point.
        '''
        ## GETTING LIST
        mu_storage = []
        
        ## LOOPING
        for idx, p_N in enumerate(self.unnorm_p_N):
            ## COMPUTING MU VALUES
            output_dict = self.compute_mu_value_for_single_p_N(p_N= p_N)
            ## STORING
            mu_storage.append(output_dict)
        
        ## GENERATING DATAFRAME
        mu_storage = pd.DataFrame(mu_storage)
        
        return mu_storage
    
    ## FUNCTION TO COMPUTE MU VALUE
    @staticmethod
    def compute_mu_value_for_single_p_N(p_N,
                                        want_shift = False):
        '''
        The purpose of this function is to compute the mu 
        value given the unnormalized distribution.
        NOTES:
            - This function first takes only data that has 
            probabilities that are > 0. Therefore, occurences
            that have zero probabilities are not accounted for.
            - This function does NOT shift the distribution, which 
            may be erroneous. 
        INPUTS:
            p_N: [np.array]
                Array of p(N) distribution
            
        OUTPUTS:
            output_dict: [dict]
                output dictionary containing the following:
                -----
                mu_value: [float]
                    mu value that is outputted
                p_func: [obj]
                    object for fitting (Gaussian distribution)
                    Fitting is: Ax^2 + Bx + C
                p_func_norm: [obj]
                    Same a p_func, but p_function is using normalized N / <N>
                x: [np.array]
                    number of occurences array
                y: [np.array]
                    probability distribution of x
                x_norm: [np.array]
                    normalized x array based on the probability distribution
                log_y: [np.array]
                    log values of y
                -----
        '''
        ## NORMALIZING BY SUM
        norm_p = p_N / p_N.sum()
        
        ## DEFINING LOGICAL FOR DATA THAT IS AVAILABLE
        data_available = norm_p>0
        
        ## GETTING X RANGE
        x = np.arange(0, len(p_N), 1)[data_available]
        
        ## GETTING Y VALUES
        y = norm_p[data_available]                                
        
        ## GETTING NORMALIZED X
        x_norm = normalize_x(x = x, 
                             P_x = y)
        
        ## TAKING LOG OF DISTRIBUTION
        log_y = np.log(y)
        
        ## FITTING
        _, p_func = fit_polynomial(x = x, 
                                   y = log_y, 
                                   order = 2)
        

        ## FITTING
        _, p_func_norm = fit_polynomial(x = x_norm, 
                                       y = log_y, 
                                       order = 2)

        ## DEFINING MU VALUE
        mu_value = - p_func(0)
        
        ## DEFINING OUTPUT DICTIONARY
        output_dict = {
                'mu_value': mu_value,
                'p_func': p_func,
                'p_func_norm': p_func_norm,
                'x': x,
                'y': y,
                'x_norm': x_norm,
                'log_y': log_y
                }
        
        return output_dict
    
    ### FUNCTION THAT PLOTS P(N) DISTRIBUTIONS
    def plot_p_N_dist_for_grid(self,
                               grid_index =0,
                               fig_size_cm = FIGURE_SIZE,
                               fig = None,
                               ax = None,):
        '''
        The purpose of this function is to plot p(N) distributions 
        for specific grid indices. 
        INPUTS:
            self: [obj]
                class object
            grid_index: [int]
                specific grid index you would like to plot 
                p_N for. 
        OUTPUTS:
            fig, ax: [obj]
                figure and axis of the p(N) distribution. 
        '''
    
        ## PLOTTING P N
        fig, ax = plot_p_N_dist(x = self.mu_storage.loc[grid_index]['x'],
                                y = self.mu_storage.loc[grid_index]['y'],
                                ylabel = 'P(N)',
                                fig_size_cm = fig_size_cm,
                                fig = fig,
                                ax = ax,)
        return fig, ax
    
    ### FUNCTION THAT PLOTS LOG Y DISTRIBUTION AND FIT
    def plot_log_p_N_for_grid(self,
                              grid_index = 0,
                              fig_size_cm = FIGURE_SIZE,
                              fig = None,
                              ax = None,
                              scatterkwargs={
                                      'color' : 'k'},
                                line_dict={
                                        'linestyle': '--',
                                        'color': 'blue'
                                        },
                              num_points = 100,
                              ):
        '''
        The purpose of this function is to plot the log P(N) vs
        N for a specific grid index.
        INPUTS:
            self: [obj]
                class object
            grid_index: [int]
                specific grid index you would like to plot 
                p_N for. 
            fig, ax:
                figure and axis to plot on
            scatterkwargs: [dict]
                dictionary for the dots
            line_dict: [dict]
                dictionary for the line plot
            num_points: [int]
                number of points used for plotting gaussian fit
        OUTPUTS:
            fig, ax: [obj]
                figure and axis of the log p(N) distribution. 
        
        '''
        
        ## PLOTTING P_N DISTRIBUTION
        fig, ax = plot_p_N_dist(x =  self.mu_storage.loc[grid_index]['x_norm'], 
                                y =  self.mu_storage.loc[grid_index]['log_y'],
                                ylabel = 'log P(N)',
                                xlabel = "N/<N>",
                                fig = fig,
                                ax = ax,
                                fig_size_cm = fig_size_cm,
                                scatterkwargs = scatterkwargs)
        ## ADDING GAUSSIAN FIT 
        add_fit_to_plot(ax = ax, 
                        p_func = self.mu_storage.loc[grid_index]['p_func_norm'],
                        line_dict = line_dict,
                        num_points = num_points)
        
        return fig, ax

### FUNCTION TO GET ALL UNNORM STORAGE
def find_all_p_N_dist_for_sampling(num_neighbors_array,
                                   frame_list,
                                   grid_index):
    '''
    This function finds all p_N distributions for each 
    frame list
    INPUTS:
        num_neighbors_array: [np.array]
            number of neighbors array
        frame_list: [list]
            list of arrays for frame list
    '''

    ## STORING
    unnorm_p_N_storage = []
    
    ## LOOPING THROUGH FRAMES
    for idx, frame_array in enumerate(frame_list):
    
        ## FINDING THE NUMBER DISTRIBUTION FOR SPECIFIC GRID
        num_neighbors_for_grid = num_neighbors_array[[[grid_index]], frame_array]

        ## COMPUTING NUMBER DIST
        unnorm_p_N = compute_num_dist(num_neighbors_array = num_neighbors_for_grid, # num_neighbors_array,
                                      max_neighbors = MAX_N)
        ## APPENDING
        unnorm_p_N_storage.append(unnorm_p_N)
        
    return unnorm_p_N_storage


### FUNCTION TO PLOT P_N DIST AND PLOTS
def plot_p_N_for_sampling_time(unnorm_p_N_storage,
                               grid_index,
                               frame_list,):
    '''
    This function plots p_N across multiple sampling times
    '''
        
    
    ## DEFINING NUMBER OF FRAMES
    length_frame_list = len(unnorm_p_N_storage)
    
    ## GETTING DIFFERENT COLORS
    cmap_colors = plot_tools.get_cmap(length_frame_list)
    
    ## DEFINING FIGURE AND AXIS
    fig, ax = None, None
    
    ## DEFINING MAXIMUM COLS
    subplot_max_cols = 5
    
    ## FINDING NUMBER OR ROWS
    num_rows = int(np.rint(length_frame_list/subplot_max_cols))
    
    ## CREATING SUBPLOTS 
    figs, axs = plt.subplots( num_rows, subplot_max_cols )
    
    ## LOOPING THROUGH FRAMES
    for idx, frame_array in enumerate(frame_list):
        
        ## DEFINING PROBABILITY DIST
        unnorm_p_N = unnorm_p_N_storage[idx]
        
        ## COMPUTING MU DISTRIBUTION
        mu_dist = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
        
        ## GETTING X
        x = mu_dist.mu_storage.loc[0]['x']
        y = mu_dist.mu_storage.loc[0]['log_y']
        p_func = mu_dist.mu_storage.loc[0]['p_func']
        mu = mu_dist.mu_storage.loc[0]['mu_value']
        
        ## DEFINING STYLES
        line_dict={
                "linestyle": '-',
                "linewidth": 2,
                "alpha": 1,
                "color" : cmap_colors(idx),
                "label": '-'.join( [ str(frame_array[0]) , str(frame_array[-1])  ] )
                }
        
        ## PLOTTING SINGLE GRID POINT
        fig, ax = plot_histogram_for_prob_dist(unnorm_p_N = unnorm_p_N,
                                               grid_index = 0,
                                               line_dict = line_dict,
                                               normalize = True,
                                               fig = fig,
                                               ax = ax)
        
        ## PLOTTING GAUSSIAN FIT
        fig_gaus, ax_gaus = plot_gaussian_fit_to_find_mu(unnorm_p_N = unnorm_p_N,
                                                         x = x,
                                                         y = y,
                                                         mu = mu,
                                                         p_func= p_func,
                                                         grid_index=0,
                                                         want_shapiro= False,
                                                         fig = figs,
                                                         ax = figs.axes[idx])
        
        ## ADDING TITLE
        ax_gaus.set_title("%s"%(line_dict["label"] ) )

    
    ## SETTING LABELS
    for each_row in range(num_rows):
        axs[each_row][0].set_xlabel("Non-negative occurances")
        axs[each_row][0].set_ylabel("-log(P(N))")
    
    ## TIGHT LAYOUT
    figs.subplots_adjust(wspace = 0.3, hspace = 0.5)
    
    ## ADDING TITLE
    ax.set_title("Grid index: %d"%(grid_index) )    
    
    ## ADDING LEGEND
    ax.legend()
    
    return fig, ax, figs, axs

### FUNCTION TO PLOT THE CONVERGENCE OF MU
def plot_convergence_of_mu( mu_storage_reverse,
                            x,
                            grid_index,
                            theoretical_bounds,
                            index_converged,
                            fig = None,
                            ax = None,
                            color = 'k',
                            alpha = 0.5,
                            xlabel = "Total frames used",
                            ylabel="$\mu$ (kT)"
                            ):
    '''
    The purpose of this function is to plot mu vs. time for different sampling 
    times
    INPUTS:
        mu_storage_reverse: [np.array, (num_sampling, num_grid_pts)]
            mu array for each sampling sets. Note that this function assumes 
            that the last sampling takes into account the entire trajectory. 
            Therefore, we will focus on getting that value. 
        x: [np.array]
            x array (e.g. frames)
        grid_index: [int]
            grid index that you want to print
        theoretical_bounds: [list]
            list of theoretical bounds
        index_converged: [list]
            list of indices of converged values
    OUTPUTS:
        fig, ax: 
            figure and axis for convergence
    '''

    ## PLOTTING
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## ADDING AXIS LABELS
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ## GETTING Y VALUE
    y = mu_storage_reverse[:, grid_index]

    ## PLOTTING
    ax.plot(x,y, marker='.', linestyle = '-', color=color, markersize = 10)
    
    ## FILLING
    ax.fill_between(x, theoretical_bounds[grid_index][0][0], theoretical_bounds[grid_index][0][1], color=color,
                     alpha=0.5)
        
    ## GETTING OPTIMAL
    opt_x = x[index_converged[grid_index]]
    
    ## PLOTTING X
    ax.axvline(x = opt_x, linestyle = '--', color = color)
    
    ## FITTING
    fig.tight_layout()
    return fig, ax


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":

    ## DEFINING LIGANDS
    ligands = [ "dodecanethiol",
                "C11OH",
                "C11NH2",
                "C11COOH",
                "C11CONH2",
                "C11CF3",                
                ]
    
    ## DEFINING RELATIVE PATH
    relative_path = "25.6-0.24-0.1,0.1,0.1-0.33-all_heavy"
    
    ## DEFINING COMPUTE RELATIVE PATH
    compute_path = r"compute_neighbors"
    
    ## DEFINING NEIGHBOR PICKLE
    neighbor_pickle_name = r"0-50000.pickle"
    
    ## DEFINING MU PICKLE
    mu_pickle_name = "mu.pickle"
    

    
    #%%
    
    ## CHECKING SPYDER
    if check_spyder() is True:
        ##%% GETTING PLANAR MU DISTRIBUTIONS + GNP DISTRIBUTIONS
        
        #####################################
        ### PLOTTING PLANAR AND SPHERICAL ###
        #####################################
        
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 30 ]    
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING TYPES
        shape_types = ['planar_frozen', 'spherical_frozen']
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = [ 'Planar_sims', 'NP_sims' ]
        
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = ligands,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
        figure_name = "Planar_spherical_mu_dist"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        #%% 
        
        ################################################
        ### COMPARISON OF PLANAR FROZEN AND UNFROZEN ###
        ################################################
            
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 30 ]    
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING LIGANDS
        ligands = [ "dodecanethiol",
                    "C11OH",
                    "C11NH2",
                    "C11COOH",
                    "C11CONH2",
                    "C11CF3",                
                    ]
        ## DEFINING TYPES
        shape_types = ['planar_frozen', 'planar_unfrozen']
        ## DEFINING DESIRED FOLDERS
        desired_folders = [ 'Planar_sims', 'Planar_sims_unfrozen' ]
    
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = ligands,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        
        ## STORING FIGURE
        figure_name = "planar_frozen_unfrozen_comparison"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
                                 
         #%%
         
         
        ###########################################################
        ### COMPARISON OF PLANAR FROZEN AND UNFROZEN AND SPRING ###
        ###########################################################
            
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 30 ]    
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING LIGANDS
        ligands = [ "dodecanethiol",
                    "C11OH",
                    "C11NH2",
                    "C11COOH",
                    "C11CONH2",
                    "C11CF3",                
                    ]
        ## DEFINING TYPES
        shape_types = ['planar_frozen', 'planar_unfrozen', 'planar_spr_600', 'planar_spr_50']
        ## DEFINING DESIRED FOLDERS
        desired_folders = [ 'Planar_sims', 'Planar_sims_unfrozen', 'Planar_SPR_600', 'Planar_SPR_50' ]
    
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = ligands,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        
        ## STORING FIGURE
        figure_name = "planar_frozen_unfrozen_spring_comparison"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
                                 

         
        
    
        #%%
        
        ########################################################
        ### COMPARISON OF GNP FROZEN AND UNFROZEN AND SPRING ###
        ########################################################
            
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 30 ]    
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING TYPES
        shape_types = ['spherical_frozen', 'gnp_spr_600',  'gnp_spr_50',]
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = ['NP_sims', 'GNP_SPR_600', 'GNP_SPR_50' ]
        
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = ligands,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
        figure_name = "spherical_frozen_spring_comparison_mu_dist"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        
        #%% 
        
        ###########################################
        ### COMPARISON OF GNP AND PLANAR SPRING ###
        ###########################################
            
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 30 ]    
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING TYPES
        shape_types = ['planar_spr_600', 'gnp_spr_600']
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = ['Planar_SPR_600', 'GNP_SPR_600', ]
        
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = ligands,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
        figure_name = "planar_vs_spherical_spring_const_600"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
        #%%
        
        ###########################################
        ### COMPARISON OF GNP AND PLANAR SPRING ###
        ###########################################
            
        ## DEFINING RELATIVE PATH
        relative_path = "25.6-0.24-0.1,0.1,0.1-0.33-all_heavy-0-100000"
        ## DEFINING NEIGHBOR PICKLE
        neighbor_pickle_name = r"0-100000.pickle"
        
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 25 ]
        # [0, 30]
        # [ 0, 30 ]    
        ## DEFINING STEP SIZE
        step_size = 0.5
        # HISTOGRAM_STEP_SIZE
        
        ## DEFINING TYPES
        shape_types = ['planar_spr_50', 'gnp_spr_50']
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = ['Planar_SPR_50_equil_with_vacuum_5nm', 'GNP_SPR_50_nptequil', ]
        # ['Planar_SPR_50_notempanneal', 'GNP_SPR_50', ]
        
        # ['Planar_SPR_50_long_z_8nm', 'GNP_SPR_50', ]
        
        # ['Planar_SPR_50_long_z', 'GNP_SPR_50', ]
        # ['Planar_SPR_50', 'GNP_SPR_50', ]
        
        ## DEFINING RELATIVE PATH
        relative_path = [ "30-0.24-0.1,0.1,0.1-0.33-all_heavy-0-150000",
                          "30-0.24-0.1,0.1,0.1-0.33-all_heavy-0-100000", ]
        relative_path = [ "27-0.24-0.1,0.1,0.1-0.25-all_heavy-0-150000",
                          "27-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000", ]
        relative_path = [ "25.6-0.24-0.1,0.1,0.1-0.33-all_heavy",
                          "25.6-0.24-0.1,0.1,0.1-0.33-all_heavy" ]
#        relative_path = [ "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-150000",
#                          "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000" ]
        
        relative_path = [ "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-50000",
                          "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000" ]
        
        relative_path = [ "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",
                          "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000" ]
#        relative_path = [ "norm-0.80-0.24-0.1,0.1,0.1-0.33-all_heavy-0-150000",
#                          "norm-0.80-0.24-0.1,0.1,0.1-0.33-all_heavy-0-100000" ]
        ## DEFINING MU PICKLE
        mu_pickle_name = "mu.pickle"
    
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = ligands,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
#        figure_name = "planar_vs_spherical_spring_const_50"+ '-' + relative_path[0]
#        plot_tools.store_figure( fig = fig,
#                                 path = os.path.join(STORE_FIG_LOC,
#                                                     figure_name),
#                                 save_fig = SAVE_FIG,
#                                 )
                                 

        

        #%% ADDING PURE WTAER PLOT
         
        ## LOADING PURE WATER SIMULATIONS
        parent_dir_name = DIRECTORY_DICT['Water_only'] 
         
        ## DEFINING SIMULATION DIRN AME
        simulation_dir_name = "wateronly-6_nm-tip3p-300.00_K"
        
        ## DEFINING RELATIVE PATH
        relative_path = '0.24-0.25-0-50000'
        relative_path = '0.24-0.33-0-50000'
        
        ## DEFINING MU PICKLE
        mu_pickle_name = "mu.pickle"
        
        ## DEFINING NEIGHBOR PICKLE
        neighbor_pickle_name = r"0-50000.pickle"
        
        ## DEFINING COMPUTE RELATIVE PATH
        compute_path = r"compute_neighbors"
        
        ## PATH TO SIMULATION
        path_to_sim = os.path.join(MAIN_SIM_FOLDER,
                                   parent_dir_name,
                                   simulation_dir_name,
                                   relative_path)
        
        ## PATH TO NEIGHBORS
        path_to_neighbors = os.path.join(path_to_sim,
                                   compute_path,)
        
        ## LOADING PICKLE
        hydration_map = extract_hydration_maps()
        mu_values = hydration_map.load_neighbor_values(main_sim_list = [path_to_sim],
                                                           pickle_name = mu_pickle_name)
        
        
        ## adding mu value to plot
        [axes.axvline(x = mu_values[0], linestyle='--', color='k', label='Pure water') for axes in ax]
        
        ## adding legend
        [axes.legend() for axes in ax]
        
        ## STORING FIGURE
        figure_name = "planar_vs_spherical_spring_const_50_with_bulk_water"+ '-' + relative_path[0]
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
        
        
        
        #%% DEBUGGING FOR PURE WATER
        
        
        ## debugging information
        percent_error = 2
        convergence_type = "value"
        frame_rate = 5000
        
        ## DEFINING NEIGHBORS LIST
        neighbor_list = hydration_map.load_neighbor_values(main_sim_list = [path_to_neighbors],
                                                           pickle_name = neighbor_pickle_name)
        
        ## GETTING NEIGHBORS ARRAY
        num_neighbors_array = neighbor_list[0]
        
        ## COMPUTING NUMBER DIST
        unnorm_p_N = compute_num_dist(num_neighbors_array = num_neighbors_array,
                                      max_neighbors = MAX_N)
        
        ## COMPUTING NUMBER DIST
        unnorm_p_N = compute_num_dist(num_neighbors_array = num_neighbors_array,
                                      max_neighbors = MAX_N)
    
        ## GETTING MU DIST
        mu_dist = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
        
        ## DEFINING GRID INDEX
        grid_index = 0
        ## PLOTTING P(N)
        fig, ax = mu_dist.plot_p_N_dist_for_grid(grid_index=grid_index)
        
        ## STORING FIGURE
        figure_name = "PURE_WATER_PN_DIST_%s"%(relative_path)
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        ## PLOTTING LOG P(N)
        fig, ax = mu_dist.plot_log_p_N_for_grid(grid_index=grid_index)
        
        ## STORING FIGURE
        figure_name = "PURE_WATER_LOGPN_DIST_%s"%(relative_path)
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        
        ## GENERATING DEBUGGING MU CLASS
        mu_debug = debug_mu_convergence()
        
        ## COMPUTING CONVERGENCE TIME -- DEBUGGED, CHECKED CONVERGENCE
        mu_storage_reverse, frame_list = mu_debug.compute_mu_convergence_time(num_neighbors_array = num_neighbors_array, # [:,10000:]
                                                                              frame_rate = frame_rate,
                                                                              want_reverse = False,
                                                                              method_type = "new") 
        
        ## GETTING CONVERGENCE INFORMATION
        theoretical_bounds, index_converged, sampling_time_x_converged, x \
                        = mu_debug.main_compute_sampling_time_from_reverse(mu_storage_reverse = mu_storage_reverse,
                                                                           frame_list = frame_list,
                                                                           percent_error = percent_error,
                                                                           convergence_type = convergence_type,
                                                                           )

        ## PLOTTING SAMPLING TIME VERSUS GRID
        fig, ax = mu_debug.plot_sampling_time_vs_grid_index(sampling_time_x_converged)
        
        ## STORING FIGURE
        figure_name = "PUREWATER_sampling_vs_grid" + "_" + convergence_type + "_" + str(percent_error) + "_perc_%s"%(relative_path)
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
        ## GETTING INDICES 
        indices_maxima = np.argmax(sampling_time_x_converged)

        
        ## PLOTTING CONVERGENCE
        fig, ax = mu_debug.plot_convergence_of_mu( mu_storage_reverse = mu_storage_reverse,  
                                                   x = x,
                                                   grid_index = indices_maxima,
                                                   theoretical_bounds = theoretical_bounds,
                                                   index_converged = index_converged
                                                   )
        
        ## STORING FIGURE
        figure_name = "PUREWATER_sampling_vs_grid" + "_" + convergence_type + "_" + str(percent_error) + "_perc_%s"%(relative_path)
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        
        
     #%%
        ###########################################
        ### COMPARISON OF GNP AND PLANAR SPRING ###
        ###########################################
            
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 5, 30 ]    
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING TYPES
        shape_types = ['gnp_spr_600']
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = ['GNP_SPR_600']
        
        ## DEFINING LIGANDS
        current_ligs = ['dodecanethiol', 'C11OH']
        
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = current_ligs,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
        figure_name = "spring_const_600_50ns"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
         #### RUNNING FOR 100 NS SIMS
        ## DEFINING TYPES
        shape_types = ['gnp_spr_600']
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = ['GNP_SPR_600_extended']
        
        ## DEFINING LIGANDS
        current_ligs = ['dodecanethiol', 'C11OH']
        
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = current_ligs,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
        figure_name = "spring_const_600_100ns"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
        
        
        
        #%%
        
        
        ### PLOTTING WITH DOUBLE BONDS
        
        ## DEFINING HISTOGRAM RANGE
        histogram_range =  [ 0, 15 ]
        
        ## DEFINING STEP SIZE
        step_size = HISTOGRAM_STEP_SIZE
        
        ## DEFINING TYPES
        shape_types = ['gnp_spr_50', 'gnp_spr_50'] # , 'gnp_spr_50'
    
        ## DEFINING DESIRED FOLDERS
        desired_folders = ['GNP_SPR_50', 'GNP_SPR_50_double_bonds']
        
        ## DEFINING LIGANDS
        current_ligs = [['dodecanethiol', 'C11OH',], ['C11double67OH', 'dodecen-1-thiol']]
        
        relative_path = [ "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000",
                          "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-50000" ]
        
        
        ## GETTING MULTIPLE EXTRACTION
        fig, ax, histogram_data_storage = extract_histogram_multiple_ligands(main_path = MAIN_SIM_FOLDER,
                                                                             folder_dict = DIRECTORY_DICT,
                                                                             desired_folders = desired_folders,
                                                                             shape_types = shape_types,
                                                                             relative_path = relative_path,
                                                                             ligands = current_ligs,
                                                                             mu_pickle_name = mu_pickle_name,
                                                                             histogram_range = histogram_range,
                                                                             step_size = step_size,
                                                                             )
        ## STORING FIGURE
        figure_name = "spring_const_50_double_bond_comparison"
        plot_tools.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
                                 
        
        #%%

        
        
        #%%
        
    ## CHECK SPYDER
    if check_spyder() is False:
        
        ## DEFINING LIGANDS
        ligands = [ 
                "dodecanethiol",
                    "C11OH",
                    "C11NH2",
                    "C11COOH",
                    "C11CONH2",
                    "C11CF3",                
                    ]
        
        ## DEFINING EACH
        sims_dict = {
                'Planar':{
                        'desired_sims': {
                            'type'   : ['planar_spr_50'],  # planar_spr_600 planar_spr_50
                            'ligand' : ligands,
                            },
                        'desired_folders': ['Planar_SPR_50_equil_with_vacuum_5nm', ], # Planar_SPR_600_extended Planar_SPR_50
                        'relative_path': [
                                            "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",

                                        # "norm-0.80-0.24-0.1,0.1,0.1-0.33-all_heavy-0-150000",
                                        # "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-150000",
                                          #  '27-0.24-0.1,0.1,0.1-0.25-all_heavy-0-150000', 
#                                          '30-0.24-0.1,0.1,0.1-0.10-all_heavy-0-150000',
#                                          '30-0.24-0.1,0.1,0.1-0.16-all_heavy-0-150000',
#                                          '30-0.24-0.1,0.1,0.1-0.20-all_heavy-0-150000',
#                                          '30-0.24-0.1,0.1,0.1-0.25-all_heavy-0-150000',
#                                          '30-0.24-0.1,0.1,0.1-0.33-all_heavy-0-150000',
                                          ],
                        'neighbor_pickle_name': '2000-50000.pickle',
                            # r"0-150000.pickle",
                        },
                'GNP': {
                        'desired_sims': {
                            'type'   : ['gnp_spr_50'], 
                            'ligand' : ligands,
                            }, # GNP_SPR_600_extended
                        'desired_folders': ['GNP_SPR_50_nptequil' ],
                        'relative_path': [
                                "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000",
                                # "norm-0.80-0.24-0.1,0.1,0.1-0.33-all_heavy-0-100000",
                                #  "norm-0.70-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000",
                                # '27-0.24-0.1,0.1,0.1-0.25-all_heavy-0-100000'
                                ],
                        'neighbor_pickle_name': '2000-50000.pickle'
                            # r"0-100000.pickle",
                        },
                        }

        ## DEFINING FRAME RATE
        frame_rate = 2000
        # 5000
        # 2000
        # 10000
        convergence_type = "value"
        percent_error = 2
        
        ## LOOPING
        for idx, each_key in enumerate(sims_dict):
            # if idx == 0:
            desired_sims = sims_dict[each_key]['desired_sims']
            desired_folders = sims_dict[each_key]['desired_folders']
            neighbor_pickle_name = sims_dict[each_key]['neighbor_pickle_name']
            current_rel_paths = sims_dict[each_key]['relative_path']

            ## GETTING MAIN PATH LIST
            main_path_list = get_all_paths(main_path = MAIN_SIM_FOLDER, 
                                           desired_folders = desired_folders, 
                                           folder_dict = DIRECTORY_DICT )
            ## TRACTING DIRECTORIES
            directories = track_directories( main_path_list = main_path_list )
            
            ## GETTING DATAFRAMES THAT YOU DESIRE
            df = directories.get_desired_sims(desired_sims = desired_sims)
        

            ## LOOP THROUGH RELATIVE PATHS
            for relative_path in current_rel_paths:
            
                ## GETTING PATH
                path_compute_neighbors = [ os.path.join(each_path, relative_path, compute_path) for each_path in list(df['PATH']) ]
                
            
                ## LOOPING
                for neighbor_idx, specific_path in enumerate(path_compute_neighbors):
                    # if neighbor_idx == 0 :  # == len(path_compute_neighbors)-1: # 
                    ## DEFINING CURRENT NAME
                    current_name = os.path.basename(os.path.dirname(os.path.dirname(specific_path))) + '-' + relative_path + '-'
                    ## LOADING PICKLE
                    hydration_map = extract_hydration_maps()
                    neighbor_list = hydration_map.load_neighbor_values(main_sim_list = [specific_path],
                                                                       pickle_name = neighbor_pickle_name)
                    ## GETTING NEIGHBORS ARRAY
                    num_neighbors_array = neighbor_list[0]
                    
                    '''
                    ## GETTING MU RESULTS
                    mu_value_array = hydration_map.load_mu_values(main_sim_list = [os.path.dirname(specific_path)],
                                                                  pickle_name = mu_pickle_name)[0]
                    
                    ## COMPUTING NUMBER DIST
                    unnorm_p_N = compute_num_dist(num_neighbors_array = num_neighbors_array,
                                                  max_neighbors = MAX_N)
                    
                    ## COMPUTING MU
                    mu = calc_mu(p_N_matrix = unnorm_p_N,
                                 d = MAX_N)
                    
                    ## GETTING P_N VALUE
                    x, y, p, p_func = get_x_y_from_p_N(p_N = unnorm_p_N[0],
                                                       d = MAX_N)
                    '''
                    #%%
    
                    ## COMPUTING NUMBER DIST
                    unnorm_p_N = compute_num_dist(num_neighbors_array = num_neighbors_array,
                                                  max_neighbors = MAX_N)
    
                    ## GETTING MU DIST
                    mu_dist = compute_mu_from_unnorm_p_N(unnorm_p_N = unnorm_p_N)
                    
                    # ''' PLOTS THE MU DISTRIBUTION
                    ## DEFINING GRID INDEX
                    grid_index = 0
                    ## PLOTTING P(N)
                    fig, ax = mu_dist.plot_p_N_dist_for_grid(grid_index=grid_index)
                    
                    ## STORING FIGURE
                    figure_name = current_name + "plot_p_N_for_grid" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = fig,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )
                    ## PLOTTING LOG P(N)
                    fig, ax = mu_dist.plot_log_p_N_for_grid(grid_index=grid_index)
                    
                    ## STORING FIGURE
                    figure_name = current_name + "plot_log_p_N" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = fig,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )
                    
                    # '''
                    
                    
                    
                    ## PLOTTING MU VALUES
                    mu_values = mu_dist.mu_storage['mu_value'].to_numpy()
                    fig, ax = plot_scatter_mu_values(mu_values = mu_values  )
                    ## ADDING HORIZONTAL LINE FOR MEAN
                    ax.axhline( y =  np.mean(mu_values), color='r', linestyle = '--', label='Avg')
                    ax.legend()
                    
                    
                    ## STORING FIGURE
                    figure_name = current_name + "plot_scatter_mu" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = fig,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )
                    
                    #%%
                    
                    ## GENERATING DEBUGGING MU CLASS
                    mu_debug = debug_mu_convergence()
                    
                    ## COMPUTING CONVERGENCE TIME -- DEBUGGED, CHECKED CONVERGENCE
                    mu_storage_reverse, frame_list = mu_debug.compute_mu_convergence_time(num_neighbors_array = num_neighbors_array, # [:,10000:]
                                                                                          frame_rate = frame_rate,
                                                                                          want_reverse = False,
                                                                                          method_type = "new") 
                    
                    #%%
                    
                    ## GETTING CONVERGENCE INFORMATION
                    theoretical_bounds, index_converged, sampling_time_x_converged, x \
                                    = mu_debug.main_compute_sampling_time_from_reverse(mu_storage_reverse = mu_storage_reverse,
                                                                                       frame_list = frame_list,
                                                                                       percent_error = percent_error,
                                                                                       convergence_type = convergence_type,
                                                                                       )
    
                    ## PLOTTING SAMPLING TIME VERSUS GRID
                    fig, ax = mu_debug.plot_sampling_time_vs_grid_index(sampling_time_x_converged)
                    
                    ## STORING FIGURE
                    figure_name = current_name + "sampling_vs_grid" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = fig,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )
                    #%%
                    
                    
                    ## GETTING INDICES 
                    indices_maxima = np.argmax(sampling_time_x_converged)
                    # 1
                    # 
                    # np.where(np.array(sampling_time_x_converged) == 50000)[0]
                    
                    ## PLOTTING CONVERGENCE
                    fig, ax = mu_debug.plot_convergence_of_mu( mu_storage_reverse = mu_storage_reverse,  
                                                               x = x,
                                                               grid_index = indices_maxima,
                                                               theoretical_bounds = theoretical_bounds,
                                                               index_converged = index_converged
                                                               )
                    
                    ## STORING FIGURE
                    figure_name = current_name + "converg_of_mu" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = fig,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )
                    
                    
                    #%%
                    
                    ### DEFINING GRID INDEX
                    grid_index = indices_maxima
                    
                    ## COMPUTING STORAGE
                    unnorm_p_N_storage = find_all_p_N_dist_for_sampling(num_neighbors_array = num_neighbors_array,
                                                                        frame_list = frame_list,
                                                                        grid_index = grid_index)
                    
                    ## PLOTTING FOR SAMPLING TIME
                    fig, ax, figs, axs = plot_p_N_for_sampling_time(unnorm_p_N_storage = unnorm_p_N_storage,
                                                                    grid_index = grid_index,
                                                                   frame_list = frame_list)
                    
                    ## STORING FIGURE
                    figure_name = current_name + "subplot" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = figs,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )
                        
                    ## STORING FIGURE
                    figure_name = current_name + "unnorm_p_N" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
                    plot_tools.store_figure( fig = fig,
                                             path = os.path.join(STORE_FIG_LOC,
                                                                 figure_name),
                                             save_fig = SAVE_FIG,
                                             )

                        
                        #%% METHOD FOR PLOTTING MAYAVI
#                        
#                        ## DEFINING RELATIVE PATH TO GRID
#                        path_grid = os.path.join(os.path.dirname(specific_path),
#                                     'grid-0_1000',
#                                     'out_willard_chandler.dat')
#                        
#                        ## DEFINING PATH TO SIM
#                        path_to_sim = os.path.dirname(os.path.dirname(specific_path))
#                        
#                        ## DEFINING THE GRID
#                        grid = load_datafile(path_grid)
#                        
#                        ## DEFINING GRO AND XTC
#                        gro_file = r"sam_prod_0_50000-heavyatoms.gro"
#                        xtc_file = r"sam_prod_0_50000-heavyatoms.xtc"
#                        
#                        ## DEFINING FRAME
#                        frame = 0
#                        
#                        ## LOADING TRAJECTORY
#                        traj_data = import_tools.import_traj(directory = path_to_sim,
#                                                             structure_file = gro_file,
#                                                             xtc_file = xtc_file,
#                                                             index = frame)
#                        #%%
#
#                        ## LIGAND ATOM INDICES
#                        _, atom_index = get_atom_indices_of_ligands_in_traj(traj = traj_data.traj)
#                        
#                        ## PLOTTING MAYAVI
#                        fig = plot_tools.plot_intersecting_points(grid = grid,
#                                                                  avg_neighbor_array = sampling_time_x_converged)
#                        
#                        ## FIGURE FROM 
#                        fig = plot_tools.plot_mayavi_atoms(traj = traj_data.traj,
#                                                atom_index = atom_index,
#                                                frame = 0,
#                                                figure = fig,
#                                                dict_atoms = plot_tools.ATOM_DICT,
#                                                dict_colors = plot_tools.COLOR_CODE_DICT)

                        #%%
                        

                        
                        
    
    #%% COMPUTING SAMPLING TIME

#        ## DEFINING EACH
#        sims_dict = {
#                'Planar':{
#                        'desired_sims': {
#                            'type'   : ['planar_spr_50'], 
#                            'ligand' : ligands,
#                            },
#                        'desired_folders': ['Planar_SPR_50', ]
#                        },
#                'GNP': {
#                        'desired_sims': {
#                            'type'   : ['gnp_spr_50'], 
#                            'ligand' : ligands,
#                            }, # GNP_SPR_600_extended
#                        'desired_folders': ['GNP_SPR_50' ],
#                        },
#    
#                        }
#        ## DEFINING NEIGHBOR PICKLE
#        neighbor_pickle_name = r"0-50000.pickle"
#        
#        ## DEFINING EACH
#        sims_dict = {
#                'Planar':{
#                        'desired_sims': {
#                            'type'   : ['planar_spr_600'], 
#                            'ligand' : ligands,
#                            },
#                        'desired_folders': ['Planar_SPR_600_extended', ]
#                        },
##                'GNP': {
##                        'desired_sims': {
##                            'type'   : ['gnp_spr_50'], 
##                            'ligand' : ligands,
##                            }, # GNP_SPR_600_extended
##                        'desired_folders': ['GNP_SPR_50' ],
##                        },
#    
#                        }
#        ## DEFINING NEIGHBOR PICKLE
#        neighbor_pickle_name = r"0-100000.pickle"
#        # r"0-50000.pickle"
#        # 
#        
#        
#        ################
#        ### DEFAULTS ###
#        ################
#        ## DEFINING STYLES
#        line_dict={
#                "linestyle": '-',
#                "linewidth": 2,
#                "alpha": 1,
#                "color" :'black',
#                }
#        
#        ## DEFINING FRAME RATE
#        frame_rate = 2000
#        
#        ## DEFINING CONVERGENCE TYPE
#        # convergence_type = "percent_error"
#        convergence_type = "value"
#        
#        ## DEFINING PERCENT ERROR
#        percent_error_list = [2]
#        # [2.5, 3] # 2, 1
#        for percent_error in percent_error_list:
#            
#            ## LOOPING
#            for each_key in sims_dict:
#                desired_sims = sims_dict[each_key]['desired_sims']
#                desired_folders = sims_dict[each_key]['desired_folders']
#            
#                ## GETTING MAIN PATH LIST
#                main_path_list = get_all_paths(main_path = MAIN_SIM_FOLDER, 
#                                               desired_folders = desired_folders, 
#                                               folder_dict = DIRECTORY_DICT )
#            
#                ## TRACTING DIRECTORIES
#                directories = track_directories( main_path_list = main_path_list )
#                
#                ## GETTING DATAFRAMES THAT YOU DESIRE
#                df = directories.get_desired_sims(desired_sims = desired_sims)
#                
#                ## GETTING PATH
#                path_compute_neighbors = [ os.path.join(each_path, relative_path, compute_path) for each_path in list(df['PATH']) ]
#            
#                ## LOOPING
#                for specific_path in path_compute_neighbors:
#                    
#                    ## FINDING FIG NAME
#                    current_name = os.path.basename(os.path.dirname(os.path.dirname(specific_path)))
#                    print("Working on: %s"%(current_name))
#                    
#                    ## LOADING PICKLE
#                    hydration_map = extract_hydration_maps()
#                    neighbor_list = hydration_map.load_neighbor_values(main_sim_list = [specific_path],
#                                                                       pickle_name = neighbor_pickle_name)
#                
#                    ## GETTING NEIGHBORS ARRAY
#                    num_neighbors_array = neighbor_list[0]
#                            
#                    ## GENERATING DEBUGGING MU CLASS
#                    mu_debug = debug_mu_convergence()
#                
#                    ## COMPUTING CONVERGENCE TIME
#                    mu_storage_reverse, frame_list = mu_debug.compute_mu_convergence_time(num_neighbors_array = num_neighbors_array,
#                                                                                          frame_rate = frame_rate,
#                                                                                          want_reverse = True)
#                    
#                    ## GETTING CONVERGENCE INFORMATION
#                    theoretical_bounds, index_converged, sampling_time_x_converged, x \
#                                    = mu_debug.main_compute_sampling_time_from_reverse(mu_storage_reverse = mu_storage_reverse,
#                                                                                       frame_list = frame_list,
#                                                                                       percent_error = percent_error,
#                                                                                       convergence_type = convergence_type,
#                                                                                       )
#                    ## PLOTTING CONVERGENCE
#                    fig, ax = mu_debug.plot_convergence_of_mu( mu_storage_reverse = mu_storage_reverse,  
#                                                               x = x,
#                                                               grid_index = 0,
#                                                               theoretical_bounds = theoretical_bounds,
#                                                               index_converged = index_converged
#                                                               )
#                    
#                    ## STORING FIGURE
#                    figure_name = current_name + "converg_of_mu" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
#                    plot_tools.store_figure( fig = fig,
#                                             path = os.path.join(STORE_FIG_LOC,
#                                                                 figure_name),
#                                             save_fig = SAVE_FIG,
#                                             )
#            
#                    
#                    ## PLOTTING SAMPLING TIME VERSUS GRID
#                    fig, ax = mu_debug.plot_sampling_time_vs_grid_index(sampling_time_x_converged)
#                    
#                    ## STORING FIGURE
#                    figure_name = current_name + "sampling_vs_grid" + "_" + convergence_type + "_" + str(percent_error) + "_perc"
#                    plot_tools.store_figure( fig = fig,
#                                             path = os.path.join(STORE_FIG_LOC,
#                                                                 figure_name),
#                                             save_fig = SAVE_FIG,
#                                             )
#            
        

#    #%%
#    ## TOTAL FRAME FOR CONVERGENCE
#    frame_for_convergence = max_sampling_time
#    # 40000
#    
#    ## INCREMENTS
#    inc = 1000
#    
#    ## COMPUTING EQUILIBRATION TIME
#    mu_storage, frame_list_equil = compute_mu_equil_time(num_neighbors_array = num_neighbors_array,
#                                                         frame_for_convergence = frame_for_convergence,
#                                                         inc = inc,
#                                                         want_reverse = False 
#                                                         )
#    
#    #%%
#    ## GETTING THE DEVIATION
#    mu_storage_deviation = mu_storage
#    
##    ## GETTING THEORETICAL BOUND
##    theoretical_bounds =  [ find_theoretical_error_bounds(value = each_value, percent_error = percent_error) 
##                            for each_value in mu_storage_deviation[-1] ]
##    
#    ## GETTING CONVERGENCE
#    theoretical_bounds, index_converged = compute_bounds_for_convergence_mu(mu_storage,
#                                                                            percent_error = percent_error)
#    
#    
#    # mu_storage - mu_storage[-1]
#    
#    ## PLOTTING
#    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
#    
#    ## ADDING AXIS LABELS
#    ax.set_xlabel("Initial frame (%d intervals)"%(frame_for_convergence) )
#    ax.set_ylabel("$\mu$ (kT)")
#    
#    ## PLOTTING FOR A SINGLE GRID
#    grid_index = 10
#    # for grid_index in range(len(mu_storage[0])):
#    x = [ each_array[0] for each_array in frame_list_equil]
#    y = [ mu_storage_deviation[idx][grid_index] for idx in range(len(mu_storage_deviation))]
#    
#    ## PLOTTING BOUNDS
#    ax.axhline(y = theoretical_bounds[grid_index][0][0], linestyle = '--')
#    ax.axhline(y = theoretical_bounds[grid_index][0][1], linestyle = '--')
#    
#    ## GETTING INDEX CLOSEST TO THEORETICAL BOUND
#    index = index_converged[grid_index]
#    
#    ## GETTING OPTIMAL
#    opt_x = x[index]
#    opt_y = y[index]
#    
#    ## PLOTTING X
#    ax.axvline(x = opt_x, linestyle = '--', color = 'r')
#    
#    ## SETTING AXIS
#    # ax.set_ylim([-5, 5])
#    
#    ## PLOTTING
#    ax.plot(x,y, color='k')
#
#    ## FITTING
#    fig.tight_layout()
#    

    
    
    
