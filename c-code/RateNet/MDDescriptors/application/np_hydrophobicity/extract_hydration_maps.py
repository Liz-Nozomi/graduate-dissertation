# -*- coding: utf-8 -*-
"""
extract_hydration_maps.py
The purpose of this script is to extract the hydration maps. We will make 
several plots that could assist in understanding what these hydration maps 
mean. 

Written by: Alex K. Chew (01/06/2020)
"""

## IMPORTING TOOLS
import os
import numpy as np
import glob as glob
## IMPORTING PANDAS
import pandas as pd
## IMPORTING MATPLOTLIB
import matplotlib.pyplot as plt

## CHECK PATH
from MDDescriptors.core.check_tools import check_path, check_dir

## PICKLE FUNCTIONS
from MDDescriptors.core.pickle_tools import load_pickle_results, pickle_results

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## IMPORTING CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## IMPORTING GLOBAL VARS
from MDDescriptors.surface.willard_chandler_global_vars import MAX_N, MU_MIN, MU_MAX
from MDDescriptors.surface.core_functions import get_x_y_from_p_N
## IMPORTING NUM DIST FUNCTION
from MDDescriptors.surface.generate_hydration_maps import compute_num_dist

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
    
##############################################
### PLOTTING FUNCTIONS
##############################################

## PLOTTING HISTOGRAM
def plot_histogram_for_prob_dist( unnorm_p_N, 
                                  grid_index=0,
                                  line_dict = {},
                                  normalize = True,
                                  fig = None,
                                  ax = None,
                                  ):
    '''
    The purpose of this function is to plot the probability distribution for a
    given grid point. 
    INPUTS:
        unnorm_p_N: [np.array, shape=(num_grid, num_occurances)]
            unnormalized number distribution
        grid_index: [int]
            grid index that you are interested in
        normalize: [logical, default=True]
            True if you want to normalize by dividing the total sum
        fig: [default=None]
            Figure object. If available, we will just add to the figure
        ax: [default=None]
            Axis object. If available, we will just add to the figure
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''
    ## DEFINING DISTRIBUTION
    dist = unnorm_p_N[grid_index]
    ## NORMALIZING
    if normalize is True:
        dist = dist / dist.sum()
    
    ## DEFINING DISTRIBUTION
    xs = np.arange(len(dist))
    
    ## CREATING PLOT
    if fig is None or ax is None:
        ## CREATING PLOT
        fig, ax = plot_funcs.create_plot()
        ## ADDING LABELS
        ax.set_xlabel("Number within grid point")
        ax.set_ylabel("Probability density function")
    
    ## CREATING HISTOGRAM PLOT
    ax.plot(xs, dist, **line_dict)
    return fig, ax
    
### FUNCTION TO GET SHAPIRO
def get_shapiro(data, alpha=0.05, verbose = False):
    '''
    The purpose of this function is to get the shapiro value. p values that 
    are greater than 0.05 are approximately gaussian. Otherwise, it does not 
    look Gaussian.
    
    INPUTS:
        data: [np.array]
            data that you want to test if Gaussian
        alpha: [float]
            confidence value you want to test
        verbose: [logical]
            True if you want verbose
    OUTPUTS:
        stat, p: outputs from shapiro
    '''

    # REF: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    # REF2: https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
    ## TESTING THE SHAPIRO-WILK TEST
    from scipy.stats import shapiro
    
    ## GENERATING TEST
    stat, p = shapiro(data)
    
    ## INTERPRETING
    if verbose is True:
        ## DEFINING STATISTIC
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
        	print('Sample looks Gaussian (fail to reject H0)')
        else:
        	print('Sample does not look Gaussian (reject H0)')
    return stat, p

### FUNCTION TO GET DISTRIBUTION FOR A SINGLE INDEX
def plot_gaussian_fit_to_find_mu(unnorm_p_N,
                                 x = None,
                                 y = None,
                                 mu = None,
                                 p_func = None,
                                 grid_index=0,
                                 d = MAX_N,
                                 want_shapiro=True,
                                 fig = None,
                                 ax = None,):
    '''
    INPUTS:
        unnorm_p_N: [np.array, shape=(num_grid, num_occurances)]
            unnormalized number distribution
        grid_index: [int]
            grid index that you are interested in
        d: [int]
            maximum
    OUTPUTS:
        fig, ax:
            figure and axis objects
    '''
    ## IMPORTING FUNCTION
    from matplotlib.offsetbox import AnchoredText

    ## DEFINING PROBABILITY
    p_N = unnorm_p_N[grid_index]
    
    ## GETTING VALUES
    if x is None or y is None:
        x, y, p, p_func = get_x_y_from_p_N(p_N = p_N,
                                           d = d)
        
    if mu is None:
        mu = p_func(0)
        
    ## CREATING PLOT
    if fig is None or ax is None:
        fig, ax = plot_funcs.create_plot()
        ## ADDING LABELS
        ax.set_xlabel("Non-negative occurances")
        ax.set_ylabel("-log(P(N))")
    
    ## CREATING HISTOGRAM PLOT
    ax.plot(x, y, '.-', color="k")
    
    ## GETTING X IMITS
    x_limits = ax.get_xlim()
    
    ## DEFINING LINESPACE
    x_lin = np.linspace(0,x_limits[1],100)
    ax.plot(x_lin, p_func(x_lin), '--', color="blue")
    
    ## ADDING TEXT
    ax.text(x_lin[0], p_func(x_lin[0]), "%.2f"%(mu), 
            horizontalalignment='center',
            verticalalignment='bottom') 
    ## ADDHING SHAPIRO R
    if want_shapiro is True:
        ## GETTING SHAPIRO
        stats, p = get_shapiro( data = y )
        
        ## CREATING BOX TEXT
        box_text = "%s: %.2f"%( "Shapiro-p", p,
                                ) 
        ## ADDING TEXT BOX
        text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax.add_artist(text_box)
    
    return fig, ax


## FUNCTION TO GET SHAPIRO P ARRAY FROM PROBABILITY DISTRIBUTION
def compute_shapiro_p_for_p_N( unnorm_p_N, d = MAX_N ):
    '''
    The purpose of this function is to compute the shapiro p values across 
    probability distribution functions.
    INPUTS:
        unnorm_p_N: [np.array, shape=(num_grid, num_occurances)]
            unnormalized probability distribution function
        d: [int]
            max number of occurances
    OUTPUTS:
        shapiro_p_array: [np.array, shape=(num_grid)]
    '''

    ## GENERATING ARRAY
    shapiro_p_array=np.zeros(len(unnorm_p_N))
    
    ## FINDING ALL P VALUES
    for idx, p_N in enumerate(unnorm_p_N):
        ## COMPUTING P VALUES
        x, y, p, p_func = get_x_y_from_p_N(p_N = p_N,
                                           d = d)
        ## GETTING SHAPIRO P
        stats, p = get_shapiro( data = y )
        
        ## STORING
        shapiro_p_array[idx] = p
    return shapiro_p_array

### FUNCTION TO PLOT
def plot_scatter_shapiro_p_vs_grid_index(shapiro_p_array,
                                         threshold=0.05):
    '''
    The purpose of this function is to plot the shapiro versus grid index
    INPUTS:
        shapiro_p_array: [np.array]
            shapiro p values
        threshold: [float]
            threshold below which is not a Gaussian distribution
    OUTPUTS:
        fig, ax: figure and axis for plot
    '''    
    ## CREATING PLOT
    fig, ax = plot_funcs.create_plot()
    
    ## DEFINING GRID INDEXES
    grid_index = np.arange(0, len(shapiro_p_array))
    
    ## ADDING LABELS
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Shapiro P")
    
    ## FINDING ALL BELOW A THRESHOLD
    idx_below_threshold = shapiro_p_array<=threshold 
    idx_above_threshold = np.logical_not(idx_below_threshold)
    ## CREATING HISTOGRAM PLOT
    ax.scatter(grid_index[idx_above_threshold], 
               shapiro_p_array[idx_above_threshold], 
               color="k")

    ## CREATING HISTOGRAM PLOT
    ax.scatter(grid_index[idx_below_threshold], 
               shapiro_p_array[idx_below_threshold], color="r")
    
    ## ADDING Y = THRESHOLD LINE
    ax.axhline(y=threshold, linestyle = '--', color='black')
    
    ## DEFINING TEXT
    plot_text="Num below %.2f: %d"%(threshold, np.sum(idx_below_threshold))
    
    ## ADDING TEXT TO FIGURE
    plt.text(0.005, 0.990, plot_text,
              ha='left', va='top',
              transform=ax.transAxes)
    
    ## SETTING Y LIMITS
    ax.set_ylim([0,1])
    return fig, ax





#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    MAIN_SIM_FOLDER=r"S:\np_hydrophobicity_project\simulations"
    
    ## DEFINING DIRECTORIES
    DIRECTORY_DICT = {
            'Planar_sims': '20200114-NP_HYDRO_FROZEN',
            'NP_sims': 'NP_SPHERICAL_REDO',
            'Planar_sims_unfrozen': 'PLANAR',
            'GNP_SPR_600': '20200215-GNP_spring_const',
            'Planar_SPR_600': '20200215-planar_SAM_frozen_600spring',
            }
    
    ## DEFINING FIGURE INFORMATIONS
    save_fig = True
    store_fig_loc = r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200217\images\hydration_maps"
    
    ## DEFINING CONTOUR DICT
    CONTOUR_DICT = {
            'water_only': "25.6-0.24-0.1,0.1,0.1-0.33-HOH",
            'with_counterions': "25.6-0.24-0.1,0.1,0.1-0.33-HOH_CL_NA",
            'heavy_atoms': "25.6-0.24-0.1,0.1,0.1-0.33-all_heavy",
            }
    
    ## DEFINING RELATIVE PATH TO PICKLE
    relative_pickle_to_neighbors = os.path.join( "compute_neighbors",
                                                 "0-50000.pickle")
    
    ## DEFINING MU PICKLE
    mu_pickle_name = "mu.pickle"
    
    ## DEFINING SUFFIX
    folder_suffix = "25.6-0.24-0.1,0.1,0.1-0.33-HOH_CL_NA"
    # folder_suffix = "-0.24-0.1,0.1,0.1-0.33-HOH"
    
    
    ## CHECKING PATH
    MAIN_SIM_FOLDER = check_path(MAIN_SIM_FOLDER)
    
    

    ## DEFINING JOB TYPE
    job_type="np_charge"
    
    '''
    contour_comparison: contour comparison
    charge_comparison: charge comparison
    
    '''
    ## DEFINING SPECIFIC SIMULATION
    simulation_location = "PLANAR"
    if job_type == "np_contour_comparison" or job_type == "np_charge":
        simulation_location = "NP_SPHERICAL_REDO"
    
    #%% CHARGED SIMS
    
    ## DEFINING DESIRED FOLDERS
    desired_folders = [ 'Planar_sims' ]
    
    ## GETTING MAIN PATH LIST
    main_path_list = get_all_paths(main_path = MAIN_SIM_FOLDER, 
                                   desired_folders = desired_folders, 
                                   folder_dict = DIRECTORY_DICT )
    
    ## DEFINING DESIRED DICTIONARY
    desired_sims = {
            'type'   : [ 'planar' ],
            'ligand' : [ "C11COO", "C11NH3" ],
            }
    
    ## TRACTING DIRECTORIES
    directories = track_directories( main_path_list = main_path_list )
    
    ## GETTING DATAFRAMES THAT YOU DESIRE
    df = directories.get_desired_sims(desired_sims = desired_sims)
    
    ## DEFINING PATH TO NEIGHBOR LIST
    neighbor_list_path = {each_key: [os.path.join(each_path, CONTOUR_DICT[each_key]) 
                            for each_path in df['PATH'].to_list() ] 
                            for each_key in CONTOUR_DICT}
    
    ## DEFINING WAY TO COMPUTE HYDRATION MAPS
    hydration_map = extract_hydration_maps()
    
    ## DEFINING NEIGHBOR LIST TYPE
    neighbor_list_type = 'with_counterions'
    
    ## LOADING EACH TYPE
    ## GETTING ALL NEIGHBORS LIST
    neighbor_list = [ hydration_map.load_neighbor_values(main_sim_list = neighbor_list_path[neighbor_list_type],
                                                       pickle_name = relative_pickle_to_neighbors)
                                for neighbor_list_type in neighbor_list_path.keys() ]
    
    ## GETTING NUMBER DISTRIBUTION
    neighbor_list_num_dist = [[ compute_num_dist(num_neighbors_array = num_neighbors_array,
                                                max_neighbors = MAX_N) for num_neighbors_array in neighbor_list[idx]]
                                                    for idx,neighbor_list_type in enumerate(neighbor_list_path.keys())]
    
    #%% GENERATING PLOTS

    ## DEFINING STYLES
    line_dict={
            "linestyle": '-',
            "linewidth": 2,
            "alpha": 1,
            "color" :'black',
            }
    
    ## DEFINING SHAPIRO THRESHOLD
    shapiro_threshold=0.05
    
#    ## LOOPING THROUGH EACH PATH
#    for idx,neighbor_list_type in enumerate(neighbor_list_path.keys()):
#        ## LOOPING THROUGH EACH DATAFRAME
#        for each_index in range(len(df)):
#            
    # neighbor_list_type = 'with_counterions'
    idx = list(neighbor_list_path.keys()).index('with_counterions')
    each_index = 1

    ## CLOSING ALL
    plt.close('all')
    ## GETTING NAME PREIFX
    fig_name_prefix = neighbor_list_type + '-' + os.path.basename(df.iloc[each_index]['PATH'])
    ## DEFINING UNNORM P_N
    unnorm_p_N = neighbor_list_num_dist[idx][each_index]
    ## GETTING HISTOGRAM FOR PROBABILITY DISTRIBUTION
    fig, ax = plot_histogram_for_prob_dist( unnorm_p_N = unnorm_p_N, 
                                            grid_index=0,
                                            line_dict = line_dict,
                                            normalize = False,
                                            fig = None,
                                            ax = None,
                                            )
    
    ## STORING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(store_fig_loc,
                                                 fig_name_prefix + "_prob_dist"),
                             save_fig = save_fig
                             )
    
    ## GETTING SHAPIRO ARRAY
    shapiro_p_array = compute_shapiro_p_for_p_N(unnorm_p_N)
    
    ## PLOTTING SCATTER
    fig, ax = plot_scatter_shapiro_p_vs_grid_index(shapiro_p_array = shapiro_p_array,
                                                   threshold = shapiro_threshold)
    
    ## STORING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(store_fig_loc,
                                                 fig_name_prefix + "_shapiro"),
                             save_fig = save_fig
                             )
    
    
    
    ## GETTING ALL INDEXES THAT ARE BELOW THRESHOLD
    below_threshold =np.where( shapiro_p_array < shapiro_threshold )[0]
    
    ## DEFININGI NDEX
    lowest_shapiro_idx=np.argmin(shapiro_p_array)
    
    ## PLOTTING HISTOGRAM PROBABILITY DISTRIBUTION
    fig, ax = plot_gaussian_fit_to_find_mu(unnorm_p_N = unnorm_p_N,
                                           grid_index = lowest_shapiro_idx
                                           )
    
    ## STORING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(store_fig_loc,
                                                 fig_name_prefix + "_lowest_shapiro"),
                             save_fig = save_fig
                             )
            
    
    #%% NORMAL PLANAR SIMS
    
    ## DEFINING DESIRED FOLDERS
    desired_folders = [ 'Planar_sims' ]
    
    ## GETTING MAIN PATH LIST
    main_path_list = get_all_paths(main_path = MAIN_SIM_FOLDER, 
                                   desired_folders = desired_folders, 
                                   folder_dict = DIRECTORY_DICT )
    
    ## DEFINING DESIRED DICTIONARY
    desired_sims = {
            'type'   : [ 'planar' ],
            'ligand' : [ "dodecanethiol" ,"C11CF3", "C11CONH2", "C11COOH", "C11NH2", "C11OH"],
            }
    
    ## TRACKING DIRECTORIES
    directories = track_directories( main_path_list = main_path_list )
    
    ## GETTING DATAFRAMES THAT YOU DESIRE
    df = directories.get_desired_sims(desired_sims = desired_sims)
    
    ## DEFINING PATH TO NEIGHBOR LIST
    neighbor_list_path = {each_key: [os.path.join(each_path, CONTOUR_DICT[each_key]) 
                            for each_path in df['PATH'].to_list() ] 
                            for each_key in CONTOUR_DICT}
    
    ## DEFINING WAY TO COMPUTE HYDRATION MAPS
    hydration_map = extract_hydration_maps()
    
    ## DEFINING NEIGHBOR LIST TYPE
    neighbor_list_type = 'heavy_atoms'
    
    ## LOADING MU VALUES
    mu_value_array = hydration_map.load_mu_values(main_sim_list = neighbor_list_path[neighbor_list_type] ,
                                                       pickle_name = mu_pickle_name) 
    
    
    
    plot_scatter_mu_values(mu_value_array[0])
    
    
    #%%
    
    p_N = unnorm_p_N[lowest_shapiro_idx]
    
    ## NORMALIZING P(N) DISTRIBUTION
    norm_p = p_N / p_N.sum()
    ## GETTING X VALUES THAT ARE GREATER THAN 0
    x = np.arange( 0, MAX_N, 1 )[ norm_p > 0. ]
    
    
    #%%
    
    ## GETTING SHAPIRO P
    # stats, p = get_shapiro( data = np.arange(0,5000) )
    
    
    
    
    #%%
    ## LOCATING THE DIRECTORY
    
    
    
    ## LOCATING ALL SIMULATIONS
    
   
    
    
    
    ## DEFINING SPECIFIC DIRECTORIES
    
    
    
    
    
    ## DEFINING SPECIFIC SIM
#    specific_sim="MostlikelynpNVT_EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1"
#    
#    
#    ## FULL PATH
#    path_to_sim = os.path.join(MAIN_SIM_FOLDER,
#                               simulation_location,
#                               specific_sim
#                               )
#    
#    ## PATH TO MU
#    path_to_mu = os.path.join(path_to_sim,
#                              '28.8' + folder_suffix,
#                              mu_pickle_name
#                              )
#    
#    ## LOADING MU
#    mu_values = load_pickle_results(path_to_mu)[0][0]
#    
#    #%%
#    ## LOADING EACH CONTOUR
#    path_to_neighbor = os.path.join(path_to_sim,
#                                    '20.8' + folder_suffix,
#                                    relative_pickle_to_neighbors
#                                    )
#    
#    ## LOADING PICKLE
#    neighbors_array = load_pickle_results(path_to_neighbor)[0][0]
#    
    
    
    
    #%%
    if job_type == "contour_comparison":
        ## DEFINING SPECIFIC SIMS
        specific_sim_list = ["FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps",
                             "FrozenGoldPlanar_300.00_K_C11OH_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"]
    elif job_type == "charge_comparison":
        specific_sim_list = ["FrozenGoldPlanar_300.00_K_C11COO_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps",
                             "FrozenGoldPlanar_300.00_K_C11NH3_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"]
    elif job_type == "np_contour_comparison":
        specific_sim_list = ["MostlikelynpNVT_EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1",
                             "MostlikelynpNVT_EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1_likelyindex_1"]
        
    ## DEFINING A WAY TO STORE MU VALUES
    avg_mu_list_storage = []
    
    ## LOPOING THROUGH EACH SIM
    for specific_sim in specific_sim_list:

        ## FULL PATH
        path_to_sim = os.path.join(MAIN_SIM_FOLDER,
                                   simulation_location,
                                   specific_sim
                                   )
        
        ## CREATING MU LIST
        mu_list = []
        ## LOOPING THROUGH EACH CONTOUR 
        for each_contour in contours:
            
            ## PATH TO MU
            path_to_mu = os.path.join(path_to_sim,
                                      str(each_contour) + folder_suffix,
                                      mu_pickle_name
                                      )
            
            ## LOADING MU
            mu_values = load_pickle_results(path_to_mu)[0][0]
            
            ## APPENDING
            mu_list.append(mu_values)
        
            ''' METHOD FOR LOADING ALL NEIGHBORS
            ## LOADING EACH CONTOUR
            path_to_neighbor = os.path.join(path_to_sim,
                                            "25.6" + folder_suffix,
                                            relative_pickle_to_neighbors
                                            )
            
            ## LOADING PICKLE
            neighbors_array = load_pickle_results(path_to_neighbor)[0][0]
            '''
        
        #%%

        
        ## GETTING AVERAGE MU VALUE
        avg_mu_list = np.array([each_mu_array.mean() for each_mu_array in mu_list])
        
        ##  STORING
        avg_mu_list_storage.append(avg_mu_list)
    
    ## CREATING PLOT
    fig, ax = plot_funcs.create_plot()
    
    ## ADDING LABELS
    ax.set_xlabel("Contour levels")
    ax.set_ylabel("Average mu value")
    
    ## LOOPING THROUGH EACH 
    for idx, specific_sim in enumerate(specific_sim_list):
    
        ## EXTRACTING NAME
        name_dict=  extract_most_likely_name(specific_sim)
        
        ## CREATING HISTOGRAM PLOT
        line_plot = ax.plot(contours, 
                            avg_mu_list_storage[idx], 
                            linestyle = "-", 
                            marker = 'o', 
                            # color="k",
                            label = name_dict['ligand']
                            )
    
    ## ADDING LEGEND
    ax.legend()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

