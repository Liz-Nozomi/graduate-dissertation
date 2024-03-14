#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_np_cosolvent_mapping_extract.py
This script extracts the cosolvent mapping information. 

Written by: Alex K. Chew (04/21/2020)

"""

## IMPORTING MODULES
import os
import glob
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools
import MDDescriptors.core.plot_tools as plot_funcs
## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools
## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
SAVE_FIG = True
# False

## IMPORTING KDE    
from scipy.stats import kde

store_image_location = "/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200629/images/np_cosolvent_mapping"
# r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_hydrophobicity_manuscript/Figures/svg_output"
# r"/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200504/images"

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## IMPORTING EXTRACT TRAJ FUNCTION
from MDDescriptors.traj_tools.loop_traj_extract import load_results_pickle, ANALYSIS_FOLDER, RESULTS_PICKLE

from MDDescriptors.core.pickle_tools import load_pickle_results, save_and_load_pickle, pickle_results


### IMPORTING GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, NP_SIM_PATH, LIGAND_COLOR_DICT

## MAIN FUNCTION
from MDDescriptors.application.np_hydrophobicity.compute_np_cosolvent_mapping import main_compute_np_cosolvent_mapping, find_original_np_hydrophobicity_name

## IMPORTING NUM DIST FUNCTION
from MDDescriptors.surface.generate_hydration_maps import compute_num_dist

### IMPORTING LIGAND REISDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj, get_atom_indices_of_ligands_in_traj

### FUNCTION TO EXTRACT UNNORMALIZED DISTRIBUTION
def extract_unnormalized_dist(path_sim_list,
                              want_num_dist_storage = False,
                              analysis_folder = ANALYSIS_FOLDER,
                              results_pickle = RESULTS_PICKLE
                              ):
    '''
    The purpose of this function is to extract unnormalized distribution
    INPUTS:
        path_sim_list: [list]
            list of simulation files
        want_num_dist_storage: [logical]
            True if you want number distribution storage
        analysis_folder: [str]
            analysis folder within simulation directory
        results_pickle: [str]
            pickle results
    OUTPUTS:
        unnorm_dict: [dict]
            dictionary of unnormalized distributions
        mapping_list: [list]
            list of the mapping class
        num_dist_storage: [list]
            list of normalized distributions stored
            
    '''

    ## STORING NUM DIST
    if want_num_dist_storage is True:
        num_dist_storage = []

    unnorm_dict = {}
    mapping_list = []
    
    ## DEFINING SIMULATION LIST
    for sim_idx, path_to_sim in enumerate(path_sim_list):
        print("Working on index: %d"%(sim_idx))
        
        ## PATH TO ANALYSIS
        path_to_pickle=os.path.join(path_to_sim,
                                    analysis_folder,
                                    main_compute_np_cosolvent_mapping.__name__,
                                    results_pickle)
        
        ## LOADING RESULTS
        num_neighbors_array, mapping = load_pickle_results(file_path = path_to_pickle,
                                                           verbose = True)[0]
    
        ## LOOPING THROUGH SOLVENT LIST
        solvent_list = mapping.solvent_list
        
        ## APPENDING MAP
        mapping_list.append(mapping)
            
        ## CREATING EMPTY DICTIONARY
        if want_num_dist_storage is True:
            num_dist_stor = {}
 
        ## LOOPING THROUGH SOLVENT LIST
        for idx, solvent_resname in enumerate(solvent_list):
            ## GETTING NEIGHBORS
            neighbors_array = num_neighbors_array[:,idx, :].T
            num_dist = compute_num_dist(num_neighbors_array = neighbors_array,
                                        max_neighbors = N_MAX)
            
            ## STORING NUM DIST
            if want_num_dist_storage is True:
                num_dist_stor[solvent_resname] = num_dist[:]
            
            ## ADDING FOR THE INITIAL SOLVENTS
            if solvent_resname not in unnorm_dict:
                unnorm_dict[solvent_resname] = num_dist[:]
            else:
                unnorm_dict[solvent_resname] = unnorm_dict[solvent_resname][:] + num_dist[:]
        ## STORING NUMB DISTRIBUTION
        num_dist_storage.append(num_dist_stor)
        
    if want_num_dist_storage is True:
        return unnorm_dict, mapping_list, num_dist_storage
    else:
        return unnorm_dict, mapping_list

### FUNCTION TO COMPUTE E(X)
def compute_e_x(unnorm_dict):
    '''
    The purpose of this function is to compute E(x) for each grid 
    point. The idea is that E(x) = sum(x_i * P_xi)). If P_xi is zero, then 
    that particular xi contributes nothing towards the final value. 
    
    INPUTS:
        unnorm_dict: [dict]
            dictionary unnormalized distribution
    OUTPUTS:
        e_x_storage: [dict]
            dictionary of E(X) value
        p_n_storage: [dict]
            dictionary of P(N), normalized probabiilty distribution function.
    '''
    ## DEVELOPING N ARRAY
    n_array = np.arange(N_MAX)
    
    ## COMPUTING P_N
    p_n_storage = {}
    e_x_storage = {}
    for each_key in unnorm_dict:
        norm_dist = unnorm_dict[each_key] / unnorm_dict[each_key].sum(axis=1)[:, np.newaxis]
        p_n_storage[each_key] = norm_dist[:]
        e_x_storage[each_key] = np.sum(norm_dist * n_array,axis=1)
        
    return e_x_storage, p_n_storage

### FUNCTION TO PLOT EX VS MU
def plot_ex_vs_mu(mu_array,
                  e_x_values,
                  color = 'k',
                  lower_e_x_value = 0,
                  want_legend = True,
                  want_fitted_line = True,
                  label = None,
                  fig = None,
                  ax = None,
                  alpha=0.5):
    '''
    This function simply plots E(N) versus mu for each grid point.
    INPUTS:
        mu_array: [np.array]
            mu values
        e_x_values: [np.array]
            E(N) values
        color: [str]
            color for the plot
        lower_e_x_value: [float]
            lower bound for E(x)
        want_legend: [logical]
            True if you want water
        fig, ax:
            figure / axis for object
        alpha: [float]
            transparency value
    OUTPUTS:
        fig, ax:
            figure and axis fro the image
    '''


    ## CREATING PLOT
    if fig is None or ax is None:
        
        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
        ## ADDING LABELS
        ax.set_xlabel("$\mu$ (kT)")
        ax.set_ylabel("E(N)")
        
    ## INDEXES GREATER THAN ZERO
    indexes_greater_than_zero = e_x_values >= lower_e_x_value
    
    x = mu_array[indexes_greater_than_zero]
    y = e_x_values[indexes_greater_than_zero]
    
    ## PLOTTING
    ax.plot(x, y, '.', 
            alpha=0.5,
            color=color,
            label = label)
    
    ## FITTING TO LINE
    if want_fitted_line is True:
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
    
    ## ADDING LEGEND
    if want_legend is True:
        ax.legend()
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax


### FUNCTION TO COMPUTE FRACTION OF OCCURENCES
def compute_frac_occurrences(num_neighbors_array, mapping):
    '''
    This function computes the fraction of occurences for a given neighbors 
    array and mapping (which contains the solvent list)
    INPUTS:
        num_neighbors_array: [np.array, shape = (num_frames, num_solvents, num_grid_pts)]
            number of neighbors array for a given raction occurence
        mapping: [obj]
            mapping object containing all information neighbors array
    OUTPUTS:
        storage_dict: [dict]
            dictionary with keys of the solvent list. Each dictionary contains 
            the number of occurences per grid point, fraction of occurences, and 
            number of frames
    '''

    ## LOOPING THROUGH SOLVENT LIST
    solvent_list = mapping.solvent_list
    
    ## DEFINING STORAGE DICT
    storage_dict = {}
    
    ## LOOPING THROUGH SOLVENT LIST
    for idx, solvent_resname in enumerate(solvent_list):
        ## GETTING NEIGHBORS
        neighbors_array = num_neighbors_array[:,idx, :].T
        ## SHAPE: NUM GRIDS, NUM FRAMES
        num_grids, num_frames = neighbors_array.shape
        ## GETTING FRACTION ACROSS TIME
        num_occur = np.sum(neighbors_array > 0, axis = 1)
        num_occur_across_time = (neighbors_array.T>0).astype('int')
        frac_occur = num_occur / num_frames
        ## STORING
        storage_dict[solvent_resname] = {
                'num_occur': num_occur,
                'frac_occur': frac_occur,
                'num_frames': num_frames,
                'num_occur_across_time': num_occur_across_time,
                }

    return storage_dict

### FUNCTION TO EXTRACT LIGAND NAME
def extract_nomenclature(name):
    '''
    This function extracts the nomenclature of the name.
    INPUTS:
        name: [str]
            name of the file. e.g.:
                planar cosolvent mapping: comap_PRO_1_1_NVTspr_50_Planar_300.00_K_C11CONH2_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps
            
    OUTPUTS:
        nomenclature: [dict]
            dictionary of the nomenclature, e.g.
                'ligand': name of the ligand
    '''
    ## SPLIT BY _
    name_split = name.split('_')
    print(name_split)
    
    ## FINDING DIFFERENT TYPES
    if name_split[6] == 'Planar':
        nomenclature = {
                'type': 'Planar',
                'ligand': name_split[9],
                }
    
    return nomenclature

### FUNCTION TO LOAD FRACTION OF OCCURENCES
def load_frac_of_occur(path_sim_list):
    '''
    This function loads the fraction of occurences. 
    INPUTS:
        path_sim_list: [list]
            path to simulation list
    OUTPUTS:
        storage_frac_occurences: [list]
            list of storage dicts for fraciton of occurences
        mapping_list: [list]
            list of mapping class objects
    '''

    ## STORAGE FOR DICTS
    storage_frac_occurences = []
    mapping_list = []
    
    ## LOOPING THROUGH EACH SIM
    for sim_path in path_sim_list:
        ## PATH TO ANALYSIS
        path_to_pickle=os.path.join(sim_path,
                                    ANALYSIS_FOLDER,
                                    main_compute_np_cosolvent_mapping.__name__,
                                    RESULTS_PICKLE)
        
        ## LOADING RESULTS
        num_neighbors_array, mapping = load_pickle_results(file_path = path_to_pickle,
                                                           verbose = True)[0]
    
        ## COMPUTING FRACTION OF OCCURENCES
        storage_dict = compute_frac_occurrences(num_neighbors_array = num_neighbors_array, 
                                                mapping = mapping)
        
        ## APPENDING
        storage_frac_occurences.append(storage_dict)
        mapping_list.append(mapping)
        
    return storage_frac_occurences, mapping_list


### FUNCTION TO SUM FRACTION OF OCCURENCES
def sum_frac_of_occurences(storage_frac_occurences):
    '''
    The purpose of this function is to generate a fraction sim of occurence dictionary 
    by summing all number of occurences and dividing by total number of rames.
    INPUTS:
        storage_frac_occurences: [list]
            list of storage frac occurrences
    OUTPUTS:
        fraction_sim_occur_dict: [dict]
            dictionary with solvents as key. Each key has:
                'num_occur': array with number of occurences
                'num_frames': number of frames for each partition
                'frac_occur': ratio of num_occur to total number of rames
    '''

    ## DEFINING FRACTION SIM DICT
    fraction_sim_occur_dict = {}
    
    ## DEFINING STORAGE
    for idx, current_dict in enumerate(storage_frac_occurences):        
        ## LOOPING THROUGH EACH SOLVENT
        for each_solvent in current_dict:
            ## DEFINING NUM OCCUR
            num_occur = np.array(current_dict[each_solvent]['num_occur'])
            num_frames = current_dict[each_solvent]['num_frames']
            
            ## ADDING KEY
            if each_solvent not in fraction_sim_occur_dict:
                fraction_sim_occur_dict[each_solvent] = {}
                fraction_sim_occur_dict[each_solvent]['num_occur'] = num_occur[:]
                fraction_sim_occur_dict[each_solvent]['num_frames'] = []
            else:
                fraction_sim_occur_dict[each_solvent]['num_occur'] += num_occur[:]
            
            ## STORING 
            fraction_sim_occur_dict[each_solvent]['num_frames'].append(num_frames) 
    ## NORMALIZING EACH DICT
    for each_key in fraction_sim_occur_dict:
        each_dict = fraction_sim_occur_dict[each_key]
        each_dict['frac_occur'] = each_dict['num_occur'] / np.sum(each_dict['num_frames'])
    return fraction_sim_occur_dict

### FUNCTION TO COMPUTE SAMPLING TIME DICT
def generate_sampling_occur_dict(storage_frac_occurences):
    '''
    This function generates the occurences over time.
    INPUTS:
        storage_frac_occurences: [list]
            list of storage frac occurrences
    OUTPUTS:
        sampling_time_occur_dict: [dict]
            dictionary of solvents containing sampling time occurences     
    '''

    ## DEFINING SOLVENT DICT
    sampling_time_occur_dict = {}
    
    ## GETTING FULL ARRAY
    for idx, current_dict in  enumerate(storage_frac_occurences):
        ## LOOPING THROUGH EACH SOLVENT
        for each_solvent in current_dict:
            ## DEFINING NUM OCCUR
            num_occur_across_time = np.array(current_dict[each_solvent]['num_occur_across_time'])
            ## ADDING KEY
            if each_solvent not in sampling_time_occur_dict:
                sampling_time_occur_dict[each_solvent] = num_occur_across_time[:]
            else:
                ## CONCATENATING
                sampling_time_occur_dict[each_solvent] = \
                    np.concatenate( ( sampling_time_occur_dict[each_solvent] ,num_occur_across_time ), axis =0 ) 
    return sampling_time_occur_dict

### FUNCTION TO COMPUTE FRACTION OF OCCURENCES
def compute_frac_of_occur(occur_array):
    '''
    This function computes the fraction of occurences given an occurence array 
    that is of 0, 1 values. 
    INPUTS:
        occur_array: [np.array, shape = (time, num_grids)]
            occurences array
    OUTPUTS:
        frac_array: [np.array, shape=num_grids]
            fractional array for number of grids
    '''
    frac_array = np.sum(occur_array,axis=0) / len(occur_array)
    
    return frac_array

### FUNCTION TO COMPUTE SAMPLING TIME DATAFRAME
def compute_sampling_time_frac_occur(sampling_time_occur_dict,
                                     solvent='HOH',
                                     frame_rate = 500,
                                     ):
    '''
    This function computes the sampling time of occurences.
    INPUTS:
        sampling_time_occur_dict: [dict]
            dictionary containing the sampling time occurences as a function of time.
        solvent: [str]
            solvent name for the key within samplign time occurences dict
        frame_rate: [int]
            frame rate
    OUTPUTS:
        sampling_time_df: [pd.dataframe]
            dataframe containing sampling time for each subset of trajectories
        
        
    '''
    ## DEFINING ARRAY
    num_occur_array = sampling_time_occur_dict[solvent]
    
    ## DEFINING FRAME RATE
    num_frames = len(num_occur_array)
    ## DEFINING SAMPLING TIMES
    sampling_values = np.arange(frame_rate, num_frames, frame_rate)
    
    ## GETTING FRAME LIST
    frame_list = [ np.arange(0, each_final + 1)  for each_final in sampling_values]
    
    ## DEFINING SAMPLING TIME DICT
    sampling_time_dict = []
    
    ## LOOPING THROUGH FRAME LIST
    for idx, each_indices in enumerate(frame_list):
        ## DEFINING CURRENT ARRAY
        current_occurences = num_occur_array[each_indices]
        ## COMPUTING FRACTION
        frac_occur = compute_frac_of_occur(occur_array = current_occurences)
        ## STORING
        sampling_time_dict.append({
                'first': each_indices[0],
                'last': each_indices[-1],
                'frac_occur': frac_occur
                })
        
    ## GETTING DATAFRAME
    sampling_time_df = pd.DataFrame(sampling_time_dict)
    return sampling_time_df

## PLOTTING SAMPLING TIME DF
def plot_sampling_time_df(sampling_time_df,
                          grid_index = 0,
                          x_key = "last",
                          y_key = "frac_occur",
                          fig_size_cm = FIGURE_SIZE,
                          delta = None,
                          convergence_type = None,
                          color = 'k',
                          x_scalar = None,
                          want_desired_y_line = False,
                          fig = None,
                          ax = None):
    '''
    This function plots the sampling time for a single grid index
    INPUTS:
        sampling_time_df: [dataframe]
            pandas dataframe containing fraction of occurences, e.g. 
               first  last                                         frac_occur
            0      0   500  [0.9960079840319361, 0.9940119760479041, 0.998...
            1      0  1000  [0.998001998001998, 0.997002997002997, 0.99900...
        x_key: [str]
            string denoting x key for dataframe
        y_key: [str]
            string denoting y key for dataframe
        delta: [float]
            delta value for finding minimum sampling time
        x_scalar: [float]
            if Not None, we will multipy all x values by this
        convergence_type: [str]
            convergence type, either percent_error or value
    OUTPUTS:
        fig, ax:
            figure and axis 
    '''
    ## CREATING FIGURE
    if fig is None or ax is None:
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    ## LOOPING THROUGH AND GETTING ALL X AND Y VALUES
    x = []
    y = []
    ## LOOPING
    for idx, row in sampling_time_df.iterrows():
        x.append(row[x_key])
        y.append(row[y_key][grid_index])
    
    ## CONVERTING TI NUMPY    
    x = np.array(x)
    
    if x_scalar is not None:
        x = x * x_scalar
    
    ## PLOTTING
    ax.plot(x,y, color ='k')
    
    ## ADDING AXIS
    ax.set_xlabel("Sampling time")
    ax.set_ylabel("Fraction of occurrences")
    
    ## ADDING LINE FOR LAST VALUE
    desired_y = y[-1]
    if want_desired_y_line is True:
        ax.axhline(y = desired_y, color = color, linestyle = '--')
    
    ## GETTING BOUNDS
    if delta is not None and convergence_type is not None:
    
        ## GETTING BOUND
        bounds = calc_tools.find_theoretical_error_bounds(value = desired_y, 
                                                 percent_error = delta,
                                                 convergence_type = convergence_type )
        
        ## GETTING CONVERGENCE INDEX
        index = calc_tools.get_converged_value_from_end(y_array = y, 
                                                        desired_y = desired_y,
                                                        bound = bounds[1],
    #                                                        bound = delta,
                                                        )
        
        ## FILLING
        ax.fill_between(x, bounds[0][0], bounds[0][1], color=color,
                         alpha=0.5)
        
        ## GETTING OPTIMAL
        opt_x = x[index]
        
        ## PLOTTING X
        ax.axvline(x = opt_x, linestyle = '--', color = color)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT POINTS
def plot_3d_points_with_scalar(xyz_coordinates, 
                               scalar_array,
                               points_dict={
                                       'scale_factor': 0.5,
                                       'scale_mode': 'none',
                                       'opacity': 1.0,
                                       'mode': 'sphere',
                                       'colormap': 'blue-red',
                                       'vmin': 9,
                                       'vmax': 11,
                                       },
                                figure = None):
    '''
    The purpose of this function is to plot the 3d set of points with scalar
    values to indicate differences between the values
    INPUTS:
        xyz_coordinates: [np.array]
            xyz coordinate to plot
        scalar_array: [np.array]
            scalar values for each coordinate
        points_dict: [dict]
            dicitonary for the scalars
        figure: [obj, default = None]
            mayavi figure. If None, we will create a new figure
    OUTPUTS:
        figure: [obj]
            mayavi figure
    '''
    ## IMPORTING MLAB
    from mayavi import mlab

    if figure is None:
        ## CLOSING MLAB
        try:
            mlab.close()
        except AttributeError:
            pass
        ## CREATING FIGURE
        figure = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    
    ## PLOTTING POINTS
    points = mlab.points3d(
                    xyz_coordinates[:,0], 
                    xyz_coordinates[:,1], 
                    xyz_coordinates[:,2], 
                    scalar_array,
                    figure = figure,
                    **points_dict
                    )
    return figure, points

### FUNCTION TO PLOT SCALAR WITH MOLECULE
def plot_scalar_with_gnp(fraction_sim_occur_dict,
                         mapping_list,
                         traj_data,
                         desired_solvent = 'HOH',
                         index = 0,
                         frame =0 ):
    '''
    This function plots a scalar value for the grid points 
    INPUTS:
        fraction_sim_occur_dict: [dict]
            dictionary of fraction occurence. This could be replaced with another scalar if needed.
        mapping_list: [list]
            list of mapping information
        desired_solvent: [str]
            desired solvent
        traj_data: [obj]
            trajectory data
        index: [int]
            index for mapping to use
        frame: [int]
            frame to use
    OUTPUTS:
        fig: 
            mayavi figure
    '''
    from mayavi import mlab
    ## GETTING GOLD INDEX    
    au_index = [atom.index for atom in traj_data.topology.atoms if atom.name == 'Au' or atom.name == 'BAu']

    ## GETTING ATOM INDICES AND LIGAND NAME
    ligand_names, atom_index = get_atom_indices_of_ligands_in_traj( traj = traj_data.traj )

    
    ## GETTING MAPPING OBJECT
    current_mapping = mapping_list[index]
    
    ## GENERATING NEW GIRD
    grid = current_mapping.get_new_grid_given_time_index(time_idx = frame)
    
    
    ## DEFINING SCALAR VALUES
    scalar_values = fraction_sim_occur_dict[desired_solvent]['frac_occur']
    
    ## GETTING FIGURE
    figure, points = plot_3d_points_with_scalar(xyz_coordinates = grid,
                                                   scalar_array = scalar_values,
                                                   points_dict={
                                                           'scale_factor': 0.5,
                                                           'scale_mode': 'none',
                                                           'opacity': 0.1,
                                                           'mode': 'sphere',
                                                           'colormap': 'blue-red',
                                                           'vmin': 0,
                                                           'vmax': 1,
                                                           },
                                                  figure = None
                                                  )
    
    
    ## ADDING COLOR BAR
    current_colobar = mlab.colorbar(object = points,
                                    label_fmt="%.1f",
                                    nb_labels=5)
    
    ## PLOTTING LIGANDS
    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                              atom_index = atom_index,
                              frame = frame,
                              figure = figure,
                              dict_atoms = plot_funcs.ATOM_DICT,
                              dict_colors = plot_funcs.COLOR_CODE_DICT)
    
    ## PLOTTING GOLD FIGURE
    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                       atom_index = au_index,
                                       frame = frame,
                                       figure = fig,
                                       dict_atoms = plot_funcs.ATOM_DICT,
                                       dict_colors = plot_funcs.COLOR_CODE_DICT)
        
    return fig

### FUNCTION TO GET MIN SAMPLING TIME
def compute_min_sampling_time(sampling_time_df,
                              delta = 0.05,
                              convergence_type = "value",
                              x_key = 'last',
                              y_key = 'frac_occur'):
    '''
    This function computes the minimum sampling time for each grid point.
    INPUTS:
        sampling_time_df: [dataframe]
            pandas dataframe containing fraction of occurences, e.g. 
               first  last                                         frac_occur
            0      0   500  [0.9960079840319361, 0.9940119760479041, 0.998...
            1      0  1000  [0.998001998001998, 0.997002997002997, 0.99900... 
        delta: [float]
            delta fraction that is used for bound. For instance, if this is 0.05, the
            sampling time will look for minimum time for deviations from +- 0.05.
        x_key: [str]
            string denoting x key for dataframe
        y_key: [str]
            string denoting y key for dataframe
    OUTPUTS:
        stored_min_sampling: [list]
            list of minimum sampling time per grid point
    '''
    ## FINDING TOTAL GRID POINTS
    total_grid_pts = len(sampling_time_df[y_key][0])
    
    ## GETTING OCCURENCE ARRAY AS A NUMPY
    occur_array = np.vstack(sampling_time_df[y_key].to_numpy())
    # Shape: 10 x 10752 or num sampling times, and total grids    
    ## LOOPING
    x = [ row[x_key] for idx, row in sampling_time_df.iterrows() ]
    
    ## STORING BOUNDS
    stored_min_sampling = []
    
    ## LOOPING THROUGH GRIDS
    for each_grid in range(total_grid_pts):
        ## GETTING Y VALUES
        y = occur_array[: , each_grid]
        
        ## DECIDING DESIRED Y
        desired_y = y[-1]
        
        ## GETTING BOUND
        bounds = calc_tools.find_theoretical_error_bounds(value = desired_y, 
                                                 percent_error = delta,
                                                 convergence_type = convergence_type )
        
        ## GETTING CONVERGENCE INDEX
        index = calc_tools.get_converged_value_from_end(y_array = y, 
                                                        desired_y = desired_y,
                                                        bound = bounds[1],
#                                                        bound = delta,
                                                        )
        ## STORING
        stored_min_sampling.append(x[index])
    
    return stored_min_sampling

### FUNCTION TO PLLOT MIN SAMPLING TIME
def plot_min_sampling_time(stored_min_sampling,
                           fig_size_cm = FIGURE_SIZE):
    '''
    This function simply plots the minimum sampling time per grid point.
    INPUTS:
        stored_min_sampling: [list]
            list of minimum sampling time per grid point
        fig_size_cm: [tuple]
            figure size in centimeters
    OUTPUTS:
        fig, ax:
            figure and axis for plot
    '''

    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = fig_size_cm)
    
    ## GETTING X VALUES
    x = np.arange(len(stored_min_sampling))
    
    ## PLOTTING
    ax.scatter(x, stored_min_sampling, color = 'k', s = 5)
    
    ## ADDING AXIS
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Min. sampling time")
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    return fig, ax

### FUNCTION TO PLOT GAUSSIAN KDE OF SCATTER
def plot_gaussian_kde_scatter(fraction_sim_occur_dict,
                              mu_array,
                              figsize,
                              solvent_list = [ 'HOH', 'PRO' ],
                              y_range = (-0.25, 1.25),
                              x_range = (8, 13),
                              nbins = 100,
                              vmin = 0,
                              vmax = 2,
                              cmap = plt.cm.jet,
                              ):
    '''
    This functions plots the kde as a scatter plot
    ## EXAMPLE OF GAUSSIAN KDE
    # https://python-graph-gallery.com/86-avoid-overlapping-in-scatterplot-with-2d-density/
    INPUTS:
        fraction_sim_occur_dict: [dict]
            dictionary of occurences
        mu_array: [float]
            mu array
        figsize: [float]
            fiugre size
        solvent_list: [list]
            list of solvents
        y_range: [tuple]
            lower and upper bound of y range
        x_range: [tuple]
            lower and upper bound of x range
        nbins: [int]
            number of bins
        vmin: [float]
            min of color bar
        vmax: [float]
            max of color bar
        cmap: [obj]
            cmap object
    OUTPUTS:
        fig, ax: [obj]
            figure and axis for kde scatter
    '''
    ## DEFINING X
    x = mu_array

    ## CREATING SUBPLOTS
    fig, axes = plt.subplots(nrows=1, ncols=2,
                             figsize=figsize)
    
    ## DEFINING SOLVENT
    for solvent_idx, desired_solvent in enumerate(solvent_list):
        ## DEFINING AXIS
        ax = axes[solvent_idx] 
        ## GETTING FRAC OCCURANCE
        frac_occur = fraction_sim_occur_dict[desired_solvent]['frac_occur']
        
        ## DEFINING X, Y   

        y = frac_occur
        
        ## DEFINING DATA            
        data = np.vstack([x, y])
        
        k = kde.gaussian_kde(data)
        if y_range is None or y_range is None:
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        else:
            xi, yi = np.mgrid[x_range[0]:x_range[1]:nbins*1j, y_range[0]:y_range[1]:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        cf = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), 
                           shading='gouraud', 
                           cmap=cmap,
                           vmin = vmin,
                           vmax = vmax,
                           )
        ## SETTING LABELX
        ax.set_xlabel("$\mu$ (kT)")
        ax.set_ylabel("f")
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        ## SETTING AXIS LIMITS
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

    ## ADDING COLOR BAR
    fig.subplots_adjust(right=0.85)
    ## third - width
    cbar_ax = fig.add_axes([0.9, 0.2 , 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax)
    return fig, ax

### FUNCTION TO EXTRACTC DETAILS FROM NAME
def extract_nomenclature_cosolvent_mapping(name):
    '''
    This function extracts the nomenclature of cosolvent name, e.g.
    'comap_PRO_1_1_50-EAM_300.00_K_2_nmDIAM_C11double67COOH_CHARMM36jul2017_Trial_1_likelyindex_1'
    INPUTS:
        name: [str]
            name of the coslvent mapping folder
    OUTPUTS:
        output_dict: [dict]
            dictionary with details about the name
    '''
    name_split = name.split("_")
    output_dict = {
            'name': name,
            'ligand': name_split[9],
            'trial': int(name_split[3]),
            }
    
    return output_dict

### FUNCTION TO LOOP THROUGH AND GE ALL PATHS
def get_all_paths(np_parent_dirs,
                  main_dir):
    '''
    This function locates all paths
    '''
    

    ## LOOPING THROUGH EACH
    for idx, parent_dir in enumerate(np_parent_dirs):
        ## GETTING ALL FILES
        all_files = glob.glob( os.path.join(main_dir,
                                            parent_dir) + "/*")
        ## EXRACTING DETAILS
        all_files_basename = [os.path.basename(each) for each in all_files]
        extracted_details = [extract_nomenclature_cosolvent_mapping(each) for each in all_files_basename ]
        
        ## CREATING DATAFRAME
        path_dataframe = pd.DataFrame(extracted_details)
        path_dataframe['path'] = all_files[:]
        
        ## APPENDING
        if idx == 0:
            full_path_dataframe = copy.copy(path_dataframe)
        else:
            full_path_dataframe = full_path_dataframe.append(path_dataframe)
    
    ## SORT BY TRIAL
    full_path_dataframe = full_path_dataframe.sort_values(by="trial")
    return full_path_dataframe

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    main_dir = NP_SIM_PATH
    
    ## DEFINING PARENT DIRECTORIES
    np_parent_dirs =[
             "20200625-GNP_cosolvent_mapping_sims_unsat_branched"
            ]

    ## FINDING ALL PATHS
    full_path_dataframe = get_all_paths(np_parent_dirs = np_parent_dirs,
                                        main_dir = main_dir)
    
    ## DEFINING PARENT WC FOLDER
    parent_wc_folder = "20200618-Most_likely_GNP_water_sims_FINAL"
    
    ## DEFINING LIGAND OF INTEREST
    desired_lig = 'C11double67COOH'
    for desired_lig in np.unique(full_path_dataframe.ligand):
    
        ## GETTING PATH
        path_sim_list = full_path_dataframe.loc[full_path_dataframe.ligand == desired_lig]['path'].to_list()
        
        # #%% GENERATING FRACTION OF OCCURENCES
        
        ## SORTING PATH
        path_sim_list.sort()
        
        ## DEFINING INPUTS
        inputs_frac_occur={'path_sim_list' : path_sim_list}
        
        
        ## EXTRACTING
        storage_frac_occurences, mapping_list = load_frac_of_occur(**inputs_frac_occur)
        
        # #%% SUMMING ALL FRACTION OF OCCURENCES
        
    
        ## GETTING FRACTION OF SIMULATION OCCURENCES
        #    fraction_sim_occur_dict = sum_frac_of_occurences(storage_frac_occurences)
    
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
        
        
        
        # #%% CORRELATING FRACTION OF OCCUPANCY TO MU VALUES
        
        #### LOADING MU VALUES
        
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
        
        
        # #%% CORRELATING MU VALUES
        
    
        ## CREATING NEW COLOR MAP
        tmap = plot_funcs.create_cmap_with_white_zero(cmap = plt.cm.jet,
                                                      n = 100,
                                                      perc_zero = 0.125)

    
        ## GENERATING PLOT
        figsize = plot_funcs.cm2inch(FIGURE_SIZE)
        figsize = (figsize[0]*2, figsize[1])
    
        ## ADDING FIGURE
        fig, ax = plot_gaussian_kde_scatter(fraction_sim_occur_dict = fraction_sim_occur_dict,
                                            mu_array = mu_array,
                                            figsize = figsize,
                                            solvent_list = [ 'PRO', 'HOH' ],
                                            y_range = (-0.2, 1.2),
                                            x_range = (8, 13),
                                            nbins = 100,
                                            vmin = 0,
                                            vmax = 2,
                                            cmap = tmap
                                            )
        
        ## DEFINING FIGURE NAME
        figure_name="%s-gaussian_kde"%(desired_lig)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(store_image_location,
                                         figure_name), 
                     fig_extension = 'png',
                     save_fig = True,
                     )
                     
    
    
    
    #%% LOADING TRAJECTORY
    ## DEFINING INDEX
    index = 0
    frame = 0
    
    ## DEFINING PATH TO SIM
    path_to_sim = path_sim_list[index]
    
    ## DEFINING GRO FILE
    prefix="sam_prod-2000-12000--pbc-mol"
    
    output_gro_file = prefix + ".gro"
    output_xtc_file = prefix + ".xtc"
    
    ## LOADING TRAJECTORY
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = output_gro_file,
                                         xtc_file = output_xtc_file,
                                         discard_overlapping_frames = True,
                                         )
    
    
    
    
    #%% PLOTTING ALL FRACTION OF SIM OCCURENCES FOR WATER
    
    ## GENERATING FIGURE
    fig = plot_scalar_with_gnp(fraction_sim_occur_dict =  fraction_sim_occur_dict,
                               mapping_list = mapping_list,
                               traj_data = traj_data,
                               desired_solvent = 'HOH',
                               index = index,
                               frame = frame)
    #%% STORING FIGURE
    ## DEFINING FIG NAME
    fig_name = "%s-HOH"%(sim_basename)
    mlab.savefig(os.path.join(store_image_location,
                              fig_name + '.png'))
    
    
    
    #%% PLOTTING ALL FRACTION OF SIM OCCURENCES FOR PROPANE
    
    ## GENERATING FIGURE
    fig = plot_scalar_with_gnp(fraction_sim_occur_dict =  fraction_sim_occur_dict,
                               mapping_list = mapping_list,
                               traj_data = traj_data,
                               desired_solvent = 'PRO',
                               index = index,
                               frame = frame)
    #%% STORING FIGURE
    ## DEFINING FIG NAME
    fig_name = "%s-PRO"%(sim_basename)
    mlab.savefig(os.path.join(store_image_location,
                              fig_name + '.png'))    
    
    #%% COMPUTING SAMPLING TIME
    
    ## GETTING OCCURENCES AS A FUNCTION OF TIME
    sampling_time_occur_dict = generate_sampling_occur_dict(storage_frac_occurences)
    
    ## DEFINING SOLVENT LIST
    solvent_list = ['HOH', 'PRO']
    
    ## LOOPING
    for solvent in solvent_list:
    
        ## COMPUTING SAMPLING TIME OF OCCURENCES
        sampling_time_df = compute_sampling_time_frac_occur(sampling_time_occur_dict,
                                                            frame_rate = 500,
                                                            solvent = solvent,
                                                            )
        
        ## CREATING SAMPLING TIME FIGURE
        fig, ax = plot_sampling_time_df(sampling_time_df,
                                        grid_index = 1000,
                                        x_key = "last",
                                        y_key = "frac_occur",
                                        fig_size_cm = FIGURE_SIZE)
            
            
        ## DEFINING FIGURE NAME
        figure_name="%s-%s-sampling_time_for_grid1000"%(sim_basename, solvent)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(store_image_location,
                                         figure_name), 
                     fig_extension = 'png',
                     save_fig = True,
                     )
                     
        # #%% FINDING CONVERGENCE SAMPLING TIME FOR EACH GRID POINT
        
        ## COMPUTING MIN SAMPLING TIME
        stored_min_sampling = compute_min_sampling_time(sampling_time_df,
                                                        delta = 0.05,
                                                        x_key = 'last',
                                                        y_key = 'frac_occur')
        # #%% PLOTTING MINIMUM SAMPLINGTIME
        fig, ax = plot_min_sampling_time(stored_min_sampling,
                                         fig_size_cm = FIGURE_SIZE)
    
        ## DEFINING FIGURE NAME
        figure_name="%s-%s-min_sampling_times"%(sim_basename, solvent)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(store_image_location,
                                         figure_name), 
                     fig_extension = 'png',
                     save_fig = True,
                     )


#%%            
            
            #%%

    
    
    

    
    #%%

    
    
    
    
    
    
    #%% PLOTTING WITH RESPECT TO SCALAR
    
    
    ## IMPORTING MLAB
    from mayavi import mlab
    


    
    #%%
    
    
    ## DEFINING SOLVENT LIST    
    solvent_list = [
#                     'PRO', 
                         'HOH', 
                    ]
    

    
    ## PLOTTING
    
        
    
    
    #%%
    
    ## STORING 
    storage_dict = {}
    
    ## LOOPING THROUGH EACH AND GENERATING DATA
    for each_basename in sim_dict_unique:
        ## GETTING DIRS LIST
        path_sim_list=  sim_dict_unique[each_basename]
        
        ## DEFINING INPUTS
        inputs_extract = {
                'path_sim_list': path_sim_list,
                'want_num_dist_storage': True,
                }
        
        ## DEFINING PICKLE PATH
        pickle_path = os.path.join(path_sim_list[0],
                                   ANALYSIS_FOLDER,
                                   main_compute_np_cosolvent_mapping.__name__,
                                   "stored_combined.pickle"
                                   )
        
        ## EXTRACTION PROTOCOL WITH SAVING
        unnorm_dict, mapping_list, num_dist_storage = save_and_load_pickle(function = extract_unnormalized_dist, 
                                                                           inputs = inputs_extract, 
                                                                           pickle_path = pickle_path,
                                                                           rewrite = False,
                                                                           verbose = True)
        ## GETTING EX
        e_x_storage, p_n_storage = compute_e_x(unnorm_dict = unnorm_dict)            
        
        ## DEFINING PICKLE PATH
        e_x_pickle_path = os.path.join(path_sim_list[0],
                                       ANALYSIS_FOLDER,
                                       main_compute_np_cosolvent_mapping.__name__,
                                       "e_x.pickle"
                                       )
        
        ## STORING
        storage_dict[each_basename] = {
                'e_x': e_x_storage,
                'path_sim_list': path_sim_list,
                }
        
    #%% PLOTTING EX VS. MU
    

    
    
    ## DEFINING SOLVENT LIST    
    solvent_list = [
                    'PRO', 
                    'HOH', 
                    ]
    ## DEFINING PLANAR LIST
    planar_sims_list=["20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_COOH",
                      "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_dod",]
    
    
    ## DEFINING SOLVENT
    for solvent_idx, desired_solvent in enumerate(solvent_list):
        
        ## INITIAL FIG
        fig = None
        ax = None
        
        ## LOOPING THROUGH STORAGE
        for idx, storage_name in enumerate(storage_dict):
        
            ## EXTRACTING NOMENCLATURE
            path_sim_list = storage_dict[storage_name]['path_sim_list']
            sim_basename = os.path.basename( path_sim_list[0] )
            nomenclature = extract_nomenclature(name = sim_basename)
            
            ## defining ex
            e_x = storage_dict[storage_name]['e_x']
            
            ## SEEING TYPE
            if nomenclature['type'] == 'Planar':
                parent_wc_folder = "20200403-Planar_SAMs-5nmbox_vac_with_npt_equil"
            else:
                parent_wc_folder = "20200401-Renewed_GNP_sims_with_equil"
            
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
            
            
            ## DEFINIG LIGAND
            ligand = nomenclature['ligand']
            
            ## DEFINING COLOR
            color = LIGAND_COLOR_DICT[ligand]
            
            
    
            ## DEFINING E_X VALUES
            e_x_values = e_x[desired_solvent]
            
            ## PLOTTING E_X VS MU
            fig, ax = plot_ex_vs_mu(mu_array = mu_array,
                                    e_x_values = e_x_values,
                                    color = color,
                                    lower_e_x_value = 0,
                                    want_fitted_line = False,
                                    want_legend = False,
                                    fig = fig,
                                    label = ligand,
                                    ax = ax)
        ## CREATING LEGEND
        ax.legend()
            
        
        ## SETTING LIMITS
        ax.set_ylim([-0.5, 6])
        ax.set_xlim([5,30])
        
        ## DEFINING FIGURE NAME
        figure_name="Multiple_surfaces_%s"%(desired_solvent)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(store_image_location,
                                         figure_name), 
                     fig_extension = 'png',
                     save_fig = True,
                     )
            
        
            
        
    #%%
    

    

    #%%
    
    for parent_dir in np_parent_dirs:
        
        ## DEFINING LIST        
        path_sim_list = glob.glob( os.path.join(main_dir,
                                                parent_dir) + "/*")
        
        ## SORTING
        path_sim_list.sort()
        
        ## storing map
        mapping_list = []
        
        ## SEEING IF YOU WANT TO STORE NUMBER STORAGE
        want_num_dist_storage = False
        
        ## DEFINING INPUTS
        inputs_extract = {
                'path_sim_list': path_sim_list,
                'want_num_dist_storage': True,
                }
        
        ## DEFINING PICKLE PATH
        pickle_path = os.path.join(path_sim_list[0],
                                   ANALYSIS_FOLDER,
                                   main_compute_np_cosolvent_mapping.__name__,
                                   "stored_combined.pickle"
                                   )
        
        ## EXTRACTION PROTOCOL WITH SAVING
        unnorm_dict, mapping_list, num_dist_storage = save_and_load_pickle(function = extract_unnormalized_dist, 
                                                                           inputs = inputs_extract, 
                                                                           pickle_path = pickle_path,
                                                                           rewrite = False,
                                                                           verbose = True)
        
                    
        #%% COMPUTE PROBABILITY DISTRIBUTION
        
        ## GETTING EX
        e_x_storage, p_n_storage = compute_e_x(unnorm_dict = unnorm_dict)            
        
        ## DEFINING PICKLE PATH
        e_x_pickle_path = os.path.join(path_sim_list[0],
                                       ANALYSIS_FOLDER,
                                       main_compute_np_cosolvent_mapping.__name__,
                                       "e_x.pickle"
                                       )
        
        ## STORING E(X)
        pickle_results(results = [e_x_storage,p_n_storage],
                       pickle_path = e_x_pickle_path,
                       verbose = True)
        
        
        #%% LOADING TRAJ
        
        ## DEFINING INDEX
        index = 0
        frame = 0
        
        ## IMPORTING TOOLS
        import MDDescriptors.core.import_tools as import_tools
        
        ## DEFINING PATH TO SIM
        path_to_sim = path_sim_list[index]
        
        ## DEFINING GRO FILE
        prefix="sam_prod-2000-12000--pbc-mol"
        
        output_gro_file = prefix + ".gro"
        output_xtc_file = prefix + ".xtc"
        
        ## LOADING TRAJECTORY
        traj_data = import_tools.import_traj(directory = path_to_sim,
                                             structure_file = output_gro_file,
                                             xtc_file = output_xtc_file,
                                             discard_overlapping_frames = True,
                                             )
    
        #%% PLOTTING WITH MAYAVI
        
        ## IMPORTING MLAB
        from mayavi import mlab
        ### IMPORTING LIGAND REISDUE NAMES
        from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj, get_atom_indices_of_ligands_in_traj
        
        ## GETTING GOLD INDEX    
        au_index = [atom.index for atom in traj_data.topology.atoms if atom.name == 'Au' or atom.name == 'BAu']
    
        ## GETTING ATOM INDICES AND LIGAND NAME
        ligand_names, atom_index = get_atom_indices_of_ligands_in_traj( traj = traj_data.traj )
        
        ## DEFINING SOLVENT LIST    
        solvent_list = [
                         'PRO', 
#                         'HOH', 
#                          'FMA',
                        ]
        # 'HOH', 'PRO', 'FMA'
        
        figure = None
        
        ## DEFINING SOLVENT
        for desired_solvent in solvent_list:
        
            ## GETTING MAPPING  OBJECT
            current_mapping = mapping_list[index]
            
            ## GENERATING NEW GIRD
            grid = current_mapping.get_new_grid_given_time_index(time_idx = frame)
            
            # PLOTTING WITH MAYAVI
            mlab.clf(figure = figure)
            figure = mlab.figure('Scatter plot',
                                 bgcolor = (0.5, 0.5, 0.5))
            
            ## DEFINING SCALAR VALUES
            scalar_values = e_x_storage[desired_solvent]
            
            points = mlab.points3d(grid[:,0],
                                   grid[:,1],
                                   grid[:,2],
                                   scalar_values,
                                   figure = figure,
                                   # color=(.5,.5,.5),
                                   opacity=1,# 0.05
                                   transparent=False,
                                   )
            
            
            ## ADDING COLOR BAR
            current_colobar = mlab.colorbar(object = points,
                                            label_fmt="%.1f",
                                            nb_labels=4)
            
            ## PLOTTING LIGANDS
            fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                      atom_index = atom_index,
                                      frame = frame,
                                      figure = figure,
                                      dict_atoms = plot_funcs.ATOM_DICT,
                                      dict_colors = plot_funcs.COLOR_CODE_DICT)
            
            ## PLOTTING GOLD FIGURE
            fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                               atom_index = au_index,
                                               frame = frame,
                                               figure = fig,
                                               dict_atoms = plot_funcs.ATOM_DICT,
                                               dict_colors = plot_funcs.COLOR_CODE_DICT)
        
            ## DEFINIGN FIGURE NAME        
            fig_name= parent_dir + "_mayavi_" + desired_solvent

#            ## SAVING FIGURE
#            mlab.savefig(os.path.join(store_image_location,
#                                      fig_name + '.png'))
            
        #%% CORRELATING E(X) TO MU
        
        

        
        ## DEFINING FIGURE SIZE
        fig_size_cm = (9, 8) # in cm
        
        ## DEFINING PLANAR LIST
        planar_sims_list=["20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_COOH",
                          "20200427-planar_cosolvent_mapping_planar_rerun_GNP_noformic_dod",]
        
        if parent_dir in planar_sims_list:
            parent_wc_folder = "20200403-Planar_SAMs-5nmbox_vac_with_npt_equil"
        else:
            parent_wc_folder = "20200401-Renewed_GNP_sims_with_equil"
        
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
        
    
        
        colors=['red', 'blue', 'red',]
        
        ## DEFINING SOLVENT LIST    
        solvent_list = [
                        'PRO', 
                        'HOH', 
                        ]
        

        
        ## DEFINING SOLVENT
        for idx, desired_solvent in enumerate(solvent_list):
            
            ## PLOTTING ALL THE COLORS
            color=colors[idx]
            
            ## DEFINING E_X VALUES
            e_x_values = e_x_storage[desired_solvent]
            

            ## PLOTTING E_X VS MU
            fig, ax = plot_ex_vs_mu(mu_array = mu_array,
                                    e_x_values = e_x_values,
                                    color = color,
                                    lower_e_x_value = 0,
                                    want_fitted_line = True,
                                    want_legend = True)
            
            ## SETTING Y LIM
            ax.set_ylim([-0.5, 6])
            ax.set_xlim([8,14])
            
        
            figure_name = "7_cosolvent_mapping_%s"%(desired_solvent)
            # parent_dir + '_EN_vs_mu-' + desired_solvent
            
            ## UPDATING FIGURE SIZE
            fig = plot_funcs.update_fig_size(fig,
                                             fig_size_cm = fig_size_cm)
            ## SETTING AXIS
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(store_image_location,
                                             figure_name), 
                         fig_extension = 'svg',
                         save_fig = True,
                         )

    
    