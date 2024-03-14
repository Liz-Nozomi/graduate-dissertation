# -*- coding: utf-8 -*-
"""
extract_np_most_likely.py
The purpose of this code is to extract the np_most_likely_config code. We can use 
this code to generate plots for publication and so on.

Written by: Alex K. Chew (alexkchew@gmail.com, 10/1/2019)

"""

### IMPORTING MODULES
import numpy as np
import pandas as pd
import os
import mdtraj as md
import glob

## IMPORTING PLOTTING TOOLS
import MDDescriptors.core.plot_tools as plotter

## IMPORTING PICKLE FUNCTIONS
from MDBuilder.core.pickle_funcs import store_class_pickle, load_class_pickle

## IMPORTING MD DESCRIPTOR MOST LIKELY CONFIGURATION
from MDDescriptors.application.nanoparticle.np_most_likely_config import find_most_probable_np

## DEFINING IMAGE LOCATION
from MDDescriptors.application.np_hydrophobicity.global_vars import IMAGE_LOCATION, NP_SIM_PATH

## SETTING DEFAULTS
plotter.set_mpl_defaults()

## DEFINING FIGURE SIZES
FIGURE_SIZE = plotter.FIGURE_SIZES_DICT_CM['1_col']
## DEFINING FIGURE PATH
FIGURE_PATH= IMAGE_LOCATION
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200217\images\np_water_bundling_grps"
## DEFINING SAVE FIG
SAVE_FIG = True
## EXTENSION
FIG_EXTENSION = "png"

### FUNCTION TO PLOT PROBABILITY IN BUNDLE
def plot_prob_in_bundle(ligand_index,
                        prob_in_bundle,
                        figure_size = FIGURE_SIZE,
                        ax_kwargs = {}):
    '''
    The purpose of this function is to plot the probability of the ligand 
    is in a bundle or not in a bundle
    INPUTS:
        ligand_index: [list]
            list of ligand indexes
        prob_in_bundle: [list]
            list of probabilities in a bundle
        figure_size: [tuple, 2]
            figure size in cm
        ax_kwargs: [dict]
            dictionary for arguments for axis
    OUTPUTS:
        fig, ax:
            figure and axis
    '''

    ## CREATING FIGURE
    fig, ax = plotter.create_fig_based_on_cm(fig_size_cm = figure_size)

    ## UPDATING AXIS
    ax.set_xlabel('Ligand index')
    ax.set_ylabel("Probability of bundle")
    
    ## PLOTTING
    ax.plot(ligand_index, prob_in_bundle, markersize = 8, color ='k', marker = '.', linestyle = "None", linewidth = 1.5 )
    
    ## SETTING TICKS
    ax = plotter.adjust_x_y_ticks(ax = ax,
                                  **ax_kwargs)
    
    ## FITTING
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT CROSS ENTROPY
def plot_cross_entropy_vs_bundles(most_probable_np,
                                  figure_size = FIGURE_SIZE,
                                  ax_kwargs = {}):
    '''
    The purpose of this function is to plot the cross entropy versus the 
    number of bundles. 
    INPUTS:
        most_probable_np: [obj]
            most probable nanoparticle object
        figure_size: [tuple, 2]
            figure size in cm
        ax_kwargs: [dict]
            dictionary for arguments for axis
    OUTPUTS:
        fig, ax:
            figure and axis
    '''
    ## CREATING FIGURE
    fig, ax = plotter.create_fig_based_on_cm(fig_size_cm = figure_size)
    
    ## UPDATING AXIS
    ax.set_xlabel('Number of bundles')
    ax.set_ylabel("Cross entropy of ligands")
    
    ## PLOTTING LINE
    ax.plot(most_probable_np.total_bundling_groups, 
               most_probable_np.cross_entropy_summed, 
               markersize = 8, color ='k', marker = '.', linestyle = "None", linewidth = 1.5   )
    
    ## PLOTTING MINIMA
    ax.axhline( y = most_probable_np.cross_entropy_lowest, linestyle='--', color = 'r', linewidth = 2,
               label = 'Lowest cross entropy')
    ax.axvline( x = most_probable_np.avg_bundling, linestyle='--', color = 'b', linewidth = 2, 
               label = 'Average bundling group')
    
    ## PLOTTING MOST LIKELY POINT
    idx_most_likely = most_probable_np.closest_structure_database.iloc[0]['Python_index']
    idx_least_likely = most_probable_np.closest_structure_database.iloc[-1]['Python_index']
    ## PLOTTING MOST LIKELY
    ax.plot( most_probable_np.total_bundling_groups[idx_most_likely],
                most_probable_np.cross_entropy_summed[idx_most_likely],
                color='g',
                linewidth=1.5,
                markersize = 8,
                linestyle="None",
                marker = "o",
                label='most_likely')
    
    ## PLOTTING LEAST LIKELY
    ax.plot( most_probable_np.total_bundling_groups[idx_least_likely],
                most_probable_np.cross_entropy_summed[idx_least_likely],
                color='purple',
                linewidth=1.5,
                markersize = 8,
                linestyle="None",
                marker = "o",
                label='least_likely')
    
    ## SETTING TICKS
    ax = plotter.adjust_x_y_ticks(ax = ax,
                                  **ax_kwargs)
    ## ADDING LEGEND
    ax.legend(loc='upper left', fontsize = 8)

    ## FITTING
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO GET NON ASSIGNED
def plot_num_bundles_and_nonbundle_vs_frame(most_probable_np,
                                            figsize = plotter.FIGURE_SIZES_DICT_CM['1_col_landscape'],
                                            want_nonbundle = True,
                                            frame_to_ns = 0.01,
                                            relative_zero = 0):
    '''
    This function generates number of bundles and non-bundles
    INPUTS:
        most_probable_np: [obj]
            most probable nanoparticle object
        figure_size: [tuple, 2]
            figure size in cm
        want_nonbundle: [logical]
            True if you want to print non-bundled
        frame_to_ns: [float]
            frame to nanosecond conversion
        relative_zero: [float]
            zero value you want to set. By default, this is zero. 
    OUTPUTS:
        fig, axs:
            figure and axis
    '''
    ## GETTING TOTAL FRAMES
    total_frames = most_probable_np.bundling_groups.total_frames
    ## DEFINING FRAME ARRAY
    frame_array = np.arange(total_frames)
    ## GETTING TOTAL BUNDLING GROUP
    total_bundling_groups = np.array(most_probable_np.bundling_groups.lig_total_bundling_groups)
    ## GETTING TOTAL NONASSIGNMENTS
    total_non_assignments = np.concatenate(most_probable_np.bundling_groups.lig_nonassignments_list)
    ## GETTTING AVERAGE BUNDLING
    avg_bundling = most_probable_np.bundling_groups.results_avg_std_bundling_grps['avg']
    
    ## GETTING TIME INDEX
    time_index_probable = most_probable_np.closest_structure_time_index[0]
    
    ## SEEING IF YOU WANT BUNDLED / NONBUNDLED
    if want_nonbundle is True:
        num_rows = 2
        bundling_idx = 1
        non_bundling_idx = 0
    else:
        num_rows = 1
        bundling_idx = 0
    
    ## GETTING FIGURE
    fig, axs = plotter.create_subplots(num_rows = num_rows, 
                                       num_cols = 1, 
                                       figsize = figsize,
                                       wspace = 0,
                                       hspace = 0)
    
    ## PLOTTING BOTTOM PLOT
    axs[bundling_idx].plot( frame_array * frame_to_ns + relative_zero, total_bundling_groups, 
                               color ='k', marker = '.', linestyle = "None", linewidth = 1.5
                               , markersize = 8,)
    axs[bundling_idx].set_xlabel('Time (ns)')
    axs[bundling_idx].set_ylabel("Num. of bundles")
    
    ## ADDING AVERAGE BUNDLING
    axs[bundling_idx].axhline( y = avg_bundling, linestyle = '--', color ='b', linewidth = 2, label='avg_bundling' )
#    axs[bundling_idx].axvline( x = frame_array[time_index_probable], linestyle = '--', color ='g', linewidth = 2, label = 'most_likely' )
    if want_nonbundle is True:
        axs[non_bundling_idx].axvline( x = frame_array[time_index_probable], linestyle = '--', color ='g', linewidth = 1.5, label = 'most_likely' )
        
        ## PLOTTING NON BUNDLES
        axs[non_bundling_idx].plot(frame_array * frame_to_ns+ relative_zero, total_non_assignments,color ='r', linewidth = 1.5   )
        axs[non_bundling_idx].set_ylabel('Num. of non-bundles')
        
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, axs


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINIGN MAIN DIRECTORY
    main_dir =NP_SIM_PATH
    
    ### DIRECTORY TO WORK ON    
    simulation_dir=r"EAM_COMPLETE"
    
    ## DEFINING PREFIX
    sim_prefix = r"EAM_300.00_K_2_nmDIAM"
    sim_suffix = r"CHARMM36jul2017_Trial_1"
    
    ## DEFINING DESIRED LIGANDS
    desired_ligands = [
#            'dodecanethiol',
#            'C11CF3',
#            'C11CONH2',
#            'C11COOH',
#            'C11NH2',
#            'C11OH',
    
            "C11double67OH"
            
            
            ]
    
    ## SEEING IF YOU WANT ALL LIGANDS
    want_all_ligands = True
    
    if want_all_ligands is True:
        path_sim = os.path.join(main_dir,
                                simulation_dir)
        
        ## GETTING ALL PATHS
        directory_list = glob.glob(path_sim + "/*")
        
        ## SORTING
        directory_list.sort()
        ## GETTING ALL BASENAMES
        desired_ligands = directory_list[:]
    
    
#            'dodecanethiol',
#            'C11CF3',
#            'C11CONH2',
#            'C11COOH',
#            'C11NH2',
#            'C11OH',
    
    ## DEFINING DICT MOST LIKELY
    most_likely_dict = {
            'avg_heavy_atom': 'most_likely-avg_heavy_atom',
            # 'terminal_heavy_atom': 'most_likely-terminal_heavy_atom',
            }
    
    ## DEFINING PICKLE FILE
    pickle_file="np_most_likely.pickle"
    
    
    ## DEFINING IMAGES DESIRED
    image_list = ['bundles_vs_frames',
                  'prob_vs_index',
                  'cross_entropy',]
    # 'prob_vs_index', 'cross_entropy', 
    
    # 'prob_vs_index', 'cross_entropy', 'bundles_vs_frames'
    ## LOOPING THROUGH EACH LIGAND
    for each_ligand in desired_ligands:
    
        ## DEFINING SIM PATH            
        if want_all_ligands is True:
            path_to_sim = each_ligand
            specific_dir = os.path.basename(each_ligand)
        else:
            path_to_sim = os.path.join(main_dir, 
                                       simulation_dir,
                                       specific_dir)
            
            ## DEFINING SPECIFIC DIRECTORY FOR ANALYSIS
            specific_dir='_'.join([sim_prefix, each_ligand, sim_suffix])
        
        ## LOOPING THROUGH TYPES
        for most_likely_type in most_likely_dict.keys():
            
            ## DEFINING PATH TO PICKLE
            path_pickle = os.path.join(path_to_sim,
                                       most_likely_dict[most_likely_type],
                                       pickle_file)
            
            ## LOADING PICKLE
            most_probable_np = pd.read_pickle(path_pickle)[0] # load_class_pickle(path_pickle)
            
            #################################################
            ### PLOT FOR PROBABILITY OF LIGANDS VS. INDEX ###
            #################################################
            if 'prob_vs_index' in image_list:
            
                ## DEFINING FIGURE NAME
                figure_name = '_'.join([specific_dir, most_likely_type,"prob_vs_index"])
                
                ## DEFINING LIGAND INDEX
                ligand_index = np.arange(len(most_probable_np.prob_ligand_in_bundle))
                prob_in_bundle = most_probable_np.prob_ligand_in_bundle
            
                ## DEFINING AXIS KWARGS
                ax_kwargs = {
                        'x_axis_labels': (0, 80, 20),
                        'y_axis_labels': (0, 1, 0.2),
                        'ax_x_lims'    : [-5, 85 ],
                        'ax_y_lims'    : [0, 1.1],
                        }
                
                ## PLOTTING FIGURE
                fig, ax = plot_prob_in_bundle(ligand_index = ligand_index,
                                              prob_in_bundle = prob_in_bundle,
                                              ax_kwargs = ax_kwargs
                                              )
                
                ## STORING FIGURE
                plotter.store_figure( fig = fig,
                                      path = os.path.join(FIGURE_PATH, figure_name),
                                      fig_extension = FIG_EXTENSION,
                                      save_fig = SAVE_FIG,
                                      )
    
            ##################################
            ### CROSS ENTROPY WITH BUNDLES ###
            ##################################
            if 'cross_entropy' in image_list:
                ## DEFINING AXIS KWARGS
                ax_kwargs = {
                        'x_axis_labels': None, # (0, 80, 20),
                        'y_axis_labels': None, # (0, 1, 0.2),
                        'ax_x_lims'    : None, # [-5, 85 ],
                        'ax_y_lims'    : None, # [0, 1.1],
                        }
                
                ## DEFINING FIGURE NAME
                figure_name = '_'.join([specific_dir, most_likely_type,"cross_entropy"])
            
                ## GENERATING PLOT
                fig, ax = plot_cross_entropy_vs_bundles(most_probable_np = most_probable_np,
                                                        ax_kwargs = ax_kwargs)
                
                ## STORING FIGURE
                plotter.store_figure( fig = fig,
                                      path = os.path.join(FIGURE_PATH, figure_name),
                                      fig_extension = FIG_EXTENSION,
                                      save_fig = SAVE_FIG,
                                      )
                
            ##############################
            ### BUNDLES AND NONBUNDLES ###
            ##############################
            if 'bundles_vs_frames' in image_list:
                ## GENERATING FIGURES
                fig, axs = plot_num_bundles_and_nonbundle_vs_frame(most_probable_np = most_probable_np,
                                                                   figsize = plotter.FIGURE_SIZES_DICT_CM['1_col_landscape'])
            
                ## DEFINING FIGURE NAME
                figure_name = '_'.join([specific_dir, most_likely_type,"bundles_vs_frames"])
                
                ## STORING FIGURE
                plotter.store_figure( fig = fig,
                                      path = os.path.join(FIGURE_PATH, figure_name),
                                      fig_extension = FIG_EXTENSION,
                                      save_fig = SAVE_FIG,
                                      )
            
            
            
                
    
    #%%
    

    ## CROSS ENTROPY VERSUS NUMBER OF BUNDLES
    
    ## PLOTTING CROSS ENTROPY VERSUS NUMBER OF BUNDLES
    # most_probable_np.plot_cross_entropy_vs_num_bundles()
    
    ## PLOTTING PROBABILITY OF LIGAND VERSUS INDEX
    # most_probable_np.plot_prob_ligand_vs_index()