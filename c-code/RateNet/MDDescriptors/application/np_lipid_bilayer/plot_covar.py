#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_covar.py
The purpose of this script is to plot the covar information. 

Written by: Alex K. Chew (05/01/2020)




"""
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

## IMPORTING COVAR READING
from MDDescriptors.core.import_tools import read_plumed_covar as read_covar

## IMPORTING PLOTTING TOOLS
import MDDescriptors.core.plot_tools as plot_tools

## DEFAULTS
plot_tools.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_tools.FIGURE_SIZES_DICT_CM['1_col_landscape']


## DEFINING IMAGE LOC
from MDDescriptors.application.np_lipid_bilayer.global_vars import IMAGE_LOC

## DEFINING DICTIONARY
DEFAULT_LABEL_DICT = {
        'time': 'Time (ps)',
        'coord': 'Number of contacts',  
        'cn0': 'Number of contacts',  
        'cn10': 'Number of contacts',  
        }

### FUNCTION TO PLOT DATA
def plot_xy_data(data,
                 x_label = 'time',
                 y_label = 'coord',
                 fig_size = FIGURE_SIZE,
                 fig = None,
                 ax = None):
    '''
    This function plots the xy data
    INPUTS:
        data: [dataframe]
            pandas dataframe
    OUTPUTS:
        fig, ax:
            figure and axis for the plot
    '''

    ## PLOTTING
    if fig is None or ax is None:
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = fig_size)
    
    ## SEEING IF Y LABEL IS A LIST
    if type(y_label) is list:
        y_label_list = y_label
        color_list = plot_tools.get_cmap(len(y_label_list))
    else:
        y_label_list = [y_label]
    
    ## SETTING LEGEND
    if x_label in DEFAULT_LABEL_DICT:
        set_x_label = DEFAULT_LABEL_DICT[x_label]
    else:
        set_x_label = x_label
    
    ## SETTING LEGEND
    if y_label_list[0] in DEFAULT_LABEL_DICT:
        set_y_label = DEFAULT_LABEL_DICT[y_label_list[0]]
    else:
        set_y_label = y_label_list[0]

    ax.set_xlabel(set_x_label)
    ax.set_ylabel(set_y_label)

    ## DEFINING X ARRAY
    x = np.array(data[x_label])
    
    ## LOOPING
    for idx, y_label in enumerate(y_label_list):
    
        ## PLOTTING SPECIFIC
        y = np.array(data[y_label])
        
        ## FINDING COLOR
        if len(y_label_list) > 1:
            color =  color_list(idx)
        else:
            color = 'k'
        
        ## PLOTTING
        ax.plot(x, y, linestyle='-', color = color, label = y_label)
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
    ## ADDING LEGEND
    if len(y_label_list) > 1:
        ax.legend()
    
    return fig, ax

### FUNCTION TO PLOT ALL XY INFORMATION
def plot_all_xy_data(data,
                     x_label = 'time',
                     want_subplots = False,
                     fig_size = FIGURE_SIZE):
    '''
    This function plots all xy data.
    INPUTS:
        data: [pd.dataframe]
            dataframe for the data
        x_label: [str]
            x-label for all the plots
        want_subplots: [logical]
            True if you want subplots. It will generate subplots based on 
            3 columns
    
    '''
    ## GETTING COLUMNS
    cols = list(data.columns)
    
    ## DROPPING ALL COLS EXCEPT FOR XLABELS
    cols.remove(x_label)
    
    ## SETTING UP SUBPLOTS
    if want_subplots is True:
        ncols = 3 # default
        nrows = math.ceil(len(cols) / ncols)
        
        ## GENERATING SUBPLOTS
        fig, axs = plt.subplots(nrows, ncols,
                                 )
        
        axs_list = [ ax for ax in axs.flat ]
        # figsize = [ fig_size[0] * nrows, fig_size[1] * ncols ] 
        
    ## CREATING OUTPUT DICT FOR THE FIGURE
    fig_dict = {}
    
    ## LOOPING
    for idx, y_label in enumerate(cols):
        
        ## NO FIG, AX
        if want_subplots is False:
            fig = None
            ax = None
        else:
            fig = fig
            ax = axs_list[idx]

        ## PLOTTING THE DATA
        fig, ax = plot_xy_data(data,
                               x_label = x_label,
                               y_label = y_label,
                               fig_size = fig_size,
                               fig = fig,
                               ax = ax)
        
        fig_dict[y_label] = {'fig' : fig, 
                             'ax': ax}
        
    ## LOOPING THROUGH AXIS THAT ARE NOT USED
    for idx in range(len(cols), len(axs_list)):
        axs_list[idx].set_axis_off()
    
    if want_subplots is True:
        return fig
    else:
        return fig_dict

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    ## DEFINING PATH TO SIM
    path_to_sim="/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200608-Pulling_with_hydrophobic_contacts/NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # path_to_sim="/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200608-Pulling_with_hydrophobic_contacts/NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"
    # "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200517-plumed_test_pulling_new_params_debugging/NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"
    # "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200517-plumed_test_pulling_new_params_debugging_stride1000_6ns/NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200517-plumed_test_pulling_new_params_debugging_stride1000_6ns/NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"
    # "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200507-full_pulling_plumed_larger_strides/NPLMplumedcontactspulling-5.100_2_50_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200430-debugging_nplm_plumed_ROT012_neighbor_list/NPLMplumedcontactspulling-5.100_2_25_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200430-debugging_nplm_plumed_ROT012_neighbor_list_pt2/NPLMplumedcontactspulling-5.100_2_25_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200430-debugging_nplm_plumed_ROT012_neighbor_list/NPLMplumedcontactspulling-5.100_2_25_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200430-debugging_nplm_plumed_ROT012_neighbor_list/NPLMplumedcontactspulling-5.100_2_25_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    # r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200430-debugging_nplm_plumed_ROT012/NPLMplumedcontactspulling-5.100_2_50_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    ## DEFINING DATA FILE
    data_file="COVAR.dat"
    
    ## PATH TO DATA
    path_data = os.path.join(path_to_sim,
                             data_file)

    ## READING THE DATA
    data = read_covar(path_data = path_data)
    
    ## DEFINING BASENAME
    sim_basename = os.path.basename(path_to_sim)
    
    #%%
    
    ## PLOTTING
    fig, ax = plot_xy_data(data,
                           x_label = 'time',
                           y_label = 'coord',
                           fig_size = FIGURE_SIZE )

    ## STORING FIGURE
    figure_name = "Coord_vs_time-%s"%(sim_basename)
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
    
    #%%
    
    ## PLOTTING
    fig, ax = plot_xy_data(data,
                           x_label = 'time',
                           y_label = ['coord', 'cn1','cn10', 'cn100', 'cn1000',],
                           fig_size = FIGURE_SIZE )
    
    ## STORING FIGURE
    figure_name = "Coord_vs_time_multiple_coords"
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
    
    #%% PLOTTING ALL DATA AS AS SUBPLOT
    
    ## GETTING FIG DICT
    fig = plot_all_xy_data(data,
                            x_label = 'time', 
                            want_subplots = True)
    

    figure_name = "All_covar_data-%s"%(sim_basename)
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
        
        
    