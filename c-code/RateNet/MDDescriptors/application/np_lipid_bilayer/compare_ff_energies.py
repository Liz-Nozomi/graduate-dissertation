#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ff_energies.py

The purpose of this script is to compare forcefield energies between 
unmodified and modified charmm36 force fields

Written by: Alex K. Chew (08/12/2020)

"""
## IMPORTING MODULES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## LOADING SPECIFIC MODULES
import MDDescriptors.core.read_write_tools as read_write_tools

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING STORAGE FIG LOCATION
STORE_FIG_LOC=r"/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200907-clean/images"

## DEFINING COLORS
COLOR_TYPES = {
        'LJ-SR:NP_LIGANDS-NP_LIGANDS': 'k', 
        'LJ-14:NP_LIGANDS-NP_LIGANDS': 'r',
        'LJ-SR:NP_LIGANDS-HEADGRPS': 'b', 
#        'LJ-14:NP_LIGANDS-HEADGRPS': 'g',
        }


## INTERACTION TYPES FOR Z PULLING
INTERACTION_TYPES=[
       'Total Energy',
#       'LJ-SR:NP_ALK_RGRP-NP_NGRP', 
#       'LJ-SR:NP_ALK_RGRP-LM_HEADGRPS',
#       'LJ-SR:NP_NGRP-LM_HEADGRPS',
       
       'LJ-SR:NP_ALK-NP_NGRP',
       'LJ-SR:NP_NGRP-NP_RGRP',
        ]

## DEFINING STYLE
COMPARISON_STYLES={
        "Original": {
                "linestyle": '-',
                "color": 'k',
                },
        "Modified":{
                "linestyle": '-',
                "color": 'r',
                }
        }
        
### FUNCTION TO PLOT INTERACTIONS
def subplot_interactions_vs_time(interaction_list,
                                 xvg_file_storage,
                                 fig_size = (8, 18),
                                 nrows = 3,
                                 ncols = 1,
                                 alpha = 1):
    '''
    The purpose of this function is to plot the interactions
    INPUTS:
        interaction_list: [list]
            list of interactions
        xvg_file_storage: [dict]
            dictionary storing all the data
        nrows: [int]
            number of rows
        ncols: [int]
            number of columns
    OUTPUTS:
        fig, axs:
            figure and axes for plot
    '''
    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, 
                             figsize = plot_funcs.cm2inch( *fig_size ))
    
    ## FLATTENING AXES
    axs= axs.flatten()
    
    ## LOOPING THROUGH EACH KEY
    for idx, each_interaction in enumerate(interaction_list):
        ## DEFINING AXIS
        ax = axs[idx]
        
        ## LOOPING THROUGH ORIG AND MODIFIED
        for xvg_key in xvg_file_storage:
            ## DEFINING XVG DATA
            xvg_data = xvg_file_storage[xvg_key]
            
            x = xvg_data['time']/ 1000.000
            y = xvg_data[each_interaction]
            
            ## GETTING COLOR
            linestyles = COMPARISON_STYLES[xvg_key]
            
            ## PLOTTING
            ax.plot(x,y, label = xvg_key, alpha = alpha, **linestyles)
        ## ADDING LEGEND
        ax.legend()
        
        ## ADDING TITLE
        ax.set_title(each_interaction, fontsize=10)
        
        ## ADDING TICKS
        ax.tick_params(axis='both', which='both', labelsize=8, labelbottom = True)
        
        ## ADDING LABELS
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Energy (kJ/mol)")
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    return fig, axs

#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    sim_list = [
            "0.0",
            "150.0",
            ]
    
    ## DEFINING PARENT DIRECTORY
    parent_directory = "20200615-US_PLUMED_rerun_with_10_spring"
    
    ## DEFINING PARENT SIM
    parent_sim="UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1"
    parent_sim="UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    
    ### FOR Z -PULLING
    sim_list = [
            "1.900"
            ]
    
    
    ## DEFINING PARENT DIRECTORY
    parent_directory = "20200613-US_otherligs_z-com"
    
    ## DEFINING PARENT SIM
    parent_sim="US-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1"    
    
    #-------
    ## FOR NEW FORCEFIELD
    
    ## DEFINING PARENT DIRECTORY
    parent_directory = "20200818-Bn_US_modifiedFF"
    
    ## DEFINING PARENT SIM
    parent_sim="US-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1"    
    
    sim_list = [
            "5.100",
            "1.900",
            ]
    
## INTERACTION TYPES FOR Z PULLING
    interaction_list=[
       'Total Energy',
       'LJ-SR:NP_RGRP-NP_NGRP', 
       'LJ-SR:NP_RGRP-LM_HEADGRPS',
       'LJ-SR:NP_NGRP-LM_HEADGRPS',
        ]
    
    ## DEFINING TYPE
    for sim_folder in sim_list:
    
        ## DEFINING PATH TO SIMULATION    
        path_to_sim_folder=r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/%s/%s/4_simulations/%s"%(parent_directory, parent_sim,sim_folder)
        
        ## GETTING BASENAME
        sim_basename=parent_sim
        # os.path.dirname(os.path.basename(path_to_sim_folder))
        
        ## DEFINING XVG FILE
        xvg_file_dict = {
                'Original': r"ffcomparison_orig.xvg",
                'Modified': r"ffcomparison_mod.xvg",
                }
        
        ## GETTING XVG FILES
        xvg_file_storage = {}
        
        ## LOADING EACH XVG FILE
        for each_key in xvg_file_dict:
            ## PATH TO XVG
            path_to_xvg = os.path.join(path_to_sim_folder,
                                       xvg_file_dict[each_key])
            
            ## READING XVG
            xvg_file = read_write_tools.read_xvg(path_to_xvg)
            
            
            ## CREATING DATAFRAME
            xvg_file_df = xvg_file.get_df()        
            
            ## APPENDING
            xvg_file_storage[each_key] = xvg_file_df
        
        ## CLOSING ALL FIGURES
        plt.close('all')
        
        ## DEFINING FIGURE SIZE
        fig_size = (16, 16)
        
        ## PLOTTING
        fig, axs = subplot_interactions_vs_time(interaction_list,
                                         xvg_file_storage,
                                         fig_size = fig_size,
                                         nrows = 2,
                                         ncols = 2,
                                         alpha = 0.75)
        
                
        ## STORING FIGURE5
        ## DEFINING FIGURE NAME
        figure_name = "%s-Interactions_vs_time_%s"%(sim_basename,sim_folder)
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 fig_extension = 'png',
                                 save_fig = True,
                                 )
    #%% PLOTTING FOR NP IN WATER
    
    ## DEFINING PATH TO SIM
    path_to_sim_parent=r"/Volumes/akchew/scratch/nanoparticle_project/simulations"
    
    ## DEFINING RELATIVE PATHS
    relative_paths = ["ROT_WATER_SIMS/EAM_300.00_K_2_nmDIAM_ROT017_CHARMM36jul2017_Trial_1",
                      "ROT_WATER_SIMS_MODIFIEDFF/EAM_300.00_K_2_nmDIAM_ROT017_CHARMM36jul2017mod_Trial_1",
                      ]
    
    ## INTERACTION TYPES FOR Z PULLING
    interaction_list=[
           'Total Energy',       
           'LJ-SR:NP_ALK-NP_NGRP',
           'LJ-SR:NP_NGRP-NP_RGRP',
            ]
    
    ## DEFINING FIGURE SIZE
    fig_size = (17.1, 6)
    nrows = 1
    ncols = 3
    
    # ---------------------------
    
    ## DEFINING PATH TO SIM
    path_to_sim_parent=r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims"
    
    ## DEFINING RELATIVE PATHS
    relative_paths = ["20200902-phenol_tma_tests/phenol-tetramethylammonium-0.4-charmm36-jul2017.ff",
                      "20200902-phenol_tma_tests/phenol-tetramethylammonium-0.4-charmm36-jul2017-mod.ff",
                      ]
    
    ## INTERACTION TYPES FOR Z PULLING
    interaction_list=[
           'Total Energy',       
           'LJ-SR:PHE-TMA',
            ]
    
    fig_size = (12, 6)
    nrows = 1
    ncols = 2
    # ---------------------------
    
    ## DEFINING XVG FILE
    xvg_file_dict = {
            'Original': r"ffcomparison_orig.xvg",
            'Modified': r"ffcomparison_mod.xvg",
            }
    
    ## LOOPING THROUGH PATHS
    for each_rel_path in relative_paths:
        ## DEFINING PATH
        path_to_sim_folder = os.path.join(path_to_sim_parent,
                                          each_rel_path)
        
        ## GETTING XVG FILES
        xvg_file_storage = {}
        
        ## LOADING EACH XVG FILE
        for each_key in xvg_file_dict:
            ## PATH TO XVG
            path_to_xvg = os.path.join(path_to_sim_folder,
                                       xvg_file_dict[each_key])
            
            ## READING XVG
            xvg_file = read_write_tools.read_xvg(path_to_xvg)
            
            
            ## CREATING DATAFRAME
            xvg_file_df = xvg_file.get_df()        
            
            ## APPENDING
            xvg_file_storage[each_key] = xvg_file_df
        
        ## CLOSING ALL FIGURES
#        plt.close('all')
        
        ## PLOTTING
        fig, axs = subplot_interactions_vs_time(interaction_list,
                                         xvg_file_storage,
                                         fig_size = fig_size,
                                         nrows = nrows,
                                         ncols = ncols)
        
#        ## SETTING XAXIS
#        for ax in axs:
#            ax.set_xlim([-5, 55])
#            ax.set_xticks(np.arange(0,60,10))
#        
        
        ## STORING FIGURE5
        ## DEFINING FIGURE NAME
        figure_name = "Interactions_vs_time-%s"%(os.path.basename(path_to_sim_folder))
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 fig_extension = 'png',
                                 save_fig = True,
                                 )
        
    
    
    
    