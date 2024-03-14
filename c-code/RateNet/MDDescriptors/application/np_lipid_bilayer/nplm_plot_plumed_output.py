#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nplm_plot_plumed_output.py

The purpose of this script is to plot the outputs from plumed in terms of the 
number of contacts and so forth. The idea would be to extract contacts versus 
time and perhaps correlate them with center of mass distances. 

Written by: Alex. K. Chew (05/19/2020)

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, PARENT_SIM_PATH, nplm_job_types, GROUP_COLOR_DICT

## PLUMED INPUT
from MDDescriptors.application.np_lipid_bilayer.compute_contacts_for_plumed import extract_plumed_input_and_covar

## IMPORTING TOOLS FOR GETTING TIME ARRAY
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import get_time_array_and_z_dist_from_com_distances, get_indices_for_truncate_time

## SETTING DEFAULTS
import MDDescriptors.core.plot_tools as plot_funcs
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING DEFAULTS
PLUMED_INPUT_FILE="plumed_analysis_input.dat"

## DEFINING TIME STRIDE
TIME_STRIDE=100

## DEFINING IMAGE LOCATION
IMAGE_LOC=r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_lipid_membrane_binding/Figures/svg_output"
#IMAGE_LOC = r"/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200727_clean/images"
# "/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200622/images/nplm_project"
# "/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_lipid_membrane_binding/Figures/svg_output"

### FUNCTION TO SHORTEN LABELS
def nplm_shorten_label(current_label="NP_LIGAND_CARBON_ONLY-LM_TAILGRPS",
                       want_np_only = False,
                       return_separated=False,
                       return_orig_label = False,):
    '''
    This function simply shortens the label by removing the "NP" and "LM" portion of the name.
    INPUTS:
        current_label: [str]
            label
        want_np_only: [logical]
            True if you want only np label
        return_separated: [logical]
            True if you want to return separated
        return_orig_label: [logical]
            True if you want origin label tagged on the end
    OUTPUTS:
        new_label: [str]
            label that has been updated
    '''
    ## SPLITTING
    np_label, lm_label = current_label.split('-')
    ## SPLITTING NP
    if np_label.startswith("NP"):
        new_np_label = '_'.join(np_label.split('_')[1:])
    else:
        new_np_label = np_label
    if lm_label.startswith("LM"):
        new_lm_label = '_'.join(lm_label.split('_')[1:])
    else:
        new_lm_label = lm_label
    
    ## JOINING
    if want_np_only is True:
        new_label = new_np_label
    elif return_separated is True:
        new_label = [new_np_label, new_lm_label]
    else:
        new_label='-'.join([new_np_label, new_lm_label])
    
    ## ADDING ORIGINAL
    if return_orig_label is True:
        if type(new_label) is list:
            new_label+= [current_label]
        else:
            new_label = [new_label, current_label]
    
    return new_label
    
### FUNCTION TO TRUNCATE DICT BASED ON TIME
def truncate_df_based_on_time(current_dict,
                                last_time_ps=50000,
                                time_label='time'):
    '''
    This function truncates a dataframe according to time and relabels it.
    INPUTS:
        my_df: [dataframe]
            dataframe containing data
        last_time_ps: [int]
            last time to search in picoseconds
        time_label: [str]
            time label
    OUTPUTS:
        truncated_dict: [dict]
            truncated dictionary with only times up to a certain time
            
    '''
    ## FINDING ALL INDICES WITH correct TIME
    indices = get_indices_for_truncate_time(time_array = current_dict[time_label].to_numpy(), last_time_ps = last_time_ps)
    ## FINDING DICT
    truncated_dict = current_dict.iloc[indices]
            
    return truncated_dict

### FUNCTIO NTO LOAD PLUMED FILES AND DISTANCES
def load_plumed_contacts_and_com_distances(sim_list_for_dict,
                                           plumed_input_file = PLUMED_INPUT_FILE,
                                           time_stride = TIME_STRIDE):
    '''
    This function loads the plumed contacts and center of mass distance given 
    the simulation key list.
    INPUTS:
        sim_list_for_dict: [list]
            list of simulation keys
        plumed_input_file: [str]
            input plumed file
        time_stride: [float]
            time stride between the steps for plumed
    OUTPUTS:
        sims_covar_storage: [dict]
            dictionary with the details stored
    '''

    ## STORAGE
    sims_covar_storage = {}
    
    ## LOOPING THROUGH EACH SIM
    for each_sim in sim_list_for_dict:
        ## GETTING MAIN SIM
        main_sim_dir = NPLM_SIM_DICT[each_sim]['main_sim_dir']
        specific_sim = NPLM_SIM_DICT[each_sim]['specific_sim']
        
        ## GETTING ALL JOB TYPES
        job_types = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                   main_sim_dir = main_sim_dir,
                                   specific_sim = specific_sim,)
        
        ## GENERATING DICT TO STORE
        covar_storage = {}
        
        ## LOOPING THROUGH EACH LIBRARY
        for idx, path_to_sim in enumerate(job_types.path_simulation_list):
            ## DEFINING CONFIG NAME
            config_name = job_types.config_library[idx]
            
            ## TRYING TO LOAD
            try:
            
                ## LOADING PLUMED FILE
                plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sim,
                                                                                          plumed_input_file=plumed_input_file,
                                                                                          time_stride=time_stride )
                
                ## GETTING TIME ARRAY AND Z DISTANCE
                time_array, z_dist = get_time_array_and_z_dist_from_com_distances(path_to_sim = path_to_sim)        
                
                ## FINDING TIME ARRAY
                covar_time_array = covar_output['time'].to_numpy()
                
                ## FINDING INDICES THAT MATCH IN TERMS OF TIME
                indices_match = np.nonzero(np.in1d(time_array,covar_time_array))[0]
                
                ## GETTING Z DISTANCE
                z_distance_match = z_dist[indices_match]
                
                ## ADDING TO COVAR
                covar_output['z_dist'] = z_distance_match[:]
                
                ## STORING
                covar_storage[config_name] = covar_output
            
            except FileNotFoundError:
                print("--> Skipping %s since it is missing input file!"%(config_name))
                pass
            
        ## STORING FOR EACH KEY
        sims_covar_storage[each_sim] = covar_storage
        
    return sims_covar_storage


### FUNCTION TO CONVERT COVAR TO SIM OUTPUT
def convert_covar_to_output_dict(sims_covar_storage,
                                 last_time_ps = 50000,
                                 num_split = 1,
                                 lm_groups =[
                                            'HEADGRPS',
                                            'TAILGRPS',
                                            ],
                                np_grps=[
                                        'GOLD',
                                        'ALK',
                                        'PEG',
                                        'RGRP',
                                        'NGRP',
                                        ]):
    '''
    This function converts the output of the covars into ouptut dict that could 
    be plotted.
    INPUTS:
        sims_covar_storage: [dict]
            dictionary containing information for each simulation
        last_time_ps: [int]
            last time in picoseocnds that is desired. For instance, 50000 means 
            that you want the last 50 ns for the contacts calculation.
        lm_groups: [list]
            list of lipid membrane groups
        np_grps: [list]
            list of nanoparticle groups
        num_split: [int]
            splitting the last n trajectories. If this is 1, then no splitting 
            is performed. This tool is useful when trying to get average and 
            standard deviation by splitting a trajectory into 2, or even 3. 
    OUTPUTS:
        sims_output_dict: [dict]
            outputs containing information about contacts and mean distance.
    '''
    ## CONVERTING KEYS INTO OUTPUT DICT
    sims_output_dict = {}
    
    for sim_key in sims_covar_storage:
        ## DEFINING COVAR STORAGE
        covar_storage = sims_covar_storage[sim_key]
    
        ## FINDING FIRST DICT
        first_dict = covar_storage[next(iter(covar_storage))]
        
        ## GETTING ALL COLUMNS
        cols = first_dict.columns.to_list()
        
        ## FINDING ALL LABELS
        nplm_keys = [ nplm_shorten_label(each_key,
                                         return_separated = True,
                                         return_orig_label = True) for each_key in cols if each_key.startswith("NP")]
        
        ## CREATING SEPARATED DICT
        output_dict = {}
        
        ## LOOPING THROUGH LM GROUPS
        for each_lm in lm_groups:
            ## FINDING ALL GROUPS MATCHING
            idx_matched = [idx for idx, name_list in enumerate(nplm_keys) 
                        if name_list[1] == each_lm and name_list[0] in np_grps ]
            
            ## GETTING THE COLUMN HEADINGS
            column_headings = [ nplm_keys[each_idx][-1] for each_idx in idx_matched ]
            
            ## INCLUDING Z DIST
            column_headings = ['z_dist'] + column_headings
            column_with_time = ['time'] + column_headings
            
            ## STORING DICT
            mean_dict_storage = []

            ## LOOPING THROUGH EACH DATAFRAME
            for each_config in covar_storage:
                ## DEFINING CURRENT DICT
                current_dict = covar_storage[each_config][column_with_time]
                
                ## FINDING DICT
                truncated_dict = truncate_df_based_on_time(current_dict = current_dict,
                                                           last_time_ps = last_time_ps,
                                                           time_label='time')
                ## SPLITTING DICT IF GREATER THAN 1
                if num_split > 1:
                    print("Taking data from %s last trajectory (ps) from %.3f ps: %.3f"%(each_config, truncated_dict.iloc[-1][0], last_time_ps))
                    print("Splitting by number of splits: %d"%(num_split))
                    ## SPLITTING BY NUMBER OF SPLITS
                    dfs = np.array_split(truncated_dict, num_split)
                    
                    ## STORING
                    dfs_storage_avg = [ each_df.mean() for each_df in dfs]
                    
                    ## GETTING MEAN AND STD
                    dfs_combined = pd.concat(dfs_storage_avg, axis = 1).T
                    
                    ## RENAMING
                    config_df = dfs_combined
                else:
                    config_df = truncated_dict
                
                ## AVERAGING ACROSS ROWS
                mean_dict = config_df.mean()
                ## GETTING STD
                std_dict = config_df.std()
                ## GETTING SPECIFIC COLUMNS
                mean_dict_specific_cols = mean_dict[column_headings] # .to_dict()
                ## GETTING STD DICT
                std_dict_specific_cols =  std_dict[column_headings] # .to_dict()
                
                ## ADDING SUFFIX
                mean_dict_specific_cols = mean_dict_specific_cols.add_suffix("_avg")
                std_dict_specific_cols = std_dict_specific_cols.add_suffix("_std")
                
                ## GETTING COMBINED
                combined_cols = pd.concat([mean_dict_specific_cols, std_dict_specific_cols ]).to_dict()
                ## APPENDING
                mean_dict_storage.append(combined_cols)
                
            ## CREATING A DATAFRAME
            mean_df = pd.DataFrame(mean_dict_storage)
            
            ## ADDING Z VALUES
            mean_df['folder'] = covar_storage.keys()
                    
            ## SORTING
            mean_df = mean_df.sort_values(by='z_dist_avg',
                                          ).reset_index(drop=True)
            
            ## STORING
            output_dict[each_lm] = mean_df
        ## STORING
        sims_output_dict[sim_key] = output_dict
        
    return sims_output_dict


### FUNCTION TO PLOT FOR SPECIFIC CONTACTS VERSUS DISTANCE
def plot_specific_contacts_vs_distance(output_dict,
                                       each_lm,
                                       xlabel = "z (nm)",
                                       ylabel = "# DOPC tail contacts",
                                       fig = None,
                                       ax = None,
                                       figsize = FIGURE_SIZE,
                                       color = None,
                                       default_sort_list = [ 'ALK', 'PEG', 'NGRP_RGRP' ],
                                        ## DEFINING CONFIG LIBRARY
                                        errorbar_format={
                                                'linestyle' : "-",
                                                "fmt": ".",
                                                "capthick": 1.5,
                                                "markersize": 8,
                                                "capsize": 3,
                                                "elinewidth": 1.5,
                                                }):
    '''
    This function plots specific contacts versus distance
    INPUTS:
        output_dict: [dict]
            output dictionary for contacts versus distance
        each_lm: [str]
            label of lipid membrane that you want to show
        default_sort_list:[list]
            list to sort the contacts
        errorbar_format: [dict]
            error bar format dictionary
    
    '''
    
    ## CREATING FIGURE
    if fig is None or ax is None:
        fig, ax = plot_funcs.create_fig_based_on_cm(FIGURE_SIZE)
    
    ## DEFINING CURRENT DICT
    current_dict = output_dict[each_lm]
    
    ## GETTING EACH NAME
    np_name = [ nplm_shorten_label(each_key[:-4], # Removing _avg / _std
                                 want_np_only = True,
                                 return_orig_label = True) for each_key in current_dict.columns if each_key.startswith("NP")]

    ## GETTING UNIQUE
    np_name = [ list(x) for x in set(tuple(x) for x in np_name)] 
    ## SORTING LIST
    np_name.sort()
    
    ## SORTING
    np_names_first = [ each_name[0] for each_name in np_name]
    
    ## DEFINING NEW LIST
    new_np_name_idx = []
    
    ## DEFAULT LIST
    for each_key in default_sort_list:
        if each_key in np_names_first:
            idx = np_names_first.index(each_key)
            ## STORING IDX
            new_np_name_idx.append(idx)
    
    ## LOCATING IDX AND APPENDING IDXES 
    idx_not_found = np.delete(np.arange(len(np_names_first)), new_np_name_idx)
    idx_sorted = new_np_name_idx + list(idx_not_found)
    
    ## SORTING
    np_name = [ np_name[idx] for idx in idx_sorted ]    
    
    ## GETTING LABELS
    labels_orig_name = [name[1] for name in np_name]
    labels = [ name[0] for name in np_name]
    if color is None:
        colors = [ GROUP_COLOR_DICT[each_y] if each_y in GROUP_COLOR_DICT else 'k' for each_y in labels ]      
    else:
        colors = color

    
    ## GETTING THE ERROR
    y_err = current_dict[[ each_key + '_std' for each_key in labels_orig_name ]]
    x_err = current_dict['z_dist_std']
    
    ## LOOPING
    for idx, each_label in enumerate(labels):
        ## DEFINING X
        x = np.array(current_dict['z_dist_avg'])
        y = np.array(current_dict[ labels_orig_name[idx] + '_avg'  ])
        y_err =  np.array(current_dict[ labels_orig_name[idx] + '_std'  ])
        
        ## DEFINING COLOR
        if color is None:
            color = colors[idx]
        
        ## ADDING LABEL IF NOT PRESENT
        if 'label' not in errorbar_format:
            errorbar_format['label'] = each_label
        ## ADDING ERROR BAR AS SHADED
        ax.fill_between(x, y-y_err, y+y_err, color = color, alpha = 0.3)
        
        ## PLOTTING
        ax.errorbar(x = x,
                    y = y,
                    color = color,
                    **errorbar_format
                    )
    ## SETTING AXIS LABELS
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ## GETTING FIG
    fig = plt.gcf()
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT FOR EACH
def plot_contacts_vs_distance_plumed(sims_output_dict,
                                     xlabel = "z (nm)",
                                     ylabel = "# DOPC tail contacts",
                                     figsize = FIGURE_SIZE,
                                     fig_prefix = ""):
    '''
    The purpose of this function is to plot the contacts versus distance 
    using PLUMED coordination results. This function will simply generate 
    individual plots for each simulation listed. Then it will plot for 
    each lipid membrane label.
    INPUTS:
        sims_output_dict: [dict]
            dictionary with covar information  
        xlabel: [str]
            x labels
        ylabel: [str]
            y labels
        figsize: [tuple]
            figure size in cm
        fig_prefix: [str]
            figure prefix to include in the label
    OUTPUTS:
        figs, axs, labels_storage:
            figure, axis, and labels for the plots
        
    '''
    ## STORING FIG
    figs =[]
    axs = []
    labels_storage = []
    ## LOOPING THROUGH SIM KEYS
    for sim_key in sims_output_dict:
        ## DEFINING OUTPUT DICT
        output_dict = sims_output_dict[sim_key]
        
        ## TURNING FIG AND AX OFF
        fig = None
        ax = None
        
        ## LOOPING THROUGH EACH LIPID MEMBRNAE
        for each_lm in output_dict:
            ## GENERATING FIGURES
            fig, ax = plot_specific_contacts_vs_distance(output_dict = output_dict,
                                                         each_lm = each_lm,
                                                         xlabel = xlabel,
                                                         ylabel = ylabel,
                                                         fig = fig,
                                                         ax = ax)
            
            ## FIGURE NAME
            figure_name = fig_prefix + "contacts_vs_z_dist-%s-"%(each_lm) + sim_key
            
            ## STORING
            figs.append(fig)
            axs.append(ax)
            labels_storage.append(figure_name)
    return figs, axs, labels_storage

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING SIM LIST
    sim_list_for_dict = [
#            'unbiased_ROT012_1.300',
#            'unbiased_ROT012_1.900',
#            'unbiased_ROT001_1.700',
#            'unbiased_ROT001_3.500',
#            'unbiased_ROT012_3.500',

#            'plumed_pulling_ROT001',
#            'plumed_pulling_ROT012',
#            'plumed_unbiased_after_US_ROT012',
#            'plumed_unbiased_after_US_ROT001',
#            'pullthenunbias_ROT004',
#            'pullthenunbias_ROT012',
            
#            'plumed_unbiased_20ns_US_ROT012_40',
#            'plumed_unbiased_20ns_US_ROT017_120',
#            'plumed_unbiased_20ns_US_ROT017_40',
            
#            'unbiased_after_us_ROT017_1.900nm',
#            'modified_FF_unbiased_after_us_ROT017_1.900nm'
            'plumed_unbiased_20ns_US_ROT012_40',
            'plumed_unbiased_15ns_US_ROT012_40',
            'plumed_unbiased_10ns_US_ROT012_40',
            ]
    
    sim_list_for_dict=[each_key for each_key in NPLM_SIM_DICT 
                       if each_key.startswith("modified_FF_hydrophobic_pmf_unbiased_ROT017")] # _20ns_US
#    # or each_key.startswith("unbiased_after_us_")
    
    SIM_LIST_RELABEL={
            'plumed_pulling_ROT012': {
                    'label': 'C$_{10}$',
                    'style': {'linestyle': '-',}
                    },
            'plumed_pulling_ROT001': {
                    'label': 'C$_{1}$',
                    'style': {'linestyle': '--',}
                    },
            }
    
    ## DEFINING SPECIFIC TO PLOT
    list_to_plot = [ 
#            'NP_LIGAND_HEAVY_ATOMS-LM_TAILGRPS',
#            'NP_LIGAND_CARBON_ONLY-LM_TAILGRPS',
            'NP_ALK-LM_TAILGRPS',
            'NP_PEG-LM_TAILGRPS',
            'NP_RGRP-LM_TAILGRPS',
            'NP_ALK_RGRP-LM_TAILGRPS',
            'NP_ALK_RGRP-LM_HEADGRPS',
       ]
    
    ## DEFINING RELABEL
    RELABEL_DICT = {
            'LIGAND_HEAVY_ATOMS-TAILGRPS' : 'All',
            'ALK_RGRP-TAILGRPS': 'Hydrophobic',
            'ALK-TAILGRPS': 'ALK',
            'RGRP-TAILGRPS': 'R',
            'PEG-TAILGRPS': 'PEG',
            }
    
    ## DEFINING COLOR DICT
    COLOR_DICT={
            'NP_LIGAND_HEAVY_ATOMS-LM_TAILGRPS': 'black',
            'NP_LIGAND_CARBON_ONLY-LM_TAILGRPS': 'blue',    
            'NP_ALK_RGRP-LM_TAILGRPS': 'black',
            'NP_ALK-LM_TAILGRPS': 'gray',
            'NP_RGRP-LM_TAILGRPS': 'red',
            'NP_PEG-LM_TAILGRPS': 'green',
            'NP_ALK_RGRP-LM_HEADGRPS': 'purple',
            }
    
    ## SEEING IF YOU WANT COMBINED PLOTS
    want_combined_plots = False
    # True
    
    # if want_combined_plots is True:
    ## LOOPING THROUGH EACH SIM
    for idx, each_sim in enumerate(sim_list_for_dict):
        ## GETTING MAIN SIM
        main_sim_dir = NPLM_SIM_DICT[each_sim]['main_sim_dir']
        specific_sim = NPLM_SIM_DICT[each_sim]['specific_sim']
        
        ## GETTING ALL JOB TYPES
        job_types = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                   main_sim_dir = main_sim_dir,
                                   specific_sim = specific_sim,)
        
        
        ## DEFINING PATH TO SIMULATION
        path_to_sims = job_types.path_simulation_list[0]
        
        if each_sim.startswith('plumed') and 'unbiased' not in each_sim:
            time_stride=10
        else:
            time_stride = TIME_STRIDE
        
        ## LOADING PLUMED FILE
        plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sims,
                                                                                  plumed_input_file=PLUMED_INPUT_FILE,
                                                                                  time_stride=time_stride )
        
        ## GETTING LABELS
        labels = [ nplm_shorten_label(each_y) for each_y in list_to_plot]
        
        ## SEEING IF IN RELABELING DICT
        labels = [ RELABEL_DICT[each_label] if each_label in RELABEL_DICT else each_label for each_label in labels]
        
        colors = [ COLOR_DICT[each_y] if each_y in COLOR_DICT else 'k' for each_y in list_to_plot ]
        
        ## FOR ZERO
        if idx == 0:
            ax = None
            
        if want_combined_plots is True:
            ax = ax
            
            ## DEFINING LABELS
            labels = [ SIM_LIST_RELABEL[each_sim]['label'] + '-' + each_label 
                          if each_sim in SIM_LIST_RELABEL else each_label for each_label in labels]
            
            ## DEFINING STYLES
            if each_sim in SIM_LIST_RELABEL:
                styles = SIM_LIST_RELABEL[each_sim]['style']
            else:
                styles = {}
            
            
        else:
            ax = None
            styles = {}
        ## DEFINING CONFIG LIBRARY
        figsize=plot_funcs.cm2inch( *FIGURE_SIZE )
        ax = covar_output.plot(x='time', 
                               y=list_to_plot, 
                               figsize = figsize,
                               label = labels,
                               color = colors,
                               ax = ax,
                               **styles)
        
        ## GETTING FIG
        fig = plt.gcf()
        
        ## DEFINING LABELS
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Hydrophobic contacts")
        
        ## REMOVING LEGEND
#        ax.get_legend().remove()
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        ## FIGURE NAME
        figure_name = "heavy_vs_carbon_contacts_unbiased-" + specific_sim
        if want_combined_plots is False:
            ## SETTING AXIS
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(IMAGE_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)
    if want_combined_plots is True:
        ## SETTING AXIS
        figure_name = "combined_unbiased"
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'png', 
                     save_fig=True,)            
    
                     
    #%% DRAWING CONTACTS VS. DISTANCE FOR UMBRELLA SAMPLING SIMULATIONS
    
    ## DEFINING SIM LIST
    sim_list_for_dict = [
            'us_forward_R12',
            'us_reverse_R12',
            'us_forward_R01',
            'us_reverse_R01',
            ]
    
    ## DEFINING GROUPS TO PRINT
    np_grps=[
            'GOLD',
            'ALK',
            'PEG',
            'RGRP',
            'NGRP',
            ]
        
    ## DEFINING LIPID MEMBRANE GROUPS
    lm_groups =[
            'HEADGRPS',
            'TAILGRPS',
            ]
    
    ## GETTING COVAR DETAILS
    sims_covar_storage = load_plumed_contacts_and_com_distances(sim_list_for_dict = sim_list_for_dict,
                                                                plumed_input_file = PLUMED_INPUT_FILE,
                                                                time_stride = TIME_STRIDE)
    
    #%%
    ## FINDING OUTPUTS FOR SIMS
    sims_output_dict = convert_covar_to_output_dict(sims_covar_storage,
                                     last_time_ps = 50000)
        
    #%% PLOTTING FOR EACH GROUP
    ## PLOTTING
    figs, axs, labels = plot_contacts_vs_distance_plumed(sims_output_dict,
                                                         xlabel = "z (nm)",
                                                         ylabel = "# DOPC tail contacts",
                                                         figsize = plot_funcs.cm2inch( *FIGURE_SIZE ))
    
    

    #%%
    
    ## SETTING AXIS
    plot_funcs.store_figure(fig = fig, 
                 path = os.path.join(IMAGE_LOC,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
            
            
        
    #%% GETTING TABLE OF VALUES FOR UNBIASED SIMULATIONS
    
    ## GETTING ALL UNBIASED SIMS
    sim_type_list = [each_key for each_key in NPLM_SIM_DICT if 'unbiased' in each_key]    
    
    ## SORTING
    sim_type_list.sort()
    
    ## CREATING STORAGE
    storage_list = []
    
    ## DEFINING SPECIFIC TO PLOT
    list_to_store = [ 
            'NP_LIGAND_HEAVY_ATOMS-LM_HEADGRPS',
            'NP_LIGAND_HEAVY_ATOMS-LM_TAILGRPS',
            'NP_LIGAND_CARBON_ONLY-LM_HEADGRPS',
            'NP_LIGAND_CARBON_ONLY-LM_TAILGRPS',
            'NP_ALK_RGRP-LM_HEADGRPS',
            'NP_ALK_RGRP-LM_TAILGRPS',
       ]
    
    ## DEFINING OUTPUT DICT
    output_list = []
    
    ## DEFINING SIMULATION TYPE
    for sim_idx, sim_type in enumerate(sim_type_list):
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ### GETTING DETAILS FROM UMBRELLA SAMPLING SIMULATIONS
        if "unbiased" in sim_type:
            ## FINDING UMBRELLA SAMPLING THAT MATCHES IT
            if "rev" in sim_type:
                if "ROT001" in sim_type:
                    us_dir_key = "us_reverse_R01"
                elif "ROT012" in sim_type:
                    us_dir_key = "us_reverse_R12"
                else:
                    print("Error! No reverse sim of this type is found: %s"%(sim_type) )
            else:
                if "ROT001" in sim_type:
                    us_dir_key = "us_forward_R01"
                elif "ROT012" in sim_type:
                    us_dir_key = "us_forward_R12"
                else:
                    print("Error! No reverse sim of this type is found: %s"%(sim_type) )
        ## FINDING CONFIG
        config_key = sim_type.split('_')[2]
        
        print("Umbrella sampling extraction point (%s): %s, config: %s"%(sim_type, us_dir_key, config_key) )
        
        us_main_sim_dir= NPLM_SIM_DICT[us_dir_key]['main_sim_dir']
        us_specific_sim= NPLM_SIM_DICT[us_dir_key]['specific_sim']
        
        ## GETTING JOB INFORMATION
        job_info_us = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                  main_sim_dir = us_main_sim_dir,
                                  specific_sim = us_specific_sim)
        
        idx = job_info_us.config_library.index(config_key)
        path_to_sim = job_info_us.path_simulation_list[idx]
        
        ## LOADING PLUMED FILE
        plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sim,
                                                                                  plumed_input_file=PLUMED_INPUT_FILE,
                                                                                  time_stride=TIME_STRIDE )
        
        ## TRUNCATING THE DICT
        truncated_dict = truncate_df_based_on_time(current_dict = covar_output,
                                                   last_time_ps=10000, # Last 10 ns
                                                   time_label='time')
        
        ## AVERAGING ACROSS ROWS
        mean_dict = truncated_dict.mean()
        ## GETTING SPECIFIC COLUMSN
        mean_dict_specific_cols = mean_dict[list_to_store].to_dict()
        
        ## adding data key
        mean_dict_specific_cols = {'sim_key': sim_type, **mean_dict_specific_cols}
        
        ## APPENDING
        output_list.append(mean_dict_specific_cols)
    ## CREATING DATAFRMAE
    df_output = pd.DataFrame(output_list)
    
    ## WRITING TO DATAFRAME
    csv_name="unbiased_coordination_plumed.csv"
    path_to_csv = os.path.join(IMAGE_LOC,csv_name)
    df_output.to_csv(path_to_csv)
    
    
    #%% TO PLOT PLUMED HYDROPHOBIC CONTACTS VERSUS ALL CONTACTS
    
    ## DEFINING TIME STRIDE
    time_stride = 100
    
    ## DEFINING SIM LIST
    sim_list_for_dict = [
            'us-PLUMED_ROT012',
            'us-PLUMED_ROT001',
            ]
    
    ## DEFINING DESIRED INDEXES
    desired_keys = {
            'y': 'NP_ALK_RGRP-LM_HEADGRPS',
            'x': 'NP_LIGAND_HEAVY_ATOMS-LM_TAILGRPS',
            }
    
    ## DEFINING SIMULATION TYPE
    for sim_idx, sim_type in enumerate(sim_list_for_dict):
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING PATH TO SIMULATION
        path_to_sims = job_info.path_simulation_list
        
        ## CREATING LIST
        stored_df = []
        
        ## LOOPING THROUGH AND RESTORING
        for each_sim_path in path_to_sims:
            ## GETTING BASENAME
            current_basename = os.path.basename(each_sim_path)
            
            ## LOADING PLUMED FILE
            plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=each_sim_path,
                                                                                      plumed_input_file=PLUMED_INPUT_FILE,
                                                                                      time_stride=time_stride )
            
            ## STORING INFORMATION
            output_dict = {
                    'label': float(current_basename),
                    'x': np.array(covar_output[desired_keys['x']]),
                    'y': np.array(covar_output[desired_keys['y']]),
                    'time': np.array(covar_output['time'])
                    }
            ## APPENDING
            stored_df.append(output_dict)

        ## CREATING DATAFRAME
        df = pd.DataFrame(stored_df)
        
        ## SORTING BY LABEL
        df = df.sort_values(by='label').reindex()
        
        #### PLOTTING ALL CONTACTS VS. HYDROPHOBIC CONTACTS
        
        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(FIGURE_SIZE)
        
        ## CREATING AXIS LABEL
        ax.set_xlabel('All contacts') # desired_keys['x']
        ax.set_ylabel('Hydrophobic contacts') # desired_keys['y']
        
        ## PLOTTING EACH ONE
        for idx, row in df.iterrows():
            ## GETTING X AND Y
            x = row['x']
            y = row['y']
            label = row['label']
            
            ## PLOTING
            ax.scatter(x, y, label = label)
        
        ## TIGHT LAYOUT
        fig.tight_layout()

        ## SETTING AXIS
        figure_name = "US_HYDROPHOBIC_VS_CONTACTS-%s"%(sim_type)
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'png', 
                     save_fig=True,)
                     
                     
        
        #### PLOTTING HYDROPHOBIC CONTACTS OVER TIME

        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(FIGURE_SIZE)
        
        ## DEFINING FREQUENCY
        freq = 3
        df_truncated = df.iloc[::freq, :]
        
        ## LOOPING
        image_dict = {
                'Hydrophobic contacts': 'y',
                'All contacts': 'x',
                }
        
        for each_key in image_dict:
            ## CREATING PLOT
            fig, ax = plot_funcs.create_fig_based_on_cm(FIGURE_SIZE)
            ## CREATING AXIS LABEL
            ax.set_xlabel('Time (ps)') # desired_keys['x']
            ax.set_ylabel(each_key) # desired_keys['y']
            
            ## GENERATING COLOR 
            colors = plot_funcs.get_cmap(len(df), 'hsv')
            
            ## ADDING AXIS LABEL AT ZERO
            # ax.axhline(y=0, color = 'k', linestyle = '--')
            
            ## PLOTTING EACH ONE
            for idx, row in df_truncated.iterrows():
                ## FINDING TIME AND CONTACTS
                x = row['time']
                y = row[image_dict[each_key]]
                
                ## NORMALIZE BY Y
                # y = y - y[-1]
                ## DEFINING LABEL
                label = int(row['label'])
                
                ## PLOTING
                line = ax.plot(x, y, 
                               label = label,
                               color = colors(idx))
                
            
            ## ADDING LEGEND
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            ## TIGHT LAYOUT
            fig.tight_layout()
            
            ## SETTING AXIS
            figure_name = "CONTACTS_VS_TIME-%s-%s"%(sim_type, each_key)
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(IMAGE_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)
                         
            
            
#            ## GETTING LABELS
#            all_labels = np.array(df_truncated.label).astype('int')
#            
#            sm = plt.cm.ScalarMappable(cmap=colors) #  norm=plt.normalize(min=0, max=1)
#            ## ADDING COLOR BAR
#            cbar = plt.colorbar(sm, ticks = np.linspace(0, 1, len(all_labels))) #  
#        
#            ## SETTING LABELS
#            cbar.ax.set_yticklabels(all_labels)
    
    
    
    
    