# -*- coding: utf-8 -*-
"""
compute_nplm_density_maps_extract.py
The purpose of this script is to extract the density maps computed.

"""


import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import MDDescriptors.core.plot_tools as plot_tools
from MDDescriptors.core.decoder import decode_name

## DEFAULTS
plot_tools.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_tools.FIGURE_SIZES_DICT_CM['1_col_landscape']

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path

## IMPORTING EXTRACT TRAJ FUNCTOIN
from MDDescriptors.traj_tools.loop_traj_extract import extract_multiple_traj

## GETTING FUNCTION
from MDDescriptors.application.np_lipid_bilayer.compute_nplm_density_maps import plot_2d_histogram, main_compute_nplm_densities

## IMPORTING EXTRACT TRAJ FUNCTION
from MDDescriptors.traj_tools.loop_traj_extract import load_results_pickle, ANALYSIS_FOLDER, RESULTS_PICKLE

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types

## IMPORTING FUNCTION
from MDDescriptors.application.np_lipid_bilayer.compute_nplm_density_maps import main_compute_nplm_densities

## DEFINING FIGURE NAME
SAVE_FIG = True

## BACKEND FOR SAVING FIGURES
if SAVE_FIG is True:
    import matplotlib
    matplotlib.use('Agg')

### FUNCTION TO NORMALIZE DENSITIES
def normalize_densities(densities, norm_type="bin_volume_and_time"):
    '''
    The purpose of this function is to normalize densities.
    INPUTS:
        densities: [obj]
            object density class
        norm_type: [str]
            type of normalization
                bin_volume_and_time: [default] 
                    normalize by bin volume and time
    OUTPUTS:
        density_dict: [dict]
            dictionary of densities that is normalized
    '''
    ## COPYING DICTIONARY
    density_dict = densities.density_dict.copy()
    
    ## NORMALIZING BY BIN OLUME
    if norm_type == "bin_volume_and_time":
        ## GETTING EDGES
        _, edges = np.histogramdd(np.zeros((1, len(densities.bins))), bins=densities.bins, range=densities.arange, normed=False)
    
        ## GETTING R EDGES
        r_edge = edges[0]
        
        ## FINDING RADIUS ARRAY
        # r = 0.5 * (r_edge[1:] + r_edge[:-1])
        
        ## FINDING OF BIN RELATIVE TO R
        # Volume = pi * L ( r_2^2 - r_1^2 )
        bin_volume =  np.pi * densities.bin_width * (np.power(r_edge[1:], 2) - np.power(r_edge[:-1], 2))
        # 4*
        
        ## LOOPING THROUGH AND STORING
        for each_key in density_dict.keys():
            density_dict[each_key]['grid'] = density_dict[each_key]['grid'] / bin_volume[:, np.newaxis] / densities.time_length
    
    return density_dict

### FUNCTION TO GET CMAP AND ADD TRANSPARENCY
def get_cmap_with_transparency(cmap_type):
    '''
    The purpose of this function is to get the cmap with the 
    inclusion of transparency at the lower values.
    INPUTS:
        cmap_type: [str]
            cmap type
    OUTPUTS:
        my_cmap: [obj]
            cmap object with updated transparency
    '''
    from matplotlib.colors import ListedColormap
    ## GETTING CMAP
    cmap  = plt.cm.__dict__[cmap_type]
    
    ##  SETTING ALPHA
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    
    ## GENERATING CMAP
    my_cmap = ListedColormap(my_cmap)
    return my_cmap

## FUNCTION TO GET CMAP TYPE
def get_cmap_type(current_type = 'GNP'):
    ''' This finds all cmap types '''

    ## SELECTING TYPES
    if 'GNP' in current_type:
        cmap_type = 'Blues'
    elif 'LM' in current_type:
        cmap_type = 'Reds'
    elif 'CL' in current_type:
        cmap_type = 'Greens'
    return cmap_type

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING SIMULATION DIRECTORY
#    simulation_dir=r"20200120-US-sims_NPLM_rerun_stampede"
#    
#    ## DEFINING SPECIFIC DIRECTORY
#    specific_dir = r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    #########################
    ### TUNABLE VARIABLES ###
    #########################
    
    ## DEFINING V MIN AND VMAX
    v_range={'vmin': 0,
             'vmax': 40,
             'alpha': 1}
    
    default_vmax = v_range['vmax']

    ## DEFINING DESIRED TYPES
    desired_types = ['GNP-RGRP']
    # ['GNP', 'GNP-GOLD', 'GNP-RGRP', 'LM-HEADGRPS', 'LM-TAIL_GRPS', "CL" , "LM", "GNP-PEG", "GNP-ALK"]

    ## DEFINING MERGED TYPES
    merged_type_list = [
#                        ['LM-HEADGRPS', 'GNP-RGRP'],
                        ['LM-TAIL_GRPS', 'GNP-ALK'],
                        ['LM-TAIL_GRPS', 'GNP-RGRP'],
#                        ['LM-HEADGRPS', 'GNP-PEG'],
                        ['LM', 'GNP']
                        ]

    ## DEFINING PARENT SIM PATH
    parent_sim_path = PARENT_SIM_PATH    

    sim_type = 'us_reverse_R12'
    # 'us_reverse_R01'
    # 
    ## GETTING DICTIONARY
    sim_folders = NPLM_SIM_DICT[sim_type]
    
    ## GETTING JOB INFOMRATION
    job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                              main_sim_dir = sim_folders['main_sim_dir'],
                              specific_sim = sim_folders['specific_sim'])

    ## LOOPING THROGUH THE LIST
    for sim_idx, sim_path in enumerate(job_info.path_simulation_list):
        # if sim_idx ==0:
        if job_info.config_library[sim_idx] == "5.300" : # "5.300"
            ## CLOSING PLOTS
            plt.close('all')
            ## LOADING RESULTS
            densities = load_results_pickle(path_to_sim = sim_path,
                                            func = main_compute_nplm_densities)
            
            ## NORMALIZING DENSITIES
            density_dict = normalize_densities(densities, norm_type="bin_volume_and_time")
        
    
            ### NORMAL TYPES
            for type_idx, each_type in enumerate(desired_types):
                ## GETTING CMAP
                cmap_type = get_cmap_type(current_type = each_type)
                my_cmap = get_cmap_with_transparency(cmap_type = cmap_type)
                
                ## CHANGING RANGE FOR CL
                if each_type == "CL":
                    v_range['vmax'] = 1
                else:
                    v_range['vmax'] = default_vmax
                
                ## PLOTTING
                fig, ax = plot_2d_histogram(grid = density_dict[each_type]['grid'],
                                            r_range = densities.r_range, 
                                            z_range = densities.z_range,
                                            increments = 1,
                                            cmap_type = my_cmap, # cmap_type
                                            interpolation = 'gaussian', # 'gaussian', # 'none'
                                            want_color_bar = True,
                                            **v_range
                                            )
                
                ## DEFINING FIG NAME
                figure_name = '_'.join([job_info.specific_sim,  job_info.config_library[sim_idx], each_type])
                
                ## STORING FIGURE
                plot_tools.save_fig_png(fig = fig,
                                        label = os.path.join(IMAGE_LOC,
                                                          figure_name),
                                         save_fig = SAVE_FIG)
            
            ########################
            ### FOR MERGED TYPES ###
            ########################
            
            ## LOOPING THROUGH EACH
            for merged_types in merged_type_list:
            
                ## LOOPING AND STORING
                for type_idx, each_type in enumerate(merged_types):
                    ## SET FIGURE TO NONE
                    if type_idx == 0:
                        fig, ax = None, None
        
                    ## CHANGING RANGE FOR CL
                    if each_type == "CL":
                        v_range['vmax'] = 1
                    else:
                        v_range['vmax'] = default_vmax
                        
                    ## GETTING CMAP
                    cmap_type = get_cmap_type(current_type = each_type)
                    my_cmap = get_cmap_with_transparency(cmap_type = cmap_type)
                        
                    ## PLOTTING
                    fig, ax = plot_2d_histogram(grid = density_dict[each_type]['grid'],
                                                r_range = densities.r_range, 
                                                z_range = densities.z_range,
                                                increments = 1,
                                                cmap_type = my_cmap, # cmap_type
                                                interpolation = 'gaussian', # 'gaussian', # 'none'
                                                fig = fig,
                                                ax = ax,
                                                want_color_bar = True,
                                                **v_range
                                                )
                    
                ## DEFINING FIG NAME
                figure_name = '_'.join([job_info.specific_sim, job_info.config_library[sim_idx], 'merge', merged_types[0], merged_types[1]])
                
                ## STORING FIGURE
                plot_tools.save_fig_png(fig = fig,
                                        label = os.path.join(IMAGE_LOC,
                                                          figure_name),
                                         save_fig = SAVE_FIG)
#
#
#
#    #%%
#    ## DEFINING SIMULATION DIRECTORY
#    simulation_dir=r"20200124-US-sims_NPLM_reverse_stampede"
#    
#    ## DEFINING SPECIFIC DIRECTORY
#    specific_dir = r"USrev-1.5_5_0.2-pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT012_1"
#
#    ## DEFINING RELATIVE PATH TO SIM
#    relative_path= "4_simulations"
#
#    ## DEFINING PATH TO LIST
#    path_to_sim_list = os.path.join(MAIN_DIR, simulation_dir, specific_dir, relative_path)    
#    
#    ## GETTING TRAJECTORY OUTPUT
#    traj_output = extract_multiple_traj(path_to_sim_list = path_to_sim_list )
#
#
#
#
#    # 'GNP', 'GNP-GOLD', 'GNP-RGRP', 'LM-HEADGRPS', 'LM-TAIL_GRPS', "CL" , "LM", "GNP-PEG", "GNP-ALK"
#    # ['LM-HEADGRPS', 'GNP']# 'LM', 'LM-HEADGRPS'
#    # 'LM-HEADGRPS', "GNP", 'LM-TAIL_GRPS', 'LM-HEADGRPS' 
#    # 'LM-HEADGRPS'
#    # 'LM-HEADGRPS', 'GNP-RGRP'
#    # ['GNP', 'GNP-GOLD', 'GNP-RGRP', 'LM-HEADGRPS', 'LM-TAIL_GRPS', "CL" , "LM", "GNP-PEG", "GNP-ALK"]
#
#                
#    ## LOOPING THROUGH EACH
#    for idx, sim_path in enumerate(traj_output.full_sim_list):
#        ## CLOSING ALL PLOTS
#        plt.close('all')
#        ## GETTING BASENAME
#        sim_basename = os.path.basename(sim_path)
#        ## LOADING ONE
#        densities = traj_output.load_results(idx =idx,
#                                           func = main_compute_nplm_densities)
#        
#        ## NORMALIZING DENSITIES
#        density_dict = normalize_densities(densities, norm_type="bin_volume_and_time")
#        
#        ### NORMAL TYPES
#        for type_idx, each_type in enumerate(desired_types):
#            ## GETTING CMAP
#            cmap_type = get_cmap_type(current_type = each_type)
#            my_cmap = get_cmap_with_transparency(cmap_type = cmap_type)
#            
#            ## CHANGING RANGE FOR CL
#            if each_type == "CL":
#                v_range['vmax'] = 1
#            else:
#                v_range['vmax'] = default_vmax
#            
#            ## PLOTTING
#            fig, ax = plot_2d_histogram(grid = density_dict[each_type]['grid'],
#                                        r_range = densities.r_range, 
#                                        z_range = densities.z_range,
#                                        increments = 1,
#                                        cmap_type = my_cmap, # cmap_type
#                                        interpolation = 'gaussian', # 'gaussian', # 'none'
#                                        want_color_bar = True,
#                                        **v_range
#                                        )
#            
#            ## DEFINING FIG NAME
#            figure_name = '_'.join([specific_dir, sim_basename, each_type])
#            
#            ## STORING FIGURE
#            plot_tools.save_fig_png(fig = fig,
#                                    label = os.path.join(IMAGE_LOC,
#                                                      figure_name),
#                                     save_fig = SAVE_FIG)
#        
#        ########################
#        ### FOR MERGED TYPES ###
#        ########################
#        
#        ## LOOPING AND STORING
#        for type_idx, each_type in enumerate(merged_types):
#            ## SET FIGURE TO NONE
#            if type_idx == 0:
#                fig, ax = None, None
#
#            ## CHANGING RANGE FOR CL
#            if each_type == "CL":
#                v_range['vmax'] = 1
#            else:
#                v_range['vmax'] = default_vmax
#                
#            ## GETTING CMAP
#            cmap_type = get_cmap_type(current_type = each_type)
#            my_cmap = get_cmap_with_transparency(cmap_type = cmap_type)
#                
#            ## PLOTTING
#            fig, ax = plot_2d_histogram(grid = density_dict[each_type]['grid'],
#                                        r_range = densities.r_range, 
#                                        z_range = densities.z_range,
#                                        increments = 1,
#                                        cmap_type = my_cmap, # cmap_type
#                                        interpolation = 'gaussian', # 'gaussian', # 'none'
#                                        fig = fig,
#                                        ax = ax,
#                                        want_color_bar = True,
#                                        **v_range
#                                        )
#            
#        ## DEFINING FIG NAME
#        figure_name = '_'.join([specific_dir, sim_basename, 'merge', merged_types[0], merged_types[1]])
#        
#        ## STORING FIGURE
#        plot_tools.save_fig_png(fig = fig,
#                                label = os.path.join(IMAGE_LOC,
#                                                  figure_name),
#                                 save_fig = SAVE_FIG)

