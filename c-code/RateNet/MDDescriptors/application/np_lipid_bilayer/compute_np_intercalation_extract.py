#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_np_intercalation_extract.py

This script is designed to analyze the intercalation of the nanoparticles.

Written by: Alex K. Chew (04/06/2020)

"""

## IMPORTING OS
import os
import numpy as np

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types

from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import \
    get_time_array_and_z_dist_from_com_distances
    
## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORTING INTERCALATION FUNCTIONS
from MDDescriptors.application.np_lipid_bilayer.compute_np_intercalation import \
    analyze_lm_groups,compute_np_intercalation, main_compute_np_intercalation, \
    plot_lig_intercolated_vs_time, get_indices_for_truncate_time
    
    
## IMPORTING EXTRACT TRAJ FUNCTION
from MDDescriptors.traj_tools.loop_traj_extract import load_results_pickle, ANALYSIS_FOLDER, RESULTS_PICKLE

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
SAVE_FIG = True
# False

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()


from MDDescriptors.core.pickle_tools import load_pickle_results, save_and_load_pickle


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    

    ## GOING THROUGH ALL SIMULATIONS
    sim_type_list = [
            'unbiased_ROT012_5.300',
            'unbiased_ROT012_5.300_rev',
            'unbiased_ROT012_1.300',
            'unbiased_ROT012_2.100',
            ]
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        path_to_sim = job_info.path_simulations
        
        ## DEFINING PATH TO ANALYSIS
        path_analysis = os.path.join(path_to_sim,
                                     ANALYSIS_FOLDER,
                                     main_compute_np_intercalation.__name__)
        
        ## DEFINING RESULTS
        path_pickle = os.path.join(path_analysis,
                                   RESULTS_PICKLE
                                   )
        
        ## LOADING CLASS OBJECT
        np_intercalation, lm_details, center_of_mass = load_pickle_results(file_path = path_pickle,
                                                                           verbose = True)[0]
        
        ## PLOTTING INTERCALATION
        fig, ax = plot_lig_intercolated_vs_time(time_array = np_intercalation.time_array,
                                                num_residue_within = np_intercalation.num_residue_within)
        
        ## ADDING 0 LINE
        ax.axhline(y=0, linestyle='--', color = 'k')
        
        ## FINDING AVERAGE INTERCOLATED (LAST 100)
        avg_intercolation = np.mean(np_intercalation.num_residue_within[-200:])
        
        ## FIG NAME
        figure_name = specific_sim + "_np_intercolation_vs_time"
        
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        
    #%% PLOTTING AVERAGE INTERCALATION FOR US SIMS
    
    ## UNBIASED SIMULATIONS
    sim_type_list = [
            'us_forward_R12',
            'us_reverse_R12',
            'us_forward_R01',
            'us_reverse_R01',
            ]
    
    last_time_ps = 50000
    
    ## CREATING DICTINOARY
    storage_dict = {}
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## STORING
        com_distance_storage = []
        avg_intercalation_storage = []
        
        ## LOOPING THROUGH LIST
        for path_to_sim in job_info.path_simulation_list:
            ## LOADING THE INTERCALATION RESULTS
            ## DEFINING PATH TO ANALYSIS
            path_analysis = os.path.join(path_to_sim,
                                         ANALYSIS_FOLDER,
                                         main_compute_np_intercalation.__name__)
            
            ## DEFINING RESULTS
            path_pickle = os.path.join(path_analysis,
                                       RESULTS_PICKLE
                                       )
            
            ## LOADING CLASS OBJECT
            np_intercalation, lm_details, center_of_mass = load_pickle_results(file_path = path_pickle,
                                                                               verbose = True)[0]
            
            ## FINDING AVERAGE INTERCOLATED (LAST 20 NS)
            avg_intercolation = np.mean(np_intercalation.num_residue_within[-200:])
            
            ## appending
            avg_intercalation_storage.append(avg_intercolation)
            
            ## FINDING COM
            ## GETTING TIME ARRAY AND Z DISTANCE
            time_array, z_dist = get_time_array_and_z_dist_from_com_distances(path_to_sim = path_to_sim)
            ## DEFINING DISTANCE AND TIME 
            
            ## GETTING INDICES
            indices = get_indices_for_truncate_time(time_array,
                                                    last_time_ps =last_time_ps)
            ## REDEFINING TIME ARRAY (starting at zero)
            time_array = time_array[indices] - time_array[indices][0]
            z_dist = z_dist[indices]
            
            ## GETTING AVG Z DISTANCE
            z_mean_dist = np.mean(z_dist)
            
            ## APPENDING
            com_distance_storage.append(z_mean_dist)
            
        ## STORING
        storage_dict[sim_type] = {
                'com_distance_storage': com_distance_storage,
                'avg_intercalation_storage': avg_intercalation_storage,
                }
        
    ## PLOTTING
    for key in storage_dict:
        com_distance_storage = storage_dict[key]['com_distance_storage']
        avg_intercalation_storage = storage_dict[key]['avg_intercalation_storage']
    
        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
        
        ## ADDING LABELS
        ax.set_xlabel("z (nm)")
        ax.set_ylabel("Avg. intercalation")
        
        ## SORTING THE ARRAY
        sort_index = np.argsort(com_distance_storage)
        
        ## UPDATED
        x  = np.array(com_distance_storage)[sort_index]
        y  = np.array(avg_intercalation_storage)[sort_index]
        
        ## PLOTTING
        ax.plot(x, y, color = 'k', linestyle = '-')
        
        ## ADDING 0 LINE
        ax.axhline(y=0, linestyle='--', color = 'k')
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        ## DEFINING LIMITS
        ax.set_ylim([-5, 70])
        ax.set_yticks( np.arange(0, 75, 10) )
        
        ax.set_xlim([1, 6])
        ax.set_xticks( np.arange(1, 7, 1) )
        
        figure_name = key + "_avg_intercalation"
        
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
    
    
    
            
    
    
    
    
    
    
    
    
    
    
    