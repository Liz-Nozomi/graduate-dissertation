# -*- coding: utf-8 -*-
"""
umbrella_sampling_extraction.py
This script contains code to extract umbrella sampling simulations for the 
nplm project.

Written by: Alex K. Chew (03/17/2020)

"""


## IMPORTING OS
import os
import numpy as np

import matplotlib.pyplot as plt


## IMPORTING VARIABLES
from MDBuilder.core.global_vars import k_B # Boltzmann constant in kJ/(mol K)

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING STORING FIG LOCATION
import MDDescriptors.application.np_lipid_bilayer.global_vars as global_vars

## DEFINING GLOBAL VARS
STORE_FIG_LOC = global_vars.IMAGE_LOC
PARENT_DIR = global_vars.PARENT_SIM_PATH
SIM_DICT = global_vars.NPLM_SIM_DICT
SAVE_FIG = True

## IMPORTING FUNCTIONS FROM MD BUILDERS
from MDBuilder.umbrella_sampling.umbrella_sampling_analysis import read_xvg, plot_us_histogram, pmf_correction_entropic_effects, truncate_by_maximum, \
                                                                   plot_us_pmf, plot_sampling_time_pmf, plot_multiple_pmfs

#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    
    ## DEFINING SIM TYPE LIST
    sim_type_list = [
#                     'us_forward_R12',
#                     'us_reverse_R12',
                     'us_forward_R01',
                     'us_reverse_R01',
                     ]
            # 'us_reverse_R12']
    # 'us_reverse_R01', 
    
    ################
    ### DEFAULTS ###
    ################
    
    ## DEFINING ANALYSIS DIR
    analysis_dir_within_folder="5_analysis"
    ## DEFINING OUTPUT XVG FILE
    sampling_time_directory="sampling_time"
    histo_xvg       =   "histo.xvg"
    profile_xvg     =   "profile.xvg"
    temperature     = 300.00 # Kelvins
    end_truncate_dist = 0.05
    ## DEFINING UNITS
    units           =   "kJ/mol"
    
    ## LOOPING THROUGH SIM TYPE
    for idx, sim_type in enumerate(sim_type_list):

        ## GETTING DICTIONARY
        sim_folders = SIM_DICT[sim_type]
        
        ## GETTING PATH
        path_sim = os.path.join(PARENT_DIR,
                                sim_folders['main_sim_dir'],
                                sim_folders['specific_sim'])
        
        ## PATH TO ANALYSIS DIR
        path_to_analysis_directory = os.path.join(path_sim,
                                                  analysis_dir_within_folder)
        
        ## DEFINING ANALYSIS DIR
        analysis_dir = sim_folders['specific_sim']
        
        # #%% PLOTTING HISTOGRAM 
        
        ## DEFINING FULL PATHS
        path_histo_xvg = os.path.join(path_to_analysis_directory, histo_xvg)
        path_profile_xvg = os.path.join(path_to_analysis_directory,  profile_xvg)
        
        ## READING XVG FILES
        profile_xvg_data = read_xvg(path_profile_xvg)
        histo_xvg_data = read_xvg(path_histo_xvg)
        
        ##########################
        ### PLOTTING HISTOGRAM ###
        ##########################
    
        ## DEFINING FIGURE NAME
        figure_name = analysis_dir + '_histo' 
        
        ## PLOTTING UMBRELLA SAMPLING RANGE
        fig, ax = plot_us_histogram(histo_xvg_data, 
                                      save_fig = False, 
                                      data_range = np.arange(0.0,8.0,0.5),
                                      output_fig_name = figure_name,
                                      fig_size_cm = FIGURE_SIZE)
        ## STORING FIUURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        ####################
        ### PLOTTING PMF ###
        ####################
        
        ## PLOTTING PROFILE
        figure_name = analysis_dir + '_pmf' 
        fig, ax, distance, pmf_data = plot_us_pmf(profile_xvg_data, 
                                                  units = units, 
                                                  ylim= (-100, 900, 200) , # None,    (-1000, 100, 100)
                                                  data_range = np.arange(1.0,7.0,1),
                                                  want_correction = False, 
                                                  temp = temperature,
                                                  end_truncate_dist = end_truncate_dist,
                                                  want_smooth_savgol_filter = False, 
                                                  savgol_filter_params={'window_length':101, 'polyorder': 0, 'mode': 'nearest'},
                                                  output_fig_name = figure_name, 
                                                  save_fig = False)
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        
        # #%% SAMPLING TIME
        
        ## PLOTTING PROFILE
        figure_name = analysis_dir + '_samplingtime' 
        
        ## PLOTTING SAMPLING TIME
        fig, ax, data_storage = plot_sampling_time_pmf(path_to_analysis_directory, 
                                         sampling_time_directory, 
                                         units = units, 
                                         sampling_time_prefix="profile", 
                                         save_fig = False,
                                         want_data = True,
                                         fig_size_cm = (7.12, 6.325 ))
        
        
        ax.set_xticks(np.arange(1.0,8.0,1))
        ax.set_yticks(np.arange(-100, 900, 200))
        
        ## CREATING LEGEND
        ax.legend(loc = 'right')
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        ## ADDING LEGEND
        # ax.legend()
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
        #%%
        
#        ## SAVING FIGURE
#        plot_funcs.store_figure( fig = fig,
#                                 path = os.path.join(STORE_FIG_LOC,
#                                                     figure_name),
#                                 fig_extension = 'svg',
#                                 save_fig = True,
#                                 )
#        
        '''
        STORE_FIG_LOC = "/Users/alex/Box/VanLehnGroup/14.Grants/2020_XSEDE_PROPOSAL"
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 fig_extension = 'svg',
                                 save_fig = True,
                                 )
        
                                 
        # %% CONCATENATING DATA AND OUTPUTTING
        new_dict =  [ {each_key + '_x': data_storage[each_key]['x'],
                      each_key + '_y': data_storage[each_key]['y']} for each_key in data_storage ]
    
        ## COMBINING INTO ONE DICT
        import pandas as pd
        new_dict_combined = dict(pair for d in new_dict for pair in d.items())
        ## GETTING DATAFRAME
        df = pd.DataFrame(new_dict_combined)
        
        ## OUTPUTTING
        output_csv_path = os.path.join(STORE_FIG_LOC, "sampling_time_output.csv")
        print("Outputting to: %s "%(output_csv_path) )
        df.to_csv(output_csv_path)
        '''
        
        
    #%% COMBINING MULTIPLE PMFS - FOWARD AND REVERSE
    
    ## DEFINING TYPES
    analysis_types_dict={
            
            'forward':{
                'C1': 
                    {'type': 'us_forward_R01',
                       'specifications': { 'color': 'r' },
                       },
                'C10': 
                    { 'type': 'us_forward_R12',
                       'specifications': { 'color': 'k' },
                     },
                     },
            'reverse':{
                'C1': 
                    {'type': 'us_reverse_R01',
                     'specifications': { 'color': 'r' },
                     'color': 'r'},
                'C10': 
                    { 'type': 'us_reverse_R12',
                     'specifications': { 'color': 'k' },
                     },
                     },
#             'combined': {
#                'C1': 
#                    {'type': 'us_forward_R01',
#                       'specifications': { 'color': 'r',
#                                            'analysis_dir_within_folder': '5_analysis_combined'},
#                       
#                       },
#                'C10': 
#                    { 'type': 'us_forward_R12',
#                       'specifications': { 'color': 'k', 
#                                           'analysis_dir_within_folder': '5_analysis_combined'},
#                     },
#                     },
#            'combined_c10_only':{
#                'C10_forward': 
#                    {'type': 'us_forward_R12',
#                     'specifications': { 'color': 'b' },
#                     'color': 'r'},
#                     
#                'C10_reverse': 
#                    { 'type': 'us_reverse_R12',
#                     'specifications': { 'color': 'k' },
#                     },
#                
#                        }
                }
                    
    ## LOOPING
    for analysis_key in analysis_types_dict:
        ## DEFINING ANALYSIS DICT
        analysis_types = analysis_types_dict[analysis_key]
    
        ## GETTING DICTIONARY
        analysis_dir_dict = { each_key: {'main_analysis_dir': SIM_DICT[analysis_types[each_key]['type']]['main_sim_dir'],
                                         'dirname': SIM_DICT[analysis_types[each_key]['type']]['specific_sim'],
                                         **analysis_types[each_key]['specifications']
                                         
                                         }  for each_key in analysis_types}
        
        ## DEFINING COLOR
        fig, ax = plot_multiple_pmfs(parent_dir = PARENT_DIR,
                                     analysis_dir_dict = analysis_dir_dict,
                                     analysis_dir_within_folder=analysis_dir_within_folder,
                                     profile_xvg = profile_xvg,
                                     units = units,
                                     ylim= (-100, 900, 200), # None,    (-1000, 100, 100)
                                     temperature = temperature,
                                     end_truncate_dist = end_truncate_dist,
                                     data_range = np.arange(1.0,7.0,1),)
        
        ## FIGURE NAME
        figure_name = "combined_pmf_%s" %(analysis_key)
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = True,
                                 )
        ## STORING FIGURE FOR SMALLER INSET
        
        ## GETTING LIMS
        ax.set_xlim([3,7])
        ax.set_ylim([-100,100])
        
        ## SETTING LABELS
        ax.set_xticks(np.arange(3, 8, 1))
        ax.set_yticks(np.arange(-100, 200, 50))
        
        ## FIGURE NAME
        figure_name = "combined_pmf_%s_smaller" %(analysis_key)
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = True,
                                 )
    
    
    #%%
    

    
    
    
    #%% PLOTTING COMBINED WITH UNBIASED SIMULATIONS
    
    ## DEFINING TYPES
    analysis_types_dict={
             'combined': {
                'C1': 
                    {'type': 'us_forward_R01',
                      'specifications': { 'color': 'r',
                                         'analysis_dir_within_folder': '5_analysis_combined'},
                     'unbiased_sims': {
                             'unbiased_ROT001_1.900': {
                                     'linestyle': '--',
                                     'color': 'r',
                                     },
                             'unbiased_ROT001_2.100': {
                                     'linestyle': ':',
                                     'color': 'r',   
                                     }
                             }
                       
                       },
                'C10': 
                    { 'type': 'us_forward_R12',
                       'specifications': { 'color': 'k', 
                                           'analysis_dir_within_folder': '5_analysis_combined'},
                         'unbiased_sims': {
                                 'unbiased_ROT012_1.900': {
                                         'linestyle': '--',
                                         'color': 'k',
                                         },
                                 'unbiased_ROT012_2.100': {
                                         'linestyle': ':',
                                         'color': 'k',
                                         }
                                 }
                     },
                     }
            }
                                 
    ## GETTING NUM CONTACTS
    from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import load_num_contacts, \
        get_time_array_and_z_dist_from_com_distances, get_indices_for_truncate_time
                        

    ## DEFINING LAST TIME
    last_time_ps = 10000 # Last times to averge    
    
    
    ## GETTING FIGURE IZE
    fig_size_cm = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
    fig_size_cm = (fig_size_cm[0]*1.5, fig_size_cm[1])
    
    ## LOOPING
    for analysis_key in analysis_types_dict:
        ## GETTING DICTIONARY
        analysis_dir_dict = { each_key: {'main_analysis_dir': SIM_DICT[analysis_types[each_key]['type']]['main_sim_dir'],
                                         'dirname': SIM_DICT[analysis_types[each_key]['type']]['specific_sim'],
                                         **analysis_types[each_key]['specifications']
                                         
                                         }  for each_key in analysis_types}
        
        ## DEFINING COLOR
        fig, ax = plot_multiple_pmfs(parent_dir = PARENT_DIR,
                                     analysis_dir_dict = analysis_dir_dict,
                                     analysis_dir_within_folder=analysis_dir_within_folder,
                                     profile_xvg = profile_xvg,
                                     units = units,
                                     ylim= (-100, 900, 200), # None,    (-1000, 100, 100)
                                     temperature = temperature,
                                     end_truncate_dist = end_truncate_dist,
                                     data_range = np.arange(1.0,9.0,1),
                                     fig_size_cm = fig_size_cm,)
        
        ## GETTING KEYS FOR EACH UNBIASED SIMS
        unbiased_sim_key = [ list(analysis_types_dict[analysis_key][each_key]['unbiased_sims'].keys())
                             for each_key in analysis_types_dict[analysis_key].keys()]
        
        ## LOOPING THROUGH EACH
        for idx, unbiased_sim_list in enumerate(unbiased_sim_key):
            for curr_idx,current_unbiased_sim in enumerate(unbiased_sim_list):
                ## GETTTING SIMULATION DIR
                current_sim_folders = SIM_DICT[current_unbiased_sim]
                
                ## GETTING PATH
                path_sim = os.path.join(PARENT_DIR,
                                        current_sim_folders['main_sim_dir'],
                                        current_sim_folders['specific_sim'])
                
                ## GETTING TIME ARRAY AND Z DISTANCE
                time_array, z_dist = get_time_array_and_z_dist_from_com_distances(path_to_sim = path_sim)
                
                ## GETTING INDICES
                indices = get_indices_for_truncate_time(time_array,
                                                        last_time_ps = last_time_ps)
                
                ## REDEFINING TIME ARRAY (starting at zero)
                time_array = time_array[indices] - time_array[indices][0]
                z_dist = z_dist[indices]
                
                ## GETTING AVG Z DISTANCE
                z_mean_dist = np.mean(np.abs(z_dist))
                print(z_mean_dist)
                
                ## GETTING LIN INFO
                line_dict = analysis_types_dict[analysis_key][list(analysis_dir_dict.keys())[idx]]['unbiased_sims'][current_unbiased_sim]
                
                ## ADDING TO PLOT
                ax.axvline(x = z_mean_dist, label = current_unbiased_sim, **line_dict )
        
        
        
        ## UPDATING LEGEND
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
    
        ## FIGURE NAME
        figure_name = "combined_pmf_with_unbiased_%s" %(analysis_key)
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 )
    
    
    