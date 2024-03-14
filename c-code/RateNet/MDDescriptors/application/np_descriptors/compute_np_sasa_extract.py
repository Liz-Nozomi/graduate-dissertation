#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_np_sasa_extract.py

This script extracts the nanoparticle sasa information. 

Written by: Alex K. chew (03/31/2020)

"""
## IMPORTIGNG FUNCTIONS
import numpy as np
import math
import os

## MATPLOTLIB
import matplotlib.pyplot as plt

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_descriptors.global_vars import SIM_DICT, PARENT_SIM_PATH, ANALYSIS_FOLDER, PICKLE_NAME
from MDDescriptors.application.np_descriptors.compute_np_sasa import main_compute_np_sasa

## PLOT TOOLS
import MDDescriptors.core.plot_tools as plotter

## CONVERGENCE TOOLS
from MDDescriptors.core.calc_tools import find_theoretical_error_bounds, get_converged_value_from_end

## DEFINING FIG SIZE
FIGURE_SIZE = plotter.FIGURE_SIZES_DICT_CM['1_col_landscape']

## SETTING DEFAULTS
plotter.set_mpl_defaults()

## IMPORTING FUNCTIONS
from MDDescriptors.traj_tools.loop_traj_extract import load_results_pickle

### FUNCTION TO PLOT SASA VS. TIME
def plot_sasa_vs_time(time_array,
                      sasa_array,
                      fig_size = FIGURE_SIZE,
                      fig = None,
                      ax = None,
                      color = 'k',
                      **args):
    ''' This plots sasa vs. time '''
    if fig is None or ax is None:
        ## PLOTTING
        fig, ax = plotter.create_fig_based_on_cm(fig_size)
        
        ## ADDING AXIS
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("SASA (nm$^2$)")
    
    ## PLOTTING
    ax.plot(time_array,
            sasa_array,
            color=color,
            **args
            )    
    ## FITTING
#    fig.tight_layout()
    return fig, ax

### FUNCTION TO OUTPUT SASA AFTER TIME
def output_sasa_after_time(time_array,
                           sasa_array,
                           time_value = 0,
                           count_type='end',
                           return_time = False):
    '''
    This function outputs the sasa array after a given time.
    INPUTS:
        time_array: [np.array]
            time array in ps
        sasa_array: [np.array]
            sasa values in nm2 versus time
        time_value: [int]
            time value in ps that is desired
        count_type: [str]
            whether to count from specific part of time:
                'end': count from the end
                'begin': count from beginning
        return_time: [logical]
            True if you want time array
    OUTPUTS:
        sasa_array_new_time: [np.array]
            sasa array after a set time
    '''
    ## GETTING TIME FROM END
    if count_type == 'end':
        time_from_end = time_array[-1] - time_value
    elif count_type == 'begin':
        time_from_end = time_value
    
    ## GETTING SASA VALUE AFTER TIME
    index_converged = time_array > time_from_end
    sasa_array_new_time =  sasa_array[index_converged]
    if return_time is False:
        return sasa_array_new_time
    else:
        return sasa_array_new_time, time_array[index_converged]


### FUNCTION TO RUN RUNNING AVERAGE
def compute_running_average(x):
    '''
    This function computes the running average of an array x
    INPUTS:
        x: [np.array]
            some array value
    OUTPUTS:
        running_avg: [np.array]
            running average array values
    '''
#    ## DEFINING RUNNING AVG
#    running_avg = []
#    
#    total_value = 0
#    for idx, value in enumerate(x):
#        ## ADDING
#        total_value += value
#        ## DIVIDING BY INDEX
#        running_avg.append(total_value / (idx+1)) # + 1 b/c python counts from zero
#    
#    ## NUMPY
#    running_avg = np.array(running_avg)
    
    ## ALTERNATIVE NUMPY CUMMSUM
    running_avg = np.cumsum(x) / np.arange(1, len(x)+1)
    
    return running_avg

#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING PATH
    path_to_sim="/Volumes/akchew/scratch/nanoparticle_project/simulations/ROT_DMSO_SIMS/switch_solvents-50000-dmso-EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
    # "/Volumes/akchew/scratch/nanoparticle_project/simulations/ROT_WATER_SIMS/EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
    
    ## SIM DICT
    sim_dict = {
            'Water': {
                    'sim_path': "/Volumes/akchew/scratch/nanoparticle_project/simulations/ROT_WATER_SIMS/EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1",
                    'color': 'black',
                    },
            'DMSO': {
                    'sim_path': "/Volumes/akchew/scratch/nanoparticle_project/simulations/ROT_DMSO_SIMS/switch_solvents-50000-dmso-EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1",
                    'color': 'red',
                    }
            
            }
            
    
    ## GETTING CONVERGED VALUE
    converged_time_from_end = 20000 # ps    
    
    ## DEFINING SOME PERCENT ERROR
    percent_error=10 # percent
    convergence_type = "percent_error"
    
    ## DEFINING FIG SIZE
    fig_size = (5.200 , 4.0)
    
    ## CREATING FIG AND AX
    fig, ax = None, None
    
    ## DEFINING ANALYSIS
    for current_key in sim_dict:
        path_to_sim = sim_dict[current_key]['sim_path']
        color = sim_dict[current_key]['color']
        ## LOADING THE DATA
        full_data, extracted_data = load_results_pickle(path_to_sim,
                                                        func = main_compute_np_sasa,
                                                        )

        # SASA VS TIME
    
        ## GETTING NUMPY 
        sasa_over_time = np.array(extracted_data).astype('float')
        
        ## DEFINING INPUTS
        time_array = sasa_over_time[:,0] / 1000 # ns
        sasa_array = sasa_over_time[:,1]
        
        ## PLOTTING SASA OVER TIME
        fig, ax = plot_sasa_vs_time(time_array = time_array,
                                    sasa_array = sasa_array,
                                    fig_size = fig_size,
                                    color = color,
                                    fig = fig,
                                    ax = ax,
                                    label = current_key)
        
    ## SETTING X AXIS 
    ax.set_xticks(np.arange(0, 60, 10))
    
    ## SETTING Y AXIS
    ax.set_yticks(np.arange(180, 280, 20))
    
    ## ADDING LEGEND
    ax.legend()
    
    ## TIGHT LAYOUT
    # fig.tight_layout()
    
    STORE_FIG_LOC = "/Users/alex/Box/VanLehnGroup/14.Grants/2020_XSEDE_PROPOSAL"
    figure_name = "SASA_vs_time"
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg',
                             save_fig = True,
                             )
                             
                             
    #%% PLOTTING RUNNING AVERAGE
    
    ## CREATING FIG AND AX
    fig, ax = None, None
    
    ## DEFINING ANALYSIS
    for current_key in sim_dict:
        path_to_sim = sim_dict[current_key]['sim_path']
        color = sim_dict[current_key]['color']
        ## LOADING THE DATA
        full_data, extracted_data = load_results_pickle(path_to_sim,
                                                        func = main_compute_np_sasa,
                                                        )
    
        
        ## GETTING OPTIMAL TIME TO NEAREST TEN THOUSAND
        rounding_value=1E4 # 1000
        opt_x_rounded = 10000
        # math.ceil(opt_x / rounding_value) * rounding_value
        # opt_x
        # 
        
        ## GETTING NUMPY 
        sasa_over_time = np.array(extracted_data).astype('float')
        
        ## DEFINING INPUTS
        time_array = sasa_over_time[:,0]  # ns
        sasa_array = sasa_over_time[:,1]
        
        
        ## GETTING SASA VALUES
        sasa_array_after_converg, time_array_after_converg= output_sasa_after_time(time_array = time_array,
                                                                                   sasa_array = sasa_array,
                                                                                   time_value = opt_x_rounded,
                                                                                   count_type = 'begin',
                                                                                   return_time = True)
        
        
        ## GETTING RUNNING AVERAGE
        running_avg_sasa = compute_running_average(x = sasa_array_after_converg)
        
        ## PLOTTING SASA OVER TIME
        fig, ax = plot_sasa_vs_time(time_array = time_array_after_converg/1000, # ns
                                    sasa_array = running_avg_sasa,
                                    fig = fig,
                                    color = color,
                                    fig_size = fig_size,
                                    ax = ax,
                                    label = current_key)
    ax.set_xticks(np.arange(10, 60, 10))
    
    ## SETTING Y AXIS
    ax.set_yticks(np.arange(180, 280, 20))
    
    ## ADDING LEGEND
    ax.legend()
    
    ## TIGHT LAYOUT
    # fig.tight_layout()
    
    STORE_FIG_LOC = "/Users/alex/Box/VanLehnGroup/14.Grants/2020_XSEDE_PROPOSAL"
    figure_name = "Running_avg_SASA_vs_time"
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg',
                             save_fig = True,
                             )
                             
    

        #%% INCLUDING CONVERGENCE 

        ## GETTING SASA AFTER A TIME
        sasa_array_new_time = output_sasa_after_time(time_array = time_array,
                                                     sasa_array = sasa_array,
                                                     time_value = converged_time_from_end )
        sasa_converged =  np.mean(sasa_array_new_time)
        
        ## DRAWING THE X 
        ax.axhline(y = sasa_converged, linestyle = '--', color = 'b', label = "Avg. SASA (nm$^2$): %.1f"%(sasa_converged))
        
        ## GETTING CONVERGENCE
        theoretical_bounds, bound = find_theoretical_error_bounds(value = sasa_converged, 
                                                           percent_error = percent_error,
                                                           convergence_type = convergence_type) 
        
        ## GETTING INDEX OF CONVERGED VALUE
        index = get_converged_value_from_end(y_array = sasa_array, 
                                             desired_y = sasa_converged,
                                             bound = bound,
                                             )
        
        ## FILLING
        ax.fill_between(time_array, theoretical_bounds[0], theoretical_bounds[1], color='gray', label="Perc. Error: %d"%(percent_error) )
        
        ## GETTING OPTIMAL
        opt_x = time_array[index]
        
        ## PLOTTING X
        ax.axvline(x = opt_x, linestyle = '--', color = 'r', label = "Opt. Time (ps): %d"%(opt_x) )
        
        ## ADDING LEGEND
        ax.legend()
    
    #%%
    


    
    #%% 
    
