#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nplm_check_spring_constants.py

The purpose of this scripts is to check the spring constants for a job. 
The idea would be to run several umbrella sampling windows with different 
spring constants and measure the changes in coordination number (or any 
other collective variable)

Written by: Alex K. Chew (05/18/2020)

"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## IMPORTING TOOLS
from MDDescriptors.core.import_tools import read_plumed_covar

## SETTING DEFAULTS
import MDDescriptors.core.plot_tools as plot_funcs
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING IMAGE LOCATION
IMAGE_LOC="/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200622/images"

### FUNCTION TO EXTRACT SPRING CONSTANT
def extract_spr_constant_from_name(name):
    '''
    This function simply extracts sprinc constant from the name, e.g.
    
    US_5-NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1
    
    '''
    ## SPLIT BY HYPHEN
    name_split = name.split('-')
    name_split_under = name_split[0].split('_')
    ## STORING SPRING CONSTANT
    spr_constant = float(name_split_under[1])
    
    return spr_constant


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING PATH TO SIM
    parent_path = "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims"
    
    ## DEFINING SIM LIST
    sim_list = [
#            "20200520-plumed_US_debugging",
#            "20200520-plumed_US_debugging_RO1"
            "20200615-Debugging_C10_spring_hydrophobic_contacts"
            ]
    
    ## DEFINING REFERENCE SPRING CONSTANT
    ref_contacts=5
    # 110
    # 100
    
    ## DEFINING HISTOGRAM RANGE
    step_size = 1
    # 5
    histogram_range = (0, 10)
    # (50, 150)
    # (0, 200)

    ## DEFINING X VALUES
    xs = np.arange( histogram_range[0], histogram_range[1], step_size ) + 0.5 * step_size
    
    ## LOOPING THROUGH DIRECTORY
    for sim_idx, sims in enumerate(sim_list):
        ## DEFINING PATH
        sim_path = os.path.join(parent_path,
                                sims)
    
        ## FINDING ALL DIRECTORIES
        dir_list = glob.glob(sim_path + "/*")
        
        ## DEFINING PATH TO SIM
        relative_path_to_sim="4_simulations/%s"%(ref_contacts)
        
        ## DEFINING COVAR FILE
        covar_file="nplm_prod_COVAR.dat"
        
        ## DEFINING COORD LABEL
        coord_label = "coord"
        
        ## GETTING THE SPRING CONSTANT
        spr_constants = [ extract_spr_constant_from_name(os.path.basename(each_file)) for each_file in dir_list]
        
        ## GETTING PATH TO COVARS
        path_list_covars = [ os.path.join(each_dir,
                                          relative_path_to_sim,
                                          covar_file) for each_dir in dir_list]
        
        ## STORING
        covar_output = []
        
        ## LOOPING AND STORING
        for idx, each_covar_file in enumerate(path_list_covars):
            
            try:
            
                ## LOADING EACH COVAR FILE
                covar_file = read_plumed_covar(each_covar_file)
            
                ## STORING THE TIME AND DATAFRAME
                current_coord = covar_file[coord_label]
                
                ## GETTING HISTOGRAM
                hist = np.histogram(current_coord, 
                                    density = True, 
                                    range = histogram_range,
                                    bins = xs.size)[0]
                
                ## GETTING AVERAGE AND STD
                output_dict = {
                        'spring_constant': spr_constants[idx],
                        'avg': np.mean(current_coord),
                        'std': np.std(current_coord),
                        'hist': hist,
                        'xs': xs,
                        'min': np.min(current_coord),
                        'max': np.max(current_coord),
                        }
                
                covar_output.append(output_dict)
                # covar_file.plot(x="time", y= coord_label)
            except FileNotFoundError:
                pass
        
        ## CREATING DF
        df = pd.DataFrame(covar_output)
        ## SORTING
        df = df.sort_values(by = "spring_constant")
        
        #### PLOTTING MIN AND MAX
        figsize=plot_funcs.cm2inch( *FIGURE_SIZE )
        ax = df.plot(x='spring_constant', 
                     y=['min', 'avg', 'max'], 
                     figsize = figsize,
                     color = ['red', 'black', 'blue'],
                     linestyle = "-",
                     marker = ".",
                     )
        ## CHANGING AXIS
        ax.set_xlabel("Spring constant")
        ax.set_ylabel("Coordination number")
        
        ## ADDING LABEL
        ax.legend()
        
        ## DRAWING HORIZONTAL LINE
        ax.axhline(y = ref_contacts, color = 'k', linestyle = '--')
        
        
        ## GETTING FIG
        fig = plt.gcf()
        
        ## GETTING TIGHT LAYOUT
        fig.tight_layout()
        
        ## FIGURE NAME
        figure_name = "Avg_minmax_vs_spring_constant-%s-"%(sims)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'png', 
                     save_fig=True,)
                     
        ## SETTING X LIM
        ax.set_xlim([-0.5, 5])
        
        ## FIGURE NAME
        figure_name = "Avg_minmax_vs_spring_constant_SHRINK-%s-"%(sims)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'png', 
                     save_fig=True,)

        
        
        
        #%%
        #### PLOTTING STANDARD DEVIATION VERSUS SPRING CONSTANT ####
        ## GETTING FIGURE SIZE
        figsize=plot_funcs.cm2inch( *FIGURE_SIZE )
        ax = df.plot(x='spring_constant', 
                     y='avg', 
                     figsize = figsize,
                     color = 'k',
                     linestyle = "-",
                     marker = ".",
                     )
        
        ## CHANGING AXIS
        ax.set_xlabel("Spring constant")
        ax.set_ylabel("AVG coordination number")
        
        ## DRAWING HORIZONTAL LINE
        ax.axhline(y = ref_contacts, color = 'k', linestyle = '--')
        
        ## REMOVING LEGEND
        ax.get_legend().remove()
        
        ## GETTING FIG
        fig = plt.gcf()
        
        ## GETTING TIGHT LAYOUT
        fig.tight_layout()

        ## FIGURE NAME
        figure_name = "Avg_vs_spring_constant-%s-"%(sims)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'png', 
                     save_fig=True,)
        
        #### PLOTTING HISTOGRAM ####
        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
        
        ## ADDING LEGENDS
        ax.set_xlabel("Coordination number")
        ax.set_ylabel("Probability density function")
        
        ## RESETTING COLOR CYCLE
        plt.gca().set_prop_cycle(None)
        
        ## GETTING CMAP
        cmap = plot_funcs.get_cmap(n = len(df),
                                   name = 'hsv')
        
        #### PLOTTING HISTOGRMS 
        for idx, rows in df.iterrows():
            ## GETTING DETAILS
            hist = rows['hist']
            bins = rows['xs']
            
            ## DEFINING COLOR
            color = cmap(idx)
            
            ## PLOTTING HISTOGRAM
            ax.plot(bins, hist, label = rows['spring_constant'], color = color)
            # ax.hist(hist, bins = bins)
        ## DRAWING VERTICAL LINE
        ax.axvline( x = ref_contacts, color = 'k', linestyle = '--')
        
        ## ADDING LEGEND
        ax.legend()
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        

        ## FIGURE NAME
        figure_name = "Histogram_of_spring_constant-%s-"%(sims)
        
        ## SETTING AXIS
        plot_funcs.store_figure(fig = fig, 
                     path = os.path.join(IMAGE_LOC,
                                         figure_name), 
                     fig_extension = 'png', 
                     save_fig=True,)
        