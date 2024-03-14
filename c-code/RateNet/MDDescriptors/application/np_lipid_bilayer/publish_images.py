#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_images.py

Written by: Alex K. Chew (05/12/2020)

Note:
    xlrd required for analyzing excel:
        pip install xlrd



"""

## IMPORTING MODULES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import mdtraj as md
import sys

## IMPORTING CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING STORING FIG LOCATION
import MDDescriptors.application.np_lipid_bilayer.global_vars as global_vars

from MDDescriptors.application.np_lipid_bilayer.global_vars import find_us_sampling_key

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types

## DEFINING GLOBAL VARS
STORE_FIG_LOC = r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_lipid_membrane_binding/Figures/svg_output"
#  global_vars.IMAGE_LOC
PARENT_DIR = global_vars.PARENT_SIM_PATH
SIM_DICT = global_vars.NPLM_SIM_DICT
SAVE_FIG = True
FIG_EXTENSION='svg'

## IMPORTING FUNCTIONS FROM MD BUILDERS
from MDBuilder.umbrella_sampling.umbrella_sampling_analysis import read_xvg, plot_us_histogram, pmf_correction_entropic_effects, truncate_by_maximum, \
                                                                   plot_us_pmf, plot_sampling_time_pmf, plot_multiple_pmfs
                                                                                  
## IMPORTING AVG CONTACTS
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import create_group_avg_contacts_df, extract_num_contacts

## IMPORTING MODULES
from MDDescriptors.application.np_lipid_bilayer.nplm_plot_plumed_output import \
    load_plumed_contacts_and_com_distances, convert_covar_to_output_dict, plot_contacts_vs_distance_plumed, plot_specific_contacts_vs_distance

## PLUMED INPUT
from MDDescriptors.application.np_lipid_bilayer.compute_contacts_for_plumed import extract_plumed_input_and_covar

## IMPORTING TRUNCATION TECHNIQUES
from MDDescriptors.application.np_lipid_bilayer.nplm_plot_plumed_output import truncate_df_based_on_time


## LOADING FREE ENERGIES
from MDBuilder.umbrella_sampling_plumed.plot_grossfield_wham_fe import \
     plot_free_energies_for_wham_folder, main_sampling_time, plot_covar_histogram, \
     plot_PMF_histogram, plot_PMF_sampling_time

## IMPORTING TOOLS FOR EXTRACTING RDFS
from gmx_extraction_tools.plot_rdf import read_xvg, extract_gmx_rdf, combine_rdf_dict, generate_rdf_dict

################
### DEFAULTS ###
################                                                                   
## DEFINING PARENT SIMULATION PATH
parent_sim_path = PARENT_SIM_PATH

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

## DEFINING COLOR DICT
COLOR_DICT = {
        'C1':{
            'color': 'r',
            },
        'C10':{
            'color': 'k'
                },
        'Bn':{
            'color': 'b'
            },
        'Bn_oldFF':{
            'color': 'orange'
            },                
        }
        
## DEFINING DEFAULTS
PLUMED_INPUT_FILE="plumed_analysis_input.dat"

## DEFINING GLOBAL CONTACT LABEL
HYDROPHOBIC_CONTACT_LABEL = 'NP_ALK_RGRP-LM_TAILGRPS'

## DEFINING TIME STRIDE
TIME_STRIDE=100

## DEFINING ALL UNBIASED SIMULATIONS
UNBIASED_SIM_LIST = [
        'unbiased_ROT012_1.300',
        'unbiased_ROT012_1.900',
        'unbiased_ROT012_2.100',
        'unbiased_ROT012_3.500',
        
        'unbiased_ROT001_1.700',
        'unbiased_ROT001_1.900',
        'unbiased_ROT001_2.100',
        'unbiased_ROT001_2.300',
        'unbiased_ROT001_3.500',
        'unbiased_ROT012_5.300',
        
        ## INCLUDING REVERSE SIMS
        'unbiased_ROT001_4.700_rev',
        'unbiased_ROT012_5.300_rev',
        ]    


## DEFINING PATH TO EXP DATA
PATH_TO_EXP="/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_lipid_membrane_binding/Exp_data/exp_data_to_plot.xlsx"

## DEFINING OTHER VARIABLES
## DEFINING PARENT DICT
PARENT_FOLDER_DICT = {
        'all_np_contacts': '20200522-plumed_US_initial',
        'hydrophobic_contacts': "20200615-US_PLUMED_rerun_with_10_spring",
        'hydrophobic_contacts_extended': "20200615-US_PLUMED_rerun_with_10_spring_extended",
        'hydrophobic_contacts_spr1': "20200609-US_with_hydrophobic_contacts",
        'hydrophobic_contacts_iter2': "20200713-US_PLUMED_iter2",
        'Bn_new_FF_hydrophobic_contacts': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF",
        }
    
## DEFINING PATH DICT
PATH_DICT = {'all_np_contacts': { 
                'C1': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['all_np_contacts'],
                                   'US_1-NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1/wham'),
                'C10': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['all_np_contacts'],
                                   'US_1-NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/wham'), },
            'hydrophobic_contacts': { 
                'C1': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['hydrophobic_contacts_extended'],
                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1/wham'),
                'Bn': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['Bn_new_FF_hydrophobic_contacts'],
                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1/wham'), 
                'C10': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['hydrophobic_contacts_extended'],
                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/wham'), 
#                'Bn_oldFF': os.path.join(parent_sim_path,
#                                   PARENT_FOLDER_DICT['hydrophobic_contacts'],
#                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1/wham'), 
                                   },
                    
                    
            'hydrophobic_contacts_iter2': { 
                'C1': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['hydrophobic_contacts_iter2'],
                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-25000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1/wham'),
                'C10': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['hydrophobic_contacts_iter2'],
                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-25000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/wham'), 
#                'Bn': os.path.join(parent_sim_path,
#                                   PARENT_FOLDER_DICT['hydrophobic_contacts_iter2'],
#                                   'UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-25000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1/wham'), 
#                                    },
                },
                    
            'hydrophobic_contacts_spr1': { 
                'C1': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['hydrophobic_contacts_spr1'],
                                   'UShydrophobic_1-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1/wham'),
                'C10': os.path.join(parent_sim_path,
                                   PARENT_FOLDER_DICT['hydrophobic_contacts_spr1'],
                                   'UShydrophobic_1-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/wham'), 
                                    },
            'US-z-pulling': {
                    'C1': os.path.join(parent_sim_path,
                                       NPLM_SIM_DICT['us_forward_R01']['main_sim_dir'],
                                       NPLM_SIM_DICT['us_forward_R01']['specific_sim'],
                                       "wham",),
                                       
                    'C10': os.path.join(parent_sim_path,
                                       NPLM_SIM_DICT['us_forward_R12']['main_sim_dir'],
                                       NPLM_SIM_DICT['us_forward_R12']['specific_sim'],
                                       "wham",),
                            },
                                        
            'US-z-pushing': {
                    'C1': os.path.join(parent_sim_path,
                                       NPLM_SIM_DICT['us_reverse_R01']['main_sim_dir'],
                                       NPLM_SIM_DICT['us_reverse_R01']['specific_sim'],
                                       "wham",),
                                       
                    'C10': os.path.join(parent_sim_path,
                                       NPLM_SIM_DICT['us_reverse_R12']['main_sim_dir'],
                                       NPLM_SIM_DICT['us_reverse_R12']['specific_sim'],
                                       "wham",),
                    
                    
                    }
            }

## DEFINING FILES
FILES_DICT = {
        'z_com':{
            'covar_file': "nplm_prod_pullx_clean.xvg",
            'times_from_end': 40000.000,
            'delta': 0.5, # Variations from below and above
            'bin_width': 0.1,
            'xlabel': "z (nm)"
            },
        'hydrophobic_contacts':{
            'covar_file': "nplm_prod_COVAR.dat",
            'times_from_end': 10000.000,
            'delta': 5, # Variations from below and above
            'bin_width': 1,
            'xlabel': "Contacts",
            }
        }
        
## COPYING
FILES_DICT['hydrophobic_contacts_iter2'] = ['hydrophobic_contacts']

## DEFINING PREFIX - uSED FOR SAMPLING TIME
FREE_ENERGY_PREFIX_INFO = {
        'C1': {
            'forward': {
                'fe_file_prefix': "umbrella_free_energy-forward-%s"%(60000.000),
                'sort_values': 'end',
                },
            'reverse': {
                'fe_file_prefix': "umbrella_free_energy-rev-%s"%(60000.000),
                'sort_values': 'begin',
                },
                },
                    
        'C10': {
            'forward': {
                'fe_file_prefix': "umbrella_free_energy-forward-%s"%(60000.000),
                'sort_values': 'end',
                },
            'reverse': {
                'fe_file_prefix': "umbrella_free_energy-rev-%s"%(60000.000),
                'sort_values': 'begin',
                },
                },
                    
        'Bn': {
            'forward': {
                'fe_file_prefix': "umbrella_free_energy-forward-%s"%(40000.000),
                'sort_values': 'end',
                },
            'reverse': {
                'fe_file_prefix': "umbrella_free_energy-rev-%s"%(40000.000),
                'sort_values': 'begin',
                },
                },
                
        }

### FUNCTION TO LOAD UNBIASED WITH US
def load_unbiased_with_US_plumed_contacts(sim_type_list,
                                          last_time_to_extract_ps = 10000,
                                          desired_contacts = 'NP_ALK_RGRP-LM_TAILGRPS'):
    '''
    This function loads the unbiased simulation plumed results, the unbiased 
    simulation center of mass z-distances, and umbrella sampling simulation 
    z-distance
    INPUTS:
        sim_type_list: [list]
            list of simulation types (preferably unbiased sims)
        last_time_to_extract_ps: [int]
            last time from end to extract. So if this is 10000, it will look 
            for the last 10 ns of the data.
        desired_contacts: [str]
            desired contacts you want to extract from PLUMED. Ideally, this 
            should be something within your output COVAR file. 
    OUTPUTS:
        storage_list: [list]
            list of storage information in a form of output dict:                
                'sim_key': simulation key
                'avg_contacts': average number o contacts from the last time
                'avg_z': average z-distance from US sims
                'avg_unbiased_': average z-distance from unbiased sims
    '''
    ## CREATING EMPTY DICT
    storage_list = []
    ## DEFINING SIMULATION TYPE
    for idx, sim_type in enumerate(sim_type_list):
            
        ########################################
        ### GETTING Z-DISTANCE FROM UNBIASED ###
        ########################################
        
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
    
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING PATH TO SIMULATION
        path_to_sims = job_info.path_simulation_list[0]
    
        ## DEFINING INPUTS
        contact_inputs= {
                'job_info': job_info,
                'last_time_ps': last_time_to_extract_ps,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts_unbiased = extract_num_contacts(**contact_inputs)
        
        ## GETTING CONTACTS PER GROUP
        contacts_dict_unbiased = extract_contacts_unbiased.analyze_num_contacts(path_to_sim = path_to_sims,
                                                                                want_nplm_grouping = True,
                                                                                want_com_distance = True,
                                                                                skip_load_contacts = True,)               


        ###############################
        ### GETTING PLUMED CONTACTS ###
        ###############################
        
        ## LOADING PLUMED FILE
        unbiased_plumed_input, unbiased_df_extracted, unbiased_covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sims,
                                                                                                             plumed_input_file=PLUMED_INPUT_FILE,
                                                                                                             time_stride=TIME_STRIDE )    

        ## TRUNCATING THE DICT
        truncated_dict_unbiased = truncate_df_based_on_time(current_dict = unbiased_covar_output,
                                                            last_time_ps=last_time_to_extract_ps, # Last 10 ns
                                                            time_label='time')
        
         
        ##########################################
        ## GETTING UMBRELLA SAMPLING DISTANCES ###
        ##########################################
        us_dir_key, config_key, lig_name = find_us_sampling_key(sim_type, want_config_key = True, want_shorten_lig_name = True)
        
        ## pRINTING
        print("Config key", config_key)
        
        ## PRINTING
        print("Umbrella sampling extraction point (%s): %s, config: %s"%(sim_type, us_dir_key, config_key) )
        
        ## DEFINING MAIN SIM DIR
        us_main_sim_dir= NPLM_SIM_DICT[us_dir_key]['main_sim_dir']
        us_specific_sim= NPLM_SIM_DICT[us_dir_key]['specific_sim']
        
        ## GETTING JOB INFORMATION
        job_info_us = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = us_main_sim_dir,
                                  specific_sim = us_specific_sim)
        
        us_idx = job_info_us.config_library.index(config_key)
        path_to_sim = job_info_us.path_simulation_list[us_idx]
        
        ### GETTING Z DIMENSION
        
        ## DEFINING INPUTS
        contact_inputs_us= {
                'job_info': job_info_us,
                'last_time_ps': last_time_to_extract_ps,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs_us)
        
        ## GETTING CONTACTS PER GROUP
        contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                                              want_nplm_grouping = True, # True,
                                                              want_com_distance = True,
                                                              skip_load_contacts = True)
                
        ###############################
        ### GETTING PLUMED CONTACTS ###
        ###############################
        
        ## LOADING PLUMED FILE
        plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sim,
                                                                                  plumed_input_file=PLUMED_INPUT_FILE,
                                                                                  time_stride=TIME_STRIDE )    

        ## TRUNCATING THE DICT
        truncated_dict = truncate_df_based_on_time(current_dict = covar_output,
                                                   last_time_ps=last_time_to_extract_ps, # Last 10 ns
                                                   time_label='time')
        
        #################
        ### AVERAGING ###
        #################
        
        ## GETTING AVERAGE
        avg_contacts = truncated_dict[desired_contacts].mean()
        
        ## GETTING CONTACTS FTER UNBIASED
        unbiased_avg_contacts = truncated_dict_unbiased[desired_contacts].mean()
        
        ## GETTING AVERAGE Z DISTANCE
        avg_z = np.abs(contacts_dict['com_z_dist']).mean()
        
        ## STORING UNBIASED Z
        avg_unbiased_z = np.abs(contacts_dict_unbiased['com_z_dist']).mean()
        
        ## DEFINING OUTPUT DICT
        output_dict = {
                'sim_key': sim_type,
                'lig_name': lig_name,
                'avg_contacts': avg_contacts,
                'avg_z': avg_z,
                'avg_unbiased_z': avg_unbiased_z,
                'avg_unbiased_contacts': unbiased_avg_contacts,
                }
        
        
        ## STORING
        storage_list.append(output_dict)
        
    return storage_list

### FUNCTION TO PLOT HYDROPHOBIC CONTACTS FOR UNBIASED
def plot_hydrophobic_contacts_vs_z(storage_df,
                                   fig = None,
                                   ax = None,
                                   fig_size = FIGURE_SIZE,
                                   ):
    '''
    This function plots the hydrophobic contacts versus center of mass 
    z distance (nm).
    INPUTS:
        storage_df: [df]
            dataframe containing all information for avg contacts and so forth, e.g.
                                 sim_key lig_name  ...  avg_unbiased_z  desorbed
                0  unbiased_ROT012_1.300      C10  ...        4.060079     False
    OUTPUTS:
        fig, ax:
            figure and axis
    '''

    ## PLOTTING
    ## PLOTTING FIGURE
    if fig is None or ax is None:
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size)
        
    ## ADDING LABELS
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("Hydrophobic contacts")
    
    ## LOOPING THROUGH LIG NAME
    for lig_name in np.unique(storage_df['lig_name']):
        ## GETTING ALL DATA
        current_lig_df = storage_df.loc[storage_df['lig_name'] == lig_name]
        
        ## LOOPING THROUGH UNIQUE TYPES
        for types in np.unique(current_lig_df['desorbed']):
            ## GETTING NEW DATAFRAME
            current_df_for_type = current_lig_df.loc[current_lig_df['desorbed'] == types]
            
            ## DEFINING X AND Y
            x_values = current_df_for_type['avg_z']
            y_values = current_df_for_type['avg_contacts']
            
            ## DEFINING COLOR
            color = COLOR_DICT[lig_name]['color']
            
            ## DEFINING MARKER TYPE
            if types == True:
                markerfill = 'none'
                type_label="ad"
            else:
                markerfill = 'full'
                type_label="de"
            ## PLOTTING
            ax.plot(x_values, y_values, 
                    fillstyle = markerfill, 
                    markersize = 8,
                    color = color, 
                    marker = 'o', 
                    linestyle = "None",
                    label = '_'.join([lig_name, type_label]),
                    markeredgewidth=1.5)
    
    ## ADDING LEGEND
    ax.legend()
    
    ## FITTING
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO CREATE DATAFRAME AND GENERATE ADSORPTION / DESORPTION
def generate_dataframe_for_unbiased(storage_list,
                                    adsorption_cutoff = 6,
                                    hydrophobic_cutoff = None
                                    ):
    '''
    The purpose of this function is to create a dtaframe from 'load_unbiased_with_US_plumed_contacts'
    function. The idea is then to estimate which unbiased simulations are 
    actually desorbed from the lipid membrane. This will require constraints 
    on the adsorption cutoff and hydrophobic cutoff.
    INPUTS:
        storage_list: [list]
            list of storage details from 'load_unbiased_with_US_plumed_contacts'
        adsorption_cutoff: [float]
            adsorption cutoff in nanometers, default = 6 nm
        hydrophobic_cutoff: [float]
            number of hydrophobic contacts less than this number will 
            be considered desorbed. If None, then this will not be used
    OUTPUTS:
        storage_df: [df]
            dataframe with 'desorb' as True/False column
    '''

    ## CREATING DATAFRAME
    storage_df = pd.DataFrame(storage_list)
    
    ## SEEING WHEN THIS IS TRUE
    if hydrophobic_cutoff is not None:
        desorbed_array = (storage_df['avg_contacts'] < hydrophobic_cutoff) & (storage_df['avg_unbiased_z'] >= adsorption_cutoff)
    else:
        desorbed_array = storage_df['avg_unbiased_z'] >= adsorption_cutoff
    ## STORING
    storage_df['desorbed'] = desorbed_array[:]
    
    return storage_df

### FUNCTION TO PLOT SCATTER
def plot_df_exps(df,
                 ax,
                 label_dict={
                    'x': 'lipofilicity',
                    'y1': 'average max',
                    'y1_err': 'average max error',
                    'y2': 'average after rinse',
                    'y2_err': 'after rinse error',
                    'label': 'label'
                    },
                labels = [ "Avg. max",  "After rinse"],
                colors = ['k', 'r'],
                y_list = ['y1', 'y2'],
                errorbar_format={
                        'linestyle' : "None",
                        "fmt": "o",
                        "capthick": 1.5,
                        "markersize": 8,
                        "capsize": 2,
                        "elinewidth": 1.5,
                        },
                want_zero = True,
                    ):
    '''
    This functnion plots the experimental data dataframe
    INPUTS:
        df: [dataframe]
            pandas dataframe containing experimental data
        ax: [obj]
            axis object
        label_dict: [dict]
            dictionary of labels
        labels: [list]
            list of labels between the different sets
        colors: [list]
            list of colors
        y_list: [list]
            list of y values to plot
        errorbar_format: [dict]
            dictionary of error bars
        want_zero: [logical]
            True if you want zero dotted lines
    OUTPUTS:
        ax: [obj]
            axis object
    '''    
    ## ADDING ZERO
    if want_zero is True:
        ax.axhline(y=0, color = "k", linestyle = "--", linewidth = 2)

    ## LOOPING
    for idx, each_label in enumerate(labels):
        
        ## DEFINING Y LABEL
        y_label = y_list[idx]
        
        ## PLOTTING
        ax.errorbar(x = df[label_dict['x']],
                    y = df[label_dict[y_label]],
                    yerr = df[label_dict['%s_err'%(y_label)]],
                    color = colors[idx],
                    label = each_label,
                    **errorbar_format
                    )
        
        ## REMOVING ERROR BARS FROM LABELS
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        
        ## ADDING LABELS
        ax.legend(handles = handles, labels = labels)
        
        
    return ax

## DEFINING RDF XVG
RDF_XVG_DICT = {
#                    'Gold': 'np_rdf-AUNP_gold.xvg',
                'Counterions': "np_rdf-AUNP_counterions.xvg",
                'Ligands': "np_rdf-AUNP_lig.xvg",
                'Water': "np_rdf-AUNP_SOL.xvg",
#                    'Nitrogen': "np_rdf-AUNP_lig_nitrogens.xvg"
                }
            
## GETTING VOLUME DICT
VOLUME_XVG_DICT = {
        'Volume': 'np_rdf_full_volume.xvg',
        'NP_volume': 'np_rdf_np_volume.xvg',
        }

## DEFINING COLORS
RDF_COLOR_DICT = {
        'Gold': "gold",
        'Counterions': "purple",
        "Water": "gray",
        "Ligands": "black",
        "Nitrogen": "blue",
        }

#### FUNCTION TO PLOT
def plot_np_rdfs(rdf_xvg_list = RDF_XVG_DICT,
                 fig_size = (8.55, 15)):
    '''
    This function plots the RDFs
    INPUTS:
        rdf_xvg_list: [dict]
            dictionary list that has rdf file paths
        
    OUTPUTS:
        figs, axs: [obj]
            figure and axis objects
    '''

    ## CREATING SUBPLOTS
    figs, axs = plt.subplots(nrows = len(NP_WATER_SIM_DICT), ncols = 1, sharex = True, 
                             figsize = plot_funcs.cm2inch( *fig_size ))
    
    if type(axs) is not list:
        axs = [axs]
    
    ## LOOPING THROUGH EACH ONE
    for sim_idx, sim_key in enumerate(NP_WATER_SIM_DICT):
        ## DEFINING FIGURE
        fig = figs
        ax = axs[sim_idx]
        
        ## GETTING PATH
        path_to_sim = os.path.join(global_vars.NP_PARENT_SIM_PATH,
                                   NP_WATER_SIM_DICT[sim_key]['main_sim_dir'],
                                   NP_WATER_SIM_DICT[sim_key]['specific_sim'],
                                   )
        
        ## DEFINING STORAGE
        storage_rdf = []
        ## LOOPING THROUGH EACH XVG
        for rdf_xvg in rdf_xvg_list:
            ## DEFINING PATH TO XVG
            path_xvg = os.path.join(path_to_sim, rdf_xvg_list[rdf_xvg] )
            
            ## READING XVG FILE
            xvg_file = read_xvg(path_xvg)
            
            ## DEFINING DATA
            data = xvg_file.xvg_lines
            
            ## GETTING INFORMATION FROM RESIDUE-RESIDUE
            rdf = extract_gmx_rdf(data = data)
            
            ## STORING RDF INFO
            storage_rdf.append(rdf)
    
        ## GETTING VOLUME
        path_vol = [ os.path.join(path_to_sim, VOLUME_XVG_DICT[each_key] ) for each_key in VOLUME_XVG_DICT ]
        vol_data = { each_key: np.array(read_xvg(each_path).xvg_data).astype(float) for each_key, each_path in zip(VOLUME_XVG_DICT,path_vol) }
        
        ## MAKING VOLUME DATA CONSISTENT
        vol_data['Volume']= vol_data['Volume'][np.isin(vol_data['Volume'][:,0], vol_data['NP_volume'][:,0])]
        
        ## GETTING AVERAGE VOLUME
        avg_volume = { each_key: np.mean(vol_data[each_key][:, 1]) for each_key in vol_data}
        
        ## ADJUSTING THE VOLUME
        box_volume = avg_volume['Volume']
        box_volume_without_NP = box_volume - avg_volume['NP_volume']
        
        ## GETTING ADJUSTMENT RATIO
        adjustment_ratio = box_volume / box_volume_without_NP 
        
        ## COMBINING RDFS
        combined_name_list, Extract_data = combine_rdf_dict( storage_rdf )
    
        ## GENETING DICTIONARY FOR RDF
        rdf_dict = generate_rdf_dict( combined_name_list = rdf_xvg_list.keys(),
                                      Extract_data = Extract_data)
        
        ## ADDING HORIZONTAL LINE
        ax.axhline(y=1, linestyle='--', color='black', linewidth=1)

        ## ADDING TITLE            
        ax.text(.5,.9,sim_key,
            horizontalalignment='center',
            transform=ax.transAxes)
        
        ## LOOPING THROUGH EACH RDF
        for idx, each_key in enumerate(rdf_dict):
            ## DEFINING R AND GR
            r = rdf_dict[each_key]['r']
            g_r = rdf_dict[each_key]['g_r'] / adjustment_ratio # Adjustment ratio accounts for nanoparticle size
            ax.plot(r, g_r, '-', color = RDF_COLOR_DICT[each_key], label = each_key)
                
        ## ADDING TICKS
        axs[-1].tick_params(axis='both', which='both', labelsize=8, labelbottom = True)
        
        ## SETTING Y LABEL
        ax.set_ylabel("g(r)")
        
    ## SETTING LEGEND
    ax.legend()
    
    ## SETTING LABELX
    ax.set_xlabel("r (nm)")
    
    ## TIGHT LAYOUT
    fig.tight_layout()    
    
    ## ADJUSTING SPACE
    plt.subplots_adjust(wspace=0, hspace=0)
    

    return figs, axs

### FUNCTION TO LOAD SAMPLING TIME INFORMATION
def load_sampling_time_data(desired_key,
                           free_energy_prefixes):
    '''
    This function loads all sampling time data. 
    INPUTS:
        desired_key: [str]
            value that you want to begin loading the data from. This is 
            a key from PATH_DICT variable. We will use this variable 
            to define the wham_folder_dict.
        free_energy_prefixes: [key]
            keys for the prefixes in forward and reverse cases
    OUTPUTS:
        
    '''
    ## DEFINING WHAM FOLDERS
    wham_folder_dict = {
            **PATH_DICT[desired_key]
            }
    
    ## DEFINING STORAGE DICT
    storage_sampling_time = {}
    
    ## LOOPING THROUGH EACH KEY
    for each_key in wham_folder_dict:
        ## DEFINING FOLDER
        wham_folder = wham_folder_dict[each_key]
        
        ## CREATING DICT
        storage_sampling_time[each_key] = {}
        
        ## CHECKING PREFIXES
        if 'forward' not in free_energy_prefixes:
            
            ## CHECKING IF KEY IS WITHIN
            if each_key not in free_energy_prefixes:
                print("Error! %s not defined in free_energy_prefixes"%(each_key))
                sys.exit()
            ## TRYING TO RECREATE PREFIX
            current_free_energy_prefix = free_energy_prefixes[each_key]
        else:
            current_free_energy_prefix = free_energy_prefixes
        
        ## LOOPING THROUGH FORWARD AND REVERSE KEYS
        for fe_key in current_free_energy_prefix:
            ## DEFINING DETAILS
            current_dict = current_free_energy_prefix[fe_key]
            ## EXTRACTING WHAM INFO
            wham_storage, sampling_time = main_sampling_time(wham_folder = wham_folder,
                                                             **current_dict)
    
            ## STORING
            storage_sampling_time[each_key][fe_key] = {
                    'wham_storage': wham_storage,
                    'sampling_time': sampling_time
                    }
    
    
    return storage_sampling_time

### FUNCTION TO GET FILES
def get_files_info(desired_key):
    '''
    This function gets dictionary of desired keys
    INPUTS:
        desired_key: [str]
            desired key
    OUTPUTS:
        files: [dict]
            dictionary of file information
    '''
    ## GETTING FILES
    if desired_key.startswith("US"):
        files = FILES_DICT['z_com']
    else:
        files = FILES_DICT['hydrophobic_contacts']
        
    return files

### FUNCTION TO LOAD HISTOGRAM DATA
def load_histogram_data(desired_key):
    '''
    This function loads the histogram information.
    INPUTS:
        desired_key: [str]
            desired key to load from PATH_DICT
    OUTPUTS:
        storage_histogram_data: [dict]
            dictionary containing histogram information used for PMFS
    '''
    ## DEFINING STORAGE
    storage_histogram_data = {}
    
    ## DEFINING WHAM FOLDERS
    wham_folder_dict = {
            **PATH_DICT[desired_key]
            }
    
    ## GETTING FILE INFORMATION
    files = get_files_info(desired_key = desired_key)
        
    ## GETTING COVAR
    covar_file = files['covar_file']
    
    ## GETTING DELTA
    delta = files['delta']
    
    ## LOOPING THROUGH EACH KEY
    for wham_idx, each_key in enumerate(wham_folder_dict):
        ## IF NONE, IT'LL LOOK FOR ALL OF THEM
        contacts = None
        
        ## DEFINING FOLDER
        wham_folder = wham_folder_dict[each_key]
    
        ## DEFINING SIMULATION PATH
        sim_folder = os.path.dirname(wham_folder)
        
        ## DEFINING RELATIVE PATH TO SIM
        relative_sim_path = "4_simulations"
        
        ## DEFINING PATH TO SIM
        path_to_sim = os.path.join(sim_folder,
                                   relative_sim_path)
        
        if contacts is None:
            folders = [ str(each) for each in glob.glob(path_to_sim + "/*") ]
            ## SORTING
            folders.sort()
            contacts = [ float(os.path.basename(each)) for each in folders ]
            histogram_range = (np.min(contacts)-delta, np.max(contacts)+delta)
        else:
            histogram_range = (-5, 105)
        
        ## GENERATING HISTOGRAM
        hist = plot_covar_histogram()
        
        ## LOADING DATA
        covar_df = hist.load_covar(path_to_sim = path_to_sim,
                                   contacts = folders,
                                   covar_file =covar_file,
                                   )
        
        ## STORING FOR EACH KEY
        storage_histogram_data[each_key] = {
                'hist': hist,
                'covar_df': covar_df,
                'histogram_range': histogram_range,
                }

    return storage_histogram_data

### FUNCTION TO PLOT HISTOGRAM AND CONVERGENCE INFORMATION
def plot_us_histo_and_convergence(storage_histogram_data = None,
                                  storage_sampling_time = None,
                                  xlabel = "z (nm)",
                                  axis_dict = {},
                                  want_zero_line = False,
                                  zero_location = None,
                                  plot_type = None,
                                  free_energy_prefixes = None,
                                  ):
    '''
    This function plots the umbrella sampling histogram and convergence 
    information.
    INPUTS:
        storage_histogram_data: [dict]
            dictionary containing histogram information
        storage_sampling_time: [dict]
            sampling time information
        xlabel: [str]
            xlabel string
        axis_dict: [dict]
            dictionary containig axis label information
        want_zero_line: [logical]
            True if you want zero line for the PMFs
        zero_location: [int]
            location to set zero
        plot_type: [str]
            type that you want. Examples:
                'histo_vertical':
                    plots histograms only in a vertical fashion
                'sampling_time_vertical':
                    plots sampling time as vertical axis
        free_energy_prefixes: [dict]
            prefixes of free energy - used to get the last time
    OUTPUTS:
        fig, axs:
            figure and axis information
    '''
    ## SEEING WHICH TYPE
    if plot_type == 'histo_vertical':
        nrows = len(storage_histogram_data)
        ncols = 1
    elif plot_type == 'sampling_time_vertical':
        nrows = len(storage_sampling_time)
        ncols = len(storage_sampling_time[next(iter(storage_sampling_time))])
    else:
        nrows = 3
        ncols = len(storage_sampling_time.keys())
    
    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = nrows, 
                            ncols = ncols, 
                            sharex = True, 
                            figsize = plot_funcs.cm2inch( *fig_size ))

    ## DEFINING AXIS ROW INDEX
    axes_row_idx = 0 
    
    ## TURNING OFF FOR SAMPLING TIME PLOT
    if plot_type != 'sampling_time_vertical':
    
        ## LOOPING THROUGH EACH HISTOGRAM DATA
        for idx_key, each_key in enumerate(storage_histogram_data):
            
            if plot_type == 'histo_vertical':
                ax = axs[idx_key]
            else:
                ## DEFINING AXIS
                ax = axs[axes_row_idx][idx_key]
            
            ## GETTING HISTOGRAM DATA
            histogram_data = storage_histogram_data[each_key]
            
            ## GETTING FILE INFORMATION
            files = get_files_info(desired_key = desired_key)
        
            ## GETTING DEFAULT FILE INFORMATION
            bin_width = files['bin_width']

            if free_energy_prefixes is not None:
                print("Using free energy prefixes as a way of determining free energies")
                if 'forward' in free_energy_prefixes:                    
                    times_from_end = float(free_energy_prefixes['forward']['fe_file_prefix'].split('-')[-1])
                else:
                    times_from_end = float(free_energy_prefixes[each_key]['forward']['fe_file_prefix'].split('-')[-1]) 
            else:
                times_from_end = files['times_from_end']                
        
            ## GENERATING PLOT
            fig, ax = plot_PMF_histogram(covar_df = histogram_data['covar_df'],
                                histogram_range = histogram_data['histogram_range'],
                                bin_width = bin_width, # 2
                                fig_size = (9.55, 6.5),
                                want_legend = False, 
                                time_starting_from = 0,
                                times_from_end = times_from_end,
                                fig = fig,
                                ax = ax,
                                want_black_color = True,)
            
            ## SETTING Y LABEL
            if plot_type == 'histo_vertical':
                if idx_key + 1 == np.round( (len(axs) ) /2):
                    ax.set_ylabel("Probability density function")
            else:
                ax.set_ylabel("Probability density function")
            
            ## ADDING TITLE            
            ax.text(.5,.9,"%s-histogram"%(each_key),
                horizontalalignment='center',
                transform=ax.transAxes)
            
            ## ADJUSTING LIMITS
            if 'histogram' in axis_dict:
                ## DEFINIG CURRENT_DICT
                current_axis_dict = axis_dict['histogram']
                if 'ylim' in current_axis_dict:
                    ax.set_ylim(current_axis_dict['ylim'])
                if 'yticks' in current_axis_dict:
                    ax.set_yticks(current_axis_dict['yticks'])
        
    ## CHECKING
    if plot_type != 'histo_vertical':
    
        ## GETTING TOTAL NUMBERS
        max_colors = np.max([ len(storage_sampling_time[each_key][forward_reverse]['wham_storage']) for each_key in storage_sampling_time for forward_reverse in storage_sampling_time[each_key] ])
        ## GETTING COLORS
        colors = plot_funcs.get_cmap(max_colors)
        
        ## GETTING COLOR LIST
        color_list = [ colors(each_idx) for each_idx in range(max_colors)]
        
        ## GETITNG LAST ONE TO BLACK
        color_list[-1] = 'k'
        
        
        #### PLOTTING SAMPLING TIME INFORMATION
        for idx_key, current_key in enumerate(storage_sampling_time):
            
            ## GETTING CURRENT DICT
            current_dict = storage_sampling_time[current_key]
            
            ## LOOPING FOR EACH SAMPLING
            for fr_idx, forward_and_reverse in enumerate(current_dict):
                if plot_type != 'sampling_time_vertical':
                    ## GETTING CURRENT AXIS ROW
                    current_row_idx = axes_row_idx + fr_idx + 1
                    ## GETTING AXIS
                    ax = axs[current_row_idx][idx_key]
                else:
                    ax = axs[idx_key][fr_idx]
                                
                ## GETTING THE DATA
                wham_storage = current_dict[forward_and_reverse]['wham_storage']
    
                ## ADDING 0 LINE                
                if want_zero_line is True:
                    ax.axhline(y=0, linestyle='--', color = 'gray', linewidth = 1.5)
                
                ## PLOTTING SAMPLING TIME
                fig, ax = plot_PMF_sampling_time(wham_storage = wham_storage,
                                                 fig = fig,
                                                 ax = ax,
                                                 want_legend = False,
                                                 want_total_time = True,
                                                 zero_location = zero_location,
                                                 colors = color_list
                                                 )
                
                ## ADDING LEGEND
                if idx_key >= 1 and fr_idx == 0:
                    ax.legend()
                
                ## SETTING Y LABEL
                ax.set_ylabel("PMF (kJ/mol)")
                
                ## ADDING TITLE            
                ax.text(.5,.9,'-'.join([current_key,forward_and_reverse]),
                    horizontalalignment='center',
                    transform=ax.transAxes)
                
                ## ADJUSTING LIMITS
                if 'pmf' in axis_dict:
                    ## DEFINIG CURRENT_DICT
                    current_axis_dict = axis_dict['pmf']
                    if 'ylim' in current_axis_dict:
                        ax.set_ylim(current_axis_dict['ylim'])
                    if 'yticks' in current_axis_dict:
                        ax.set_yticks(current_axis_dict['yticks'])
                ## ADDING LABELS
                if plot_type == 'sampling_time_vertical':
                    if idx_key == len(storage_sampling_time) - 1:
                        ## SETTING AXIS X LABEL
                        ax.set_xlabel(xlabel)

    ## SETTING AXIS X LABEL
    if plot_type != 'sampling_time_vertical':
        ax.set_xlabel(xlabel)
            
    ## ADJUSTING X TICKS
    if 'xlim' in axis_dict:
        ax.set_xlim(axis_dict['xlim'])
    if 'xticks' in axis_dict:
        ax.set_xticks(axis_dict['xticks'])

    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## ADJUSTING SPACE
    plt.subplots_adjust(hspace=0)

    return fig, axs
    


#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    
    #%% FIGURE 2 - EXPERIMENTAL QCM-D DATA
    
    fig_prefix="2_exp_qcmd_results"
    
    ## DEFINING FIGURE SIZE
    fig_size = (8.2439, 12.3683)
    
    ## PLOTTING MASS UPTAKE AND DISSIPATION
    
    
    ## DEFINING LABEL DICT
    label_dict={
            'x': 'lipofilicity',
            'y1': 'average max',
            'y1_err': 'average max error',
            'y2': 'average after rinse',
            'y2_err': 'after rinse error',
            'label': 'label'
            }
    
    ## LOADING THE DATA
    mass_uptake_df = pd.read_excel(PATH_TO_EXP, 'mass')
    diss_df =  pd.read_excel(PATH_TO_EXP, 'dissipation')
    
    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, 
                             figsize = plot_funcs.cm2inch( *fig_size ))
    
    ## DEFINNG FORMAT
    errorbar_format={
            'linestyle' : "None",
            "fmt": "o",
            "capthick": 1.5,
            "markersize": 8,
            "capsize": 3,
            "elinewidth": 1.5,
            }
    
    ## PLOTTING MASS UPTAKE
    ax =  plot_df_exps(df = mass_uptake_df,
                       ax = axs[0],
                       label_dict= label_dict,
                       labels = [ "Avg. max",  "After rinse"],
                       colors = ['k', 'r'],
                       y_list = ['y1', 'y2'],
                       errorbar_format = errorbar_format)
    
    ## SETTING LABELS
    ax.set_ylabel("Mass uptake")
    
    ## SETTING LIMITS
    ax.set_xlim([0,6])
    ax.set_ylim([-20,60])
    ax.set_yticks(np.arange(-20,80,20))

    ## PLOTITNG DISSIPATION
    ax =  plot_df_exps(df = diss_df,
                       ax = axs[1],
                       label_dict= label_dict,
                       labels = [ "Avg. max",  "After rinse"],
                       colors = ['k', 'r'],
                       y_list = ['y1', 'y2'],
                       errorbar_format = errorbar_format)
    ## SETTING Y LABELS
    ax.set_xlabel("Lipophilicity")
    ax.set_ylabel("Dissipiation")
    
    ## SETTING LIMITS
    ax.set_ylim([-0.20,0.80])
    ax.set_yticks(np.arange(-0.2,1,0.2))
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## ADJUSTING HORIZONTAL SPACING
    plt.subplots_adjust(hspace=0.3)
    
    ## ADDING TICKS
    axs[0].tick_params(axis='both', which='both', labelsize=8, labelbottom = True)
    axs[0].set_xlabel('Lipophilicity')
    
    ## FIGURE NAME
    figure_name = fig_prefix
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
    
    
    #%% FIGURE 6 - EXPERIMENTAL RATES
    
    fig_prefix="6_exp_rates"
    
    ## DEFINING FIGURE SIZE
    fig_size = (8.2439, 14)
    
    ## DEFINNG FORMAT
    errorbar_format={
            'linestyle' : "None",
            "fmt": "o",
            "capthick": 1.5,
            "markersize": 8,
            "capsize": 3,
            "elinewidth": 1.5,
            }
    
    ## DEFINING IF YOU WANT HORIZONTAL
    want_horizontal = True
    if want_horizontal is True:
        fig_size = (18.4, 5)
        # (fig_size[1], fig_size[0])
    
    ## DEFINING LIST OF K 
    k_list = ["ka", "kb", "kd"]
    
    ## DEFINING AXIS LIMITS
    k_axis_limits={
            'ka': {
                    'ylim': [-0.5e-4, 1.5e-4],
                    'yticks': np.arange(-0.5e-4, 2.0e-4, 0.5e-4),
                    },
                    
            'kb': {
                    'ylim': [0, 0.25],
                    'yticks': np.arange(-0.05, 0.30, 0.10),
                    },    
            'kd': {
                    'ylim': [0, 0.015],
                    'yticks': np.arange(-0.005, 0.020, 0.005),
                    },    
                    
            
            }
    
    ## LOADING
    k_df = [ pd.read_excel(PATH_TO_EXP, each_k) for each_k in k_list]
    
    ## DEFINING LABEL DICT
    label_dict={
            'x': 'lipofilicity',
            'y': 'value',
            'y_err': 'error',
            }
    
    if want_horizontal is True:
        nrows = 1
        ncols = len(k_list)
        sharex = False
    else:
        nrows = len(k_list)
        ncols = 1
        sharex = True
    
    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex = sharex, 
                             figsize = plot_funcs.cm2inch( *fig_size ))
    
    
    ## PLOTITNG DISSIPATION
    for idx, each_k in enumerate(k_list):
        ## PLOTTING
        ax =  plot_df_exps(df = k_df[idx],
                           ax = axs[idx],
                           label_dict= label_dict,
                           labels = ["Value"],
                           colors = ['k'],
                           y_list = ['y',],
                           errorbar_format = errorbar_format,
                           want_zero = True)
        ## REMOVE LEGEND
        ax.get_legend().remove()
        
        ## SETTING XLIM
        ax.set_xlim([0,6])
        ## SETTING XLABEL
        ax.set_xlabel("Lipophilicity")
        
        ## SETTING Y LIM
        ax.set_ylabel(each_k)
        
        ## SETTING AXIS
        if each_k == "ka":
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
        ## EDITING LIMITS
        if each_k in k_axis_limits:
            ax.set_ylim(k_axis_limits[each_k]['ylim'])
            ax.set_yticks(k_axis_limits[each_k]['yticks'])
        
        
    ## SETTING LIMITS
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## ADDING LABELS
    [ax.tick_params(axis='both', which='both', labelsize=8, labelbottom = True) for ax in axs]
    
    ## ADJUSTING HORIZONTAL SPACING
    if want_horizontal is False:
#        plt.subplots_adjust(wspace=0.4)
        plt.subplots_adjust(hspace=0.4)
    
    ## FIGURE NAME
    figure_name = fig_prefix
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
    
    
    #%% FIGURE 3 A,B
    ##############################
    ### UMBRELLA SAMPLING SIMS ###
    ##############################    

    ## DEFINING FIGURE SIZE
    fig_size = FIGURE_SIZE

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
                }
    
    ## DEFINING FIGURE PREFIX
    fig_prefix="COMP_2"
    
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
                                     ylim= (-100, 1000, 200), # None,    (-1000, 100, 100)
                                     temperature = temperature,
                                     end_truncate_dist = end_truncate_dist,
                                     fig_size_cm = fig_size,
                                     data_range = np.arange(1.0,8,1),)
        
        ## SETTING XLIM
        ax.set_xlim([1,7])
        
        ## SETTING Y LIM
        ax.set_ylim([-100, 1000])
        
        ## FIGURE NAME
        figure_name = fig_prefix + "_combined_pmf_%s" %(analysis_key)
        
        #%%
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 fig_extension = FIG_EXTENSION,
                                 save_fig = SAVE_FIG,
                                 )
                                 
        ''' 
        ## STORING FIGURE FOR SMALLER INSET
        
        ## GETTING LIMS
        ax.set_xlim([3,7])
        ax.set_ylim([-100,50])
        
        ## SETTING LABELS
        ax.set_xticks(np.arange(3, 8, 1))
        ax.set_yticks(np.arange(-100, 100, 50))
        
        ## FIGURE NAME
        figure_name = fig_prefix + "_combined_pmf_%s_smaller" %(analysis_key)
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 fig_extension = FIG_EXTENSION,
                                 )
        '''
    
    #%% FIGURE 2C,D: Number of contacts
        
    ## DEFINING SIM LIST
    sim_list_for_dict = [
            'us_forward_R12',
            'us_reverse_R12',
#            'us_forward_R01',
#            'us_reverse_R01',
            ]
    
    ## GETTING COVAR DETAILS
    sims_covar_storage = load_plumed_contacts_and_com_distances(sim_list_for_dict = sim_list_for_dict,
                                                                plumed_input_file = "plumed_analysis_input.dat",
                                                                time_stride = 100)
    
    #%%
    ## FINDING OUTPUTS FOR SIMS
    sims_output_dict = convert_covar_to_output_dict(sims_covar_storage,
                                     last_time_ps = 40000,
                                     lm_groups =[
                                                # 'HEADGRPS',
                                                'TAILGRPS',
                                                ],
                                        np_grps=[
#                                                'GOLD',
                                                'ALK',
                                                'PEG',
                                                'NGRP_RGRP'
#                                                'RGRP',
#                                                'NGRP',
                                                ],
                                        num_split = 2)

    #%% PLOTTING FOR EACH GROUP
    
    ## DEFINING FIGURE DETAILS
    fig_prefix = "2CD_"
    figsize = FIGURE_SIZE
    
    ## DEFINING LIMITS FOR DISTANCE PLUMED
    ylim = (0, 600)
    ylabels=np.arange(0, 700, 100)
    xlim = (1,7)
    xlabels = np.arange(1,8,1)
    
    ## PLOTTING
    figs, axs, labels = plot_contacts_vs_distance_plumed(sims_output_dict,
                                                         xlabel = "z (nm)",
                                                         ylabel = "# DOPC tail contacts",
                                                         figsize = plot_funcs.cm2inch( *figsize ),
                                                         fig_prefix = fig_prefix)
    
    ## UPDATING FOR EACH FIG
    for idx, each_fig in enumerate(figs):    
        ax = axs[idx]
        
        ## LABELS
        ax.set_xlim(xlim)
        ax.set_xticks(xlabels)

        ax.set_ylim(ylim)
        ax.set_yticks(ylabels)
        
        ## TIGHT LAYOUT
        each_fig.tight_layout()
        
        #%%
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = each_fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     labels[idx]),
                                 fig_extension = FIG_EXTENSION,
                                 save_fig = SAVE_FIG,
                                 )
    
    #%% COMBINING FIGURE 2 INTO TWO SUBPLOTS
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
                }
    
    ## GETTING SIM LIST
    sim_list_for_dict = [
            'us_forward_R12',
            'us_reverse_R12',
            'us_forward_R01',
            'us_reverse_R01',
            ]
    
    ## GETTING COVAR DETAILS
    sims_covar_storage = load_plumed_contacts_and_com_distances(sim_list_for_dict = sim_list_for_dict,
                                                                plumed_input_file = "plumed_analysis_input.dat",
                                                                time_stride = 100)
    
    ## FINDING OUTPUTS FOR SIMS
    sims_output_dict = convert_covar_to_output_dict(sims_covar_storage,
                                     last_time_ps = 40000,
                                     lm_groups =[
                                                # 'HEADGRPS',
                                                'TAILGRPS',
                                                ],
                                        np_grps=[
#                                                'ALK',
#                                                'PEG',
#                                                'NGRP_RGRP'
                                                'ALK_RGRP'
                                                ])
    
    #%% FIGURE 3 COMBINED (FINAL)
    
    ## DEFINING FIGURE PREFIX
    fig_prefix="3-combined"
    
    ## DEFINING FIGURE SIZE
    fig_size = (8.3, 16)
    # 18.55245
    # 12.3683
    # (FIGURE_SIZE[0], FIGURE_SIZE[1]*2)
    # (20, 5)
    # 
    ## CLOSING ALL
    plt.close('all')
    
    ## CREATING SUBPLOTS
    figs, axs = plt.subplots(nrows = 3, ncols = 1, sharex = True, 
                             figsize = plot_funcs.cm2inch( *fig_size ))
    
    ## LOOPING
    for idx, analysis_key in enumerate(analysis_types_dict):
        
        ## DEFINING AXIS
        ax = axs[idx]
        
        ## DEFINING DESIRED DICT
        if analysis_key == 'forward':
            desired_key = 'US-z-pulling'
            zero_location = None
        elif analysis_key == 'reverse':
            desired_key = 'US-z-pushing'
            zero_location = -1
        
        ## DEFINING WHAM FOLDERS
        wham_folder_dict = {
                **PATH_DICT[desired_key]
                }
        
        ## DEFINING FILE
        fe_file = "umbrella_free_energy.dat"
                
        ## ADDING 0 LINE
        ax.axhline(y=0, linestyle='--', color = 'gray', linewidth = 1.5)
        
        ## PLOTTING FREE ENERGIES
        fig, ax, storage_list = plot_free_energies_for_wham_folder(wham_folder_dict,
                                                     want_peaks = False,
                                                     print_min = True,
                                                     wham_styles_dict = COLOR_DICT,
                                                   fe_file_prefix_suffix = {
                                                           'prefix': 'umbrella_free_energy_',
                                                           'suffix': '.dat',
                                                           },
                                                     fig_size = None,
                                                     fig = figs,
                                                     ax = ax,
                                                     zero_location = zero_location,
                                                    errorbar_format={
                                                            "fmt": "None",
                                                            "capthick": 1.5,
                                                            "markersize": 8,
                                                            "capsize": 3,
                                                            "elinewidth": 1.5,
                                                            },
                                                     )
        
        ## ADDING LABELS
        ax.set_ylabel('PMF (kJ/mol)')
        
        ## SETTING X LIMS
        xlim = (1,7)
        xlabels = np.arange(1,8,1)
        if analysis_key == 'forward':
            ylim = (-100, 1100)
            ylabels = np.arange(-100,1200,200)
        else:
            ylim = (-100, 1100)
            ylabels = np.arange(-100,1000,200)
            
        ## LABELS
        ax.set_xlim(xlim)
        ax.set_xticks(xlabels)

        ax.set_ylim(ylim)
        ax.set_yticks(ylabels)
        ax.legend()
        
    ## PLOTTING CONTACTS FOR LAST AXIS
    
    ## DEFINING AX
    ax = axs[-1]
    
    ## ADDING 0 LINE
    ax.axhline(y=0, linestyle='--', color = 'gray', linewidth = 1.5)
    
    ## COLOR DICT
    color_dict = {
            'C1': 'red',
            'C10': 'black',
            }
    
    ## LOOPING
    for idx, analysis_key in enumerate(analysis_types_dict):
        ## LOOPING THROUGH EACH KEY
        for each_key in ['C1', 'C10']:
            ## NOW, ADDING CONTACTS VERSUS DISTANCES
            sim_key = analysis_types_dict[analysis_key][each_key]['type']
            
            ## GETTING COLOR
            color = color_dict[each_key]
            
            ## DEFINING KEY
            output_dict = sims_output_dict[sim_key] 
            
            ## DEFINING LINSTYLE
            if analysis_key == "reverse":
                linestyle = ':'
                label = "reverse"
                fillstyle = 'none'
                fillstyle = 'none'
                fmt = '^'
                markerfacecolor = 'white'
            else:
                linestyle = '-'
                label = "forward"
                fillstyle = 'full'
                fmt = 'o'
                markerfacecolor = None
            
            ## PLOTTING SPECIFIC CONTACTS
            fig, ax = plot_specific_contacts_vs_distance(output_dict = output_dict,
                                                         each_lm = 'TAILGRPS',
                                                         xlabel = "z (nm)",
                                                         ylabel = "Hydrophobic contacts",
                                                         fig = figs,
                                                         ax = ax,
                                                         figsize = None,
                                                         color = color,
                                                         errorbar_format={
                                                                'linestyle' :linestyle,
                                                                "fmt": fmt,
                                                                'fillstyle': fillstyle,
    #                                                            "capthick": 1.5,
                                                                "markersize": 6,
                                                                'markerfacecolor': markerfacecolor,
    #                                                            "capsize": 3,
    #                                                            "elinewidth": 1.5,
                                                                "label" : label,
                                                                }
                                                         )
    
        ## ADDING LEGEND
        ax.legend()
        
        ## ADDING TICKS
        axs[-1].tick_params(axis='both', which='both', labelsize=8, labelbottom = True)
        
        # CHECKING SPECIFIC VALUES
        # sims_output_dict['us_forward_R12']['TAILGRPS'][['z_dist_avg', 'NP_ALK_RGRP-LM_TAILGRPS_avg', 'folder']]
        
    ## DEFINING LIMITS FOR DISTANCE PLUMED
    ylim = (-50, 1200)
    ylabels=np.arange(0, 1200, 200)
        
    ax.set_ylim(ylim)
    ax.set_yticks(ylabels)

    
    ## SETTING X LABEL
    axs[-1].set_xlabel('z (nm)')
    
    ## TIGHT LAYOUT
    figs.tight_layout()
    
    ## REMOVING WHITE SPACE
    plt.subplots_adjust(wspace = 0, hspace = 0)
    #%%
    
    ## FIGURE NAME
    figure_name = fig_prefix + "_combined_pmf"

    ## SAVING FIGURE
    plot_funcs.store_figure( fig = figs,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
    
    
    
    #%% FIGURE 4: UNBIASED SIMULATIONS
    # This figure is designed to plot the hydrophobic contacts versus initial 
    # z value    
    ## DEFINING SIM TYPE LIST
    sim_type_list = UNBIASED_SIM_LIST
    
    ## EXTRACTION OF UNBIASED SIMS
    storage_list = load_unbiased_with_US_plumed_contacts(sim_type_list = sim_type_list,
                                                         last_time_to_extract_ps = 10000,
                                                         desired_contacts = 'NP_ALK_RGRP-LM_TAILGRPS')
    
    ## CREATING DATAFRAME FROM UNBIASED
    storage_df = generate_dataframe_for_unbiased(storage_list = storage_list,
                                                 adsorption_cutoff = 6,
                                                 hydrophobic_cutoff = None
                                                 )
    
    
    ## ADDING THE BENZENE RESULTS
    storage_list_addn = load_unbiased_with_US_plumed_contacts(sim_type_list =[ "modified_FF_unbiased_after_us_ROT017_1.900nm" ],
                                                              last_time_to_extract_ps = 10000,
                                                              desired_contacts = 'NP_ALK_RGRP-LM_TAILGRPS')
    
    
    ## STORAGE LIST
    storage_list_with_addn = storage_list + storage_list_addn
    
    ## GETTING DF WITH ADDITION
    storage_df_with_addn = generate_dataframe_for_unbiased(storage_list = storage_list_with_addn,
                                                 adsorption_cutoff = 6,
                                                 hydrophobic_cutoff = None
                                                 )
    
    
    ## CREATING SHORTERNED DF
    shortened_df = storage_df_with_addn[['sim_key','avg_z','avg_contacts','avg_unbiased_z', 'avg_unbiased_contacts', 'desorbed']]
    #%%
    
    ## STORING
    path_csv = os.path.join(STORE_FIG_LOC, "SI_COMP_TABLE_S2_unbiased_sim_details.csv")
    shortened_df.to_csv(path_csv)
    print(shortened_df)
    print("Writing to: %s"%(path_csv))
    
    #%% PLOTTING FIGURE
    
    ## DEFINING FIGURE SIZE
    figsize = (8, 7)
    
    ## GENERATING FIGURE AND AXIS FOR HYDROPHOBIC CONTACTS
    fig, ax = plot_hydrophobic_contacts_vs_z(storage_df,
                                             fig = None,
                                             ax = None,
                                             fig_size = figsize,
                                             )
            
    ## SETTING LIMITS
    ax.set_xlim([1, 7])
    ax.set_ylim([-50, 800])
    
    ## SETTING TICKS
    ax.set_yticks(np.arange(0,900,100))
    
    ## DRAWING LINE
    ax.axhline(y=40, linestyle='--', color = 'k', linewidth = 1.5)
    
    #%%
    
    ## DEFINING FIG PREFIX
    fig_prefix="COMP_4"
    
    ## FIGURE NAME
    figure_name ="-".join([fig_prefix, 'unbiased_hydrophobic_vs_z'])
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
    
    
    #%% FIGURE 5 - PLOTTING US FOR HYDROPHOBIC CONTACTS
    ## DEFINING FIGURE SIZE
    figsize = (8, 6)
    
    ## DEFINING FIG PREFIX
    fig_prefix="COMP_5"
                
    ## DEFINING DESIRED DICT
    desired_key = "hydrophobic_contacts"
    # "hydrophobic_contacts_iter2"
    # "hydrophobic_contacts"
    # "US-z-pulling"
    # "hydrophobic_contacts"
    # 'hydrophobic_contacts'
    # desired_key = 'all_np_contacts'
    
    ## DEFINING WHAM FOLDERS
    wham_folder_dict = {
            **PATH_DICT[desired_key]
            } 
    
    ## DEFINING FILE
    fe_file = "umbrella_free_energy.dat"
            
    
    ## PLOTTING FREE ENERGIES
    fig, ax, storage = plot_free_energies_for_wham_folder(wham_folder_dict,
                                                 want_peaks = False,
                                                 wham_styles_dict = COLOR_DICT,
                                                 fig_size = figsize,
                                               fe_file_prefix_suffix = {
                                                       'prefix': 'umbrella_free_energy_',
                                                       'suffix': '.dat',
                                                       },
                                                    errorbar_format={
                                                            "fmt": "None",
                                                            "capthick": 1.5,
                                                            "markersize": 8,
                                                            "capsize": 3,
                                                            "elinewidth": 1.5,
                                                            },
                                                x_line = 40,
                                                 )
    
    ## MOVING LEGEND
    ax.legend(loc="upper left")
    

    
    #%%
    ## DEFINING AXIS
    ax.set_xlim([-5,150])
    ax.set_xticks(np.arange(0,165,15))
    
    ax.set_ylim([-5, 250]) # 350
    ax.set_yticks(np.arange(0,300, 50)) # 400
    
    #%%
    
    ## FIGURE NAME
    figure_name ="-".join([fig_prefix, desired_key])
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg', # 'svg'
                             save_fig = SAVE_FIG,
                             )
                             
    #%%

    ###########################################################################
    ### SUPPORTING INFORMATION
    ###########################################################################         
    
    #%% FIG_COMP_S2: NANOPARTICLE RDF
        
    ## DEFINING FIGURE SIZE
    fig_size = (8.55, 15) # 6.4125
    
    ## GETTING DICT
    NP_WATER_SIM_DICT = {
            'C1':
                {
                        'main_sim_dir': r"ROT_WATER_SIMS",
                        'specific_sim': r"EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_Trial_1",
                        },
            'Bn':
                {
                        'main_sim_dir': r"ROT_WATER_SIMS_MODIFIEDFF",
                        'specific_sim': r"EAM_300.00_K_2_nmDIAM_ROT017_CHARMM36jul2017mod_Trial_1",
                        },
            'C10':
                {
                        'main_sim_dir': r"ROT_WATER_SIMS",
                        'specific_sim': r"EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1",
                        },
            }
                
    ## PLOTTING
    figs, axs = plot_np_rdfs(rdf_xvg_list = RDF_XVG_DICT)
    
    ## SETTING XLIM
    axs[-1].set_xlim([0, 5])
    
    ## SETTING Y LIMS
    [ax.set_ylim([0,24]) for ax in axs]
    [ax.set_yticks(np.arange(0,24,4)) for ax in axs]
    
    ## DEFINING FIGURE NAME
    figure_name = "SI_COMP2_NP_RDF"
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = figs,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
                             
                             
                             
    #%% FOR RVL CLASS
    
    ## DEFINING FIGURE SIZE
    fig_size = (8.55, 8.55) # 6.4125
    
    ## GETTING DICT
    NP_WATER_SIM_DICT = {
#            'C1':
#                {
#                        'main_sim_dir': r"ROT_WATER_SIMS",
#                        'specific_sim': r"EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_Trial_1",
#                        },
#            'Bn':
#                {
#                        'main_sim_dir': r"ROT_WATER_SIMS_MODIFIEDFF",
#                        'specific_sim': r"EAM_300.00_K_2_nmDIAM_ROT017_CHARMM36jul2017mod_Trial_1",
#                        },
            'C10':
                {
                        'main_sim_dir': r"ROT_WATER_SIMS",
                        'specific_sim': r"EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1",
                        },
            }
                
    ## PLOTTING
    figs, axs = plot_np_rdfs(rdf_xvg_list = RDF_XVG_DICT,
                             fig_size = fig_size)
    
    ## SETTING XLIM
    axs[-1].set_xlim([0, 5])
    
    ## SETTING Y LIMS
    [ax.set_ylim([0,24]) for ax in axs]
    [ax.set_yticks(np.arange(0,28,4)) for ax in axs]
    
    ## DEFINING FIGURE NAME
    figure_name = "SI_COMP2_NP_RDF_COURSEWORK"
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = figs,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg',
                             save_fig = SAVE_FIG,
                             )
    
    
    #%% FIG COMP S3: SAMPLING TIME / HISTOGRAMS FOR UMBRELLA SAMPLING SIMULATIONS
        
    #################################    
    ### LOADING DEFAULT VARIABLES ###
    #################################
    ## DEFINING TIME
    time_from_end = "40000.000"
    
    ## DEFINING FREE ENERGY FILE PREFIXES
    free_energy_prefixes = {
            'forward': {
                'fe_file_prefix': "umbrella_free_energy-forward-%s"%(time_from_end),
                'sort_values': 'end',
                },
            'reverse': {
                'fe_file_prefix': "umbrella_free_energy-rev-%s"%(time_from_end),
                'sort_values': 'begin',
                },
            }
    
    ## DEFINING FIGURE SIZE
    fig_size = (17.1, 17.1)    
    
    ## GETTING LIMITS
    CONVERGENCE_LIMITS= {
            'histogram':
                {'ylim': [-1, 10],
                 'yticks': np.arange(0, 12, 2)
                 },
            'pmf':{
                    'ylim': (-100, 1100),
                    'yticks': np.arange(-100,1100,200),
                    },
            'xlim': [1,7],
            'xticks': np.arange(1,8,1)
            }

    #%%
    ## DEFINING DESIRED KEY
    desired_key = "US-z-pulling"
    
    ## LOADING SAMPLING TIME
    storage_sampling_time = load_sampling_time_data(desired_key = desired_key,
                                                    free_energy_prefixes = free_energy_prefixes)
    
    ## GETTING HISTOGRAM DATA
    storage_histogram_data = load_histogram_data(desired_key = desired_key)
    
    #%% PLOTTING FOR HISTOGRAM WITH SAMPLING TIME
    
    ## CLOSING ALL FIGS
    plt.close('all')
    

    ## PLOTTING
    fig, axs = plot_us_histo_and_convergence(storage_histogram_data = storage_histogram_data,
                                             storage_sampling_time = storage_sampling_time,
                                             axis_dict = CONVERGENCE_LIMITS,
                                             want_zero_line = True,
                                             free_energy_prefixes = free_energy_prefixes,
                                             )
    
    ## FIGURE NAME
    figure_name = "SI_COMP3_US_convergence"

    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
                             
    #%% FIGURE S4: US CONVERGENCE FOR PUSHING SIMULATIONS
                             
    ## DEFINING DESIRED KEY
    desired_key = "US-z-pushing"
    
    ## LOADING SAMPLING TIME
    storage_sampling_time = load_sampling_time_data(desired_key = desired_key,
                                                   free_energy_prefixes = free_energy_prefixes)
    
    ## GETTING HISTOGRAM DATA
    storage_histogram_data = load_histogram_data(desired_key = desired_key)
    
    #%% PLOTTING FOR HISTOGRAM WITH SAMPLING TIME
    
    ## CLOSING ALL FIGS
    plt.close('all')
    
    ## DEFINING FIGURE SIZE
    fig_size = (17.1, 17.1)    
    
    ## GETTING LIMITS
    CONVERGENCE_LIMITS= {
            'histogram':
                {'ylim': [-1, 10],
                 'yticks': np.arange(0, 12, 2)
                 },
            'pmf':{
                    'ylim': (-100, 1100),
                    'yticks': np.arange(-100,1100,200),
                    },
            'xlim': [1,7],
            'xticks': np.arange(1,8,1)
            }
    
    ## PLOTTING
    fig, axs = plot_us_histo_and_convergence(storage_histogram_data = storage_histogram_data,
                                             storage_sampling_time = storage_sampling_time,
                                             axis_dict = CONVERGENCE_LIMITS,
                                             want_zero_line = True,
                                             zero_location = -1, # Setting zero location at end
                                             )
    
    ## FIGURE NAME
    figure_name = "SI_COMP4_US_convergence"

    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
                             
                             
    #%% FIGURE S5/S6: US CONVERGENCE OF HYDROPHOBIC CONTACT SIMULATIONS
    ## DEFINING DESIRED KEY
    desired_key = "hydrophobic_contacts"
    
    ## REDEFININE FREE ENERGY PREFIXES
    free_energy_prefixes = FREE_ENERGY_PREFIX_INFO
    
    ## LOADING SAMPLING TIME
    storage_sampling_time = load_sampling_time_data(desired_key = desired_key,
                                                   free_energy_prefixes = free_energy_prefixes)
    
    ## GETTING HISTOGRAM DATA
    storage_histogram_data = load_histogram_data(desired_key = desired_key)
        

    #%% FIGURE S5: HISTOGRAMS
    
    ## GETTING LIMITS
    CONVERGENCE_LIMITS= {
            'histogram':
                {'ylim': [-0.2, 1.3],
                 'yticks': np.arange(0, 1.3, 0.4)
                 },
            'pmf':{
                    'ylim': ([0, 300]),
                    'yticks': np.arange(0,300, 50),
                    },
            'xlim': [-5,155],
            'xticks': np.arange(0,165,15)
            }
            
    #%%
    
    ## CLOSING ALL FIGS
    plt.close('all')
    
    ## DEFINING FIGURE SIZE
    fig_size = (17.1, 12)  
    
    ## PLOTTING
    fig, axs = plot_us_histo_and_convergence(storage_histogram_data = storage_histogram_data,
                                             storage_sampling_time = storage_sampling_time,
                                             axis_dict = CONVERGENCE_LIMITS,
                                             want_zero_line = False,
                                             zero_location = None, 
                                             xlabel = "Hydrophobic contacts",
                                             plot_type = "histo_vertical",
                                             free_energy_prefixes = free_energy_prefixes,
                                             )    
    
    #%%
    ## FIGURE NAME
    figure_name = "SI_COMP5_hydrophobic_PMF_histograms"

    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
    
    #%% FIGURE S6: SAMPLING TIME OF PMFS
    
    ## CLOSING ALL FIGS
    plt.close('all')
    
    ## DEFINING FIGURE SIZE
    fig_size = (15, 14)
    
    ## PLOTTING
    fig, axs = plot_us_histo_and_convergence(storage_histogram_data = storage_histogram_data,
                                             storage_sampling_time = storage_sampling_time,
                                             axis_dict = CONVERGENCE_LIMITS,
                                             want_zero_line = False,
                                             zero_location = None, 
                                             xlabel = "Hydrophobic contacts",
                                             plot_type = "sampling_time_vertical",
                                             free_energy_prefixes = free_energy_prefixes,
                                             )    
    #%%
    ## FIGURE NAME
    figure_name = "SI_COMP6_hydrophobic_PMF_sampling_time"

    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
    
    #%% FIGURE S7: UNBIASED SIMULATIONS FROM THE HYDROPHOBIC PMFS
    
    ## DEFINING SIM DICT
    UNBIASED_AFTER_HYDROPHOBIC_PMF_DICT = {
            'C1': {
                    '10 ns': 'plumed_unbiased_10ns_US_ROT001_40',
                    '15 ns': 'plumed_unbiased_15ns_US_ROT001_40',
                    '20 ns': 'plumed_unbiased_20ns_US_ROT001_40',
                    '50 ns': 'plumed_unbiased_50ns_US_ROT001_40',
                    },
            'Bn': {
                    '10 ns': 'modified_FF_hydrophobic_pmf_unbiased_ROT017_40_10000',
                    '15 ns': 'modified_FF_hydrophobic_pmf_unbiased_ROT017_40_15000',
                    '20 ns': 'modified_FF_hydrophobic_pmf_unbiased_ROT017_40_20000',
                    '50 ns': 'plumed_unbiased_50ns_US_ROT017_40',
                    },
            'C10': {
                    '10 ns': 'plumed_unbiased_10ns_US_ROT012_40',
                    '15 ns': 'plumed_unbiased_15ns_US_ROT012_40',
                    '20 ns': 'plumed_unbiased_10ns_US_ROT012_40',
                    '50 ns': 'plumed_unbiased_50ns_US_ROT012_40',
                    }
            }
    
    ## DEFINING TYPES
    TIME_DICT = {
            '10 ns': {
                    'color': 'blue',
                    'label': 'Config 1',
                    },
            '15 ns': {
                    'color': 'green',
                    'label': 'Config 2',
                    },
            '20 ns': {
                    'color': 'red',
                    'label': 'Config 3',
                    },
            '50 ns': {
                    'color': 'black',
                    'label': 'Config 4',
                    },            
            }
    
    ## CREAITNG SUBPLOT
    nrows = len(UNBIASED_AFTER_HYDROPHOBIC_PMF_DICT)
    ncols = 1
    
    ## DEFINING FIGURE SIZE
    fig_size = (8.55, 15) # 6.4125

    ## DEFINING Y LIMITS
    

    ## CREATING SUBPLOTS
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, 
                             figsize = plot_funcs.cm2inch( *fig_size ))
    
    ## DEFINING TO PLOT
    contact_label = HYDROPHOBIC_CONTACT_LABEL
    
    ## DEFINING Y LIM
    ylim = [-5, 75]
    yticks = np.arange(0, 75, 10)
    
    ## LOOPING
    for key_idx, each_key in enumerate(UNBIASED_AFTER_HYDROPHOBIC_PMF_DICT):
        ## GETTING DICT
        current_sim_key_dict = UNBIASED_AFTER_HYDROPHOBIC_PMF_DICT[each_key]
        
        ## DEFINING AXIS
        ax = axs[key_idx]
        
        ## ADDING TITLE            
        ax.text(.5,.9,each_key,
            horizontalalignment='center',
            transform=ax.transAxes)
        
        ## ADDING HORIZONTAL LINE
        ax.axhline(y=0, linestyle='--', color='gray', linewidth=1)
        
        ## LOOPING THROUGH EACH KEY
        for each_sim_key in current_sim_key_dict:
            ## GETTING CURRENT SIM
            current_sim_key = current_sim_key_dict[each_sim_key]
            
            ## GETTING MAIN SIM
            main_sim_dir = NPLM_SIM_DICT[current_sim_key]['main_sim_dir']
            specific_sim = NPLM_SIM_DICT[current_sim_key]['specific_sim']
            
            ## GETTING ALL JOB TYPES
            job_types = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                       main_sim_dir = main_sim_dir,
                                       specific_sim = specific_sim,
                                       )
            
            ## DEFINING PATH TO SIMULATION
            path_to_sims = job_types.path_simulation_list[0]
                                       
            
            if current_sim_key.startswith('plumed') and 'unbiased' not in current_sim_key:
                time_stride=10
            else:
                time_stride = TIME_STRIDE
                
            ## LOADING PLUMED FILE
            plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sims,
                                                                                      plumed_input_file=PLUMED_INPUT_FILE,
                                                                                      time_stride=time_stride )
            
            ## GETTING STYLES
            styles = TIME_DICT[each_sim_key]
            
            ## PLOTTING
            x = covar_output['time'] / 1000.0
            y = covar_output[contact_label]
            
            ## plotting
            ax.plot(x, y, **styles)
        
        ## ADDING LABELS
        ax.set_ylabel("Contacts")
        
        ## SETTING LIMITS
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        
        if key_idx == 0:
            ## ADDING LEGEND 
            ax.legend()
    
    ## ADDING X LABEL
    ax.set_xlabel("Simulation time (ns)")
        
    ## TIGHT LAYOUT
    fig.tight_layout()
    ## ADJUSTING SPACE
    plt.subplots_adjust(wspace=0, hspace=0)
                
     ## SAVIG FIG   
    figure_name = "SI_7_COMP_unbiased_sim"
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg',
                             save_fig = SAVE_FIG,
                             )
            
            
    #%% TABLE S1 - Number of atoms, etc.
    
    ## GETTING DICT FOR INFORMATION
    TOTAL_ATOM_TABLE_DICT = {
            'DOPC-water':
                {
                        '-': {
                                'parent_sim_dir': "/Volumes/akchew/scratch/nanoparticle_project/prep_files/lipid_bilayers",
                                'main_sim_dir': 'DOPC-300.00-196_196',
                                'specific_sim': 'gromacs',
                                'prefix': 'step7_9',
                                },
                },
            'NP-water':
                {
                        'C1': {
                                'parent_sim_dir': global_vars.NP_PARENT_SIM_PATH,
                                'main_sim_dir': 'ROT_WATER_SIMS',
                                'specific_sim': 'EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_Trial_1',
                                'prefix': 'sam_prod',
                                },
                        'Bn': {
                                'parent_sim_dir': global_vars.NP_PARENT_SIM_PATH,
                                'main_sim_dir': 'ROT_WATER_SIMS_MODIFIEDFF',
                                'specific_sim': 'EAM_300.00_K_2_nmDIAM_ROT017_CHARMM36jul2017mod_Trial_1',
                                'prefix': 'sam_prod',
                                },
                        'C10': {
                                'parent_sim_dir': global_vars.NP_PARENT_SIM_PATH,
                                'main_sim_dir': 'ROT_WATER_SIMS',
                                'specific_sim': 'EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1',
                                'prefix': 'sam_prod',
                                },
                                
                        },
            'NP-DOPC_US-sims': {
                    'C1': {
                                'parent_sim_dir': global_vars.PARENT_SIM_PATH,
                                **global_vars.NPLM_SIM_DICT['us_forward_R01'],
                                'prefix': 'nplm_prod',
                                'sim_folder': '5.100',
                            },
                    'Bn': {
                                'parent_sim_dir': global_vars.PARENT_SIM_PATH,
                                **global_vars.NPLM_SIM_DICT['modified_FF_us_forward_R17'],
                                'prefix': 'nplm_prod',
                                'sim_folder': '5.100',
                            },
                    'C10': {
                                'parent_sim_dir': global_vars.PARENT_SIM_PATH,
                                **global_vars.NPLM_SIM_DICT['us_forward_R12'],
                                'prefix': 'nplm_prod',
                                'sim_folder': '5.100',
                            },
                    
                    },
                            
            'NP-DOPC_hydrophobic-sims': {
                    'C1': {
                                'parent_sim_dir': global_vars.PARENT_SIM_PATH,
                                'main_sim_dir': PARENT_FOLDER_DICT['hydrophobic_contacts'],
                                'specific_sim': os.path.dirname(PATH_DICT['hydrophobic_contacts']['C1']),
                                'prefix': 'nplm_prod',
                                'sim_folder': '0.0',
                            },
                    'Bn': {
                                'parent_sim_dir': global_vars.PARENT_SIM_PATH,
                                'main_sim_dir': PARENT_FOLDER_DICT['Bn_new_FF_hydrophobic_contacts'],
                                'specific_sim': os.path.dirname(PATH_DICT['hydrophobic_contacts']['Bn']),
                                'prefix': 'nplm_prod',
                                'sim_folder': '0.0',
                            },
                    'C10': {
                                'parent_sim_dir': global_vars.PARENT_SIM_PATH,
                                'main_sim_dir': PARENT_FOLDER_DICT['hydrophobic_contacts'],
                                'specific_sim': os.path.dirname(PATH_DICT['hydrophobic_contacts']['C10']),
                                'prefix': 'nplm_prod',
                                'sim_folder': '0.0',
                            },                    
                    },
            }
    
    ## DEFINING DEFAULT US SIM
    default_us_sim="4_simulations"
    
    ## CREATING STORAGE LIST
    storage_list = []
    
    ## LOOPING THROUGH
    for each_type in TOTAL_ATOM_TABLE_DICT:
        ## LOOPING THROUGH EACH ONE
        for lig_idx, each_ligand in enumerate(TOTAL_ATOM_TABLE_DICT[each_type]):
            ## DEFINING CURRENT DICT
            current_sim_dict = TOTAL_ATOM_TABLE_DICT[each_type][each_ligand]
            ## GETTING PARENT SIM DIRECTORY
            sim_path = os.path.join(current_sim_dict['parent_sim_dir'],
                                    current_sim_dict['main_sim_dir'],
                                    current_sim_dict['specific_sim'],
                                    )
            
            ## CHECKING IF SIM FOLDER IN FILE (US SIMS)
            if 'sim_folder' in current_sim_dict:
                sim_path = os.path.join(sim_path,
                                        default_us_sim,
                                        current_sim_dict['sim_folder']
                                        )
            
            ## DEFINING GRO FILE
            gro_file = current_sim_dict['prefix'] + '.gro'
            path_gro_file = os.path.join(sim_path,
                                         gro_file)
            
            ## LOADING TRAJECTORY
            print("Loading gro file: %s"%(path_gro_file))
            traj = md.load(path_gro_file,
                           path_gro_file)
            
            ## PRINTING
            print(traj)
            
            ## GETTING TOTAL NUMBER OF WATERS
            n_waters = calc_tools.find_total_residues(traj = traj, resname = 'HOH')[0]
            
            ## GETTING TOTAL NUMBER OF ATOMS
            n_atoms = len([atom.index for atom in traj.topology.atoms])
            
            ## GETTING BOX LENGTHS
            box_lengths = traj.unitcell_lengths[0] # 1 x 3 vector
            
            ## CREATING DICTIONARY
            output_dict = {
                    'Type': each_type,
                    'Ligand': each_ligand,
                    'N_H2O': n_waters,
                    'N_atoms': n_atoms,
                    'Box_x': box_lengths[0],
                    'Box_y': box_lengths[1],
                    'Box_z': box_lengths[2],
                    }
            
            ## APPENDING
            storage_list.append(output_dict)
            
    ## CREATING DATAFRAME
    storage_df = pd.DataFrame(storage_list)
    
    
    ## DEFINING PREFIX
    table_csv="SI_COMP_1_System_size.csv"
    
    ## STORING
    path_table = os.path.join(STORE_FIG_LOC,
                              table_csv)
    
    ## STORING TO CSV
    storage_df.to_csv(path_table)
    
    #%% SI FIGURE : BN-AUNP CONTACTS
    
    fig_size = (8.5, 8.5)
    ## DEFINING TO PLOT
    contact_label = HYDROPHOBIC_CONTACT_LABEL
    
    ## DEFINING SIMULATION KEY DICTIONARY
    sim_key_dict = {
            'Bn': 'modified_FF_unbiased_after_us_ROT017_1.900nm'
            }
    
    ## DEFINING STYLES
    styles = {
            'color': 'k',
            }
    
    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size)
    
    ## ADDING LABELS
    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel("Hydrophobic contacts")
    
    
    ## LOOPING THROUGH EACH KEY
    for each_sim_key in sim_key_dict:
        ## GETTING CURRENT SIM
        current_sim_key = sim_key_dict[each_sim_key]
        
        ## GETTING MAIN SIM
        main_sim_dir = NPLM_SIM_DICT[current_sim_key]['main_sim_dir']
        specific_sim = NPLM_SIM_DICT[current_sim_key]['specific_sim']
        
        ## GETTING ALL JOB TYPES
        job_types = nplm_job_types(parent_sim_path = PARENT_SIM_PATH,
                                   main_sim_dir = main_sim_dir,
                                   specific_sim = specific_sim,
                                   )
        
        ## DEFINING PATH TO SIMULATION
        path_to_sims = job_types.path_simulation_list[0]
                                   
        
        if current_sim_key.startswith('plumed') and 'unbiased' not in current_sim_key:
            time_stride=10
        else:
            time_stride = TIME_STRIDE
            
        ## LOADING PLUMED FILE
        plumed_input, df_extracted, covar_output = extract_plumed_input_and_covar(path_to_sim=path_to_sims,
                                                                                  plumed_input_file=PLUMED_INPUT_FILE,
                                                                                  time_stride=time_stride )
        ## PLOTTING
        x = covar_output['time'] / 1000.0
        y = covar_output[contact_label]
        
        ## plotting
        ax.plot(x, y, **styles)
    
    ## SETTING AXIS
    ax.set_ylim([200, 400])
    ax.set_yticks(np.arange(200, 450, 50))
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
     ## SAVIG FIG   
    figure_name = "SI_8_COMP_unbiased_Bn"
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg',
                             save_fig = SAVE_FIG,
                             )
                             
                             
    #%% FIGURE SX: Second iteration
    
    ## DEFINING FIGURE SIZE
    figsize = (8, 6)
    
    ## DEFINING FIG PREFIX
    fig_prefix="COMP_5"
                
    ## DEFINING DESIRED DICT
    desired_key = "hydrophobic_contacts_iter2"
    # "hydrophobic_contacts_iter2"
    # "hydrophobic_contacts"
    # "US-z-pulling"
    # "hydrophobic_contacts"
    # 'hydrophobic_contacts'
    # desired_key = 'all_np_contacts'
    
    ## DEFINING WHAM FOLDERS
    wham_folder_dict = {
            **PATH_DICT[desired_key]
            } 
    
    ## DEFINING FILE
    fe_file = "umbrella_free_energy.dat"
            
    
    ## PLOTTING FREE ENERGIES
    fig, ax, storage = plot_free_energies_for_wham_folder(wham_folder_dict,
                                                 want_peaks = False,
                                                 wham_styles_dict = COLOR_DICT,
                                                 fig_size = figsize,
                                                 fe_file = fe_file,
#                                               fe_file_prefix_suffix = {
#                                                       'prefix': 'umbrella_free_energy_',
#                                                       'suffix': '.dat',
#                                                       },
                                                    errorbar_format={
                                                            "fmt": "None",
                                                            "capthick": 1.5,
                                                            "markersize": 8,
                                                            "capsize": 3,
                                                            "elinewidth": 1.5,
                                                            },
                                                x_line = 40,
                                                 )
    
    ## MOVING LEGEND
    ax.legend(loc="upper left")
    
    ## DEFINING AXIS
    ax.set_xlim([-5,150])
    ax.set_xticks(np.arange(0,165,15))
    
    ax.set_ylim([-5, 450]) # 350
    ax.set_yticks(np.arange(0,500, 100)) # 400
    
    #%%
    ## SAVIG FIG   
     
    figure_name = "SI_9_COMP_iteration2_PMF"
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'svg',
                             save_fig = SAVE_FIG,
                             )
                             


    
    
    
    #%%
    
    ## LOOPING THROUGH EACH LIST
    for desired_key in key_list:
        ## DEFINING WHAM FOLDERS
        wham_folder_dict = {
                **PATH_DICT[desired_key]
                }
        
        ## GETTING FILES
        if desired_key.startswith("US"):
            files = FILES_DICT['z_com']
        else:
            files = FILES_DICT['hydrophobic_contacts']
            
        ## GETTING COVAR
        covar_file = files['covar_file']
        times_from_end = files['times_from_end']
        
        ## GETTING HISTOGRAM INFORMATION
        delta = files['delta']
        bin_width = files['bin_width']
        
        ## GETTING X LABEL
        xlabel = files['xlabel']
        
        ## LOOPING THROUGH EACH KEY
        for wham_idx, each_key in enumerate(wham_folder_dict):
            ## IF NONE, IT'LL LOOK FOR ALL OF THEM
            contacts = None
            
            ## DEFINING FOLDER
            wham_folder = wham_folder_dict[each_key]
        
            ## DEFINING SIMULATION PATH
            sim_folder = os.path.dirname(wham_folder)
            
            ## DEFINING RELATIVE PATH TO SIM
            relative_sim_path = "4_simulations"
            
            ## DEFINING PATH TO SIM
            path_to_sim = os.path.join(sim_folder,
                                       relative_sim_path)
            
            if contacts is None:
                folders = [ str(each) for each in glob.glob(path_to_sim + "/*") ]
                ## SORTING
                folders.sort()
                contacts = [ float(os.path.basename(each)) for each in folders ]
                histogram_range = (np.min(contacts)-delta, np.max(contacts)+delta)
            else:
                histogram_range = (-5, 105)
            
            ## GENERATING HISTOGRAM
            hist = plot_covar_histogram()
            
            ## LOADING DATA
            covar_df = hist.load_covar(path_to_sim = path_to_sim,
                                       contacts = folders,
                                       covar_file =covar_file,
                                       )
            ## GENERATING PLOT
            fig, ax = hist.plot(covar_df = covar_df,
                                histogram_range = histogram_range,
                                bin_width = bin_width, # 2
                                fig_size = (9.55, 6.5),
                                want_legend = False, 
                                time_starting_from = 0,
                                times_from_end = times_from_end,
                                xlabel = xlabel)
            
            ## MODIFYING LEGEND
            # Removing legend
            # ax.get_legend().remove()
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=3)
            
            ## FIGURE NAME
            figure_name = "%s-WHAM_HIST-"%(desired_key) + os.path.basename(os.path.dirname(wham_folder))
            #%%
            ## SAVING FIGURE
            plot_funcs.store_figure(fig = fig, 
                         path = os.path.join(STORE_FIG_LOC,
                                             figure_name), 
                         fig_extension = 'png', 
                         save_fig=True,)
#
    
    
    #%% PLOTTING HISTOGRAMS
    
    
    
    
    
    #%% FIGURE 4
    ######################################
    ### PULLING THEN UNBIASED CONTACTS ###
    ######################################            
    
    sim_type_list = [
            'pullthenunbias_ROT012',
            ]
    
    fig_prefix = "COMP_4"
    ## GETTING FIGURE SIZE
    fig_size=(8,6)
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## GETTING JOB INFOMRATION
        job_info = nplm_job_types(parent_sim_path = parent_sim_path,
                                  main_sim_dir = main_sim_dir,
                                  specific_sim = specific_sim)
        
        ## DEFINING INPUTS
        contact_inputs= {
                'job_info': job_info,
                'last_time_ps': 50000,
                }
        
        ## DEVELOPING SCRIPT FOR CONTACTS
        extract_contacts = extract_num_contacts(**contact_inputs)
        
        ## DEFINING PATH TO SIMULATION
        path_to_sim = job_info.path_simulation_list[0]
        
        ## GETTING CONTACTS PER GROUP
        contacts_dict = extract_contacts.analyze_num_contacts(path_to_sim = path_to_sim,
                                                  want_nplm_grouping = True,
                                                  want_com_distance = True)
        
        
        ## PLOTTING CONTACT VERSUS TIME
        fig, ax, ax2 = extract_contacts.plot_num_contacts_vs_time_for_config(path_to_sim = path_to_sim,
                                                                             want_com_distance = True,
                                                                             want_com_distance_abs = True,
                                                                             want_tail_groups = True)
        ## SETTING Y LABEL
        ax.set_ylabel("DOPC tail contacts")
        ax2.set_ylabel("z (nm)")
        
        ## SETTING LIMITS
        ax.set_ylim([15, 160])
        ax.set_yticks(np.arange(25, 175, 25))
        
        ax.set_xlim([0, 50000])
        ax.set_xticks(np.arange(0,60000,10000))
        
        ax2.set_ylim([5.1,6.5])
        ax2.set_yticks(np.arange(5.2, 6.5, 0.2))
        
        ## CHANGING COLOR
        ax.spines['right'].set_color('red')
        ax.spines['top'].set_color('red')
        ax2.spines['top'].set_color('red')
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')
        
        ## UPDATING FIGURE SIZE
        fig = plot_funcs.update_fig_size(fig,
                                         fig_size_cm = fig_size)
        
        ## GETTING TIGHT LAYOUT
        fig.tight_layout()
        
        ## SAVING FIGURE
        figure_name = fig_prefix + "_sim_type-%s"%(sim_type)

        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 save_fig = SAVE_FIG,
                                 fig_extension = FIG_EXTENSION,
                                 )
        
        
        #%%
        
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(IMAGE_LOC,
                                                     figure_name),
                                 save_fig = True,
                                 )
    #%%  
    ###########################################################################
    ### GARBAGEO
    ###########################################################################         
    
    
    #%% PLOTTING FREE ENERGIES FOR FORWARD AND REVERSE
    
    ## DEFINING FIGURE SIZE
    figsize = (8, 6)
    
    ## DEFINING DESIRED DICT
    desired_key ="US-z-pulling"
    
    ## DEFINING WHAM FOLDERS
    wham_folder_dict = {
            **PATH_DICT[desired_key]
            }
    
    ## DEFINING FILE
    fe_file = "umbrella_free_energy.dat"
            
    ## PLOTTING FREE ENERGIES
    fig, ax, _ = plot_free_energies_for_wham_folder(wham_folder_dict,
                                                 want_peaks = False,
                                                 wham_styles_dict = COLOR_DICT,
                                                 fig_size = figsize,
                                                 xlabel = "z (nm)")
    
    ## SETTING LIMITS
    ax.set_xlim([1,7])
    ax.set_ylim([-100,1100])
    ax.set_yticks(np.arange(-100,1200,200))
    
    ## ADDING 0 LINE
    ax.axhline(y=0, linestyle='--', color = 'k', linewidth = 1.5)
    
    ## DEFINING FIGURE NAME
    figure_name="%s-PMF"%(desired_key)
    #%%
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'png',
                             save_fig = SAVE_FIG,
                             )
    
    #%% REVERSE US
    
    ## DEFINING DESIRED DICT
    desired_key ="US-z-pushing"
    
    ## DEFINING WHAM FOLDERS
    wham_folder_dict = {
            **PATH_DICT[desired_key]
            }
    
    ## DEFINING FILE
    fe_file = "umbrella_free_energy.dat"
            
    ## PLOTTING FREE ENERGIES
    fig, ax = plot_free_energies_for_wham_folder(wham_folder_dict,
                                                 want_peaks = False,
                                                 wham_styles_dict = COLOR_DICT,
                                                 fig_size = figsize,
                                                 xlabel = "z (nm)",
                                                 zero_location = -1)
    
    ## SETTING LIMITS
    ax.set_xlim([1,7])
    ax.set_ylim([-100,1100])
    ax.set_yticks(np.arange(-100,1200,200))
    
    ## ADDING 0 LINE
    ax.axhline(y=0, linestyle='--', color = 'k', linewidth = 1.5)
    
    ## DEFINING FIGURE NAME
    figure_name="%s-PMF"%(desired_key)
    
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = 'png',
                             save_fig = SAVE_FIG,
                             )
    



    ## LOOPING THROUGH EACH
    for desired_key in key_list:
        ## DEFINING WHAM FOLDERS
        wham_folder_dict = {
                **PATH_DICT[desired_key]
                }
        
        ## LOOPING THROUGH EACH KEY
        for each_key in wham_folder_dict:
            ## DEFINING FOLDER
            wham_folder = wham_folder_dict[each_key]
            
            ## LOOPING THROUGH FORWARD AND REVERSE KEYS
            for fe_key in free_energy_prefixes:
                ## DEFINING DETAILS
                current_dict = free_energy_prefixes[fe_key]
                ## EXTRACTING WHAM INFO
                wham_storage, sampling_time = main_sampling_time(wham_folder = wham_folder,
                                                                 **current_dict)
                
                ## PLOTTING WHAM STORAGE        
                fig, ax = sampling_time.plot_sampling_time(wham_storage = wham_storage,
                                                          fig_size = figsize,
                                                          xlabel="z (nm)") # 6.5
                
                ## DRAWING ZERO LINE
                ax.axhline(y=0, color = 'k', linestyle = '--', linewidth = 1.5)
                
                ## SETTING LIMITS
                if desired_key.startswith("US"):
                    ax.set_xlim([1,7])
                    ax.set_ylim([-100,1100])
                    ax.set_yticks(np.arange(-100,1200,200))
                else:
                    ax.set_xlim([-5,155])
                    ax.set_xticks(np.arange(0,160,20))
                    
                    ax.set_ylim([-50,350])
                    ax.set_yticks(np.arange(-50,400,50))
                    
                    ## MOVING LEGEND
                    ax.legend(loc="lower right")
                    
                    ## tIGHT LAYTOUT
                    fig.tight_layout()
                    
                ## FIGURE NAME
                figure_name = "%s-WHAM_SAMPLING-%s-%s-"%(desired_key, time_from_end, fe_key) + os.path.basename(os.path.dirname(wham_folder))

                ## SAVING FIGURE
                plot_funcs.store_figure(fig = fig, 
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name), 
                             fig_extension = 'png', 
                             save_fig=False,)
                             
                             
    #%% SI MATERIALS: PLOTTING HISTOGRAMS
                           
    ## TODO: Fix importing tool for different covars
    
    plt.close('all')
    ## DEFINING DESIRED DICT
    key_list = [
                "US-z-pulling",
#                "US-z-pushing",
#                "hydrophobic_contacts",
#                "hydrophobic_contacts",
#                "hydrophobic_contacts_iter2"
                ]
    

    
    
    