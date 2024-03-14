#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_gnp_database.py

This script is designed to analyze gold nanoparticle databases and output 
information for them. The idea is that we want to visualize the database and 
selectively choose nanoparticles with representative ligands. 

Written by: Alex K. Chew (11/09/2020

"""
## IMOPRTING MODULES
import os
import numpy as np
import pandas as pd

## DEFINING GLOBAL VARIABLES
GNP_COL_NAME = 'Index' # Name of column like GNP1, GNP2, .... etc.


## IMPORTING RDKIT
from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
%matplotlib inline

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORTING CUSTOM FUNCTIONS
from MDDescriptors.application.np_descriptors.convert_database_to_ligands import \
    create_lig_and_gnp_database_from_input

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING EXTENSION
FIG_EXTENSION = "png"
SAVE_FIG = True

## DEFINING STORAGE OUTPUT
STORE_FIG_LOC=r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_descriptors_manuscript/Figures/Output_images"

## GETTING INDICES
def get_indices_by_limit_dict(total_points,
                              limit_dict):
    '''
    This function gets indices baesd on the limit dictionary. It will also 
    remove any potential repeats from the limits.
    INPUTS:
        total_points: [int]
            total nubmer of points
        limit_dict: [dict]
            dictionary showing the lower, middle, and upper limits of a dict
    OUTPUTS:
        
    '''
    ## CREATING EMPTY INDICES
    indices = np.arange(total_points)
    
    ## GETTING INDICES OF LOWER
    lower_indices = np.arange(limit_dict['lower'])
    
    ## GETTING UPPER INDICES
    upper_value = total_points - limit_dict['upper']
    upper_indices = np.arange(upper_value, total_points)
    
    ## GETTING MIDDLE VALUES
    middle_index = int(total_points / 2) - 1
    ## GETTING LIMIT
    middle_num_pts = limit_dict['middle']
    middle_indices = np.arange(middle_num_pts)
    middle_indices_center = int(middle_num_pts / 2) - 1
    ## SUBTRACTING MIDDLE
    middle_indices -= middle_indices_center
    ## SHIFTING
    middle_indices += middle_index
    
    ## COMBINING ALL INDICES
    indices = np.unique(np.concatenate( (lower_indices, middle_indices, upper_indices) ))
    
    return indices

### FUNCTION TO PLOT DESCRIPTOR
def plot_horizontal_bar_descriptor(df,
                                  descriptor_name,):
    '''
    This function plots a horizontal bar plot for each GNP based on 
    inputs. 
    INPUTS:
        df: [dataframe]
            pandas dataframe containing information about your GNPs
        descriptor_name: [str]
            name of descriptors
    OUTPUTS:
        
    '''
    ## CREATING FIGURE
    fig_size = FIGURE_SIZE
    # (8, 20)
    # FIGURE_SIZE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size)
    
    ## GETTING X VALUES
    x = df[GNP_COL_NAME]
    y = df[descriptor_name]
    
    ## GETTING POSITIONS
    y_pos = np.arange(len(x))
    
    ## PLOTTING HORIZONTAL
    ax.barh(y_pos, y, align='center', height = 0.6, color = 'k')
    
    ## ADDING Y LABEL
    ax.set_xlabel(descriptor_name)
    
    ## SETTING Y TICKS    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    ## GETTING TIGHT
    fig.tight_layout()
    
    return fig, ax


#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    
    ## DEFINING PATH
    path_to_database = r"/Volumes/akchew/scratch/MDLigands/database_ligands"
    
    ## DEFINING DATABASE NAME
    database_name = "logP_exp_data.csv"
    database_path = os.path.join(path_to_database, database_name )
    
    ## LOADING DATAFRAME
    database_df = pd.read_csv(database_path)
    
    ## DEFINING DESCRIPTOR
    descriptor_name = "logP"
    
    ## DEFINING WHICH ONES YOU WANT
    limit_dict = {
            'lower' : 2,
            'middle': 2,
            'upper': 2
            } # means how many points you want in the median, lower, and upper

    ## GETTING INDICES
    indices = get_indices_by_limit_dict(total_points = len(database_df),
                                        limit_dict = limit_dict)
    
    ## SORTING
    df_sorted = database_df.sort_values(by = descriptor_name)
    
    ## DOWN SELECTING BASED ON CRITERIA (NO MULTI LIGANDS)
    df_down_selected = df_sorted.loc[df_sorted['Multilig'] == False]
    
    ## GETTING INDICES
    indices = get_indices_by_limit_dict(total_points = len(df_down_selected),
                                        limit_dict = limit_dict)
    
    ## DF FINAL
    df_shortened = df_down_selected.iloc[indices]
    
    #%%
    
    ## PLOTTING HORIZONTAL BAR DESCRIPTOR
    fig, ax = plot_horizontal_bar_descriptor(df = df_shortened,
                                             descriptor_name = descriptor_name,)
    
    ## STORING FIGURE    
    figure_name = '_'.join([descriptor_name, "shortened"])
    
    ## STORING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = FIG_EXTENSION,
                             save_fig = SAVE_FIG,
                             )
                             
    #%% GETTING GNP INFORMATION
    
    ## CREATING DATAFRAMES FOR LIG AND GNP
    output_df, storage_for_gnps_df = create_lig_and_gnp_database_from_input(database_path = database_path,
                                                                            want_output_csv = False)
    
    
    #%%
    ## GETTING DF
    df_gnp_with_ligs = storage_for_gnps_df[storage_for_gnps_df[GNP_COL_NAME].isin(df_shortened[GNP_COL_NAME])]
    
    ## DEFINING LIGAND PREFIX
    ligand_prefix="Ligand"
    
    ## GETTING LIGANDS PRESENT
    for idx, row in df_gnp_with_ligs.iterrows():
        ## GETTING LIGAND COLS
        lig_cols = [each_key for each_key in row.keys() if each_key.startswith(ligand_prefix) ]
        
        ## GETTING SMILES
        ligand_names = row[lig_cols].to_list()
        
        ## REMOVING ALL NONES
        ligand_names = [each_name for each_name in ligand_names if each_name != None]
        
        ## GETTING LIGAND
        ligand_smiles = output_df[output_df.name.isin(ligand_names)]        
        ## LOOPING THROUGH EACH LIGAND AND GETTING STRUCTURE
        for lig_idx, each_smiles in enumerate(ligand_smiles['SMILES'].to_list()):
            print(each_smiles)
            ## PLOTTING EACH SMILES
            smiles_list = [each_smiles]
            ms = [Chem.MolFromSmiles(x) for x in (smiles_list)]
            fig=Draw.MolsToGridImage(ms,
                                     molsPerRow=1,
                                     subImgSize=(200, 200),
                                     legends=[ligand_smiles.iloc[lig_idx]['name']],
                                     )
            
            ## SHOWING FIG
            figure_name = '_'.join([row[GNP_COL_NAME], "lig_%d.png"%(lig_idx)])
            path_image = os.path.join(STORE_FIG_LOC,
                                      figure_name)
            fig.save(fp = path_image)
    
    
    