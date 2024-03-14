#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_database.py

This script loads and visualizes the database

# Uses rdkit
Installation for rdkit:
    source activate py36_mdd
    conda install -c conda-forge rdkit
    
    
FUNCTIONS:
    plot_all_ligands: 
        plots all the ligands using rdkit
    plot_size_distribution:
        plots size distribution of the database
    plot_multi_lig_dist:
        plots total number of ligands that have multiple or single ligands
    check_if_multiple_ligs:
        checks dataframe to see if there are multiple ligands

"""
import os
import numpy as np
import pandas as pd
import math
    

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
%matplotlib inline

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING EXTENSION
FIG_EXTENSION = "png"
SAVE_FIG = True

## DEFINING PATH TO DATABASE
PATH_TO_DATABASE=r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_membrane_binding_descriptors_manuscript/Exp_Database/refined_databased.xlsx"
STORE_FIG_LOC=r"/Users/alex/Box Sync/VanLehnGroup/0.Manuscripts/NP_membrane_binding_descriptors_manuscript/Figures/Output_ligands"

### FUNCTION TO PLOT ALL LIGANDS
def plot_all_ligands(database,
                     index_str = "Index",
                     smiles_str = "Ligand1 SMILES",
                     num_ligands_prefix = "#Ligand",
                     subImgSize=(200,200),
                     molsPerRow=3,
                     **kwargs
                     ):
    '''
    This function plots all the ligands
    INPUTS:
        database: [df]
            pandas dataframe with all the data
        index_str: [str]
            index that you want
        smiles_str: [str]
            smiles str within database to plot
        subImgSize: [tuple]
            size of each image
        molsPerRow: [int]
            molecules per row
    OUTPUTS:
        fig: [PIL]
            pil image file
    '''    
    ## FINDING STR
    if type(smiles_str) == str:
        print("Smiles str is a string!")
        smiles_list = database[smiles_str].to_numpy()
        ## FINDING GNP NAMES
        names = list(database[index_str].to_numpy())
    elif type(smiles_str) == list:
        print("Smiles str is a list!")
        ## LOOPING THROUGH EACH
        smiles_list = []
        names = []
        
        ## FINDING NUMBER OF LIGANDS
        num_lig_keys =[each_col for each_col in database.columns if num_ligands_prefix in each_col]
        print(num_lig_keys)
        
        ## LOOPING THROUGH DATABASE
        for idx, row in database.iterrows():
            ## GETTING DATA
            data_list = row[smiles_str].to_list()
            
            ## GETTING NUMBER OF LIGANDS
            num_ligands = row[num_lig_keys].to_list()
            
            print(data_list)
            # LOOPING THROUGH DATA LIST
            for data_idx, each_data in enumerate(data_list): # .iteritems()
                ## GETTING TOTAL NUMBER OF LIGANDS
                total_ligs = num_ligands[data_idx]
                if total_ligs == '-':
                    total_ligs = 0
                if each_data != '-' and total_ligs > 0:
                    smiles_list.append(each_data)
                    name_str = row[index_str] + "_%d"%(data_idx)
                    names.append(name_str)
            
    
    ## PLOTTING
    ms = [Chem.MolFromSmiles(x) for x in (smiles_list)]
    fig=Draw.MolsToGridImage(ms,
                             molsPerRow=molsPerRow,
                             subImgSize=subImgSize,
                             legends=names,
                             **kwargs)
    
    
    
    return fig

### FUNCTION TO GET MULTIPLE LIGANDS
def check_if_multiple_ligs(df,
                           lig_cols = ['#Ligand1','#Ligand2','#Ligand3','#Ligand4']):
   '''
   The purpose of this function is to check if a dataframe has more than 
   one ligand. The idea is to extract the total number of ligands, then 
   check if you have multiple ligands.
   INPUTS:
       df: [pd.dataframe]
           dataframe
       lig_cols: [list]
           list of columns
   OUTPUTS:
       multiple_lig_log: [list]
           list of True/False based on whether a NP has multiple ligands. If True, then 
           the ligand has multiple ligands
   '''
   
   ## COMPUTING NUMBER OF HOMOGENOUS LIGANDS
   num_lig_df = df[lig_cols].to_numpy()
   
   ## STORING LIST
   multiple_lig_log = []
   for each_list in num_lig_df:
       ## REMOVING
       no_hyphen = each_list[each_list != '-']
       ## REMOVING ZEROS
       no_zeros = no_hyphen[no_hyphen > 0]

       ## SEEING IF LEN > 1
       if len(no_zeros) > 1:
           multiple_lig_log.append(True)
       else:
           multiple_lig_log.append(False)
       
   return multiple_lig_log

### FUNCTION TO PLOT THE SIZE DISTRIBUTION
def plot_size_distribution(df,
                           column_key = "Size",
                           fig_size = FIGURE_SIZE,
                           bins = np.arange(0,11,1),
                           **kwargs):
    '''
    This function plots the size distribution
    INPUTS:
        df: [dataframe]
            pandas dataframe for sizes
        column_key: [str]
            key for the Size
        bins: [np.array]
            bins used for histogram
    OUTPUTS:
        fig, ax:
            figure and axis for image
        
    '''
    ## FINDING DATA
    size_data = df[column_key].to_numpy()

    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size)
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Size (nm)")
    ax.set_ylabel("# Occurrences")
    
    ## GENERATING HISTOGRAM
    ax.hist(x = size_data, bins = bins, align='left', **kwargs)
    
    ## SETTING X LABELS
    ax.set_xticks(bins)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    

    return fig, ax
    
### FUNCTION TO PLOT NUMBER DISTRIBUTION BETWEEN MULTIPLE LIGANDS
def plot_multi_lig_dist(df,
                        column_key = 'Multilig',
                        fig_size = FIGURE_SIZE,
                        bins = [0.,0.5, 1.],
                        xticks = ["Single", "", "Multiple"],
                        **kwargs):
    '''
    This function plots the distribution of GNPs with multiple ligands.
    INPUTS:
        df: [dataframe]
            pandas dataframe
        column_key: [str]
            key for the logicals
    OUTPUTS:
        
    '''
    ## FINDING DATA
    current_data = df[column_key].to_numpy().astype(float)

    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size)
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Ligand distribution")
    ax.set_ylabel("# Occurrences")
    
    ## GETTING HISTGRAM
    hist, bin_edges = np.histogram(current_data, bins = bins)
    
    ## PLOTTING
    ax.bar(x = [bins[0], bins[-1]],height =  hist, **kwargs)
            
    ## SETTING X LABELS
    ax.set_xticks(bins)
    ax.set_xticklabels(xticks)
    
    ## GETTING Y LIM
    y_lims = ax.get_ylim()
    
    ## ADDING 10 MORE TO YLIM
    new_ylim = math.ceil( y_lims[1] + 10 )
    
    ## SETTING LIMITS
    ax.set_ylim([0, new_ylim])
    
    ## ADDING LABELS
    rects = ax.patches
    
    # Make some labels.
    labels = [ hist[i] for i in range(len(rects))]
    
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 3, label,
                ha='center', va='bottom')
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    
    ## LOADING THE DATA
    data = pd.read_excel(PATH_TO_DATABASE,
                         sheet_name = "main")
    
    ## PLOTTING THE SMILES
    
    ## GETTING SMILES STRING
    smiles_strings = data['Ligand1 SMILES']
    
    ## FINDING UNIQUE SMILES STRINGS
    unique_smiles = np.unique(smiles_strings)
    print("Total SMILES: %d"%(len(smiles_strings)))
    print("Total unique SMILES: %d"%(len(unique_smiles)))
    
    #%%
    
    ## READING DATA FOR CELL UPTAKE 
    ## LOADING THE DATA
    uptake_data = pd.read_excel(PATH_TO_DATABASE,
                         sheet_name = "cell_uptake")
    
    #%% GETTING AL LLIGANDS
    
    ## LOADING RAW DATA
    raw_data = pd.read_excel(PATH_TO_DATABASE,
                         sheet_name = "Raw_database")
    
    ## LOADING GOLD ONLY
    gold_data_only = raw_data.loc[ np.all( (raw_data['Core']=='Gold',
                                            raw_data['Shape'] == 'Sphere',
                                            ), axis = 0 )]
    
    
    ## LOADING SPECIFIC SIZE
    gold_data_lower = gold_data_only.loc[gold_data_only['Size'] < 10]
    
    ## GETTING THOSE WITH UPTAKE DATA
    uptake_with_constraints = gold_data_lower.loc[gold_data_lower['Index'].isin(uptake_data['Index'])]
    
    ## COMPUTING NUMBER OF HOMOGENOUS LIGANDS
    num_lig_df = uptake_with_constraints[['#Ligand1','#Ligand2','#Ligand3','#Ligand4']].to_numpy()
                                          
    ## STORING LIST
    multiple_lig_log = []
    for each_list in num_lig_df:
        ## REMOVING
        no_hyphen = each_list[each_list != '-']
        ## REMOVING ZEROS
        no_zeros = no_hyphen[no_hyphen > 0]
        
        ## SEEING IF LEN > 1
        if len(no_zeros) > 1:
            multiple_lig_log.append(True)
        else:
            multiple_lig_log.append(False)
    
    ## GETTING DATAFRAME WITH UPDATED CONSTRAINTS
    uptake_single_lig = uptake_with_constraints.loc[np.logical_not(multiple_lig_log)]
    
    ## FINDING FIRST LIGAND
    uptake_single_lig = uptake_single_lig.head(1)
    
    '''
    ## PLOTTING
    fig = plot_all_ligands(database = uptake_single_lig,
                     index_str = "Index",
                     smiles_str = ['Ligand1 SMILES','Ligand2 SMILES'],
                     subImgSize=(200,200),
                     molsPerRow=1,# 4
                     )

    fig
    
    ## PATH IMAGE
    figure_name = "cell_uptake_11_gnps.png"
    path_image = os.path.join(STORE_FIG_LOC,
                              figure_name)
    
    fig.save(fp = path_image)
    '''
    
    
    
    #%% LOG P DATA ANALYSIS
    
    ## LOADING THE DATA
    logP_data = pd.read_excel(PATH_TO_DATABASE,
                         sheet_name = "logP")
    
    
    ## DEFINING DATA BASES
    databases = {
#            'celluptake': uptake_data,
            'logP': logP_data,
            }
    
    ## storage
    database_with_constraint_storage = {}
    
    for each_key in databases:
        specific_database = databases[each_key]
    
        ## GETTING THOSE WITH UPTAKE DATA
        database_with_constraint = gold_data_lower.loc[gold_data_lower['Index'].isin(specific_database['Index'])]
        
        ## SETTING IF MULTIPLE LIGANDS ARE PRESENT
        multiple_lig_log = check_if_multiple_ligs(df = database_with_constraint )
        
        ## ADDING TO DICT
        database_with_constraint = database_with_constraint.assign(Multilig = multiple_lig_log)
        
        ## STORING
        database_with_constraint_storage[each_key] = database_with_constraint
        
        ## PLOTTING LOG P DISTRIBUTION
        updated_specific_database = specific_database.loc[specific_database['Index'].isin(database_with_constraint['Index'])]
        
        #%%
        
        ## DEFINING MERGED DF
        merged_df = updated_specific_database.merge(database_with_constraint)
        
        ## DEFINING UNIQUE CORE SIZE
        unique_core_sizes =  database_with_constraint_storage['logP']['Size'].unique()
        
        ## CREATING DATAFRAME
        core_sizes_diameter_df = pd.DataFrame(unique_core_sizes, columns = ['diameter'])
        
        ## SORTING
        core_sizes_diameter_df['diameter'] = core_sizes_diameter_df['diameter'].astype(float)
        core_sizes_diameter_df = core_sizes_diameter_df.sort_values('diameter')
        
        ## OUTPUTTING TO LOCATION
        database_location = "/Volumes/akchew/scratch/nanoparticle_project/database"
        database_name = "logP_diameters.csv"
        path_database = os.path.join(database_location,
                                     database_name
                                     )
        ## PRINTING
        '''
        core_sizes_diameter_df.to_csv(path_database, float_format='%.2f', index = False)
        '''

        ## OUTPUTTING TO CSV
        merged_df_output_path = os.path.join(database_location,
                                             "logP_exp_data.csv")
        merged_df.to_csv(merged_df_output_path, index = False)

        
        
        #%%
        
        ## GETTING SMILES
        smiles_string = database_with_constraint['Ligand1 SMILES'].iloc[0]
        
        ## GETTING MOLECULE
        m = Chem.MolFromSmiles(smiles_string)
        
        ## OUTPUTING TO MOL2
        ''' Example of getting molecule
        uncharged_mol_1D = Chem.MolFromSmiles(smile)
        
        uncharged_mol_3D = Chem.AddHs(uncharged_mol_1D)
        AllChem.EmbedMolecule(uncharged_mol_3D)
        AllChem.UFFOptimizeMolecule(uncharged_mol_3D)
        
        charged_mol_3D = uncharged_mol_3D
        AllChem.ComputeGasteigerCharges(charged_mol_3D)
        
        fout = Chem.SDWriter('./charged_test.mol')
        fout.write(charged_mol_3D)
        fout.close()
        '''
        
        
        
        #%%
        
        ## GETTING SMILES
        smiles = database_with_constraint[['Ligand1 SMILES',
                                           'Ligand2 SMILES',
                                           'Ligand3 SMILES',
                                           'Ligand4 SMILES']].to_numpy()
        
        ## LOOPING TO GET TOTAL ATOMS PER LIGAND AND TOTAL SULFUR ATOMS
        total_atoms_array = np.zeros(smiles.shape)
        total_sulfur_atoms = np.zeros(smiles.shape)
        for row_idx, each_smiles in enumerate(smiles):
            for col_idx,current_smiles in enumerate(each_smiles):
                if current_smiles != '-':
                    m = Chem.MolFromSmiles(current_smiles)
                    total_atoms = m.GetNumAtoms(onlyExplicit = False)
                    ## IMPORTING TO TOTAL ATOMS
                    total_atoms_array[row_idx, col_idx] = total_atoms
                    
                    ## GETTING TOTAL SULFURS
                    num_sulfur = len([ True for a in m.GetAtoms() if a.GetAtomicNum() == 16 ])
                    total_sulfur_atoms[row_idx, col_idx] = num_sulfur
#    ## COMPARING DATABASE
#    overlap = database_with_constraint_storage['logP'].loc[database_with_constraint_storage['logP']['Index'].isin(database_with_constraint_storage['celluptake']['Index'])]
        
        #%%
        
        
#        ## CREATING FIGURE
#        fig, ax = plot_size_distribution(df = database_with_constraint,
#                                         column_key = "Size",
#                                         color = 'k',)
#        
#        ## STORING FIGURE    
#        figure_name = '_'.join([each_key, "size_distribution"])
#        
#        ## STORING FIGURE
#        plot_funcs.store_figure( fig = fig,
#                                 path = os.path.join(STORE_FIG_LOC,
#                                                     figure_name),
#                                 fig_extension = FIG_EXTENSION,
#                                 save_fig = SAVE_FIG,
#                                 )
#    
#        
#        ## PLOTTING MULTI LIG
#        fig, ax = plot_multi_lig_dist(df = database_with_constraint,
#                                      color = 'k',
#                                      )
#        
#        ## STORING FIGURE    
#        figure_name = '_'.join([each_key, "lig_distribution"])
#    
#        ## STORING FIGURE
#        plot_funcs.store_figure( fig = fig,
#                                 path = os.path.join(STORE_FIG_LOC,
#                                                     figure_name),
#                                 fig_extension = FIG_EXTENSION,
#                                 save_fig = SAVE_FIG,
#                                 )
#                                 
        ## PLOTTING
        fig = plot_all_ligands(database = database_with_constraint,
                         index_str = "Index",
                         smiles_str = ['Ligand1 SMILES','Ligand2 SMILES'],
                         subImgSize=(200,200),
                         molsPerRow=16,# 4
                         maxMols = 5000,
                         )
    
        fig
        
        ## PATH IMAGE
        figure_name = '_'.join([each_key, "lig_snapshots.png"])
        path_image = os.path.join(STORE_FIG_LOC,
                                  figure_name)
        fig.save(fp = path_image)
    
    
    
    #%%
    
    ## PLOTTING
    fig = plot_all_ligands(database = uptake_with_constraints,
                     index_str = "Index",
                     smiles_str = ['Ligand1 SMILES','Ligand2 SMILES'],
                     subImgSize=(200,200),
                     molsPerRow=12
                     )
    ## PATH IMAGE
    figure_name = "cell_uptake_all_ligs.png"
    path_image = os.path.join(STORE_FIG_LOC,
                              figure_name)
    
    fig.save(fp = path_image)
    
                                          
                                          
                                          
                                          
                                          
                                             
                                             
                                             
    
    
    
    #%%
    
    ## REMOVING LIGANDS WITH S1S?
    fig = plot_all_ligands(database = uptake_with_constraints,
                         index_str = "Index",
                         smiles_str = "Ligand1 SMILES",
                         subImgSize=(200,200),
                         molsPerRow=10,
                         )
    
    fig
    
    
    ## SPHERE ONLY
    
    
    
    ## GETTING ALL LIGANDS REGARDLESS OF DOUBLE LIGAND
    
    
    
    
    
    
    
    #%%
    
    ## S1S SMILES STRING SEARCH
    current_smiles = data['Ligand1 SMILES'].to_list()
    
    ## GETTING LIGADS WITHOUT S1S
    ligs_without_double_sulfur = [ each_smiles for each_smiles in current_smiles if 'S1S' not in each_smiles]
    
    
    
    
    #%%
    
    
    
    ms = [Chem.MolFromSmiles(x) for x in (unique_smiles[0:10])]
    img=Draw.MolsToGridImage(ms[:8],molsPerRow=3,subImgSize=(200,200),legends=['hello' for x in ms[:8]]) # x.GetProp("_Name") 
    print(img)
    img
    
#    figs = Draw.MolsToGridImage(ms)
    
    #%%
#    fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, 
#                         # figsize = plot_funcs.cm2inch( *fig_size )
#                         )
    
    # fig, axs = plt.subplots()
    fig = plt.figure()
    
    ## DEFINING SIZE
    size = (120, 120)
    
    ## DEFINING INDEX
    index = 0
    
    ## LOOPING
    for index in range(len(unique_smiles)):
    
        ## TRYING ONE
        smiles_each = unique_smiles[index]
    
        ## DRAWING
        draw_smiles = Chem.MolFromSmiles(smiles_each)
    
        ## MAKING FIGURE
        fig_mpl = Draw.MolToMPL(draw_smiles, size=size) # , ax = axs[0]
        
        ## SAVING FIGURE
        figure_name = "%s"%(index)
        plot_funcs.store_figure( fig = fig_mpl,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 fig_extension = 'png',
                                 save_fig = True,
                                 )
        
    #%%
    ## GETTING AXIS
    ax = fig_mpl.get_axes()
    
    fig.axes.append(ax[0])
     
    
    #%%
    
