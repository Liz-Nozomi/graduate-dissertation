#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_self_assembly_sims.py

This script analyzes self-assembly simulations from ligands onto gold cores. 

Written by: Alex K. Chew (08/07/2020)

"""
import os
import glob
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import pandas as pd

## IMPORTING
from MDDescriptors.application.nanoparticle.self_assembly_structure import \
    self_assembly_structure, GOLD_RESIDUE_NAME, ATTACHED_SULFUR_ATOM_NAME, GOLD_SULFUR_CUTOFF

from MDDescriptors.application.nanoparticle.extract_self_assembly_structure import \
    plot_self_assembly_structure

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## DEFINING FIGURE SIZE
FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING LOCATION TO STORE IMAGE
STORE_FIG_LOC = r"/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200817/images"

### FUNCTION TO COUNT THE TWO SULFURS
def count_two_sulfurs(traj_data,
                      structure):
    '''
    This function counts the number of double-suflurs there are 
    for a given residue. 
    INPUTS:
        traj_data: [obj]
            trajectory object
        structure: [obj]
            object using for self-assembly structure
    OUTPUTS:
        num_sulfurs_multiple_sulfurs: [np.array, shape = (num_frames)]
            number of ligands with two sulfurs 
    '''

    ## STORAGE ARRAY
    num_sulfurs_multiple_sulfurs = []
    ## LOOPING
    for each_row in structure.num_gold_sulfur_bonding_indices:
        sulfurs_indices_attached = structure.sulfur_index[np.where(each_row)[0]]
        
        ## CONVERTING TO RESINDEX
        res_index = np.array( [ traj_data.topology.atom(each_index).residue.index for each_index in sulfurs_indices_attached ] )
        
        ## COUNTING NUMBER OF REPEATS
        unique, counts = np.unique(res_index, return_counts = True)
        # counted = np.bincount(res_index)
        
        ## COMPUTING NUMBER OF MORE THAN 2 SULFURS
        num_more_than_2 = (counts == 2).sum()
        num_sulfurs_multiple_sulfurs.append(num_more_than_2)
    
    num_sulfurs_multiple_sulfurs = np.array(num_sulfurs_multiple_sulfurs)
    
    return num_sulfurs_multiple_sulfurs

### fUNCTION TO EXTRACT DIAMETER
def extract_self_assembly_name(name = 'spherical_6.50_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_bidente_Trial_1'):
    '''
    This function extracts the self-assembly name.
    INPUTS:
        name: [str]
            name of the self-assembly simulations
    OUTPUTS:
        output_dict: [dict]
            dictionary for self-assembly simulations
    '''
    ## SPLITTING
    name_split=name.split('_')
    
    ## CREATING DICT
    output_dict = {
            'diameter': float(name_split[1]),
            'ligand': name_split[-3],            
            }
    
    return output_dict

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    #%% LOADING DIAMETERS
    
    ## DEFINING PATH
    path_to_sim = r"/Volumes/akchew/scratch/nanoparticle_project/prep_system/prep_gold/self_assembly_simulations/spherical/"
    path_to_sim = r"/Volumes/akchew/scratch/nanoparticle_project/prep_system/prep_gold/self_assembly_simulations_nonfrag/spherical"
    
    ## SORTING
    sim_list = glob.glob(path_to_sim + "/*"
                         )
    sim_list.sort()

    ## CREATING STORAGE
    storage_list = []
    
    
    ## LOOPING
    for sim_path in sim_list:
        ## DEFINING SIM NAME
        sim_name = os.path.basename(sim_path)
        
        ## EXTRACTING DICT NAME
        output_sim_dict = extract_self_assembly_name( name = sim_name )
        
        if output_sim_dict['ligand'] == 'bidente':
        
            ## DEFINING GRO AND XTC
            gro_file="gold_ligand_equil.gro"
            xtc_file="gold_ligand_equil.gro"
            # "gold_ligand_equil.xtc"
            
            ## PATH TO SIM
            full_path_to_sim = os.path.join(path_to_sim,
                                            sim_name)
            
            ### LOADING TRAJECTORY
            traj_data = import_tools.import_traj( directory = full_path_to_sim, # Directory to analysis
                                                  structure_file = gro_file, # structure file
                                                  xtc_file = xtc_file, # trajectories
                                                  )
            
            ### DEFINING INPUT DATA
            input_details={ 'gold_residue_name' : GOLD_RESIDUE_NAME,         # Gold residue name
                            'sulfur_atom_name'  : None, # Sulfur atom name
                            'gold_sulfur_cutoff': GOLD_SULFUR_CUTOFF,        # Gold sulfur cutoff for bonding
                            'gold_shape'        : "spherical",             # shape of the gold
                            'coord_num_surface_to_bulk_cutoff': 11,         # Cutoff between surface and bulk
                            'coord_num_facet_vs_edge_cutoff':   7,          # Cutoff between facet and edge atoms
                            'ps_per_frame'      : 50,                        # Total picoseconds per frame
                            'split_traj'        : 200,                        # Total frames to run calculation every time
                            'gold_optimize'     : True,                     # True if you want gold optimization procedure
                            }
            
            ### FINDING SELF ASSEMBLY STRUCTURE
            structure = self_assembly_structure(traj_data, **input_details)
            
            ## GETTING NUMBER OF LIGANDS WITH TWO SULFURS
            num_two_sulfurs = count_two_sulfurs(traj_data = traj_data,
                                                structure = structure)
            
            ## GETTING DIAMETER
            storage_output = {
                    **output_sim_dict,
                    'Num_adsorbed_2S': num_two_sulfurs
                    }
            
            ## APPENDING
            storage_list.append(storage_output)
    #%%
    
    ## CREATING DATAFRAME
    sim_df = pd.DataFrame(storage_list)
    
    ## CONVERTING TYPE
    sim_df['Num_adsorbed_2S'] = sim_df['Num_adsorbed_2S'].astype(int)
            
    
    #%% LOADING DATABASE
    ## LOADING DATABASE INFORMATION
    parent_database = "/Volumes/akchew/scratch/nanoparticle_project/database"
    database_name = "logP_exp_data.csv"
    path_to_database = os.path.join(parent_database,
                                    database_name)
    ## LOADING
    database_df = pd.read_csv(path_to_database)
    #%%
    
    ## CREATING FIGURE
    fig, ax = plot_funcs.create_fig_based_on_cm(FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Adsorbed ligands")
    
        
    ## PLOTTING
    ax.plot(sim_df['diameter'], 
            sim_df['Num_adsorbed_2S'],
            color = 'k',
            label = "Comp. data")

    ## GETTING CORE SIZE
    exp_sizes = database_df['Size'].to_numpy()
    
    ## GETTING SULFURS
    comp_sulfurs_storage = []
    
    ## GETTING DATAFRAME NUM LIGS
    for idx, each_row in enumerate(exp_sizes):
        print(each_row)
        ## GETTING COMP
        row_with_diameter = sim_df.iloc[ np.where(sim_df['diameter'] == each_row)[0] ]
        ## GETTING NUM SUFLURS
        comp_sulfurs = row_with_diameter['Num_adsorbed_2S'].iloc[0]
        ## APPENDING
        comp_sulfurs_storage.append(comp_sulfurs)
        
    ## STORING IN DATAFRAME
    database_df['Comp_adsorbed'] = comp_sulfurs_storage
    
    ## GETTING LIST
    columns_with_numbers = [each_col for each_col in database_df.columns if each_col.startswith("#")] 
    ## GETTING TOTAL
    df_with_nums = database_df[columns_with_numbers]
    ## REPLACING
    df_with_nums = df_with_nums.replace('-', np.nan)
    
    ## GETTING TOTAL LIGANDS
    exp_total_ligands = df_with_nums.astype(float).sum(axis=1).to_numpy()
    
    ## GETTING NUMBER OF LIGANDS GREATER
    num_ligs_comp_greater = (comp_sulfurs_storage > exp_total_ligands).sum()
    
    ## GETTING 
    planar_sizes = 4.62 / 4*np.pi(exp_sizes/2.0)**2
    
    
    
    ## ADDING TITLE
    ax.set_title("Num comp > exp: %d of %d"%(num_ligs_comp_greater, len(comp_sulfurs_storage)))
    
    ## PLOTTING
    ax.plot(exp_sizes, exp_total_ligands, 
            marker = '.', 
            linestyle = "None", 
            color = 'red',
            label = "Exp. data")
    
    ## legend
    ax.legend()
                     
    #%%
    
    ## DEFINING FIGURE NAME
    figure_name = "Comp_vs_exp_assembly_logP"
    ## SAVING FIGURE
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(STORE_FIG_LOC,
                                                 figure_name),
                             fig_extension = "png",
                             save_fig = True,
                             )            
    
    
    
    
    
    #%%    
            

        
        '''
        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(FIGURE_SIZE)
        
        ## ADDING LABELS
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Adsorbed ligands")
        
        
        ## PLOTTING
        ax.plot(structure.frames_ns, 
                structure.num_gold_sulfur_bonding_per_frame,
                color = 'k',
                label = "total sulfurs")
        
        ## PLOTTING MULTIPLE
        ax.plot(structure.frames_ns,
                num_sulfurs_multiple_sulfurs,
                color = 'b',
                label = "2 sulfurs"
                )
        
        ## ADDING LEGEND
        ax.legend()
        
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        ## DEFINING FIGURE NAME
        figure_name = sim_name + "_lig_adsorbed"
        ## SAVING FIGURE
        plot_funcs.store_figure( fig = fig,
                                 path = os.path.join(STORE_FIG_LOC,
                                                     figure_name),
                                 fig_extension = "png",
                                 save_fig = True,
                                 )
    
        '''
        
        #%%
        

    
    #%%
    
    ### PLOTTING
#    plot_structure = plot_self_assembly_structure(structure)