#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_lm_rearrangement.py

The purpose of this function is to compute the lipid membrane re-arrangement

Written by: Alex K. Chew

The algorithm will be to look for first the z-dimension locations for the 
upper and lower leaflets. Then, look for all tail groups that are outside 
the range. If they are outside, we designate them as lipids that have 
rearranged. 
"""

## IMPORTING TOOLS
import os
import numpy as np
import MDDescriptors.core.import_tools as import_tools
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types

## FUNCTIONS
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups
from MDDescriptors.application.np_lipid_bilayer.compute_np_intercalation import analyze_lm_groups

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv

## IMPORTING LAST FRAME TOOL
from MDDescriptors.core.traj_itertools import get_traj_from_last_frame

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    sim_type_list = [
            'unbiased_ROT012_1.300'
            ]
    
    ## GRO AND FILE
    input_prefix="nplm_prod-center_pbc_mol"
    gro_file = input_prefix + ".gro"
    xtc_file = input_prefix + ".xtc"
    
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## PATH TO SIMULATION
        path_to_sim=os.path.join(PARENT_SIM_PATH,
                              main_sim_dir,
                              specific_sim)
        
        ## LOADING THE TRAJECTORY
        traj_data = import_tools.import_traj(directory = path_to_sim,
                                             structure_file = gro_file,
                                             xtc_file = xtc_file,
                                             )
        
        #%%
        lm_res_name = "DOPC"
        ## GETTING LM GROUPS
        lm_details = analyze_lm_groups(traj = traj_data.traj,
                                       lm_res_name = lm_res_name)
    
        ## FINDING COM
        mean_z_top, mean_z_bot, center_of_mass = lm_details.find_top_and_bottom_leaflet_positions(traj = traj_data.traj)
        
        ## FINDING ALL INDEXES FOR LM TAIL GROUPS
        lm_tail_group_atom_index = lm_details.lm_groups_atom_index['TAILGRPS']
        
        
        traj = traj_data.traj
        
        ### FUNCTION TO GET ALL ATOM INDICES OUTSIDE
        def find_lm_outside_range(traj,
                                  lm_tail_group_atom_index,
                                  mean_z_top,
                                  mean_z_bot
                                  ):
            '''
            This function finds lipid membranes outside of the range.
            INPUTS:
                traj: 
                    trajectory object
                lm_tail_group_atom_index: [np.array]
                    lipid membrane tail group atom index
                mean_z_top: [float]
                    mean z top
                mean_z_bot: [float]
                    mean z bottom
            OUTPUTS:
                atom_index_out: [np.array]
                    atom indices that are outside the range
            '''
        
            ## FINDING ALL Z POSITIONS
            lm_tail_z_pos = traj.xyz[:,lm_tail_group_atom_index,-1]
            
            ## FINDING ALL Z POSITIONS WITHIN THE TOP AND BOTTOM
            log_outside = np.logical_or( lm_tail_z_pos > mean_z_top )
                                        # , lm_tail_z_pos < mean_z_bot)
            
            ## GETTING ATOM INDEX
            atom_index_out = [lm_tail_group_atom_index[each_frame_array] for each_frame_array in log_outside]
            
            return atom_index_out
        
        ### FUNCTION TO GET THE RES INDEX
        def find_res_index_for_lm(atom_index_out,
                                  traj,):
            '''
            This function finds the unique residue index for the lipid membrane
            INPUTS:
                atom_index_out: [np.array]
                    atom indices as a list for each frame
                traj: [obj]
                    trajectory object
            OUTPUTS:
                resindex_list: [np.array]
                    array for residue index
            '''
        
            ## CREATING RES INDEX LIST
            resindex_list = []
            
            ## FINDING ALL UNIQUE RESIDUES
            for each_array in atom_index_out:
                ## GETTING ALL RESIDUE INDEX
                res_index = np.unique([ traj.topology.atom(each_atom).residue.index for each_atom in each_array ])
                
                ## PPAENDING
                resindex_list.append(res_index)
                
            ## CREATING ARRAY
            resindex_list = np.array(resindex_list)
            
            return resindex_list
        
        ## GETTING LOGICALS
        atom_index_out = find_lm_outside_range(traj = traj,
                                  lm_tail_group_atom_index = lm_tail_group_atom_index,
                                  mean_z_top = mean_z_top,
                                  mean_z_bot = mean_z_bot,
                                  )
        
        ## DEFINING THE CUTOFF
        cutoff_radius = 0.5 # 0.5 nm cutoff
        
        ## GETTING GROUPS
        
        
        
        
        
        ## GETTING RES INDEX LIST
        resindex_list = find_res_index_for_lm(atom_index_out = atom_index_out,
                                              traj = traj,)

        ## GETTING NUMBER PER FRAME
        num_residue_within = np.array([ len(each_list) for each_list in resindex_list])
        
        
        
        
        
        