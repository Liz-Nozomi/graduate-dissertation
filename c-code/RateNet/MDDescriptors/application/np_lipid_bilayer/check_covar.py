#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_covar.py

The purpose of this code is to check the covar and make sure that the coordination 
is computing the correct values. We can do this by the following algorithm:
    - load gro / xtc file
    - compute contacts per frame
    - Split contacts into groups, e.g.
        - Group 1: nanoparticle ligands
        - Group 2: lipid membrane tail groups
    - Run analysis based on the separated groups
    - Compute the PLUMED coordination value, as stated online:
        https://www.plumed.org/doc-v2.5/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
        
    - Lastly, see if the COVAR matches correctly
    
Written by: Alex K. Chew (05/06/2020)

KbT: 2.494339
"""
## IMPORTING OS
import os
import numpy as np
import mdtraj as md
## IMPORTING FUNCTION
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import main_compute_contacts, compute_NPLM_contacts

## PATH TO PARENT SIM
from MDDescriptors.application.np_lipid_bilayer.global_vars import PARENT_SIM_PATH, IMAGE_LOC

## DEFINING PICKLING TOOLS
from MDDescriptors.core.pickle_tools import load_pickle_results, save_and_load_pickle

## IMPORTING PLOT TOOLS
import matplotlib.pyplot as plt
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

### FUNCTION TO COMPUTE SIJ
def compute_coord_from_plumed(distance,
                              d0 = 0,
                              n = 6,
                              m = 0,
                              r0 = 0.5):
    '''
    Tnhis function computes the coordination number from plumed.
    INPUTS:
        distance: [np.array]
            distance array
        d0: [float]
            d0 switching parameters
        n: [int]
            exponent on numerator
        m: [int]
            exponent in denominator. By default, if m = 0, then this is 2 * n
        r0: [float]
            radius of cutoff for the coordination number
    OUTPUTS:
        sij: [np.array]
            sij numpy array matrix
        
    '''
    if m == 0:
        m = 2 * n
    ## GETTING SIJ
    sij = ( 1 - ( (distance - d0) / r0 ) ** n ) / ( 1 - ( (distance - d0) / r0  )**m )
    return sij

#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    
    ## DEFINING PARENT SIM
    parent_dir="20200430-debugging_nplm_plumed_ROT012_neighbor_list"
    specific_sim = "NPLMplumedcontactspulling-5.100_2_25_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    parent_dir = r"20200430-debugging_nplm_plumed_ROT012_neighbor_list_pt2"
    specific_sim = r"NPLMplumedcontactspulling-5.100_2_25_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    parent_dir = r"20200505-debugging_nplm_plumed_ROT001_neighbor_list_pt3"
    specific_sim = r"NPLMplumedcontactspulling-5.100_2_50_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"

    parent_dir = r"20200505-full_pulling_plumed"
    specific_sim = r"NPLMplumedcontactspulling-5.100_2_50_1000_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"   
    
#    parent_dir = r"20200505-debugging_nplm_plumed_ROT001_neighbor_list_pt3"
#    specific_sim = r"NPLMplumedcontactspulling-5.100_2_50_500_0.5-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"  

    ## DEFINING SIMULATION PATH
    path_to_sim = os.path.join(PARENT_SIM_PATH,
                               parent_dir,
                               specific_sim)
    
#    #%% COMPUTING NUMBER OF CONTACTS
#    
#    ## DEFINING INPUTS
#    contacts_input = {
#            'cutoff': 0.5,
#            'lm_res_name': 'DOPC',
#            }
#    
#    ## DEFINING FUNCTION INPUT
#    func_inputs = {'path_to_sim': path_to_sim,
#                   'input_prefix': 'nplm_pulling',
#                   'func_inputs': contacts_input,
#                   'last_time_ps': 50000,
#                   'selection': 'non-Water',
#                   'gro_output_time_ps': 0,
#                   'n_procs': 28,
#                   'rewrite': False,
#                   }
#    
#    ## defining pickle name
#    pickle_name = func_inputs['input_prefix'] + "_contacts.pickle"
#    pickle_path = os.path.join(path_to_sim, pickle_name)
#    
#    ## EXTRACTION PROTOCOL WITH SAVING
#    contacts_obj, results = save_and_load_pickle(function = main_compute_contacts, 
#                                                 inputs = func_inputs, 
#                                                 pickle_path = pickle_path,
#                                                 rewrite = False,
#                                                 verbose = True)
#    
    #%% COMPUTING DISTANCES BETWEEN ALL POSSIBLE PAIRS
    
    ## DEFINING INPUTS
    input_prefix = "nplm_pulling"
    selection = "non-Water"
    rewrite=False
    gro_output_time_ps = 0
    
    ## IMPORTING COMMANDS 
    from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv
    import MDDescriptors.core.import_tools as import_tools
    
#    ## CONVERTING TRAJECTORY
#    trjconv_func = convert_with_trjconv(wd = path_to_sim)
#    ## GETTING ONLY SPECIFIC SELECTION
#    gro_file, xtc_file, ndx_file = trjconv_func.generate_gro_xtc_specific_selection(input_prefix = input_prefix,
#                                                                                    selection = selection,
#                                                                                    rewrite = rewrite,
#                                                                                    gro_output_time_ps = gro_output_time_ps)
#    
#    ## LOADING FILES
##    traj_data = import_tools.import_traj(directory = path_to_sim,
##                                         structure_file = gro_file,
##                                         xtc_file = xtc_file,
##                                         )
#    
    ## LOADING FILES
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = "nplm_em.gro",
                                         xtc_file = "nplm_pulling.xtc",
                                         )
    
    #%% CREATING GROUPS
    
    ## DEFINING TRAJECTORY
    traj = traj_data.traj
    
    ## DEFINING LIPID MEMBRANE NAME
    lipid_membrane_resname = "DOPC"
    
#    ## GETTING CONTACTS
#    contacts = compute_NPLM_contacts(traj = traj,
#                                     cutoff = 0.5,
#                                     lm_res_name = lipid_membrane_resname,
#                                     )

    
    #%% COMPUTING GROUPS
    
    ## CALC TOOLS
    import MDDescriptors.core.calc_tools as calc_tools
    
    from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups
    from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import get_nplm_heavy_atom_details
    
    ## GETTING INFORMATION FOR LIGAND DETAILS
    ligand_names, \
    np_heavy_atom_index, \
    np_heavy_atom_names, \
    lm_heavy_atom_index, \
    lm_heavy_atom_names = get_nplm_heavy_atom_details(traj = traj,
                                                           lm_res_name = lipid_membrane_resname)
    ## GETTING GROUPS
    np_groups = generate_rotello_np_groups(traj = traj,
                                           atom_names_np =  np_heavy_atom_names,
                                           np_heavy_atom_index = np_heavy_atom_index )
    
    ## GETTING LM GROUPS
    lm_groups = generate_lm_groups(traj = traj,
                                   atom_names = lm_heavy_atom_names,
                                   lm_heavy_atom_index = lm_heavy_atom_index,
                                   verbose = True,
                                   )
    
    ## DEFINING SPECIFIC GROUPS
    np_groups_key = [each_key for each_key in np_groups.keys() if each_key != "GOLD"]
    lm_groups_key = ['TAILGRPS']
    
    
    ## GETTING ALL ATOM NAMES
    np_atom_names = calc_tools.flatten_list_of_list([np_groups[each_key] for each_key in np_groups_key])
    lm_atom_names = calc_tools.flatten_list_of_list([lm_groups[each_key] for each_key in lm_groups_key])
    
    ## GETTING ALL ATOM INDICES THAT MATCH
    np_atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                       residue_name = ligand_names[0], # assuming one
                                                       atom_names = np_atom_names)[1]
    lm_atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                       residue_name = lipid_membrane_resname, # assuming one
                                                       atom_names = lm_atom_names)[1]
    
    ## FLATTENING
    np_atom_index = np.array(calc_tools.flatten_list_of_list(np_atom_index))
    lm_atom_index = np.array(calc_tools.flatten_list_of_list(lm_atom_index))

    #%%

    
    
    ### FUNCTION TO COMPUTE DISTANCES
    def compute_nplm_distances(traj,
                               atom_pairs,
                               np_atom_index,
                               lm_atom_index):
        '''
        This function computes the distances for a given set of atom pairs
        INPUTS:
            traj: [md.traj]
                trajectory ojbect
            atom_pairs: [list]
                list of atom pairs
            
        '''
    
        ## TOTAL FRAME
        total_frames = len(traj)
        
        ## COMPUTING DISTANCES
        distances = md.compute_distances(
                                        traj = traj,
                                        atom_pairs = atom_pairs,
                                        periodic = False
                ) ## RETURNS TIMEFRAME X (NUM_ATOM_1 X NUM_GOLD) NUMPY ARRAY
        
        ## RESHAPING THE DISTANCES
        distances = distances.reshape(total_frames, 
                                      len(np_atom_index), 
                                      len(lm_atom_index))
        
        return distances
    

    ## GENERATING ATOM PAIRS
    atom_pairs = calc_tools.create_atom_pairs_list(atom_1_index_list = np_atom_index, 
                                                   atom_2_index_list = lm_atom_index)
    
    ## COMPUTING DISTANCES
    distances_nplm = compute_nplm_distances(traj = traj[0:110:10], # [0]
                                       atom_pairs = atom_pairs,
                                       np_atom_index = np_atom_index,
                                       lm_atom_index = lm_atom_index)
    
    #%%
    
#    ## TESTING ATOM PAIRS
#    atom_pairs_test = calc_tools.create_atom_pairs_list(atom_1_index_list = [0,1], 
#                                                   atom_2_index_list = [2,3])
#    
#    
#    ### FUNCTION TO COMPUTE DISTANCES
#    def compute_nplm_distances(traj,
#                atom_pairs,
#                np_atom_index,
#                lm_atom_index,
#                periodic = True):
#        '''
#        This function computes the distances for a given set of atom pairs
#        INPUTS:
#            traj: [md.traj]
#                trajectory ojbect
#            frames: [list]
#                list of frames to run
#            periodic: [logical]
#                True if you want to account for PBC
#        OUTPUTS:
#            distances: [np.array, shape = (T, N, M)]
#                Distances array with the shape of num frames x num nanoparticle atoms x num lipid membrane atoms
#        '''
#        ## TOTAL FRAME
#        total_frames = len(traj)
#        
#        ## COMPUTING DISTANCES
#        distances = md.compute_distances(
#                                        traj = traj,
#                                        atom_pairs = atom_pairs,
#                                        periodic = True,
#                ) ## RETURNS TIMEFRAME X (NUM_ATOM_1 X NUM_GOLD) NUMPY ARRAY
#        
#        ## RESHAPING THE DISTANCES
#        distances = distances.reshape(total_frames, 
#                                      len(np_atom_index), 
#                                      len(lm_atom_index))
#        return distances
#    
#    ## COMPUT DISTANCE
#    distance_test = compute_nplm_distances(traj[0:110:10],
#                                   atom_pairs = atom_pairs_test,
#                                   # [[0,2],
##                                                 [1,2],
##                                                 [0,3],
##                                                 [1,3]],
#                                   np_atom_index=[0,1],
#                                   lm_atom_index=[2,3])
#    
#    ## ROUNDING
#    # distance_test = np.around(distance_test, decimals = 12)
#   fram
    frame = 0
    d0 = 0
    n = 6
    m = 2*n
    #2*n
    # 2*n
    r0 = 0.5
#    
#    sij = compute_coord_from_plumed(distance = distance_test,
#                                  d0 = d0,
#                                  n = n,
#                                  m = m,
#                                  r0 = r0)
    dmax = d0+r0* 0.00001**(1/(n-m))
    
    current_distance = distances_nplm[frame][distances_nplm[frame]<dmax]
    # current_distance = distances_nplm[0]
    sij = compute_coord_from_plumed(distance = current_distance,# distances_nplm,
                                  d0 = d0,
                                  n = n,
                                  m = m,
                                  r0 = r0)
    
    print(sij.sum())
    
    
    #%%
#    # sij = np.around(sij, decimals = 12)
#    print(np.sum(sij, axis = 1).sum(axis=1)) # [0:110:10]
    
    #%% COMPUTING 
    

#    distances_nplm = np.around(distances_nplm, decimals = 12)
    
    
    ## DEFINING FAMRE
    frame = 0
    
    ## COMPUTING SIJ FROM PLUMED
    
    
    ## COMPUTING SUM OF SIJ FOR EACH FRAME
    def compute_sij_for_each_frame(distances,
                                   sij_values_greater_than = np.exp(1)*10**(-5),
                                   d0 = 0,
                                   n = 6,
                                   m = 2*n,
                                   r0 = 0.5):
        '''
        This function computes sij for each frame.
        INPUTS:
            distances: [np.array]
                distances array
            sij_values_greater_than: [float]
                values of sij that you care about. For PLUMED, some data is 
                omitted to get a lower overall sij value. It has something 
                to do with the neighbors list, although I am not sure to what 
                extent the neighbors are removed. Set this value to 0 if you don't 
                want any removals
            d0, n, m, r0 
        OUTPUTS:
            
        
        '''
        ## DEFINING FRAMES
        num_frames = len(distances)
        
        ## DEFINING DICT
        output_dict = {
                'sij': [],
                'rij': [],
                'sum_sij' : []
                
                }
        ## LOOPING
        for each_frame in range(num_frames):
            sij = compute_coord_from_plumed(distance = distances[each_frame],# distances_nplm,
                                  d0 = d0,
                                  n = n,
                                  m = m,                                  
                                  r0 = r0)
            ## GETTING SPECIFIC SIJ VALUES
            idx = sij > sij_values_greater_than
            sij_trunc = sij[idx]
            rij_trunc = distances[each_frame][idx]
            
            ## GETTING SUM
            sum_sij = np.sum(sij_trunc)
            
            ## APPENDING
            output_dict['sij'].append(sij_trunc)
            output_dict['rij'].append(rij_trunc)
            output_dict['sum_sij'].append(sum_sij)
            
        return output_dict
    
    
    ## LOOPING AND GETTING SIJ
    output_dict = compute_sij_for_each_frame(distances = distances_nplm)
    
    #%%
    
            
    
    #%%
    import sys
    epsilon = sys.float_info.epsilon
    #%%
    print(sij[sij>np.exp(1)*10**(-5)].sum())
    # 2.71828e-5
    
    #%%
    
    ## PRINTING
    print(np.sum(sij, axis = 1).sum(axis=1)) # [0:110:10]
    
    
    #%% PLOTTINGplt.

    ## DEFINING FIGURE SIZE
    FIGURE_SIZE = plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
    SAVE_FIG = False
    # False
    ## PLOTTING SIJ VS DISTANCE
    def plot_sij_vs_distance(sij, distance, r0=0.5):
        ''' Plots sij vs. distance '''
        ## CREATING FIGURE
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=FIGURE_SIZE)
        
        ## ADDING LABELS
        ax.set_xlabel("$r_{ij}$ (nm)")
        ax.set_ylabel("$s_{ij}$")
        
        ## DEFINING X AND Y
        x = distance
        y = sij
        
        ## PLOTTING
        ax.plot(x, y,linestyle = "None", marker = '.',  color = 'k')
        
        ## DRAWING LINE
        ax.axvline(x=r0, color='b', linestyle='--')
        
        ## GETTING TIGHT LAYOUT
        fig.tight_layout()
        return fig, ax
    ## PLOTTING
    fig, ax = plot_sij_vs_distance(sij = output_dict['sij'][0], 
                                   distance = output_dict['rij'][0], 
                                   r0=0.5)
    #%%
    figure_name = specific_sim+ 'example_rij'
    
    plot_funcs.store_figure( fig = fig,
                             path = os.path.join(IMAGE_LOC,
                                                 figure_name),
                             save_fig = True,
                             )
    
    