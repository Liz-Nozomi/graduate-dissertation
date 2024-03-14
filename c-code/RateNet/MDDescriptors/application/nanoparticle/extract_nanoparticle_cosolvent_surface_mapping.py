#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_nanoparticle_cosolvent_surface_mapping.py
The purpose of this code is to extract cosolvent surface mapping. 

CREATED ON: 11/2/2019

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""
## IMPORT
import os
import pandas as pd
import mdtraj as md
import numpy as np


## CHECKING PATH
from MDDescriptors.core.check_tools import check_path, check_testing

## IMPORTING COSOLVENT MAPPING
from MDDescriptors.application.nanoparticle.stored_parallel_scripts import compute_cosolvent_mapping
from MDDescriptors.application.nanoparticle.nanoparticle_cosolvent_surface_mapping import cosolvent_mapping_main

## PICKLE TOOLS
import MDDescriptors.core.pickle_tools as pickle_tools
    

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    
    if testing == True:
    
        ## MAIN SIM PATH
        main_sim_path=r"R:\scratch\nanoparticle_project\simulations\191017-mixed_sams_most_likely_sims"
        
        ## DEFINING PATH TO ANALYSIS
        ## DEFINING SIM TYPE
        ## DEFINING FRAMES
        frame=["9500","9600","9700","9800","9900","10000"]
        
        ## GETTING LIST OF DIRECTORIES
        ligand_name="C11COO"
        # "dodecanethiol"
        # "C11COOH"
        
        ## PREFIX
        sim_prefix=r"MostNP-EAM_300.00_K_2_nmDIAM_"
        sim_middle=r"_CHARMM36jul2017_Trial_1-lidx_1-cosf_"
        sim_suffix=r"-aceticacid_formate_methylammonium_propane_1_molfrac_300"
        
        ## GETTING FULL LIST
        sim_name_list = [ sim_prefix + ligand_name + sim_middle + each_frame +  sim_suffix for each_frame in frame]
        ## GETTING FULL PATH
        path_analysis_list = [ check_path(os.path.join(main_sim_path, each_sim)) for each_sim in sim_name_list]
        
#        sim_name= r"MostNP-EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1-lidx_1-cosf_10000-aceticacid_formate_methylammonium_propane_1_molfrac_300"
#        sim_name= r"MostNP-EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1-lidx_1-cosf_10000-aceticacid_formate_methylammonium_propane_1_molfrac_300"
#        path_analysis=os.path.join(,
#                                   sim_name)
        ## CHECKING PATH
        # path_analysis = check_path(path_analysis)
        
        ## GRO AND XTC
        gro_file="sam_prod.gro"
        xtc_file="sam_prod.xtc"

        
        ## DEFINING PICKLE TYPE
        pickle_type="new"
        # "old"
        
        ## DEFINING PDB FILE
        grid_pdb_filename="wc_aligned.pdb"
        
        if pickle_type == "old":
            ## DEFINING OUTPUT PICKLING PATH
            pickle_name = "cosolvent_map.pickle"
            # path_pickle = os.path.join(path_analysis, pickle_name)
            path_pickle_list = [os.path.join(each_path, pickle_name) for each_path in path_analysis_list]
            ## DEFINING PATH
            # path_pdb = os.path.join(path_analysis, grid_pdb_filename)
            
        else:
        
            ## DEFINING COSOLVENT MAPPING FOLDER
            cosolvent_map_folder="cosolvent_mapping"
            ## DEFINING OUTPUT PICKLING PATH
            pickle_name = "map-10-0.33.pickle"
            # path_pickle = os.path.join(path_analysis, cosolvent_map_folder, pickle_name)
            path_pickle_list = [os.path.join(each_path, cosolvent_map_folder, pickle_name) for each_path in path_analysis_list]
            

            ## DEFINING PATH
            path_pdb = os.path.join(path_analysis_list[0], cosolvent_map_folder, grid_pdb_filename)
        
        ## LOADING PDB FILE
        grid_pdb_file = md.load(path_pdb)
        
        ## GETTING RID
        grid_xyz = grid_pdb_file.xyz[0]
        
        ## DEFINING PICKLES AND MAX N
        max_N = 10
        cutoff = 0.33
        
        ## DEFINING PATH TO PDB
        '''
        path_pdb = os.path.join( path_analysis,
                                 grid_pdb_filename)
        '''
    
    ## UNLOADING THE RESULTS
    # results = pd.read_pickle(path_pickle)

    
    ## DEFINING EMPTY ARRAY
    stored_mapping = []
    stored_unnorm_p_N = []
    
    ## LOOPING THROUGH EACH PICKLE
    for path_pickle in path_pickle_list:
    
        ## RESTORE PICKLE INFORMATION
        results = pickle_tools.load_pickle_results(path_pickle)[0]
        
        ## EXTRACTING DETAILS
        mapping, unnorm_p_N = results[0], results[1]
        ## APPENDING
        stored_mapping.append(mapping)
        stored_unnorm_p_N.append(unnorm_p_N)

    '''
    unnorm_p_N is 5, 10792, 10
    5 is the number of solvents
    10792 is the grid points
    10 is the histogram. number of occurances for specific grid point. 
    '''
    
    #%%
    
    ## DEFINING EMPTY ARRAY
    total_unnorm_p_N = np.zeros(stored_unnorm_p_N[0].shape)
    total_solvent_list = stored_mapping[0].solvent_list
    
    ## LOOPING AND FINDING SOLVENT LIST
    for each_idx, unnorm_p_N in enumerate(stored_unnorm_p_N):
        ## GETTING SOLVENT LIST
        current_solvent_list=stored_mapping[each_idx].solvent_list
        ## LOOPING AND ADDING INDEX
        for current_idx, each_solvent in enumerate(current_solvent_list):
            ## FINDING INDEX
            solv_index = total_solvent_list.index(each_solvent)
            ## ADDING TO SPECIFIC INDEX
            total_unnorm_p_N[solv_index] += unnorm_p_N[current_idx]
        
    
    
    #%%
    
    ## SOLVENT LIST
    # solvent_list = mapping.solvent_list
    solvent_list =  total_solvent_list
    unnorm_p_N = total_unnorm_p_N
    
    
    ## CREATING DICTIONARY
    prob_occurances_dict = {}
    
    ## GETTING HISTOGRAM INFORMATION
    for idx,each_solvent in enumerate(solvent_list):
        ## COMPUTING PROBABILITY DISTRIBUTION
        prob_occurances = unnorm_p_N[idx].sum(axis=0)/unnorm_p_N[idx].sum()
        ## STORING
        prob_occurances_dict[each_solvent] = prob_occurances[:]
        
    
    #%%
    
    
#    #####################################
#    ### PLOTTING THE PROBABILITY DIST ###
#    #####################################
#    
#    ## IMPORTING FUNCTIONS
#    import MDDescriptors.core.plot_tools as plot_funcs
#    
#    ## DEFINING SOLVENT DICTIONARY
#    SOLVENT_COLOR_DICT={
#            'HOH': 'blue',
#            'FMT': 'pink',
#            'MTA': 'cyan',
#            'PRO': 'red',
#            'ACA': 'green',
#            }
#        
#    # 'red',
#    ## GETTING OCCURANCE ARRAY
#    occurance_array = np.arange(0,mapping.max_N)
#    
#    ## CREATING PLOT
#    fig, ax = plot_funcs.create_plot()
#    
#    ## PLOTTING PROBABILITY OF OCCURANCES
#    for idx,each_solvent in enumerate(solvent_list):
#        ## GETTING Y  VALUES
#        y = prob_occurances_dict[each_solvent]
#        
#        ## GETTING COLOR
#        color = SOLVENT_COLOR_DICT[each_solvent]
#        
#        ## PLOTTING
#        ax.plot(occurance_array, y, 
#                linestyle='-',
#                linewidth=2, 
#                color = color, 
#                label = each_solvent)
#    
#    ## ADDING AXIS TITLE
#    ax.set_xlabel("Number of occurrences")
#    ax.set_ylabel("Probability")
#    
#    ## ADDING LEGEND
#    ax.legend()
#    
#    ## PRINTING
#    store_fig_loc = r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20191104\Images\cosolvent_map"
#    ## NAME
#    fig_name = "C11COOH_prob"
#    ## SAVE FIG
#    save_fig=False
#    ## STORING FIG
#    plot_funcs.store_figure( fig = fig,
#                             path = os.path.join(store_fig_loc,
#                                                 fig_name), # _charged
#                             save_fig = save_fig
#                             )
#    
    
    #%%
    
    
    ### FUNCTION TO COMPUTE EXPECTED VALUE FOR A GIVEN ARRAY
    def compute_expected_grid_histogram(array):
        '''
        The purpose of this function is to compute the expected value for a 
        hisogram with the number of occurances.
        INPUTS:
            array: [np.array]
                array with shape: n_atoms, num_occurances
                Note that the number of occurances should start with zero
        OUTPUTS:
            expected_value: [np.array]
                array with expected value for each grid atom (shape: n_atom)
        '''
        ## GETTING SHAPE
        row, col = array.shape
        ## 0 TO N ARRAY
        number_array = np.arange(0,col)
        ## GETTING PROBABILITIES
        prob_dist = array/ array.sum(axis=1)[:,np.newaxis]
        ## GETTING EXPECTED VALUE
        expected_value = np.sum(prob_dist * number_array, axis = 1)
        return expected_value
    
    ## DEFINING EXPECTED VALUE
    expected_value_dict = {}
    
    ## GETTING HISTOGRAM INFORMATION
    for idx,each_solvent in enumerate(solvent_list):
        ## GETTING EXPECTED VALUE
        expected_value_hist = compute_expected_grid_histogram(array = unnorm_p_N[idx])
        ## STORING
        expected_value_dict[each_solvent] = expected_value_hist
        
    #%%
    
    ## IMPORTING FUNCTION FOR RESCALING
    from sklearn.preprocessing import MinMaxScaler
    
    ### FUNCTION TO PRINT PDB FILE
    def print_pdb_file( path_pdb_file,
                        xyz,
                        b_factors,
                        box_dims,
                        verbose = True):
        '''
        The purpose of this function is to print PDB box for a given file.
        INPUTS:
            path_pdb_file: [str]
                path to output pdb file
            xyz: [np.array]
                xyz positions
            b_factors: [np.array]
                b factors associated with the PDB file
            box_dims: [np.array, shape=3]
                box dimensions to output in nanometers
        OUTPUTS:
            pdb file with the xyz positions and b factors printed
        '''
        ## PRINTING
        if verbose is True:
            print( '--- PDB file written to %s' %(  path_pdb_file ) )
        
        ## OPENING PDB FILE
        with open(path_pdb_file, 'w+') as pdbfile:
            ## WRITING HEADER
            pdbfile.write( 'TITLE     frame t=1.000 in water\n' )
            pdbfile.write( 'REMARK    THIS IS A SIMULATION BOX\n' )
            pdbfile.write( 'CRYST1{:9.3f}{:9.3f}{:9.3f}{:>7s}{:>7s}{:>7s} P 1           1\n'.format( box_dims[0]*10, box_dims[1]*10, box_dims[2]*10, '90.00', '90.00', '90.00' ) )
            pdbfile.write( 'MODEL        1\n' )
            ## WRITING GRID
            for idx, coord in enumerate(xyz):
                ## DEFINING LINE
                line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format( \
                        'ATOM', idx+1, 'C', '', 'SUR', '', 1, '', coord[0]*10, coord[1]*10, coord[2]*10, 1.00, b_factors[idx], '', '' )
                ## ADDING TO PDB
                pdbfile.write( line )
        
            ## ENDING PDB FILE
            pdbfile.write( 'TER\n' )
            pdbfile.write( 'ENDMDL\n' )
            
        return
    
    ## LOOPING THROUGH EACH SOLVENT
    for residue in solvent_list:
    
        ## DEFINING UNIT CELL LENGTHS
        box_dims = grid_pdb_file.unitcell_lengths[0]
        
        ## DEFINING GRID
        grid = grid_xyz
        
        ## GETTING EXPECTED VALUE
        expected_values = expected_value_dict[residue]
        
        ## MIN MAX SCALAR
        scaler = MinMaxScaler()
        scaler.fit(expected_values[:,np.newaxis])
        expected_values_rescaled = scaler.transform(expected_values[:,np.newaxis])[:,0]
        
        ## DEFINING PATH
        path_pdb_file = os.path.join(main_sim_path,
                                     "cosolvent_map" + '_' + ligand_name + '_' + residue + ".pdb")
                                     
        ## PRINTING OUT PDB
        print_pdb_file( path_pdb_file = path_pdb_file,
                        xyz = grid,
                        b_factors = expected_values_rescaled,
                        box_dims = box_dims,
                        verbose = True)
    
    #%%
    
#    ## CREATING PLOT
#    fig, ax = plot_funcs.create_plot()
#    
#    ## GETTING GRID
#    grid_index=np.arange(0,mapping.total_grid)
#    
#    ## PLOTTING PROBABILITY OF OCCURANCES
#    for idx,each_solvent in enumerate(solvent_list):
#        ## GETTING EXPECTED VALUE
#        expected_values = expected_value_dict[each_solvent]
#        
#        ## GETTING COLOR
#        color = SOLVENT_COLOR_DICT[each_solvent]
#        
#        ## PLOTTING
#        ax.plot(grid_index, 
#                expected_values, 
#                linestyle="-",
#                linewidth=.5,
#                color = color,
#                label = each_solvent)
#        
#    
#    ## ADDING AXIS TITLE
#    ax.set_xlabel("Grid index")
#    ax.set_ylabel("Expected value")
#    
#    ## ADDING LEGEND
#    ax.legend()
#    
#    ## NAME
#    fig_name = "C11COOH_expected_value"
#    ## SAVE FIG
#    save_fig=True
#    ## STORING FIG
#    plot_funcs.store_figure( fig = fig,
#                             path = os.path.join(store_fig_loc,
#                                                 fig_name), # _charged
#                             save_fig = save_fig
#                             )
                             
    #%%
    
    
    
    # cmd.select("SURFACE", "resname SUR")
    # cmd.spectrum("b","red_white_blue", "SURFACE", minimum=0, maximum=1)
    # cmd.show("surface", "SURFACE") # ISOSURFACE
    # cmd.select( "SURFACE_HOH","cosolvent_map_HOH")
    # cmd.set("transparency", 0.20, "SURFACE" )
    
    ## RENORMALIZE EXPECTED VALUE BETWEEN 0 AND 1
    
    ## OUTPUTTING PDB FILE
    
    
    
    
        