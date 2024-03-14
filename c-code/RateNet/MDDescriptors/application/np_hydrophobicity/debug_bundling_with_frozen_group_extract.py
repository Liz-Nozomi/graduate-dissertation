# -*- coding: utf-8 -*-
"""
debug_bundling_with_frozen_group_extract.py
The purpose of this function is to extract the output from the frozen groups

Written by: Alex K. Chew (alexkchew@gmail.com, 02/05/2020)

"""
import os
import numpy as np
import glob
import pandas as pd
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path

## PICKLING FUNCTIONS
from MDDescriptors.core.pickle_tools import load_pickle_results
import MDDescriptors.core.plot_tools as plot_tools
from MDDescriptors.core.decoder import decode_name

## DEFAULTS
plot_tools.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_tools.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING IMAGE LOCATION
IMAGE_LOC = r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200217\images\bundling"


#%% MAIN SCRIPT
if __name__ == "__main__":
        
    ## DEFINING PATHS
    ## DEFINING MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
    
    simulation_dir=r"20200212-Debugging_GNP_spring_constants_heavy_atoms"
    # r"20200129-Debugging_GNP_spring_constants_heavy_atoms"
    
    ## DEFINING PATH TO LIST
    path_to_sim_list = os.path.join(main_dir,simulation_dir)
    
    ## GETTING FULL LIST
    full_sim_list = np.array(glob.glob(path_to_sim_list + "/*"))
    
    # specific_dir = r"MostlikelynpNVTspr_25-EAM_300.00_K_2_nmDIAM_C11NH3_CHARMM36jul2017_Trial_1_likelyindex_1"
    specific_dir = r"MostlikelynpNVTspr_2000-EAM_300.00_K_2_nmDIAM_C11NH3_CHARMM36jul2017_Trial_1_likelyindex_1"
    specific_dir = r"MostlikelynpNVTspr_2000-EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1"
    
    ## DEFINING DECODED NAME
    decoded_name = decode_name(specific_dir, decode_type='nanoparticle')
    
    #%%

    ## GENERATING COLORS FOR LIGAND
    LIGAND_LINESTYLES = {
            'dodecanethiol': "solid",
            "C11NH3": "dotted",
            }
    
    LIGAND_COLORS = {    
            'dodecanethiol': "k",
            "C11NH3": "r",
            "C11OH": "b",
            }
    ## GENERATING LINE STYLES
    SPRING_CONSTANT_COLORS = {
            "25": "red",
            "50": "black",
            "100": "blue",
            '500': "magenta",
            '1000': 'green',
            "2000": "cyan",
            '10000': 'yellow',
            }
    
    relative_path_to_pickle = os.path.join("analysis",
                                           "main_compute_bundles",
                                           "results.pickle")
    
    ## GETTING DECODED NAME
    decoded_name_list = [ decode_name(os.path.basename(path_to_sim), decode_type='nanoparticle') for path_to_sim in full_sim_list]
    
    decoded_name_df = pd.DataFrame(decoded_name_list)
    
    ## DEFINING UNIQUE LIGANDS
    unique_ligand_list = np.unique(decoded_name_df.ligand)
    
    #%%
    
    ## DEFINING FIGURE TYPE
    figure_type="avg_bundles_vs_spring_constant"
    # figure_type="num_bundles_vs_frames"
    
    ## GENERATING STORAGE DICTIONARY
    if figure_type == "avg_bundles_vs_spring_constant":
        ## DEFINING STORAGE
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## LOOPING THROUGH EACH LIGAND
    for ligand_type in unique_ligand_list:
        
        ## GETTING INDICES
        current_df = decoded_name_df[decoded_name_df.ligand == ligand_type]
    
        ## CONVERTING
        current_df = current_df.astype({'spring_constant': float})
        ## SORTING
        current_df = current_df.sort_values(by=['spring_constant'])
        
        ## GETTING COLORS
        cmap = plot_tools.get_cmap(len(current_df))
        
        ## GETTING INDEX
        index = current_df.index
        
        ## CREATING PLOT
        if figure_type == "num_bundles_vs_frames":
            fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
        
        ## ADDING AXIS
        if figure_type == "num_bundles_vs_frames":
            ax.set_xlabel("Frames")
            ax.set_ylabel("Number of bundling groups")
        elif figure_type == "avg_bundles_vs_spring_constant":
            ax.set_xlabel("Spring constant")
            ax.set_ylabel("Avg. number of bundling groups")
            
        ## GENERATING STORAGE DICTIONARY
        if figure_type == "avg_bundles_vs_spring_constant":
            ## DEFINING STORAGE
            storage_dict = {}
        
        ## GETTING LIST OF SIMS
        sims_list = full_sim_list[index]
    
        ## LOOPING THROUGH EACH
        for idx, path_to_sim in enumerate(sims_list):
    
            ## GETTING SPRING CONSTANT
            spring_constant = current_df.spring_constant.iloc[idx]
            
            ## GETTING COLOR
            color = cmap(idx)
            # SPRING_CONSTANT_COLORS[spring_constant]
            
            ## DEFINING PATH
            path_to_pickle=os.path.join(path_to_sim,
                                        relative_path_to_pickle)
        
            ## LOADING PICKLE
            results = load_pickle_results(path_to_pickle)[0][0]
        
            ## GETTING ARRAY FOR TOTAL BUNDLING GROUPS
            total_bundling_groups = np.array(results.bundling_groups.lig_total_bundling_groups)
            
            ## GETTING TIME ARRAY
            time = np.arange(len(total_bundling_groups))
        
            ## PLOTTING
            if figure_type == "num_bundles_vs_frames":
                ax.plot(time, 
                        total_bundling_groups, 
                        color = color,
                        label = spring_constant
                        )
            
            ## STORING
            if figure_type == "avg_bundles_vs_spring_constant":
                storage_dict[spring_constant] = {
                        'Avg bundle' : np.mean(total_bundling_groups),
                        'Std bundle' : np.std(total_bundling_groups), 
                        }
        ## PLOTTING
        if figure_type == "avg_bundles_vs_spring_constant":
            x = np.array(list(storage_dict.keys()) ).astype('float')
            y = [storage_dict[each_key]['Avg bundle'] for each_key in storage_dict ]
            y_err = [storage_dict[each_key]['Std bundle'] for each_key in storage_dict ]
            color = LIGAND_COLORS[ligand_type]
            ## PLOTTING
            ax.errorbar(x = x, 
                        y = y,
                        yerr = y_err,
                        color = color,
                        fmt = '.',
                        linestyle = None,
                        label = ligand_type)
            
        ## TIGHT LAYOUT
        fig.tight_layout()
        ## ADDING LEGEND
        ax.legend()
        
        ## STORING FIGURE
        if figure_type == "num_bundles_vs_frames":
            figure_name = figure_type + "-" + ligand_type
            ## STORING FIGURE
            plot_tools.save_fig_png( fig = fig,
                                    label = os.path.join(IMAGE_LOC,
                                                      figure_name))
        
    ## STORING FIGURE
    if figure_type == "avg_bundles_vs_spring_constant":
        figure_name = figure_type
        ## STORING FIGURE
        plot_tools.save_fig_png( fig = fig,
                                label = os.path.join(IMAGE_LOC,
                                                  figure_name))

    