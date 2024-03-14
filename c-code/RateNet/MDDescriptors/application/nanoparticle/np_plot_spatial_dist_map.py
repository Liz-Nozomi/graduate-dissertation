# -*- coding: utf-8 -*-
"""
np_plot_spatial_dist_map.py
Created on Tue Sep 10 13:51:00 2019

@author: akchew
"""
## IMPORTING MODULES
import numpy as np
from mayavi import mlab
import os

## IMPORTING CUSTOM MODULES
import MDDescriptors.core.pickle_tools as pickle_tools

## IMPORTING SPATIAL DIST MAPS
from MDDescriptors.application.nanoparticle.np_spatial_dist_map import calc_np_spatial_dist_map

## IMPORTING PROBABILITY DENSITY MAPS
from MDDescriptors.visualization.plot_probability_density_map import plot_prob_density_map, MAP_INPUT, PRINT_FIGURE_INFO, update_image_combined_image
    
#%% MAIN SCRIPT
if __name__ == "__main__":

    ## DEFINING PICKLE DIRECTORY TO LOOK INTO
    pickle_path = r"R:\scratch\nanoparticle_project\scripts\analysis_scripts\PICKLE_ROT"
    # r"R:\PICKLE_ROT"
    # 
    
    ## DEFINING DIFFERENT TYPES
    cosolvent_list = ["formamide", "phenol"]
    #  "dimethylsulfide", "indole", "methanethiol"
    # [ "methanol", "toluene", "formicacid"]
    ligand_list = ["C11COOH","dodecanethiol" ]
    
    # "formicacid"
    # "methanol"
    # "toluene"
    ## LOOPING
#    for cosolvent in cosolvent_list:
#        for ligand in ligand_list:
#        
    ## DEFINING DESIRED FILE
    Pickle_loading_file = r"MostNP-EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1-lidx_1-cosf_10000-formicacid_methanol_phenol_1_molfrac_300"

    
    ### EXTRACTING THE DATA
    ## LOADING THE PICKLE
    np_spat_dist = pickle_tools.load_pickle_from_descriptor(pickle_dir = pickle_path,
                                                       descriptor_class = calc_np_spatial_dist_map,
                                                       pickle_name = Pickle_loading_file,
                                                       verbose = False
                                                       )
    
    #%%
    iso_map_dict={
            "HOH":{
                    "contours": np.arange(2, 3+0.1, 0.2),   # np.arange(1.3, 3+0.1, 0.1),
                    "vmin": 2, # 1.5 # 0
                    "vmax": 3,
                    'nb_labels': 5, # Number of labels
                    },
            "COSOLVENT":{
                    # "contours": np.arange(1.5, 3+0.1, 0.3),   # np.arange(1.3, 3+0.1, 0.1),
                    "contours": np.arange(2, 3+0.1, 0.2),   # np.arange(1.3, 3+0.1, 0.1),
                    "vmin": 2, # 0
                    "vmax": 3,
                    'nb_labels': 5, # Number of labels
                    },
            }
    mlab.close(all=True)
    ### RUNNING PROBABILITY DENSITY MAPS
    prob_density_map = plot_prob_density_map(prob_density = np_spat_dist,
                                             iso_map_dict = iso_map_dict,
                                             combine_plot = True,
                                             want_water_only = False,
                                             want_color_bar = True,
                                             opacity = 0.2,
                                             skip_bonds = True, # Skipping bonds
                                             **MAP_INPUT
                                             )
    ## SETTING TO Z AXIS
    mlab.view(azimuth=90, elevation = 0, figure = prob_density_map.figures[0])
            #%%
            ## ADDING OUTLINE
            # prob_density_map.add_outline()
            # #%%
            ## LOADING PICKLE INFORMATION
            from MDBuilder.core.pickle_funcs import store_class_pickle, load_class_pickle
            PICKLE_PATH_SPATIAL_DIST=r"R:\scratch\nanoparticle_project\scripts\analysis_scripts\spatial_dist_storage\spatial_dist_map_info.pickle"
            SPATIAL_DIST_IMAGE_STORAGE_PATH = r"R:\scratch\nanoparticle_project\scripts\analysis_scripts\spatial_dist_storage"
            ### RELOADING IMAGE
            updated_image = update_image_combined_image( prob_density_map = prob_density_map, 
                                                         input_pickle_file_name = Pickle_loading_file, 
                                                         pickle_path = PICKLE_PATH_SPATIAL_DIST)
            
            ## DEFINING PATH TO STORE IMAGE
            image_path = r"R:\scratch\nanoparticle_project\scripts\analysis_scripts"
            
            ### STORING SPECIFIC IMAGE
            fig_name = os.path.join( image_path, Pickle_loading_file + '_front' + '.png' ) # '.eps'
            mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO ) 
        
            
        #%%
        ### ZOOMING
        # updated_image.zoom()
        
        #%%
        
        ### UPDATING IMAGE
        updated_image.reload_view()
        
        #%%
        ### STORING IMAGE
        updated_image.save_view_data(want_overwrite=True)
        
        
        #%%
        
        # test = prob_density_map.figures[0].scene.camera
        
        #%%
        

    
    
    #%%
#    
#    ## DEFINING COORDINATES
#    solute_atom_coord = prob_density_map.prob_density.solute_atom_coord
#    
#    ## DEFINING BONDS INDEX
#    bonds_index = prob_density_map.prob_density.solute_structure.bonds - 1 # -1 to resolve issue with python starting with 0
    
    #%%
    
    
#    
#    ## DEFINING ATOMNAME
#    atom_atomname = np.array(prob_density_map.prob_density.solute_structure.atom_atomname)
#    
#
#    
#    ## DEFINING BOND ATOMNAME
#    bonds_atomname = np.array(prob_density_map.prob_density.solute_structure.bonds_atomname)
#    ## STORING SHAPE
#    bonds_shape = bonds_atomname.shape
#    ## FLATTENING
#    bonds_atomname_flatten = bonds_atomname.flatten()
#    ## COPYING ARRAY
#    bonds_atomname_flatten_idx = np.copy(bonds_atomname_flatten)
#    ## LOOPING THROUGH ATOMNAMES
#    for atom_index, each_atomname in enumerate(atom_atomname):
#        ## DEFINING LOCATIONS
#        locations = np.where(bonds_atomname_flatten==each_atomname)
#        ## CHANGING LABELS
#        bonds_atomname_flatten_idx[locations] = atom_index
#        
#    ## RESHAPING
#    bonds_atomname_flatten_idx_reshape = bonds_atomname_flatten_idx.reshape(bonds_shape)
#    #%%
#        
#        
#    
#    
#    ## prob_density_map.prob_density.solute_structure.atom_atomname
#    
#    ## FINDING COORDINATES FOR EACH ATOM
#    bond_coordinates = np.array( [ index \
#                                  for index in [ prob_density_map.prob_density.solute_structure.atom_atomname.index(eachAtom) for eachAtom in bonds_atomname[0]] ] )
#    # prob_density_map.prob_density.solute_atom_coord[index]
#    
#    #%%
#    
#    import mayavi.mlab as mlab
#    
#    black = (0,0,0)
#    white = (1,1,1)
#    mlab.figure(bgcolor=white)
#    mlab.plot3d([0, 1000], [0, 0], [0, 0], color=black, tube_radius=10.)
#    mlab.plot3d([0, 0], [0, 1500], [0, 0], color=black, tube_radius=10.)
#    mlab.plot3d([0, 0], [0, 0], [0, 1500], color=black, tube_radius=10.)
#    mlab.text3d(1050, -50, +50, 'X', color=black, scale=100.)
#    mlab.text3d(0, 1550, +50, 'Y', color=black, scale=100.)
#    mlab.text3d(0, -50, 1550, 'Z', color=black, scale=100.)    
#    