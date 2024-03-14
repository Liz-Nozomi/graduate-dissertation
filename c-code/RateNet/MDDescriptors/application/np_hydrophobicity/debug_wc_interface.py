#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_wc_inteface.py

The purpose of this code is to debug the wc interface by loading the densities 
and seeing how they are contoured.

This will also take into account ligand overlap and so forth

Written by: Alex K. Chew (03/19/2020)


"""
## LOADING MODULES
import os
import matplotlib
# ; matplotlib.use('agg')
import numpy as np
import MDDescriptors.core.pickle_tools as pickle_tools
import mdtraj as md
import glob
## IMPORTING PLOTTING FUNCTION
from MDDescriptors.surface.willard_chandler import mlab_plot_density_field
from MDDescriptors.surface.willard_chandler_parallel import normalize_density_field

## DEFINNIG SIM PATH
from MDDescriptors.application.np_hydrophobicity.global_vars import \
    PARENT_SIM_PATH, COMBINE_NEIGHBORS_DIR, COMBINE_NEIGHBORS_LOG, GRID_LOC, GRID_OUTFILE, \
    OUT_HYDRATION_PDB, MU_PICKLE, PROD_GRO
    
from MDDescriptors.application.np_hydrophobicity.remove_grids_for_planar_SAMs import remove_grid_for_planar_SAMs, extract_new_grid_for_planar_sams
    
## GETTING OVERLAP
from MDDescriptors.application.np_hydrophobicity.check_ligand_to_grid_overlap import check_lig_overlap

## IMPORT TOOLS
import MDDescriptors.core.import_tools as import_tools

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## IMPORITNG GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, IMAGE_LOCATION

## IMPORTING MLAB
from mayavi import mlab

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

#### GLOBAL VARS
save_fig = True
# False
#  True

### FUNCTION TO GET FRAC OF GRID POINTS WITH LIGANDS
def compute_frac_of_lig_in_grid(traj_data,
                                lig_overlap,
                                grid):
    '''
    The purpose of this script is to compute the fraction of ligands that 
    are overlapped with the grid points.
    INPUTS:
        lig_overlap: [class]
            object containing ligand overlap information
        traj_data: [obj]
            trajectory data object
        grid: [np.array]
            N x 3 grid points to test for
    OUTPUTS:
        fraction_grid_points: [np.array]
            fraction of grid ponts that are overlapping
        num_neighbors_array: [np.array]
            number of neighbors for each grid point / heavy atom
    '''
    ## GETTING NEIGHBORS
    num_neighbors_array = lig_overlap.compute_neighbors(traj = traj_data.traj,
                                                        grid = grid,
                                                        neighbor_search="kdtree",
                                                        verbose = False)

    ## FINDING AVERAGE NEIGHBOR ARRAY
    avg_neighbor_array = np.mean(num_neighbors_array,axis=1)

    ## COMPUTING FRACTION OF GRID POINTS WITH LIGANDS
    fraction_grid_points = np.sum(avg_neighbor_array>0) / avg_neighbor_array.size
    
    return fraction_grid_points, num_neighbors_array



############################################    
### CLASS FUNCTION TO DEBUG WC INTERFACE ###
############################################
class debug_wc_interface():
    '''
    This class holds functions to debug the wc interface. 
    FUNCTIONS:
        plot_contour_map_of_contour_level_vs_probe_radius:
            plots contour map vs. probe radius heat map
        loop_to_get_contour_vs_probe_radius:
            loops through many contour and probe radius
        plot_wc_density_field_with_traj:
            plots density of the wc interface
    
    '''
    def __init__(self):
        
        return
    
    ### FUNCTION TO PLOT WC DENSITY
    @staticmethod
    ### FUNCTION TO GET MAYAVI PLOT WITH DENSITY FIELDS
    def plot_wc_density_field_with_traj(traj_data,
                                        avg_density_field,
                                        interface,
                                        size = (500, 500)):
        '''
        The purpose  of this script is to plot the density field with heavy atoms. 
        INPUTS:
            traj_data: [obj]
                trajectory object
            avg_density_field: [np.array]
                average density field outputted from wc interface code
            interface: [obj]
                interface object from wc interface
        OUTPUTS:
            fig: [obj]
                mayavi figure object
        '''
        ## IMPORTING MLAB
        from mayavi import mlab
        ## COMPUTING OVERLAP
        lig_overlap = check_lig_overlap(traj_data = traj_data,)
        
        ## PLOTTING FIELD
        figure = mlab_plot_density_field(density_field = avg_density_field,
                                grid = interface.grid,
                                num_grid_pts = interface.num_grid_pts,
                                size = size)
        
        ## FIGURE FROM 
        fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                  atom_index = lig_overlap.atom_index,
                                  frame = 0,
                                  figure = figure,
                                  dict_atoms = plot_funcs.ATOM_DICT,
                                  dict_colors = plot_funcs.COLOR_CODE_DICT)
        
        ## ORIENTING
        mlab.view(figure = fig, elevation = 90, azimuth = 0,
                  distance = 20)
        
        return fig
    
    ### FUNCTION TO GET FRACTION OF GRID POINTS
    @staticmethod
    def loop_to_get_contour_vs_probe_radius(avg_density_field,
                                            interface,
                                            traj_data,
                                            contour_array = np.arange(0, 32, 2),
                                            probe_radius = np.arange(0, 0.40, 0.05),
                                            remove_grid_func = None):
        '''
        The purpose of this function is to loop through the contours versus probe 
        radius to get fraction of grid points that are overlapped with the ligands. 
        INPUTS:
            avg_density_field: [np.array]
                average density field outputted from wc interface code
            interface: [obj]
                interface object
            traj_data: [obj]
                trajectory object
            contour_array: [np.array]
                contour array
            probe_radius: [np.array]
                probe radius to search for
            remove_grid_func: [obj]
                object for removing grid points for planar samS
        OUTPUTS:
           storage_frac_grid_points: [np.array, shape = (num_contour, num_probe_radius) ]
               fraction of grid points for each contour and probe radius
        '''
        
    
        ## GETTING ARRAY
        storage_frac_grid_points = np.zeros( (len(contour_array), len(probe_radius)) )
        
        ## LOOPING RANGE
        for idx_contour, contour in enumerate(contour_array):
            print("Contour level: %.2f [%d of %d]"%(contour, idx_contour, len(contour_array)) )
            ## GETTING GRID FOR CONTOUR 
            grid_pts = interface.get_wc_interface_points(density_field = avg_density_field,
                                                         contour = contour,
                                                         )
            if remove_grid_func is not None:
                ## REMOVING GRID POINTS
                print("Removing grid points based on cutoffs!")
                grid_pts = extract_new_grid_for_planar_sams(grid = grid_pts,
                                                        water_below_z_dim = remove_grid_func.water_below_z_dim,
                                                        water_above_z_dim = remove_grid_func.water_above_z_dim )[0]
            
            ## LOOPING THROUGH PROBE RADIUS
            for idx_radius, radius_cutoff in enumerate(probe_radius):
                print(" --> Probe Radius: %.2f [%d of %d]"%(radius_cutoff, idx_radius, len(probe_radius)) )
                ## GENERATING LIGAND OVERLAP
                lig_overlap = check_lig_overlap(traj_data = traj_data,
                                                cutoff_radius = radius_cutoff)
                
                ## COMPUTING FRACTION OF GRID POINTS WITHIN GRID
                fraction_grid_points, num_neighbors_array = compute_frac_of_lig_in_grid(traj_data = traj_data,
                                                                                        lig_overlap =lig_overlap,
                                                                                        grid = grid_pts)
                ## STORING
                storage_frac_grid_points[idx_contour, idx_radius] =  fraction_grid_points
                
        return storage_frac_grid_points
    
    ### FUNCTION TO PLOT CONTOUR LEVELS
    @staticmethod
    def plot_contour_map_of_contour_level_vs_probe_radius(x,y,heatmap,
                                                          levels = [0, 0.00000000000001, 0.05, 0.25, 0.5, 0.75, 1.0],
                                                          cmap = 'jet',
                                                          want_grid = False,
                                                          fig = None,
                                                          ax = None,
                                                          want_color_bar = True,
                                                          return_contourf = False,
                                                          ):
        '''
        The purpose of this function is to plot the contour level vs. probe 
        radius and generate a map of regions where the fraction of grid points are 
        zero. The idea is to find values where there is no overlap in the ligand 
        space. 
        INPUTS:
            x: [np.array]
                contour array
            y: [np.array]
                probe radius array
            heatmap: [np.array]
                map of the different combinations
            levels: [list]
                list of values to plot
            fig, ax: [obj]
                figure and object to add to plot
            want_color_bar: [logical]
                True if you want color bar
            return_contourf: [logical]
                True if you want to return contourf
        OUTPUTS:
            fig, ax:
                figure and axis for the contour level vs. probe radius
            cont: [obj]
                optional object
        '''
        ## PLOTTING
        if fig is None or ax is None:
            fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
        
        ## GETTING AXIS
        ax.set_xlabel('Contour level')
        ax.set_ylabel('Probe radius')
            
        ## PLOTTING CONTOUR
        cont = ax.contourf(x, y, heatmap.T, levels = levels, cmap=cmap)
        
        ## ADDING COLOR BAR
        if want_color_bar is True:
            fig.colorbar(cont)
        
        ## ADDING GRID
        if want_grid is True:
            ax.grid(which='major',linewidth=0.5)
    
        ## FIT PLOT
        fig.tight_layout()
        if return_contourf is True:
            return fig, ax, cont
        else:
            return fig, ax
    



#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    
    ## DEFINING PARENT SIM PATH
    parent_sim_path = PARENT_SIM_PATH
    
    ## DEFINING DIRECTORY
    directory_list = ['20200224-planar_SAM_spr50',
                      '20200224-GNP_spr_50']
    directory_list = [
                # "20200224-GNP_spr_50",
                # "20200328-Planar_SAMs_new_protocol-shorterequil_spr50"
               #  '20200331-Planar_SAMs_no_temp_anneal_or_vacuum_withsprconstant'
            # '20200326-Planar_SAMs_with_larger_z_8nm_frozen'
            # '20200326-Planar_SAMs_with_larger_z_frozen',
            
            '20200401-Renewed_GNP_sims_with_equil',
            '20200403-Planar_SAMs-5nmbox_vac_with_npt_equil',
            
                      ]
    
    ## LOOPING THROUGH DIRECTORY
    dir_list = calc_tools.flatten_list_of_list([glob.glob(os.path.join(parent_sim_path, dirlist) + "/*") 
                                                for dirlist in directory_list])
    
    ## DEFINING IF YOU WANT NORMALIZATION
    want_norm = False
    
    ## DEFINING CONTOUR LEVEL
    contour_loc = "26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000"
    
    ## DEFINING GRO FILE NAME
    gro_file_name = "sam_prod_2000_50000-heavyatoms.gro" 
    # "sam_prod.gro"
    
    ## DEFINING WC INTERFACE
#    grid_loc = GRID_LOC
    grid_loc = "grid-45000_50000"
    
    ## PICKLE FOR REMOVING GRO
    remove_grid_pickle = "remove_grid.pickle"
    
    ## DEFINIGN PLANAR SAMS
    planar_sim_list = [
            '20200403-Planar_SAMs-5nmbox_vac_with_npt_equil'
            ]
    
    ## lOOPING THROUGH EACH
    for idx, each_dir in enumerate(dir_list):
        # if idx == 0:
        ## GETTING DIRNAME
        dirname = os.path.basename(os.path.dirname(each_dir))
        
        ## SEEING IF PLANAR SIM
        if dirname in planar_sim_list:
            is_planar = True
        else:
            is_planar = False

        ## GETTING BASENAME
        dir_basename = os.path.basename(each_dir)

        ## DEFINING SIMULATION NAME
        prefix = dir_basename
        
        ## DEFINING PATH TO LOAD
        path_to_load = os.path.join(each_dir, contour_loc)
        
        ## GETTING PATH TO GRO FILE
        path_to_gro = os.path.dirname(path_to_load)
        
        ## DEFINING DEBUG
        debug_pickle = "debug.pickle"
        
        ## DEFINING PICKLE FOR OUTPUT
        storage_pickle = debug_wc_interface.__name__
        
        ## DEFINING PATH TO IT
        path_to_pickle = os.path.join(path_to_load,
                                      grid_loc,
                                      debug_pickle)
        ## LOADING GRO FILE
        path_gro = os.path.join(path_to_gro,
                                gro_file_name)
        
        ## LOADING THE GRO FILE
        traj_data = import_tools.import_traj(directory = path_to_gro,
                                             structure_file = gro_file_name,
                                             xtc_file = gro_file_name)
        
        ## LOADING THE PICKLE
        grid, interface, avg_density_field = pickle_tools.load_pickle(file_path = path_to_pickle)[0]
        
        if is_planar is True:
            remove_grid_path = os.path.join(path_to_load,
                                            grid_loc,
                                            remove_grid_pickle)
            ## REMOVING GRID FUNCTION
            remove_grid_func = pickle_tools.load_pickle(file_path = remove_grid_path)[0][0]
            
            ## REMOVING GRID POINTS
            grid = extract_new_grid_for_planar_sams(grid = grid,
                                                    water_below_z_dim = remove_grid_func.water_below_z_dim,
                                                    water_above_z_dim = remove_grid_func.water_above_z_dim )[0]
        else:
            remove_grid_func = None
    
    
    #%%
        ## NORMALIZING
        if want_norm is True:
            norm_density_field, maxima_value = normalize_density_field(density_field = avg_density_field)
            print("Normalizing density field -- max value: %.2f"%(maxima_value) )
            
            ## RE-DEFINING DENSITY FIELD
            avg_density_field = norm_density_field[:]
            prefix = prefix + "_NORMALIZED"
            
            ## GENERATING ARRAY OF CONTOURS
            contour_array = np.arange(0, 1, 0.05)
            
            ## RELABELING
            path_to_storage_pickle = os.path.join(path_to_load,storage_pickle + "_NORMALIZED")
            
            
        else:
            ## GENERATING ARRAY OF CONTOURS
            contour_array = np.arange(0, 32, 2)
            ## PATH TO STORAGE PICKLE
            path_to_storage_pickle = os.path.join(path_to_load,storage_pickle)
        
    
        #%% PLOTTING DENSITY FIELD WITH ATOMS
        
        ## DEBUGGIN WC NTERFACE
        wc_debug = debug_wc_interface()
    
        ## DEFINING CLASS FUNCTION
        fig = wc_debug.plot_wc_density_field_with_traj(traj_data = traj_data,
                                                        avg_density_field = avg_density_field ,
                                                        interface = interface,
                                                        size = (1000, 1000),
                                                     )    
        ## SAVING
        fig_name=prefix + '-mayavi'
        mlab.savefig(os.path.join(IMAGE_LOCATION,
                                  fig_name + '.png'))
        
    
        #%% LOOPING AND GETTING ALL POSSIBLE CONTOUR LEVELS AND PROBE RADIUS
        
        ## GENERATING PROBE RADIUS
        probe_radius = np.arange(0, 0.40, 0.05)
        
        ## DEFINING INPUTS
        input_dict = {
                'contour_array': contour_array,
                'probe_radius': probe_radius,
                'traj_data': traj_data,
                'avg_density_field': avg_density_field,
                'remove_grid_func' : remove_grid_func,
                }
        
        # bash extract_hydration_maps_with_python.sh > hydration.out 2>&1 &
        ## RUNNING FUNCTION
        storage_frac_grid_points = pickle_tools.save_and_load_pickle(function = wc_debug.loop_to_get_contour_vs_probe_radius,
                                                                     inputs = input_dict,
                                                                     pickle_path =  path_to_storage_pickle,
                                                                     rewrite=True)
        
        #%% GENERATING PLOT FOR IT
        fig, ax =wc_debug.plot_contour_map_of_contour_level_vs_probe_radius(x = contour_array,
                                                                   y = probe_radius,
                                                                   heatmap = storage_frac_grid_points,
                                                                   levels = [0, 0.00000000000001, 0.05, 0.25, 0.5, 0.75, 1.0],
                                                                   )
        
        fig_name=prefix + '-heatmap_contour_space'
        plot_funcs.store_figure(fig = fig, 
                                path = os.path.join(IMAGE_LOCATION,
                                                    fig_name), 
                                fig_extension = 'png', 
                                save_fig=True, 
                                dpi=600)
        
    
    #%%
    
    ## GETTING BULK WATER
    dir_loc="PURE_WATER_SIMS"
    bulk_water_loc="wateronly-6_nm-tip3p-300.00_K"
    contour_level="0.24-0.25-0-50000"
    ## DEFINING DEBUG
    debug_pickle = "debug.pickle"
    path_to_bulk_water=os.path.join(PARENT_SIM_PATH,
                                    dir_loc,
                                    bulk_water_loc,
                                    contour_level,
                                    GRID_LOC,
                                    debug_pickle
                                    )
    
    from MDDescriptors.surface.willard_chandler_parallel import plot_wc_interface_across_z_dist
    
    ## LOADING THE PICKLE
    grid, interface, avg_density_field = pickle_tools.load_pickle(file_path = path_to_bulk_water)[0]
    
    ## PLOTTING WC INTERFACE
    fig, ax = plot_wc_interface_across_z_dist(avg_density_field = avg_density_field,
                                              interface = interface)
    
    ax.set_ylim([25, 35])
    
    fig.tight_layout()
    
    fig_name = bulk_water_loc

    plot_funcs.store_figure(fig = fig, 
                            path = os.path.join(IMAGE_LOCATION,
                                                fig_name), 
                            fig_extension = 'png', 
                            save_fig=True, 
                            dpi=600)
    
    '''
    np.max(avg_density_field)
    Out[12]: 30.811162293762063
    np.min(avg_density_field)
    Out[13]: 30.11736443548398
    np.mean(avg_density_field)
    Out[14]: 30.456303096857805
    '''
    
