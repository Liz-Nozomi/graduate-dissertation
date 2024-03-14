# -*- coding: utf-8 -*-
"""
check_ligand_to_grid_overlap.py

This script checks if the ligand heavy atoms are within the grid points of 
the self-assembled monolayer. The inputs are the ligand names, 

Written by: Alex K. Chew (01/17/2020)

"""
## IMPORTING MODULES
import os
import mdtraj as md
import numpy as np
import pandas as pd

## LOADING DATA
from MDDescriptors.surface.core_functions import load_datafile

## PLOTTING FUNCTIONS
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORT TOOLS
import MDDescriptors.core.import_tools as import_tools

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## LOADING MD LIGANDS
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_atom_indices_of_ligands_in_traj

## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## CRITICAL FUNCTION TO USE PERIODIC CKD TREE
from MDDescriptors.surface.periodic_kdtree import PeriodicCKDTree

## IMPORTING TRACK TIME
from MDDescriptors.core.track_time import track_time

## IMPORTING MATPLOTLIB
import matplotlib.pyplot as plt

## IMPORITNG GLOBAL VARS
from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH, IMAGE_LOCATION

## PLANAR CHARACTERISTICS
from MDDescriptors.application.np_hydrophobicity.debug_monolayer_characteristics import nanoparticle_structure, \
                                                                                        compute_monolayer_properties
                    
## IMPORTING MLAB
from mayavi import mlab

#### GLOBAL VARS
save_fig = True

## FUNCTION TO COMPUTE NUMBER OF ATOMS WITHIN SPHERE
def count_atoms_in_sphere( traj, pairs, r_cutoff = 0.33, periodic = True ):
    r"""
    The purpose of this function is to comptue the number of atoms within some 
    radius of cutoff.
    INPUTS:
        traj: [obj]
            trajectory object
        pairs: [np.array, shape=(N,2)]
            pairs array that you're interested in
        r_cutoff: [float]
            cutoff that you are interested in
    OUTPUTS:
        N: [np.array]
            array that is the same size as pairs across trajectories
    """
    dist = md.compute_distances( traj, pairs, periodic = periodic )
    N = np.sum( dist <= r_cutoff, axis = 1 )
    return N

### FUNCTION TO COMPUTE NEIGHBORS
def compute_neighbor_array_with_md_traj(traj,
                                        grid,
                                        atom_index,
                                        cutoff_radius,
                                        verbose = True):
    '''
    This function computes the neighbor array using md traj functions. First, 
    an atom is moved to the grid point, then we utilize the compute_distances 
    function to estimate the number of atoms in the sphere.
    INPUTS:
        traj: [obj]
            trajectory object
        grid: [np.array]
            grid points in x, y, z positions
        atom_index: [list]
            list of atom indices
        cutoff_radius: [float]
            cutoff radius to check
    OUTPUTS:
        num_neighbors_array: [num_grid, num_frames]
            neighbors within the grid points
    '''
    ## GETTING TOTAL FRAMES
    total_frames = traj.time.size

    ## GENERATING ZEROS 
    num_neighbors_array = np.zeros( shape = ( len(grid), total_frames ) )
    
    ## FINDING NO ATOM INDEX
    atom_not_index = np.array([ atom.index for atom in traj.topology.atoms if atom.index not in atom_index ])
    
    ## DECIDING WHICH ATOM TO CHANGE, USEFUL FOR MDTRAJ DISTANCES FUNCTION
    atom_index_to_change = atom_not_index[0] # First atom index that were not interested in

    ## GENERATING PAIRS TO THE SYSTEM
    pairs = np.array([ [ atom_index, atom_index_to_change ] for atom_index in atom_index ])
    ## LOOPING THROUGH THE GRID
    for idx in range(len(grid)):
        ## PRINTING
        if idx % 100 == 0 and verbose is True:
            print("Working on the grid: %d of %d"%(idx, len(grid)) )
        ## COPYING TRAJ
        copied_traj = traj[:]
        ## CHANGING THE POSITION OF ONE ATOM IN THE SYSTEM TO BE THE GRID ATOM
        copied_traj.xyz[:,atom_index_to_change,:] = np.tile( grid[idx,:], (total_frames, 1) )
        ## COMPUTE NUMBER OF ATOMS WITHIN A SPHERE
        N = count_atoms_in_sphere(traj = copied_traj, 
                                  pairs = pairs, 
                                  r_cutoff = cutoff_radius, 
                                  periodic = True )
        ## STORING IN NEIGHBORS
        num_neighbors_array[idx, :] = N[:]
    
    return num_neighbors_array

### FUNCTION TO COMPUTE NEIGHBOR ARRAY
def compute_neighbor_array_KD_tree(traj,
                                   grid,
                                   atom_index,
                                   cutoff_radius):
    '''
    The purpose of this function is to compute number of neighbor arrays using 
    periodic KD tree
    INPUTS:
        traj: [obj]
            trajectory object
        grid: [np.array]
            grid points in x, y, z positions
        atom_index: [list]
            list of atom indices
        cutoff_radius: [float]
            cutoff radius to check
    OUTPUTS:
        num_neighbors_array: [num_grid, num_frames]
            neighbors within the grid points
    '''
    ## GETTING TOTAL FRAMES
    total_frames = traj.time.size
    ## GETTING ARRAY
    num_neighbors_array = np.zeros( shape = ( len(grid), total_frames ) )
    
    ## DEFINING INDEX
    index = 0

    ## GETTING BOXES
    box = traj.unitcell_lengths[ index, : ] # ASSUME BOX SIZE DOES NOT CHANGE!
    ## DEFINING POSITIONS
    pos = traj.xyz[index, atom_index, :] # ASSUME ONLY ONE FRAME IS BEING RUN
    
    ### FUNCTION TO GET TOTAL NUMBER OF GRID
    T = PeriodicCKDTree(box, pos)
    
    ## COMPUTING ALL NEIGHBORS
    neighbors = T.query_ball_point(grid, r=cutoff_radius)

    ## LOOPING THROUGH THE LIST
    for n, ind in enumerate(neighbors):
        num_neighbors_array[n][index] += len(ind)
    return num_neighbors_array

### CLASS FUNCTION TO GET THE NEIGHBORS ARRAY
class check_lig_overlap():
    '''
    The purpose of this function is to plot the number of ligands that 
    are within the grid points. 
    INPUTS:
        traj_data: [obj]
            trajectory data from import_tools
        want_gold: [logical]
            True if you want gold atoms in the planar case

    OUTPUTS:
        self.atom_index: [index]
            atom indices that you are interested in (ligands)
    '''
    def __init__(self,
                 traj_data,
                 cutoff_radius = 0.33,
                 want_gold = True,
                 ):
        
        ## STORING RADIUS
        self.cutoff_radius = cutoff_radius
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## GETTING ATOM INDICES AND LIGAND NAME
        self.ligand_names, self.atom_index = get_atom_indices_of_ligands_in_traj( traj )
        
        ## GETTING RESIDUE NAMES
        if want_gold is True:
            resnames = calc_tools.find_unique_residue_names(traj = traj)
            
            ## SEEING IF INSIDE
            if 'AUI' in resnames:
                ## GETTING ALL ATOM INDICES
                gold_atom_indices = np.array([atom.index for atom in traj.topology.atoms if atom.residue.name == 'AUI'])
                self.atom_index = np.concatenate( (self.atom_index, gold_atom_indices) )
        
        return
    
    ### FUNCTION TO COMPUTE NEIGHBORS ARRAY
    def compute_neighbors(self,
                          traj,
                          grid,
                          neighbor_search = "kdtree",
                          verbose = True):
        '''
        The purpose of this function is to compute the neighbors with KD tree. 
        INPUTS:
            traj: [obj]
                trajectory object
            grid: [np.array]
                grid in x, y, z positions
            neighbor_search: [str]
                neighbor searching method.
                    kdtree: uses fast periodic boundary tree
                    mddistances: uses mdtraj's distance function
                
        OUTPUTS:
            num_neighbors_array: [np.array, shape = (num_grid, num_frames)]
                number of neighbors array 
        '''
            
        ## DEFINING TIME TRACKER
        time_tracker = track_time()
        
        ## COMPUTING NEIGHBORS WITH KD TREE
        if neighbor_search == "kdtree":
            num_neighbors_array = compute_neighbor_array_KD_tree(traj = traj,
                                                                   grid = grid,
                                                                   atom_index = self.atom_index,
                                                                   cutoff_radius = self.cutoff_radius,
                                                                   )
        elif neighbor_search == "mddistances":
            num_neighbors_array = compute_neighbor_array_with_md_traj(traj = traj,
                                                                      grid = grid,
                                                                      atom_index = self.atom_index,
                                                                      cutoff_radius = self.cutoff_radius,
                                                                      verbose = False,
                                                                      )
        else:
            print("Error! No neighbor searching method found for: %s"%(neighbor_search) )
            print("Maybe try kdtree or mddistances? Both should reproduce the same results.")
        if verbose is True:
            time_tracker.time_elapsed()
        
        return num_neighbors_array
    
    ### FUNCTION TO PLOT THE NUMBER OF LIGANDS INSIDE GRID POINTS
    @staticmethod
    def plot_num_ligands_within_grid(num_neighbors_array):
        '''
        The purpose of this function is to plot the number of ligands 
        within the grid points
        INPUTS:
            num_neighbors_array: [np.array, shape = (num_grid)]
                number of neighbors array
        OUTPUTS:
            fig, ax: [obj]
                figure and axis
        '''
        ## PLOTTING
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
#        ## CREATING PLOT
#        fig, ax = plot_funcs.create_plot()
#        
        ## ADDING LABELS
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Number of ligands within grid")
        
        ## GENERATING INDICES
        indices = np.arange(len(num_neighbors_array))
        
        ## PLOTTING
        ax.scatter(indices, 
                   num_neighbors_array, 
                   color = "k",
                   s = 2)
        
        return fig, ax
    
    
### FUNCTION TO GET THE MONOLAYER CHARACTERISTICS
def compute_monolayer_characteristics(ligand_names,
                                      traj_data,
                                      path_grid,
                                      ):
    '''
    The purpose of this function is to compute monolayer height difference to 
    the interface. NOTE: This only works for the planar case. If it is not 
    planar, many of these assumptions are broken!
    INPUTS:
        ligand_names: [list]
            list of ligand names in your monolayer
        traj_data: [obj]
            trajectory data object
        path_grid: [str]
            path to the grid wc interface
        
    OUTPUTS:
        monolayer_properties: [obj]
            class object of the monolayer properties
            contains: monolayer_properties.top_bot_dict -- which is the top and bottom distance to grid
    '''


    ### DEFINING INPUT DATA
    input_details = {
                        'ligand_names': ligand_names,      # Name of the ligands of interest
                        'structure_types': ['trans_ratio'],         # Types of structurables that you want , 'distance_end_groups' # 'trans_ratio'
                        'save_disk_space': False,                    # Saving space
                        'separated_ligands':True,
                        }
    
    ## RUNNING CLASS
    structure = nanoparticle_structure(traj_data, **input_details )
    
    
    ## GENERATING MONOLAYER PROPERTIES
    monolayer_properties = compute_monolayer_properties( traj_data = traj_data,
                                                         structure = structure)
    
    
    ## GETTING GRID DIFFERENCE
    monolayer_properties.compute_grid_difference(path_grid)
    '''
    top_bot_dict = monolayer_properties.top_bot_dict
    '''
    return monolayer_properties

### FUNCTION TO PLOT CONTOUR VS. Z-DISTANCE
def plot_fxn_vs_contour_levels(contour_storage_df,
                               contour_key = 'contour',
                               fxn_key = 'fraction_grid_points',
                               line_dict={'color': 'k'},
                               ylabel = "Fraction of ligands within the grid",
                               xlabel = "Contour level",
                               fig = None,
                               ax = None):
    '''
    The purpose of this function is to plot a function versus contour level. 
    INPUTS:
        contour_key: [str]
            contour key
        contour_storage_df: [df]
            dataframe containing storage information, e.g. 
               contour  fraction_grid_points       top       bot
        0     25.6              0.024545  0.338463  0.339960
        1     27.0              0.000000  0.371260  0.370718
            - Contour is the contour level
            - fraction of grid points is the fraction of grid points with overlapping ligands
            - top / bot is the z-distance from the top or bottom
        fxn_key: [key]
            key to plot the function
        line_dict: [dict]
            dictionary with the line information
    OUTPUTS:
        fig, ax: [obj]
            figure and axis for the image
    '''
    ## CREATING FUNCTION
    if fig is None or ax is None:
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
        ## ADDING LABELS
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    ## DEFINING X AND Y LABELS
    x = contour_storage_df[contour_key].to_numpy()
    y = contour_storage_df[fxn_key].to_numpy()
    
    ## PLOTTING
    ax.plot(x, y, **line_dict)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    return fig, ax


### FUNCTION TO PLOT THE DENSITY AND CONTOUR LEVEL
def plot_density_vs_z_with_contour(path_pickle,
                                   monolayer_properties,
                                   contour_level = None):
    '''
    The purpose of this function is to plot the density versus z for a 
    contour level.
    INPUTS:
        path_pickle: [str]
            path to the pickle with WC debugging
        monolayer_properties: [obj]
            monolayer properties object
        
    OUTPUTS:
        fig, ax:
            figure and axis for the image
    '''

    ## LOADING PICKLE
    import MDDescriptors.core.pickle_tools as pickle_tools
    from MDDescriptors.surface.willard_chandler_parallel import plot_wc_interface_across_z_dist
    
    ## FINDING CENTER
    z_top = monolayer_properties.end_group_z_positions[0, monolayer_properties.top_index]
    z_bot = monolayer_properties.end_group_z_positions[0, monolayer_properties.bot_index]
    
    ## AVGING Z-POSITIONS
    avg_z_top = np.mean(z_top)
    avg_z_bot = np.mean(z_bot)
    avg_z_top_bot = [ avg_z_top, avg_z_bot  ]
    ## GETTING CENTER
    center = np.mean( avg_z_top_bot )
    
    ## GETTING THE INTERFACE DATA
    data, interface, avg_density_field = pickle_tools.load_pickle(path_pickle)[0]
    
    ## PLOTTING WC INTERFACE
    fig, ax = plot_wc_interface_across_z_dist(avg_density_field = avg_density_field,
                                              interface = interface)
    
    ## PLOTTING CENTER
    ax.axvline(x = center, linestyle = '--', color='k', label='center')
    
    ## PLOTTING LIGAND END
    for idx, each_value in enumerate(avg_z_top_bot):
        if idx == 0:
            label = 'lig_end'
        else:
            label = None
        ## PLOTTING CENTER
        ax.axvline(x = each_value, linestyle = '--', color='gray', label=label)
    
    ## GETTING AVERAGE TOP GRID AND BOTTOM GRID
    avg_grid_top = np.mean(monolayer_properties.top_grid_z)
    avg_grid_bot = np.mean(monolayer_properties.bot_grid_z)
    avg_grid_top_bot = [avg_grid_top, avg_grid_bot]
    
    ## PLOTTING LIGAND END
    for idx, each_value in enumerate(avg_grid_top_bot):
        if idx == 0:
            if contour_level is not None:
                label = 'c=%.2f'%(contour_level)
            else:
                label = 'c'
        else:
            label = None
        ## PLOTTING CENTER
        ax.axvline(x = each_value, linestyle = '--', color='blue', label=label)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## ADD LABEL
    ax.legend()
    
    return fig, ax

    
#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    MAIN_SIM_FOLDER=PARENT_SIM_PATH
    # r"S:\np_hydrophobicity_project\simulations"
    
    ## DEFINING LOCATION OF STORAGE
    store_image_location = IMAGE_LOCATION
    # r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200316\images\check_overlap"
    # r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200120\check_overlap_images"
    
    
    ## DEFINING SIMULATION LIST
    simulation_list = [ 
                        '20200403-Planar_SAMs-5nmbox_vac_with_npt_equil',
#                        '20200401-Renewed_GNP_sims_with_equil',
#                        '20200411-mixed_sam_frozen',
                        ]
    
    ## DEFINING CONTOURS
    contour_levels = [26]
            # 0.80] # 30
    # 0.70
    # 27
    # 25.6, 30, 30.5
    #  31, 32,
    # "30" "31" "32" "32.5"  32.5
    # [25.6, 27, 30, 32, 32.5]
    # [25.6, 27, 30, 32, 32.5] # , 33
    
    ## DEFINING DEBUG PICKLE
    debug_pickle_name="debug.pickle"
    
    ## DEFINING GRO AND XTC
    prefix="sam_prod_2000_50000-heavyatoms"
    # "sam_prod_2000_50000-heavyatoms"
    # "sam_prod_0_50000-heavyatoms"
    structure_file = prefix + ".gro"
    xtc_file = prefix + ".xtc"

    ## DEFINING CUTOFF
    cutoff_radius = 0.33
    # 0.25
    # 0.33
    
    ## FRAME
    frame = 0
    
    ## DEFINING SUFFIX
    suffix="-2000-50000-wc_45000_50000"
    
    ## DEFINING LIGANDS
    ligands = [ 
                "dodecanethiol",
                "C11OH",
                "C11NH2",
                "C11COOH",
                "C11CONH2",
                "C11CF3",   
#                'C11COO,C11COOH',
#                'C11NH3,C11NH2',

                ]
    
    ## DEFINING GRID
    grid_folder_name = "grid-45000_50000"
    # "grid-49000_50000"
    ## DEFINING PREFIX
    contour_prefix=""
    # "norm-"

    ## DEFINING SIMULATION NAME
    for simulation_dir in simulation_list:    
        ## DEFINING PREFIX
        sim_prefix="FrozenPlanar_300.00_K_"
        sim_suffix="_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
        
        sim_prefix="MostlikelynpNVT_EAM_300.00_K_2_nmDIAM_"
        sim_suffix="_CHARMM36jul2017_Trial_1_likelyindex_1"
        
        ## DEFINING CONTOUR SUFFIX
        if simulation_dir == "20200224-planar_SAM_spr50" \
            or simulation_dir == '20200328-Planar_SAMs_new_protocol-shorterequil_spr50' \
            or simulation_dir == '20200403-Planar_SAMs-5nmbox_vac_with_npt_equil':
            contour_suffix =  "-0.24-0.1,0.1,0.1-" + str(cutoff_radius) +  "-all_heavy" + suffix
            # "-all_heavy-0-150000"
            is_planar = True
        else:
            contour_suffix = "-0.24-0.1,0.1,0.1-" + str(cutoff_radius) + "-all_heavy" + suffix
            is_planar = False
        
        if is_planar is True:
        # simulation_dir == "20200224-planar_SAM_spr50":
            sim_prefix="NVTspr_50_Planar_300.00_K_"
            sim_suffix="_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps"
        elif simulation_dir == "20200224-GNP_spr_50" or simulation_dir == '20200401-Renewed_GNP_sims_with_equil':
            sim_prefix="MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_"
            sim_suffix="_CHARMM36jul2017_Trial_1_likelyindex_1"
        elif simulation_dir == "20200411-mixed_sam_frozen":
            sim_prefix="MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_"
            sim_suffix="_0.53,0.47_CHARMM36jul2017_Trial_1_likelyindex_1"
        
        ## DEFINING VECTOR 
        lig_storage = {}
        
        ## DEFINING COLORS
        cmap_colors = plot_funcs.get_cmap(len(ligands))
            
        ## PLOTTING
        fig_frac_vs_contour, ax_frac_vs_contour = None, None
        fig_top_vs_contour, ax_top_vs_contour = None, None
        
        ## LOOPING THROUGH SPECIFIC SIM
        for lig_index, current_lig_name in enumerate(ligands):
            # if lig_index == 0:
        
            ## DEFINING SIMULATION NAME
            specific_sim = sim_prefix + current_lig_name + sim_suffix
        
            ## DEFINING PATHS
            path_sim = os.path.join(MAIN_SIM_FOLDER,
                                    simulation_dir,
                                    specific_sim)
    
            ## IMPORTING TRAJECTORIES
            traj_data = import_tools.import_traj(directory = path_sim,
                                                 structure_file = structure_file,
                                                 xtc_file = xtc_file,
                                                 index = frame)
            
            ## DEFINING STORAGE CONTOUR
            contour_storage = []
            
            ## LOOPING THROUGH EACH CONTOUR
            for cont_idx, contour in enumerate(contour_levels):
            
                ## DEFINING LOCATION OF CONTOUR
                contour_analysis = contour_prefix + "%.0f"%(contour) + contour_suffix
                
                ## DEFINING GRID PATH
                relative_grid_path = os.path.join(contour_analysis, 
                                                  grid_folder_name,
                                                  "out_willard_chandler.dat")
                
                ## DEFINING PATH TO GRID
                path_grid = os.path.join(path_sim,
                                         relative_grid_path)
                if os.path.exists(path_grid):
                    ## LOADING GRID
                    grid = load_datafile(path_grid)
                    
                    ## DEFINING PATH TO PICKLE
                    path_pickle = os.path.join(path_sim,
                                               contour_analysis,
                                               grid_folder_name,
                                               debug_pickle_name)
        
                    
                    ## RUNNING LIGAND OVERLAP FUNCTION
                    lig_overlap = check_lig_overlap( traj_data = traj_data,
                                                     cutoff_radius = cutoff_radius)
                    
                    ## GETTING NEIGHBORS
                    num_neighbors_array = lig_overlap.compute_neighbors(traj = traj_data.traj,
                                                                        grid = grid,
                                                                        neighbor_search="kdtree")
                
                    ## FINDING AVERAGE NEIGHBOR ARRAY
                    avg_neighbor_array = np.mean(num_neighbors_array,axis=1)
                
                    ## COMPUTING FRACTION OF GRID POINTS WITH LIGANDS
                    fraction_grid_points = np.sum(avg_neighbor_array>0) / avg_neighbor_array.size
                    
                    ## STORING
                    contour_storage_dict = {
                            'contour': contour,
                            'fraction_grid_points': fraction_grid_points,
                            }
                    
                                        
                    ##### PLOTTING FIGURES
                    fig_name_suffix = "_overlap" + "_" + str(contour)
                    ## FOR PLANAR SIMS
                    if is_planar is True:
                        ## GETING MONOLAYER CHARACTERISTICS
                        monolayer_properties = compute_monolayer_characteristics(ligand_names = lig_overlap.ligand_names,
                                                                                  traj_data = traj_data,
                                                                                  path_grid = path_grid
                                                                                  )
                        ## DEFINING MONOLAYER PROPERTIES
                        top_bot_dict = monolayer_properties.top_bot_dict
                        
                        
                        ## ADD TO DICT
                        contour_storage_dict = {**contour_storage_dict, **top_bot_dict}
                        
                        ## PLOTTING FIGURE FOR DENSITY
                        if os.path.exists(path_pickle) is True:
                            fig, ax = plot_density_vs_z_with_contour(path_pickle = path_pickle,
                                                                       monolayer_properties = monolayer_properties,
                                                                       contour_level = contour)
                            ## DEFINING FIGURE NAME
                            fig_name=specific_sim + "_density" + fig_name_suffix
                            plot_funcs.store_figure(fig = fig, 
                                                    path = os.path.join(store_image_location,
                                                                        fig_name), 
                                                    fig_extension = 'png', 
                                                    save_fig=save_fig, 
                                                    dpi=600)
                        
                        ## PLOTTING OTHER DETAILS
                        ## TILT ANGLE PLOT
                        fig, ax = monolayer_properties.plot_tilt_angle_dist()
                        plot_funcs.save_fig_png(fig = fig, 
                                                label =  os.path.join(store_image_location,
                                                                      specific_sim + "-tilt_angle"), 
                                                save_fig = save_fig)
                        
                        ## MONOLAYER HEIGHT
                        fig, ax = monolayer_properties.plot_monolayer_height()
                        plot_funcs.save_fig_png(fig = fig, 
                                                label =  os.path.join(store_image_location,
                                                                      specific_sim + "-height"),
                                                save_fig = save_fig)
                        
                        ## GRID DIFERENCE
                        fig, ax = monolayer_properties.plot_bar_grid_position_difference()
                        plot_funcs.save_fig_png(fig = fig, label =  os.path.join(store_image_location,
                                                                      specific_sim + "-grid_diff"),
                                                save_fig = save_fig)
                    
                    
                    ## STORING CONTOUR 
                    contour_storage.append(contour_storage_dict)
                    ## PLOTTING NUMBER OF LIGANDS PER GRID POINT
                    fig, ax = lig_overlap.plot_num_ligands_within_grid(avg_neighbor_array)
                    
                    ## TIGHT LAYOUT
                    fig.tight_layout()
    
                    ## DEFINING FIG NAME
                    fig_name=specific_sim + "_numwithin" + fig_name_suffix
                    plot_funcs.store_figure(fig = fig, 
                                            path = os.path.join(store_image_location,
                                                                fig_name), 
                                            fig_extension = 'png', 
                                            save_fig=save_fig, 
                                            dpi=600)
                    
                    ## DEFINING STORAGE
                    lig_storage[current_lig_name] = {
                            'fraction_grid_points' : fraction_grid_points
                            }
                    
                    ### PLOTTING MAYAVI 
                    
                    ## PLOTTING MAYAVI
                    fig = plot_funcs.plot_intersecting_points(grid = grid,
                                                   avg_neighbor_array = avg_neighbor_array)
                    
                    ## FIGURE FROM 
                    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                              atom_index = lig_overlap.atom_index,
                                              frame = 0,
                                              figure = fig,
                                              dict_atoms = plot_funcs.ATOM_DICT,
                                              dict_colors = plot_funcs.COLOR_CODE_DICT)
                    
                    ## PLOTTING GOLD ATOMS
                    au_index = [atom.index for atom in traj_data.topology.atoms if atom.name == 'Au' or atom.name == 'BAu']
                 
                    ## PLOTTING GOLD FIGURE
                    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                                                       atom_index = au_index,
                                                       frame = 0,
                                                       figure = fig,
                                                       dict_atoms = plot_funcs.ATOM_DICT,
                                                       dict_colors = plot_funcs.COLOR_CODE_DICT)
                    ## DEFINIGN FIGURE NAME        
                    fig_name=specific_sim + "_mayavi" + fig_name_suffix
    
                    ## SAVING FIGURE
                    if save_fig is True:
                        mlab.savefig(os.path.join(store_image_location,
                                                  fig_name + '.png'))
                        
            ## AFTER CONTOUR LEVEL LOADING
            ## CREATING DATAFRAME
            contour_storage_df = pd.DataFrame(contour_storage)
                
            ## PLOTTING CONTOUR LEVELS
            line_dict = {'color': cmap_colors(lig_index),
                         'label': current_lig_name}
            
            ## PLOTTING FRACTION VERSUS CONTOUR
            fig_frac_vs_contour, ax_frac_vs_contour = plot_fxn_vs_contour_levels(contour_storage_df,
                                               contour_key = 'contour',
                                               fxn_key = 'fraction_grid_points',
                                               line_dict=line_dict,
                                               ylabel = "Fraction of ligands within the grid",
                                               xlabel = "Contour level",
                                               fig = fig_frac_vs_contour,
                                               ax = ax_frac_vs_contour)
            ## FOR PLANAR SIMS
            if is_planar is True:
                ## PLOTTING TOP Z DISTANCE
                fig_top_vs_contour, ax_top_vs_contour = plot_fxn_vs_contour_levels(contour_storage_df,
                                                   contour_key = 'contour',
                                                   fxn_key = 'top',
                                                   line_dict=line_dict,
                                                   ylabel = "Distance to top z-grid (nm)",
                                                   xlabel = "Contour level",
                                                   fig = fig_top_vs_contour,
                                                   ax = ax_top_vs_contour)
    
        ## FOR PLANAR SIMS
        if is_planar is True:
            ## ADDING Z DISTANCE TO THE TOP
            ax_top_vs_contour.axhline(y = cutoff_radius, linestyle='--', color='k', label='Probe cutoff')
            ax_top_vs_contour.legend()
        
            ## FIGURE FOR OVERLAP AND TOP
            fig_name=simulation_dir + '_check_ligand_overlap-top_vs_contour'
            plot_funcs.store_figure(fig = fig_top_vs_contour, 
                                    path = os.path.join(store_image_location,
                                                        fig_name), 
                                    fig_extension = 'png', 
                                    save_fig=True, 
                                    dpi=600)
        
        ## ADDING Z DISTANCE TO THE TOP
        ax_frac_vs_contour.axhline(y = 0.00, linestyle='--', color='k', )
        
        ## AFTER EVERYTHING, ADD LEGEND
        ax_frac_vs_contour.legend()
        
        
        ## STORING FIGURE
        fig_name=simulation_dir + '_check_ligand_overlap-frac_vs_contour'
        plot_funcs.store_figure(fig = fig_frac_vs_contour, 
                                path = os.path.join(store_image_location,
                                                    fig_name), 
                                fig_extension = 'png', 
                                save_fig=True, 
                                dpi=600)

            
            
    
    #%%


    
#
#    #%%
#    
#    
#    ## PLOTTING PICKLE FILE
#    path_to_grid=r"S:/np_hydrophobicity_project/simulations/20200224-planar_SAM_spr50/NVTspr_50_Planar_300.00_K_C11COOH_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps/30-0.24-0.1,0.1,0.1-0.33-all_heavy-0-150000/grid-0_1000"
#    
#
#    
#    ## PATH TO PICKLE
#    path_pickle = os.path.join(path_to_grid,
#                               debug_pickle_name)
#    
#    
#
#    
#    ## PLOTTING FIGURE
#    fig, ax = plot_density_vs_z_with_contour(path_pickle = path_pickle,
#                                               monolayer_properties = monolayer_properties,
#                                               contour_level = 2)
#    
#    
#    
#    #%%
#
#    
#    ## DEFINING PREFIX
#    # fig_prefix = os.path.join(path_save_fig,job_type)
#    
#    ## TILT ANGLE PLOT
#    fig, ax = monolayer_properties.plot_tilt_angle_dist()
#    ## MONOLAYER HEIGHT
#    fig, ax = monolayer_properties.plot_monolayer_height()
#    ## GRID DIFERENCE
#    fig, ax = monolayer_properties.plot_bar_grid_position_difference()
#    plot_tools.save_fig_png(fig = fig, 
#                            label = fig_prefix + "-tilt_angle", 
#                            save_fig = save_fig)
#    
#    ## MONOLAYER HEIGHT
#    fig, ax = monolayer_properties.plot_monolayer_height()
#    plot_tools.save_fig_png(fig = fig, label = 
#                            fig_prefix + "-height", 
#                            save_fig = save_fig)
#    
#    ## GRID DIFERENCE
#    fig, ax = monolayer_properties.plot_bar_grid_position_difference()
#    plot_tools.save_fig_png(fig = fig, label = 
#                            fig_prefix + "-grid_diff", 
#                            save_fig = save_fig)
#
#    
    
    #%%
    
#    ## CREATING PLOT
#    fig, ax = plot_funcs.create_plot()
#    objects = list(lig_storage.keys())
#    y_pos = np.arange(len(objects))
#    performance = [ lig_storage[each_object]['fraction_grid_points'] for each_object in objects]
#    
#    ## PLOTTING BAR PLOT
#    ax.bar(y_pos, performance, color = 'k', align='center', alpha=1)
#    plt.xticks(y_pos, objects)
#    ax.set_xlabel("Ligands")
#    ax.set_ylabel("Fraction of ligand overlap on grid")
#    
#    ## DEFINING FIG NAME
#    fig_name="Fraction_of_lig_overlap_np"
#    plot_funcs.store_figure(fig = fig, 
#                            path = os.path.join(store_image_location,
#                                                fig_name), 
#                            fig_extension = 'png', 
#                            save_fig=save_fig, 
#                            dpi=600)
#                            
                            
    #%%
    
    
    
    
    
    
    
    
    
