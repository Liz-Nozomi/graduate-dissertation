# -*- coding: utf-8 -*-
"""
willard_chandler_parallel.py
This functions debug the WC interface using a parallel code. This uses all previously 
generated code from willard_chandler_with_debugging.py and tries to parallelize.

Written by: Alex K. Chew (12/12/2019)
"""

## IMPORTING MODULES
import os
import mdtraj as md
import numpy as np

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## IMPORTING FUNCTIONS
from MDDescriptors.surface.willard_chandler import wc_interface

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

## IMPORTONG TRACKING TIME
from MDDescriptors.core.track_time import track_time

### FUNCTION TO PLOT Z DISTRIBUTION
import MDDescriptors.core.plot_tools as plot_funcs

### FUNCTION TO NORMALIZE DENSITY FIELD
def normalize_density_field(density_field):
    '''
    This function normalizes by the maximum of the density field
    INPUTS:
        density_field: [np.array]
            density field
    OUTPUTS:
        norm_density_field: [np.array]
            normalized density field
        maxima_value: [float]
            maximum that was used to normalize
    '''
    ## FINDING MAXIMA
    maxima_value = np.max(density_field)
    
    ## NORMALIZING
    norm_density_field = density_field / maxima_value

    return norm_density_field, maxima_value

### FUNCTION TO COMPUTE GRID
def compute_wc_grid(traj, 
                    sigma,
                    mesh, 
                    contour,
                    n_procs = 1,
                    residue_list = ['HOH'],
                    print_freq = 1,
                    want_normalize_c = False,
                    want_embarrassingly_parallel = True):
    '''
    The purpose of this function is to compute the WC grid in parallel. 
    INPUTS:
        traj: [md.traj]
            trajectory object
        sigma: [float]
            standard deviation of the gaussian distributions
        mesh: [list, shape = 3]
            list of mesh points
        contour: [float]
            contour level that is desired. If None, then we will not compute 
            the interface points -- it will simply be defined at the origin. 
        n_procs: [int]
            number of processors desired
        residue_list: [list]
            list of residues to look for when generating the WC surface
        print_freq: [int]
            frequency for printing output
        want_normalize_c: [logical]
            True if you want normalized contour levels from 0 to 1
        want_embarrassingly_parallel: [logical]
            True if you want embarrassingly parallel code (on by default)
    OUTPUTS:
        interface_points: [np.array, shape = (N, 3)]
            interfacial points
        interface: [obj]
            interfacial object
        avg_density_field: [np.array]
            average density field
    '''
    ## TRACKING TIME
    timer = track_time()
    ## DEFINING INPUTS FOR COMPUTE
    kwargs = {'traj'  : traj,
              'mesh'  : mesh,
              'sigma' : sigma,
              'residue_list': residue_list,
              'verbose': True, 
              'print_freq': print_freq, 
              'norm': False  }
    
    ## GETTING NORMALIZATION
    if n_procs == 1:
        kwargs['norm'] = True
    
    ## DEFINING INTERFACE
    interface = wc_interface(**kwargs)
    
    ## PARALELLIZING CODE
    if n_procs != 1:
        ## GETTING AVERAGE DENSITY FIELD
        avg_density_field = parallel_analysis_by_splitting_traj(traj = traj, 
                                                                class_function = interface.compute, 
                                                                n_procs = n_procs,
                                                                combine_type="sum",
                                                                want_embarrassingly_parallel = want_embarrassingly_parallel)
        ## NORMALIZING
        avg_density_field /= traj.time.size
    else:
        ## GETTING AVERAGE DENSITY FIELD
        avg_density_field = interface.compute(traj = traj)
    
    # NORMALIZE DENSITY FIELD IF CONTOUR DESIRED
    if want_normalize_c is True:
        norm_density_field, maxima_value = normalize_density_field(avg_density_field)
        print("Normalizing c contour with maxima value of: %.2f"%(maxima_value))
        density_field_to_find_interface = norm_density_field
    else:
        density_field_to_find_interface = avg_density_field
    
    ## AFTER COMPLETION, GET INTERFACE
    interface_points = interface.get_wc_interface_points(density_field = density_field_to_find_interface, 
                                                         contour = contour)
        
    ## OUTPUTTING TIME
    timer.time_elasped()
    return interface_points, interface, avg_density_field
    
### FUNCTION TO PLOT THE DISTRIBUTION
def plot_wc_interface_across_z_dist( avg_density_field, interface ):
    '''
    Parameters
    ----------
    avg_density_field : [np.array, shape=(num_density_points)]
        average density field
    interface : [class]
        class from compute_wc_grid

    Returns
    -------
    fig, ax: figure and axis for interface

    '''
    ## DEFINING FIGURE SIZE
    FIGURE_SIZE=plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']
    ## CREATING PLOT
    fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm = FIGURE_SIZE)
    
    ## ADDING LABELS
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("Density (num/nm$^3$)")

    ## GETTING RESHAPES
    avg_field_reshaped = avg_density_field.reshape(interface.num_grid_pts)
    
    ## AVERAGING ALONG THE X, Y AXIS
    avg_field_z = np.mean(np.mean(avg_field_reshaped, axis = 0),axis=0)
    
    ## GENERATING NUMPY ARRAY
    z_array = np.linspace(0, interface.box[-1], int(interface.num_grid_pts[-1]) )
 
    ## CREATING HISTOGRAM PLOT
    ax.plot(z_array, avg_field_z, color="k")
    
    return fig, ax


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING sigma, CONTOUR, AND MESH
    sigma = WC_DEFAULTS['sigma']
    contour = WC_DEFAULTS['contour']
    ## DEFINING MESH
    mesh = WC_DEFAULTS['mesh']
    #%%
    ##########################
    ### LOADING TRAJECTORY ###
    ##########################
    ## DEFINING MAIN SIMULATION
    main_sim=r"S:\np_hydrophobicity_project\simulations\191210-annealing_try4"
    ## DEFINING SIM NAME
    sim_name=r"FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
    ## DEFINING WORKING DIRECTORY
    wd = os.path.join(main_sim, sim_name)
    ## DEFINING GRO AND XTC
    gro_file = r"sam_prod-0_1000-watO_grid.gro"
    xtc_file = r"sam_prod-0_1000-watO_grid.xtc"
    ## DEFINING PATHS
    path_gro = os.path.join(wd, gro_file)
    path_xtc = os.path.join(wd, xtc_file)
    
    ## PRINTING
    print("Loading trajectory")
    print(" --> XTC file: %s"%(path_xtc) )
    print(" --> GRO file: %s"%(path_gro) )
    ## LOADING TRAJECTORY
    traj = md.load(path_xtc, top = path_gro)
    #%%
    
    ## GETTING INTERFACE POINTS
    interface_points, interface, avg_density_field = compute_wc_grid(traj = traj[0:10], 
                                                                     sigma = sigma, 
                                                                     mesh = mesh, 
                                                                     contour = contour,
                                                                     n_procs = 1,
                                                                     residue_list = ['HOH'],
                                                                     print_freq = 1,
                                                                     )
    #%%
    ## GETTING INTERFACE POINTS
    # interface_points_2, interface_2, avg_density_field_2 = compute_wc_grid(traj = traj[0:10], 
    #                                                                        sigma = sigma, 
    #                                                                        mesh = mesh, 
    #                                                                        contour = contour,
    #                                                                        n_procs = 2,
    #                                                                        residue_list = ['HOH'],
    #                                                                        print_freq = 1,
    #                                                                        )
    
    # ## TOTALLING
    # total = np.sum(interface_points_2 - interface_points)
    # if total != 0.0:
    #     print("Error! Something is not right between serial and parallel codes!")
    # # THIS VALUE SHOULD BE 0. OTHERWISE, THE MULTIPROCESSING DID NOT WORK!
    
    #%%
    
    ## GETTING WC INTERFACE
    interface_points = interface.get_wc_interface_points(density_field = avg_density_field, 
                                                         contour = contour)
    
    ## PLOTTING MLAB
    interface.mlab_plot_density_field(density_field = avg_density_field,
                                      interface_points = interface_points,
                                      num_grid_pts = interface.num_grid_pts,
                                      grid = interface.grid,
                                      pos = None, 
                                      )
    
    #%%
    
    

    
    ## CREATING FIG, AX
    fig, ax = plot_wc_interface_across_z_dist( avg_density_field = avg_density_field, 
                                               interface = interface)
    
    
        
