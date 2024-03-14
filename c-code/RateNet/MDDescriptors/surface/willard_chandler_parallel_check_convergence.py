# -*- coding: utf-8 -*-
"""
willard_chandler_parallel_check_convergence.py
This functions checks the convengence of WC interface using a parallel code. 
This uses all previously generated code from willard_chandler_parallel.py 
and tries to parallelize.

Author: Bradley C. Dallin (01/07/2020)
"""

## IMPORTING MODULES
import os
import sys
import pickle
import mdtraj as md
import numpy as np

if r"R:\bin\python_modules" not in sys.path:
    sys.path.append( r"R:\bin\python_modules" )

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## IMPORTING FUNCTIONS
from MDDescriptors.surface.willard_chandler_check_convergence import wc_interface

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

## IMPORTONG TRACKING TIME
from MDDescriptors.core.track_time import track_time

### FUNCTION TO COMPUTE GRID
def compute_wc_grid(traj, 
                    sigma,
                    mesh, 
                    contour,
                    n_procs = 1,
                    residue_list = ['HOH'],
                    print_freq = 1):
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
            contour level that is desired
        n_procs: [int]
            number of processors desired
        residue_list: [list]
            list of residues to look for when generating the WC surface
        print_freq: [int]
            frequency for printint output
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
        ## GETTING DENSITY FIELD
        density_field = parallel_analysis_by_splitting_traj(traj = traj, 
                                                            class_function = interface.compute, 
                                                            n_procs = n_procs,
                                                            combine_type="concatenate_axis_0")      
    else:
        ## GETTING AVERAGE DENSITY FIELD
        density_field = interface.compute(traj = traj)
           
    ## OUTPUTTING TIME
    timer.time_elasped()
    return interface, density_field


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    ## RECOMPUTE DENSITY FIELD
    recompute = False
    analysis = True
    n_procs = 1
    ## TEST CONVERGENCE OR EQUILIBRATION
    compare_ensemble_avg = True
    check_equilibration = True
    converge_time = 200
    check_convergence = True
    equil_time = 0
    check_step_size = 10
    ## DEFINING sigma, CONTOUR, AND MESH
    sigma = WC_DEFAULTS['sigma']
    contour = 16. # WC_DEFAULTS['contour']
    ## DEFINING MESH
    mesh = WC_DEFAULTS['mesh']
    #%%
    ##########################
    ### LOADING TRAJECTORY ###
    ##########################
    ## DEFINING MAIN SIMULATION
    main_sim=r"/home/bdallin/simulations/polar_sams/indus/2x2x0.3nm/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36_2x2x0.3nm"
    # main_sim=r"R:\simulations\polar_sams\indus\2x2x0.3nm\sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36_2x2x0.3nm"
    ## DEFINING SIM NAME
    sim_name=r"equil"
    ## DEFINING WORKING DIRECTORY
    wd = os.path.join(main_sim, sim_name)
    ## DEFINING GRO AND XTC
    gro_file = r"sam_wc.gro"
    xtc_file = r"sam_wc.xtc"
    pkl_file = r"density_field.pkl"
    ensem_file = r"willard_chandler_ensemble_average.csv"
    equil_file = r"willard_chandler_equilibration.csv"
    converg_file = r"willard_chandler_convergence.csv"
    ## DEFINING PATHS
    path_gro = os.path.join(wd, gro_file)
    path_xtc = os.path.join(wd, xtc_file)
    path_pkl = os.path.join(main_sim + '/output_files', pkl_file)
    path_ens = os.path.join(main_sim + '/output_files', ensem_file)
    path_eq = os.path.join(main_sim + '/output_files', equil_file)
    path_conv = os.path.join(main_sim + '/output_files', converg_file)
    
    #%%
    ## GETTING INTERFACE POINTS
    if recompute is True:
        ## PRINTING
        print("Loading trajectory")
        print(" --> XTC file: %s"%(path_xtc) )
        print(" --> GRO file: %s"%(path_gro) )
        ## LOADING TRAJECTORY
        traj = md.load(path_xtc, top = path_gro)
        # traj = traj[0:10]
    
        interface, density_field = compute_wc_grid( traj = traj, 
                                         sigma = sigma, 
                                         mesh = mesh, 
                                         contour = contour,
                                         n_procs = n_procs,
                                         residue_list = ['HOH'],
                                         print_freq = 10 )
        
        print( "Writing pickle to {:s}".format( path_pkl ) )
        to_pkl = { 'interface': interface, 'density_field': density_field, 'box': traj.unitcell_lengths[0,:] }
        with open( path_pkl, 'wb') as handle:
            pickle.dump( to_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL )
        
    #%%
    with open( path_pkl, 'rb' ) as handle:
        from_pkl = pickle.load(handle)
        interface = from_pkl['interface']
        df = np.array(from_pkl['density_field'])
        box = from_pkl['box']
        density_field = np.array(df[0])
        for dd in df[1:]:
            density_field = np.vstack(( density_field, dd ))

    if analysis is True:      
        ## DETERMINE ENSEMBLE AVERAGE
        if compare_ensemble_avg is True:
            z_ensemble = []
            for ii in range(density_field.shape[0]):
                interface_points = interface.get_wc_interface_points(density_field = density_field[ii,:], 
                                                                      contour = contour)
                points_on_monolayer = interface_points[interface_points[:,2]<4.5,:]
                
                dist_sq_center = ( points_on_monolayer[:,0] - box[0] )**2. + ( points_on_monolayer[:,1] - box[1] )**2.
                closest_ndx = np.nanargmin( dist_sq_center )    
                z_ensemble.append( points_on_monolayer[closest_ndx,2] )
    
            z_ensemble = np.array( z_ensemble )
            with open( path_ens, 'w+' ) as ens_file:
                ens_file.write( '<z>={:.2f}\n'.format( np.mean(z_ensemble) ) )
                ens_file.write( "frame,z\n")
                for ii in range(len(z_ensemble)):
                    ens_file.write( '{:.1f},{:.3f}\n'.format( ii, z_ensemble[ii] ) )
        
        ## CHECK WC INTERFACE EQUILIBRATION
        if check_equilibration is True:
            frames = np.arange( 0, density_field.shape[0] - converge_time, check_step_size )
            z_equil = []
            for n in frames:
                avg_density_field = density_field[0:n+converge_time,:].mean(axis=0)
                interface_points = interface.get_wc_interface_points(density_field = avg_density_field, 
                                                                     contour = contour)
                points_on_monolayer = interface_points[interface_points[:,2]<4.5,:]
                
                dist_sq_center = ( points_on_monolayer[:,0] - box[0] )**2. + ( points_on_monolayer[:,1] - box[1] )**2.
                closest_ndx = np.nanargmin( dist_sq_center )    
                z_equil.append( points_on_monolayer[closest_ndx,2] )
    
            z_equil = np.array( z_equil )
            with open( path_eq, 'w+' ) as eq_file:
                eq_file.write( "frame,z\n")
                for ii in range(len(frames)):
                    eq_file.write( '{:.1f},{:.3f}\n'.format( frames[ii], z_equil[ii] ) )
                
        ## CHECK WC INTERFACE CONVERGENCE
        if check_convergence is True:
            frames = np.arange( equil_time, density_field.shape[0], check_step_size )
            z_converge = []
            for n in frames:
                avg_density_field = density_field[equil_time:n+check_step_size,:].mean(axis=0)
                interface_points = interface.get_wc_interface_points(density_field = avg_density_field, 
                                                                     contour = contour)
                points_on_monolayer = interface_points[interface_points[:,2]<4.5,:]
                
                dist_sq_center = ( points_on_monolayer[:,0] - box[0] )**2. + ( points_on_monolayer[:,1] - box[1] )**2.
                closest_ndx = np.nanargmin( dist_sq_center )    
                z_converge.append( points_on_monolayer[closest_ndx,2] )
    
            z_converge = np.array( z_converge )
            with open( path_conv, 'w+' ) as conv_file:
                conv_file.write( "frame,z\n")
                for ii in range(len(frames)):
                    conv_file.write( '{:.1f},{:.3f}\n'.format( frames[ii]-equil_time, z_converge[ii] ) )
                    