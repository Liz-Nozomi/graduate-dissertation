# -*- coding: utf-8 -*-
"""
rdf.py
The purpose of this script is to compute the interfacial water rdf

CREATED ON: 02/18/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT NUMPY
import numpy as np  # Used to do math functions
## IMPORT MDTRAJ
import mdtraj as md
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path
## IMPORT TRAJECTORY FUNCTION
from MDDescriptors.application.sams.trajectory import load_md_traj
## IMPORT COMPUTE DISPLACEMENT FUNCTION
from MDDescriptors.core.calc_tools import compute_displacements, compute_com
## IMPORT WILLARD-CHANDER HEIGHT FUNCTION
from MDDescriptors.application.sams.willard_chandler import compute_willad_chandler_height
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl
## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj
## IMPORTING TRACKING TIME
from MDDescriptors.core.track_time import track_time

##############################################################################
## CLASSES & FUNCTION
##############################################################################
def compute_hydration_residence_time( sim_working_dir = "None",
                                      input_prefix    = "None",
                                      rewrite         = False,
                                      **kwargs ):
    r'''
    Function loads gromacs trajectory and computes triplet angle distribution
    '''
    if sim_working_dir  == "None" \
        or input_prefix == "None":
        print( "ERROR: missing inputs. Check inputs and try again" )
        return 1
    else:
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_hydration_residence_time_profile.pkl" )
        if rewrite is not True and os.path.exists( path_pkl ):
            profile = load_pkl( path_pkl )
        else:
            ## PREPARE DIRECTORY FOR ANALYSIS
            path_to_sim = check_server_path( sim_working_dir )
            ## USE WC IF NOT OTHERWISE SPECIFIED
            if "z_ref" not in kwargs:
                ## GET Z_REF FROM WC GRID
                z_ref = compute_willad_chandler_height( path_wd = path_to_sim, 
                                                        input_prefix = input_prefix,
                                                        **kwargs )            
            ## LOAD MD TRAJECTORY
            traj = load_md_traj( path_traj    = sim_working_dir,
                                 input_prefix = input_prefix,
                                 **kwargs )
            ## COMPUTE DENSITY PROFILE
            print( "\nCOMPUTING HYDRATION LAYER RESIDENCE TIME WITH REFERENCE POSITION: %.3f nm" % z_ref )
            profile = hydration_residence_time_profile( traj     = traj,
                                                        z_ref    = z_ref,
                                                        **kwargs )
            ## SAVE PICKLE FILE
            save_pkl( profile, path_pkl )
        ## COMPUTE HYDRATION LAYER RESIDENCE TIME
        path_pkl_residence_time = os.path.join( sim_working_dir, r"output_files", input_prefix + "_hydration_residence_time.pkl" )
        path_pkl_residence_time_stats = os.path.join( sim_working_dir, r"output_files", input_prefix + "_hydration_residence_time_stats.pkl" )
        profile = load_pkl( path_pkl )
        ## ONLY LOOK AT FIRST 50 PS
        tau = fit_exp( profile[1:50,0], profile[1:50,1] )
        save_pkl( tau["tau"], path_pkl_residence_time )
        save_pkl( tau, path_pkl_residence_time_stats )        

### FUNCTION TO COMPUTE HBONDS
def hydration_residence_time_profile( traj,
                                      z_ref        = 0.0,
                                      z_cutoff     = 0.3,  # Shell ACS Nano used 0.5, INDUS uses 0.3 nm cavity
                                      n_procs      = 20,
                                      use_com      = True,
                                      periodic     = True,
                                      residue_list = [ 'SOL', 'HOH' ],
                                      verbose      = True,
                                      print_freq   = 100,
                                      rewrite      = False,
                                      **kwargs ):
    r'''
    The purpose of this function is to compute the triplet angle distribution
    in parallel.
    '''
    ## TRACKING TIME
    timer = track_time()
    if use_com is True:
        print( "--- USING COM OF GROUP ---" )
        ## GET ATOM GROUPS TO COMPUTE DENSITY
        residue_group_indices = [ [ atom.index for atom in residue.atoms ] 
                                    for residue in traj.topology.residues 
                                    if residue.name in residue_list ]
        atom_indices = np.array([ residue[0] for residue in residue_group_indices ])    
        ## UPDATE TRAJ SO HEAVY ATOM HAS COM POSITION
        for ndx, res_indices in zip( atom_indices, residue_group_indices ):
            traj.xyz[:,ndx,:] = compute_com( traj, res_indices )
    else:
        ## EXTRACT HEAVY ATOM INDICES IN RESIDUE LIST
        atom_indices = np.array([ atom.index for atom in traj.topology.atoms 
                                   if atom.residue.name in residue_list
                                   and atom.element.symbol == "O" ])
    ## GATHER Z POSITIONS OF WATER ATOMS
    z_water_positions = traj.xyz[:,atom_indices,2]
    ## DEFINE Z UPPER AND LOWER BOUNDS
    z_upper = z_ref + z_cutoff
    z_lower = z_ref - 0.5
    ## APPLY HEAVISIDE STEP FUNCTION (1 IF IN CAVITY, 0 OTHERWISE)
    theta_heaviside = np.logical_and( z_water_positions > z_lower, z_water_positions < z_upper )
    ## COMPUTE AUTOCORRELATION FUNCTION
    C_res = acf( theta_heaviside )
    ## OUTPUTTING TIME
    timer.time_elasped()
    # ## RETURN RESULTS    
    return np.array([ traj.time, C_res ]).transpose()

def acf( X ):
    r'''
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    ## GET NUM FRAMES
    n_frames = X.shape[0]
    norm_array = np.arange(n_frames, 0, -1)
    C = np.zeros(n_frames)
    nn = 0.
    for n_col in range(X.shape[1]):
        x = X[:,n_col]
        if x.std() > 0.:
            ## NORMALIZE INPUTS
            a = ( x - x.mean() ) / x.std() / x.size
            b = ( x - x.mean() ) / x.std()
            ## DETERMINE AUTOCORRELATION
            C += np.correlate( a, b, mode = 'full' )[-n_frames:]
            nn += 1.
    ## AVERAGE PROFILE
    # C /= X.sum(axis=1).mean()
    ## RETURN ACF
    return C / nn
        
def fit_exp( t, R ):
    r'''
    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    
    Returns
    -------
    None.
    '''
    log_R = np.log(R)
    p = np.polyfit( t, np.log(R), 1 )
    R_fit = np.polyval( p, t )
    MSE = np.sum( ( log_R - R_fit )**2. ) / R.size
    SS_tot = np.sum( ( log_R - log_R.mean() )**2. )
    SS_res = np.sum( ( log_R - R_fit )**2. )
    R_sq = 1 - SS_res / SS_tot
    print( "tau: %.2f" % ( -1. / p[0] ) )
    print( "RMSE: %.2f" % ( np.sqrt( MSE ) ) )
    print( "R^2: %.2f" % ( R_sq ) )
    return { "tau"  : -1. / p[0],
             "RMSE" : np.sqrt(MSE),
             "R2"   : R_sq }
    