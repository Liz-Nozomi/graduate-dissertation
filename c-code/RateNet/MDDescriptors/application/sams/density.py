# -*- coding: utf-8 -*-
"""
rdf.py
The purpose of this script is to compute the water density

CREATED ON: 02/21/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os
import numpy as np  # Used to do math functions
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

##############################################################################
## FUNCTIONS
##############################################################################
## FUNCTION TO COMPUTE DENSITY PROFILES
def compute_density_profile( sim_working_dir = "None",
                             input_prefix    = "None",
                             rewrite         = False,
                             **kwargs ):
    r'''
    Function loads gromacs trajectory and computes density profile
    '''
    if sim_working_dir  == "None" \
        or input_prefix == "None":
        print( "ERROR: missing inputs. Check inputs and try again" )
        return 1
    else:
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_density_profile.pkl" )
        if rewrite is not True and os.path.exists( path_pkl ):
            profile = load_pkl( path_pkl )
        else:
            ## PREPARE DIRECTORY FOR ANALYSIS
            path_to_sim = check_server_path( sim_working_dir )
            
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
            print( "\nCOMPUTING DENSITY PROFILE WITH REFERENCE POSITION: %.3f nm" % z_ref )
            profile = density( traj     = traj,
                               path_pkl = path_pkl,
                               z_ref    = z_ref,
                               **kwargs )
            ## SAVE PICKLE FILE
            save_pkl( profile, path_pkl )

### FUNCTION TO COMPUTE DENSITY
def density( traj,
             path_pkl,
             z_ref,
             use_com = True,
             periodic = True,
             residue_list = [ 'SOL', 'HOH' ],
             verbose = True,
             rewrite = False,
             bin_width = 0.005,
             z_range = ( -0.5, 2.0 ),
             bulk_density = 33.4,
             **kwargs ):
    r'''
    The purpose of this function is to compute water density profile. 
    INPUTS:
        traj: [md.traj]
            trajectory object
        z_ref: [float]
            reference position for the density
        periodic: [bool]
            apply periodic boundary conditions
        residue_list: [list]
            list of residues to look for when generating the density profile
        verbose: [bool]
            print out updates
        rewrite: [bool]
            rewrite pkl file
        bin_width: [float]
            histogram spacing
        z_range: [tuple]
            (min, max) bounds on density profile
        bulk_density: [float]
            default: bulk density of water
    OUTPUTS:
        profile: [np.array shape=(n,2)]
            density profile with first column distance from z_ref and second
            column density         
    '''
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
                                   and atom.element.symbol != "H" ])
    ## REDUCE TRAJECTORY TO ONLY ATOMS OF INVOLVED IN CALCULATION
    traj = traj.atom_slice( atom_indices, inplace = False )
    ## WATER ATOM INDICES
    atom_indices = np.array([ atom.index for atom in traj.topology.atoms 
                              if atom.residue.name in residue_list ])    
    ## COMPUTE Z DISTANCE FROM Z_REF
    ref_coords = [ 0.5*traj.unitcell_lengths[0,0], 0.5*traj.unitcell_lengths[0,1], z_ref ]
    z_dist = compute_displacements( traj,
                                    atom_indices = atom_indices,
                                    box_dimensions = traj.unitcell_lengths,
                                    ref_coords = ref_coords,
                                    periodic = periodic )[:,:,2]
    ## BIN ATOM POSITIONS
    n_bins = int( np.abs( z_range[1] - z_range[0] ) / bin_width )
    histo, dist = np.histogram( z_dist, bins = n_bins, range = z_range, )
    ## NORMALIZE DENSITY
    z = dist[:-1]
    volume = bin_width * traj.unitcell_lengths[0,0] * traj.unitcell_lengths[0,1]
    rho = histo / volume / traj.n_frames / bulk_density
    profile = np.array([ z, rho ]).transpose()
    return profile
