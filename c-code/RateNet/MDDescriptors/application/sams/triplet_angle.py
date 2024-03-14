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
from MDDescriptors.core.calc_tools import compute_displacements
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
def compute_hydration_decay_profile( sim_working_dir = "None",
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
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_hydration_decay_profile.pkl" )
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
            print( "\nCOMPUTING HYDRATION DECAY PROFILE WITH REFERENCE POSITION: %.3f nm" % z_ref )
            distribution = H( traj     = traj,
                                                        z_ref    = z_ref,
                                                        **kwargs )
            ## SAVE PICKLE FILE
            save_pkl( distribution, path_pkl )

### FUNCTION TO COMPUTE HBONDS
def triplet_angles_distribution( traj,
                                 z_ref        = 0.0,
                                 z_cutoff     = 0.3,  # Shell ACS Nano used 0.5, INDUS uses 0.3 nm cavity
                                 r_cutoff     = 0.33, # cutoff from Head-Gordon 1993 paper
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
    ## DEFINING INPUTS FOR COMPUTE
    kwargs = { 'z_ref'        : z_ref,
               'z_cutoff'     : z_cutoff,
               'r_cutoff'     : r_cutoff,
               'periodic'     : periodic,
               'residue_list' : residue_list,
               'verbose'      : verbose,
               'print_freq'   : print_freq,
             }
    ## EXTRACT HEAVY ATOM INDICES IN RESIDUE LIST
    atom_indices = np.array([ atom.index for atom in traj.topology.atoms 
                              if atom.residue.name in residue_list 
                              and atom.element.symbol != "VS" 
                              and atom.element.symbol != "H" ])    
    ## REDUCE TRAJECTORY TO ONLY ATOMS OF INVOLVED IN CALCULATION
    traj.atom_slice( atom_indices, inplace = True )    
    ## DEFINING INTERFACE
    triplet_angle_object = triplet_angle( **kwargs )    
    ## PARALELLIZING CODE
    if n_procs != 1:
        ## GETTING TRIPLET ANGLES
        p_theta = parallel_analysis_by_splitting_traj( traj = traj, 
                                                       class_function = triplet_angle_object.compute, 
                                                       n_procs = n_procs,
                                                       combine_type="sum",
                                                       want_embarrassingly_parallel = False )
    else:
        ## GETTING TRIPLET ANGLES
        p_theta = triplet_angle_object.compute( traj = traj )   
    ## NORMALIZE HISTOGRAM
    theta = triplet_angle_object.theta
    p_theta = p_theta / np.trapz( p_theta, x = triplet_angle_object.theta )
    distribution = np.array([ theta, p_theta ]).transpose()    
    ## OUTPUTTING TIME
    timer.time_elasped()
    ## RETURN RESULTS    
    return distribution

class triplet_angle:
    r'''
    Class to compute water triplet angles
    '''
    ## INITIALIZING
    def __init__( self, 
                  z_ref = 0.0, 
                  z_cutoff = 0.3,
                  r_cutoff = 0.33,
                  periodic = True,
                  residue_list = ['HOH'],
                  verbose = True, 
                  print_freq = 100,
                  theta_range = ( 0., 180. ),
                  bin_width = 2.,
                  ):        
        ## STORING INPUT VARIABLES
        self.z_ref = z_ref
        self.z_cutoff = z_cutoff
        self.r_cutoff = r_cutoff
        self.periodic = periodic
        self.residue_list = residue_list
        self.verbose = verbose
        self.print_freq = print_freq        
        ## DETERMINING HISTOGRAM SIZE/BINS
        self.theta_range = theta_range
        self.n_bins = np.ceil( theta_range[-1] / bin_width ).astype('int')
        self.theta = np.arange( theta_range[0], theta_range[-1], bin_width )        
        return

    ### FUNCTION TO COMPUTE TRIPLET ANGLE OF ATOMS IN FRAME
    def calc_triplet_angle( self, vector ):
        r'''
        '''
        theta = []
        magnitude = np.sqrt( np.sum( vector**2., axis = 1 ) )
        n_neighbors = np.sum( magnitude < self.r_cutoff )
        sorted_neighbors = magnitude.argsort()[:n_neighbors]
        for nn, ii in enumerate( sorted_neighbors[:-1] ):
            for jj in sorted_neighbors[nn+1:]:
                cos_theta = np.sum( vector[ii,:] * vector[jj,:] ) / magnitude[ii] / magnitude[jj]
                # ADJUST FOR ROUNDING ERROR
                cos_theta = np.round( cos_theta, 4 )                
                theta.append( np.rad2deg( np.arccos( cos_theta ) ) )            
        return theta
    
    ### FUNCTION TO COMPUTE DENSITY FOR A SINGLE FRAME
    def compute_single_frame( self, 
                              traj, 
                              frame = 0, ):
        r'''
        The purpose of this function is to compute the triplet angles for a single 
        frame. 
        INPUTS:
            traj: [md.traj]
                trajectory object
            frame: [int]
                frame that you are interested in
        OUTPUTS:
            theta: [list]
                water triplet angles
        '''
        ## REDUCE TRAJ TO SINGLE FRAME
        traj = traj[frame]                        
        ## GET WATER OXYGEN INDICES (ASSUMES ATOM SLICED TRAJ OF ONLY OXYGENS)
        atoms = [ atom for atom in traj.topology.atoms ]
        atom_indices = np.array([ atom.index for atom in atoms ])        
        ref_coords = [ traj.unitcell_lengths[0,0], traj.unitcell_lengths[0,1], self.z_ref ]       
        z_dist = compute_displacements( traj,
                                        atom_indices = atom_indices,
                                        box_dimensions = traj.unitcell_lengths,
                                        ref_coords = ref_coords,
                                        periodic = self.periodic )[:,2]        
        ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
        mask = z_dist < self.z_cutoff
        mask_sliced = z_dist < self.z_cutoff + 1.05*self.r_cutoff
        target_atom_indices = atom_indices[mask]
        sliced_atom_indices = atom_indices[mask_sliced]        
        ## COMPUTE ATOM-ATOM CARTESIAN DISTANCES
        atom_pairs = np.array([ [ aa, target_atom_indices[0] ] for aa in sliced_atom_indices if aa != target_atom_indices[0] ])
        dist_vector = md.compute_displacements( traj, 
                                                atom_pairs = atom_pairs,
                                                periodic = self.periodic ).squeeze()        
        ## COMPUTE TRIPLET ANGLES
        theta = []
        theta += self.calc_triplet_angle( dist_vector )        
        ## REPEAT FOR EACH ATOM        
        for atom_ndx in target_atom_indices[1:]:
            atom_pairs = np.array([ [ aa, atom_ndx ] for aa in sliced_atom_indices if aa != atom_ndx ])
            dist_vector = md.compute_displacements( traj, 
                                                    atom_pairs = atom_pairs,
                                                    periodic = self.periodic ).squeeze()                
            theta += self.calc_triplet_angle( dist_vector )        
        p_theta = np.histogram( theta, bins = self.n_bins, range = self.theta_range )[0]        
        return p_theta
            
    ### FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, 
                 traj,
                 frames = [],
                 ):
        r'''
        The purpose of this function is to compute the interfacial water rdf for a single 
        frame. 
        INPUTS:
            traj: [md.traj]
                trajectory object
            frame: [list]
                list of frames that you are interested in
        OUTPUTS:
            r:  [np.array, shape=(300,)]
                radial distance from group
            g_r: [np.array, shape=(300,)]
                 density at distance from group
        '''
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[tuple(frames)]
        ## DEFINING TOTAL TRAJECTORY SIZE
        total_traj_size = traj.time.size
        if self.verbose is True:
            if len(frames) == 0:
                print("--- Calculating density field for %s simulations windows ---" % (str(total_traj_size)) )                
        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for frame in np.arange(0, total_traj_size):
            if frame == 0:
                ## GETTING TRIPLET ANGLES FOR FIRST FRAME
                sum_p_theta = self.compute_single_frame( traj = traj,
                                                         frame = 0 )
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                p_theta = self.compute_single_frame( traj = traj,
                                                     frame = frame )
                sum_p_theta += p_theta
            ## PRINTING 
            if traj.time[frame] % self.print_freq == 0:
                print("====> Working on frame %d"%(traj.time[frame]))                            
        return sum_p_theta    
