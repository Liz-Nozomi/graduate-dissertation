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
import os, sys
import pickle
import numpy as np  # Used to do math functions
import mdtraj as md # Used for distance calculations
import MDDescriptors.core.import_tools as import_tools      # Loading trajectory details
from MDDescriptors.core.calc_tools import compute_displacements, compute_com # Loading calculation tools

## IMPORTING TRACKING TIME
from MDDescriptors.core.track_time import track_time
from MDDescriptors.core.initialize import checkPath2Server

## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

##############################################################################
## CLASSES & FUNCTION
##############################################################################
### FUNCTION TO COMPUTE HBONDS
def compute_interfacial_rdf( traj,
                             path_pkl,
                             z_ref = 0.0,
                             z_cutoff = 0.3,
                             n_procs = 20,
                             use_com = True,
                             periodic = True,
                             residue_list = [ 'SOL', 'HOH' ],
                             verbose = True,
                             print_freq = 100,
                             rewrite = False,
                             **kwargs ):
    R'''
    The purpose of this function is to compute the interfacial water rdf in parallel. 
    '''
    if os.path.exists( path_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
            data = load_pkl( path_pkl )
            r = data[0]
            g_r = data[1]
    else:
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
                                      and atom.element.symbol != "VS" 
                                      and atom.element.symbol != "H" ])

        ## REDUCE TRAJECTORY TO ONLY ATOMS OF INVOLVED IN CALCULATION
        traj = traj.atom_slice( atom_indices, inplace = False )

        ## DEFINING INPUTS FOR COMPUTE
        kwargs = { 'z_ref'        : z_ref,
                   'z_cutoff'     : z_cutoff,
                   'periodic'     : periodic,
                   'residue_list' : residue_list,
                   'verbose'      : verbose,
                   'print_freq'   : print_freq,
                 }
        
        ## DEFINING INTERFACE
        rdf = interfacial_rdf( **kwargs )
        
        ## PARALELLIZING CODE
        if n_procs != 1:
            ## GETTING AVERAGE DENSITY FIELD
            g_r = parallel_analysis_by_splitting_traj( traj = traj, 
                                                       class_function = rdf.compute, 
                                                       n_procs = n_procs,
                                                       combine_type="sum",
                                                       want_embarrassingly_parallel = True)
            g_r /= traj.n_frames
        else:
            ## GETTING AVERAGE DENSITY FIELD
            g_r = rdf.compute( traj = traj )
                    
        ## OUTPUTTING TIME
        timer.time_elasped()
        
        r = np.arange( 0, 2.0, 0.005 )
        
        save_pkl( [ r, g_r ], path_pkl )
    
def load_pkl( path_pkl ):
    r'''
    Function to load data from pickle file
    '''
    print( "LOADING PICKLE FROM %s" % ( path_pkl ) )
    with open( path_pkl, 'rb' ) as input:
        data = pickle.load( input )
        
    return data

def save_pkl( data, path_pkl ):
    r'''
    Function to save data as pickle
    '''
    print( "PICKLE FILE SAVED TO %s" % ( path_pkl ) )
    with open( path_pkl, 'wb' ) as output:
        pickle.dump( data, output, pickle.HIGHEST_PROTOCOL )

class interfacial_rdf:
    r'''
    Class to compute interfacial water rdf
    '''
    ## INITIALIZING
    def __init__( self, 
                  z_ref = 0.0, 
                  z_cutoff = 0.5,
                  periodic = True,
                  residue_list = ['HOH'],
                  verbose = True, 
                  print_freq = 100,
                  ):
        
        ## STORING sigma AND MESH
        self.z_ref = z_ref
        self.z_cutoff = z_cutoff
        self.periodic = periodic
        self.residue_list = residue_list
        self.verbose = verbose
        self.print_freq = print_freq
        
        return

    ### FUNCTION TO COMPUTE DENSITY FOR A SINGLE FRAME
    def compute_single_frame( self, 
                              traj, 
                              frame = 0,
                              r_range = ( 0, 2.0 ),
                              bin_width = 0.005 ):
        r'''
        The purpose of this function is to compute the interfacial water rdf for a single 
        frame. 
        INPUTS:
            traj: [md.traj]
                trajectory object
            frame: [int]
                frame that you are interested in
        OUTPUTS:
            r:  [np.array, shape=(300,)]
                radial distance from group
            g_r: [np.array, shape=(300,)]
                 density at distance from group
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
                                        periodic = self.periodic )[frame,:,2]
        
        ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
        mask = np.logical_and( z_dist > -self.z_cutoff,
                               z_dist < self.z_cutoff )
        sliced_atom_indices = atom_indices[mask]
        n_waters = float(len(sliced_atom_indices))
        
        atom_pairs = np.array([ [ aa, sliced_atom_indices[0] ] for aa in atom_indices if aa != sliced_atom_indices[0] ])
        r, g_r = md.compute_rdf( traj, 
                                 pairs = atom_pairs, 
                                 r_range = r_range, 
                                 bin_width = bin_width, 
                                 periodic = self.periodic )
        
        for atom_ndx in sliced_atom_indices[1:]:
            atom_pairs = np.array([ [ aa, atom_ndx ] for aa in atom_indices if aa != atom_ndx ])
            _, tmp_g_r = md.compute_rdf( traj, 
                                         pairs = atom_pairs, 
                                         r_range = r_range, 
                                         bin_width = bin_width, 
                                         periodic = self.periodic )
            g_r += tmp_g_r
        
        g_r /= n_waters
        
        return r, g_r
            
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
                ## GETTING DENSITY FIELD FOR FIRST FRAME
                _, sum_g_r = self.compute_single_frame( traj = traj,
                                                        frame = 0 )
            else:
                ## COMPUTING DENSITY FIELD
                _, g_r = self.compute_single_frame( traj = traj,
                                                    frame = frame )
                ## ADDING TO TOTAL DENSITY
                sum_g_r += g_r
            
            ## PRINTING 
            if traj.time[frame] % self.print_freq == 0:
                print("====> Working on frame %d"%(traj.time[frame]))
                            
        return sum_g_r    
