# -*- coding: utf-8 -*-
"""
h_bond_debug.py
The purpose of this script is to debug the hydrogen bond parameters script

CREATED ON: 01/16/2020

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
# FUNCTIONS AND CLASSES
##############################################################################
### FUNCTION TO COMPUTE HBOND TRIPLETS
def compute_hbond_triplets( sim_working_dir = "None",
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
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_hbond_triplets.pkl" )
        if rewrite is not True and os.path.exists( path_pkl ):
            triplets = load_pkl( path_pkl )
            ## RETURN RESULTS
            return triplets
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
            print( "\nCOMPUTING HBOND TRIPLETS WITH REFERENCE POSITION: %.3f nm" % z_ref )
            triplets = hbond_triplets( traj     = traj,
                                       z_ref    = z_ref,
                                       **kwargs )
            ## SAVE PICKLE FILE
            save_pkl( triplets, path_pkl )
            ## RETURN RESULTS
            return triplets

def hbond_triplets( traj,
                    z_ref = 0.0,
                    z_cutoff = 0.3, # 0.3 was used in INDUS cavity
                    r_cutoff = 0.35,
                    angle_cutoff = 0.523598,
                    periodic = True,
                    n_procs = 20,
                    verbose = True,
                    residue_list = [ 'SOL', 'HOH' ],
                    print_freq = 100,
                    rewrite = False,
                    **kwargs ):
    R'''
    The purpose of this function is to compute the hbonding triplets in parallel
    '''
    ## TRACKING TIME
    timer = track_time()
    ## DEFINING INPUTS FOR COMPUTE
    kwargs = { 'z_ref'        : z_ref,
               'z_cutoff'     : z_cutoff,
               'r_cutoff'     : r_cutoff,
               'angle_cutoff' : angle_cutoff,
               'periodic'     : periodic,
               'verbose'      : True, 
               'print_freq'   : print_freq
             }        
    ## DEFINING INTERFACE
    hbonds = hbond( **kwargs )
    ## PARALELLIZING CODE
    if n_procs != 1:
        ## GETTING HBOND TRIPLETS
        triplets = parallel_analysis_by_splitting_traj( traj = traj, 
                                                        class_function = hbonds.compute, 
                                                        n_procs = n_procs,
                                                        combine_type = "append_list",
                                                        want_embarrassingly_parallel = False )
    else:
        ## GETTING AVERAGE DENSITY FIELD
        triplets = hbonds.compute( traj = traj )
    ## OUTPUTTING TIME
    timer.time_elasped()
    ## RETURN RESULTS
    return triplets
    
class hbond:
    r'''
    Class containing hydrogen bonding information from a given trajectory
    '''    
    def __init__( self, traj = None,
                  z_ref = 0.0, 
                  z_cutoff = 0.3, 
                  r_cutoff = 0.35, 
                  angle_cutoff = 0.523598, 
                  periodic = True, 
                  verbose = True, 
                  print_freq = 100,
                  **kwargs ):
        R'''
        '''
        self.z_ref = z_ref
        self.z_cutoff = z_cutoff
        self.r_cutoff = r_cutoff
        self.angle_cutoff = angle_cutoff
        self.periodic = periodic
        self.verbose = verbose
        self.print_freq = print_freq  

    def compute_single_frame( self,
                              traj,
                              frame = 0, ):
        r'''
        The purpose of this function is to compute the hbond triplets for a single 
        frame. 
        INPUTS:
            traj: [md.traj]
                trajectory object
            frame: [int]
                frame that you are interested in
        OUTPUTS:
            triplets: [np.array, shape=(Nx3)]
                hbond triplets
        '''
        ## REDUCE TRAJ TO SINGLE FRAME
        traj = traj[frame]
                        
        ## GET POTENTIAL DONOR AND ACCEPTOR INDICES
        atoms = [ atom for atom in traj.topology.atoms if atom.element.symbol in [ "N", "O" ] ]
        atom_indices = np.array([ atom.index for atom in atoms ])
        
        ## COMPUTE DISTANCE VECTORS BETWEEN REF POINT AND DONORS/ACCEPTORS
        ref_coords = [ 0.5*traj.unitcell_lengths[0,0], 0.5*traj.unitcell_lengths[0,1], self.z_ref ]
        z_dist = compute_displacements( traj,
                                        atom_indices = atom_indices,
                                        box_dimensions = traj.unitcell_lengths,
                                        ref_coords = ref_coords,
                                        periodic = self.periodic )[:,2]
        
        ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
        mask = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                               z_dist < self.z_cutoff )
        mask_sliced = np.logical_and( z_dist > -(self.z_cutoff + 0.5), # always includes head groups
                                      z_dist < self.z_cutoff + 1.05*self.r_cutoff ) # include hbonders above
        
        ## MASK OUT TARGET ATOMS AND ATOMS TO SLICE
        target_atoms = atom_indices[mask]
        atoms_to_slice = atom_indices[mask_sliced]
        
        ## ADD HYDROGENS BACK TO ATOM LIST
        atom_indices_to_slice = []
        for aa in atoms_to_slice:
            group = [ aa ]
            for one, two in traj.topology.bonds:
                if aa == one.index and two.element.symbol == "H":
                    group.append( two.index )
                elif one.element.symbol == "H" and aa == two.index:
                    group.append( one.index )
            atom_indices_to_slice += group
        atom_indices_to_slice = np.array(atom_indices_to_slice)
    
        ## SLICE TRAJECTORY TO ONLY TARGET ATOMS AND THOSE WITHIN CUTOFF
        sliced_traj = traj.atom_slice( atom_indices_to_slice, inplace = False )
        
        ## COMPUTE TRIPLETS
        sliced_triplets = self.luzar_chandler( sliced_traj, 
                                               distance_cutoff = self.r_cutoff, 
                                               angle_cutoff = self.angle_cutoff )

        ## GET ARRAY OF INDICES OF WHERE TRIPLETS CORRESPOND TO NDX_NEW
        triplets = np.array([ atom_indices_to_slice[ndx] for ndx in sliced_triplets.flatten() ])
        triplets = triplets.reshape( sliced_triplets.shape )   

        return [ target_atoms, triplets ]
    
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
                triplets = [self.compute_single_frame( traj = traj, 
                                                       frame = 0 )]
            else:
                ## COMPUTING TRIPLET ANGLES AND CONCATENATE
                triplets += [self.compute_single_frame( traj = traj,
                                                        frame = frame )]
            ## PRINTING 
            if traj.time[frame] % self.print_freq == 0:
                print("====> Working on frame %d"%(traj.time[frame]))
                            
        return triplets
    
    ## FUNCTION TO COMPUTE HBONDS USING LUZAR-CHANDLER CRITERIA
    def luzar_chandler( self, traj, 
                        distance_cutoff = 0.35, 
                        angle_cutoff = 0.523598 ):
        R'''Identify hydrogen bonds based on cutoffs for the Donor...Acceptor
        distance and H-Donor...Acceptor angle. Works best for a single trajectory
        frame, anything larger is prone to memory errors.
        
        The criterion employed is :math:'\\theta > \\pi/6' (30 degrees) and 
        :math:'r_\\text{Donor...Acceptor} < 3.5 A'.
        
        The donors considered by this method are NH and OH, and the acceptors 
        considered are O and N.
        
        Input
        -----
        traj : md.Trajectory
            An mdtraj trajectory.
        distance_cutoff : 0.35 nm (3.5 A)
            Default 'r_\\text{Donor..Acceptor}' distance
        angle_cutoff : '\\pi/6' (30 degrees)
            Default '\\theta' cutoff
        
        Output
        ------
        hbonds : np.array, shape=[n_hbonds, 3], dtype=int
            An array containing the indices atoms involved in each of the identified 
            hydrogen bonds. Each row contains three integer indices, '(d_i, h_i, a_i)',
            such that 'd_i' is the index of the donor atom , 'h_i' the index of the 
            hydrogen atom, and 'a_i' the index of the acceptor atom involved in a 
            hydrogen bond which occurs (according to the definition above).
            
        References
        ----------
        Luzar, A. & Chandler, D. Structure and hydrogen bond dynamics of water–
        dimethyl sulfoxide mixtures by computer simulations. J. Chem. Phys. 98, 
        8160–8173 (1993).
        '''    
        def _get_bond_triplets( traj ):    
            def get_donors(e0, e1):
                # Find all matching bonds
                elems = set((e0, e1))
                atoms = [(one, two) for one, two in traj.topology.bonds if set((one.element.symbol, two.element.symbol)) == elems]
        
                # Get indices for the remaining atoms
                indices = []
                for a0, a1 in atoms:
                    pair = (a0.index, a1.index)
                    # make sure to get the pair in the right order, so that the index
                    # for e0 comes before e1
                    if a0.element.symbol == e1:
                        pair = pair[::-1]
                    indices.append(pair)
        
                return indices
        
            nh_donors = get_donors('N', 'H')
            oh_donors = get_donors('O', 'H')
            xh_donors = np.array(nh_donors + oh_donors)
        
            if len(xh_donors) == 0:
                # if there are no hydrogens or protein in the trajectory, we get
                # no possible pairs and return nothing
                return np.zeros((0, 3), dtype=int)
        
            acceptor_elements = frozenset(('O', 'N'))
            acceptors = [ a.index for a in traj.topology.atoms if a.element.symbol in acceptor_elements ]
        
            # Make acceptors a 2-D numpy array
            acceptors = np.array(acceptors)[:, np.newaxis]
        
            # Generate the cartesian product of the donors and acceptors
            xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
            acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
            bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))
        
            # Filter out self-bonds
            self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
            return bond_triplets[np.logical_not(self_bond_mask), :]
        
        def _compute_bounded_geometry( traj, 
                                       triplets, 
                                       distance_indices = [ 0, 2 ], 
                                       angle_indices = [ 1, 0, 2 ] ):
                '''this function computes the distances between the atoms involved in
                the hydrogen bonds and the H-donor...acceptor angle using the law of 
                cosines.
                
                Inputs
                ------
                traj : md.traj
                triplets : np.array, shape[n_possible_hbonds, 3], dtype=int
                    An array containing the indices of all possible hydrogen bonding triplets
                distance_indices : [LIST], [ donor_index, acceptor_index ], default = [ 0, 2 ]
                    A list containing the position indices of the donor and acceptor atoms
                angle_indices : [LIST], [ h_index, donor_index, acceptor_index ], default = [ 1, 0, 2 ]
                    A list containing the position indices of the H, donor, and acceptor 
                    atoms. Default is H-donor...acceptor angle
                  
                Outputs
                -------
                distances : np.array, shape[n_possible_hbonds, 1], dtype=float
                    An array containing the distance between the donor and acceptor atoms
                angles : np.array, shape[n_possible_hbonds, 1], dtype=float
                    An array containing the triplet angle between H-donor...acceptor atoms
                '''  
                # Calculate the requested distances
                distances = md.compute_distances( traj, triplets[ :, distance_indices ], periodic = True )
                
                # Calculate angles using the law of cosines
                abc_pairs = zip( angle_indices, angle_indices[1:] + angle_indices[:1] )
                abc_distances = []
                
                # calculate distances (if necessary)
                for abc_pair in abc_pairs:
                    if set( abc_pair ) == set( distance_indices ):
                        abc_distances.append( distances )
                    else:
                        abc_distances.append( md.compute_distances( traj, triplets[ :, abc_pair ], ) )
                        
                # Law of cosines calculation to find the H-Donor...Acceptor angle
                #            c**2 = a**2 + b**2 - 2*a*b*cos(C)
                #                        acceptor
                #                          /\
                #                         /  \
                #                      c /    \ b
                #                       /      \ 
                #                      /______(_\
                #                     H    a     donor
                a, b, c = abc_distances
                cosines = ( a ** 2 + b ** 2 - c ** 2 ) / ( 2 * a * b )
                np.clip(cosines, -1, 1, out=cosines) # avoid NaN error
                angles = np.arccos(cosines)
                
                return distances, angles
    
        if traj.topology is None:
            raise ValueError( 'hbond requires that traj contain topology information' )
        
        # get the possible donor-hydrogen...acceptor triplets    
        bond_triplets = _get_bond_triplets( traj )
        
        distances, angles = _compute_bounded_geometry( traj, bond_triplets )
        
        # Find triplets that meet the criteria
        presence = np.logical_and( distances < distance_cutoff, angles < angle_cutoff )
        
        return bond_triplets.compress( presence.flatten(), axis = 0 )
