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
### IMPORTING MODULES
import os, sys
import pickle
import time
import numpy as np # Used to do math functions
import mdtraj as md # Used for distance calculations
import MDDescriptors.core.import_tools as import_tools          # Loading trajectory details
from MDDescriptors.core.read_write_tools import extract_itp     # Loading itp reading tool
## IMPORTONG TRACKING TIME
from MDDescriptors.core.track_time import track_time
from MDDescriptors.core.initialize import checkPath2Server
## IMPORTING PARALLEL CODE
from MDDescriptors.parallel.parallel import parallel_analysis_by_splitting_traj

##############################################################################
# hbond class
##############################################################################
### FUNCTION TO COMPUTE HBONDS
def compute_hbond_triplets( traj,
                            path_pkl,
                            ref_ndx = -1,
                            r_cutoff = 0.5,     
                            dist_cutoff = 0.35,
                            angle_cutoff = 0.523598,
                            periodic = True,
                            n_procs = 20,
                            verbose = True,
                            residue_list = [ 'HOH' ],
                            print_freq = 100,
                            rewrite = False,
                            **kwargs ):
    R'''
    The purpose of this function is to compute the hbonding triplets in parallel
    '''
    if os.path.exists( path_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
            triplets = load_pkl( path_pkl )
    else:
        ## TRACKING TIME
        timer = track_time()
                
        ## DEFINING INPUTS FOR COMPUTE
        kwargs = { 'ref_ndx'      : ref_ndx,
                   'r_cutoff'     : r_cutoff,
                   'dist_cutoff'  : dist_cutoff,
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
                                                            want_embarrassingly_parallel = True)
        else:
            ## GETTING AVERAGE DENSITY FIELD
            triplets = [hbonds.compute_single_frame( traj = traj )]
                    
        ## OUTPUTTING TIME
        timer.time_elasped()
    
        ## SAVE OUTPUT TO PICKLE        
        save_pkl( triplets, path_pkl )    
        
    return triplets

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
    
class hbond:
    r'''
    Class containing hydrogen bonding information from a given trajectory
    '''    
    def __init__( self, ref_ndx, 
                  r_cutoff = 0.3, 
                  dist_cutoff = 0.35, 
                  angle_cutoff = 0.523598, 
                  periodic = True, 
                  verbose = True, 
                  print_freq = 100 ):
        R'''
        '''
        self.ref_ndx = ref_ndx
        self.r_cutoff = r_cutoff
        self.dist_cutoff = dist_cutoff
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
                        
        ## GET POTENTIAL WATER HBONDERS INDICES
        atoms = [ atom for atom in traj.topology.atoms        \
                  if atom.element.symbol in set(( "N", "O" )) \
                  and atom.residue.name in [ "HOH" ] ]
        atom_indices = np.array([ atom.index for atom in atoms ])
        
        ## GET REFERENCE PAIRS REF_NDX >= 0, OR BULK REF_NDX < 0
        if self.ref_ndx > -1:
            atom_pairs = np.array([ [ ndx, self.ref_ndx ] for ndx in atom_indices if ndx != self.ref_ndx ])
            r_dist = md.compute_distances( traj,
                                           atom_pairs = atom_pairs,
                                           periodic = self.periodic ).flatten()
            ## REDUCE ATOMS TO THOSE INSIDE CUTOFFS
            mask = r_dist < self.r_cutoff
            mask_sliced = r_dist < self.r_cutoff + 1.05*self.dist_cutoff
 
            ## CREATE ATOM INDICES LISTS                         
            target_atoms = atom_indices[mask]
            atoms_to_slice = atom_indices[mask_sliced]
    
            ## ADD HYDROGENS BACK TO ATOM LIST
            atom_indices_to_slice = []
            for aa in atoms_to_slice:
                group = [ aa ]
                for one, two in traj.topology.bonds:
                    if aa == one.index and two.element.symbol == "H":
                        group.append( two.index )
                atom_indices_to_slice += group
            atom_indices_to_slice = np.array(atom_indices_to_slice)

            ## SLICE TRAJECTORY TO ONLY TARGET ATOMS AND THOSE WITHIN CUTOFF
            sliced_traj = traj.atom_slice( atom_indices_to_slice, inplace = False )
            
            ## COMPUTE TRIPLETS
            sliced_triplets = self.luzar_chandler( sliced_traj, 
                                                   distance_cutoff = self.dist_cutoff, 
                                                   angle_cutoff = self.angle_cutoff )    
            ## GET ARRAY OF INDICES OF WHERE TRIPLETS CORRESPOND TO NDX_NEW
            triplets = np.array([ atom_indices_to_slice[ndx] for ndx in sliced_triplets.flatten() ])
            triplets = triplets.reshape( sliced_triplets.shape )   
        else:
            ## ASSUMING BULK SIMULATION, USES ALL WATERS
            ## COMPUTE TRIPLETS
            target_atoms = atom_indices
            triplets = self.luzar_chandler( traj, 
                                            distance_cutoff = self.dist_cutoff, 
                                            angle_cutoff = self.angle_cutoff )
        return [ target_atoms, triplets ]
    
    ### FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute( self, 
                 traj,
                 frames = [],
                 ):
        r'''
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
                        abc_distances.append( md.compute_distances( traj, triplets[ :, abc_pair ], periodic = True ) )
                        
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

def compute_hbond_types( traj,
                         triplet_list,
                         path_pkl,
                         hbond_types = [ "all", 
                                         "water", 
                                         "sam", 
                                         "water-water", 
                                         "sam-water", 
                                         "sam-sam" ],
                        **kwargs ):
    R'''
    Computes hydrogen bonding states given triplets and molecule indices, outputs
    from compute_hbonds function
    '''
    def count_hbonds( aa, triplets, atom_indices ):
        r'''
        '''
        ## FIND ALL INSTANCES AA IS DONOR
        atom_aa_donor_mask = ( triplets[:,0] == aa )
        ## FIND ALL ACCEPTORS WITH DONOR AA
        acceptors_with_donor_aa = triplets[:,2][atom_aa_donor_mask]
        ## ONLY COUNT ACCEPTORS IN ATOM_INDICES
        donor_aa_with_acceptors_in_atom_indices = [ ndx in atom_indices 
                                                    for ndx in acceptors_with_donor_aa ]
        ## COUNT DONORS
        n_donors = np.sum( donor_aa_with_acceptors_in_atom_indices )
        
        ## FIND ALL INSTANCES AA IS ACCEPTOR
        atom_aa_acceptor_mask = ( triplets[:,2] == aa )
        ## FIND ALL DONORS WITH ACCEPTOR AA
        donors_with_acceptor_aa = triplets[:,0][atom_aa_acceptor_mask]
        ## ONLY COUNT DONORS IN ATOM_INDICES
        acceptor_aa_with_donors_in_atom_indices = [ ndx in atom_indices 
                                                    for ndx in donors_with_acceptor_aa ]
        ## COUNT DONORS
        n_acceptors = np.sum( acceptor_aa_with_donors_in_atom_indices )         

        ## COUNT NUM HBONDS
        return n_donors + n_acceptors
    
    ## TRACKING TIME
    timer = track_time()

    ## MAKE DICTIONARY FOR OUTPUT DATA
    hbond_data = {}
    hbond_data['hbonds_average'] = {}
    hbond_data['hbonds_total'] = {}
    hbond_data['hbonds_distribution'] = {}
    for hb in hbond_types:
        hbond_data['hbonds_average'][hb] = 0.
        hbond_data['hbonds_total'][hb] = 0.
        hbond_data['hbonds_distribution'][hb] = np.zeros( 10 ) # 10 is arbitrarily high for num. hbonds, water is usually <5
                    
    ## GET WATER AND SAM INDICES
    atoms = [ atom for atom in traj.topology.atoms if atom.element.symbol in set(( "N", "O" )) ]
    atom_indices = {}
    atom_indices["water"] = np.array([ atom.index for atom in atoms if atom.residue.name == "HOH" ])
    atom_indices["sam"] = np.array([ atom.index for atom in atoms if atom.residue.name not in [ "HOH", "MET", "CL" ] ])
    atom_indices["all"] = np.concatenate(( atom_indices["sam"], atom_indices["water"] ))

    ## GET NUMBER OF END GROUPS
    num_end_groups = len([ residue for residue in traj.topology.residues if residue.name not in [ "HOH", "MET", "CL" ] ])

    ## LOOP THROUGH TRIPLET LIST FRAME BY FRAME    
    for ii in range(int(traj.n_frames)):
        ## GET CURRENT TRIPLETS   
        target_atoms = triplet_list[ii][0]
        triplets = triplet_list[ii][1]
        
        ## LOOP THROUGH HBOND TYPES
        for hb in hbond_types:
            if '-' in hb:
                ## SPLIT HBOND TYPES
                hb1 = hb.split('-')[0]
                hb2 = hb.split('-')[1]
            else:
                ## HBOND TYPE TO ANY
                hb1 = hb
                hb2 = "all"
                
            ## COUNT NUMBER OF HBONDS AND PER ATOM/LIGAND
            num_molecules = 0.
            num_hbonds = 0.
            for aa in target_atoms:
                if aa in atom_indices[hb1]:
                    ## COUNT HBONDS
                    nbonds = count_hbonds( aa, triplets, atom_indices[hb2] )
                    num_hbonds += nbonds
                    ## ADD TO HISTOGRAM
                    hbond_data['hbonds_distribution'][hb][int(nbonds)] += 1.
                    ## UPDATE NUM MOLECULES
                    if aa in atom_indices["water"]:
                        num_molecules += 1.
                                            
            ## NORMALIZE TO NUMBER MOLECULES/HBONDS
            if hb1 in [ "all", "sam" ] and \
               np.any( np.isin( target_atoms, atom_indices["sam"] ) ) is True:
                num_molecules += num_end_groups
            hbond_data['hbonds_average'][hb] += num_hbonds / (num_molecules+1.)
            hbond_data['hbonds_total'][hb] += num_hbonds

    ## AVERAGE, TOTAL, AND NORMALIZE HISTOGRAM
    num_hbonds = np.arange( 0, 10, 1 )
    for hb in hbond_types:
        hbond_data['hbonds_average'][hb] /= traj.n_frames
        hbond_data['hbonds_total'][hb] /= traj.n_frames
        normalize = np.max([ np.trapz( hbond_data['hbonds_distribution'][hb], x = num_hbonds ), 1 ])
        hbond_data['hbonds_distribution'][hb] /= normalize # 10 is arbitrarily high for num. hbonds, water is usually <5

    ## OUTPUTTING TIME
    timer.time_elasped()
    
    if "average" in path_pkl:
        path_pkl = path_pkl.strip("average.pkl")
    if "total" in path_pkl:
        path_pkl = path_pkl.strip("total.pkl")
    if "distribution" in path_pkl:
        path_pkl = path_pkl.strip("distribution.pkl")
        
    ## SAVE OUTPUT TO PICKLE
    save_pkl( hbond_data['hbonds_average'], path_pkl + "average.pkl" )
    save_pkl( hbond_data['hbonds_total'], path_pkl + "total.pkl" )
    save_pkl( hbond_data['hbonds_distribution'], path_pkl + "distribution.pkl" )
        
    return hbond_data

def compute_hbonds_average( traj,
                            path_pkl,
                            rewrite = False,
                            **kwargs ):
    r'''
    '''
    ## PATH TO HBOND TRIPLETS PKL
    path_triplets_pkl = path_pkl.replace("average.pkl", "triplets.pkl")
    
    ## READ HBOND TYPES PKL OR RECALCULATE
    if os.path.exists( path_pkl ) and rewrite is False:
        print( "--- PICKLE FILE FOUND! ---" )
        data = load_pkl( path_pkl )
    else:
        ## READ TRIPLETS FROM PKL OR RECALCULATE
        if os.path.exists( path_triplets_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
            triplet_list = load_pkl( path_triplets_pkl )
        else:
            triplet_list  = compute_hbond_triplets( traj,
                                                    path_pkl = path_triplets_pkl,
                                                    **kwargs )
        ## COMPUTE HBOND TYPES
        data = compute_hbond_types( traj,
                                    triplet_list,
                                    path_pkl,
                                    **kwargs )
        data = data["hbonds_average"]
    
    return data
    
def compute_hbonds_total( traj,
                          path_pkl,
                          rewrite = False,
                          **kwargs ):
    r'''
    '''
    ## PATH TO HBOND TRIPLETS PKL
    path_triplets_pkl = path_pkl.replace("total.pkl", "triplets.pkl")
    
    ## READ HBOND TYPES PKL OR RECALCULATE
    if os.path.exists( path_pkl ) and rewrite is False:
        print( "--- PICKLE FILE FOUND! ---" )
        data = load_pkl( path_pkl )
    else:
        ## READ TRIPLETS FROM PKL OR RECALCULATE
        if os.path.exists( path_triplets_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
            triplet_list = load_pkl( path_triplets_pkl )
        else:
            triplet_list  = compute_hbond_triplets( traj,
                                                    path_pkl = path_triplets_pkl,
                                                    **kwargs )
        ## COMPUTE HBOND TYPES
        data = compute_hbond_types( traj,
                                    triplet_list,
                                    path_pkl,
                                    **kwargs )
        data = data["hbonds_total"]
    
    return data
    
def compute_hbonds_distribution( traj,
                                 path_pkl,
                                 rewrite = False,
                                 **kwargs ):
    r'''
    '''
    ## PATH TO HBOND TRIPLETS PKL
    path_triplets_pkl = path_pkl.replace("distribution.pkl", "triplets.pkl")
    
    ## READ HBOND TYPES PKL OR RECALCULATE
    if os.path.exists( path_pkl ) and rewrite is False:
        print( "--- PICKLE FILE FOUND! ---" )
        data = load_pkl( path_pkl )
    else:
        ## READ TRIPLETS FROM PKL OR RECALCULATE
        if os.path.exists( path_triplets_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
            triplet_list = load_pkl( path_triplets_pkl )
        else:
            triplet_list  = compute_hbond_triplets( traj,
                                                    path_pkl = path_triplets_pkl,
                                                    **kwargs )
        ## COMPUTE HBOND TYPES
        data = compute_hbond_types( traj,
                                    triplet_list,
                                    path_pkl,
                                    **kwargs )
        data = data["hbonds_distribution"]
    
    return data 

def compute_hbonds_configurations( traj,
                                   path_pkl,
                                   rewrite = False,
                                   hbond_states = [ "Nlt1", "1geNle3", "Nge4" ],
#                                  hbond_states = [ "NONE", "D", "A", "DD", "AA", "DA", "DDA", "DAA", "DDAA", "Other" ],
                                   **kwargs ):
    R'''
    Computes hydrogen bonding states given triplets and molecule indices, outputs
    from compute_hbonds function
    '''
    ## PATH TO HBOND TRIPLETS PKL
    path_triplets_pkl = path_pkl.replace("configurations", "triplets")

    ## READ HBOND TYPES PKL OR RECALCULATE
    n_pkl = 0
    for ss in hbond_states:
        path_state_pkl = path_pkl.replace("configurations", "configurations_" + ss)
        n_pkl += int(os.path.exists( path_state_pkl ))
        
    if n_pkl < 1 or rewrite is True:
        ## READ TRIPLETS FROM PKL OR RECALCULATE
        if os.path.exists( path_triplets_pkl ) and rewrite is False:
            print( "--- PICKLE FILE FOUND! ---" )
            triplet_list = load_pkl( path_triplets_pkl )
        else:
            triplet_list  = compute_hbond_triplets( traj,
                                                    path_pkl = path_triplets_pkl,
                                                    **kwargs )
        ## CREATE EMPTY HISTOGRAM
        hbond_states_hist = np.zeros( len(hbond_states) ) # [ "N<1", "1<=N<=3", "N>=4" ]
        ## LOOP THROUGH TRIPLET LIST FRAME BY FRAME 
        for ii in range(len(triplet_list)):
            ## GET CURRENT TRIPLETS
            target_atoms = triplet_list[ii][0]
            triplets = triplet_list[ii][1]
            
            for jj in target_atoms:
                # count number of hbonds total and per water molecule
                n_donors = np.sum( triplets[:,0] == jj )
                n_acceptors = np.sum( triplets[:,2] == jj )
                n_bonds = n_donors + n_acceptors
                if n_bonds < 1.:
                    hbond_states_hist[0] += 1.
                elif n_bonds < 4.:
                    hbond_states_hist[1] += 1.
                else:
                    hbond_states_hist[2] += 1.
                            
#                if n_donors < 1. and n_acceptors < 1.: # no hbonds
#                    hbond_states_hist[0] += 1.
#                
#                elif n_donors == 1 and n_acceptors < 1.: # one donor
#                    hbond_states_hist[1] += 1.
#                    
#                elif n_donors < 1. and n_acceptors == 1.: # one acceptor\
#                    hbond_states_hist[2] += 1.
#                    
#                elif n_donors > 1. and n_acceptors < 1.: # two donors
#                    hbond_states_hist[3] += 1.
#                    
#                elif n_donors < 1. and n_acceptors > 1.: # two acceptors
#                    hbond_states_hist[4] += 1.
#                    
#                elif n_donors == 1 and n_acceptors == 1: # one donor / one acceptors
#                    hbond_states_hist[5] += 1.
#                    
#                elif n_donors > 1. and n_acceptors == 1: # two donors / one acceptors
#                    hbond_states_hist[6] += 1.
#                    
#                elif n_donors == 1 and n_acceptors > 1.: # one donor / two acceptors
#                    hbond_states_hist[7] += 1.
#                    
#                elif n_donors > 1. and n_acceptors > 1.: # two donors / two acceptors
#                    hbond_states_hist[8] += 1.
#                    
#                else: # Other
#                    hbond_states_hist[9] += 1.
                    
        ## NORMALIZE HISTOGRAM
        hbond_states_hist = hbond_states_hist / hbond_states_hist.sum()
#        hbond_states_hist = hbond_states_hist / np.trapz( hbond_states_hist, x = x )
        
        ## SAVE DATA TO PICKLE
        for ss, hist in zip( hbond_states, hbond_states_hist ):
            path_state_pkl = path_pkl.replace("configurations", "configurations_" + ss)
            save_pkl( hist, path_state_pkl )
