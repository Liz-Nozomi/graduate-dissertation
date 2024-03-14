# -*- coding: utf-8 -*-
"""
h_bond.py
The purpose of this script is to calculate the average number of hydrogen bonds for a given trajectory

CLASSES:
    calc_hbond: calculates the hydrogen bonding between solute and solvent
            input_details={
                    'solute_name'        : 'PDO',                                   # Solute of interest
                    'solvent_name'       : ['HOH', 'DIO', 'GVLL'],                  # Solvents you want radial distribution functions for
                    'itp_files'          : ['12-propanediol.itp', 'dioxane.itp'],   # itp files to look into
                    'want_solute_donor_breakdown': True                             # True if you want solute donor breakdown
                    }
            
            IMPORTANT NOTE: This script uses PDB files for the gro. This is necessary for water bonds to correctly be accounted for!!!

CREATED ON: 04/04/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
** UPDATES **
2018-04-05: Completed draft of hydrogen bonding script
2018-04-15: Fixed bug in hydrogen bonding script

TODO: Need to parallelize the for-loop for calculating hydrogen bonds
"""
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools
from MDDescriptors.core.read_write_tools import extract_itp # Extraction of itp files
import sys
import mdtraj as md # Used for distance calculations
import numpy as np # Used to do math functions
import itertools # Used for combinations/permutations
import time # Used to keep track of time
from MDDescriptors.core.initialize import convert2HoursMinSec
import glob # Used to look for itp files
from MDDescriptors.core.initialize import checkPath2Server
## INCLUDING PARALLELIZATION CODE
import MDDescriptors.core.parallel as parallel

### FUNCTION TO EXTRACT SPECIFIC INDEX
def extract_index_from_dict(my_dict, index):
    ''' This function extracts index value from dictionary'''
    my_dict_index = { each_key: my_dict[each_key][index]  for each_key in my_dict}
    return my_dict_index

### FUNCTION TO EXTRACT NUM ACCEPTOR AND DONOR INDEX
def extract_index_acceptor_and_donor( acceptor_donor_dict, index, donor_acceptor_key=['num_donor','num_acceptor'] ):
    ''' This finds index from acceptor/donor array'''
    avg_dict = { each_key:{each_donor_acceptor_pair: acceptor_donor_dict[each_key][each_donor_acceptor_pair][index]
                        for each_donor_acceptor_pair in donor_acceptor_key} for each_key in acceptor_donor_dict.keys() }
    return avg_dict
####################################################
### CLASS FUNCTION TO CALCULATE HYDROGEN BONDING ###
####################################################
class calc_hbond:
    '''
    The purpose of this function is to find the number of hydrogen bonds between a solute and solvent. Note that this will go through some itp files to try to find bonding information.
    We assume that all itp files are within the same directory as your trajectories
    INPUTS:
        solute_name: [str] Name of the solute
        solvent_name: [list] List of solvent names (as many as you like! We will check if solvent is there)
        want_solute_donor_breakdown: [logical] True if you want OH/NH groups of your solute to be observed
        debug_mode: [logical] True if you want output for everything
        want_parallel: [logical] True if you want to parallel the hbond script (default: False)
    OUTPUTS:
        ## ITP FILE
            self.itp_info: [list] itp information for bonding
        
        ## BONDING INFORMATION
            self.bonds: [np.array, shape=(num_bonds, 2)]: bonds between atom indexes. These bonds are unique! (No duplications)
            
        ## H BONDING INFO
            ## DONORS
                self.donor_bonds: [np.array, shape=(num_donor, 2)] Outputs atom index that have donor bonds
                self.num_donor_bonds: number of donor bonds
            ## ACCEPTORS
                self.acceptors: [np.array, shape=(num_acceptors, 1)] index of all acceptor atoms
                self.acceptors_solute: [np.array, shape=(num_solute_acceptors, 1)]: index of all solute acceptor atoms
                self.acceptors_solute_names: [list] names of all solute acceptor atoms
                self.num_acceptors: [int] total number of acceptor atoms
            ## TRIPLETS
                self.triplet: [np.array, shape=(num_triplets, 3)]: Triplet of the atom indexes
                self.num_triplet: [int] total number of triplets
        ## SOLUTE SOLVENT COMBINATIONS
            self.residue_list: full residue list
            self.residue_combinations: combinations between solute and solvent (includes solvent-solvent interactions)
            self.total_residues: total residue lists
                
        ## RESULTS
            self.hbond_btn_residues: [dict] contains dictionary of residue-residue interactions as a form of the list (the list will be the same size as the time vector)
            self.hbond_total: [list] total number of hydrogen bonds per frame
            self.hbond_solute_donor: [dict] contains dictionary of solute-heavy atoms to solvent molecules
            
    FUNCTIONS:
        load_itp_file: loading itp file
        create_bond_list: create bonding list
        find_donors: find all donor bonds
        find_acceptors: finds all acceptor atoms from trajectory
        find_triplets: finds all possible triplets
        find_solute_solvent_combinations: finds all combinations between solute and solvent
        compute_bounded_geometry: [staticmethod]: simply finds bound geometry (i.e. distances and angles)
        calc_hydrogen_bonds: computes hydrogen bonding given the geometry and triplets (preferably for a single trajectory frame)
        find_total_num_hbond_btn_groups: find total number of hydrogen bonds between residue groups
        
    ALGORITHM:
        - Find donor, acceptor, triplet bonds
        - Calculate hydrogen bonds
        - Use sorting to get which is hydrogen bonded to which    
    TODO:
        - Get the solute acceptors atom names
    '''

    
    ### INITIALIZING
    def __init__(self,
                 traj_data, 
                 solute_name = None, 
                 solvent_name = [], 
                 distance_cutoff = 0.35, 
                 angle_cutoff = np.pi / 6.,
                 want_solute_donor_breakdown = False, 
                 debug_mode=False, 
                 want_solute_acceptor_donor_info = False,
                 want_serial = True):
        ## STORING INPUT VARIABLES
        self.solute_name=solute_name
        self.solvent_name=solvent_name
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
        self.want_solute_donor_breakdown = want_solute_donor_breakdown
        self.debug_mode = debug_mode
        self.want_solute_acceptor_donor_info = want_solute_acceptor_donor_info
        self.want_serial = want_serial 
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## DEFINING TIMES TO PRINT FRAME
        self.frame_print_freq = 100
        ## DEFINING LENGTH OF TRAJECTORY
        self.traj_length = len(traj)
        ## DEFINING TRAJECTORY TIMES
        self.traj_time = traj.time
        
        ## CHECK IF SOLUTE EXISTS IN TRAJECTORY
        if self.solute_name not in traj_data.residues.keys():
            print("ERROR! Solute (%s) not available in trajectory. Stopping here to prevent further errors. Check your 'Solute' input!")
            sys.exit()
        
        ## CHECK THE SOLVENTS
        self.solvent_name = [ each_solvent for each_solvent in self.solvent_name if each_solvent in traj_data.residues.keys() ]
        
        ## GENERATING LIST OF ALL RESIDUES
        self.residue_list = [ self.solute_name ] + self.solvent_name
        
        ## FINDING TOTAL RESIDUES
        self.total_residues = [ calc_tools.find_total_residues(traj, resname)[0] for resname in self.residue_list ]
        
        ## FINDING TOTAL RESIDUE ATOM INDEX
        self.residue_dict_atoms = { resname: np.array(calc_tools.find_total_atoms(traj, resname)[1]) 
                                                for resname in self.residue_list }
        
        ## LOADING EACH TRAJECTORY
        self.load_itp_file(itp_path = traj_data.directory )
        
        ## FINDING BONDING INFORMATION FOR THE TOPOLOGY
        self.create_bond_list(traj)
        
        ## FINDING ALL DONORS
        self.find_donors(traj)
        
        ## FINDING ALL ACCEPTORS
        self.find_acceptors(traj)
        
        ## FINDING ALL TRIPLETS BETWEEN DONORS AND ACCEPTORS
        self.find_triplets()
        
        ## FINDING COMBINATIONS OF SOLUTE AND SOLVENT
        self.find_solute_solvent_combinations()
        
        ## GENERATING EMPTY DICTIONARY FOR APPENDING HYDROGEN BONDING INFORMATION
        self.hbond_btn_residues = {}
        ## CREATING EMPTY LISTS BASED ON THE COMBINATIONS
        for each_combo in self.residue_combinations:
            self.hbond_btn_residues[each_combo[0] + '-' + each_combo[1]  ] = [None] * self.traj_length
        
        ## SOLUTE DONOR BREAKDOWN: GENERATING EMPTY DICTIONARY FOR APPENDING HYDROGEN BONDING INFORMATION
        if want_solute_donor_breakdown is True or self.want_solute_acceptor_donor_info is True:
            self.hbond_solute_donor={}
            self.solute_acceptor_donor_hbond = {}
            ## CREATING EMPTY LIST BASED ON SOLVENTS
            for each_solute_atom_index in self.acceptors_solute:
                ## FINDING NAME OF THE ATOM
                atom_name = traj.topology.atom(each_solute_atom_index).name
                ## LOOPING THROUGH SOLVENTS
                for each_solvent in self.solvent_name:
                    if want_solute_donor_breakdown is True:
                        self.hbond_solute_donor[self.solute_name +'-' + atom_name + '-' + each_solvent] = [None] * self.traj_length
                    if self.want_solute_acceptor_donor_info is True:
                        self.solute_acceptor_donor_hbond[self.solute_name +'-' + atom_name + '-' + each_solvent] ={'num_donor': [None] * self.traj_length,
                                                                                                                  'num_acceptor': [None] * self.traj_length,}

        ## STORING TOTAL NUMBER OF HBONDS
        self.hbond_total = [None] * self.traj_length
        
        
        ## COMPUTING 
        if self.want_serial is True:
            self.compute( traj = traj, want_serial=self.want_serial )
        
        return
    ##################################
    ### CALCULATING HYDROGEN BONDS ###
    ##################################
    
    ## LOOPING THROUGH THE TRAJECTORY
    def compute(self, traj, index = None, want_serial=False):
        if want_serial is False:
            # PARALLEL IMPLEMENTATION
            self.hbond_single_loop(traj = traj, index = index)
        
            ## DEFINING A WAY TO STORE RESULTS
            results = {
                    'hbond_btn_residues': extract_index_from_dict( self.hbond_btn_residues, index = index),
                    }
            if self.want_solute_donor_breakdown is True:
                results['hbond_solute_donor'] = self.hbond_solute_donor
            if self.want_solute_acceptor_donor_info is True:
                results['want_solute_acceptor_donor_info'] = extract_index_acceptor_and_donor( acceptor_donor_dict = self.solute_acceptor_donor_hbond,
                                                                                               index = index    )
            return results
        else:
            for each_index in range(len(traj)):
                self.hbond_single_loop(traj = traj, 
                                       index = each_index)
        return
    
    ### FUNCTION TO LOAD EACH TRAJECTORY AND FIND ITS RESIDUE NAME
    def load_itp_file(self, itp_path):
        '''
        The purpose of this function is to load the itp file, and see if it matches with any of the residues
        INPUTS:
            self: class object
            itp_path: directory of itp files
        OUTPUTS:
            self.itp_dict: [dict] itp information for bonding
                NOTE: only itp information for the solutes and solvents within is saved
        '''
        ## CREATING VARIABLE TO SAVE ITP FILES
        self.itp_dict = {}
        ## USING GLOB TO FIND ALL ITP FILES
        itp_files = glob.glob( itp_path + '/*.itp' )
        ## LOOPING THROUGH EACH ITP FILE
        for full_itp_path in itp_files:
            ## ATTEMPTING TO LOAD IT
            try:
                ## FINDING THE ITP FILE
                itp_info = extract_itp(full_itp_path)
                ## EXTRACTING RESIDUE NAME
                residue_name = itp_info.residue_name
                ## STORE ONLY IF THE RESIDUE NAME MATCHES THE SOLUTE/SOLVENT
                if residue_name in self.residue_list and residue_name not in self.itp_dict.keys() :
                    self.itp_dict[residue_name]=itp_info 
            except:
                pass
        
        ## AT THE END, CHECK IF ALL THE MOLECULES HAVE AN ITP FILE
        for each_residue in self.residue_list:
            if each_residue not in self.itp_dict.keys() and each_residue != 'HOH':
                print("ERROR! Missing itp file for residue: %s"%(each_residue) )
                print("Pausing so you can see this error! This will bring issues in computing h-bonding, which need bonding information")
                time.sleep(5)
        
        return

    ### FUNCTION TO CREATE BONDING LIST
    def create_bond_list(self, traj):
        '''
        The purpose of this function is to go through the itp files and find all bonding information.
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.bonds: [np.array, shape=(num_bonds, 2)]: bonds between atom indexes. These bonds are unique! (No duplications)
            self.num_bonds: [int] number of bonds
        '''
        ## LOADING BONDING INFORMATION FROM MD.TRAJ
        table, bonds = traj.topology.to_dataframe()
        ## FINDING ALL UNIQUE BONDS
        self.bonds = np.unique(bonds,axis=0)
        ## LOOPING THROUGH EACH SOLUTE/SOLVENT AND INCLUDING BONDING INFORMATION
        for each_residue in self.residue_list:
            ## IGNORING WATER -- ASSUME THAT IS TAKEN BY THE PDB STRUCTURE FILE
            if each_residue != 'HOH':
                ## FINDING ITP FILE THAT CORRESPONDS TO THE RESIDUE
                itp_info = self.itp_dict[each_residue]
                print("*** LOADING BONDING INFORMATION FOR %s ***"%(each_residue))
                ## CREATING BOND STORAGE
                bond_storage = []
                ## FINDING ALL RESIDUES
                num_residues, index_residues = calc_tools.find_total_residues(traj, resname = each_residue)
                ## LOOPING THROUGH EACH RESIDUE AND BONDING EACH ATOM
                for index in index_residues:
                    ## FINDING RESIDUE
                    residue = traj.topology.residue(index)
                    ## FINDING ALL BONDING INDEX
                    atom_bond_index = [ [residue.atom(each_bond[0]-1).index, residue.atom(each_bond[1]-1).index ] for each_bond in itp_info.bonds]
                    ## APPENDING TO BONDS
                    bond_storage.extend(atom_bond_index)
                    
                ## CONCATENATING TO BONDS
                self.bonds = np.concatenate( (self.bonds,np.array(bond_storage)), axis=0)
                ## FINDING NUMBER OF BONDS
                self.num_bonds = len(self.bonds)
                ## PRINTING
                print("ADDING %d BONDS TO BONDING INFORMATION, TOTAL BONDS IS: %d"%(len(bond_storage), self.num_bonds))
        return
      
    ### FUNCTION TO FIND THE DONORS
    def find_donors(self, traj):
        '''
        The purpose of this function is to find all the donors (e.g. O-H, N-H, etc.)
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.donor_bonds: [np.array, shape=(num_donor, 2)] Outputs atom index that have donor bonds
            self.num_donor_bonds: number of donor bonds
        '''
        ### GOING THROUGH ALL BONDS AND FINDING IF WE HAVE AN OH GROUP
        self.donor_bonds = [] # Creating empty list that can store the donor bonds
        for each_bond in self.bonds:
            element_list = [ traj.topology.atom(each_atom).element.symbol for each_atom in each_bond ]
            if ( 'O' in element_list and 'H' in element_list ) or ( 'N' in element_list and 'H' in element_list ):
                ## FINDING DONOR INDEX
                if 'O' in element_list:
                    donor_index = element_list.index('O')
                elif 'N' in element_list:
                    donor_index = element_list.index('N')
                ## FINDING HYDROGEN INDEX
                hyd_index = element_list.index('H')
                self.donor_bonds.append( [each_bond[donor_index], each_bond[hyd_index]])
        ## CONVERTING DONOR BONDS TO ARRAY
        self.donor_bonds = np.array(self.donor_bonds)
        ### FINDING TOTAL DONOR BONDS
        self.num_donor_bonds = len(self.donor_bonds)
        ## PRINTING
        print("FOUND %d DONOR BONDS OUT OF A TOTAL OF %d BONDS"%( self.num_donor_bonds, self.num_bonds ))
        return
    
    ### FUNCTION TO FIND ALL ACCEPTOR ATOMS
    def find_acceptors(self, traj):
        '''
        The purpose of this function is to find all acceptors (e.g. O, N, etc.)
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.acceptors: [np.array, shape=(num_acceptors, 1)] index of all acceptor atoms
            self.acceptors_solute: [np.array, shape=(num_solute_acceptors, 1)]: index of all solute acceptor atoms
            self.acceptors_solute_names: [list] names of all solute acceptor atoms
            self.num_acceptors: [int] total number of acceptor atoms
        '''
        ## LOOPING THROUGH EACH ATOM AND FINDING ALL ACCEPTOR ELEMENTS
        acceptor_elements = frozenset( ( 'O', 'N' ) )
        self.acceptors = [ a.index for a in traj.topology.atoms 
                  if a.element.symbol in acceptor_elements ]
        ## LOOPING THROUGH AND FINDING ALL ACCEPTORS FOR THE SOLUTE
        self.acceptors_solute = [ each_acceptor for each_acceptor in self.acceptors if traj.topology.atom(each_acceptor).residue.name == self.solute_name ]
        self.acceptors_solute_names =  [ traj.topology.atom(each_acceptor).name for each_acceptor in self.acceptors_solute ]
        ## FINDING TOTAL ACCEPTORS
        self.num_acceptors = len(self.acceptors)
        print("***FINDING TOTAL ACCEPTORS***")
        print("TOTAL ACCEPTORS: %d"%(self.num_acceptors))
        
    ### FUNCTION TO FIND ALL TRIPLETS
    def find_triplets(self):
        '''
        The purpose of this function is to find all triplets between donor and acceptors
        INPUTS:
            self: class object
        OUTPUTS:
            self.triplet: [np.array, shape=(num_triplets, 3)]: Triplet of the atom indexes
            self.num_triplet: [int] total number of triplets
        '''
        ## MAKING ACCEPTORS A 2D NUMPY ARRAY
        acceptors = np.array( self.acceptors )[ :, np.newaxis ]
        
        ## GENERATE CARTESIAN PRODUCT OF DONOARS AND ACCEPTORS
        donors_repeated = np.repeat( self.donor_bonds, acceptors.shape[0], axis = 0 )
        # RETURNS REPEATED DONORS, e.g.:
            # array([[10472, 10473],
            #    [10472, 10473],
            #    [10472, 10473],...
        # SHAPE OF THIS IS: (NUMBER OF DONORS * NUMBER OF ACCEPTORS, 2)
        
        ## EXPANDING THE ACCEPTOR MATRIX
        acceptors_tiled = np.tile( acceptors, ( self.donor_bonds.shape[0], 1 ) )
        # RETURNS REPEATED ACCEPTORS, e.g.:
            #   array([[    2],
            #    [    5],
            #    [   16],
        # SHAPE OF THIS IS: (NUMBER OF DONORS * NUMBER OF ACCEPTORS, 1)
        
        ## STACKING THE BOND AND TRIPLETS TOGETHER
        stack_bond_triplets = np.hstack( ( donors_repeated, acceptors_tiled ) )
        # RETURNS COMBINED REPEATED DONOR + ACCEPTOR, e.g.:
            # array([[10472, 10473,     2],
            #         [10472, 10473,     5],
            #         [10472, 10473,    16],
        # NOTE: This contains repeats! Some of the donors can be acceptors and vice-versa
        
        ## FINDING DUPLICATES ACCEPTOR/DONOR
        bond_triplet_filter = ( stack_bond_triplets[ :, 0 ] == stack_bond_triplets[ :, 2 ] )
        # RETURNS TRUE OR FALSE, WHERE TRUE INDICATES THAT WE HAVE IT DUPLICATED, SHAPE = NUMBER OF DONORS * NUMBER OF ACCEPTORS
        
        ## CORRECTING BOND TRIPLETS
        self.triplet =stack_bond_triplets[ np.logical_not(bond_triplet_filter) ]
        
        ## FINDING TOTAL TRIPLETS
        self.num_triplet = len(self.triplet)
        
        ## PRINTING
        print("*** FINDING ALL TRIPLETS ***")
        print("TOTAL NUMBER OF TRIPLETS: %d"%(self.num_triplet))
        return
    ### FUNCTION TO CALCULATE THE GEOMETRY
    @staticmethod
    def compute_bounded_geometry( traj, triplets, distance_cutoff, freq=0.0, distance_indices = [ 0, 2 ], angle_indices = [ 1, 0, 2 ], periodic = True ):
        '''
        This function computes the distances between the atoms involved in
        the hydrogen bonds and the H-donor...acceptor angle using the law of 
        cosines.
        
        INPUTS
        ------
        traj : md.traj
        triplets : np.array, shape[n_possible_hbonds, 3], dtype=int
            An array containing the indices of all possible hydrogen bonding triplets
        distance_indices : [LIST], [ donor_index, acceptor_index ], default = [ 0, 2 ]
            A list containing the position indices of the donor and acceptor atoms
        angle_indices : [LIST], [ h_index, donor_index, acceptor_index ], default = [ 1, 0, 2 ]
            A list containing the position indices of the H, donor, and acceptor 
            atoms. Default is H-donor...acceptor angle
            
        OUTPUTS
        -------
        distances : np.array, shape[n_possible_hbonds, 1], dtype=float
            An array containing the distance between the donor and acceptor atoms
        angles : np.array, shape[n_possible_hbonds, 1], dtype=float
            An array containing the triplet angle between H-donor...acceptor atoms
        '''
        # Calculate the requested distances
        distances = md.compute_distances( traj, triplets[ :, distance_indices ], periodic = periodic)
        
        '''
        # Now we discover which triplets meet the distance cutoff often enough
        prevalence = np.mean(distances < distance_cutoff, axis=0)
        mask = prevalence > freq
        
        # Update data structures to ignore anything that isn't possible anymore
        triplets = triplets.compress(mask, axis=0)
        distances = distances.compress(mask, axis=1)
        '''
        # Calculate angles using the law of cosines
        abc_pairs = zip( angle_indices, angle_indices[1:] + angle_indices[:1] )
        abc_distances = []
        
        # calculate distances (if necessary)
        for abc_pair in abc_pairs:
            if set( abc_pair ) == set( distance_indices ):
                abc_distances.append( distances )
            else:
                abc_distances.append( md.compute_distances( traj, triplets[ :, abc_pair ], periodic=periodic ) )
                
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
    
    ### FUNCTION TO CALCULATE HYDROGEN BONDS
    def calc_hydrogen_bonds(self, traj, freq=0.0, periodic=True ):
        '''
        This function takes your triplet, and trajectory, then finds whether or not there has been a hydrogen bond
        INPUTS:
            self: class objects
            traj: trajectory from md.traj
        OUTPUTS:
            hbonds: [np.array, shape=(number_h_bonds, 3)] Triplet of the hydrogen bonding donor to acceptor
        NOTE: It may be preferred that you do trajectories as single frames. Although this can handle multiple frames, the hydrogen bonds get convoluted you consider multiple frames!
        '''
        ## FINDING GEOMETRIC DISTANCES AND ANGLES
        distances, angles = self.compute_bounded_geometry(traj = traj,
                                                    distance_cutoff = self.distance_cutoff,
                                                     triplets = self.triplet,
                                                     periodic = periodic,
                                                     )
        ## RETURNS: distances: TIME FRAME, EACH_TRIPLET: distances in nm between the donor and acceptor
        ## RETURNS: angles: TIME_FRAME, EACH_TRIPLET: returns angles between specified angles
    
        ## FINDING TRIPLETS THAT MATCH THE CRITERIA
        presence = np.logical_and( distances < self.distance_cutoff, angles < self.angle_cutoff )

        ## COMPRESSING INTO HYDROGEN BONDS
        hbonds = self.triplet.compress( presence.flatten(), axis = 0 )
        return hbonds
    
    ### FUNCTION TO FIND TOTAL NUMBER OF HYDROGEN BONDS BETWEEN TWO GROUPS
    def find_total_num_hbond_btn_groups(self, traj, hbonds, group_1, group_2, group_type='residue_to_residue'):
        '''
        The purpose of this function is to find the total number of hydrogen bonds between two residues
        INPUTS:
            self: class objects
            traj: trajectory from md.traj 
            hbonds: [np.array, shape=(number_h_bonds, 3)] Triplet of the hydrogen bonding donor to acceptor
            group_1: name of residue 1
            group_2: name of residue 2
            group_type: type of grouping, e.g:
                residue_to_residue: [DEFAULT] based on residue name, residue name of 1 to residue name of 2
                atom_to_residue: based on atom index to residue. NOTE: group_1 must be some atom index! Similarly, group_2 is some residue name
        OUTPUTS:
            total_hbond_residues: [int] total hydrogen bonding between residues
        '''            
        ## DEFINING STORAGE
        total_hbond_residues = 0
        
        if group_type == 'residue_to_residue':
            ## DEFINING RESIDUE INDEX
            group_1_atom_index = self.residue_dict_atoms[group_1]
            group_2_atom_index = self.residue_dict_atoms[group_2]
            ## GETTING TOTAL HBOND
            total_hbond_residues = ( np.isin( hbonds[:,0] , group_1_atom_index, ) & np.isin( hbonds[:,2], group_2_atom_index ) |
                                      np.isin( hbonds[:,2] , group_1_atom_index, ) & np.isin( hbonds[:,0], group_2_atom_index )).sum()
        
        ## LOOP THROUGH EACH HYDROGEN BOND AND FIND THE DONOR AND ACCEPTOR RESIDUE NAMES
#        for each_hbond in hbonds:
#            ## DEFINING DONOR AND ACCEPTOR ATOM INDEX
#            donor_atom_index, acceptor_atom_index = each_hbond[0], each_hbond[2]
#            ## FINDING RESIDUE NAME OF THE DONOR RESIDUE NAME
#            donor_group_name = traj.topology.atom(donor_atom_index).residue.name
#            ## FINDING ACCEPTOR RESIDUE NAME
#            acceptor_group_name = traj.topology.atom(acceptor_atom_index).residue.name
#            if group_type == 'residue_to_residue':
#                ## SEEING IF EITHER DONOR AND ACCEPTOR MATCHES
#                if (donor_group_name == group_1 and acceptor_group_name == group_2) or \
#                   (donor_group_name == group_2 and acceptor_group_name == group_1):
#                       ## ADDING ONE TO THE HBOND RESIDUES
#                       total_hbond_residues += 1
#            elif group_type == 'atom_to_residue':
#                ## SEEING IF EITHER DONOR AND ACCEPTOR MATCHES
#                if (donor_atom_index == group_1 and acceptor_group_name == group_2) or \
#                   (donor_atom_index == group_2 and acceptor_group_name == group_1) or \
#                   (acceptor_atom_index == group_1 and donor_group_name == group_2) or \
#                   (acceptor_atom_index == group_2 and donor_group_name == group_1):
#                       ## ADDING ONE TO THE HBOND RESIDUES
#                       total_hbond_residues += 1
#            ## WOULD LIKE TO DISTINGUISH DONOR AND ACCEPTOR
#            elif group_type == 'group_1_donor':
#                ## SEEING IF DONOR 
#                if (donor_atom_index == group_1 and acceptor_group_name == group_2) or \
#                   (donor_atom_index == group_2 and acceptor_group_name == group_1) or \
#                   (acceptor_atom_index == group_1 and donor_group_name == group_2) or \
#                   (acceptor_atom_index == group_2 and donor_group_name == group_1):
#                       ## ADDING ONE TO THE HBOND RESIDUES
#                       total_hbond_residues += 1
                       

        ## PRINTING
        if self.debug_mode is True:
            if group_type == 'atom_to_residue':
                group_1_name = traj.topology.atom(group_1).residue.name + '-' + traj.topology.atom(group_1).name
            else:
                group_1_name = group_1
        
            print("TOTAL HBONDS BETWEEN %s AND %s is: %d"%(group_1_name, group_2, total_hbond_residues) )
        
        return total_hbond_residues 
    
    ### FUNCTION TO FIND ALL COMBINATIONS OF SOLUTE-SOLVENT
    def find_solute_solvent_combinations(self):
        '''
        The purpose of this function is to find all solute-solvent combinations, which can allow us to find all contributions between solute-solvent systems
        INPUTS:
            self: class objects
        OUTPUTS:
            self.residue_list: [list] list of all residues
                e.g. ['HOH', ...]
            self.residue_combinations: [list] set of all unique residue combinations
                e.g.
                    [['DIO', 'PDO'],
                     ['PDO', 'PDO'],
        '''
        ## GENERATING COMBINATIONS (INCLUDING SELF TERMS)
        residue_combinations = list(set(list(itertools.combinations(self.residue_list, 2))))
        ## INCLUDING SOLVENTS WITH ITSELF
        solvent_solvent_combinations = [ [each_solvent, each_solvent] for each_solvent in self.solvent_name]
        ## CLEANING UP RESIDUE COMBINATIONS TO BE A LIST OF LIST
        self.residue_combinations = [ [each_combo[0], each_combo[1]] for each_combo in residue_combinations] + \
                                    solvent_solvent_combinations
        return
    
    ### FUNCTION TO FIND TOTAL HYDROGEN BONDS BETWEEN ALL RESIDUES
    def find_h_bonds_btn_all_residues(self, traj, index, hbonds):
        '''
        The purpose of this function is to find hydrogen bonds between all residues
        INPUTS:
            self: class objects
            traj: trajectory from md.traj 
            index: index to store in
            hbonds: [np.array, shape=(number_h_bonds, 3)] Triplet of the hydrogen bonding donor to acceptor
        OUTPUT:
            self.hbond_btn_residues: [dict] Hydrogen bonding between residues
            self.hbond_solute_donor: [dict] (OPTIONAL) Hydrogen bonding between solute acceptor atoms to solvents
        '''
        ## LOOPING THROUGH COMBINATIONS LIST
        for each_residue_combination in self.residue_combinations:
            ## FINDING TOTAL HYDROGEN BONDING BETWEEN RESIDUES
            total_hbond_residues = self.find_total_num_hbond_btn_groups(traj, hbonds, 
                                                            group_1=each_residue_combination[0], group_2 = each_residue_combination[1])
            ## STORING TOTAL HYDROGEN BONDING
            self.hbond_btn_residues[each_residue_combination[0] + '-' + each_residue_combination[1]][index] = total_hbond_residues
        
#        ## INCLUDE SOLUTE-SOLVENT HYDROGEN BONDING IF NECESSARY
#        if self.want_solute_donor_breakdown is True:
#            ## LOOPING THROUGH EACH KEY IN COMBINATIONS
#            for each_solute_atom_index in self.acceptors_solute:
#                ## LOOPING THROUGH SOLVENTS
#                for each_solvent in self.solvent_name:
#                    ## FINDING NAME OF THE ATOM
#                    atom_name = traj.topology.atom(each_solute_atom_index).name
#                    ## FINDING TOTAL HYDROGEN BONDING
#                    total_hbond_residues = self.find_total_num_hbond_btn_groups(traj, hbonds, 
#                                                            group_1=each_solute_atom_index, group_2 = each_solvent, group_type = 'atom_to_residue')
#                    self.hbond_solute_donor[self.solute_name +'-' + atom_name + '-' + each_solvent][index] = total_hbond_residues
        ## INCLUDE SOLUTE-SOLVENT ACCEPTOR-DONOR INFORMATION
        if self.want_solute_acceptor_donor_info is True:
            ## LOOPING THROUGH EACH SOLUTE
            for idx, each_solute_atom_index in enumerate(self.acceptors_solute):
                ## FINDING NAME OF THE ATOM
                atom_name = self.acceptors_solute_names[idx]
                ## LOOPING THROUGH SOLVENTS
                for each_solvent in self.solvent_name:
                    ## FINDING RESIDUE
                    current_solvent_index = self.residue_dict_atoms[each_solvent]
                    ## COMPUTING ACCEPTOR AND DONOR
                    num_donor, num_acceptor = self.find_num_acceptor_and_donor( hbonds = hbonds, 
                                                                                atom_index = each_solute_atom_index,
                                                                                acceptable_atom_indices = current_solvent_index)
                    ## APPENDING
                    self.solute_acceptor_donor_hbond[self.solute_name +'-' + atom_name + '-' + each_solvent]['num_donor'][index] = num_donor
                    self.solute_acceptor_donor_hbond[self.solute_name +'-' + atom_name + '-' + each_solvent]['num_acceptor'][index] = num_acceptor
                
        return
    
    ### FUNCTION TO FIND NUM DONOR AND ACCEPTOR
    @staticmethod
    def find_num_acceptor_and_donor( hbonds, atom_index, acceptable_atom_indices = None):
        '''
        The purpose of this function is to find the acceptor and donors given hydrogen bonds 
        and a specific atom index. 
        INPUTS:
            hbonds: [np.array]
                hydrogen bonding array
            atom_index: [int]
                atom index you are interested in computing number of acceptors and donor
            acceptable_atom_indices: [np.array, default=None]
                acceptable atom indices that you want to check. If None, we will find all possible number of acceptors and donors
        OUTPUTS:
            num_donor: [int]
                number of donor bonds
            num_acceptor: [int]
                number of acceptor bonds
        '''
        ## FINDING RELEVANT HBONDS
        relevant_hbonds = hbonds[np.where(hbonds == atom_index)[0]]
        
        ## FINDING NUM DONOR AND NUM ACCEPTOR
        if acceptable_atom_indices is None:
            num_donor = (relevant_hbonds[:,0] == atom_index).sum()
            num_acceptor = (relevant_hbonds[:,2] == atom_index).sum()
        else:
            num_donor = ((relevant_hbonds[:,0] == atom_index) & (np.isin( relevant_hbonds[:,2], acceptable_atom_indices) ) ).sum()
            num_acceptor = ((relevant_hbonds[:,2] == atom_index) & (np.isin( relevant_hbonds[:,0], acceptable_atom_indices) ) ).sum()
        return num_donor, num_acceptor

    ### FUNCTION FOR A SINGLE LOOP
    def hbond_single_loop(self, traj, index=0):
        '''
        The purpose of this function is to run hydrogen bonding analysis for a single loop
        INPUTS:
            self: 
                class object
            traj: [obj]
                trajectory from md.traj
            index: [int, default=0]
                index of that trajectory
        OUTPUTS:
            self.hbond_total: [int]
                total number of hydrogen bonds
        '''
        ## DEFINING CURRENT TRAJECTORY
        each_traj = traj[index]
        print("RUNNING HBOND ANALYSIS FOR %d"%(index))
        ## PRINTING
        if (index % self.frame_print_freq) == 0:
            print("Calculating hydrogen bonds for %d ps of %d ps"%(each_traj.time[0], self.traj_time[-1]))
            start_time = time.time()
        ## CALCULATING HYDROGEN BONDS
        hbonds = self.calc_hydrogen_bonds(each_traj)
        ## STORING HBONDS (TEMP)
        # self.hbonds=hbonds
        ## STORING TOTAL
        self.hbond_total[index] = len(hbonds)
        ## CALCULATING TOTAL HYDROGEN BONDS BETWEEN RESIDUES
        self.find_h_bonds_btn_all_residues(each_traj, index, hbonds )  
        
        if (index % self.frame_print_freq) == 0:
            if self.debug_mode is True:
                print("--> TOTAL HBONDS: %d"%(len(hbonds)))
            ## CALCULATING TIME AND PRINTING
            total_time = time.time()-start_time
            ## CONVERSION TO HOURS, MINUTES, SECONDS
            h, m, s = convert2HoursMinSec(total_time)
            print("--> TOTAL TIME FOR SINGLE FRAME: %d hours, %d min, %d seconds"%(h, m, s))
            ## CALCULATING PROJECTED TIME OF COMPLETION
            num_frames_left = self.traj_length - index
            proj_time = total_time * num_frames_left
            h, m, s = convert2HoursMinSec(proj_time)
            print("--> PROJECTED TIME OF COMPLETION: %d hours, %d min, %d seconds"%(h,m,s))
        return index


#%% MAIN SCRIPT
if __name__ == "__main__":
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"190207-PDO_Mostlikely_Configs_10000" # Analysis directory
    # analysis_dir=r"180316-ACE_PRO_DIO_DMSO"
    specific_dir="PDO\\Mostlikely_433.15_6_nm_PDO_100_WtPercWater_spce_Pure" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    specific_dir="PDO\\Mostlikely_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="ACE/mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir=r"Planar_310.15_ROT_TMMA_10x10_CHARMM36_withGOLP" # Directory within analysis_dir
    path2AnalysisDir=checkPath2Server(r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir) # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod_structure.pdb" # Structural file <-- must be a pdb file!
    xtc_file=r"mixed_solv_prod_10_ns_whole_center_150000.xtc"
    xtc_file=r"mixed_solv_prod_10_ns_whole_center_190000.xtc"
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    ### SHORTING THE TRAJECTORY
    traj_data.traj = traj_data.traj
    
    #%%
    
    ### DEFINING INPUT DATA
    input_details={ 'traj_data'          : traj_data,     
                    'solute_name'        : 'PDO',                                   # Solute of interest
                    'solvent_name'       : ['HOH', 'DIO', 'GVLL'],                  # Solvents you want radial distribution functions for
#                     'want_solute_donor_breakdown': True,                            # True if you want solute donor breakdown
                    'want_serial': False,
                    'debug_mode' : False,
                    'want_solute_acceptor_donor_info': True,
                    }
    #%%
    
    ### RUNNING CLASS
    hbond = calc_hbond(**input_details)
    
    
    
    #%%
    # hbond.compute(traj=traj_data.traj,index =0 )
    
    
    #%%
    # print(hbond.solute_acceptor_donor_hbond)
    # hbond.hbond_btn_residues

    
    #%%

    #%%
    
    ### FUNCTION TO DEAL WITH MULTIPROCESSING -- NOTE, ONLY WORKS WITH SINGLE VALUE INDEXES
    def parallel_for_loop( function, args, input_index, num_cores = None):
        '''
        The purpose of this function is to parallel for loop
        INPUTS:
            function: [func]
                function you want to run through a multiple for-loop
            args: [tuple]
                arguments in the form of a tuple
            input_index: [list]
                list of index you want, e.g. range(10)
            num_cores: [idx, default=None]
                number of cores you want to parallelize
        OUTPUTS:
            results: [list]
                list of results outputted from each function in the for loop
        '''
        ## IMPORTING MULTIPROCESSING TOOLS
        from joblib import Parallel, delayed
        import multiprocessing
        ## COMPUTING NUMBER OF CORES
        if num_cores == None:
            num_cores = multiprocessing.cpu_count()
        ## RUNNING "multiprocessing" vs. "threading"
        results = Parallel(n_jobs=num_cores, backend="threading")(delayed(function)(args,i) for i in input_index)
        return results
#    
        
    
    total_index = 100
    start_time = time.time()
    ### RUNNING PARALLEL FOR LOOP
    results = parallel_for_loop(function = hbond.compute, 
                                args = (traj_data.traj), 
                                input_index = range(total_index), 
                                num_cores = 3)# 3None
    
    elapsed_time = time.time() - start_time
    print("PARALLEL TIME")
    print(elapsed_time)
    
    ## PRINTING
    print(hbond.hbond_btn_residues['PDO-HOH'][0:total_index])
    
    
    
    start_time = time.time()
    ## TESTING SLOW FOR LOOP
    for i in range(total_index):
        results = hbond.compute( traj=traj_data.traj, index = i )
    elapsed_time = time.time() - start_time
    print("FOR LOOP TIME")
    print(elapsed_time)
    
    ## PRINTING
    print(hbond.hbond_btn_residues['PDO-HOH'][0:total_index])
    
    
    #%%
#    
#    total_index = 10
#    
#    
#    
#    ### FINDING DEPTH OF DICTIONARY
#    def depth(d, level=1):
#        if not isinstance(d, dict) or not d:
#            return level
#        return max(depth(d[k], level + 1) for k in d)
#    
#    ## GETTING THE DEPTH
#    test = {}
#    current_depth = depth( hbond.solute_acceptor_donor_hbond)
#    test_depth = depth(test)
#    
#    ### FUNCTION TO PUT THE VALUES BACK
#    def place_results_back_into_class(my_class, results):
#        '''
#        The purpose of this function is to place results taken from parallel for-loop back into
#        a class. 
#        INPUTS:
#            my_class: [obj]
#                class that you want to put your results back in
#            results: [list]
#                list of results coming from the parallel process
#        '''
#        
#        ## LOOPING THROUGH EACH RESULT
#        for idx,each_result in enumerate(results):
#            ## LOOPING THROUGH EACH ATTRIBUTE KEY
#            for each_key in each_result:
#                ## DEFINING CURRENT RESULTS
#                current_result = each_result[each_key]
#                ## FINDING THE ATTRIBUTE IN CLASS
#                current_attribute = my_class.__getattribute__(each_key)
#                ## FINDING DEPTH OF THE CURRENT ATTRIBUTE
#                depth_current_attribute = depth(current_attribute)
#                ## DEALING WITH WHEN THE ATTIBUTES
#                if depth_current_attribute  == 2:
#                    ## LOOPING THROUGH THE KEYS
#                    for attribute_key in current_result:
#                        ## UPDATING CURRENT ATTRIBUTE VALUES
#                        current_attribute[attribute_key][idx] = current_result[attribute_key]
#                        
#                        ## SINGLE DEPTH, ALLOWS US TO PUT ONE TO ONE
#                        my_class.__setattr__( each_key, current_attribute )
#        return
#    
#    place_results_back_into_class( my_class = hbond,
#                                   results = results)
#    
#    # hbond.hbond_btn_residues['PDO-DIO'][0:10]
#    # [1, 2, 2, 2, 1, 0, 1, 2, 1, 1]
#    
#    #%%
#    
#    
#    
#    ### FUNCTION TO PLACE RESULTS BACK
#    def recursive_place_results_back( current_result_dict, current_attribute_dict, index ):
#        ## LOOPING THROUGH ITEMS
#        for each_key, v in current_result_dict.items():
#            print(each_key)
#            if isinstance(v, dict):
#                print("TRUE")
#                recursive_place_results_back( current_result_dict[each_key], current_attribute_dict, index  )
#            else:
#                print("FALSE")
#                current_attribute_dict[each_key][index] = v
#        # return current_attribute_dict
#    
#    index = 0
#    current_variable = 'hbond_btn_residues'
#    current_result_dict = results[index][current_variable]
#    current_attribute_dict = hbond.__getattribute__(current_variable)
#    # current_attribute_dict['PDO-HOH'][0:11]
#    
#    
#    #%%
#    ## RECURIVELY PLACE BACK
#    recursive_place_results_back( current_result_dict = current_result_dict,
#                                  current_attribute_dict = current_attribute_dict,
#                                  index = index)
#    
#    
#    
#    
#    #%%
#    
#    
#    def myprint(d):
#      for k, v in d.items():
#        if isinstance(v, dict):
#          myprint(v)
#        else:
#          print("{0} : {1}".format(k, v))
#    
#    
#    myprint( current_result_dict ) # hbond.solute_acceptor_donor_hbond
#    
#    
#    #%%
#    
#                
#                
#                
#                
#                ## RENUMBERING THE RESULT setattr
#                
#    # DATA STRUCTURE REFERENCE: https://stackoverflow.com/questions/17107973/python-tree-like-implementation-of-dict-datastructure
#        
#    
#    
#    
#    
#    #%%
#
#    
#
#    num_cores =3   # multiprocessing.cpu_count()
#    parallel_stop_watch = time.time()
#    input_index = range(total_index)
#    results = Parallel(n_jobs=num_cores, backend="threading")(delayed(hbond.compute)(traj_data.traj,i) for i in input_index)
#    elapsed_time = time.time() - parallel_stop_watch
#    print("PARALLEL LOOP TIME")
#    print(elapsed_time)
#    #%%
#    
#    
#    total_index = 10
#    start_time = time.time()
#    ## TESTING SLOW FOR LOOP
#    for i in range(total_index):
#        results = hbond.compute( traj=traj_data.traj, index = i )
#    elapsed_time = time.time() - start_time
#    print("FOR LOOP TIME")
#    print(elapsed_time)
#    
#    #%%

    
    
    
    #%%
    
    ### FUNCTION TO DEAL WITH MULTIPROCESSING
    def parallel_for_loop( function, args, input_index, num_cores = None):
        '''
        The purpose of this function is to parallel for loop
        INPUTS:
            function: [func]
                function you want to run through a multiple for-loop
            args: [tuple]
                arguments in the form of a tuple
            input_index: [list]
                list of index you want
            num_cores: [idx, default=None]
                number of cores you want to parallelize
        OUTPUTS:
            results: [list]
                list of results outputted from each function in the for loop
        '''
        ## COMPUTING NUMBER OF CORES
        if num_cores == None:
            num_cores = multiprocessing.cpu_count()
#        
#        ## RUNNING 
#        results = Parallel(n_jobs=num_cores)(delayed(function)(args,i) for i in input_index)
#        return results
#    
#    start_time = time.time()
#    results = parallel_for_loop( function = hbond.compute, 
#                                 args = (traj_data.traj), 
#                                 input_index = range(10), 
#                                 num_cores = None )
#    elapsed_time = time.time() - start_time
#    print("PARALLEL LOOP TIME")
#    print(elapsed_time)
    
    
#    inputs = range(10) 
#    def processInput(i):
#        return i * i
#     
#    num_cores = multiprocessing.cpu_count()
#         
#    results = Parallel(n_jobs=num_cores)(delayed(hbond.compute)(traj_data.traj,i) for i in inputs)
#    print(num_cores)
#    print(results)
    
    
#    import multiprocessing
#    pool = multiprocessing.Pool(4)
#    out1, out2, out3 = zip(*pool.map(calc_hbond.compute(), range(0, 10 * offset, offset)))
    
#    ## RUNNING PARALLEL
#    results = parallel.parallel( traj = traj_data.traj, 
#                       func = calc_hbond,
#                       args = input_details)
#    ## PRINTING RESULTS
#    print(results.results)
#    
    #%%
    
#    import multiprocessing
#    
#    class Test(object) :
#    
#        def __init__(self):
#           self.manager = multiprocessing.Manager()
#           self.some_list = self.manager.list()  # Shared Proxy to a list
#    
#        def method(self):
#            self.some_list.append(123) # This change won't be lost
#    
#    
#    if __name__ == "__main__":
#        t1 = Test()
#        t2 = Test()
#        pr1 = multiprocessing.Process(target=t1.method)
#        pr2 = multiprocessing.Process(target=t2.method)
#        pr1.start()
#        pr2.start()
#        pr1.join()
#        pr2.join()
#        print(t1.some_list)
#        print(t2.some_list)

    
    #%%
    ''' function to find all hydrogens on acceptor atoms
    ### FINDING DONORS
    acceptors_solute = hbond.acceptors_solute
    acceptors_solute_names = hbond.acceptors_solute_names
    ## DEFINING BONDS
    hbond_bonds = hbond.bonds
    
    ## DEFINING EMPTY LIST
    solute_donor_dict = {}
    
    ## LOOPING THROUGH EACH ACCEPTOR
    for idx, acceptor_solute_index in enumerate(acceptors_solute):
        ## LOOPING THROUGH EACH SOLVENT
        for each_solvent in self.solvent_name:
        ## FINDING INDEX
        atoms_bonded_index = [ each_atom_index for each_atom_index in hbond_bonds[np.where(hbond_bonds == acceptor_solute_index)[0]].flatten() 
                                        if each_atom_index != acceptor_solute_index ]
        
        ## FINDING H-BOND INDICES
        hydrogen_bonded_indices = [each_atom_index for each_atom_index in atoms_bonded_index
                                        if traj_data.traj.topology.atom(each_atom_index).element.symbol == 'H']
        
        ## STORING
        solute_donor_dict[acceptors_solute_names[idx]] = hydrogen_bonded_indices[:]
    
    '''
    
    