# -*- coding: utf-8 -*-
"""
sam_structure.py
This script analyzes the structure of the SAM for each frame. Structural characteristic examples are shown below:
    tilt angle
    monolayer thickness
    trans/cis ratio
    hexatic order
    minimum distance between tail chains

CLASSES:
    sam_structure: characterizes the nanoparticle structure and calculates various quantities, e.g.
        trans ratio, distances between end groups, etc.

CREATED ON: 04/09/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
**UPDATES**
20180420 - AKC - Updated draft of nanoparticle structure that can measure minimum distance between tail grous and trans ratio
20180702 - AKC - Updated "calc_dihedral_ensemble_avg_ratio" to correctly account for standard deviation in trans ratio
20180828 - BCD - Updated from nanoparticle_structure.py to sam_structure.py
"""
### IMPORTING MODULES
from MDDescriptors.application.nanoparticle.monolayer_info import monolayer_info
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
import sys
import glob
import time

### DEFINING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import GOLD_ATOM_NAME, LIGAND_HEAD_GROUP_ATOM_NAME

##############################################
### CLASS TO STUDY NANOPARTICLE STRUCTURE ####
##############################################
class sam_structure:
    '''
    The purpose of this function is to study the nanoparticle structure, in particular:
        tilt angle
        monolayer thickness
        trans/cis ratio
        hexatic order
        minimum distance between tail chains
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        structure_types:   [list] list of strings that include what type of job you want, e.g.
            'all': finds all possible structural types (e.g. trans_ratio, etc.)
            None: Turns off all secondary calculations of strutural types.
            ## LIST OF STRUCTURAL TYPES
                'trans_ratio': finds the trans ratio of the ligands based on heavy atoms
                'distance_end_groups': finds the end groups and calculates the minimum distance between each head group
        separated_ligands: [logical] True if you want to avoid itp file. This assumes that your gro defines ligands separately. (e.g. more than one residue for ligands)
        save_disk_space: [logical] True if you want to save disk space by removing the following variables: [Default: True]
            Turning off: terminal_group_distances, terminal_group_min_distance_matrix, dihedrals
        
    OUTPUTS:        
        ## TRAJECTORY INFORMATION
            self.traj_time_list:[list] trajectory times as a list in picoseconds
            self.traj_time_len: [int] total number of frames
            
        ## -------- IF structure_types == None, then the following are not calculated! ------ ##
        ## GAUCHE/TRANS DIHEDRAL ANALYSIS
            self.dihedral_list:[list] list of list of all the possible dihedrals
            self.dihedrals: [np.array, shape=(time_frame, dihedral_index)] {EXPENSIVE} dihedral angles in degrees from 0 to 360 degrees
            dihedral_gauche_avg: [float] Ensemble average of gauche dihedrals
            dihedral_gauche_std: [float] Standard deviation of the average of gauche dihedrals
            dihedral_trans_avg: [float] Ensemble average of trans dihedrals
            dihedral_trans_std: [float] Standard deviation of the average of gauche dihedrals
            dihedral_ligand_trans_avg_per_frame: [np.array, shape=(num_frames,num_ligands)] {EXPENSIVE} trans ratio of each ligand per frame 
            
        ## DISTANCE BETWEEN TERMINAL GROUP ANALYSIS
            self.terminal_group_index: [list] last index of each ligand in the heavy atom index
            self.terminal_group_atom_pairs: [np.array, shape=(num_pairs, 2)] contains list of atom pairs, e.g. [[1, 2]], etc. The total length should be the total number of end groups + total - 1 + ... + 0
            self.terminal_group_distances: [np.array, shape=(time_frame, atom_pair_index)] {EXPENSIVE} distances between the terminal groups for each frame
            self.terminal_group_min_distance_matrix: [np.array, shape=(time_length, index)] {EXPENSIVE} minimum distance for each group in each frame
            self.terminal_group_min_dist_avg: [float] average minimum distance
            self.terminal_group_min_dist_std: [float] standard deviation of the minimum distances

    FUNCTIONS:
        ## GAUCHE/TRANS DIHEDRAL FUNCTIONS
            find_dihedral_list: [staticmethod] finds dihedral list based on available heavy atoms
            calc_dihedral_angles: calculates dihedral angles given dihedral index list
            calc_dihedral_ensemble_avg_ratio: calculates ensemble average of trans/gauche ratio
            calc_dihedral_trans_per_ligand: [staticmethod] calculates trans-ratio per ligand per frame basis
        ## DISTANCE BETWEEN TERMINAL GROUP FUNCTIONS
            find_all_terminal_groups: finds all terminal groups
            calc_terminal_group_distances: calculates distances between terminal groups
            calc_terminal_group_min_dist: calculates minimum distances based on the distance array
        
    ALGORITHM:
        - Find all the residue index of the ligands
        - Find the center of the nanoparticle for all frames
        tilt angle:
            - Find the sulfur atom index
            - Find the last carbon atom index
            - Calculate the tilt angle
        gauche/trans dihedral:
            - Find all the heavy atoms of the ligand (i.e. ligand backbone)
            - Find all dihedral angles for each heavy atom
            - Calculate an average cis/dihedral ratio
        monolayer thickness:
            - Find sulfur atom index
            - Find last carbon atom index
            - Calculate the monolayer thickness based on the 
            
    '''
    ### INITIALIZING
    def __init__(self, traj_data, ligand_names = ['BUT'], itp_file = None, structure_types = 'all', separated_ligands = False, save_disk_space = True ):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### STORING INITIAL VARIABLES
        self.save_disk_space = save_disk_space                             # Used to remove unnecessarily large variables
        self.separated_ligands = separated_ligands                         # Used to avoid itp file generation
        
        ## CHECKING STRUCTURAL TYPES
        available_structure_types = [ 'monolayer_thickness', 'trans_ratio', 'distance_end_groups', 'hexatic_order', 'autocorrelation' ]
        if structure_types == 'all':
            print("We are are running all possible structurals")
            print("Structural types: %s"%(', '.join(available_structure_types) ) )
            self.structure_types = available_structure_types[:]
        elif structure_types == None:
            print("Since structure types is set to None, Nothing will be calculated")
            self.structure_types = []
        else:
            ## FINDING STRUCTURE TYPES (SCREENING ALL INNCORRECT STRUCTURE TYPES)
            self.structure_types = [ each_structure for each_structure in structure_types if each_structure in available_structure_types ]
            print("Final structure type list: %s"%(', '.join(self.structure_types) ))
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## STORING INFORMATION ABOUT THE TRAJECTORY
        self.traj_time_list = traj.time
        self.traj_time_len  = len(self.traj_time_list)
        
        ## GET MONOLAYER INFORMATION
        monolayerInfo = monolayer_info( traj_data, ligand_names = ligand_names, itp_file = itp_file, separated_ligands = separated_ligands, save_disk_space = save_disk_space )
        
        ###########################
        ### MONOLAYER THICKNESS ###
        ###########################
        if 'monolayer_thickness' in self.structure_types:
            ## FIND NUMBER OF LIGANDS
            self.num_ligands = monolayerInfo.total_ligands
            ## CALCULATING EACH LIGAND VECTOR
            self.ligand_vectors = self.calc_ligand_vectors( traj, ligand_atom_list = monolayerInfo.ligand_heavy_atom_index )
            ## CALCULATING ENSEMBLE AVERAGE LIGAND LENGTH
            self.calc_ligand_length_ensemble_avg( ligand_vectors = self.ligand_vectors )
            ## CALCULATING ENSEMBLE AVERAGE THICKNESS
            self.calc_monolayer_thickness_ensemble_avg( ligand_vectors = self.ligand_vectors )
            
        ###############################
        ### DIHEDRAL ANGLE ANALYSIS ###
        ###############################
        if 'trans_ratio' in self.structure_types:
            ## FUNCTION TO FIND ALL THE DIHEDRAL ANGLES
            self.dihedral_list, self.dihedral_reference_list = self.find_dihedral_list( ligand_atom_list = monolayerInfo.ligand_heavy_atom_index )
            ## FUNCTION TO MEASURE DIHEDRAL ANGLES
            self.dihedrals = self.calc_dihedral_angles( traj, dihedral_list = self.dihedral_list )            
            ## CALCULATING ENSEMBLE AVERAGE
            self.dihedral_gauche_avg, self.dihedral_gauche_std, self.dihedral_trans_avg, self.dihedral_trans_std = self.calc_dihedral_ensemble_avg_ratio( dihedrals = self.dihedrals )
            ## FUNCTION TO RUN TRANS RATIO IN A PER LIGAND BASIS
            self.dihedral_ligand_trans_avg_per_frame = self.calc_dihedral_trans_per_ligand( dihedrals =  self.dihedrals,  reference_list = self.dihedral_reference_list )          
        
        ###########################################
        ### MINIMUM DISTANCE BETWEEN END GROUPS ###
        ###########################################
        if 'distance_end_groups' in self.structure_types:
            ## FINDING ALL TERMINAL GROUPS
            self.find_all_terminal_groups( ligand_atom_list = monolayerInfo.ligand_heavy_atom_index )
            ## FINDING TERMINAL GROUP DISTANCES
            self.calc_terminal_group_distances( traj )
            ## FINDING MINIMUM DISTANCES
            self.calc_terminal_group_min_dist()
        
        ##############################
        ### HEXATIC ORDER ANALYSIS ###
        ##############################
        if 'hexatic_order' in self.structure_types:
            if 'distance_end_groups' not in self.structure_types:
                ## FINDING ALL TERMINAL GROUPS
                self.find_all_terminal_groups( ligand_atom_list = monolayerInfo.ligand_heavy_atom_index )
            
            self.calc_hexatic_order_ensemble_avg( traj )                
          
        ################################
        ### Autocorrelation function ###
        ################################
        if 'autocorrelation' in self.structure_types:
#            if 'monolayer_thickness' in self.structure_types:
#                self.monolayer_thickness_autocorr = self.autocorrelate( self.avg_monolayer_thickness_per_frame, 'monolayer_thickness' )

            if 'trans_ratio' in self.structure_types:
                self.trans_ratio_autocorr = self.autocorrelate( self.dihedral_ligand_trans_avg_per_frame.mean(axis=1), 'trans_ratio' )
                
            if 'hexatic_order' in self.structure_types:
                self.hexatic_order_autocorr = self.autocorrelate( self.avg_hexatic_order_per_frame, 'hexatic_order' )

        ## PRINTING SUMMARY
        self.print_summary()
        
        ## TURNING OFF OBJECTS IF NECESSARY
        if save_disk_space is True:
            self.terminal_group_distances, self.terminal_group_min_distance_matrix, self.dihedrals = [], [], []
                
        return
    
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function prints the summary ''' 
        if 'monolayer_thickness' in self.structure_types:
            print("\n----- MONOLAYER THICKNESS ANALYSIS -----")
            print("TOTAL NUMBER OF LIGANDS: %d"%( self.num_ligands ) )
            print("AVG LIGAND LENGTH: %.2f +- %.2e"%( self.ligand_length_avg, self.ligand_length_std ) )
            print("AVG MONOLAYER THICKNESS: %.2f +- %.2e"%( self.monolayer_thickness_avg, self.monolayer_thickness_std ) )
            
        if 'trans_ratio' in self.structure_types:
            print("\n----- GAUCHE/TRANS RATIO ANALYSIS -----")
            print("TOTAL NUMBER OF DIHEDRALS: %d"%( len(self.dihedral_list) ) )
            print("AVG TRANS: %.2f +- %.2e"%( self.dihedral_trans_avg, self.dihedral_trans_std ) )
            print("AVG GAUCHE: %.2f +- %.2e"%( self.dihedral_gauche_avg, self.dihedral_gauche_std) )
            
        if 'distance_end_groups' in self.structure_types:
            print("\n----- DISTANCE BETWEEN TERMINAL GROUPS ANALYSIS -----")
            print("TOTAL NUMBER OF TERMINAL GROUPS: %d"%(len(self.terminal_group_index)))
            print("MINIMUM DISTANCE: %.2f +- %.2e nm"%(self.terminal_group_min_dist_avg, self.terminal_group_min_dist_std))
    
        if 'hexatic_order' in self.structure_types:
            print("\n----- HEXATIC ORDER ANALYSIS -----")
            print("AVG HEXATIC ORDER: %.2f +- %.2e"%(self.hexatic_order_avg, self.hexatic_order_std))
            
    ### FUNCTION TO CALCULATE LIGAND VECTORS
    @staticmethod
    def calc_ligand_vectors( traj, ligand_atom_list, periodic = True ):
        '''
        This function calculates the vector of each ligand in the monolayer
        INPUTS:
            traj: trajectory from md.traj
            ligand_atom_list: [list] list of heavy atoms [[ 1, 2, 3, 4 ], ... ]
            periodic: [logical] True if you want periodic boundaries
        OUTPUTS:
           ligand_vectors: [np.array, shape=(time_frame, total_ligands, 3)] ligand vector from head atoms to tail atoms
        '''
        head_n_tail_atoms = np.array([ [ ligand[0], ligand[-1] ] for ligand in ligand_atom_list ])
        ligand_vectors = md.compute_displacements( traj, head_n_tail_atoms, periodic = periodic )
        return ligand_vectors

    ### FUNCTION TO CALCULATE LIGAND LENGTHS
    def calc_ligand_length_ensemble_avg( self, ligand_vectors ):
        '''
        This function calculates the average length of the ligands in the monolayer
        INPUTS:
            ligand_vectors: [np.array, shape=(time_frame, total_ligands, 3)] ligand vector from head atoms to tail atoms
        OUTPUTS:
            avg_ligand_length_per_frame: [np.array, shape=(time_frame, 1)] surface average ligand length per frame
            std_ligand_length_per_frame: [np.array, shape=(time_frame, 1)] standard dev. in the surface average ligand length per frame
            ligand_length_avg: [float] Ensemble average of ligand length
            ligand_length_std: [float] Standard deviation of the average of ligand length
        '''
        ligand_lengths = np.sqrt( np.sum( ligand_vectors**2, axis = 2 ) )
        self.avg_ligand_length_per_frame = ligand_lengths.mean( axis = 1 )
        self.std_ligand_length_per_frame = ligand_lengths.std( axis = 1 )
        self.ligand_length_avg = self.avg_ligand_length_per_frame.mean()
        self.ligand_length_std = self.avg_ligand_length_per_frame.std()    

    ### FUNCTION TO CALCULATE MONOLAYER THICKNESS
    def calc_monolayer_thickness_ensemble_avg( self, ligand_vectors ):
        '''
        This function calculates the average thickness of the monolayer
        INPUTS:
            ligand_vectors: [np.array, shape=(time_frame, total_ligands, 3)] ligand vector from head atoms to tail atoms
        OUTPUTS:
            avg_monolayer_thickness_per_frame: [np.array, shape=(time_frame, 1)] surface average monolayer thickness per frame
            std_monolayer_thickness_per_frame: [np.array, shape=(time_frame, 1)] standard dev. in the surface average monolayer thickness per frame
            monolayer_thickness_avg: [float] Ensemble average of monolayer thickness
            monolayer_thickness_std: [float] Standard deviation of the average of monolayer thickness
        '''
        ligand_heights = np.abs( ligand_vectors[:,:,2] )
        self.avg_monolayer_thickness_per_frame = ligand_heights.mean( axis = 1 )
        self.std_monolayer_thickness_per_frame = ligand_heights.std( axis = 1 )
        self.monolayer_thickness_avg = self.avg_monolayer_thickness_per_frame.mean()
        self.monolayer_thickness_std = self.avg_monolayer_thickness_per_frame.std()  
        
    ### FUNCTION TO FIND DIHEDRAL LIST
    @staticmethod
    def find_dihedral_list( ligand_atom_list ):
        '''
        The purpose of this function is to find all dihedral indices. 
        INPUTS:
            self: class object
            ligand_atom_list: [list] list of heavy atoms [[ 1, 2, 3, 4 ], ... ]
        OUTPUTS:
            dihedral_list:[list] list of list of all the possible dihedrals
            dihedral_reference_list: [list] dihedral reference list between ligand index and the dihedral index.
                e.g.:
                    [ [1,2,3], ...] <-- indicates that 1, 2, 3 are dihedrals belonging to ligand 0
        '''
        ## CREATING A BLANK LIST
        dihedral_list = []
        ## DEFINING COUNTER AS A REFERENCE
        counter = 0 ; dihedral_reference_list = []
        ## GENERATING DIHEDRAL LIST BASED ON HEAVY ATOMS
        for atom_list in ligand_atom_list:
            ## LOOPING THROUGH TO GET A DIHEDRAL LIST (Subtract owing to the fact you know dihedrals are 4 atoms)
            ## CREATING EMPTY ARRAY THAT CAN STORE DIHEDRAL INDEXES
            atom_dihedral_reference_list = []
            for each_iteration in range(len(atom_list)-3):
                ## APPENDING DIHEDRAL LIST
                dihedral_list.append(atom_list[each_iteration:each_iteration+4])
                ## STORING REFERENCE
                atom_dihedral_reference_list.append(counter)
                ## ADDING 1 TO COUNTER
                counter+=1
            ## AT THE END, STORE THE DIHEDRAL REFERENCE LIST
            dihedral_reference_list.append(atom_dihedral_reference_list)
        return dihedral_list, dihedral_reference_list
    
    ### FUNCTION TO CALCULATE THE DIHEDRALS
    @staticmethod
    def calc_dihedral_angles( traj, dihedral_list, periodic = True ):
        '''
        The purpose of this function is to calculate all the dihedral angles given a list
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            dihedral_list:[list] list of list of all the possible dihedrals
            periodic: [logical] True if you want periodic boundaries
        OUTPUTS:
           dihedrals: [np.array, shape=(time_frame, dihedral_index)]  dihedral angles in degrees from 0 to 360 degrees
        '''
        ## CALCULATING DIHEDRALS FROM MDTRAJ
        dihedrals = md.compute_dihedrals(traj, dihedral_list, periodic = periodic )
        # RETURNS NUMPY ARRAY AS A SHAPE OF: (TIME STEP, DIHEDRAL) IN RADIANS
        dihedrals = np.rad2deg(dihedrals) # np.array( dihedrals ) * 180 / np.pi # 
        dihedrals[ dihedrals < 0 ] = dihedrals[ dihedrals < 0 ] + 360  # to ensure every dihedral is between 0-360
        return dihedrals
    
    ### FUNCTION TO CALCULATE ENSEMBLE AVERAGE DIHEDRAL ANGLES
    @staticmethod
    def calc_dihedral_ensemble_avg_ratio( dihedrals ):
        '''
        The purpose of this function is to calculate the ensemble average of the dihedral angles
        INPUTS:
            self: class object
            dihedrals: [np.array, shape=(time_frame, dihedral_index)]  dihedral angles in degrees from 0 to 360 degrees
        OUTPUTS:
            dihedral_gauche_avg: [float] Ensemble average of gauche dihedrals
            dihedral_gauche_std: [float] Standard deviation of the average of gauche dihedrals
            dihedral_trans_avg: [float] Ensemble average of trans dihedrals
            dihedral_trans_std: [float] Standard deviation of the average of gauche dihedrals
        # NOTE: dihedral_gauche and trans should sum to 1.0. If not, there is something wrong!
        '''
        ## DEFINING DIHEDRALS
        d = dihedrals[:]
        
        ## TRANS RATIO OVER TIME
        dihedral_trans_over_time = np.mean((d < 240) & (d > 120), axis=1)
        dihedral_trans_avg = np.mean(dihedral_trans_over_time)
        dihedral_trans_std = np.std(dihedral_trans_over_time)
        
        ## GAUCHE RATIO OVER TIME
        dihedral_gauche_over_time = np.mean((d > 240) | (d < 120), axis=1)
        dihedral_gauche_avg = np.mean(dihedral_gauche_over_time)
        dihedral_gauche_std = np.std(dihedral_gauche_avg)
        
        return dihedral_gauche_avg, dihedral_gauche_std, dihedral_trans_avg, dihedral_trans_std
    
    ### FUNCTION TO FIND TRANS RATIO OF EACH LIGAND PER FRAME BASIS
    @staticmethod
    def calc_dihedral_trans_per_ligand( dihedrals, reference_list ):
        '''
        The purpose of this function is to find the trans-ratio per ligand basis per frame
        INPUTS:
            dihedrals: [np.array, shape=(n_frames, n_dihedrals)] dihedral angles for each and every frame
            reference_list: [list] list of ligand to dihedral index
        OUTPUTS:
            dihedral_ligand_trans_avg_per_frame: [np.array, shape=(num_frames, num_ligands)] trans-ratio for each ligand for each frame
        '''
        ## CONVERTING TO NUMPY ARRAY
        reference_list = np.array(reference_list)
        
        ## FINDING DIHEDRALS BASED ON LIGANDS
        dihedrals_ligand_basis = dihedrals[:, reference_list] ## SHAPE: NUM_FRAMES X NUM_LIGANDS X NUM_DIHEDRALS
        
        ## FINDING AVERAGE DIHEDRALS PER FRAME BASIS
        dihedral_ligand_trans_avg_per_frame = np.mean((dihedrals_ligand_basis < 240) & (dihedrals_ligand_basis > 120), axis=2) ## SHAPE: NUM_FRAMES X NUM_LIGANDS
        ## NOTE: If you average this, you should get the ensemble average trans ratio
        
        return dihedral_ligand_trans_avg_per_frame
    
    ### FUNCTION TO FIND THE END GROUPS
    def find_all_terminal_groups( self, ligand_atom_list ):
        '''
        The purpose of this function is to find the end group, which we assume is the last heavy atom
        INPUTS:
            self: class object
        OUTPUTS:
            self.terminal_group_index: [list] last index of each ligand in the heavy atom index
        '''
        self.terminal_group_index=[each_ligand[-1] for each_ligand in ligand_atom_list ]
        return
        
    ### FUNCTION TO CALCULATE ALL DISTANCES BETWEEN THE ATOMS
    def calc_terminal_group_distances( self, traj, periodic=True ):
        '''
        The purpose of this function is to calculate the terminal group distances
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            periodic: [logical] True if you want periodic boundary conditions to be accounted for
        OUTPUTS:
            self.terminal_group_atom_pairs: [np.array, shape=(num_pairs, 2)] contains list of atom pairs, e.g. [[1, 2]], etc. The total length should be the total number of end groups + total - 1 + ... + 0
            self.terminal_group_distances: [np.array, shape=(time_frame, atom_pair_index)] distances between the terminal groups for each frame
        '''
        ## GENERATING ATOM PAIR LIST
        self.terminal_group_atom_pairs = np.array([ [i_atom, j_atom] for index, i_atom in enumerate(self.terminal_group_index) for j_atom in self.terminal_group_index[index+1:]])
        ## CALCULATING DISTANCES
        self.terminal_group_distances = md.compute_distances(traj = traj,
                                                             atom_pairs = self.terminal_group_atom_pairs,
                                                             periodic = periodic
                                                             ) ## RETURNS (TIME_FRAME, ATOM_PAIR_INDEX)
        return
    
    ### FUNCTION TO FIND MINIMUM DISTANCES OF THE TERMINAL GROUPS
    def calc_terminal_group_min_dist( self, ):
        '''
        The purpose of this function is to find the minimum distances between groups.
        INPUTS:
            self: class object
        OUTPUTS:
            self.terminal_group_min_distance_matrix: [np.array, shape=(time_length, index)] minimum distance for each group in each frame
            self.terminal_group_min_dist_avg: [float] average minimum distance
            self.terminal_group_min_dist_std: [float] standard deviation of the minimum distances
        '''
        ## CREATING EMPTY MATRIX BASED ON TRAJECTORY LENGTH AND ATOMS
        self.terminal_group_min_distance_matrix = np.empty( (self.traj_time_len, len(self.terminal_group_index) ) )
        ## WILL OUTPUT MATRIX: (time_frame, terminal group index)
        ## MAIN IDEA IS TO GET MINIMUM DISTANCE IN EACH FRAME
        
        ## LOOPING OVER TERMINAL GROUPS
        for i, each_index in enumerate(self.terminal_group_index):
            ## FINDING THE INDICES FOR YOUR ATOM
            indices = np.argwhere(self.terminal_group_atom_pairs==each_index)[:,0]
            ## NOW, FINDING ALL THE CORRESPONDING DISTANCES
            terminal_atom_distances = self.terminal_group_distances[:, indices] # RETURNS NP ARRAY OF SHAPE ( TIME FRAME, N_ATOMS -1  )
            ## FINDING MINIMUM OF THE DISTANCES AS A VECTOR (CALCULATES MINIMUM DISTANCE BETWEEN ALL GROUPS FOR A SINGLE FRAME)
            min_distances = np.min(terminal_atom_distances, axis = 1) # RETURNS NP ARRAY OF SHAPE (TIME_FRAME, 1)
            ## STORING THE INFORMATION
            self.terminal_group_min_distance_matrix[:, i] = min_distances[:]
        
        ## FINDING ENSEMBLE AVERAGE AND STD
        self.terminal_group_min_dist_avg = np.mean(self.terminal_group_min_distance_matrix)
        self.terminal_group_min_dist_std = np.std(self.terminal_group_min_distance_matrix)
        
        return

    ### FUNCTION TO CALCULATE HEXATIC ORDER PARAMETER
    def calc_hexatic_order_ensemble_avg( self, traj, cutoff = 0.55, periodic = True ):
        '''
        This function calculates the ensemble average of the hexatic order
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            periodic: [logical] True if you want periodic boundary conditions to be accounted for
        OUTPUTS:
            avg_hexatic_order_per_frame: [np.array, shape=(time_frame, 1)] surface average hexatic order per frame
            std_hexatic_order_per_frame: [np.array, shape=(time_frame, 1)] standard dev. in the surface average hexatic order per frame
            hexatic_order_avg: [float] Ensemble average of hexatic order
            hexatic_order_std: [float] Standard deviation of the average of hexatic order
        # NOTE: hexatic order should be <=1.0. If not, there is something wrong!
        '''
        refVector = np.array([ 1, 0 ])
        magRef = np.sqrt( refVector[0]**2 + refVector[1]**2 )
        expTheta_perLigand = np.zeros( (self.traj_time_len, len(self.terminal_group_index)), dtype = 'complex' ) 
        for ndxAtom, tailAtom in enumerate( self.terminal_group_index ):
            # find nearest neighbors
            potential_neighbors = np.array([ [ ndx, tailAtom ] for ndx in self.terminal_group_index if ndx != tailAtom ])
            vector = md.compute_displacements( traj, potential_neighbors, periodic = periodic ) # only care about x and y displacements
            dist = np.sqrt( np.sum( vector**2., axis = 2 ) )
#            indices = np.unravel_index( np.argsort( dist, axis=1 ), dist.shape )
            # determine atoms in cutoff
            dist[ abs( dist ) > cutoff ] = 0.0
            mask = abs( dist ) > 0.0
            nNeighbors = np.sum( mask, axis = 1 )
            
            # calculate angle using dot product and determinant
            theta = np.zeros( dist.shape )
            expTheta = np.zeros( dist.shape, dtype = 'complex' )
            dotVec = vector[:,:,0] * refVector[0] + vector[:,:,1] * refVector[1]
            detVec = vector[:,:,0] * refVector[1] - vector[:,:,1] * refVector[0]
            theta[mask] = np.arccos( dotVec[mask] / ( dist[mask] * magRef ) )
            theta[ detVec < 0.0 ] = 2.0 * np.pi - theta[ detVec < 0.0 ]
            # compute order parameter
            expTheta[mask] = np.exp( 6j * theta[mask] )
#            expTheta[ :, ndxAtom ] += np.sum( np.exp( 6j * theta[indices][:,:6] ), axis = 1 ) / 6.
            expTheta_perLigand[ :, ndxAtom ][ nNeighbors > 0.0 ] += np.sum( expTheta[ nNeighbors > 0.0 ], axis = 1 ) / nNeighbors[ nNeighbors > 0.0 ]
            
        self.avg_hexatic_order_per_frame = np.mean( np.abs(expTheta_perLigand)**2., axis = 1 )
        self.std_hexatic_order_per_frame = np.std( np.abs(expTheta_perLigand)**2., axis = 1 )
        self.hexatic_order_avg = self.avg_hexatic_order_per_frame.mean()
        self.hexatic_order_std = self.avg_hexatic_order_per_frame.std()
    
    ### FUNCTION TO FIND TRANS RATIO OF EACH LIGAND PER FRAME BASIS
    def autocorrelate( self, data, name, dt = 500 ):
        '''
        '''
        print( 'calculating autocorrelation function for {:s}'.format( name ) )
        
        t_f = len(data) - 1
        simStep = self.traj_time_list[1] - self.traj_time_list[0]
        tau_s = np.arange( simStep*t_f, 0, -simStep )
        C_tau = []
        for tau in tau_s:
            ndx_tau = int( tau / dt )
            t_max = t_f - ndx_tau 
            C_tau.append( np.sum([ data[ii] * data[ii+ndx_tau] for ii in range(t_max) ]) / t_max )        
        
        newTau = np.array(range(int(len(tau_s)/dt)))
        avgC = np.array([ np.array(C_tau[int(dt*ii):int(dt*(ii+1))]).mean() for ii in newTau ]) - data.mean()**2.
        newTau = simStep * ( t_f - dt*newTau )

        results = np.array( [ [ tau, C ] for tau, C in zip( newTau, avgC ) ] )
        
        return results
    
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"unbiased" # Analysis directory
    category_dir="autocorrelation" # category directory
    specific_dir="sam_single_8x8_300K_butanethiol_water_nvt_CHARMM36" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    path2AnalysisDir=r"R:/simulations/" + analysis_dir + '/' + category_dir + '/' + specific_dir + '/' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    # xtc_file=r"sam_prod_10_ns_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    xtc_file=r"sam_prod.xtc"
    '''
    ### DIRECTORY TO WORK ON
    analysis_dir=r"180427-Planar_sams" # Analysis directory
    category_dir="Planar" # category directory
    specific_dir="Planar_310.15_K_butanethiol_10x10_CHARMM36_intffGold" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    path2AnalysisDir=r"R:\scratch\LigandBuilder\Analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    # xtc_file=r"sam_prod_10_ns_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    xtc_file=r"sam_prod_10_ns_whole.xtc"
    '''
    
    '''
    ### DIRECTORY TO WORK ON
    analysis_dir=r"180506-spherical_hollow_EAM_2nm" # Analysis directory
    category_dir="spherical" # category directory
    specific_dir="spherical_310.15_K_2_nmDIAM_hexadecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    # xtc_file=r"sam_prod_10_ns_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    xtc_file=r"sam_prod_10_ns_whole.xtc"
    '''
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                                          structure_file = gro_file,    # structure file
                                          xtc_file = xtc_file,          # trajectories
                                          )
    
    #%%
    ### DEFINING INPUT DATA
    input_details = {
                        'ligand_names': ['OCT', 'BUT', 'HEP', 'HED', 'DEC', 'DOD'],      # Name of the ligands of interest
                        'itp_file': 'sam.itp', # 'match', # ,                      # ITP FILE
                        'structure_types': [ 'monolayer_thickness', 'trans_ratio', 'hexatic_order', 'autocorrelation' ],         # 'monolayer_thickness', 'trans_ratio', 'distance_end_groups', Types of structurables that you want , 'distance_end_groups'
                        'save_disk_space': False,                    # Saving space
                        'separated_ligands': True
                        }
#    if category_dir == 'Planar':
#        input_details['itp_file'] = 'match'
#        input_details['separated_ligands'] = True
        
    ## RUNNING CLASS
    structure = sam_structure(traj_data, **input_details )
    # topology.atom(structure.head_group_atom_index[0]).residue.atom
    # structure.ligand_atom_index_list

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot( structure.trans_ratio_autocorr[:,0], structure.trans_ratio_autocorr[:,1] )
    plt.xlabel( 'sim. time (ps)' )
    plt.ylabel( r'$\langle x(0)x(t) \rangle$' )
    
    plt.figure()
    plt.plot( structure.hexatic_order_autocorr[:,0], structure.hexatic_order_autocorr[:,1] )
    plt.xlabel( 'sim. time (ps)' )
    plt.ylabel( r'$\langle \psi (0) \psi (t) \rangle$' )
    
    #%%
    
    ### IMPORTING GLOBAL VARIABLES
#    from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
#    from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
#    
#    ## DEFINING REFERENCE
#    reference_list = np.array(structure.dihedral_reference_list)
#    dihedrals = structure.dihedrals
#    
#    ## FINDING DIHEDRALS BASED ON LIGANDS
#    dihedrals_ligand_basis = dihedrals[:, reference_list] ## SHAPE: NUM_FRAMES X NUM_LIGANDS X NUM_DIHEDRALS
#    
#    ## FINDING AVERAGE DIHEDRALS PER FRAME BASIS
#    dihedral_ligand_trans_avg_per_frame = np.mean((dihedrals_ligand_basis < 240) & (dihedrals_ligand_basis > 120), axis=2) ## SHAPE: NUM_FRAMES X NUM_LIGANDS
#    ## NOTE: If you average this, you should get the ensemble average trans ratio
#    
#    ## PLOTTING THE DISTRIBUTION
#    bin_width = 0.01
#    ## CREATING FIGURE
#    fig, ax = create_plot()
#    ## DEFINING TITLE
#    ax.set_title('Trans ratio distribution between assigned and unassigned ligands')
#    ## DEFINING X AND Y AXIS
#    ax.set_xlabel('Trans ratio', **LABELS)
#    ax.set_ylabel('Normalized number of occurances', **LABELS)  
#    ## CREATING BINS
#    bins = np.arange(0, 1, bin_width)
#    ## DEFINING THE DATA
#    data = dihedral_ligand_trans_avg_per_frame.flatten()
#    ax.hist(data, bins = bins, color  = 'k' , density=True, label="All ligands" , alpha = 0.50)
    
    

        
    
    
    
    
