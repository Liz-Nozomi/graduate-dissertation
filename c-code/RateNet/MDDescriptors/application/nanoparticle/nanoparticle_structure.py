# -*- coding: utf-8 -*-
"""
nanoparticle_structure.py
This script analyzes the structure of the nanoparticle for each frame. Structural characteristic examples are shown below:
    tilt angle
    monolayer thickness
    trans/cis ratio
    minimum distance between tail chains

CLASSES:
    nanoparticle_structure: characterizes the nanoparticle structure and calculates various quantities, e.g.
        trans ratio, distances between end groups, etc.

CREATED ON: 04/09/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
**UPDATES**
20180420 - AKC - Updated draft of nanoparticle structure that can measure minimum distance between tail grous and trans ratio
20180702 - AKC - Updated "calc_dihedral_ensemble_avg_ratio" to correctly account for standard deviation in trans ratio
"""
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
import sys
import glob
import time
import os

### DEFINING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import GOLD_ATOM_NAME, LIGAND_HEAD_GROUP_ATOM_NAME

##############################################
### CLASS TO STUDY NANOPARTICLE STRUCTURE ####
##############################################
class nanoparticle_structure:
    '''
    The purpose of this function is to study the nanoparticle structure, in particular:
        tilt angle
        monolayer thickness
        trans/cis ratio
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        structure_types:   [list] 
            list of strings that include what type of job you want, e.g.
            'all': finds all possible structural types (e.g. trans_ratio, etc.)
            None: Turns off all secondary calculations of strutural types.
            ## LIST OF STRUCTURAL TYPES
                'trans_ratio': finds the trans ratio of the ligands based on heavy atoms
                'distance_end_groups': finds the end groups and calculates the minimum distance between each head group
        separated_ligands: [logical] 
            True if you want to avoid itp file. This assumes that your gro defines ligands separately. (e.g. more than one residue for ligands)
        save_disk_space: [logical] 
            True if you want to save disk space by removing the following variables: [Default: True]
            Turning off: terminal_group_distances, terminal_group_min_distance_matrix, dihedrals
        itp_file_path: [str, default=None]
            path to itp file
        
    OUTPUTS:
        ## JOB DETAILS
            self.ligand_names: name of the ligands in your file as a list
            self.structure_types: [list] list of strings that include what type of job you want
        
        ## TRAJECTORY INFORMATION
            self.traj_time_list:[list] trajectory times as a list in picoseconds
            self.traj_time_len: [int] total number of frames
        
        ## GOLD DETAILS
            self.gold_atom_index: [list] atom index of the gold atoms
            self.gold_geom: [np.array, shape=(time,num_gold,3)]: geometry of each gold atom per frame
            self.gold_center_coord: [np.array, shape=(time_steps, 3)]: center of gold core
        
        ## LIGAND DETAILS
            self.head_group_atom_index: [list] list of head group index
            self.ligand_atom_index_list: [np.array, shape=(num_lig, total length)] atom index of each ligand
            self.ligand_heavy_atom_index: [list] list of atom index that is not hydrogen
            self.total_ligands: [int] total number of ligands
            
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
        find_gold_info: finds the gold center, geometry, and coordinates
        find_all_head_groups: finds all head groups (i.e. sulfur atoms)
        find_each_ligand_group: finds all ligand groups (i.e. single ligand subsets)
        find_all_ligand_heavy_atoms: Using the ligand groups, find all heavy atoms (e.g. not hydrogens)
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
        cis/trans dihedral:
            - Find all the heavy atoms of the ligand (i.e. ligand backbone)
            - Find all dihedral angles for each heavy atom
            - Calculate an average cis/dihedral ratio
        monolayer thickness:
            - Find sulfur atom index
            - Find last carbon atom index
            - Calculate the monolayer thickness based on the 
            
    '''
    ### INITIALIZING
    def __init__(self, 
                 traj_data, 
                 ligand_names=['BUT'], 
                 itp_file=None, 
                 structure_types = 'all', 
                 separated_ligands=False, 
                 save_disk_space = True,
                 itp_file_path = None,):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### STORING INITIAL VARIABLES
        self.ligand_names = ligand_names                        # Ligand names
        if itp_file_path is None:
            if itp_file is not None:
                if os.path.isabs(itp_file) is True:
                    self.itp_file_path = itp_file
                else:
                    self.itp_file_path = os.path.join(traj_data.directory,
                                                      str(itp_file))
            else:
                self.itp_file_path = os.path.join(traj_data.directory,
                                                  str(itp_file))
        else:
            self.itp_file_path = itp_file_path
        self.save_disk_space = save_disk_space                  # Used to remove unnecessarily large variables
        self.separated_ligands = separated_ligands              # Used to avoid itp file generation
        
        ## CHECKING STRUCTURAL TYPES
        available_structure_types = ['trans_ratio', 'distance_end_groups']
        if structure_types == 'all':
            print("We are are running all possible structurals")
            print("Structural types: %s"%(', '.join(available_structure_types) ) )
            self.structure_types = available_structure_types[:]
        elif structure_types == None:
            print("Since structure types is set to None, we will simply calculate the nanoparticle structure (i.e. ligand head groups ,etc.)")
            print("This is a quick calculation!")
            self.structure_types = []
        else:
            ## FINDING STRUCTURE TYPES (SCREENING ALL INNCORRECT STRUCTURE TYPES)
            self.structure_types = [ each_structure for each_structure in structure_types if each_structure in available_structure_types]
            print("Final structure type list: %s"%(', '.join(self.structure_types) ))
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## STORING INFORMATION ABOUT THE TRAJECTORY
        self.traj_time_list = traj.time
        self.traj_time_len  = len(self.traj_time_list)
        
        ### FINDING ALL LIGAND TYPES
        self.ligand_names = [ each_ligand for each_ligand in self.ligand_names if each_ligand in traj_data.residues.keys() ]
        
        ## CHECKING LIGAND NAMES
        if len(self.ligand_names) == 0:
            print("ERROR! No ligand names defined. Perhaps check your inputs for ligand_names?")
            print("AVAILABLE RESIDUE NAMES: %s"%(', '.join(traj_data.residues.keys())))
            sys.exit()
        
        ## SEEING IF YOU HAVE AN ITP FILE
        if itp_file == 'match' or self.separated_ligands is False:
            ## IF MATCH, I will need to find all itp files within the trajectory location!
            if itp_file == 'match':
                print("Since itp file was set to 'match', we are going to try to find an ITP file that matches your ligand!")
                ## USING GLOB TO FIND ALL ITP FILES
                itp_files = glob.glob( traj_data.directory + '/*.itp' )
                
                print("Paths for itp file:")
                for idx_itp, each_itp in enumerate(itp_files):
                    print("%d: %s"%(idx_itp, each_itp))
                print("")
                
                ## PRINTING LIGAND NAME
                print("Searching for: ", self.ligand_names)
                
                ## GETTING ITP FILE
                self.itp_dict = {}
                
                ## LOOPING THROUGH EACH PATH
                for full_itp_path in itp_files:
                    print("Checking itp path: %s"%(full_itp_path) )
                    try:
                        itp_info = read_write_tools.extract_itp(full_itp_path)
                        
                        ## PRINTING RESIDUE NAME
                        print("Current residue name: %s"%(itp_info.residue_name))
                        
                        ## STORE ONLY IF THE RESIDUE NAME MATCHES THE LIGAND
                        if itp_info.residue_name in self.ligand_names:
                            print("Storing: %s"%(itp_info.residue_name))
                            self.itp_file = itp_info
                            self.itp_dict[itp_info.residue_name] = itp_info
                            # break ## Breaking out of for loop
                    except Exception: # <-- if error in reading the itp file (happens sometimes!)
                        pass
                ## SEEING IF ITP FILE EXISTS
                try:
                    print(self.itp_file)
                    print("Found itp file! ITP file path: %s"%(self.itp_file.itp_file_path))
                except NameError:
                    print("Error! No ITP file for ligand names found: %s"%(', '.join(self.ligand_names)) )
                    print("Perhaps, check your input files and make sure an itp file with the ligand name residue is there!")
                    print("Stopping here to prevent subsequent errors!")
                    sys.exit()
            else:
                if self.separated_ligands is False:
                    self.itp_file = read_write_tools.extract_itp(self.itp_file_path)
        
        ## SEEING IF YOU HAVE SEPARATED LIGANDS
        if self.separated_ligands is False:
            ### FINDING GOLD LATTICE INFORMATION
            self.find_gold_info(traj)
        
        ### FINDING ALL HEAD GROUPS
        self.find_all_head_groups(traj)
        
        ## FINDING ALL LIGAND GROUPS
        self.find_each_ligand_group(traj)
        
        ## FUNCTION TO GET ALL HEAVY ATOMS ON THE LIGAND GROUP (not hydrogens)
        self.find_all_ligand_heavy_atoms(traj)
    
        ###############################
        ### DIHEDRAL ANGLE ANALYSIS ###
        ###############################
        if 'trans_ratio' in self.structure_types:
            ## FUNCTION TO FIND ALL THE DIHEDRAL ANGLES
            self.dihedral_list, self.dihedral_reference_list = self.find_dihedral_list(ligand_atom_list = self.ligand_heavy_atom_index)
            ## FUNCTION TO MEASURE DIHEDRAL ANGLES
            self.dihedrals = self.calc_dihedral_angles(traj, dihedral_list = self.dihedral_list)            
            ## CALCULATING ENSEMBLE AVERAGE
            self.dihedral_gauche_avg, self.dihedral_gauche_std, self.dihedral_trans_avg, self.dihedral_trans_std = self.calc_dihedral_ensemble_avg_ratio(dihedrals = self.dihedrals)
            ## FUNCTION TO RUN TRANS RATIO IN A PER LIGAND BASIS
            self.dihedral_ligand_trans_avg_per_frame = self.calc_dihedral_trans_per_ligand( dihedrals =  self.dihedrals,  reference_list = self.dihedral_reference_list )          
        
        ###########################################
        ### MINIMUM DISTANCE BETWEEN END GROUPS ###
        ###########################################
        if 'distance_end_groups' in self.structure_types:
            ## FINDING ALL TERMINAL GROUPS
            self.find_all_terminal_groups()
            ## FINDING TERMINAL GROUP DISTANCES
            self.calc_terminal_group_distances(traj)
            ## FINDING MINIMUM DISTANCES
            self.calc_terminal_group_min_dist()
            
        ## PRINTING SUMMARY
        self.print_summary()
        
        ## TURNING OFF OBJECTS IF NECESSARY
        if save_disk_space is True:
            self.terminal_group_distances, self.terminal_group_min_distance_matrix, self.dihedrals = [], [], []
        
        return
    
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function prints the summary '''       
        if 'trans_ratio' in self.structure_types:
            print("\n----- GAUCHE/TRANS RATIO ANALYSIS -----")
            print("TOTAL NUMBER OF DIHEDRALS: %d"%( len(self.dihedral_list) ) )
            print("AVG TRANS: %.2f +- %.2f"%( self.dihedral_trans_avg, self.dihedral_trans_std ) )
            print("AVG GAUCHE: %.2f +- %.2f"%( self.dihedral_gauche_avg, self.dihedral_gauche_std) )
        if 'distance_end_groups' in self.structure_types:
            print("\n----- DISTANCE BETWEEN TERMINAL GROUPS ANALYSIS -----")
            print("TOTAL NUMBER OF TERMINAL GROUPS: %d"%(len(self.terminal_group_index)))
            print("MINIMUM DISTANCE: %.2f +- %.2f nm"%(self.terminal_group_min_dist_avg, self.terminal_group_min_dist_std))
    
    ### FUNCTION TO FIND GOLD INFORMATION
    def find_gold_info(self, traj, gold_atom_name = GOLD_ATOM_NAME):
        '''
        The purpose of this function is to find all gold information (i.e. center)
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            gold_atom_name: gold atom name (i.e. 'Au', default)
        OUTPUTS:
            self.gold_atom_index: [list] atom index of the gold atoms
            self.gold_geom: [np.array, shape=(time,num_gold,3)]: geometry of each gold atom per frame
            self.gold_center_coord: [np.array, shape=(time_steps, 3)]: center of gold core
        '''
        ## FINDING ALL GOLD INDEX
        self.gold_atom_index = [atom.index for atom in traj.topology.atoms if atom.name == gold_atom_name]
        ## FINDING GOLD GEOMETRY
        self.gold_geom = traj.xyz[:, self.gold_atom_index, :]
        ## FINDING GOLD CENTER
        self.gold_center_coord = np.mean(self.gold_geom, axis = 0)
        return
        
    ### FUNCTION TO FIND ALL HEAD GROUPS
    def find_all_head_groups(self, traj, 
                             head_group_name = LIGAND_HEAD_GROUP_ATOM_NAME):
        '''
        The purpose of this function is to find all head groups (i.e. sulfur atom) indexes.
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            head_group_name: [str] atom name of the head group (i.e. sulfur atom)
        OUTPUTS:
            self.head_group_atom_index: [list] list of head group index
        '''
        ## FINDING ALL HEADGROUP INDEXES
        self.head_group_atom_index = [atom.index for atom in traj.topology.atoms if atom.name == head_group_name and atom.residue.name in self.ligand_names ]
        ## FINDING GEOMETRY OF THE HEAD GROUPS
        self.head_group_geom = traj.xyz[:, self.head_group_atom_index, :] ## RETURNS ARRAY (NUM_FRAMES, HEAD_GROUP_INDEX)
        return

    ### FUNCTION TO FIND ALL INDIVIDUAL LIGANDS
    def find_each_ligand_group(self, traj):
        '''
        The purpose of this script is to find each ligand group using bonding information
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.ligand_atom_index_list: [np.array, shape=(num_lig, total length)] atom index of each ligand
        '''
        ## GENERATING BLANK LIST
        self.ligand_atom_index_list = []

        ## DEFINING FIRST INDEX IN LIGAND
        np_first_index = self.head_group_atom_index[0] # Everything is in reference to this

        ## LOOPING THROUGH EACH SULFUR HEAD GROUP
        for each_head_group_atom_index in self.head_group_atom_index:
            ## SEEING IF YOU HAVE SEPARATED LIGANDS
            if self.separated_ligands is False:
                ## FINDING ATOM INDEX
                atom_index = traj.topology.atom(each_head_group_atom_index).index
                ## FINDING ALL POSSIBLE BONDS
                logical_no_more_bonds = False
                ## CREATING BLANK LIST FOR THIS LIGAND
                each_ligand_atom_list = [atom_index]
                ## FINDING ATOM LISTS TO LOOK INTO
                new_atom_list = [ each_ligand_atom_list[-1] ] 
                
                #########################
                ### HEAVY-ATOM SEARCH ###
                #########################
                ## USING WHILE LOOP SINCE WE DO NOT KNOW HOW FAR REACHING THIS LIGAND IS
                while logical_no_more_bonds is False:
                    ## CREATE EMPTY LIST
                    attached_atom_list = []
                    ## LOOPING THROUGH NEW ATOM LIST
                    for current_atom_index in new_atom_list:
                        ## DEFINING SERIAL NUMBER # Updated to include references
                        current_serial_index = current_atom_index + 1 - np_first_index 
                        # time.sleep(1)
                        ## FINDING ALL ATOMS THAT ARE ATTACHED
                        bonds_attached = self.itp_file.bonds[np.where(self.itp_file.bonds==current_serial_index)[0],:]
                        ## FINDING ALL ATOM NUMBERS WITHIN ATTACHED ATOMS
                        atoms_attached = bonds_attached[bonds_attached != current_serial_index]
                        ## APPENDING TO LIST, AVOIDING ALL GOLD ATOMS AND HYDROGENS
                        atoms_to_append=[ each_atom - 1 + np_first_index for each_atom in atoms_attached \
                                                                             if traj.topology.atom(each_atom -1 + np_first_index).element.symbol != 'Au' and # No Gold atoms
                                                                                     traj.topology.atom(each_atom - 1 + np_first_index).element.symbol != 'H' and  # No hydrogens
                                                                                     each_atom - 1 + np_first_index not in each_ligand_atom_list and # All old atoms
                                                                                     each_atom - 1 + np_first_index not in attached_atom_list # All new atoms
                                                                                     ] # All new atoms
                        ## STORING THE ATOMS TO APPEND
                        attached_atom_list.extend(atoms_to_append)
                        # print(attached_atom_list)
                        # time.sleep(1)
                    ## AFTER FOR-LOOP, ATTACH THE ATOMS AND RESTART IF NECESSARY
                    ## SEEING IF THE LIGAND IS DONE
                    if len(attached_atom_list) == 0:
                        ## TURNING OFF WHILE LOOP
                        logical_no_more_bonds = True                            
                    else: ## LIGAND IS NOT DONE, CONTINUE FINDING HEAVY ATOMS!
                        each_ligand_atom_list.extend(attached_atom_list)                            
                        new_atom_list = attached_atom_list[:] ## NEW ATOM LIST TO CHECK OUT
                
                ## NOW FINDING ALL ATOMS BONDED
                non_heavy_atom_list = []
                # print(each_ligand_atom_list)
                
                
                ####################################################
                ### POST-WHILE LOOP: FINDING ALL NON-HEAVY ATOMS ###            
                ####################################################
                for each_atom in each_ligand_atom_list:
                    ## DEFINING SERIAL NUMBER
                    current_serial_index = each_atom + 1 - np_first_index 
                    ## FINDING ALL ATOMS THAT ARE ATTACHED
                    bonds_attached = self.itp_file.bonds[np.where(self.itp_file.bonds==current_serial_index)[0],:]
                    ## FINDING ALL ATOM NUMBERS WITHIN ATTACHED ATOMS
                    atoms_attached = bonds_attached[bonds_attached != current_serial_index]
                    ## APPENDING TO LIST, AVOIDING ALL GOLD ATOMS
                    atoms_to_append=[ each_atom - 1 + np_first_index for each_atom in atoms_attached if traj.topology.atom(each_atom -1 + np_first_index).element.symbol != 'Au' and # No Gold atoms
                                                                                 each_atom - 1 + np_first_index not in each_ligand_atom_list and
                                                                                 each_atom - 1 + np_first_index not in non_heavy_atom_list] # All new atoms
                    ## STORING
                    non_heavy_atom_list.extend(atoms_to_append)
                    # print(non_heavy_atom_list)
                    # time.sleep(1)
                ## APPENDING TO LIST
                each_ligand_atom_list.extend(non_heavy_atom_list)
                ## SORTING AND STORAGE
                each_ligand_atom_list.sort()
                
                            
            ## LIGANDS ARE SEPARATED
            else:
                ## GETTING ALL ATOMS
                each_ligand_atom_list = [ atom.index for atom in traj.topology.atom(each_head_group_atom_index).residue.atoms ]
            ## APPENDING TO THE LIST
            self.ligand_atom_index_list.append(each_ligand_atom_list)
        return
    
    ### FUNCTION TO GET THE HEAVY ATOMS
    def find_all_ligand_heavy_atoms(self, traj):
        '''
        The purpose of this script is to find all heavy atoms of a ligand
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.ligand_heavy_atom_index: [list] list of atom index that is not hydrogen
            self.total_ligands: [int] total number of ligands
        '''
        ## CREATE EMPTY LISTS
        self.ligand_heavy_atom_index = []
        ## LOOPING THROUGH EACH LINE OF THE LIGAND AND GET ALL NON-HEAVY ATOMS
        for each_ligand_atom_index in self.ligand_atom_index_list:
            self.ligand_heavy_atom_index.append([each_atom for each_atom in each_ligand_atom_index if traj.topology.atom(each_atom).element.symbol != 'H'])
        
        ## FINDING TOTAL NUMBER OF LIGANDS
        self.total_ligands = len(self.ligand_atom_index_list)
        
        return
    
    ### FUNCTION TO FIND DIHEDRAL LIST
    @staticmethod
    def find_dihedral_list(ligand_atom_list):
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
    def calc_dihedral_angles(traj, dihedral_list, periodic = True):
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
    def calc_dihedral_ensemble_avg_ratio(dihedrals):
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
        dihedral_gauche_std = np.std(dihedral_gauche_over_time)
        
        return dihedral_gauche_avg, dihedral_gauche_std, dihedral_trans_avg, dihedral_trans_std
    
    ### FUNCTION TO FIND TRANS RATIO OF EACH LIGAND PER FRAME BASIS
    @staticmethod
    def calc_dihedral_trans_per_ligand(dihedrals, reference_list ):
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
    def find_all_terminal_groups(self):
        '''
        The purpose of this function is to find the end group, which we assume is the last heavy atom
        INPUTS:
            self: class object
        OUTPUTS:
            self.terminal_group_index: [list] last index of each ligand in the heavy atom index
        '''
        self.terminal_group_index=[each_ligand[-1] for each_ligand in self.ligand_heavy_atom_index ]
        return
        
    ### FUNCTION TO CALCULATE ALL DISTANCES BETWEEN THE ATOMS
    def calc_terminal_group_distances(self,traj, periodic=True):
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
    def calc_terminal_group_min_dist(self, ):
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
    
        
    
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180702-Trial_1_spherical_EAM_correction" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_4_nmDIAM_hexadecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    # xtc_file=r"sam_prod_10_ns_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    xtc_file=r"sam_prod_10_ns_whole.xtc"
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
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    ### DEFINING INPUT DATA
    input_details = {
                        'ligand_names': ['OCT', 'BUT', 'HED', 'DEC', 'DOD'],      # Name of the ligands of interest
                        'itp_file': 'sam.itp', # 'match', # ,                      # ITP FILE
                        'structure_types': ['trans_ratio'],         # Types of structurables that you want , 'distance_end_groups'
                        'save_disk_space': False,                    # Saving space
                        'separated_ligands':False
                        }
    if category_dir == 'Planar':
        input_details['itp_file'] = 'match'
        input_details['separated_ligands'] = True
        
    ## RUNNING CLASS
    structure = nanoparticle_structure(traj_data, **input_details )
    # topology.atom(structure.head_group_atom_index[0]).residue.atom
    # structure.ligand_atom_index_list
    
    
    #%%
    
    ### IMPORTING GLOBAL VARIABLES
    from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
    from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
    
    ## DEFINING REFERENCE
    reference_list = np.array(structure.dihedral_reference_list)
    dihedrals = structure.dihedrals
    
    ## FINDING DIHEDRALS BASED ON LIGANDS
    dihedrals_ligand_basis = dihedrals[:, reference_list] ## SHAPE: NUM_FRAMES X NUM_LIGANDS X NUM_DIHEDRALS
    
    ## FINDING AVERAGE DIHEDRALS PER FRAME BASIS
    dihedral_ligand_trans_avg_per_frame = np.mean((dihedrals_ligand_basis < 240) & (dihedrals_ligand_basis > 120), axis=2) ## SHAPE: NUM_FRAMES X NUM_LIGANDS
    ## NOTE: If you average this, you should get the ensemble average trans ratio
    
    ## PLOTTING THE DISTRIBUTION
    bin_width = 0.01
    ## CREATING FIGURE
    fig, ax = create_plot()
    ## DEFINING TITLE
    ax.set_title('Trans ratio distribution between assigned and unassigned ligands')
    ## DEFINING X AND Y AXIS
    ax.set_xlabel('Trans ratio', **LABELS)
    ax.set_ylabel('Normalized number of occurances', **LABELS)  
    ## CREATING BINS
    bins = np.arange(0, 1, bin_width)
    ## DEFINING THE DATA
    data = dihedral_ligand_trans_avg_per_frame.flatten()
    ax.hist(data, bins = bins, color  = 'k' , density=True, label="All ligands" , alpha = 0.50)
    
    

        
    
    
    
    
