# -*- coding: utf-8 -*-
"""
stored_parallel_scripts.py
This contains scripts that are used in parallel.

FUNCTIONS:
    compute_cosolvent_mapping: 
        computes cosolvent mapping
    count_atoms_in_sphere:
        counts number of atoms in the sphere
    calc_histogram_integers:
        computes number of histogram integer
    
Written by: Alex K. Chew (10/31/2019)
"""

import mdtraj as md
import numpy as np

## TRAJECTORY DETAILS
import MDDescriptors.core.calc_tools as calc_tools # Loading trajectory details

### IMPORTING LIGAND REISDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import ligand_residue_list as ligand_names

### FUNCTION TO COMPUTE HISTOGRAM
def calc_histogram_integers( N, d = -1 ):
    r"""
    This function computes the un-normalized histogram distribution. This defines 
    d, which is some large range of value.
    INPUTS:
        N: [np.array]
            numpy array containing your distribution -- number of occurances per frame
        d: [int]
            number of bins used for the histogram
    OUTPUTS:
        histogram: [np.array]
            histogram data for the un-normalized distribution
    """
    if d < 0.:
        d = np.max( N )
    return np.histogram( N, bins = int( d ), range = [ 0., d ] )[0]

## FUNCTION TO COMPUTE NUMBER OF ATOMS WITHIN SPHERE
def count_atoms_in_sphere( traj, pairs, r_cutoff = 0.33, periodic = True ):
    r"""
    The purpose of this function is to comptue the number of atoms within some 
    radius of cutoff.
    INPUTS:
        traj: [obj]
            trajectory object
        pairs: [np.array, shape=(N,2)]
            pairs array that you're interested in
        r_cutoff: [float]
            cutoff that you are interested in
    OUTPUTS:
        n_occurance_per_frame: [np.array]
            array that is the same size as pairs across trajectories
    """
    ## COMPUTING DISTANCES BETWEEN ATOMS
    distances = md.compute_distances( traj = traj, 
                                      atom_pairs = pairs, 
                                      periodic = True )
    ## RETURNS NUM_FRAMES X NUM_PAIRS
    
    ## GETTING DISTANCES THAT ARE LESS THAN A CUTOFF
    n_occurance_per_frame = np.sum(distances < r_cutoff,axis=1) 
    ## RETURNS NUM_FRAMES X 1 <-- number of occurances
    
    return n_occurance_per_frame


### MAIN CLASS FUNCTION
class compute_cosolvent_mapping:
    '''
    The purpose of this function is to generate cosolvent mapping by creating 
    a histogram of occurances across a simulation trajectory. The idea is to simply 
    output the number of occurances across trajectories for a set of grid points
    INPUTS:
        traj_data: [obj]
            trajectory data
        ligand_names: [list]
            ligand names
        grid_xyz: [np.array]
            grid xyz to compute cosolvent mapping on
        cutoff: [float]
            cutoff to check around grid point
        max_N: [int]
            maximum number of atoms that could be within a radius
        verbose: [True/False]
            True if you want to print verbosely information
    OUTPUTS:
        self.atom_pairs_dict: [dict]
            dictionary of atom pairs
    '''
    def __init__(self, 
                 traj_data, 
                 grid_xyz, 
                 ligand_names = ligand_names, 
                 cutoff = 0.33, 
                 max_N = 10, 
                 verbose = True):
        
        ## STORING DETAILS
        self.cutoff = cutoff,
        self.max_N = max_N
        self.total_grid = len(grid_xyz)
        
        ## STORING XYZ
        self.grid_xyz = grid_xyz
        
        ## GETTING SOLVENT LIST
        self.solvent_list = [each_residue for each_residue in traj_data.residues if each_residue not in ligand_names]
        self.total_solvents = len(self.solvent_list)
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## GETTING ATOMS FOR EACH RESIDUE
        self.get_atoms_each_residue(traj = traj)

        ## DETERMINING ATOM INDEX TO CHANGE
        self.atom_index_to_change = self.atom_not_in_index[0]
        
        ## GENERATING ATOM PAIR LIST
        self.generate_atom_pairs()
        
        return
    
    ### FUNCTION TO GENERATE ATOM PAIRS
    def generate_atom_pairs(self):
        '''
        The purpose of this function is to generate atom pairs
        INPUTS:
            self: [obj]
                self property
        OUTPUTS:
            atom_pairs_dict: [dict]
                dictionary of atom pairs
        '''
        ## DICTIONARY TO STORE THE PAIRS
        self.atom_pairs_dict = {}
        ## LOOPING THROUGH EACH RESIDUE INDEX
        for each_residue in self.atom_index_dict:
            ## GETTING INDEX
            atom_index = self.atom_index_dict[each_residue]
            ## CREATING ATOM PAIRS LIST
            pair_list = calc_tools.create_atom_pairs_list(atom_1_index_list = [self.atom_index_to_change],
                                                          atom_2_index_list = atom_index)
            ## STORING
            self.atom_pairs_dict[each_residue] = pair_list[:]
        return
    
    ### FUNCTION TO GET ATOM INDEX DICTIONARY
    def get_atoms_each_residue(self, traj):
        '''
        This function gets the atom for each residue.
        INPUTS:
            self: [obj]
                self property
            traj: [md.traj]
                trajectory object
        OUTPUTS:
            self.atom_index_dict: [dict]
                dictionary of heavy atom indices of each of the solvent atoms
            self.atom_not_in_index: [list]
                list of atoms that are not part of the index
        '''
        ## GETTTING ATOM INDEX FOR EACH SOLVENT
        self.atom_index_dict = { each_solvent: [] for each_solvent in self.solvent_list }
        self.atom_not_in_index = []
        ## LOOPING THROUGH EACH ATOM
        for each_atom in traj.topology.atoms:
            ## GETTING RESIDUE NAME
            current_residue_name = each_atom.residue.name
            ## SEEING IF ATOM AND IGNORING ALL HYDROGENS
            if current_residue_name in self.solvent_list and each_atom.element.symbol != 'H':
                ## STORING
                self.atom_index_dict[current_residue_name].append(each_atom.index)
            else:
                self.atom_not_in_index.append(each_atom.index)
        return
        
    
    ### FUNCTION TO COMPUTE NUMBER OF PAIRS WITHIN FOR SOME FRAMES
    def compute(self, traj, verbose = True):
        '''
        The purpose of this function is to compute the number of nearest solvents 
        within the vicinity of the grid point. 
        INPUTS:
            traj: [mdtraj.trajectory] 
                trajectory imported from mdtraj
        OUTPUTS:
            unnorm_p_N: [np.array, shape = (num_solvents, num_grid, max N)]
                unnormalizes number distribution that contains the number of solvents within a cutoff per grid basis
        '''
        ## GETTING TRAJECTORY SIZE
        n_frames = traj.time.size
        print("--- Calculating cosolvent mapping for %s simulations windows ---" % (str(n_frames)) )
        ## SELECTING ATOM INDEX TO CHANGE
        atom_index_to_change = self.atom_index_to_change
        ## DEFINING GRID
        grid_xyz = self.grid_xyz
        ## GENERATING DISTRIBUTION CONTAINING THE NUMBER OF GRID POINTS AND MAX NUMBER OF ITEMS
        unnorm_p_N = np.zeros( shape = ( len(self.solvent_list), len(grid_xyz), self.max_N ) )
        ## TOTAL GRID
        total_grid = len(grid_xyz)
        
        ## LOOPING THROUGH THE GRID
        for idx, grid_values in enumerate(grid_xyz):
            ## PRINTING
            if verbose is True and idx % 100 == 0:
                print("Working on Grid index: %d of %d"%(idx, total_grid))
            
            ## COPYING TRAJ
            copied_traj = traj[:]
            ## CHANGING THE POSITION OF ONE ATOM IN THE SYSTEM TO BE THE GRID ATOM
            copied_traj.xyz[:,atom_index_to_change,:] = np.tile( grid_xyz[idx,:], (n_frames, 1) )
            
            ## LOOPING THROUGH EACH RESIDUE
            for res_index, each_residue in enumerate(self.solvent_list):
                ## DEFINING ATOM PAIRS
                pairs = self.atom_pairs_dict[each_residue]
                ## GETTING NUMBER OF OCCURANCES
                n_occurance_per_frame = count_atoms_in_sphere( traj = copied_traj, 
                                                               pairs = pairs, 
                                                               r_cutoff = self.cutoff, 
                                                               periodic = True )
                ## COMPUTING DISTRIBUTION AND STORING IT INTO THE INDEX
                unnorm_p_N[res_index, idx,:] = calc_histogram_integers( N = n_occurance_per_frame, 
                                                                        d = self.max_N )
                
        return unnorm_p_N

