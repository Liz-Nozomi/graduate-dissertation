# -*- coding: utf-8 -*-
"""
nanoparticle_find_bundled_groups.py
The purpose of this script is to find patches of the nanoparticle that is bundled together. The main idea here is that we want to group nanoparticle parts together. 
After grouping them, we can analyze these structures, e.g. finding total number of ligands that are bundled together, size of the bundle, and directionality. 

CREATED ON: 05/16/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

INPUTS:
    - gro file
    - itp file (for bonding purposes)
    - cutoff between bundling groups
    - number of neighboring sulfur groups to consider when checking for groups
    
OUTPUTS:
    - list of list: basically, for each frame, we want to get a group index for each ligand
    
ALGORITHM:
    - NANOPARTICLE STRUCTURE
        - Find all ligands in the nanoparticle
        - Create ligand index list
    - SULFUR STRUCTURE
        - Find all distances between sulfur atoms
        - Create sulfur index of the nearest sulfur atoms for each frame, based on number of nearest neighbors criteria
    - LIGAND STRUCTURE
        - Each ligand will be defined as a single vector
            - e.g. vector between sulfur and end group
    - GROUPING LIGANDS TOGETHER
        - For each frame:
            - Find dot product of nearest neighbors
            - Find likelihood of grouping
            - Assign grouping index to the residue
            
GROMACS:
    ## CREATING TRAJECTORY WITH NO WATER
    gmx trjconv -f sam_prod_10_ns_whole.xtc -s sam_prod.tpr -pbc whole -center -n index.ndx -o sam_prod_10_ns_whole_no_water_center.xtc
    ## GRO FILE
    gmx trjconv -f sam_prod_10_ns_whole.xtc -s sam_prod.tpr -pbc whole -center -n index.ndx -o sam_prod_10_ns_whole_no_water_center.gro -b 50000

FUNCTIONS:
    ## NEIGHBOR FUNCTIONS
        find_nearest_atom_index: 
            finds nearest atom indexes for all frames
        correct_atom_numbers: 
            corrects atom numbers using a conversion legend
        check_group_list: ** DEPRECIATED ** Checks group list
        find_all_neighbors: ** DEPRECIATED ** function that finds all neighbors and completes neighborhood list
        find_angle_btwn_displacement_vectors: 
            function to find angle between displacement vectors
        find_group_assignments: 
            function that an find group assignments
        check_nearest_neighbors_list: 
            function that can find nearest neighbors
        
    ## LIGAND BUNDLING GROUPS
        calc_similarity_matrix_bundling_grps: 
            generates similarity matrix between bundling groups
        find_similarity_new_group_dict: 
            finds similarity between two groups
        find_similar_bundling_groups: 
            finds similar bundling groups between two frames
        update_lig_assignment_list: 
            updates ligand assignment list
        compute_lig_avg_displacement_array:
            computes ligand average displacement array
        
    ## TRANS RATIO FUNCTIONS
        find_assigned_vs_unassigned: finds assigned vs unassigned ligands
    
CLASSES:
    nanoparticle_sulfur_structure: ** DEPRECIATED ** calculates sulfur structure for nanoparticle
    calc_nanoparticle_bundling_trans_ratio: calculates trans ratio with bundling groups
    calc_nanoparticle_bundling_groups: calculates nanoparticle bundling groups

** UPDATES **
20180518 - AKC - Updating correct atom numbers list
20180525 - AKC - Correctly implemnts hdbscan and outputs number of bundles
20180621 - AKC - Inclusion of trans ratio calculation in bundling groups

"""
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
import sys
import time # Measures time
import copy # Copy module
### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure
### RANDOMIZING LISTS
from random import shuffle
### IMPORTING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot


### FUNCTION TO GET AVERAGE DISPLACEMENTS
def compute_lig_avg_displacement_array(traj,
                                       sulfur_atom_index,
                                       ligand_heavy_atom_index,
                                       periodic = True):
    '''
    The purpose of this function is to get the ligand average displacment 
    vector given the head group and ligand heavy atom index
    INPUTS:
        traj: [obj]
            trajectory object
        sulfur_atom_index: [list]
            list of sulfur atom index
        ligand_heavy_atom_index: [list]
            list of heavy atom indexes
        periodic: [logical] 
            True if you want periodic boundaries
    OUTPUTS:
        lig_displacements_array: [np.array, shape = (num_frames, num_sulfurs, 3)]
            ligand displacement array in x, y, z coordinates
    '''
    ## CREATING EMPTY ARRAY
    lig_displacements_array = np.empty( (len(traj), len(sulfur_atom_index), 3))
    
    ## DEFINING INDEX
    ## LOOPING
    for index in range(len(sulfur_atom_index)):
        ## DEFINING ATOM INDICES
        atom_1_index = sulfur_atom_index[index]
        atom_2_index = ligand_heavy_atom_index[index]
        
        ## GENERATING ATOM PAIRS
        atom_pairs = calc_tools.create_atom_pairs_list(atom_1_index, atom_2_index) 
        
        ## GETTING DISPLACEMENTS
        displacements = md.compute_displacements(traj = traj,
                                                 atom_pairs = atom_pairs,
                                                 periodic = True)
        
        ## GETTING AVERAGE DISPLACMENT WITH RESPECT TO ALL VECTORS
        avg_displacement = np.mean(displacements,axis=1) # SHAPE: NUM_FRAMES, 3    
        
        ## STORING
        lig_displacements_array[:, index, :] = avg_displacement[:]
        
    return lig_displacements_array

### FUNCTION TO FIND NEAREST ATOMS GIVEN ATOM PAIRS AND DISTANCES
def find_nearest_atom_index( atom_pairs, distances, atom_index, nearest_neighbors = 6 ):
    '''
    The purpose of this function is to take the atom pairs, distances, and a index. It will find all nearest atoms for all frames
    INPUTS:
        atom_pairs: [np.array, shape=(num_pairs, 2)] atom pairs with atom indexes
        distances: [np.array, shape=(frames, num_pairs)] distances of each of the atom pairs for multiple frames
        atom_index: [int] index that you are interested in finding the nearby neighbors
            NOTE: atom_index should match one of the atoms in atom_pairs! If not, then we cannot find nearest neighbors!
        nearest_neighbors: [int, default = 6] how many neighbors do you want?
    OUTPUTS:
        nearest_atom_index_array: [np.array, shape=(frames, nearest_neighbors)] Gives atom index that are matching for all frames
    '''
    ## FINDING ALL ROWS IN ATOM PAIRS THAT HAVE YOUR ATOM INDEX
    rows_in_atom_pairs_matched = np.where(atom_pairs == atom_index)[0] ## RETURNS ALL ROWS THAT MATCHES THE ATOM INDEX
    ## FINDING ATOM PAIRS FOR THE PARTICULAR ATOM INDEX
    atom_pairs_matched = atom_pairs[rows_in_atom_pairs_matched] ## RETURNS ONLY THE ATOM PAIRS THAT HAS YOUR ATOM INDEX
    ## FINDING DISTANCES THAT ARE MATCHED
    distances_matched = distances[:, rows_in_atom_pairs_matched] ## RETURNS ALL DISTANCES THAT ARE MATCHING ATOM PAIRS
    ## SORTING DISTANCES
    sorted_distances_matched = np.argsort( distances_matched, axis = 1 ) ## SORTS DISTANCES FROM SMALLEST TO LARGEST WITH RESPECT TO AXIS 1, MINIMUM IS THE LOWEST INDEX
    ## FINDING CLOSEST NEIGHBORS
    closest_neighbors_serial = sorted_distances_matched[:, range(nearest_neighbors) ] ## RETURNS ONLY SOME OF THE NEAREST NEIGHBORS
    ### NOW THAT WE HAVE THE CLOSEST ATOM INDEX, LET'S REWRITE THE DATA TO A MORE DESIRABLE TYPE
    ## FINDING RESIDUES WITHIN ATOM PAIRS THAT IS NOT YOUR ATOM INDEX
    atom_index_without_matched = atom_pairs_matched[np.not_equal(atom_pairs_matched, atom_index)] ## RETURNS ATOM INDEXES THAT ARE PAIRED BUT NOT YOUR ATOM INDEX
    ## FINALLY, GET ATOM INDEX OF THE CLOSEST
    nearest_atom_index_array = np.reshape(atom_index_without_matched[closest_neighbors_serial.flatten()], newshape=closest_neighbors_serial.shape )
    return nearest_atom_index_array

### FUNCTION TO CORRECT ATOM NUMBERS
def correct_atom_numbers( list_to_convert, conversion_legend ):
    '''
    The purpose of this function is to take a list of list and convert all the numbers according to a conversion list. The conversion list is also a list of list (i.e. [[1,2], [2,3]])
    The way this script works is that it flattens out the list of list, then converts the numbers, then re-capitulates the list of list.
    INPUTS:
        list_to_convert: list of list that you want to fix in terms of numbers (i.e. atom numbers)
            NOTE: zeroth index is the current value and 1st index is the transformed value
        conversion_legend: list of list that has indexes where the first index is the original and the next index is the new value
    OUTPUTS:
        converted_list: [np.array] array with the corrected values
    '''
    # Start by converting the list to a numpy array
    converted_list = np.array(list_to_convert).astype('int')
    ## CONVERTING CONVERSION LEGEND TO NUMPY ARRAY
    conversion_legend_array = np.array(conversion_legend).astype('int')
    # Copying list so we do not lose track of it
    orig_list = converted_list[:]
    # Looping through each conversion list value and replacing
    for legend_values in conversion_legend_array:
        ## FINDING LOCATION OF WHERE IS TRUE
        indexes = np.where( orig_list == legend_values[0] )
        ## CHANGE IF FOUND INDEX
        if len(indexes) > 0:
            converted_list[indexes] = legend_values[1]
    return converted_list

### FUNCTION TO CHECK GROUP LIST AND RENUMBER IF NECESSARY
def check_group_list( assignments, group_list , verbose = False):
    '''
    *** DEPRECIATED ***
    The purpose of this function is to see the assignments and group list, then re-number if necessary. The main goal is to minimize the number of groups.
    INPUTS:
        assignments:[np.array,  shape=(N,1)] assignments with group indexes, e.g. [ 0, 1, 2, 1, 1, 1, ...]
        group_list: [list] list of group values, assumed to start at [0, 1, 2, ...]
        verbose: [logical] True if you want to see output printed
    OUTPUTS:
        new_assignments:[np.array,  shape=(N,1)] updated assignments
        new_group: [list] updated group list
    '''
    ## FINDING UNIQUE ASSIGNMENTS
    unique_assignments = np.unique(assignments, return_counts=True)[0]
    # if len(unique_assignments) != len(group_list):
    if len(unique_assignments) != len(group_list) or np.max(unique_assignments) != np.max(group_list):
        if verbose == True:
            print("Unique assignments (%d) is not the same as total groups (%d)"%( len(unique_assignments), len(group_list) ) )
            print("Renumbering assignments to match the groups!")
        new_group = np.arange(len(unique_assignments)) #  [ value for value in range(len(unique_assignments))]
        ## CREATING LEGEND
        legend =np.append( unique_assignments[:, np.newaxis], new_group[:, np.newaxis], axis=1)  #  [ [unique_assignments[idx], idx ] for idx in new_group]
        ## CORRECTING THE ASSIGNMENTS
        new_assignments = correct_atom_numbers(assignments, legend)
    else:
        new_assignments = assignments
        new_group = group_list
    return new_assignments, new_group


### FUNCTION TO FIND ALL NEIGHBORS A NEIGHBORS LIST -- WILL COMPLETE THE NEIGHBORHOOD
def find_all_neighbors( neighbors ):
    '''
    *** DEPRECIATED ***
    The purpose of this function is to look through the neighbors list and complete it. If you have a neighbor in your list, but that neighbor does not have you -- this will correct that.
    INPUTS:
        neighbors: [np.array, shape=(N, num_neighbors)] Each index indicate a neighbors list
    OUTPUTS:
        new_neighbors_list: [list] list of new neighbors with the same length of neighbors
    '''
    ## CREATING EMPTY ARRAY
    new_neighbors_list = []
    ## LOOPING THROUGH ALL THE NEIGHBORS
    for idx, each_neighbors in enumerate(neighbors):
        ## FIND ALL NEIGHBORS WITH CURRENT INDEX
        neighbors_with_index = np.where(neighbors == idx)[0]
        ## FINDING ALL NEW NEIGHBORS
        new_neighbors = np.append( each_neighbors, neighbors_with_index[~np.isin(neighbors_with_index, each_neighbors)] ).tolist()
        ## APPENDING TO NEIGHBORS LIST
        new_neighbors_list.append(new_neighbors)
    return new_neighbors_list

### FUNCTION TO FIND ALL ANGLES BETWEEN DISPLACEMENT VECTORS
def find_angle_btwn_displacement_vectors( displacements ):
    '''
    The purpose of this function is to find the angle between a list of displacement vectors. It will return a matrix with the angles between each displacement vector.
    ## -- UPDATE: This function clips the values to fall within -1.0 and 1.0
    Note that this function fills the diagonals to be zero in angles, with the assumption that the displacement angle with itself is indeed 0 degrees
    INPUTS:
        displacements: [np.array, shape=(N, 3)] 
            N displacement vectors
    OUTPUTS:
        angles_btwn: [np.array, shape=(N,N)] 
            Angles in degrees between all displacement vectors
    '''
    ## CHANGING FLOAT TYPE
    displacements = displacements.astype('float64')
    ## FINDING NORMS OF DISPLACEMENT
    displacement_norms = np.linalg.norm(displacements, axis = 1)
    ## DEFINING DOT PRODUCTS
    dot_products_matrix = np.dot(displacements, displacements.T) ## SHAPE: num_ligands x num_ligands    
    ## DEFINING NORM MATRIX
    displacement_norms_matrix = np.matmul( displacement_norms[:, np.newaxis], displacement_norms[np.newaxis, :] ).astype('float64')
    ## FINDING THE DOT PRODUCT DIVIDED BY DISPLACEMENTS
    dot_product_over_displacement_matrix = dot_products_matrix / displacement_norms_matrix
    ## FILLING DIAGONALS TO BE 1 <-- REQUIRED TO AVOID NUMERICAL ERRORS WHERE THE DOT PRODUCTS AND DIVISION OF NORMS SLIGHTLY > 1
    # np.fill_diagonal(dot_product_over_displacement_matrix, val=1)
    ## FINDING ARC COSINE OF THE ANGLE
    angles_btwn = np.rad2deg( np.arccos( np.clip(dot_product_over_displacement_matrix, -1.0, 1.0) ) )
    return angles_btwn

### FUNCTION TO FIND GROUP ASSIGNMENTS
def find_group_assignments( assignments ):
    '''
    The purpose of this function is to find the group assignments with a given array of assignments.
    For example, suppose you are given a list of numbers [ 0 , 1 ,2, 0 ], and you want to create a new list:
        0: [0, 3]
        1: [1]
        2: [2]
    In this case, we are generating list of lists to get all the indexes that are matching
    INPUTS:
        assignments: [np.array, shape=(N,1)] assignments for each index
    OUTPUTS:
        group_list: [dict] dictionary of the group list
    '''
    ## FINDING UNIQUE ASSIGNMENTS
    unique_assignments = np.unique( assignments )
    ## CREATING EMPTY DICTIONARY LIST
    group_list = {}    
    ## LOOPING THROUGH EACH ASSIGNMENT AND FINDING ALL INDICES WHERE TRUE
    for each_assignment in unique_assignments:
        ## FINDING ALL INDEXES
        indices_with_assignments = np.where(assignments == each_assignment)[0] # ESCAPING TUPLE
        ## CREATING A DICTIONARY AND STORING THE VALUES
        group_list[str(each_assignment)] = indices_with_assignments
    return group_list

### FUNCTION TO CORRECT FOR CLUSTERS THAT ARE NOT WITHIN THE SAME NEIGHBORS LIST
def check_nearest_neighbors_list( assignments, group_list, neighbor_list, verbose = False):
    '''
    The purpose of this function is to check the assignments and ensure that the groups are correctly bundled together. This will go through the
    neighbors list
    INPUTS:
        assignments: [np.array, shape=(num_frames, num_ligands)] group assignments
        group_list: [np.array, shape=(num_frames, num_groups)] list of ligands assigned
        neighbor_list: [np.array, shape=(num_ligands, num_frames, neighbors)] neighbors list for the atoms
        verbose: [logical, default = False] True if you want to see everything
    OUTPUTS:
        assignments: [np.array, shape=(num_frames, num_ligands)] updated group assignments
        group_list: [np.array, shape=(num_frames, num_groups)] updated list of ligands assigned
    '''
    ## COPYING OVER ASSIGNMENT AND GROUP LISTS
    assignments = assignments[:]
    group_list = group_list[:]
    ## STORING TOTAL CORRECTIONS
    total_corrections = 0
    ## LOOPING THROUGH EACH FRAME
    for each_frame in range(len(assignments)):
        ## COUNTS NUMBER OF TIMES YOU FOUND AN INCORRECT ASSIGNMENT
        num_incorrect_assignment = 0
        ## LOOPING THROUGH EACH GROUP
        for each_group in group_list[each_frame]:
            if each_group != '-1':
                ## DEFINING MEMBERS
                group_members = group_list[each_frame][each_group]
                ## DEFINING FULL NEIGHBORS LIST
                group_neighbors_list = neighbor_list[group_members, each_frame, :]
                ## LOOPING THROUGH EACH MEMBER TO SEE IF IT IS WITHIN NEIGHBORS LIST
                for each_member in group_members:
                    ## SEEING TOTAL TIMES FOUND IN GROUP
                    each_member_total_times_in_neighbors_list = (group_neighbors_list==each_member).sum()
                    ## IF THE MEMBER IS == 0, THEN IT IS NOT REALLY A MEMBER!
                    if each_member_total_times_in_neighbors_list == 0:
                        ## UPDATING ASSIGNMENTS
                        assignments[each_frame][each_member] = -1
                        ## KEEPING TRACK OF INCORRECT ASSIGNMENTS 
                        num_incorrect_assignment += 1
                        print("CORRECTING ASSIGNMENT IN FRAME %d FOR MEMBER %d IN GROUP %s"%(each_frame, each_member, each_group) )
        ## SEEING IF NUMBER OF INCORRECT ASSIGNMENT IS TRUE
        if num_incorrect_assignment > 0:
            print("TOTAL INCORRECT ASSIGNMENTS: %d"%(num_incorrect_assignment))
            ## UPDATE ENTIRE GROUP LIST
            group_list[each_frame] = find_group_assignments(assignments[each_frame])
            ## ADDING TO TOTAL CORRECTIONS
            total_corrections += num_incorrect_assignment
        
    print("TOTAL CORRECTIONS OUT OF %d FRAMES: %d"%(len(assignments), total_corrections) )
    return assignments, group_list

### FUNCTION TO CALCULATE PAIR DISTANCES USING MDTRAJ
def calc_pair_distances(traj, atom_index, periodic=True):
    '''
    The purpose of this function is to caluclate pair distances between any atom pairs
    INPUTS:
        traj: [md.traj] trajectory from md.traj
        atom_index: [list] list of atom indexes you want pair distances for
        periodic: [logical, default=True] True if you want PBCs
    OUTPUTS:
        distances: [np.array, shape=(frames, pairs, 1)] Distances between all pairs
        atom_pairs: [list] list of atom pairs
        total_atoms: [int] total number of atoms used for the pairs
    '''
    ## FINDING TOTAL ATOMS
    total_atoms = len(atom_index)
    print("** COMPUTING ATOM PAIRS FOR %d ATOMS **"%(total_atoms))
    atom_pairs = np.array([ [i,j] for index, i in enumerate(atom_index) for j in atom_index[index:] if i != j ])
    print("TOTAL PAIRS: %d"%(len(atom_pairs) ) )
    ## FINDING DISTANCES
    distances = md.compute_distances(traj = traj,
                                     atom_pairs = atom_pairs,
                                     periodic = periodic
                                     ) # shape=(num_frames, atom_pair))
    return distances, atom_pairs, total_atoms

### FUNCTION TO CREATE DISTANCE MATRIX BASED ON PAIR DISTANCES
def create_pair_distance_matrix(atom_index, distances, atom_pairs, total_atoms = None):
    '''
    The purpose of this function is to get the distances and atom pairs to generate a distance matrix for pairs
    INPUTS:
        atom_index: [list] 
            list of atom indexes you want pair distances for
        distances: [np.array, shape=(frames, pairs, 1)] 
            Distances between all pairs
        atom_pairs: [list] 
            list of atom pairs
        total_atoms: [int, optional, default = None (will use atom_index)] 
            total number of atoms used for the pairs
    OUTPUTS:
        distances_matrix: [np.array, shape=(time_frame, total_atoms, total_atoms)] 
            Distance matrix between atoms
    '''
    ## GETTING TOTAL ATOMS
    if total_atoms == None:
        total_atoms = len(atom_index)
    
    ## FINDING TOTAL TIME
    total_time = len(distances)
    
    ## CREATING DICTIONARY TO MAP ATOM PAIRS TO INDICES IN MATRIX
    atom_index_mapping_dist_matrix = np.array([ [current_index, idx] for idx, current_index in enumerate(atom_index)])
    
    ## FIXING ATOM PAIRS NAMES
    dist_matrix_atom_pairs = correct_atom_numbers( atom_pairs, atom_index_mapping_dist_matrix )
    
    ## CREATING ZEROS ARRAY
    distances_matrix = np.zeros( (total_time, total_atoms, total_atoms)  )
    
    ## LOOPING THROUGH EACH FRAME
    for each_frame in range(total_time):
        ## DEFINING CURRENT DISTANCES
        frame_distances = distances[each_frame]
        ## LOOPING THROUGH EACH PAIR
        for idx, each_res_pair in enumerate(dist_matrix_atom_pairs):
            distances_matrix[each_frame][tuple(each_res_pair)] = frame_distances[idx]
    
        ## ADJUSTING FOR SYMMETRIC MATRIX
        distances_matrix[each_frame] = distances_matrix[each_frame] + distances_matrix[each_frame].T
        
    return distances_matrix


##########################
### CLUSTER ALGORITHMS ###
##########################
    
### FUNCTION TO CALCULATE BUNDLING GROUPS USING DBSCAN METHOD
def cluster_DBSCAN( X, eps, min_samples , metric='precomputed', verbose=False):
    '''
    The purpose of this function is to calculate the DBSCAN groups.
    Depreciated name: calc_bundling_groups_DBSCAN
    INPUTS:
        X: [np.array, shape=(N,N) or (N,2)] array of features
        eps: [float] cutoff for distances / neighbors
        min_samples: [int] number of minimum samples to be consiered a group
        metric: [str] metric used to calculate distance. 'precomputed' means that X is already pre-computed and is square
        verbose: [logical] True if you want to print out number of clusters
    OUTPUTS:
        label: [np.array, shape=(num_ligands, 1)] labels of the groups. Note that -1 means it is noisy labels
    '''
    ## IMPORTING SKLEARN MODULES
    from sklearn.cluster import DBSCAN
    ## USING DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)
    ## FINDING LABELS FOR DBSCAN
    labels = db.labels_        
    if verbose is True:
        cluster_print_num_cluster(labels)
    return labels

### FUNCTION TO USE HDBSCAN TO FIND CLUSTERS
def cluster_HDBSCAN(X, min_samples, metric='precomputed', verbose=False):
    '''
    The purpose of this function is to cluster using HDBSCAN using its defaults. 
    INPUTS:
        X: [np.array, shape=(N,N) or (N,2)] array of features
        min_samples: [int] number of minimum samples to be consiered a group
        metric: [str] metric used to calculate distance. 'precomputed' means that X is already pre-computed and is square
        verbose: [logical] True if you want to print out number of clusters
    OUTPUTS:
        label: [np.array, shape=(num_ligands, 1)] labels of the groups. Note that -1 means it is noisy labels
    '''
    ## IMPORTING HDBSCAN
    ## USAGE INFORMATION: http://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
    import hdbscan
    hdbscan_cluster = hdbscan.HDBSCAN(algorithm='best', 
                                      alpha=1.0, 
                                      approx_min_span_tree = True,
                                      gen_min_span_tree = False,
                                      leaf_size = 40,
                                      min_cluster_size = min_samples,
                                      metric=metric).fit(X)
    labels = hdbscan_cluster.labels_
    if verbose is True:
        cluster_print_num_cluster(labels)
    return labels
    
    
### FUNCTION TO PRINT THE NUMBER OF CLUSTERS FOR A GIVEN CLUSTER ALGORITHM
def cluster_print_num_cluster(labels):
    '''
    The purpose of this function is to print the number of clusters
    INPUTS:
        labels: [np.array] labels from cluster algorithm, e.g. [ 0, -1, 2, ...]
    OUTPUTS:
        void. We will print the number of labels
    '''
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d'%(n_clusters_))
    return n_clusters_


#######################################################
### CLASS TO FIND THE SULFUR STRUCTURAL INFORMATION ###
#######################################################
class nanoparticle_sulfur_structure():
    '''
    The purpose of this function is to extend the nanoparticle_structure class by inclusion of sulfur atom distances, etc. 
    INPUTS:
        traj_data: Data taken from import_traj class
        structure: [class] structure from nanoparticle_structure class
        num_neighbors: [int, optional, default=6] number of neighbors to look for
        skip_distance_calc: [logical] True if you want to skip all distance calculations
    OUTPUTS:
        ## SULFUR/STRUCTURE INFORMATION
            self.structure: [class] structure from nanoparticle_structure class
            self.sulfur_atom_to_residue_index: [np.array, shape=(num_sulfur,2)] mapping index from atom index to residue index (starts at zero)        
        ## ATOM PAIR DISTANCES
            self.sulfur_atom_pairs: [list] list of list of atom pairs between sulfur atoms
            self.sulfur_distances: [np.array, shape=(time_frame, num_pairs, 3)] distances between all sulfur atoms
        ## DISTANCE MATRIX
            self.sulfur_distances_matrix: [np.array, shape=(time_frame, num_ligands, num_ligands)] Distance matrix between sulfur atomms for each ligand
        ## NEAREST NEIGHBORS
            self.sulfur_nearest_neighbors_atom_index: [list] list of sulfur atom nearest neighbors with the same length as the number of sulfur atoms
            self.sulfur_nearest_neighbors_res_index: [np.array, shape=(sulfur_index, frame)] residue index where 0 refers to the first sulfur in structure head group index
    FUNCTIONS:
        calc_pair_distances: calculates pair distances between sulfur atoms
        calc_nearest_neighbors: calculates nearest neighbors
        calc_create_distance_matrix: creates distance matrix between pairs <-- TURNED OFF ()
        find_avg_nonassignments: finds average nonassigned ligands
        
    ACTIVE FUNCTIONS:
        plot_sulfur_sulfur_distance_dist: plots sulfur-sulfur normalized distance distribution
    ALGORITHM:
        - Find all sulfur atom indexes
        - Find all distances between sulfur atoms
        - Find nearest distances and nearest neighbors
    '''
    ### INITIALIZING
    def __init__(self, traj_data, structure, num_neighbors = 6):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ## STORING INITIAL VARIABLES
        self.num_neighbors          = num_neighbors                      # Number of neighbors to look for
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## CALCULATING MAPPING INDEX
        self.sulfur_atom_to_residue_index = np.array([ [head_grp_index, idx] for idx, head_grp_index in enumerate(structure.head_group_atom_index)])
        
        ## CALCULATING ALL PAIR DISTANCES
        self.calc_sulfur_pair_distances(traj, structure)
        
        ## FINDING DISTANCE MATRIX
        self.calc_create_distance_matrix(structure)
        
        ## FINDING NEAREST NEIGHBORS
        self.calc_nearest_neighbors(structure)
        
    ### FUNCTION TO FIND ATOM PAIRS AND DISTANCES
    def calc_sulfur_pair_distances(self, traj, structure, periodic = True):
        '''
        The purpose of this function is to calculate the pair distances for the sulfur head groups
        INPUTS:
            traj: [md.traj] trajectory from md.traj
            structure: [class] structure from nanoparticle_structure class
            periodic: [logical, default=True] True if you want PBCs
        OUTPUTS:
            self.sulfur_atom_pairs: [list] list of list of atom pairs between sulfur atoms
            self.sulfur_distances: [np.array, shape=(time_frame, num_pairs, 3)] distances between all sulfur atoms
        '''
        ### CALCULATING PAIR DISTANCES
        self.sulfur_distances, self.sulfur_atom_pairs, self.total_sulfur_atoms = calc_pair_distances(traj = traj,
                                                                            atom_index = structure.head_group_atom_index,
                                                                            periodic = periodic
                                                                            )
        return
    
    ### FUNCTION TO CREATE DISTANCE MATRIX
    def calc_create_distance_matrix(self, structure ):
        '''
        The purpose of this function is to create a distance matrix for each frame of sulfurs
        INPUTS:
            self.sulfur_atom_pairs: [list] list of list of atom pairs between sulfur atoms
            structure: [class] structure from nanoparticle_structure class
            self.sulfur_distances: [np.array, shape=(time_frame, num_pairs, 3)] distances between all sulfur atoms
        OUTPUTS:
            self.sulfur_distances_matrix: [np.array, shape=(time_frame, num_ligands, num_ligands)] Distance matrix between sulfur atomms for each ligand
        '''
        self.sulfur_distances_matrix = create_pair_distance_matrix( atom_index = structure.head_group_atom_index, 
                                                                    distances = self.sulfur_distances,
                                                                    atom_pairs = self.sulfur_atom_pairs,
                                                                    total_atoms = self.total_sulfur_atoms
                                                                   )
        return
    
    ### FUNCTION TO FIND NEAREST NEIGHBORS
    def calc_nearest_neighbors(self, structure):
        '''
        The purpose of this function is simply to get the nearest neighbors of each atom for each frame
        INPUTS:
            structure: [class] structure from nanoparticle_structure class
            self.sulfur_atom_pairs: [list] list of list of atom pairs between sulfur atoms
            self.sulfur_distances: [np.array, shape=(time_frame, num_pairs, 3)] distances between all sulfur atoms
            self.num_neighbors: [int] number of nearest neighbors
            self.sulfur_atom_to_residue_index: [np.array, shape=(num_sulfur,2)] mapping index from atom index to residue index (starts at zero)
        OUTPUTS:
            self.sulfur_nearest_neighbors_atom_index: [list] list of sulfur atom nearest neighbors with the same length as the number of sulfur atoms
                For each entry, there is a np.array which contains all nearest neighbors per frame, e.g.
                     [array([[4500, 4500, 4500,  850, 1800, 4500],
                        [4500, 4500, 4500,  850, 1800, 4500],
                        [4500, 4500, 4500,  850, 1800, 4500], ... ]
            self.sulfur_nearest_neighbors_res_index: [np.array, shape=(sulfur_index, frame)] residue index where 0 refers to the first sulfur in structure head group index
        '''
        print("** COMPUTING NEAREST SULFUR-SULFUR NEIGHBORS **")
        ## CREATING EMPTY ARRAY TO STORE ALL NEAREST NEIGHBORS
        self.sulfur_nearest_neighbors_atom_index = []
        ## LOOPING THROUGH EACH SULFUR ATOM INDEX
        for each_sulfur_index in structure.head_group_atom_index:
            ## FINDING NEAREST INDEX
            nearest_atom_index_array = find_nearest_atom_index( atom_pairs  = self.sulfur_atom_pairs,
                                                                distances   = self.sulfur_distances,
                                                                atom_index  = each_sulfur_index,
                                                                nearest_neighbors = self.num_neighbors,
                                                               )
            ## STORING NEAREST NEIGHBOR ARRAY
            self.sulfur_nearest_neighbors_atom_index.append(nearest_atom_index_array)
            
        ## AT THE END, CONVERT TO RESIDUE INDEX
        self.sulfur_nearest_neighbors_res_index = correct_atom_numbers(self.sulfur_nearest_neighbors_atom_index,  self.sulfur_atom_to_residue_index)

        return
    
    ### FUNCTION TO PLOT DISTANCE DISTRIBUTION
    def plot_sulfur_sulfur_distance_dist(self, bin_width = 0.2 ):
        '''
        The purpose of this function is to plot the distance distribution between sulfur atoms
        INPUTS:
            bin_width: [float, default = 0.2 nm] bin width for the histogram
        OUTPUTS:
            plot number of occurances versus distance between sulfur atoms
        '''
        ## CREATING FIGURE
        fig, ax = create_plot()
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Distance between sulfur atoms (nm))', **LABELS)
        ax.set_ylabel('Normalized number of occurances', **LABELS)      
        
        ## DEFINING DATA
        data = self.sulfur_distances.flatten()
        
        ## CREATING BINS
        bins = np.arange(0, np.max(data), bin_width)
        
        ## PLOTTING HISTOGRAM
        ax.hist(data, bins = bins, color  = 'k' , density=True )
        
        return
   
##############################
### FUNCTIONS FOR BUNDLING ###
##############################

### FUNCTION TO MEASURE SIMILARITY BETWEEN TWO GROUPS FOR BUNDLING GROUPS
def calc_similarity_matrix_bundling_grps(group_list_0, group_list_1 ):
    '''
    The purpose of this function is to calculate the similarity between two arrays based on their keys
    INPUTS:
        group_list_0: [dict] dictionary of different groups for group 1
        group_list_1: [dict] dictionary of different groups for group 2
        
    NOTE: this ignores similarity to dictionary group list -1
    '''
    ## GETTING KEYS FOR EACH INDEX (IGNORING '-1')
    group_1 = [each_key for each_key in group_list_0.keys() if each_key != '-1']
    group_2 = [each_key for each_key in group_list_1.keys() if each_key != '-1']
    
    ## CREATING SIMILARITY MATRIX
    similarity_matrix = np.zeros( (len(group_1), len(group_2)) ).astype('int')
    
    ## MEASURING SIMILARITY BETWEEN THE TWO GROUPS
    for i in range(len(group_1)):
        for j in range(len(group_2)):
            ## FINDING SIMILARITY
            similarity_matrix[i,j] = calc_tools.common_member_length( group_list_0[group_1[i]],
                                                             group_list_1[group_2[j]])
    return similarity_matrix, group_1, group_2

### FUNCTION TO FIND SIMILARITY BETWEEN TWO GROUPS
def find_similarity_new_group_dict(group_assignments_1, group_assignments_2, frame=0, verbose = False):
    '''
    The purpose of this script is to compare between two groups and find the most similar matching groups. This is intended to prevent vast changes in color over simulation time
    INPUTS:
        group_assignments_1: [dict] dictionary of the different groups for group 1
        group_assignments_2: [dict] same as group 1 but for group 2
    OUTPUTS:
        new_group_assignments_2_dict: [dict] dictionary of an updated group 2
    '''
    ## FINDING SIMILARITY MATRIX
    similarity_matrix, group_1, group_2 = calc_similarity_matrix_bundling_grps(group_assignments_1, group_assignments_2)
    if verbose is True:
        print("TOTAL GROUPS IN FIRST AND SECOND GROUP: %d vs. %d"%(len(group_1), len(group_2) ) )
    ## SEEING IF THERE IS A SIMILARITY TO BEGIN WITH!
    if len(group_1) > 0 and len(group_2) > 0:
        ## FINDING MOST SIMILAR GROUP WITH RESPECT TO GROUP 2
        maximum_similarity_values = np.max( similarity_matrix, axis = 1   )
        maximum_similarity_indexes = np.argmax( similarity_matrix, axis = 1  ) ## OUTPUTS SAME SIZE AS GROUP 1
        
        ## SEEING IF THE SIMILARITY IS UNIQUE -- IF NOT, WE NEED TO CORRECT THIS TO AVOID GROUPS BEING OVERWRITTEN!
        unique_elements, num_elements = np.unique(maximum_similarity_indexes, return_counts = True)
        
        ## CORRECT FOR ELEMENTS SIMILARITY
        if (num_elements != 1).sum() > 0:
            if verbose is True:
                print("Number of elements duplicated!")
            ## NOW, WE HAVE TO CORRECT FOR ISSUES OF MULTIPLE SIMILARITIES
            element_index_multiple_similarities = np.where(num_elements > 1)[0]
            ## LOOPING THROUGH EACH INDEX
            for each_index in element_index_multiple_similarities:
                ## DEFINING ELEMENT
                current_element = unique_elements[each_index]
                ## FINDING WHERE IS THE ELEMENT FOUND IN MAXIMUM SIMILARITY
                element_in_max_similarity = np.where( maximum_similarity_indexes == current_element  )
                ## DEFINING TEMPORARY SIMILARITY MATRIX
                element_similarity_matrix = similarity_matrix[element_in_max_similarity] # , :
                ## FINDING MOST SIMILAR VALUE
                element_maximums = np.max( element_similarity_matrix, axis = 1 )
                ## FINDING FINAL RESULT WITH MAXIMUM
                final_element_max_index = element_in_max_similarity[0][np.argmax(element_maximums)]
                ## ADJUSTING VALUES OF MAXIMUM SIMILARITY
                maximum_similarity_values[np.where(element_in_max_similarity[0] != final_element_max_index)] = 0
                
        ## FINDING MATCHED LISTS
        matched_1_to_2 = [ [group_1[idx], group_2[each_similarity]] for idx, each_similarity in enumerate(maximum_similarity_indexes) if maximum_similarity_values[idx] != 0 ] 
        
        ## FINDING UNMATCHED LISTS
        unmatched_group_2 = [ each_key for each_key in group_2 if each_key not in [ matched_keys[-1] for matched_keys in matched_1_to_2] ]
        
        ## CORRECTING IF NECESSARY
        if len(matched_1_to_2) > 0:
            ## CREATING NEW DICTIONARY
            new_group_assignments_2_dict = {}
            ## INCLUDING VALUES FOR -1
            try:
                new_group_assignments_2_dict['-1'] = group_assignments_2['-1']
            except Exception:
                pass
            ## LOOPING THROUGH EACH MATCHED LIST
            for index, each_matched in enumerate(matched_1_to_2):
                if verbose is True:
                    print("UPDATING FOR FRAME: %d, GROUP WAS %s, NOW SET TO %s, WITH SIMILARITY OF %d"%(frame+1, each_matched[1], each_matched[0], similarity_matrix[index, maximum_similarity_indexes[index]])   )
                new_group_assignments_2_dict[each_matched[0]] = group_assignments_2[each_matched[1]]
                
            if len(unmatched_group_2) > 0:
                    
                ## LOOPING THROUGH EACH UNMATCHED LIST
                for index, unmatched_list in enumerate(unmatched_group_2):
                    ## FINDING ALL KEYS
                    new_dict_keys = [ int(each_key) for each_key in new_group_assignments_2_dict.keys() if each_key != '-1']
                    ## FINDING LARGEST INDEX OF THE DICTIONARY
                    max_dict_index = np.max( np.array(list(new_group_assignments_2_dict.keys())).astype('int') )
                    ## SEEING ALL VALUES THAT ARE NOT SELECTED WITHIN THE INDEX
                    array_index_not_selected = [ each_value for each_value in np.arange(0, max_dict_index + 1 ) if each_value not in new_dict_keys ]
                    
                    ## ADDING BASED ON INDEX THAT IS AVAILABLE
                    if len(array_index_not_selected) > 0:
                        new_index = array_index_not_selected[0]
                    else:
                        new_index = max_dict_index + 1
                    ## ADDING TO NEW GROUP ASSIGNMENTS
                    if verbose is True:
                        print("UPDATING UNMATCHED LIST FOR FRAME: %d, GROUP WAS %s, NOW SET TO %s"%(frame+1, unmatched_list, new_index)   )
                    new_group_assignments_2_dict[str(new_index)] =  group_assignments_2[unmatched_list]
                    
        else:
            ## SIMPLY COPYING THE DICTIONARY IF THE ASSIGNMENTS HAVE NO CHANGES!
            if verbose is True:
                print("No changes found!")
            new_group_assignments_2_dict = group_assignments_2.copy()
    else:
        ## SIMPLY COPYING THE DICTIONARY IF THE ASSIGNMENTS HAVE NO CHANGES!
        if verbose is True:
            print("No changes found!")
        new_group_assignments_2_dict = group_assignments_2.copy()
        
    return new_group_assignments_2_dict

### FUNCTION TO FIND SIMILAR BUNDLING GROUPS
def find_similar_bundling_groups( group_assignments, verbose=False ):
    '''
    The purpose of this function is to match bundling groups. The main idea here is that we want colors schemes to match up between frames. To do this, we will use a similarity matrix to find similar groups.
    INPUTS:
        group_assignments: [list] list of dictionary with the group assignments of the ligands
        verbose: [logical] True if you want outputs every frame
    OUTPUTS:
        similar_groups_bundling_assignments: [list] list of dictionary with the corrected groups
    '''
    
    ## CREATING NEW LIST AND APPENDING FIRST ONE
    similar_groups_bundling_assignments =  copy.deepcopy(group_assignments[:])
    
    ## LOOPING THROUGH EACH FRAME
    for frame in range(len(similar_groups_bundling_assignments) - 1):
        
        ## DEFINING GROUP ASSIGNMENTS
        group_assignments_1 = similar_groups_bundling_assignments[frame]
        group_assignments_2 = similar_groups_bundling_assignments[frame+1]
        
        ## FINDING GROUP ASSIGNMENTS OF A NEW GROUP
        new_group_assignments_2_dict = find_similarity_new_group_dict( group_assignments_1, group_assignments_2 , frame, verbose = verbose  )
        
        ## STORING SIMILAR GROUPS
        similar_groups_bundling_assignments[frame+1] = new_group_assignments_2_dict
        
    return similar_groups_bundling_assignments


### FUNCTION TO UPDATE THE LIGAND ASSIGNMENT LIST
def update_lig_assignment_list( ligand_assignments, group_list ):
    '''
    The purpose of this function is to update the ligand assignment list based on some group list
    INPUTS:
        ligand_assignments: [list] list of ligand assignments
        group_list: [list] updated group assignments
    OUTPUTS:
        ligand_assignments: 
    '''
    ## CREATING COPY OF LIGAND ASSIGNMENTS
    updated_lig_assignments = copy.deepcopy(ligand_assignments)
    ## LOOPING THROUGH EACH FRAME
    for each_frame in range(len(updated_lig_assignments)):
        ## GOING THROUGH EACH GROUP AND RE-DEFINING
        for each_group in group_list[each_frame]:
            ## DEFINING LIGAND INDEX
            lig_index = group_list[each_frame][each_group]
            ## LOOPING THROUGH EACH INDEX OF THE GROUP
            updated_lig_assignments[each_frame][lig_index] = int(each_group)
    
    return updated_lig_assignments
    
#################################
### FUNCTIONS FOR TRANS RATIO ###
#################################
    
### FUNCTION TO DELINEATE BETWEEN ASSIGNED AND UNASSIGNED GROUPS
def find_assigned_vs_unassigned( group_list ):
    '''
    The purpose of this function is to find all assigned versus unassigned groups. The group assignment is such that '-1' denotes unassigned.
    INPUTS:
        group_list: [dict] dictionary of the group list, where '-1' key indicates unassigned. In addition, each key has a list of values.
            e.g.:
            {'-1': array([  2,   3,   9,  14,  22,  29,  44,  50,  56,  57,  78,  80,  85,
                 90,  91,  92,  97, 100, 103, 104, 106, 110, 118, 121, 123, 128,
                134, 168, 178, 193, 208, 209, 213], dtype=int64),
    OUTPUTS:
        assigned_list: [np.array, shape=(num_assigned)] list of integers that have been assigned. Note that this list is ordered.
            e.g.:
                array([  0,   1,   4,   5,   6,   7, .... ])
        unassigned_list: [np.array, shape=(num_unassigned)] list of integers that have not been assigned
    '''
    ## DEFINING EMPTY LIST FOR ASSIGNED AND UNASSIGNED
    unassigned_list, assigned_list = [], []
    for each_key in group_list.keys():
        if each_key == '-1':
            unassigned_list.extend( group_list[each_key] )
        else:
            assigned_list.extend( group_list[each_key] )
    ## SORING THE LIST
    assigned_list.sort(); unassigned_list.sort()
    ## CREATING NUMPY ARRAY
    assigned_list = np.array(assigned_list).astype('int')
    unassigned_list = np.array(unassigned_list).astype('int')
    return assigned_list, unassigned_list

### FUNCTION TO FIND ALL ASSIGNED AND UNASSIGNED LIGANDS
def find_all_assigned_vs_unassigned_ligands(ligand_grp_list):
    '''
    The purpose of this function is to find all assigned and unassigned ligands for all frames
    INPUTS:
        ligand_grp_list: [np.array, shape=(num_frames, num_groups)] group list containing ligand indices per frame
    OUTPUTS:
        assigned_ligands_list: [list] list of assigned ligands per frame
        unassigned_ligands_list: [list] list of unassigned ligands per frame
    '''
    ## CREATING EMPTY LISTS TO CAPTURE ALL ASSIGNED VS. UNASSIGNED LIGANDS
    assigned_ligands_list, unassigned_ligands_list = [], []
    
    ## LOOPING THROUGH EACH FRAME
    for each_frame in range(len(ligand_grp_list)):
        ## FINDING ALL ASSIGNED AND UNASSIGNED LIGANDS
        assigned_ligands, unassigned_ligands = find_assigned_vs_unassigned( ligand_grp_list[each_frame] ) 
        ## APPENDING TO LIST
        assigned_ligands_list.append(assigned_ligands)
        unassigned_ligands_list.append(unassigned_ligands)
    
    return assigned_ligands_list, unassigned_ligands_list


####################################################################################################################
### CLASS FUNCTION TO TAKE BUNDLING GROUPS AND TRAJECTORY INFORMATION TO CALCULATE TRANS RATIO OF BUNDLED GROUPS ###
####################################################################################################################
class calc_nanoparticle_bundling_trans_ratio:
    '''
    The purpose of this function is to calculate the trans ratio of the bundled group. We are interested in getting the number of trans- configuration for bundled groups and unbundled (i.e. unassigned ligands)
    INPUTS:
        traj_data: Data taken from import_traj class
        bundling_class: [class] bundling class from calc_nanoparticle_bundling_groups
        save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
            Turning off: self.dihedrals, self.dihedral_reference_list, self.assigned_ligands, self.unassigned_ligands
    OUTPUTS:
        ## ASSIGNED VS UNASSIGNED LIGANDS
            self.assigned_ligands: [list] list of assigned ligands per frame
            self.unassigned_ligands: [list] list of unassigned ligands per frame
        ## DIHEDRAL ANGLES
            self.dihedrals: [np.array, shape=(time_frame, dihedral_index)]  dihedral angles in degrees from 0 to 360 degrees
            self.dihedral_reference_list: [list] dihedral reference list between ligand index and the dihedral index.
        ## TRANS RATIO
            self.trans_ratio_unassigned_ligands_list: [list] list of trans ratio values for unassigned ligands
            self.trans_ratio_assigned_ligands_list: [list] list of trans ratio values for assigned ligands
        ## RESULTS
            self.results_avg_std_trans_ratio_unassigned_ligands: [dict] average and standard deviation of unassigned ligands
            self.results_avg_std_trans_ratio_assigned_ligands: [dict] average and standard deviation of assigned ligands
            
    FUNCTIONS:
        find_all_assigned_vs_unassigned_ligands: finds all assigned and unassigned ligands
        calc_ligand_dihedrals: calculates all ligand dihedrals
        calc_trans_ratio_single_frame: calculates trans ratio for a single frame given a ligand assignment list
        calc_trans_ratio_all_frames: calculates trans ratio for all possible frames
        clean_disk: cleans up disk space
        
    ALGORITHM:
        - Loop through each frame and identify ligands that are assigned versus non-assigned
        - For each frame, calculate the trans- ratio based on the heavy atoms
    '''
    ### INITIALIZING
    def __init__(self, traj_data, structure, ligand_grp_list, save_disk_space = True):
        ## STORING INPUTS
        self.structure = structure
        self.ligand_grp_list = ligand_grp_list

        ## DEFINING THE TRAJECTORY
        traj = traj_data.traj
        
        ## FINDING TOTAL NUMBER OF FRAMES
        self.total_frames = len(traj)  
        
        ## FINDING ALL ASSIGNED AND UNASSIGNED LIGANDS
        self.assigned_ligands, self.unassigned_ligands = find_all_assigned_vs_unassigned_ligands(ligand_grp_list = ligand_grp_list)
        
        ## FINDING ALL DIHEDRAL ANGLES
        self.calc_ligand_dihedrals(traj = traj)
        
        ## CALCULATING TRANS RATIO FOR EACH FRAME
        self.calc_trans_ratio_all_frames()
        
        ## CLEANING UP DISK SPACE
        self.clean_disk( save_disk_space = save_disk_space)
        return
    
    ### FUNCTION TO CLEAN UP THE SPACE
    def clean_disk(self, save_disk_space = True):
        ''' 
        This function cleans up disk space 
        INPUTS:
            save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
        '''
        if save_disk_space == True:
            self.dihedrals, self.dihedral_reference_list, self.assigned_ligands, self.unassigned_ligands = [], [], [], []
        return
    
    ### FUNCTION TO ENSURE THAT DIHEDRAL LIST IS AVAILABLE
    def calc_ligand_dihedrals(self, traj):
        '''
        The purpose of this function is to calculate ligand dihedrals using structural functions
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.dihedrals: [np.array, shape=(time_frame, dihedral_index)]  dihedral angles in degrees from 0 to 360 degrees
            self.dihedral_reference_list: [list] dihedral reference list between ligand index and the dihedral index.
        '''
        ## DIHEDRAL LIST
        dihedral_list, dihedral_reference_list = self.structure.find_dihedral_list(ligand_atom_list = self.structure.ligand_heavy_atom_index)            
        ## MEASURE DIHEDRALS
        self.dihedrals = self.structure.calc_dihedral_angles(traj = traj, dihedral_list = dihedral_list)
        ## CONVERTING REFERENCE LIST TO A NUMPY ARRAY
        self.dihedral_reference_list = np.array(dihedral_reference_list)
        return

    ### FUNCTION TO CALCULATE TRANS RATIO OF A SINGLE FRAME GIVEN TRAJECTORIES
    def calc_trans_ratio_single_frame(self, assignment_list, frame ):
        '''
        The purpose of this function is to calculate trans ratio of ligands based structure and selected ligand indexes
        INPUTS:
            - traj: trajectory from md.traj
            - assignment_list: [list] assignment list across all frames
            - frame: [int] frame to calculate the trans ratio
        OUTPUTS:
            - dihedral_trans_avg: [float] ratio of trans for the ligands you care about at a particular frame
            - total_dihedrals: [int] total number of dihedrals considered for the trans ratio calculation
        ALGORITHM:
            - Get the heavy atoms of the ligands of interest
            - Find dihedral angle list
            - Calculate a trans ratio
        '''
        ## DESIRED LIGAND INDEXES
        ligand_indexes = assignment_list[frame] #  bundling_trans_ratio.assigned_ligands[frame]
        
        ## FINDING ALL DIHEDRALS THAT MATTER
        assignment_dihedrals_indexes = self.dihedral_reference_list[ligand_indexes].flatten()
        assignment_dihedrals_array = np.array([self.dihedrals[frame][assignment_dihedrals_indexes]]) ## CONVERTING TO A TIME-BASIS
        
        ## FINDING TOTAL DIHEDRALS
        total_dihedrals = len(assignment_dihedrals_array)
        ## SEEING IF WE HAVE DIHEDRALS!
        # if total_dihedrals == 0:
        # return IndexError, 0
        # else:
        ## CALCULATING TRANS RATIO
        dihedral_trans_avg = self.structure.calc_dihedral_ensemble_avg_ratio(assignment_dihedrals_array)[2]
        return dihedral_trans_avg, total_dihedrals
        
    ### LOOPING THROUGH EACH FRAME AND CALCULATING TRANS RATIO FOR EACH FRAME AND LIGAND
    def calc_trans_ratio_all_frames( self , print_frequency = 100):
        '''
        The purpose of this function is to calculate the trans ratio for all frames
        INPUTS:
            self: class object
            print_frequency: [int] print frequency for the working on frame
        OUTPUTS:
            ## TRANS RATIO
            self.trans_ratio_unassigned_ligands_list: [list] list of trans ratio values for unassigned ligands
            self.trans_ratio_assigned_ligands_list: [list] list of trans ratio values for assigned ligands
            ## TOTAL DIHEDRALS
            self.trans_ratio_unassigned_total_dihedrals: [list] total number of dihedrals considered for the unassigned ligands
            self.trans_ratio_assigned_total_dihedrals: [list] total number of dihedrals considered for assigned ligands
            ## RESULTS
            self.results_avg_std_trans_ratio_unassigned_ligands: [dict] average and standard deviation of unassigned ligands
            self.results_avg_std_trans_ratio_assigned_ligands: [dict] average and standard deviation of assigned ligands
        NOTES:
            - trans ratio of unassigned or assigned ligand can be non-existing -- we will ignore those by including them as "nan"s
        '''
        ## CREATING EMPTY LISTS TO STORE THE TRANS RATIO
        self.trans_ratio_unassigned_ligands_list, self.trans_ratio_assigned_ligands_list = [], []
        ## STORING TOTAL NUMBER OF DIHEDRALS
        self.trans_ratio_unassigned_total_dihedrals, self.trans_ratio_assigned_total_dihedrals = [], []
        ## PRINTING
        print("\n---- CALCULATING TRANS RATIO ----")
        ## LOOPING THROUGH EACH FRAME
        for each_frame in range(self.total_frames):
            ## PRINTING
            if each_frame % print_frequency == 0:                
                print("WORKING ON FRAME: %d OUT OF %d"%(each_frame,self.total_frames))
            ## LOOPING THROUGH EACH LIST
            for idx, each_list in enumerate([ self.unassigned_ligands, self.assigned_ligands ]):
                ## SEEING IF WE HAVE A TRANS RATIO TO BEGIN WITH
                try:                        
                    ## CALCULATING DIHEDRAL AND TRANS FOR EACH FRAME
                    dihedral_trans, total_dihedrals = self.calc_trans_ratio_single_frame( assignment_list = each_list,
                                                                                      frame = each_frame,
                                                                                      )
                except IndexError:  ## INDEX ERROR RAISED WHEN THERE ARE NO DIHEDRALS
                    dihedral_trans = np.nan
                    total_dihedrals = 0
                    
                ## STORING VALUE
                [ self.trans_ratio_unassigned_ligands_list, self.trans_ratio_assigned_ligands_list ][idx].append(dihedral_trans)
                [ self.trans_ratio_unassigned_total_dihedrals, self.trans_ratio_assigned_total_dihedrals ][idx].append(total_dihedrals)

        ## FINDING MEAN AND STD
        # self.results_avg_std_trans_ratio_unassigned_ligands = self.trans_ratio_unassigned_ligands_list
        # self.results_avg_std_trans_ratio_assigned_ligands = self.trans_ratio_assigned_ligands_list
        
        self.results_avg_std_trans_ratio_unassigned_ligands = calc_tools.calc_avg_std_of_list(self.trans_ratio_unassigned_ligands_list)
        self.results_avg_std_trans_ratio_assigned_ligands = calc_tools.calc_avg_std_of_list(self.trans_ratio_assigned_ligands_list)
        return        


######################################
### CLASS TO CHARACTERIZE BUNDLING ###
######################################
class calc_nanoparticle_bundling_groups():
    '''
    The purpose of this function is to calculate the nanoparticle bundling groups
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        min_cluster_size: [int] minimum clusters to even consider bundling
        weights: [list] weights between the main contributors. The list should sum to 1.
        separated_ligands: [logical] True if your ligands are not attached to each other -- True in the case of planar SAMs
        save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
            Turning off: self.lig_displacements, self.tail_tail_distance_matrix, self.lig_displacement_angles
        displacement_vector_type: [str]
            Type of displacement vector to choose. By default, this is 'terminal_heavy_atom'
                'terminal_heavy_atom': 
                    Choose the last heavy atom of each of the ligands. This was 
                    used originally in the 2018 NP work.
                'avg_heavy_atom':
                    Average all the heavy atoms as a displacement vector
    OUTPUTS:
        ## INPUTS
            self.constraint_angle: [float, default = 30] angle in degrees between two ligands to constraint and determine whether it is in a group
        ## TRAJ INFORMATION
            self.total_frames: [int] total number of frames
        ## STRUCTURE AND SULFUR COORDINATES
            self.structure_np: [class] structure from nanoparticle_structure class
                - contains information about the ligand structure, such as the indexes for heavy atoms, etc.
            *** DEPRECIATED CLASS ***
            self.structure_sulfur: [class] structure from nanoparticle_sulfur_structure class
                - contains information about the sulfur atoms and its neighboring sulfur atoms
        ## DISPLACEMENT VECTOR
            self.lig_displacements: [np.array, shape=(frame, num_ligands, 3)] 
                displacement vector of ligand over time
        ## ANGLES TO ANGLES
            self.lig_displacement_angles: [np.array, shape=(frame, num_ligands, num_ligands)] 
                angles for all possible ligand pairs
        ## DISTANCE MATRIX
            self.tail_tail_distance_matrix: [np.array, shape=(time_frame, total_atoms, total_atoms)] 
                Distance matrix between tail-tail end groups
        ## BUNDLING GROUPS
            self.lig_grp_assignment: [list] 
                list of group assignments per frame
            self.lig_grp_list: [list] 
                list of groups
            self.lig_total_bundling_groups: [list] 
                total number of bundling groups over time
        ## RESULTS
            self.results_avg_std_bundling_grps: [dict] 
                Average and standard deviation of bundling groups
            self.results_avg_std_nonassignments: [dict] 
                dictionary with the average and std of nonassignments
            results_avg_std_group_size: [dict] 
                dictionary with the average and std of ligand group sizes
            self.lig_nonassignments_list: [list] 
                list of non-assignments
        ## SIMILARITY GROUPS FOR LIGANDS
            self.similarity_lig_grp_list: [list] 
                updated ligand group list with similarity between frames
            self.similarity_lig_grp_assignment: [list] 
                similar to list except with ligand assignments
        ## TRANS RATIO
            self.trans_ratio: 
                contains all trans ratio information for the assigned and unassigned ligands based on calc_nanoparticle_bundling_trans_ratio class
    FUNCTIONS:
        calc_ligand_vectors: 
            calculates ligand displacement vectors
        calc_tail_tail_distances: 
            calculates tail-to-tail distances
        calc_ligand_angles: 
            calculates ligand angles matrix
        cluster_HDBSCAN_angles_distances: 
            calculates bundling based on angles and distances
        calc_bundling_groups_all_frames: 
            calculates all bundling group per frame
        find_avg_nonassignments: 
            finds average and standard deviation of nonassignments
        find_avg_group_size: 
            finds average and standard deviation of group sizes
        
    '''
    ### INITIALIZING
    def __init__(self, 
                 traj_data, 
                 ligand_names, 
                 itp_file, 
                 min_cluster_size, 
                 weights = [0.50, 0.50], 
                 separated_ligands = False, 
                 save_disk_space = True,
                 displacement_vector_type = 'terminal_heavy_atom'):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        ## STORING INPUT VARIABLES
        self.min_cluster_size = min_cluster_size
        self.weights = weights
        self.displacement_vector_type = displacement_vector_type
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = ligand_names,        # ligand names
                                                itp_file            = itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = separated_ligands    # True if you want separated ligands 
                                                )
        ''' <-- DEPRECIATED -- do not need structure sulfur!
        ## RUNNING SULFUR STRUCTURE
        self.structure_sulfur = nanoparticle_sulfur_structure( traj_data = traj_data,
                                                                structure = self.structure_np,
                                                                num_neighbors = nearest_neighbors,
                                                                )
        '''
        ## DEFINING THE TRAJECTORY
        traj = traj_data.traj
        
        ## FINDING TOTAL NUMBER OF FRAMES
        self.total_frames = len(traj)  
        
        ## FINDING ALL LIGAND DISPLACEMENTS
        self.calc_ligand_vectors(traj)
        
        ## FINDING ALL LIGAND TO LIGAND TAIL GROUP DISTANCES
        self.calc_tail_tail_distances(traj)
        
        ## FINDING ALL LIGAND ANGLES
        self.calc_ligand_angles()
        
        ## FINDING ALL BUNDLING GROUPS
        self.calc_bundling_groups_all_frames()
        
#        ## CALCULATING TRANS RATIO
#        self.trans_ratio = calc_nanoparticle_bundling_trans_ratio(traj_data = traj_data,
#                                                                  structure = self.structure_np,
#                                                                  ligand_grp_list = self.lig_grp_list,
#                                                                  save_disk_space = save_disk_space
#                                                                  )
        
        ## PRINTING SUMMARY
        self.print_summary()
        
        ## SAVING DISK SPACE
        if save_disk_space is True:
            self.lig_displacements, self.tail_tail_distance_matrix, self.lig_displacement_angles = [], [], []
        
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function prints a summary '''
        print("\n----- SUMMARY -----")
        print("TOTAL NUMBER OF LIGANDS: %d"%(self.structure_np.total_ligands))
        print("MINIMUM CLUSTER SIZE: %d"%(self.min_cluster_size) )        
        print("AVERAGE LIGAND BUNDLING: %.2f +/- %.2f"%(self.results_avg_std_bundling_grps['avg'], self.results_avg_std_bundling_grps['std']))
        print("AVERAGE NONASSIGNMENTS: %.2f +/- %.2f"%(self.results_avg_std_nonassignments['avg'], self.results_avg_std_nonassignments['std']))
        print("AVERAGE GROUP SIZE: %.2f +/- %.2f"%(self.results_avg_std_group_size['avg'], self.results_avg_std_group_size['std'])  )
        return
        
    ### FUNCTION TO FIND ALL VECTORS ASSOCIATED WITH
    def calc_ligand_vectors(self, traj, periodic = True ):
        '''
        The purpose of this function is to find the vector associated with each ligand
        INPUTS:
            traj: [obj]
                trajectory from md.traj
            periodic: [logical] 
                True if you want periodic boundaries
        OUTPUTS:
            self.lig_displacements: [np.array, shape=(frame, num_ligands, 3)] displacement vector of ligand over time
        '''
        print("** COMPUTING LIGAND VECTORS **")
        if self.displacement_vector_type == 'terminal_heavy_atom':
            print("Working on finding terminal heavy atom ligand vectors")
            ## FINDING ALL LAST HEAVY ATOM INDEX
            self.terminal_group_index=[each_ligand[-1] for each_ligand in self.structure_np.ligand_heavy_atom_index ]
            ## GENERATING ATOM PAIRS
            atom_pairs = [ [each_sulfur_index, self.terminal_group_index[idx] ] for idx, each_sulfur_index in enumerate(self.structure_np.head_group_atom_index) ]
            print("TOTAL ATOM PAIRS: %d"%(len(atom_pairs) ) )
            ## CALCULATING DISPLACEMENTS FOR THE LIGANDS
            self.lig_displacements = md.compute_displacements( traj = traj,
                                                               atom_pairs = atom_pairs,
                                                               periodic = periodic,
                                                               )
        elif self.displacement_vector_type == 'avg_heavy_atom':
            print("Working on finding average heavy atom ligand vectors")
            ## COMPUTING DISPLACEMENTS
            self.lig_displacements = compute_lig_avg_displacement_array(traj = traj,
                                                                        sulfur_atom_index = self.structure_np.head_group_atom_index,
                                                                        ligand_heavy_atom_index = self.structure_np.ligand_heavy_atom_index,
                                                                        periodic = periodic)
            
        else:
            print("Error in 'displacement_vector_type' variable!")
            print("%s is not defined, consider using a different type"%(self.displacement_vector_type))
            sys.exit()
        return 
    
    ### FUNCTION TO COMPUTE TAIL TO TAIL DISTANCES
    def calc_tail_tail_distances( self, traj, periodic = True ):
        '''
        The purpose of this function is to calculate the tail-to-tail distances between ligand vectors
        INPUTS:
            traj: trajectory from md.traj
            periodic: [logical] True if you want periodic boundaries
        OUTPUTS:
            self.tail_tail_distance_matrix: [np.array, shape=(time_frame, total_atoms, total_atoms)] 
                Distance matrix between tail-tail end groups
        '''
        print("** COMPUTING LIGAND TAIL-TO-TAIL DISTANCES **")
        if self.displacement_vector_type == 'terminal_heavy_atom':
            ## COMPUTING PAIR DISTANCES, ETC.
            distances, atom_pairs, total_atoms = calc_pair_distances(traj = traj, 
                                                                     atom_index = self.terminal_group_index, 
                                                                     periodic=periodic)
            ## DEFINING ATOM INDEX
            atom_index = self.terminal_group_index
            
        elif self.displacement_vector_type == 'avg_heavy_atom':
            ## COMPUTING PAIR DISTANCES WITH ARBITRARY DISPLACEMENTS
            distances, atom_pairs, total_atoms, atom_index = calc_tools.compute_pair_distances_with_arbitrary_displacements(traj = traj,
                                                                                                                            displacements = self.lig_displacements,
                                                                                                                            periodic=periodic
                                                                                                                            )

        ## COMPUTING DISTANCE MATRIX
        self.tail_tail_distance_matrix = create_pair_distance_matrix( atom_index = atom_index,
                                                                      distances = distances,
                                                                      atom_pairs = atom_pairs,
                                                                      total_atoms = total_atoms
                                                                     )
            
        return
    
    ### FUNCTION TO FIND DOT PRODUCT OF ALL VECTORS
    def calc_ligand_angles(self, ):
        '''
        The purpose of this function is to calculate the angles of all the ligands and output them in a N x N form where N is the number of ligands
        INPUTS:
            self.lig_displacements: [np.array, shape=(frame, num_ligands, 3)] displacement vector of ligand over time
        OUTPUTS:
            self.lig_displacement_angles: [np.array, shape=(frame, num_ligands, num_ligands)] angles for all possible ligand pairs
        '''
        print("** COMPUTING LIGAND ANGLES **")
        ### CREATING ARRAY THAT WILL FIT THE ENTIRE VECTOR
        self.lig_displacement_angles = np.zeros( (self.total_frames, self.structure_np.total_ligands,self.structure_np.total_ligands)  )
        
        ### LOOPING THROUGH EACH FRAME AND CALCULATING ANGLES
        for each_frame in range(self.total_frames):
            self.lig_displacement_angles[each_frame] = find_angle_btwn_displacement_vectors( self.lig_displacements[each_frame] )
        return
    
    ### FUNCTION TO ACCOUNT FOR BOTH ANGLES AND DISTANCES USING SIMILARITY MATRIX
    def cluster_HDBSCAN_angles_distances(self, frame, min_samples, weights=[0.50, 0.50]): # 0.85, 0.15 0.50, 0.50
        '''
        The purpose of this script is to cluster HDBSCAN using angle and distances similarity matrix. The matrices are rescaled and averaged.
        INPUTS:
            frame:[int] frame to calculate the labels
            min_samples:[int] minimum number of samples
            weights: [list, length = 2] weight of each one as floating points, where first index is the weight of angles and second index is the weight of the distances
        OUTPUTS:
            labels: [np.array, shape=(N,1)] labels after using hdbscan
        '''
        ### FINDING ANGLE VERSUS DISTANCE
        angles = self.lig_displacement_angles[frame]
        distances = self.tail_tail_distance_matrix[frame]  # self.structure_sulfur.sulfur_distances_matrix[frame]
        
        ### RESCALING ANGLES AND DISTANCES
        angles_rescaled = calc_tools.rescale_vector( angles )
        distances_rescaled = calc_tools.rescale_vector( distances )
        
        ### SUMMING AND DIVIDING
        # feature_matrix = np.mean( (angles_rescaled, distances_rescaled ), axis =0)
        feature_matrix = np.average( (angles_rescaled, distances_rescaled ), axis = 0, weights = weights)
        ## RUNNING HDBSCAN
        labels = cluster_HDBSCAN( X = feature_matrix, min_samples = min_samples, metric='precomputed', verbose = False )
        return labels
    
    ### FUNCTION TO FIND ALL LIGANDS BUNDLED FOR ALL FRAMES
    def calc_bundling_groups_all_frames(self, print_traj_index = 100):
        '''
        The purpose of this function is to calculate bundling groups by extension of the "calc_bundling_groups_single_frame" function
        INPUTS:
            print_traj_index: [int] trajectory index to print output
        OUTPUTS:
            self.lig_grp_assignment: [list] list of group assignments per frame
            self.lig_grp_list: [list] list of groups and their assignments, each of which is in a form of a dictionary
            self.lig_total_bundling_groups: [list] total number of bundling groups over time
            self.results_avg_std_group_size: [dict] dictionary with the average and std of ligand group sizes
            self.results_avg_std_bundling_grps: [dict] Average and standard deviation of bundling groups
            self.results_avg_std_nonassignments: [dict] dictionary with the average and std of nonassignments
            ## SIMILARITY GROUPS FOR LIGANDS
            self.similarity_lig_grp_list: [list] updated ligand group list with similarity between frames
            self.similarity_lig_grp_assignment: [list] similar to list except with ligand assignments
        '''
        ## CREATING LIST OF LIST FOR 
        self.lig_grp_assignment =[ [] for each_frame in range(self.total_frames) ]
        self.lig_grp_list = self.lig_grp_assignment[:]
        
        ## LOOPING THROUGH EACH FRAME AND GETTING BUNDLING
        for frame_index in range(self.total_frames): # 
            ## PRINTING
            if frame_index % print_traj_index == 0:
                print("WORKING ON FRAME: %d OUT OF %d"%(frame_index, self.total_frames))
        
            ## RUNNING BUNDLING FUNCTION
            ligand_grp_assignment = self.cluster_HDBSCAN_angles_distances( frame = frame_index,
                                                                          min_samples = self.min_cluster_size,
                                                                          weights = self.weights,
                                                                          )

            ## APPENDING TO LIST (COPYING ARRAY)
            self.lig_grp_assignment[frame_index] = ligand_grp_assignment[:]
            self.lig_grp_list[frame_index] = find_group_assignments(ligand_grp_assignment)
        
        
        ## FINDING TOTAL BUNDLED GROUP FOR ASSIGNMENT LIST
        self.lig_total_bundling_groups = [ (np.array(list(self.lig_grp_list[each_frame].keys())).astype('int') != -1).sum() for each_frame in range(len(self.lig_grp_list)) ] # np.count_nonzero(np.where(np.array(list
        
        ## UPDATING BY SIMILAR GROUPS <-- need to double-check
        # self.lig_grp_list = find_similar_groups_bundling_index( group_assignments = self.lig_grp_list)
        # self.lig_grp_assignment = update_lig_assignment_list( ligand_assignments =self.lig_grp_assignment, group_list = self.lig_grp_list )
        
        ## FINDING AVERAGE BUNDLING GROUP
        self.results_avg_std_bundling_grps = calc_tools.calc_avg_std_of_list(self.lig_total_bundling_groups)
        
        ## FINDING AVERAGE NONASSIGNMENTS
        self.find_avg_nonassignments()
        
        ## FINDING AVERAGE GROUP SIZE
        self.find_avg_group_size()
        
        ## FINDING ASSIGNMENTS FOR PLOTTING BASED ON SIMILARITIES BETWEEN GROUPS
        self.similarity_lig_grp_list, self.similarity_lig_grp_assignment = self.find_similar_groups_btn_frames()
        
        return
    
    ### FUNCTION TO CALCULATE THE AVERAGE NUMBER OF UNASSIGNED GROUP FOR THE SIMULATION
    def find_avg_nonassignments(self):
        '''
        The purpose of this function is to calculate the average number of non-assignments
        INPUTS:
            self: class object
        OUTPUTS:
            self.results_avg_std_nonassignments: [dict] dictionary with the average and std of nonassignments
            self.lig_nonassignments_list: [list] list of non-assignments
        '''
        assignments = self.lig_grp_assignment[:]
        total_frames = len(assignments)
        self.lig_nonassignments_list = np.zeros( (total_frames , 1))
        ## LOOPING THROUGH THE FRAMES
        for each_frame in range(total_frames):
            ## FINDING TOTAL NUMBER OF UNCLASSIFIED LIGANDS
            total_negatives = np.count_nonzero(assignments[each_frame]==-1)
            ## STORING TO FRAME
            self.lig_nonassignments_list[each_frame] = total_negatives
            
        ## FINDING AVERAGE AND STD
        self.results_avg_std_nonassignments = calc_tools.calc_avg_std_of_list(self.lig_nonassignments_list)
        return
    
    ### FUNCTION TO FIND THE AVERAGE GROUP SIZE
    def find_avg_group_size(self):
        '''
        The purpose of this function is to find the average group size of ligands
        INPUTS:
            self: class object
                group_list: [list] list of groups for each frame
        OUTPUTS:
            results_avg_std_group_size: [dict] dictionary with the average and std of ligand group sizes
        '''
        group_list = self.lig_grp_list[:]
        ## DEFINING TOTAL FRAMES
        total_frames = len(group_list)
        ## DEFINING EMPTY MATRIX TO STORE AVERAGE GROUP SIZE
        avg_group_size = []
        ## LOOPING THROUGH THE FRAMES
        for each_frame in range(total_frames):
            ## DEFINING TOTAL LIGAND COUNTER AND GROUP COUNTER
            total_ligs = 0; total_grps = 0
            ## LOOPING THROUGH EACH DICTIONARY AND IGNORING -1 CLASSIFICATION
            for each_key in group_list[each_frame].keys():
                if each_key != '-1':
                    total_ligs += len( group_list[each_frame][each_key]  )
                    total_grps += 1
            ## STORING AVERAGE GROUP SIZE
            if total_grps != 0:
                avg_group_size.append(total_ligs / float(total_grps))
            
        ## FINDING AVERAGE AND STD
        self.results_avg_std_group_size = calc_tools.calc_avg_std_of_list(avg_group_size)
        return
    
    ### FUNCTION TO FIND SIMILAR GROUPS BETWEEN FRAMES
    def find_similar_groups_btn_frames(self):
        '''
        The purpose of this function is to find groups that are similar in between two frames. This way, we can avoid issues with color clashing over multiple frames.
        INPUTS:
            self: class object
        OUTPUTS:
            similar_groups_bundling_list: [list] updated lig group assignment
            similar_group_bundling_assignments: [list] updated ligand assignment list
        '''
        ### FINDING SIMILAR GROUP ASSIGNMENTS
        similar_groups_bundling_list = find_similar_bundling_groups(self.lig_grp_list)
        
        ### UPDATING LIGAND ASSIGNMENT LIST PER FRAME
        similar_group_bundling_assignments = update_lig_assignment_list( ligand_assignments = self.lig_grp_assignment, group_list = similar_groups_bundling_list  )
        
        return similar_groups_bundling_list, similar_group_bundling_assignments
        

################################################
### CLASS TO PLOT BUNDLING GROUP INFORMATION ###
################################################
class plot_nanoparticle_bundling_groups():
    '''
    The purpose of this class is to plot nanoparticle bundling group using matplotlib
    INPUTS:
        bundling: [class] bundling information from calc_nanoparticle_bundling_groups
    OUTPUTS:
        
    ACTIVE FUNCTIONS:
        ## VIEWING VECTORS
            plot_sulfur_lig_grp_all_frames: Main function that plots ligand groups for all frames
            plot_sulfur_lig_grp_frame: Plots vectors for a single frame
            plot_sulfur_atoms: Plots sulfur atoms on a figure
            plot_ligand_vectors: Plots the vectors of the ligands
        ## PLOTS
            plot_num_bundling_vs_time: Plots number of bundling groups per time
            plot_angle_distribution: plots angle distribution
    '''
    ### INITIALIZING
    def __init__(self, bundling):
        
        ## STORING BUNDLING
        self.bundling = bundling
        
        return
        
    ### FUNCTION TO PLOT SULFUR LIGANDS FOR MULTIPLE FRAMES
    def plot_sulfur_lig_grp_all_frames(self, pause_time=.5):
        '''
        The purpose of this function is to plot ligand groups for all frames
        INPUTS:
            pause_time: [float] time to pause in seconds
        OUTPUTS:
            plots of the ligand per frame
        '''
        import matplotlib.pyplot as plt
        ## LOOPING THROUGH ALL FRAMES
        total_frames = self.bundling.total_frames
        
        ## DEFAULT FIG, AX TO NONE
        fig, ax = None, None
        
        ## LOOPING THROUGH EACH FRAME
        for each_frame in range(total_frames):
            ## PRINTING
            if (each_frame % 1 == 0): # 10
                print("--> PLOTTING FRAME: %d"%(each_frame))
            ## RUNNING SULFUR LIGAND PER FRAME
            fig, ax = self.plot_sulfur_lig_grp_frame(frame_index = each_frame, fig=fig, ax = ax)
            
            ## PAUSING SO YOU CAN VIEW IT
            plt.pause(pause_time) # Pause so you can see the changes            
            ## CLEARING PLOT
            if each_frame != total_frames - 1:
                plt.cla()
        return
        
    ### FUNCTION TO PLOT SULFUR LIGANDS FOR A SINGLE FRAME
    def plot_sulfur_lig_grp_frame(self, frame_index = 1, fig = None, ax = None):
        '''
        The purpose of this script is to plot the sulfurs and the ligands for a single frame
        INPUTS:
            frame_index: index for a frame
        OUTPUTS:
            plot for the figure
        '''
        ### CREATING FIGURE
        if fig is None and ax is None:
            fig, ax = create_3d_axis_plot()
        
        ## ADDING X, Y, Z LABELS
        ax.set_xlabel('x (nm)', **LABELS)
        ax.set_ylabel('y (nm)', **LABELS)
        ax.set_zlabel('z (nm)', **LABELS)
        
        ## WRITING THE TOTAL FRAMES AS TITEL
        ax.set_title( 'FRAME: %d\nNUMBER OF CLUSTERS: %d'%(frame_index, self.bundling.lig_total_bundling_groups[frame_index])   , **LABELS )
    
        ## ADDING SULFUR ATOMS
        fig, ax = self.plot_sulfur_atoms(   geometry = self.bundling.structure_np.head_group_geom[frame_index],
                                            fig = fig, 
                                            ax = ax )
        
        ## ADDING VECTORS
        fig, ax = self.plot_ligand_vectors(    origin_vectors   = self.bundling.structure_np.head_group_geom[frame_index] , 
                                               pointing_vectors = self.bundling.lig_displacements[frame_index], 
                                               group_index      = self.bundling.lig_grp_assignment[frame_index],
                                               fig              = fig, 
                                               ax               = ax,
                                               )
        return fig, ax
        
    ### FUNCTION TO PLOT SULFUR ATOMS
    def plot_sulfur_atoms(self, geometry, fig, ax ):
        '''
        The purpose of this function is to plot all the sulfur atoms given a figure and axis
        INPUTS:
            fig, ax: figure and axis
            geometry: [np.array, shape=(N,3)] geometry of the sulfur atoms
        OUTPUTS:
            fig, ax: updated figure and axis
        '''
        ## ADDING TO PLOT
        ax.scatter(geometry[:, 0], geometry[:, 1], geometry[:, 2], marker = 'o', edgecolors = 'black', color='yellow', s=50, linewidth=2 )
        
        return fig, ax
    
    ### FUNCTION TO PLOT LIGAND VECTORS
    def plot_ligand_vectors(self,origin_vectors, pointing_vectors,group_index, fig, ax ):
        '''
        The purpose of this function is to plot ligand vectors. We will also color in the ligands appropriately
        INPUTS:
            origin_vectors: [np.array, shape=(N,3)] vectors with origin
            pointing_vectors: [np.array, shape=(N,3)] vectors that are pointing
            group_index: [np.array, shape=(N,1)] group index that it is in
            fig, ax: figure and axis
        OUTPUTS:
            fig, ax
        '''
        ## FINDING UNIQUE VALUES
        values = np.unique(group_index)
        
        ## IF VALUES ARE GREATER THAN COLOR LIST, THEN REPLICATE COLOR LIST UNTIL THE LENGTH IS SATISFIED
        if len(values) > len(COLOR_LIST):
            division_color_list = np.ceil( len(values) / float(len(COLOR_LIST)) )
            color_list = COLOR_LIST * int(division_color_list)
        else:
            color_list = COLOR_LIST
            
        ## FINDING COLORS
        colors = [color_list[each_assignment] if each_assignment != -1 else 'gray' for each_assignment in group_index]
        
        ## MAKING EDITS TO COLOR TO CORRECT FOR QUIVER 
        # See reference: https://stackoverflow.com/questions/28420504/adding-colors-to-a-3d-quiver-plot-in-matplotlib?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        # colors=[color_1, color_2, ..., color_n, color_1, color_1, color_2, color_2, ..., color_n, color_n]
        color_arrows =[ [each_color, each_color] for each_color in colors] 
        color_arrows_flat_list = [item for sublist in color_arrows for item in sublist]
        
        ## ADDING TO COLORS
        colors = colors + color_arrows_flat_list
        
        ## USING QUIVER TO DRAW THE POINTS
        ax.quiver(origin_vectors[:,0], origin_vectors[:,1], origin_vectors[:,2], pointing_vectors[:,0], pointing_vectors[:,1], pointing_vectors[:,2],
                  color = colors, alpha=.8, length = 1, normalize = True, linewidths = 2) #'black'
        return fig, ax
        
    ### FUNCTION TO PLOT NUMBER OF BUNDLING GROUPS PER TIME
    def plot_num_bundling_vs_time(self, ps_per_frame = 100,):
        '''
        The purpose of this function is to plot the number of bundling group versus simulation time
        INPUTS:
            ps_per_frame: [float] number of picoseconds per frame
        OUTPUTS:
            x: [list] time frame
            y: [list] number of bundling group
            plot with number of bundling versus time
        '''
        ## CREATING FIGURE
        fig, ax = create_plot()
        
        ## GENERATING X AND Y
        x = np.arange(self.bundling.total_frames)*ps_per_frame / 1000.0 # ns
        y = self.bundling.lig_total_bundling_groups[:]
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Simulation time (ns)', **LABELS)
        ax.set_ylabel('Number of bundling groups', **LABELS)        
        ## PLOTTING
        ax.plot(x, y, color='k', **LINE_STYLE)        
        ## FINDING AVERAGE BUNDLING GROUP
        avg_bundling_grp = np.mean(y) 
        ## PLOTTING HORIZONTAL LINE
        ax.axhline(y = avg_bundling_grp, linestyle='--', color='r', label='Average bundling', **LINE_STYLE)
        ## ADDING LEGEND
        ax.legend()
        return x, y
        
    ### FUNCTION TO PLOT ANGLE DISTRIBUTION
    def plot_angle_distribution(self, bin_array = np.arange(0, 185, 5)):
        '''
        The purpose of this function is to plot the angle distribution as a form of a historgram
        INPUTS:
            bin_array: [np.array] array of bins that you are interested in 
        OUTPUTS:
            
        '''
        ## CREATING FIGURE
        fig, ax = create_plot()
        
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Angles (degrees)', **LABELS)
        ax.set_ylabel('Normalized number of occurances', **LABELS)      
        
        ## FINDING HISTOGRAM
        # frequency, values = np.histogram(self.bundling.angles_list)
        
        ## FINDING ANGLE INFORMATION
        # angle_data = np.array(self.bundling.angles_list).flatten()
        # angle_data = np.concatenate(self.bundling.angles_list) # .flatten()
        angle_data = self.bundling.lig_displacement_angles.flatten()
        bin_array = np.arange(0, 180, 1)
        
        ## PLOTTING HISTOGRAM
        ax.hist(angle_data, bins = bin_array, color  = 'k' , density=True )
        
        # ax.hist(self.bundling.angles_list, bins = bin_array, stacked = True)
        
        ## PRINTING CUTOFF LINE
        # ax.axvline(x = self.bundling.constraint_angle, linestyle = '--', color='r',label='Angle cutoff', **LINE_STYLE )
        
        ## ADDING LEGEND
        # ax.legend()

        return
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180702-Trial_1_spherical_EAM_correction" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_4_nmDIAM_hexadecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    
    '''
    analysis_dir=r"180511-Planar_sims_alkanethiol" # Analysis directory
    category_dir="Planar" # category directory
    specific_dir="Planar_310.15_K_butanethiol_10x10_CHARMM36_intffGold" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    '''
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    # path2AnalysisDir=r"/Volumes/akchew/scratch/nanoparticle_project/analysis/" + analysis_dir + '/' + category_dir + '/' + specific_dir + '/' # MAC side

    ### DEFINING FILE NAMES
    gro_file=r"sam_prod_10_ns_whole_no_water_center.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole_no_water_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    '''
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    '''

    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    
    #%%
    
    ### DEFINING INPUT DATA
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                        'ligand_names'      :           ['OCT', 'BUT', 'HED', 'DEC'],   # Name of the ligands of interest
                        'itp_file'          :           'sam.itp',                      # ITP FILE
                        'min_cluster_size'  :           3,                              # Minimum cluster size 6
                        'save_disk_space'   :          True    ,                        # Saving space
                        'weights'           :           [0.5, 0.5],                     # Weights between particle group (should sum to 1)
                        'separated_ligands' :           False,
                        }
    #%%
    ### CALCULATE BUNDLING GROUPS    
    bundling_groups = calc_nanoparticle_bundling_groups( **input_details )   
    
    
    #%%
        

    
    plot_bundling_groups_trans_distribution(bundling_groups)
    
    
    
    
    
    #%%
    
    ### RUNNING TRANS RATIO
    trans_ratio = calc_nanoparticle_bundling_trans_ratio( traj_data = traj_data,
                                                          structure = bundling_groups.structure_np,
                                                          ligand_grp_list = bundling_groups.lig_grp_list,
                                                          save_disk_space = False)
    #%%
    test = trans_ratio.calc_trans_ratio_single_frame( assignment_list = trans_ratio.assigned_ligands,
                                              frame = 0)
    
    #%%
    
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
        dihedral_gauche_std = np.std(dihedral_gauche_avg)
        
        return dihedral_gauche_avg, dihedral_gauche_std, dihedral_trans_avg, dihedral_trans_std
    ## DESIRED LIGAND INDEXES
    frame = 0
    ligand_indexes = trans_ratio.assigned_ligands[frame] #  bundling_trans_ratio.assigned_ligands[frame]
    
    ## FINDING ALL DIHEDRALS THAT MATTER
    assignment_dihedrals_indexes = trans_ratio.dihedral_reference_list[ligand_indexes].flatten()
    assignment_dihedrals_array = trans_ratio.dihedrals[frame][assignment_dihedrals_indexes]
    
    ## FINDING TOTAL DIHEDRALS
    total_dihedrals = len(assignment_dihedrals_array)
    ## SEEING IF WE HAVE DIHEDRALS!
    # if total_dihedrals == 0:
    # return IndexError, 0
    # else:
    ## CALCULATING TRANS RATIO
    dihedral_trans_avg = calc_dihedral_ensemble_avg_ratio(assignment_dihedrals_array)
    # return dihedral_trans_avg, total_dihedrals
    
    #%%
    d = assignment_dihedrals_array
    test = np.mean((d < 240) & (d > 120))
    
        
    #%%
    
    ############################## CALIBRATION OF BUNDLING GROUPS ##############################
    
    ### FUNCTION TO CALIBRATE BUNDLING GROUPS
    def calibrate_class(class_function, extract_function, varying_variable = ['min_cluster_size', [1, 2, 3, 4, 5] ], **input_details ):
        '''
        The purpose of this function is to calibrate bundling groups in terms of bundling size
        INPUTS:
            class_function: [class] class function that you are interested in
            extract_function: [func] function to extract the data you are interested from the class -- prevents memory error! (too large space to store multiple classes)
            varying_variable: [list] ['variable', [list of inputs]] the variable that you are interested in varying
            bundling_group_function: bundling group function
            input_details [dict]: input details for class function group            
        OUTPUTS:
            storage: [list] list of the extracted results
        '''
        ## DEFINING LIST TO STORE THE CLASSES
        storage = []
        ## LOOPING THROUGH EACH VARYING VARIABLE
        for each_varying_variable in varying_variable[1]:
            ## PRINTING...
            print("\n----- VARYING %s, VALUE = %s -----\n"%(varying_variable[0], each_varying_variable) )
            ## CHANGING THE VARIABLE
            input_details[varying_variable[0]] = each_varying_variable
            ## RUNNING CLASS FOR EACH VARIABLE
            class_results = class_function( **input_details)
            ## RUNNING EXTRACTION RESULTS
            extract_results = extract_function(class_results)
            ## STORING THE CLASS RESULTS
            storage.append(extract_results)
        
        return storage
    
    ### FUNCTION TO EXTRACT INFORMATION FOR BUNDLING GROUPS
    def bundling_grps_extract( bundling_class ):
        '''
        The purpose of this function is to extract information from the bundling class
        INPUTS:
            bundling_class: bundling class from calc_bundling_grp
        OUTPUTS:
            extracted_results: [list] list of extracted results:
                0: minimum cluster size
                1: weights for the angle contribution
                2: average number of bundling groups
                3: std of bundling groups
                4: avg of nonassignments
                5: std of nonassignments
        '''
        extracted_results = [ bundling_class.min_cluster_size,
                              bundling_class.weights[0],
                              bundling_class.results_avg_std_bundling_grps['avg'],
                              bundling_class.results_avg_std_bundling_grps['std'],
                              bundling_class.results_avg_std_nonassignments['avg'],
                              bundling_class.results_avg_std_nonassignments['std'],
                              ]
        return extracted_results
    
    ## IMPORTING CSV MODULES
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add, csv_dict
    from MDDescriptors.core.initialize import checkPath2Server
    
    ## LOOPING THROUGH MULTIPLE TRAJECTORIES
    for specific_dir in ['EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1', 
                         'EAM_310.15_K_2_nmDIAM_hexadecanethiol_CHARMM36_Trial_1',
                         'EAM_310.15_K_6_nmDIAM_butanethiol_CHARMM36_Trial_1', 
                         'EAM_310.15_K_6_nmDIAM_hexadecanethiol_CHARMM36_Trial_1',]:
    
        ## DEFINING SPECIFIC PATH
        path2AnalysisDir=checkPath2Server(r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\')  # PC Side
        
        ### LOADING TRAJECTORY
        traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                     structure_file = gro_file, # structure file
                      xtc_file = xtc_file, # trajectories
                      )
        
        ### DEFINING INPUT DATA
        input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                            'ligand_names'      :           ['OCT', 'BUT', 'HED', 'DEC'],   # Name of the ligands of interest
                            'itp_file'          :           'sam.itp',                      # ITP FILE
                            'min_cluster_size'  :           3,                              # Minimum cluster size 6
                            'save_disk_space'   :          True    ,                        # Saving space
                            'weights'           :           [0.5, 0.5],                     # Weights between particle group (should sum to 1)
                            'separated_ligands' :           False,                          # True if you want separated ligands
                            }
        
        ## CREATING DICTIONARY
        csv_info = csv_dict(file_name=specific_dir, decoder_type='nanoparticle' )
        
        for each_varying_variable in [ # ['min_cluster_size', [2,3,4,5,6,7,8]], 
                                       ['weights', [ [each_value, 1-each_value] for each_value in np.arange(0, 1.05, .05)]] 
                                                    ]:
            ### CALIBRATION WITH RESPECT TO MINIMUM CLUSTER SIZE
            bundling_groups_calibration = calibrate_class( class_function = calc_nanoparticle_bundling_groups,
                                                           extract_function = bundling_grps_extract,
                                                            varying_variable = each_varying_variable,
                                                            **input_details
                                                               )
            ### GETTING CALIBRATION RESULTS
            calibration_list =   np.array(bundling_groups_calibration)
        
            ## ADDING AVERAGE ANGLES BUNDLING
            csv_info.add(  data_title = each_varying_variable[0], data = [calibration_list[:,0], 
                                                                  calibration_list[:,1], calibration_list[:,2],
                                                                  calibration_list[:,3], calibration_list[:,4],calibration_list[:,5]
                                                                  ],
                         labels = ['cluster_size', 'angle_frac', 'avg_bundling_grp', 'std_bundling_grp', 'avg_nonassignments', 'std_nonassignments' ])
        
        ## EXPORTING
        csv_info.export()
    

    