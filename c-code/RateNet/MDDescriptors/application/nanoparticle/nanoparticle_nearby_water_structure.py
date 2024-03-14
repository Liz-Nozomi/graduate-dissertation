# -*- coding: utf-8 -*-
"""
nanoparticle_nearby_water_structure.py
The purpose of this script is to analyze the water structure near the nanoparticle. We would like to see how many waters make  contact with the ligands

CREATED ON: 06/22/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
INPUTS:
    - gro file
    - itp file (for bonding purposes)
    - cutoff radius to look for water

OUTPUTS:
    - water contact of each ligand per frame
    - average water contact for each ligand
    
ALGORITHM:
    - get nanoparticle structure
    - find all heavy atoms
    - finds all nearby water molecules

CLASSES:
    nanoparticle_nearby_water_structure: calculates number of waters nearby the heavy atoms of the ligands
    
** UPDATES **



"""

### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure
### IMPORTING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot

#############################################
### CLASS TO CHARACTERIZE WATER STRUCTURE ###
#############################################
class nanoparticle_nearby_water_structure:
    '''
    The purpose of this function is to analyze the water structure nearby the nanoparticle
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        cutoff_radius: [float] cutoff radius to look for water
        water_residue_name: [str, default = 'HOH'] water residue name
        separated_ligands: [logical] True if your ligands are not attached to each other -- True in the case of planar SAMs
        split_traj: [int] number of frames to split the trajectoyr
        save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
            Turning off: atom_pairs_lig_water, ligand_heavy_atom_index_flatten
    OUTPUTS:
        ## LIGAND INFORMATION
            self.ligand_heavy_atom_index: [np.array, shape=(num_ligands, num_atoms )] heavy atom index
            self.ligand_heavy_atom_index_flatten: [np.array, shape=(num_atoms, 1)] flatten out heavy atom indices
            self.ligand_heavy_atom_index_shape: [tuple, shape=(2)] shape of the heavy atom index
        ## WATER INDICES
            self.num_water_residues: [int] total number of water molecules
            self.water_residue_index: [list] list of water residue index
            self.water_oxygen_index: [list] atom list index of water oxygen
        ## ATOM PAIRS
            self.atom_pairs_lig_water: [np.array, shape=(num_pairs,2)] atom pairs between ligand and water
        ## LIGAND-WATER CONTACT
            self.num_water_ligand_contacts: [np.array, shape=(num_frames, num_ligands)] average ligand-water contact
            self.results_num_water_ligand_contacts: [dict] average and standard deviation in the number of ligand-water contacts
    FUNCTIONS:
        clean_disk: function to clean up disk space
        find_water_index: finds water index in the simulation
        calc_ligand_water_contacts: [staticmethod] calculates ligand-water contacts for a given ligand and water
        
    ALGORITHM:
        - get nanoparticle structure
        - find all heavy atoms
        - find distances between heavy atoms and water-oxygen (SLOW STEP)
        - loop through each heavy atom -- find nearby water molecules
        - calculate density of water for each heavy atom
        - calculate average density for each ligand (based on ligand index)
        - calculate ensemble average results
    NOTES:
        - This function uses splitting trajectory tool to get nearby water structure
    '''
    ### INITIALIZING
    def __init__(self, traj_data, ligand_names, itp_file, cutoff_radius, water_residue_name = 'HOH', split_traj = 10, separated_ligands = False, save_disk_space = True ):
        ## STORING VARIABLES
        self.cutoff_radius = cutoff_radius
        
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = ligand_names,        # ligand names
                                                itp_file            = itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = separated_ligands    # True if you want separated ligands 
                                                )
        
        ## DEFINING THE TRAJECTORY
        traj = traj_data.traj # [0:100] #[0:100]
        
        ## FINDING TOTAL NUMBER OF FRAMES
        # self.total_frames = traj_data.num_frames
        
        ## DEFINING LIGAND HEAVY ATOM INDEX (FLATTEN)
        self.ligand_heavy_atom_index = np.array(self.structure_np.ligand_heavy_atom_index)
        self.ligand_heavy_atom_index_flatten = np.array(self.ligand_heavy_atom_index).flatten()
            
        ## FINDING SHAPE OF LIGAND HEAVY ATOM INDEX
        self.ligand_heavy_atom_index_shape = self.ligand_heavy_atom_index.shape
        
        ## FINDING WATER INDEXES
        self.num_water_residues, self.water_residue_index, self.water_oxygen_index  = calc_tools.find_water_index(traj = traj, water_residue_name = water_residue_name)
        
        ## CREATING ATOM PAIRS BETWEEN LIGAND AND WATER
        self.atom_pairs_lig_water = calc_tools.create_atom_pairs_list(atom_1_index_list = self.ligand_heavy_atom_index_flatten, 
                                                           atom_2_index_list = self.water_oxygen_index)
        
        ## DEFINING INPUT VARIABLES FOR LIGAND CONTACT
        average_lig_contact_input_vars= { 
                                            'atom_1_index_list'                 : self.ligand_heavy_atom_index_flatten,     # atom list for ligands
                                            'atom_2_index_list'                 : self.water_oxygen_index,                  # atom list for water molecules
                                            'atom_pairs'                        : self.atom_pairs_lig_water,                # atoms pairs between ligand and water
                                            'cutoff_radius'                     : self.cutoff_radius,                       # cutoff radius in nm to search for water molecules
                                            'ligand_heavy_atom_index_shape'     : self.ligand_heavy_atom_index_shape,       # shape of the heavy atom indexes                
                }
        
        ## CALCULATING LIGAND-WATER CONTACT USING OPTIMIZED TRAJECTORY SPLITTING FUNCTION
        self.num_water_ligand_contacts = calc_tools.split_traj_function( traj = traj,
                                                                         split_traj = split_traj,
                                                                         input_function = self.calc_ligand_water_contacts,
                                                                         optimize_memory = True, # Turn on creation of array to save memory!
                                                                         **average_lig_contact_input_vars )
        
        ## FINDING AVERAGE WATER CONTACTS
        self.results_num_water_ligand_contacts = calc_tools.calc_avg_std_of_list(self.num_water_ligand_contacts)
        
        ## CLEANING UP DISK
        self.clean_disk(save_disk_space)
        
        return
    
    ### FUNCTION TO CLEAN UP DISK
    def clean_disk(self, save_disk_space = True):
        ''' 
        This function cleans up disk space 
        INPUTS:
            save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
        '''
        if save_disk_space == True:
            self.atom_pairs_lig_water, self.ligand_heavy_atom_index_flatten  = [], []
            ## TURNING OFF WATER CONTACTS --- FLOATING POINTS FOR FRAME X NUM_LIGANDS ARE EXPENSIVE TO SAVE!
            # self.average_ligand_water_contacts = []
            
        return

    ### FUNCTION TO FIND LIGAND-WATER DISTANCES
    @staticmethod
    def calc_ligand_water_contacts( traj, atom_1_index_list, atom_2_index_list, atom_pairs, cutoff_radius, ligand_heavy_atom_index_shape ):
        '''
        The purpose of this function is to calculate the ligand-water contacts based on some cutoff
        IMPORTANT NOTE: This is highly expensive! In fact, it may be too expensive to run this function all at once. It may be preferable to divide your trajectory up!
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            atom_1_index_list: [np.array, shape=(N,1)] atom index 1 (ligand) 
            atom_2_index_list: [np.array, shape=(N,1)] atom index 2 (water)
            atom_pairs: [np.array, shape=(num_pairs, 2)] atom pairs between atom 1 and atom 2
            cutoff_radius: [float] cutoff radius in nanometers
            ligand_heavy_atom_index_shape: [tuple, shape=(1,2)] shape of atom index ** used for reshaping the matrix to be ligand basis
        OUTPUTS:
            num_water_below_cutoff: [np.array, shape=(NUM_FRAMES, NUM_LIGANDS)] number of ligand-water contact for each ligand in each frame
        ALGORITHM:
            - compute distances between all pairs
            - find all ligand-water that is below some cutoff
            - reshape the distance matrix on a ligand basis
            - calculate average number of water molecules in contact with the ligand
        '''
        ## COMPUTING DISTANCES
        distances = md.compute_distances(traj = traj, atom_pairs = atom_pairs, periodic=True)
        
        ## RECONSTRUCTION OF DISTANCE MATRIX
        distances = distances.reshape( (len(traj), len(atom_1_index_list), len(atom_2_index_list)) )
        
        ## GETTING SHAPE OF DISTANCE MATRIX
        distance_matrix_shape = distances.shape
        
        ## RESHAPING DISTANCES MATRIX TO A PER LIGAND BASIS
        distances = distances.reshape( distance_matrix_shape[0], ligand_heavy_atom_index_shape[0], ligand_heavy_atom_index_shape[1], distance_matrix_shape[-1]   )
        ## SHAPE: NUM_FRAMES X NUM_LIGANDS X NUM_HEAVY_ATOMS X NUM_WATER
        
        ## FINDING WATER DISTANCES BELOW CUTOFF RADIUS (Note: This finds all unique water molecules!)
        num_water_below_cutoff = np.sum( np.any(distances < cutoff_radius, axis=3),axis = 2 ) ## RETURNS IN SHAPE: NUM_FRAMES X TOTAL LIGANDS
        
        return num_water_below_cutoff
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180607-Alkanethiol_Extended_sims" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_dodecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    '''    
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"PLANAR_SIMS" # Analysis directory
    category_dir="Planar" # category directory
    specific_dir="Planar_310.15_K_dodecanethiol_10x10_CHARMM36_intffGold" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    '''
    ### DEFINING FULL PATH TO WORKING DIRECTORY
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side

    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file


    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%%
    ### DEFINING INPUT DATA
    input_details = {   'traj_data'          :           traj_data,                      # Trajectory information
                         'ligand_names'      :           ['OCT', 'BUT', 'HED', 'DEC', 'DOD',],   # Name of the ligands of interest
                         'itp_file'          :           'sam.itp',                      # ITP FILE
                         'cutoff_radius'     :              0.6 ,                        # Cutoff radius in nanometers
                         'save_disk_space'   :          False    ,                        # Saving space
                         'split_traj'        :          25, # Number of frames to split trajectory
                         'separated_ligands' :          False,
                         }
    ### CALCULATING WATER STRUCTURE
    water_structure = nanoparticle_nearby_water_structure( **input_details )
    
    #%%
    
    traj = traj_data.traj[0:10]
    atom_pairs = water_structure.atom_pairs_lig_water
    atom_1_index_list = water_structure.ligand_heavy_atom_index_flatten
    atom_2_index_list = water_structure.water_oxygen_index
    cutoff_radius = 0.6
    ligand_heavy_atom_index_shape = water_structure.ligand_heavy_atom_index_shape
    
    
    ## COMPUTING DISTANCES
    distances = md.compute_distances(traj = traj, atom_pairs = atom_pairs, periodic=True)
    
    ## RECONSTRUCTION OF DISTANCE MATRIX
    distance_matrix = distances.reshape( (len(traj), len(atom_1_index_list), len(atom_2_index_list)) )
    
    ## SHAPE OF LIGAND HEAVY ATOM
    heavy_atom_shape = np.array(water_structure.structure_np.ligand_heavy_atom_index).shape
    
    ## GETTING SHAPE OF DISTANCE MATRIX
    distance_matrix_shape = distance_matrix.shape
    
    ## RESHAPING DISTANCES MATRIX TO A PER LIGAND BASIS
    distance_reshape = distance_matrix.reshape( distance_matrix_shape[0], heavy_atom_shape[0], heavy_atom_shape[1], distance_matrix_shape[-1]   )
    
    #%%
        
    ## FINDING WATER DISTANCES BELOW CUTOFF RADIUS (Note: This finds all unique water molecules!)
    num_water_below_cutoff = np.sum( np.any(distance_reshape < cutoff_radius, axis=3),axis = 2 ) ## RETURNS IN SHAPE: NUM_FRAMES X TOTAL LIGANDS
    
    
    
    #%%
    ## FINDING DISTANCES BELOW THE CUTOFF RADIUS
    distance_below_cutoff = np.sum(distance_matrix < cutoff_radius, axis=2 ) ## RETURNS IN SHAPE: NUM_FRAMES X TOTAL HEAVY ATOMS, e.g. (10, 2821)
    
    ## RESHAPING DISTANCES ON LIGANDS BASIS
    distance_below_cutoff_ligand_reshape = distance_below_cutoff.reshape( (len(traj), ligand_heavy_atom_index_shape[0],  ligand_heavy_atom_index_shape[1] ))
    ## RETURNS IN SHAPE: NUM_FRAMES X LIGAND_INDEX X ATOM_INDEX, e.g. (10, 217, 13) <-- 217 ligands with 13 heavy atoms as in dodecanethiol

    ## FINDING AVERAGE NUMBER OF WATER CONTACTS
    average_ligand_water_contacts = np.mean(distance_below_cutoff_ligand_reshape,axis=2)
    ## RETURNS IN SHAPE: NUM_FRAMES X LIGAND INDEX; this informs the average number of water contacts for the ligand
    
    
    
    
