# -*- coding: utf-8 -*-
"""
nanoparticle_gold_to_ligand_structure.py
The purpose of this script is to analyze the bundling ligand structure on the gold facets.

## CLASSES:
- nanoparticle_gold_to_ligand_structure: finds structure of the gold to ligand


Written by: Alex K. Chew (alexkchew@gmail.com, 06/25/2018)
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
### IMPORTING RDF FUNCTIONS
from MDDescriptors.application.nanoparticle.nanoparticle_rdf import calc_rdf_histogram_data_pieces, compute_ranges_and_bins_for_rdf, normalize_rdf, calc_edges

### FUNCTION TO CALCULATE RDF BETWEEN GOLD-GOLD ATOMS
def calc_rdf_pieces(traj, atom_pairs, r_range, bin_width, split_traj, periodic = True):
    '''
    The purpose of this function is to calculate the RDFs in multiple pieces to avoid memory error. 
    INPUTS:
        traj: trajectory from md.traj
        atom_pairs: [np.array, shape=(num_pairs, 2)] atom pairs that you are interested in for the RDF
        periodic: [logical, default=True] True if you want PBCs to be accounted for
    OUTPUTS:
        r: [np.array] radius vector for the RDF
        g_r: [np.array] values for the RDF
    '''
    ## CALCULATING RANGES AND BINS
    r_range, n_bins = compute_ranges_and_bins_for_rdf( r_range = r_range, bin_width = bin_width  )
    
    ## DEFINING INPUTS
    split_input = {
                        'atom_pairs'    : atom_pairs, # Atom pairs
                        'periodic'      : periodic,             # Periodicity
                        'r_range'       : r_range,              # Range between RDFs
                        'n_bins'        : n_bins,               # Number of bins for RDF
                        }
    
    ## FINDING DISTANCE MATRIX BASED ON THE ATOM PAIRS
    hist = calc_tools.split_traj_function( traj = traj,
                                         split_traj = split_traj,
                                         input_function = calc_rdf_histogram_data_pieces,
                                         optimize_memory = False, # False because this does not take too much memory!
                                         **split_input )

    ## SUMMING UP THE HISTOGRAMS
    hist = np.sum(hist, axis = 0)
    
    ## FINDING EDGES
    edges = calc_edges( r_range = r_range, n_bins = n_bins )
    
    ## NORMALIZING TO GET RDF
    r, g_r = normalize_rdf( traj = traj, g_r = hist, edges = edges, atom_pairs = atom_pairs  )
    
    return r, g_r


###########################################################
### CLASS FUNCTION TO ANALYZE THE GOLD-LIGAND STRUCTURE ###
###########################################################
class nanoparticle_gold_to_ligand_structure:
    '''
    The purpose of this function is to analyze the gold structure. We are interested in the surface gold atoms, facets, distances between gold atoms, etc.
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        gold_atom_name: [str] atom name of the gold, also available in the trajectory
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        bin_width: [float, default = 0.2] bin width in nm of the RDF
        save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
        r_range: [tuple, shape=2] range to plot your RDF
        coord_num_surface_to_bulk_cutoff: [int] coordination number between surface and bulk atoms. Everything less than or equal to this number is considered a surface atom
        gold_shape: [str, default=EAM] shape of the gold:
            EAM: embedded atom model
            spherical: based on fcc {111} cut
            hollow: based on perfectly uniform sphere that has a hollow core
    OUTPUTS:
        self.atom_index: [list] list of the atom indices of gold
        self.num_atoms: [int] total number of gold atoms
        ## RDF between gold and gold
            self.gold_rdf_r: [np.array] radius vector for the RDF
            self.gold_rdf_g_r: [np.array] values for the RDF 
        ## RDF between sulfur and gold
            self.gold_surface_sulfur_rdf_r: [np.array] radius vector for the RDF between gold-sulfur surface atoms
            self.gold_surface_sulfur_rdf_g_r: [np.array] radial distribution function for the RDF between gold-sulfur surface atoms
    FUNCTIONS:
        calc_rdf_gold_gold: function that calculates the RDF between gold-gold atoms
        calc_rdf_gold_surface_sulfur: function that calculates the RDF between surface gold atoms and sulfur atoms
        
    ALGORITHM:
        - find all gold atoms of interest
        - find distances between all gold atom pairs
        - create a function that can find an rdf between all gold atoms
        - find coordination number of all gold atoms
        - plot gold atoms and its corresponding coordination numbers
        
    IMPORTANT NOTES:
        - currently, the gold atom distances are based purely on coordinates without PBC. This means that the gold core should not be at the PBC. Make sure to center the gold atom!
    
    '''
    ### INITIALIZING
    def __init__(self, traj_data, ligand_names, itp_file,  split_traj = 10,  
                 bin_width = 0.2, r_range=(0.0, 2.0), separated_ligands = False, gold_atom_name="Au",
                 gold_shape = 'EAM', coord_num_surface_to_bulk_cutoff = 11, save_disk_space = True):
        ## STORING INITIAL VARIABLES
        self.r_range = r_range
        self.bin_width = bin_width
        self.gold_atom_name = gold_atom_name
        self.split_traj = split_traj
        self.gold_shape = gold_shape
        self.coord_num_surface_to_bulk_cutoff = coord_num_surface_to_bulk_cutoff
        
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = ligand_names,        # ligand names
                                                itp_file            = itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = separated_ligands    # True if you want separated ligands 
                                                )
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj # [0:100]
        
        ## FINDING ATOM INDEX OF GOLD
        self.gold_atom_index = np.array(calc_tools.find_atom_index(   traj = traj,                    # trajectory
                                                             atom_name = self.gold_atom_name,    # gold atom name
                                                             ))
        ## FINDING TOTAL NUMBER OF ATOMS
        self.gold_num_atoms = len(self.gold_atom_index)
        
        ## FINDING GOLD-GOLD RDFs
        self.calc_rdf_gold_gold(traj = traj)
        
        ## FINDING RDF OF GOLD-SULFUR
        self.calc_rdf_gold_surface_sulfur(traj = traj)
        
        return
        
    ### FUNCTION TO CALCULATE RDF BETWEEN GOLD-GOLD ATOMS
    def calc_rdf_gold_gold(self, traj, periodic=True ):
        '''
        The purpose of this function is to find the RDF between gold-gold atoms.
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            periodic: [logical, default=True] True if you want PBCs to be accounted for
        OUTPUTS:
            self.gold_rdf_r: [np.array] radius vector for the RDF between gold-gold atoms
            self.gold_rdf_g_r: [np.array] radial distribution function for the RDF between gold-gold-atoms
        '''
        ## FINDING ATOM PAIRS
        gold_atom_pairs = calc_tools.create_atom_pairs_with_self(self.gold_atom_index)[0]

        ## FINDING RDF OF GOLD-GOLD
        self.gold_rdf_r, self.gold_rdf_g_r = calc_rdf_pieces(traj=traj, 
                                                             atom_pairs = gold_atom_pairs,
                                                             r_range = self.r_range,
                                                             bin_width = self.bin_width,
                                                             split_traj = self.split_traj,
                                                             periodic = periodic,
                                                             )
        
        return
        
    ### FUNCTION TO CALCULATE RDF BETWEEN GOLD-SULFUR ATOMS
    def calc_rdf_gold_surface_sulfur(self, traj, bin_width= 0.02, periodic=True):
        '''
        The purpose of this function is to calculate the RDF between gold atoms and the surface sulfur atoms
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            bin_width: [float, default=0.02] bin width of the gold-surface rdf
            periodic: [logical, default=True] True if you want PBCs to be accounted for
        OUTPUTS:
            self.gold_surface_sulfur_rdf_r: [np.array] radius vector for the RDF between gold-sulfur surface atoms
            self.gold_surface_sulfur_rdf_g_r: [np.array] radial distribution function for the RDF between gold-sulfur surface atoms
        '''
        ## IMPORTING SCRIPT TO FIND SURFACE ATOMS
        from MDDescriptors.application.nanoparticle.nanoparticle_sulfur_gold_coordination import full_find_surface_atoms
        
        ## FINDING COORDINATION NUMBER AND SURFACE ATOMS
        gold_coord_num, gold_gold_cutoff, gold_surface_indices, \
        gold_surface_atom_index, gold_surface_coord_num, gold_surface_num_atom = \
                    full_find_surface_atoms( traj                               = traj,
                                             gold_atom_index                    = self.gold_atom_index,
                                             gold_shape                         = self.gold_shape,
                                             coord_num_surface_to_bulk_cutoff   = self.coord_num_surface_to_bulk_cutoff,
                                             frame = -1, # Use last frame for gold-gold distances
                                             verbose = True,
                                             periodic = True,
                                            )
        ## ATOM INDICES OF SULFUR
        sulfur_atom_index = np.array(self.structure_np.head_group_atom_index)
        
        ## GETTING ATOM PAIRS BETWEEN GOLD AND SULFUR
        gold_sulfur_atom_pairs = calc_tools.create_atom_pairs_list( atom_1_index_list = gold_surface_atom_index,
                                                                         atom_2_index_list = sulfur_atom_index,)
        
        ## FINDING RDF OF GOLD TO SURFACE OF SULFUR
        self.gold_surface_sulfur_rdf_r, self.gold_surface_sulfur_rdf_g_r = \
                                            calc_rdf_pieces(traj=traj, 
                                                             atom_pairs = gold_sulfur_atom_pairs,
                                                             r_range = self.r_range,
                                                             bin_width = bin_width,
                                                             split_traj = self.split_traj,
                                                             periodic = periodic,
                                                             )
        return
        
        
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
                         'gold_atom_name'    :          'Au',                           # Atom name of gold   
                         'r_range'           :          (0.0, 3.0),                      # r range to run the rdf function
                         'bin_width'         :          0.005,                                # Bin width   
                         'save_disk_space'   :          False    ,                        # Saving space
                         'split_traj'        :          25, # Number of frames to split trajectory
                         'separated_ligands' :          False,
                         'coord_num_surface_to_bulk_cutoff'  :          11,   
                         'gold_shape'        :          'EAM',
                         }
    
    ### RUNNING NANOPARTICLE GOLD STRUCTURE
    structure_gold_lig = nanoparticle_gold_to_ligand_structure( **input_details )
    
    ## PLOTTING PAIR DISTRIBUTION
    # structure_gold.plot_dist_distribution()
    #%%
    
    num_atoms = structure_gold_lig.gold_num_atoms
    
    indices = np.array(structure_gold_lig.gold_atom_index)
    
